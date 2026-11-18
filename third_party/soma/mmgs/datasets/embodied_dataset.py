from copy import deepcopy
import os
import sys
from typing import Union, Tuple, Dict
from collections import defaultdict, OrderedDict
import copy

from scipy.spatial.transform import Rotation as scipy_R
import point_cloud_utils as pcu
# from decord import VideoReader
import numpy as np
import torch
from decord import VideoReader # Must put after import torch; if put this first, cannot use cuda
from torch import Tensor
from torch.utils.data import Dataset
# from physdreamer.data.scene_box import SceneBox
# from physdreamer.gaussian_3d.utils.rigid_body_utils import get_rigid_transform
from mmgs.datasets.utils.cameras import Camera, focal2fov, fov2focal
from mmgs.datasets.utils.io import readPKL, writePKL, readJSON, read_video_image_cv2, read_video_image_rgba_cv2_mask
from mmgs.datasets.utils.misc import tensor_container
from mmgs.utils.colmap_utils import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from mmgs.utils.colmap_utils import Image as colmap_Image
from mmgs.utils.physdreamer_utils import find_far_points
import json
from PIL import Image
import pickle
from .builder import DATASETS
from mmcv.parallel import collate
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import euclidean_distances

import sys
sys.path.append('gaussian-splatting')
from scene.gaussian_model import GaussianModel
from math import atan

def read_uint8_rgba(img_path, img_hw=None):
    if not (img_path.endswith(".png") or img_path.upper().endswith(".JPG")):
        img_path = img_path + ".png"

    if not os.path.exists(img_path):
        img_path = img_path.replace(".png", ".jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        with Image.open(img_path) as image:
            if img_hw is not None:
                image = image.resize((img_hw[1], img_hw[0]), Image.Resampling.BILINEAR)
            im_data = np.array(image.convert("RGBA"))
        return im_data
    else:
        with Image.open(img_path) as image:
            if img_hw is not None:
                image = image.resize((img_hw[1], img_hw[0]), Image.Resampling.BILINEAR)
            im_data = np.array(image.convert("RGBA"))
    return im_data

def extract_masters(cam_meta):
    unique_mapping = cam_meta['twin']
    slave_dict = defaultdict(set)
    for t_list in unique_mapping:
        slave_dict[t_list[0]].add(t_list[1])
    return slave_dict

def expand_cameras(cam_meta, cam_extrinsics):
    master_dict = extract_masters(cam_meta)

    if len(master_dict) <= 0:
        return cam_extrinsics, None
    invalid_names = cam_meta['invalid'] + cam_meta['force_invalid']
    exp_cam_extrinsics = dict()
    exp_real_image_path = dict()
    extra_pointer = max(cam_extrinsics.keys()) + 1
    for idx, key in enumerate(cam_extrinsics):
        exp_cam_extrinsics[key] = cam_extrinsics[key] # COPY
        extr = cam_extrinsics[key]
        image_name = extr.name
        image_base_name = os.path.basename(image_name)
        assert image_base_name == image_name
        seq_name = os.path.splitext(image_base_name)[0]
        seq_idx = int(seq_name.split('_')[1])
        cam_idx = int(seq_name.split('_')[3])
        if seq_idx not in master_dict.keys():
            continue
        for slave_seq_idx in master_dict[seq_idx]:
            slave_seq_name = f"seq_{str(slave_seq_idx).zfill(5)}_cam_{str(cam_idx).zfill(5)}"
            if slave_seq_name in invalid_names:
                continue
            # Expand the slaves
            ## Change the image name
            assert len(os.path.splitext(image_base_name)) == 2
            slave_image_name = slave_seq_name + os.path.splitext(image_base_name)[1]
            exp_real_image_path[extra_pointer] = image_name
            slave_extr_tmp = copy.deepcopy(extr)
            slave_extr = colmap_Image(
                    id=slave_extr_tmp.id, qvec=slave_extr_tmp.qvec, tvec=slave_extr_tmp.tvec,
                    camera_id=slave_extr_tmp.camera_id, name=slave_image_name,
                    xys=slave_extr_tmp.xys, point3D_ids=slave_extr_tmp.point3D_ids)
            exp_cam_extrinsics[extra_pointer] = slave_extr
            extra_pointer += 1
    
    return exp_cam_extrinsics, exp_real_image_path

def compute_bbox(img, margin=2):
    device = img.device
    H, W = img.shape[-2:]
    # 创建mask：任何通道有非零值的像素
    mask = torch.any(img > 0, axis=0)  # [H, W]，True表示该像素至少一个通道>0
    nonzero_coords = torch.where(mask)
    if len(nonzero_coords[0]) > 0:  # 确保有非零像素
        y_coords, x_coords = nonzero_coords
        x_min = torch.min(x_coords)
        y_min = torch.min(y_coords) 
        x_max = torch.max(x_coords) + 1  # +1使bbox包含边界像素
        y_max = torch.max(y_coords) + 1
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(W, x_max + margin)
        y_max = min(H, y_max + margin)

        if (x_max - x_min) % 2 == 1:
            x_max -= 1
        if (y_max - y_min) % 2 == 1:
            y_max -= 1

        return torch.tensor([y_min, x_min, y_max, x_max], device=device, dtype=torch.int32)

def bbox_reader(video_list, bounding_box, margin=2):
    """
    video_list: [n_cam, nf, 3, H, W]
    bounding_box: None, 'static', 'dynamic'
    return: [n_cam, nf, 4] , [x_min, y_min, x_max, y_max]
    """
    # n_cam, n_frame, _, H, W = video_list.shape
    n_cam, n_frame = len(video_list), len(video_list[0])
    H, W = video_list[0][0].shape[-2:]
    device = video_list[0][0].device
    bbox = torch.zeros((n_cam, n_frame, 4), device=device, dtype=torch.int32)
    if bounding_box is None:
        bbox[:, :, 0] = 0
        bbox[:, :, 1] = 0
        bbox[:, :, 2] = H
        bbox[:, :, 3] = W
        return bbox
        # return None
    elif bounding_box == 'static':
        for i_cam in range(n_cam):
            for i_frame in range(n_frame):
                img = video_list[i_cam][i_frame]  # [3, H, W]
                bbox[i_cam, i_frame, :] = compute_bbox(img, margin)
        return bbox
    elif bounding_box == 'dynamic':
        for i_cam in range(n_cam):
            for i_frame in range(n_frame):
                if i_frame == 0:
                    bbox[i_cam, i_frame, :] = torch.tensor([0, 0, H, W], device=device)
                
                cur_img = video_list[i_cam][i_frame]
                prev_img = video_list[i_cam][i_frame-1]
                res_img = cur_img - prev_img
                bbox[i_cam, i_frame, :] = compute_bbox(res_img, margin)
        return bbox
    else:
        raise ValueError(f"Invalid bounding box type: {bounding_box}")

def extrinsic_to_R_T(M, convention='cw'):
    R_raw = M[:3, :3]
    T_raw = M[:3, 3]
    if convention == 'cw':
        R, T = R_raw, T_raw
    else:
        R = R_raw.T
        T = -R_raw.T @ T_raw
    return R, T

def extract_extrinsics(c2w):
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]
    return R, T

def fov_from_intrinsics(intrinsic, img_hw):
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    H, W = img_hw
    FoVx = 2.0 * atan(W / (2.0 * fx))
    FoVy = 2.0 * atan(H / (2.0 * fy))
    return FoVx, FoVy

def focal2fov(focal, pixels):
    return 2.0 * atan(pixels / (2.0 * focal))

def extract_intrinsics(intrinsic, img_hw):
    focal_length_x = intrinsic[0][0]
    focal_length_y = intrinsic[1][1]

    H, W = img_hw

    FoVx = focal2fov(focal_length_x, W)
    FoVy = focal2fov(focal_length_y, H)
    return FoVx, FoVy

def readEmbodiedCameras(
    cam_extrinsics,
    cam_intrinsics,
    images_folder,
    videos_folder,
    img_hw=None,
    scale_x_angle=1.0,
    suffix_replace=None,
    max_cam_total=-1,
    max_seq=-1,
    exp_real_image_path=None,
    split_list=None,
):
    camera_dict = defaultdict(list)
    seq_info_dict = defaultdict(list)
    
    for seq_ids, split in enumerate(split_list):
        for cam_ids in range(max_cam_total):
            image_path = os.path.join(images_folder, f"{cam_ids}", f"{split[0]}.png")

            static_image_path = image_path
            real_image_path = image_path

            # 1. transfer extrinsics to camera_dict
            # print(cam_extrinsics, len(cam_extrinsics), max_cam_total)
            M = cam_extrinsics[cam_ids]
            intrinsics = cam_intrinsics[cam_ids]
            # R, T = extrinsic_to_R_T(M, convention='cw')
            # FoVx, FoVy = fov_from_intrinsics(intrinsics, img_hw)
            R, T = extract_extrinsics(M)
            FoVx, FoVy = extract_intrinsics(intrinsics, img_hw)
            cam_info = Camera(
                R=R,
                T=T,
                FoVx=FoVx,
                FoVy=FoVy,
                img_path=real_image_path,
                static_img_path=static_image_path,
                img_hw=img_hw,
                timestamp=None,
                data_device="cuda",
            )

            seq_name = f"seq_{seq_ids:05d}"
            camera_dict[seq_name].append(cam_info)

            # 2. load static images
            seq_info_dict[seq_name].append(real_image_path)
    
    return camera_dict, seq_info_dict

@DATASETS.register_module()
class EmbodiedDataset(Dataset):
    """
    Dataset for embodied scenes. (input data format according to phystwin)
        Static. ?

        Load images, which is unique for each camera information;
        Each image associates with one video, but may have multi-ple sequences recoding the same motions
            image file name: [seq_xxx_]frame_xxx.png; if don't have seq_xxx, then assume this motion has only one video;
        
        Output:
            camera, image, mask, and other metadata like flow, depth..
    """
    def __init__(self, env_cfg, phase, **kwargs) -> None:
        super(EmbodiedDataset, self).__init__()
        
        self.env_cfg = env_cfg
        self.phase = phase

        self.gravity = np.array(env_cfg['gravity'], dtype=np.float32).reshape(1, 3)
        self.real_dt = env_cfg.get('real_dt', None)
        self.comp_dt = env_cfg.get('dt', 1/30)
        # Parse attributes
        self.data_dir = env_cfg['data_dir']
        self.sub_video_dir = env_cfg['sub_video_dir']
        self.sub_img_dir = env_cfg['sub_img_dir']
        self.max_cam_num = env_cfg['max_cam_num']
        self.max_cam_total = env_cfg['max_cam_num']
        self.max_seq = env_cfg.get('max_seq', -1)
        self.llffhold = env_cfg.get('llffhold', 0)
        self.tracking_pcd = env_cfg.get('tracking_pcd', None)
        self.use_random_background = env_cfg.get('use_random_background', None)
        self.const_white_bg = env_cfg.get('const_white_bg', None)
        self.frame_gap = env_cfg.get('frame_gap', 1) 

        self.split_list = env_cfg.get('split_list', None) #TODO
        self.prompt_dict = env_cfg.get('prompt_dict', None)
        
        aligned_scene = env_cfg.get('aligned_scene', None)
        self.bounding_box = env_cfg.get('bounding_box', None)
        if aligned_scene is not None:
            self.aligned_scene_idx = self.find_obje_idx(aligned_scene)
        else:
            self.aligned_scene_idx = -1

        self.scene_list = env_cfg.get('scene_list', None) # if None, use all scene
        self.max_seq = len(self.split_list[self.scene_list[0]])
        if self.scene_list is None:
            self.scene_list = []
            for scene_dir in os.listdir(self.data_dir):
                if os.path.isdir(os.path.join(self.data_dir, scene_dir)):
                    self.scene_list.append(scene_dir)
        else:
            assert isinstance(self.scene_list, list)

        #phase attributes
        assert len(self.scene_list) > 0
        self.cam_transform_fn = env_cfg.get('cam_transform_fn', f"transforms_{phase}.json")
        self.resolution = env_cfg['resolution']
        self.scale_x_angle = env_cfg['scale_x_angle']
        self.load_imgs = env_cfg['load_imgs']
        self.load_pcs = env_cfg.get('load_pcs', True)
        # Load data info [with data]
        self.meta_dict_sclist, self.camera_list_sclist, self.np_uint8_rgba_list_sclist = self._parse_dataset(self.data_dir, self.scene_list, self.sub_img_dir, self.sub_video_dir, self.cam_transform_fn, load_imgs=self.load_imgs, max_cam_total=self.max_cam_total, max_seq=self.max_seq, llffhold=self.llffhold)
        
        self._num_frames = sum(meta_dict['num_frames'] for meta_dict in self.meta_dict_sclist)
        self.num_cameras = sum(meta_dict["num_cameras"] for meta_dict in self.meta_dict_sclist)
        #check
        # assert self._num_frames == self.num_cameras, f"num_frames {self._num_frames} != num_cameras {self.num_cameras}"

        self.idx_mapping = self._parse_idx(self.camera_list_sclist, self.max_cam_num, padding=env_cfg['pad_cam'], phase=self.phase)
        self.dataset_len = len(self.idx_mapping)

        if env_cfg.get('use_index', None) is not None:
            assert False, "Not supported yet"
            use_index = [_ for _ in use_index if _ < len(self.camera_list)]
            self.camera_list = [self.camera_list[i] for i in use_index]
            # self.np_uint8_rgba_list = [self.np_uint8_rgba_list[i] for i in use_index]
            if self.load_imgs:
                self.np_uint8_rgba_list = [
                    self.np_uint8_rgba_list[i] for i in use_index
                ]
            # self.test_camera_list = [self.test_camera_list[i] for i in use_index]
            # self.test_camera_list = self.test_camera_list
            self.dataset_len = len(self.camera_list)
        
        #TODO
        # self.pcmask_sclist, self.pinmask_sclist, self.cln_pcmask_sclist = self._load_pointcloud_mask(self.data_dir, self.meta_dict_sclist, pcmask_name=env_cfg.get('pcmask_name', 'pc_mask.pkl'), cln_pcmask_name=env_cfg.get('cln_pcmask_name', 'cln_pc_mask.pkl'), pinmask_name=env_cfg.get('pinmask_name', 'pin_mask.json'))
        cluster_type = env_cfg.get('cluster_type', '')
        clustermask_sclist = self._load_cluster_mask(self.data_dir, self.meta_dict_sclist, env_cfg['cluster_cfg'], cluster_type=cluster_type)
        controller_clustermask_sclist = self._load_controller_cluster_mask(self.data_dir, self.meta_dict_sclist, env_cfg['controller_cfg'], cluster_type=cluster_type)
        self.merged_clustermask_sclist = self._merge_cluster_mask(clustermask_sclist, controller_clustermask_sclist)

    def _parse_idx(self, camera_list_sclist, max_cam_num, padding=True, phase='train' ):
        idx_mapping = []
        for sc_idx, camera_list in enumerate(camera_list_sclist):
            for seq_idx, cam_list in camera_list.items():
                cam_idx_list = [_ for _ in range(len(cam_list))]
                cam_idx_list = np.array(cam_idx_list)
                if phase == 'train' or phase == 'all':
                    # Randomize
                    cam_idx_list = np.random.choice(cam_idx_list, size=cam_idx_list.shape[0], replace=False)
                for i in range(0, cam_idx_list.shape[0], max_cam_num):
                    cur_batch_idx = [i+_ for _ in range(max_cam_num) if i+_ < len(cam_idx_list)]
                    cur_batch = [cam_idx_list[i] for i in cur_batch_idx]
                    idx_mapping.append([sc_idx, seq_idx, cur_batch])
        return idx_mapping

    def _parse_dataset(self, data_dir, scene_list, image_dir, video_dir, cam_transform_fn, load_imgs=False, max_cam_total=-1, max_seq=-1, llffhold=0):
        meta_dict_sclist, camera_dict_sclist, np_uint8_rgba_dict_sclist = [], [], []
        for scene_n in scene_list:
            scene_data_dir = os.path.join(data_dir, scene_n)
            assert os.path.exists(scene_data_dir)
            
            if cam_transform_fn is not None:
                camera_transform_file = os.path.join(scene_data_dir, cam_transform_fn)
            
            if cam_transform_fn is not None and os.path.exists(camera_transform_file):
                print(f"=> loading {scene_n} camera from blender format {cam_transform_fn}")
                camera_dict, meta_dict = self._read_camera_transforms(
                    scene_data_dir, camera_transform_file, img_hw=self.resolution, max_cam_total=max_cam_total, max_seq=max_seq,
                )
            else:
                extrinsics_path = os.path.join(scene_data_dir, 'calibrate.pkl')
                assert os.path.exists(extrinsics_path), "calibrate.pkl not found!"
                from joblib import load
                extrinsics = load(extrinsics_path)
                intrinsics_path = os.path.join(scene_data_dir, 'metadata.json')
                assert os.path.exists(intrinsics_path), "metadata.json not found!"
                intrinsics = readJSON(intrinsics_path)

                print(f"=> loading {scene_n} camera from embodied format")
                split_list = self.split_list[scene_n]
                camera_dict, meta_dict = self._read_camera_transforms_embodied(
                    scene_data_dir, extrinsics, intrinsics, image_dir=image_dir, video_dir=video_dir, split_list=split_list, img_hw=self.resolution, max_cam_total=max_cam_total, max_seq=max_seq, eval=False, llffhold=llffhold,
                ) 

            np_uint8_rgba_dict = self._load_imgs_info(camera_dict)
            if load_imgs:
                data_np_uint8_rgba_dict = dict()
                for seq_name, rgba_list in np_uint8_rgba_dict.items():
                    rgba_data_list = self._load_imgs(rgba_list)
                    data_np_uint8_rgba_dict[seq_name] = rgba_data_list
                np_uint8_rgba_dict = data_np_uint8_rgba_dict
            
            if 'scene_name' not in meta_dict.keys():
                meta_dict['scene_name'] = scene_n
            meta_dict_sclist.append(meta_dict)
            camera_dict_sclist.append(camera_dict)
            np_uint8_rgba_dict_sclist.append(np_uint8_rgba_dict)
        
        return meta_dict_sclist, camera_dict_sclist, np_uint8_rgba_dict_sclist
    
    def _load_imgs_info(self, camera_dict):
        np_uint8_rgba_path_dict = dict()
        print("Loading images' information...")
        for seq_name, cam_list in camera_dict.items():
            rgba_path_list = [cam.img_path for cam in cam_list]
            np_uint8_rgba_path_dict[seq_name] = rgba_path_list
        return np_uint8_rgba_path_dict
    
    def _load_imgs(self, np_uint8_rgba_path_list):
        assert False
        np_uint8_rgba_list = []
        print("Loading images...")
        # for img_path in tqdm(np_uint8_rgba_path_list):
        for img_path in np_uint8_rgba_path_list:
            im_data = read_uint8_rgba(img_path, self.resolution)
            # im_data = read_uint8_jpg(img_path, self.resolution)
            np_uint8_rgba_list.append(im_data)

            if self.resolution is None:
                self.resolution = np_uint8_rgba_list[0].shape[:2]
        
        if self.load_imgs:
            # Only print if preload images
            print(
                "img dtype: ",
                np_uint8_rgba_list[0].dtype,
                # "num_frames: ",
                # self.num_frames,
                "image resolution(h,w): ",
                self.resolution,
            )
        return np_uint8_rgba_list

    def __len__(self):
        return self.dataset_len
    
    @property
    def num_frames(self):
        return self._num_frames
    
    def _merge_cluster_mask(self, clustermask_sclist, controller_clustermask_sclist):
        merged_clustermask_sclist = []
        for scene_idx, (clustermask, controller_clustermask) in enumerate(zip(clustermask_sclist, controller_clustermask_sclist)):
            merged_clustermask = []
            for num_level, (clustermask_level, controller_clustermask_level) in enumerate(zip(clustermask, controller_clustermask)):
                # num_controller = len(controller_clustermask_level)
                num_controller = controller_clustermask_level.max() + 1
                merged_clustermask_level = [controller_clustermask_level, np.array(list(map(lambda x: x + num_controller, clustermask_level)))]
                merged_clustermask.append(merged_clustermask_level)
            merged_clustermask.append([np.zeros(1), np.zeros(1)])
            merged_clustermask_sclist.append(merged_clustermask)
        return merged_clustermask_sclist

    def _load_cluster_mask(self, data_dir, meta_dict_sclist, cluster_cfg, clustermask_name='cluster_mask.pkl', sc_pc_name='clean_object_points.ply', cluster_type=''):
        clustermask_sclist = []
        aux_pc_dict = dict()
        for scene_idx, scene_meta in enumerate(meta_dict_sclist):
            scene_cluster = []
            scene_n = scene_meta['scene_name']
            if isinstance(cluster_cfg, dict):
                assert scene_n in cluster_cfg.keys()
                scene_cluster_cfg = cluster_cfg[scene_n]
            else:
                scene_cluster_cfg = cluster_cfg
            
            num_level = len(scene_cluster_cfg)
            unique_id = [f"nl_{num_level}"]
            # Get unique key
            for i in range(num_level):
                cluster_idx = i
                cluster_meta = scene_cluster_cfg[cluster_idx]
                path_key = ''
                if 'downsample_rate' in cluster_meta.keys():
                    digit_info = str(cluster_meta['downsample_rate']).replace('.', 'd')
                    path_key = f'downsample_rate_{digit_info}'
                else:
                    assert 'num_cluster' in cluster_meta.keys()
                    digit_info = str(cluster_meta['num_cluster']).replace('.', 'd')
                    path_key = f'nc_{digit_info}'
                unique_id.append(path_key)
            unique_id = '_'.join(unique_id)
            mask_dir = os.path.join(data_dir, scene_n, 'cluster_mask', cluster_type)
            os.makedirs(mask_dir, exist_ok=True)
            mask_path = os.path.join(mask_dir, f'{cluster_type}{unique_id}_{clustermask_name}')

            if not os.path.exists(mask_path):
            # if True:
                print(f"Generating cluster mapping to {mask_path}")
                prev_clustered_pcs = None
                cluster_mask_list = []
                for i in range(num_level):
                    cluster_idx = i
                    cluster_meta = scene_cluster_cfg[cluster_idx]
                    # path_key = ''
                    # Generate here
                    if i == 0:
                        if scene_n not in aux_pc_dict.keys():
                            # mov_pcs = self.pcmask_sclist[scene_idx]
                            # sc_pcs_path = os.path.join(data_dir, scene_n, 'gs', sc_pc_name)
                            try:
                                sc_pcs_path = os.path.join(data_dir, scene_n, 'pi3', 'gs', 'point_cloud', 'iteration_10000', 'point_cloud.ply')
                                scene_gaussian = GaussianModel(0)
                                print(f"Initializing gaussian scene: {sc_pcs_path}")
                                scene_gaussian.load_ply(sc_pcs_path)
                            except:
                                sc_pcs_path = os.path.join(data_dir, '..', 'gaussian_output', scene_n, 'init=hybrid_iso=True_ldepth=0.1_lnormal=0_laniso_0.0_lseg=1.0/point_cloud/iteration_10000/point_cloud.ply')
                                scene_gaussian = GaussianModel(3)
                                print(f"Initializing gaussian scene: {sc_pcs_path}")
                                scene_gaussian.load_ply(sc_pcs_path)
                            # aux_pc_dict[scene_n] = scene_gaussian.get_xyz[mov_pcs].detach().cpu().numpy()
                            aux_pc_dict[scene_n] = scene_gaussian.get_xyz.detach().cpu().numpy()
                        cur_pcs = aux_pc_dict[scene_n]
                    else:
                        cur_pcs = prev_clustered_pcs
                    # Cluster
                    if cluster_type == '':
                        if 'downsample_rate' in cluster_meta.keys():
                            num_cluster = int(cluster_meta['downsample_rate'] * cur_pcs.shape[0])
                        else:
                            num_cluster = cluster_meta['num_cluster']
                        num_cluster = max(num_cluster, 1)
                        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(cur_pcs)
                        new_pcs = kmeans.cluster_centers_
                        pc_labels = kmeans.labels_
                    elif cluster_type == 'dis_split':
                        assert 'downsample_rate' in cluster_meta.keys(), "downsample_rate now serve as maximum distance"
                        downsample_dis = cluster_meta['downsample_rate']
                        cluster_handler = AgglomerativeClustering(n_clusters=None, linkage='complete', distance_threshold=downsample_dis).fit(cur_pcs)
                        pc_labels = cluster_handler.labels_
                        clf = NearestCentroid()
                        clf.fit(cur_pcs, pc_labels)
                        new_pcs = clf.centroids_
                    elif cluster_type == 'skeleton':
                        if i != num_level -1:
                            assert 'downsample_rate' in cluster_meta.keys(), "downsample_rate now serve as maximum distance"
                            downsample_dis = cluster_meta['downsample_rate']
                            cluster_handler = AgglomerativeClustering(n_clusters=None, linkage='complete', distance_threshold=downsample_dis).fit(cur_pcs)
                            pc_labels = cluster_handler.labels_
                            clf = NearestCentroid()
                            clf.fit(cur_pcs, pc_labels)
                            new_pcs = clf.centroids_
                        # if i == num_level - 1:
                        else:
                            # For the last level, compute the skeleton clustering
                            assert 'anchor_points' in cluster_meta.keys(), "anchor_points must be provided for skeleton clustering"
                            anchor_points = cluster_meta['anchor_points']
                            cluster_points = anchor_points['cluster']
                            joint_points = anchor_points['joint']
                            #1. compute joint anchor angle
                            joint_anchor_dict = self._compute_joint_anchor_angle(cluster_points, joint_points)
                            #2. find skeleton labels
                            skeleton_label = []
                            joint_points_position = [p for p in joint_points.values()]
                            for i, c in enumerate(new_pcs):
                                #2.1 find the closest joint point
                                dist = euclidean_distances(c.reshape(1, -1), joint_points_position)
                                closest_joint_idx = np.argmin(dist)
                                closest_joint_pos = joint_points_position[closest_joint_idx]
                                closest_joint = list(joint_points.keys())[closest_joint_idx]
                                #2.2 compute angle
                                cluster_angle = self._compute_spatial_angle(closest_joint_pos, c)
                                #2.2 find the closest cluster point leverage the joint anchor angle
                                joint_anchor = joint_anchor_dict[closest_joint]
                                
                                # angle_list = [v for v in joint_anchor.values()]
                                angle_list = sum(joint_anchor.values(), [])
                                angle_keys = [[k]*len(v) for k, v in joint_anchor.items()]
                                angle_keys = sum(angle_keys, [])
                                angle_dist = [self._cosine_similarity(cluster_angle, angle) for angle in angle_list]
                                closest_angle_idx = np.argmax(angle_dist)
                                # cluster_label = list(joint_anchor.keys())[closest_angle_idx]
                                cluster_label = angle_keys[closest_angle_idx]                                
                                cluster_label_ids = list(cluster_points.keys()).index(cluster_label)
                                skeleton_label.append(cluster_label_ids)
                            #3. get center points
                            pc_labels = np.array(skeleton_label)
                            clf = NearestCentroid()
                            clf.fit(cur_pcs, pc_labels)
                            new_pcs = clf.centroids_
                            #TODO: fine level segmentation adjustment
                    else:
                        assert False, "Invalid cluster type"
                    cluster_mask_list.append(pc_labels) # labels  (n_cluster,  )
                    prev_clustered_pcs = new_pcs        # new_pcs (n_cluster, 3)
                #TODO, pinned_state?
                # pinned_state = aux_pc_dict[scene_n][self.pinmask_sclist[scene_idx][:, 0]]
                # dist = euclidean_distances(prev_clustered_pcs, pinned_state)
                # p2pin_mapping = np.argmin(dist, axis=-1)
                p2pin_mapping = np.zeros(len(prev_clustered_pcs))
                cluster_mask_list.append(p2pin_mapping)
                print(f"Saving cluster mapping list to {mask_path}")
                writePKL(mask_path, dict(p2c=cluster_mask_list))
            scene_cluster = readPKL(mask_path)['p2c']
            clustermask_sclist.append(scene_cluster)
            # self._save_cluster_image(save_path=mask_dir, clustermask_sclist=clustermask_sclist, )
        return clustermask_sclist

    def _load_controller_cluster_mask(self, data_dir, meta_dict_sclist, controller_cfg, controller_mask_name='track_process_data.pkl', cluster_type=''):
        controller_clustermask_sclist = []
        aux_pc_dict = dict()
        for scene_idx, scene_meta in enumerate(meta_dict_sclist):
            scene_controller = []
            scene_n = scene_meta['scene_name']
            tracking_pcd_path = os.path.join(self.data_dir, scene_n, controller_mask_name)
            mask_info_path = os.path.join(self.data_dir, scene_n, 'mask', 'mask_info_0.json')
            mask_info = readJSON(mask_info_path)
            # num_controller = int(len(mask_info.keys())-1)
            num_controller = 2
            controller_points = readPKL(tracking_pcd_path)['controller_points'][0]

            if isinstance(controller_cfg, dict):
                assert scene_n in controller_cfg.keys()
                scene_controller_cfg = controller_cfg[scene_n]
            else:
                scene_controller_cfg = controller_cfg
            
            num_level = len(scene_controller_cfg)
            unique_id = [f"nl_{num_level}"]
            for i in range(num_level):
                cluster_idx = i
                cluster_meta = scene_controller_cfg[cluster_idx]
                path_key = ''
                if 'downsample_rate' in cluster_meta.keys():
                    digit_info = str(cluster_meta['downsample_rate']).replace('.', 'd')
                    path_key = f'downsample_rate_{digit_info}'
                else:
                    assert 'num_cluster' in cluster_meta.keys()
                    digit_info = str(cluster_meta['num_cluster']).replace('.', 'd')
                    path_key = f'nc_{digit_info}'
                unique_id.append(path_key)
            unique_id = '_'.join(unique_id)
            mask_dir = os.path.join(data_dir, scene_n, 'controller_mask', cluster_type)
            os.makedirs(mask_dir, exist_ok=True)
            mask_path = os.path.join(mask_dir, f'controller_{cluster_type}{unique_id}_{controller_mask_name}')

            # if True:
            if not os.path.exists(mask_path):
                print(f"Generating controller mapping to {mask_path}")
                prev_clustered_pcs = None
                cluster_mask_list = []
                for i in range(num_level):
                    cluster_idx = i
                    cluster_meta = scene_controller_cfg[cluster_idx]
                    # path_key = ''
                    # Generate here
                    if i == 0:
                        if scene_n not in aux_pc_dict.keys():
                            aux_pc_dict[scene_n] = controller_points
                        cur_pcs = aux_pc_dict[scene_n]
                    else:
                        cur_pcs = prev_clustered_pcs
                    # Cluster
                    if cluster_type == 'dis':
                        if 'downsample_rate' in cluster_meta.keys():
                            num_cluster = int(cluster_meta['downsample_rate'] * cur_pcs.shape[0])
                        else:
                            num_cluster = cluster_meta['num_cluster']

                        if i == num_level - 1:
                            num_cluster = num_controller

                        num_cluster = max(num_cluster, 1)
                        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(cur_pcs)
                        new_pcs = kmeans.cluster_centers_
                        pc_labels = kmeans.labels_
                    elif cluster_type == 'dis_split':
                        if i == 0:
                            # 第一层：均匀分布 [0,0, 1,1, 2,2, ...] 顺序排列
                            n_points = cur_pcs.shape[0]
                            if 'downsample_rate' in cluster_meta.keys():
                                num_cluster = int(cluster_meta['downsample_rate'] * n_points)
                            else:
                                num_cluster = cluster_meta['num_cluster']
                            num_cluster = max(num_cluster, 1)
                            
                            # 均匀分配点到各个cluster：每个cluster连续的点
                            pc_labels = np.zeros(n_points, dtype=int)
                            points_per_cluster = n_points // num_cluster
                            extra_points = n_points % num_cluster
                            
                            start_idx = 0
                            new_pcs = []
                            for c in range(num_cluster):
                                # 前extra_points个cluster多分配1个点
                                cluster_size = points_per_cluster + (1 if c < extra_points else 0)
                                end_idx = start_idx + cluster_size
                                
                                # 为这个cluster的点分配标签
                                pc_labels[start_idx:end_idx] = c
                                
                                # 计算这个cluster的中心点
                                cluster_points = cur_pcs[start_idx:end_idx]
                                cluster_center = np.mean(cluster_points, axis=0)
                                new_pcs.append(cluster_center)
                                
                                start_idx = end_idx
                            
                            new_pcs = np.array(new_pcs)
                        else:
                            # 其他层（包括最后一层）：0.5分割
                            n_points = cur_pcs.shape[0]
                            split_idx = n_points // 2  # 整数除法，取前一半
                            # 创建标签：前一半为0，后一半为1
                            pc_labels = np.zeros(n_points, dtype=int)
                            pc_labels[split_idx:] = 1
                            
                            # 计算两个cluster的中心点
                            cluster_0_points = cur_pcs[:split_idx]
                            cluster_1_points = cur_pcs[split_idx:]
                            
                            cluster_0_center = np.mean(cluster_0_points, axis=0)
                            cluster_1_center = np.mean(cluster_1_points, axis=0)
                            
                            new_pcs = np.array([cluster_0_center, cluster_1_center])
                    else:
                        assert False, "Invalid cluster type"
                    cluster_mask_list.append(pc_labels) # labels  (n_cluster,  )
                    prev_clustered_pcs = new_pcs        # new_pcs (n_cluster, 3)
                #TODO, pinned_state?
                # pinned_state = aux_pc_dict[scene_n][self.pinmask_sclist[scene_idx][:, 0]]
                # dist = euclidean_distances(prev_clustered_pcs, pinned_state)
                # p2pin_mapping = np.argmin(dist, axis=-1)
                p2pin_mapping = np.zeros(len(prev_clustered_pcs))
                cluster_mask_list.append(p2pin_mapping)
                assert cluster_mask_list[-1].shape[0] == num_controller, "clustering num_controller is not correct"
                print(f"Saving cluster mapping list to {mask_path}")
                writePKL(mask_path, dict(p2c=cluster_mask_list))
            scene_cluster = readPKL(mask_path)['p2c']
            controller_clustermask_sclist.append(scene_cluster)
            # self._save_cluster_image(save_path=mask_dir, clustermask_sclist=clustermask_sclist, )
        return controller_clustermask_sclist
            

    
    # functions for skeleton clustering
    def _cosine_similarity(self, vec1, vec2):
        # Compute the cosine similarity between two vectors.
        dot_product = np.dot(vec1, vec2)
        return dot_product / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _compute_spatial_angle(self, pos1, pos2):
        pos1 = np.array(pos1, dtype=np.float32)
        pos2 = np.array(pos2, dtype=np.float32)
        vec = pos2 - pos1
        norm = np.linalg.norm(vec, 2)
        return vec / norm

    def _compute_joint_anchor_angle(self, cluster_points, joint_points):
        joint_anchor_dict = defaultdict(dict)
        for anchor_name, anchor_pos in joint_points.items():
            for cluster_name, cluster_poses in cluster_points.items():
                if cluster_name not in anchor_name:
                    continue
                joint_anchor_dict[anchor_name][cluster_name] = []
                # Compute the angle between the anchor point and the cluster point
                for cluster_pos in cluster_poses:
                    angle = self._compute_spatial_angle(anchor_pos, cluster_pos)
                    joint_anchor_dict[anchor_name][cluster_name].append(angle)
        return joint_anchor_dict
    
    #TODO
    def _load_pointcloud_mask(self, data_dir, meta_dict_sclist, cln_sc_pc_name='clean_object_points.ply'):
        pass        
    
    def _rotation_augmentation():
        pass

    def _horizontal_flip_augmentation(self, gravity, images, videos):
        '''
        # Must apply this before rotation if possible
        Assume the x-axis is vertical to the image plane; Must use the camera direction as x_axis
        images: n_cam, H, W, 3
        videos: n_cam, n_frame, H, W, 3
        '''
        gravity[1] *= -1

        pass

    def find_obje_idx(self, scene_name):
        for i, scene in enumerate(self.scene_list):
            if scene == scene_name:
                return i
        return -1

    def __getitem__(self, idx):
        scene_idx, seq_name, cam_list_idx = self.idx_mapping[idx]
        seq_idx = int(seq_name.replace('seq_', '')) # This one start from 0, the seq_name start from 0
        assert seq_idx >= 0
        # Dynamic padding
        if self.env_cfg['pad_cam']:
            extra_cam = len(cam_list_idx) % self.max_cam_num
            num_pad = 0
            if extra_cam > 0:
                num_pad = self.max_cam_num - extra_cam
            if num_pad > 0:
                if self.phase == 'train' or self.phase == 'all':
                    replace = False
                    if num_pad > len(cam_list_idx):
                        replace = True
                    pad_candidate = np.random.choice(cam_list_idx, size=num_pad, replace=replace).tolist()
                else:
                    pad_candidate = [cam_list_idx[-1] for _ in range(num_pad)]
                for pcandidate in pad_candidate:
                    cam_list_idx.append(pcandidate)
        cam_list = [self.camera_list_sclist[scene_idx][seq_name][i] for i in cam_list_idx]
        rgba_list = [self.np_uint8_rgba_list_sclist[scene_idx][seq_name][i] for i in cam_list_idx]
        
        img_list, video_list, mask_clip_list, full_video_list, robot_mask_clip_list = [], [], [], [], []
        mask_object_clip_list = []
        white_bg_list = []
        # video_range = []
        split_list = self.split_list[self.scene_list[scene_idx]]
        video_range = list(range(split_list[seq_idx][0], split_list[seq_idx][1], self.frame_gap))
        
        #TODO: set maximum input video; for now load all videos
        for cam, rgba in zip(cam_list, rgba_list):
            img_path = cam.img_path
            static_img_path = cam.static_img_path

            #load rest states
            if not self.load_imgs:
                rgba = read_uint8_rgba(static_img_path, self.resolution)
                # rgba = read_uint8_jpg(static_img_path, self.resolution)
                if self.resolution is None:
                    self.resolution = rgba.shape[:2]
            assert self.sub_img_dir in img_path
            # video_path = img_path.replace(self.sub_img_dir, self.sub_video_dir)
            base_dir = img_path.split('color')[0]
            seq_name = img_path.split('color')[1].split('/')[1]
            video_dir = os.path.join(base_dir, 'color', seq_name)
            mask_dir = os.path.join(base_dir, 'mask', seq_name)

            scene_use_random_background = self.use_random_background[self.meta_dict_sclist[scene_idx]['scene_name']] if self.use_random_background is not None else False
            scene_const_white_bg = self.const_white_bg[self.meta_dict_sclist[scene_idx]['scene_name']] if self.const_white_bg is not None else False
            assert (not scene_use_random_background and not scene_const_white_bg) or (scene_use_random_background ^ scene_const_white_bg)
            white_bg = np.random.rand() > 0.5 and scene_use_random_background
            if scene_const_white_bg:
                white_bg = True
            white_bg_list.append(white_bg)
            # [nf, 3, H, W], RGB, [0, 255]
            # video_dir = os.path.splitext(video_path)[0]
            start_frame = self.env_cfg.get('start_frame', 0)
            assert start_frame >= 0
            # video_reader = read_video_image_rgba_cv2
            video_reader = read_video_image_rgba_cv2_mask
            # video_range = [0] + [_ for _ in range(start_frame+1, start_frame+self.env_cfg.get('max_frame', 2))]
            video_clip_dict = video_reader(video_dir, mask_dir, video_range, self.prompt_dict, white_bg=white_bg, use_pil=self.meta_dict_sclist[scene_idx]['scene_name']=='bunny')
            
            #TODO; mask ids
            # video_clip = video_clip_dict['mask_object']
            # video_clip_fused = np.max(video_clip, axis=0)[None, :, :, :]
            
            video_clip = torch.from_numpy(video_clip_dict['mask_object']).permute(0, 3, 1, 2)
            video_clip = video_clip / 255.0

            full_video_clip = torch.from_numpy(video_clip_dict['image']).permute(0, 3, 1, 2)
            full_video_clip = full_video_clip / 255.0

            mask_clip_list_tmp = []
            robot_mask_clip_list_tmp = []
            mask_object_clip_list_tmp =[]
            for k, v in video_clip_dict.items():
                if k != 'image' and k != 'mask_object' and not k.startswith('mask_robot'):
                    mask_clip = torch.from_numpy(v).permute(0, 3, 1, 2)
                    mask_clip_list_tmp.append(mask_clip)
                if k.startswith('mask_robot'):
                    mask_clip = torch.from_numpy(v).permute(0, 3, 1, 2)
                    robot_mask_clip_list_tmp.append(mask_clip)
                # if k == 'mask_object':
                #     mask_clip = torch.from_numpy(v).permute(0, 3, 1, 2)
                #     mask_object_clip_list_tmp.append(mask_clip)
            if len(mask_clip_list_tmp) > 0:
                mask_clip_tmp = torch.stack(mask_clip_list_tmp, dim=-1)
                mask_clip_tmp = torch.sum(mask_clip_tmp, dim=-1)
                mask_clip_tmp = torch.clip(mask_clip_tmp, 0, 1)
            else:
                mask_clip_tmp = torch.zeros([video_clip.shape[0], 1, video_clip.shape[2], video_clip.shape[3]])

            if len(robot_mask_clip_list_tmp) > 0:
                robot_mask_clip_tmp = torch.stack(robot_mask_clip_list_tmp, dim=-1)
                robot_mask_clip_tmp = torch.sum(robot_mask_clip_tmp, dim=-1)
                robot_mask_clip_tmp = torch.clip(robot_mask_clip_tmp, 0, 1)
            else:
                robot_mask_clip_tmp = torch.zeros([video_clip.shape[0], 1, video_clip.shape[2], video_clip.shape[3]])
            # mask_object_clip_tmp = torch.clip(mask_object_clip_list_tmp[0], 0, 1)

            norm_data = rgba / 255.0
            img = norm_data[:, :, :3] * norm_data[:, :, 3:4]
            if white_bg:
                img += np.ones_like(img) * (1-norm_data[:, :, 3:4])
                img = np.clip(img, 0, 1.0)
            
            # shape convert from HWC to CHW
            img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)

            img_list.append(img)
            full_video_list.append(full_video_clip)
            video_list.append(video_clip)
            mask_clip_list.append(mask_clip_tmp)
            # remove this
            robot_mask_clip_list.append(robot_mask_clip_tmp)
            # mask_object_clip_list.append(mask_object_clip_tmp)
        
        base_dir = img_path.split('color')[0]
        tracking_pcd_path = os.path.join(base_dir, self.tracking_pcd)
        controller_points = readPKL(tracking_pcd_path)['controller_points'][video_range]
        
        gravity = self.gravity
        if self.real_dt is not None:
            if isinstance(self.real_dt, dict):
                c_real_dt = self.real_dt[self.meta_dict_sclist[scene_idx]['scene_name']]
            else:
                c_real_dt = self.real_dt
            time_ratio = c_real_dt / self.comp_dt
            gravity = self.gravity * time_ratio**2
        rot_est = self.env_cfg.get("rot_est", None)
        if rot_est is not None:
            assert isinstance(rot_est, dict)
            scene_quant = rot_est[self.meta_dict_sclist[scene_idx]['scene_name']]
            scene_rot = scipy_R.from_quat(scene_quant)
            gravity = scene_rot.inv().apply(gravity).astype(np.float32)
        volume_scalar = self.env_cfg['volume_scalar']
        if isinstance(volume_scalar, dict):
            volume_scalar = self.env_cfg['volume_scalar'][self.meta_dict_sclist[scene_idx]['scene_name']]
        
        bbox = bbox_reader(video_list, self.bounding_box)
        
        # valid_cam = [0, 2]
        mask_idx = self.aligned_scene_idx if self.aligned_scene_idx != -1 else scene_idx
        ret_dict = {
            "inputs": {
                "img": img_list, # [n_cam, H, W, 3] value in [0, 1]
                "full_gt_label": full_video_list, # [n_cam, nf, 3, H, W] value in [0, 1]
                "bbox": bbox, #[n_cam, nf, 4] , [x_min, y_min, x_max, y_max]
                "controller_trajectory": controller_points,
                "controller_img_mask_list": mask_clip_list, #[n_cam, nf, 1, H, W] value in [0, 1]
                "pure_robot_img_mask_list": robot_mask_clip_list, #[n_cam, nf, 1, H, W] value in [0, 1]
                # "mask_object_list": mask_object_clip_list, #[n_cam, nf, 1, H, W] value in [0, 1]
                "gs_aligned_frame": video_range[0],
                "p2c_mapping": self.merged_clustermask_sclist[mask_idx],
                "cam": cam_list, # [n_cam, 1]
                "volume_scalar": np.array([volume_scalar], dtype=np.float32).reshape(1, 1),
                "external": gravity,
            },
            "gt_label": video_list, # [n_cam, nf, 3, H, W] value in [0, 1]
            "meta": {
                "scene_idx": torch.tensor([scene_idx]).long(),
                "scene_name": self.meta_dict_sclist[scene_idx]['scene_name'],
                "idx": torch.tensor([idx]).long(),
                "seq_idx": torch.tensor([seq_idx]).long(),
                "eval_start_frame": torch.tensor([self.env_cfg.get("eval_start_frame", 0)]).long(),
                "seq_num": torch.tensor([video_range[1] - video_range[0]]).long(),
            },
        }

        if self.use_random_background:
            ret_dict["inputs"]["white_bg"] = white_bg_list
        opacity_scalar = self.env_cfg.get("opacity_scalar", None)
        if opacity_scalar is not None:
            ret_dict["inputs"]["opacity_scalar"] = opacity_scalar
        return ret_dict

    def _read_camera_transforms_embodied(self, 
        scene_data_dir,
        extrinsics,
        intrinsics,
        image_dir,
        video_dir,
        split_list,
        img_hw=None,
        max_cam_total=-1, max_seq=-1, eval=False, llffhold=0,
    ):
        #intrinsics: intrinsics, serial_numbers, fps, 'WH', frame_num, start_step, end_step
        cam_intrinsics = intrinsics['intrinsics']
        img_hw = list(reversed(intrinsics['WH']))

        #TODO
        reading_dir = "images" if image_dir == None else image_dir
        reading_video_dir = "video_images" if video_dir == None else video_dir

        camera_dict, seq_info_dict = readEmbodiedCameras(
            cam_extrinsics=extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=os.path.join(scene_data_dir, reading_dir),
            videos_folder=os.path.join(scene_data_dir, reading_video_dir),
            img_hw=img_hw,
            scale_x_angle=self.scale_x_angle,
            suffix_replace=['.jpg', '.png'] if "_mov" in reading_dir else None,
            max_cam_total=max_cam_total, max_seq=max_seq,
            split_list=split_list, 
        )

        ret_cam_infos = defaultdict(list)
        for seq_n, cam_list in camera_dict.items():
            for cam_idx in range(len(cam_list)):
                if llffhold > 0:
                    is_training = (cam_idx+1) % llffhold != 0
                    if self.phase == 'train' and is_training:
                        ret_cam_infos[seq_n].append(cam_list[cam_idx])
                    elif self.phase != 'train' and not is_training:
                        ret_cam_infos[seq_n].append(cam_list[cam_idx])
                else:
                    ret_cam_infos[seq_n].append(cam_list[cam_idx])

        meta_dict = {
            "num_frames": intrinsics['frame_num'],
            "num_cameras": len(cam_intrinsics),
            "img_hw": img_hw,
            "seq_info": seq_info_dict,
        }
        
        return ret_cam_infos, meta_dict
    
    def filter_camera_with_renderd_frames(self, cam_list, rendered_dir):
        rendered_img_names = [_ for _ in os.listdir(rendered_dir) if _.endswith(".png")]
        rendered_img_names = [_.split(".")[0] for _ in rendered_img_names]

        for cam in cam_list:
            img_name = os.path.basename(cam.img_path).split(".")[0]
            if img_name not in rendered_img_names:
                cam_list.remove(cam)
        return cam_list  
    
    def save_camera_list(self, cam_list, save_path):
        # R=R,
        # T=T,
        # FoVy=FovY,
        # FoVx=FovX,
        # img_path=img_path,
        # image_height=image_height,
        # image_width=image_width,
        camera_list = []
        assert save_path.endswith(".json"), "save_path should be a json file"

        for cam in cam_list:
            cam_dict = {
                "R": cam.R.tolist(),
                "T": cam.T.tolist(),
                "FoVy": cam.FoVy,
                "FoVx": cam.FoVx,
                "img_path": cam.img_path,
                "image_height": cam.image_height,
                "image_width": cam.image_width,
            }
            camera_list.append(cam_dict)

        with open(save_path, "w") as f:
            json.dump(camera_list, f, indent=4)

    def interpolate_camera(self, filename1, filename2, num_frames):
        for cam in self.camera_list:
            img_name = os.path.basename(cam.img_path).split(".")[0]
            if filename1.startswith(img_name):
                cam1 = cam
            if filename2.startswith(img_name):
                cam2 = cam

        interpolated_cameras = cam1.interpolate(cam2, num_frames - 1)
        return interpolated_cameras
    
    def collate(self, batch, samples_per_gpu=1):
        assert len(batch) == 1, "Batch size could only be 1 for now"
        batch_cam = batch[0]['inputs'].pop('cam')
        scene_name = batch[0]['meta'].pop('scene_name')
        batched_data = collate(batch, samples_per_gpu=samples_per_gpu)
        batched_data['inputs']['cam'] = batch_cam
        batched_data['meta']['scene_name'] = scene_name
        return batched_data
    
    def evaluate_frame(self,
        results, metric=None, logger=None, **kwargs):
        eval_results = defaultdict(list)
        for rst in results:
            acc = rst['acc']
            for key, val in acc.items():
                eval_results[key].append(val)
        
        # Mean results
        collate_results = dict()
        for g_name, g_list in eval_results.items():
            collate_results[g_name] = dict(
                mean=np.mean(g_list),
                std=np.std(g_list),
            )
        return collate_results
    
    def evaluate_rollout(self,
                 results,
                 metric=None,
                 metric_options=None,
                 logger=None,
                 **kwargs):

        def collate_rollout(data_dict, prefix='rollout'):
            collate_dict = defaultdict(list)
            for rollout_idx, val in data_dict.items():
                cur_rst = defaultdict(list)
                for acc_entry in val:
                    for acc_key, acc_val in acc_entry.items():
                        cur_rst[acc_key].append(acc_val)
                for acc_key, val in cur_rst.items():
                    collate_dict[acc_key].append(np.mean(val))
            rst = dict()
            for key, val in collate_dict.items():
                rst[key] = {
                    f'{prefix}_mean': np.mean(val),
                    f'{prefix}_std': np.std(val)
                }
            return rst
        
        collate_results = dict()

        scene_wise_results = dict() # If only one scene, then this is equal to the following one
        rollout_wise_results = defaultdict(list)
        # Collect results
        for rst in results:
            rollout_idx = rst['rollout_idx'][0]
            scene_name = rst['scene_name']
            rollout_wise_results[rollout_idx].append(rst['acc'])
            if scene_name not in scene_wise_results.keys():
                scene_wise_results[scene_name] = defaultdict(list)
            scene_wise_results[scene_name][rollout_idx].append(rst['acc'])
            
        # Calculate rollout level results
        eval_rollout = dict()
        eval_rollout['whole'] = collate_rollout(rollout_wise_results, prefix='whole')
        for key, val in scene_wise_results.items():
            eval_rollout[key] = collate_rollout(val, prefix=key)

        # # Save into logger
        # for rollout_idx, rollout_rst in eval_rollout.items():
        #     logger.info(f"Rollout {rollout_idx} {metric_key}: mean: {np.mean(rollout_rst)}; std: {np.std(rollout_rst)}")

        return eval_rollout
    
    def evaluate(self,
                 results,
                 metric=None,
                 metric_options=None,
                 logger=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict): Options for calculating metrics. Allowed
                keys are 'topk', 'thrs' and 'average_mode'.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict: evaluation results
        """
        perframe_rst = self.evaluate_frame(results=results, metric=metric, metric_options=metric_options, logger=logger, **kwargs)
        rst = dict()
        rst['per_frame'] = perframe_rst
        if self.env_cfg.get("rollout", False):
            perrollout_rst = self.evaluate_rollout(results=results, metric=metric, metric_options=metric_options, logger=logger, **kwargs)
            rst['per_rollout'] = perrollout_rst
        return rst
    
    def inject_external_forces(self, prev_state, query_pos, radius, decay_radius, delta_x):
        # Find selected verts
        dist = torch.linalg.norm(prev_state - query_pos, dim=-1, keepdim=True)
        # nearest_pos = prev_state[torch.argmin(nn_dist, dim=0)]
        # dist = torch.linalg.norm(prev_state - nearest_pos, dim=-1, keepdim=True)
        selected_mask = dist < radius
        # Apply the forces with decay according to the l2 distance
        assert decay_radius < radius and decay_radius > 0
        scalar_dist = torch.clamp_min(dist/decay_radius, 1.0)
        applied_x = delta_x / scalar_dist
        prev_state_updated = prev_state - selected_mask * applied_x
        return prev_state_updated
    
    def inject_external_forces_tocur(self, cur_state, query_pos, radius, decay_radius, delta_x):
        # Find selected verts
        dist = torch.linalg.norm(cur_state - query_pos, dim=-1, keepdim=True)
        # nearest_pos = prev_state[torch.argmin(nn_dist, dim=0)]
        # dist = torch.linalg.norm(prev_state - nearest_pos, dim=-1, keepdim=True)
        selected_mask = dist < radius
        # Apply the forces with decay according to the l2 distance
        assert decay_radius < radius and decay_radius > 0
        scalar_dist = torch.clamp_min(dist/decay_radius, 1.0)
        applied_x = delta_x / scalar_dist
        cur_state_updated = cur_state + selected_mask * applied_x
        return cur_state_updated
    
    def rot90_ccw_unit_3d(self, a: torch.Tensor, n: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        assert a.shape[-1] == 3 and n.shape[-1] == 3
        n = n / torch.linalg.norm(n, dim=-1, keepdim=True).clamp_min(eps)
        a_norm = torch.linalg.norm(a, dim=-1, keepdim=True).clamp_min(eps)
        a_par = (a * n).sum(dim=-1, keepdim=True) * n
        a_perp = a - a_par
        a_rot = a_par + torch.cross(n, a_perp, dim=-1)  # 90° CCW about n
        return a_rot / a_norm

    # def inject_external_forces_drag(self, cur_state, query_pos, radius, decay_radius, delta_x, pin_mask):
    #     # Find selected 
    #     pin_node_id = pin_mask.nonzero(as_tuple=True)[0]
    #     pin_position = cur_state[pin_node_id]
    #     R = query_pos - pin_position
    #     n = torch.cross(R, torch.tensor(delta_x), dim=-1)
    #     dist_pin = torch.linalg.norm(R, dim=-1, keepdim=True)
    #     theta = torch.linalg.norm(torch.tensor(delta_x), dim=-1) / dist_pin

    #     dist = torch.linalg.norm(cur_state - pin_position, dim=-1, keepdim=True)
    #     rot = theta * dist

        
    #     rot_vec = self.rot90_ccw_unit_3d(cur_state, n)

    #     applied_x = rot * rot_vec
    #     cur_state_updated = cur_state + applied_x
    #     return cur_state_updated.squeeze()


def camera_dataset_collate_fn(batch):
    ret_dict = {
        "cam": [],
        "img_name": [],
    }

    for key in batch[0].keys():
        if key == "cam":
            ret_dict[key].extend([item[key] for item in batch])
        elif key == "img_name":
            ret_dict[key].extend([item[key] for item in batch])
        elif key == "timestamp":
            ret_dict[key] = torch.tensor([item[key] for item in batch])
        else:
            ret_dict[key] = torch.stack([item[key] for item in batch], dim=0)

    return ret_dict


def create_camera(dataset_dir, save_path, *args):
    fname1, fname2, fname3, num_frames = args
    num_frames_each = int(num_frames / 3)

    dataset = MultiviewVideoDataset(dataset_dir, load_imgs=False)

    cam_AB = dataset.interpolate_camera(fname1, fname2, num_frames_each)
    cam_BC = dataset.interpolate_camera(fname2, fname3, num_frames_each)
    cam_CA = dataset.interpolate_camera(fname3, fname1, num_frames_each)

    cam_list = cam_AB + cam_BC + cam_CA
    dataset.save_camera_list(cam_list, save_path)


def test_speed():
    dataset_dir = "../../../../../dataset/3D_capture/purple_branches_colmap"
    dataset_dir = "../../../../../dataset/physics_dreamer/llff_flower_undistorted"

    dataset = MultiviewVideoDataset(dataset_dir)

    data = dataset[0]

    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            print(key, val.shape)
        else:
            print(key, type(val))


if __name__ == '__main__':
    env_cfg = dict(
        data_dir='/mnt/shared-storage-user/huangmu/interactable3d/SkeletonSim/data/phystwin/different_types/',
        sub_video_dir='color',
        sub_img_dir='color',
        # cam_transform_fn='transforms_train.json',
        scene_list = [
        "rope_double_hand", "single_lift_rope", "single_push_rope", "single_push_rope_1", "single_push_rope_4",
        ],
        split_list = [
            [1, 20], [20, 30]
        ],
        
        # use_white_background=False, # only for render moving obj, no actual use for now
        resolution=[576, 1024], # Decide by loaded images, the images must have the same resolution
        scale_x_angle=1.0, # Default, TODO don't know what to use
        load_imgs=False, # Preload image or not, default True
        # load_pcs=True, # Preload point clouds to load the mask, # TODO: change this api, since only need moving points mask
        gravity=[0, 0, -9.8],
        max_cam_num=3,
        pad_cam=True, # If false, then will drop the last few cams that cannot form a batch
        max_cam_total=40,
        cluster_cfg=[
            dict(num_cluster=50),
            dict(downsample_rate=0.1),
            dict(downsample_rate=0.1)], # TODO: must align with the model's cfg
    )
    dataset = EmbodiedDataset(
        env_cfg, 'train',)

    data = dataset[0]
    
    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            print(key, val.shape)
        else:
            print(key, type(val))