# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import vmap, jacrev

import numpy as np

from .. import builder
from ..builder import SIMULATORS
from .base import BaseSimulator
from mmgs.core import add_prefix, multi_apply
from mmgs.datasets.utils import to_numpy_detach, readPKL
from mmgs.utils import TensorBoardLogger
from collections import defaultdict
from functools import partial

import dgl
import dgl.function as fn
from mmgs.models.utils.dgl_graph import CLUSTER_VERT_ID, P2C_EDGE_ID, POINT_VERT_ID, C2P_EDGE_ID
from mmgs.utils.transformation_utils import apply_cov_rotations_batched, get_shs_rotations_batched
from mmgs.utils.physdreamer_utils import apply_mask_gaussian

import os
from datetime import datetime

import sys
sys.path.append('gaussian-splatting')
from scene.gaussian_model import GaussianModel

import time
import pickle

@SIMULATORS.register_module()
class GsSimulatorEmbodiedS2(BaseSimulator):
    """Encoder Decoder SIMULATORS.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 gs_scene,
                 processor_cfg,
                 cluster_cfg,
                #  neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 accumulate_gradient=False,
                 forward_last_layer=False,
                 share_weight=False,
                 opt_sim=True,
                 opt_vel=False,
                 dt=1/30,
                 static_loss=False,
                 checkpoint_rollout=25,
                 selfsup_loss=False,
                 selfsup_velonly=False,
                 avg_loss=False,
                 render_mov_only=False, # Render only the moving target object or not
                 data_aug=False,
                 fix_bug=False,
                 flag_save_gaussian=False,
                 flag_update_gaussian=False,
                 flag_update_gaussian_train=False,
                 data_dir=None,
                 frame_gap=1,
                 use_rotation=True,
                 pred_vel=False,
                 attr_init=False,
                 force_forward=-1,
                 test_rollout_mode='segmented',
                 **kwargs):
        super(GsSimulatorEmbodiedS2, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and simulator set pretrained weight'
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
            # TODO: check pretrain
            # backbone.pretrained = pretrained
        # Update common config
        backbone['dt'] = dt
        decode_head['dt'] = dt
        self.dt = dt
        # 1 for mass;
        self.attr_dim = backbone['attr_dim']
        self.attr_init = attr_init
        self.avg_loss = avg_loss
        self.render_mov_only = render_mov_only
        self.fix_bug = fix_bug
        self.use_rotation = use_rotation
        self.pred_vel = pred_vel
        self.force_forward = force_forward
        assert test_rollout_mode in ['segmented', 'continuous']
        self.test_rollout_mode = test_rollout_mode

        self._init_gs_scene(gs_scene)
        self.iters_per_epoch = None
        self.frame_gap = frame_gap

        # Model init
        self.cluster_cfg = cluster_cfg
        self.forward_last_layer = forward_last_layer
        self.share_weight = share_weight

        if share_weight:
            self.backbone = builder.build_backbone(backbone) # TODO: uncomment this
            # if neck is not None:
            #     self.neck = builder.build_neck(neck)
            self.decode_head = self._init_decode_head(decode_head)
        else:
            # Check 
            num_model = None
            for key, val in cluster_cfg.items():
                cur_len = len(val)
                if num_model == None:
                    num_model = cur_len
                else:
                    assert num_model == cur_len
            if forward_last_layer:
                num_model += 1
            self.backbone = nn.ModuleList([builder.build_backbone(backbone) for _ in range(num_model)])
            self.decode_head = nn.ModuleList([self._init_decode_head(decode_head) for _ in range(num_model)])

        if isinstance(processor_cfg, dict):
            processor_cfg = [processor_cfg]
        generator_cfg = processor_cfg[0]
        self.graph_generator = builder.build_preprocessor(generator_cfg)
        self.preprocessor = []
        for i in range(1, len(processor_cfg)):
            p_cfg = processor_cfg[i]
            self.preprocessor.append(builder.build_preprocessor(p_cfg))

        # This is for preprocess/augment train input
        self.train_cfg = train_cfg
        # This is for preprocess/augment test input
        self.test_cfg = test_cfg
        self.flag_save_gaussian = flag_save_gaussian
        self.flag_update_gaussian = flag_update_gaussian
        self.flag_update_gaussian_train = flag_update_gaussian_train
        self.flag_update = False
        self.data_dir = data_dir
        self.frame_gap = frame_gap

        # Try only sim first, then vel as second stage
        assert opt_sim or opt_vel
        self.opt_sim = opt_sim
        self.opt_vel = opt_vel
        # Estimating the initial velocity first
        self.is_est_vel = True
        self.num_iter = 0
        self.num_epoch = 0
        self.static_loss = static_loss
        self.checkpoint_rollout = checkpoint_rollout
        self.selfsup_loss = selfsup_loss
        self.selfsup_velonly = selfsup_velonly
        if selfsup_velonly:
            assert selfsup_loss and opt_vel
        self.data_aug = data_aug
        
        # To control the autoregressive
        self.accumulate_gradient = accumulate_gradient
        assert self.with_decode_head
        self._init_scene_init_pos_cov(gs_scene)

        # self.enable_tensorboard = kwargs.get('enable_tensorboard', True)
        # self.tb_log_dir = kwargs.get('tb_log_dir', '/mnt/shared-storage-user/huangmu/interactable3d/SkeletonSim/tensorboard_logs')


    def _initial_tensorboard(self, gs_scene):
        #initial tensorboard logger
        self.tb_logger = None
        self.enable_tensorboard = True
        self.tb_log_dir = 'tensorboard_logs'
        scene_name_list = [scene_cfg['name'] for scene_cfg in gs_scene]
        scene_name = f"{scene_name_list[0]}_num_{len(scene_name_list)}"
        #2. 创建date字符串：YYYYMMDD_HHMMSS
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(self.tb_log_dir, scene_name, date_str)
        os.makedirs(log_dir, exist_ok=True)
        

        if self.enable_tensorboard:
            self.tb_logger = TensorBoardLogger(
                log_dir=log_dir,
                enabled=True
                )
        self.global_step = 0


    def _init_gs_scene(self, gs_scene):
        self.gs_scene_dict = dict() # Only provide initial states and for rendering
        
        ## mass, attr_dim
        self.scene_attr = nn.ParameterDict()
        self._scene_attr_dict = dict()

        for scene_idx, scene_cfg in enumerate(gs_scene):
            # Load gaussian scene
            gaussian = GaussianModel(scene_cfg['sh_degree'])
            print(f"Initializing gaussian scene: {scene_cfg['model_path']}")
            gaussian.load_ply(scene_cfg['model_path'])
            self.gs_scene_dict[scene_cfg['name']] = gaussian
            self._scene_attr_dict[scene_cfg['name']] = scene_cfg['attr_name']
            init_pos = gaussian.get_xyz
            if scene_cfg['attr_name'] not in self.scene_attr.keys():
                attr_vec = torch.zeros((1, self.attr_dim)).to(init_pos)
                if self.attr_init:
                    attr_vec[:, 1:] += (scene_idx/len(gs_scene))
                self.scene_attr[scene_cfg['attr_name']] = nn.Parameter(attr_vec, requires_grad=True)
        return
    
    def _init_scene_init_pos_cov(self, gs_scene):
        self.scene_init_pos = dict()
        self.scene_init_prev_pos = dict()
        self.scene_init_cov = dict()
        if self.test_rollout_mode == 'continuous':
            self._init_scene_init_pos_cov_from_raw_gaussian(gs_scene)
            return
        for scene_idx, scene_cfg in enumerate(gs_scene):
            scene_init_dir = os.path.join(self.data_dir, scene_cfg['name'], 'pred_stage1')
            pred_data_files = os.listdir(scene_init_dir)
            for pred_data_file in pred_data_files:
                if not pred_data_file.endswith('.pth'):
                    continue
                pred_data = torch.load(os.path.join(scene_init_dir, pred_data_file))
                pred_pos = pred_data['pred_pos']
                pred_cov = pred_data['pred_cov']
                pos_leaf = pred_pos.detach().clone().requires_grad_(True)
                cov_leaf = pred_cov.detach().clone().requires_grad_(True)
                # self.scene_init_pos[f"{scene_name}_frame_{real_frame_idx}"] = pos_leaf
                # self.scene_init_cov[f"{scene_name}_frame_{real_frame_idx}"] = cov_leaf
                self.scene_init_pos[f"{scene_cfg['name']}_{pred_data_file.split('.')[0]}"] = pos_leaf
                self.scene_init_prev_pos[f"{scene_cfg['name']}_{pred_data_file.split('.')[0]}"] = pos_leaf.detach().clone().requires_grad_(True)
                self.scene_init_cov[f"{scene_cfg['name']}_{pred_data_file.split('.')[0]}"] = cov_leaf
        return

    def _init_scene_init_pos_cov_from_raw_gaussian(self, gs_scene):
        for scene_idx, scene_cfg in enumerate(gs_scene):
            scene_name = scene_cfg['name']
            scene_gaussian = self.gs_scene_dict[scene_name]
            init_pos = scene_gaussian.get_xyz
            init_cov = scene_gaussian.get_covariance()
            pos_leaf = init_pos.detach().clone().requires_grad_(True)
            cov_leaf = init_cov.detach().clone().requires_grad_(True)
            key = f"{scene_name}_frame_0"
            self.scene_init_pos[key] = pos_leaf
            self.scene_init_prev_pos[key] = pos_leaf.detach().clone().requires_grad_(True)
            self.scene_init_cov[key] = cov_leaf
        return

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        # Build head
        return builder.build_head(decode_head)
        # Retrieve some configs from decode head

    def init_weights(self):
        super(GsSimulatorEmbodiedS2, self).init_weights()

        # Customized init weights
        # if not (isinstance(self.init_cfg, dict)
        #         and self.init_cfg['type'] == 'Pretrained'):
        #     trunc_normal_(self.dist_token, std=0.02)
    
    def _pre_maskout_pinverts_rollout(self, input_state, future_state, vert_mask):
        assert input_state.shape == future_state.shape
        assert input_state.shape[0] == vert_mask.shape[0]
        rst_state = input_state * vert_mask + future_state * (1-vert_mask)
        return rst_state,
    
    def _preprocess(self, prev_state, cur_state, template_state,  controller_prev_state, controller_cur_state, controller_template_state, attr, diag_volume, cur_cov, external_forces, p2c_mapping, pin_mask=None, is_training=False):
        '''
            prev_state: n_points, 3
            cur_state: n_points, 3
            material_center: 1, 3
            pin_mask: n_points, 1
        '''
        #concatenate the controller state
        merged_cur_state = torch.concatenate([controller_cur_state, cur_state], axis=0)
        merged_template_state = torch.concatenate([controller_template_state, template_state], axis=0)
        merged_prev_state = torch.concatenate([controller_prev_state, prev_state], axis=0)

        num_controller_points = controller_cur_state.shape[0]
        device = cur_state.device
        controller_attr = torch.ones((num_controller_points, attr.shape[1]), device=device, dtype=attr.dtype) * attr[0]
        controller_diag_volume = torch.ones((num_controller_points, diag_volume.shape[1]), device=device, dtype=diag_volume.dtype) * diag_volume[0]
        controller_cur_cov = torch.ones((num_controller_points, cur_cov.shape[1]), device=device, dtype=cur_cov.dtype) * cur_cov[0]
        # controller_external_forces = torch.ones((num_controller_points, external_forces.shape[1]), device=device, dtype=external_forces.dtype) * external_forces[0]

        merged_attr = torch.concatenate([controller_attr, attr], axis=0)
        merged_diag_volume = torch.concatenate([controller_diag_volume, diag_volume], axis=0)
        merged_cur_cov = torch.concatenate([controller_cur_cov, cur_cov], axis=0)
        # merged_external_forces = torch.concatenate([controller_external_forces, external_forces], axis=0)

        merged_p2c_mapping = [torch.concatenate([p2c[0], p2c[1]], axis=0) for p2c in p2c_mapping]

        built_graph, connect_graph = self.graph_generator.batch_preprocess(merged_prev_state, merged_cur_state, merged_template_state, merged_attr, merged_diag_volume, merged_cur_cov, external_forces, p2c_mapping=merged_p2c_mapping, dynamic_base=self.forward_last_layer, pin_mask=pin_mask)

        for i in range(len(self.preprocessor)):
            processor = self.preprocessor[i]
            built_graph = processor.graph_preprocess(built_graph, is_training=is_training)
        
        # graph_dict = self._register_forces(graph_dict, meta_dict, state_dim=state_dim)
        return built_graph, connect_graph
    
    def _postprocess(self, inputs, pred):
        return pred
    
    def extract_feat(self, backbone_model, input_graph, connect_graph):
        """Extract features from inputs."""
        x = backbone_model(input_graph, connect_graph)
        # if self.with_neck:
        #     x = self.neck(x)
        return x
    
    def encode_decode(self, input_graph, connect_graph, gaussians, cam_list, scene_cov3D, scene_pcs, num_controller_points, register_norm=False, gt_label=None, rollout_size=1, cln_mask=None, cln_gaussian=None, white_bg_list=None, opacity_scalar=None):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.
            connect_graph[i] is for in_graph[i], for moving things(anchor_next_state)  from last(upper) hierachy.
            in_graph: 0 is the leaf, 1 is cluster one, 2 is the root cluster;

        """
        losses = dict()
        outF_lv_list = []
        outDG_lv_list = []
        for i in range(1,len(input_graph)):
            if self.force_forward > 0 and i >= self.force_forward:
                break
            lv_idx = len(input_graph)-1 - i
            if not self.forward_last_layer and lv_idx <= 0:
                break
            if self.share_weight:
                backbone_model = self.backbone
                decode_head = self.decode_head
            else:
                backbone_model = self.backbone[i]
                decode_head = self.decode_head[i]
            # Get the output prediction
            lv_in_g, lv_in_connect_g = input_graph[lv_idx], connect_graph[lv_idx]
            if i == 1:
                # First step initialize
                lv_in_g.ndata['anchor_next_state'] = lv_in_g.ndata['anchor_cur_state']
                lv_in_connect_g.ndata['anchor_next_state'] = lv_in_connect_g.ndata['anchor_cur_state']
            # TODO: encg_lv_t0 is only a pointer? Check the values in lv_in_g
            ## Should be the pointer
            ## TODO-self.extract_feat
            encg_lv_t0 = self.extract_feat(backbone_model, lv_in_g, lv_in_connect_g)
            # The pin verts are only valid for the point clouds's level (level_idx=0, i=len()-1)
            outx_lv_t1, outF_lv_t1, outFmat_lv_t1, lv_out_g = decode_head.pre_predict(encg_lv_t0, register_norm=register_norm, apply_pin=True)

            # out_g_list = out_g_list + [lv_out_g]
            if lv_idx > 0:
                # anchor_next_state is first defined here
                # Update anchor next state for next graph
                next_in_g = input_graph[lv_idx-1]
                next_connect_g = connect_graph[lv_idx-1]
                next_cluster_nids = torch.nonzero(next_connect_g.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
                next_point_nids = torch.nonzero(next_connect_g.ndata[POINT_VERT_ID][:, 0], as_tuple=False).squeeze()
                next_c2p_eids = torch.nonzero(next_connect_g.edata[C2P_EDGE_ID][:, 0], as_tuple=False).squeeze()
                next_connect_g.nodes[next_cluster_nids].data['anchor_next_state'] = lv_out_g.ndata['pred_pos']
                # This step is to distribute, both sum and mean work actually, only distribute the value
                next_connect_g.send_and_recv(next_c2p_eids, fn.copy_u('anchor_next_state', 'anchor_next_state'), fn.mean('anchor_next_state', 'anchor_next_state'))
                next_in_g.ndata['anchor_next_state'] = next_connect_g.nodes[next_point_nids].data['anchor_next_state']
                next_in_g.ndata['anchor_template_state'] = next_in_g.ndata['anchor_next_state'][:, :] # Update once the anchor_next_state is updated, only affect the neighbor one
            if i > 1:
                # lv_idx < len(input_graph)-1
                # Only apply the loss to real clusters, the top root graph is forced with pinned nodes, thus the momentum is not guarenteed
                lv_loss = decode_head.forward_train_regularize(outx_lv_t1, outF_lv_t1, lv_out_g, lv_in_connect_g)
                losses.update(add_prefix(lv_loss, 'decode'))
            
            # print(f"==> Forward to next level: {time.time()-timer}")
            # timer = time.time()
            # Broad cast to next level and future level
            for j in range(i+1, len(input_graph)):
                # Forward to all level
                ## hie_xxx variables only effective within this loop
                bc_lv_idx = len(input_graph)-1 - j
                bc_cur_connect_g = connect_graph[bc_lv_idx]
                point_nids = torch.nonzero(bc_cur_connect_g.ndata[POINT_VERT_ID][:, 0], as_tuple=False).squeeze()
                cluster_nids = torch.nonzero(bc_cur_connect_g.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
                c2p_eids = torch.nonzero(bc_cur_connect_g.edata[C2P_EDGE_ID][:, 0], as_tuple=False).squeeze()
                bc_g = input_graph[bc_lv_idx]
                # Forward the root anchor's info from last connect graph
                if j == i+1:
                    # Initialize the first prop
                    ## hie_anchor_next_state is different from anchor_next_state;
                    ## anchor_next_state is only affecting the current layer; hie_anchor_next_state affect all layers;
                    ## anchor_next_state is the last layer's new pos; hie_anchor_next_state starts from last last layer's pos
                    # if j == 1: lv_in_g.ndata['anchor_cur_state']-lv_in_g.ndata['anchor_next_state'] == 0
                    bc_cur_connect_g.nodes[cluster_nids].data['hie_anchor_next_state'] = lv_in_g.ndata['anchor_next_state']
                    bc_cur_connect_g.nodes[cluster_nids].data['hie_anchor_template_state'] = lv_in_g.ndata['anchor_template_state']
                    bc_cur_connect_g.nodes[cluster_nids].data['hie_pred_dg_mat'] = lv_out_g.ndata['pred_dg_mat']
                    bc_cur_connect_g.nodes[cluster_nids].data['hie_pred_dg'] = lv_out_g.ndata['pred_dg']
                bc_cur_connect_g.send_and_recv(c2p_eids, fn.copy_u('hie_anchor_next_state', 'hie_anchor_next_state'), fn.mean('hie_anchor_next_state', 'hie_anchor_next_state'))
                bc_cur_connect_g.send_and_recv(c2p_eids, fn.copy_u('hie_anchor_template_state', 'hie_anchor_template_state'), fn.mean('hie_anchor_template_state', 'hie_anchor_template_state'))
                bc_cur_connect_g.send_and_recv(c2p_eids, fn.copy_u('hie_pred_dg_mat', 'hie_pred_dg_mat'), fn.mean('hie_pred_dg_mat', 'hie_pred_dg_mat'))
                bc_cur_connect_g.send_and_recv(c2p_eids, fn.copy_u('hie_pred_dg', 'hie_pred_dg'), fn.mean('hie_pred_dg', 'hie_pred_dg'))
                # Assign to the compute graph
                bc_g.ndata['hie_anchor_next_state'] = bc_cur_connect_g.nodes[point_nids].data['hie_anchor_next_state']
                bc_g.ndata['hie_anchor_template_state'] = bc_cur_connect_g.nodes[point_nids].data['hie_anchor_template_state']
                bc_g.ndata['hie_pred_dg_mat'] = bc_cur_connect_g.nodes[point_nids].data['hie_pred_dg_mat']
                bc_g.ndata['hie_pred_dg'] = bc_cur_connect_g.nodes[point_nids].data['hie_pred_dg']
                # Update current state
                if bc_lv_idx >= 0:
                    # This are still hierarchical levels, the pin is invalid for these levels
                    bc_g.ndata['cur_state'] = bc_g.ndata['hie_anchor_next_state'] + torch.bmm(bc_g.ndata['hie_pred_dg_mat'], (bc_g.ndata['template_state']-bc_g.ndata['hie_anchor_template_state']).unsqueeze(-1)).squeeze(-1)
                # else:
                #     bc_g.ndata['cur_state'] = torch.logical_not(bc_g.ndata['pin_mask']) * (bc_g.ndata['hie_anchor_next_state'] + torch.bmm(bc_g.ndata['hie_pred_dg_mat'], (bc_g.ndata['template_state']-bc_g.ndata['hie_anchor_template_state']).unsqueeze(-1)).squeeze(-1)) + bc_g.ndata['pin_mask'] * bc_g.ndata['cur_state']
                if self.fix_bug:
                    bc_g.ndata['template_state'] = bc_g.ndata['cur_state'][:, :3]
                if bc_lv_idx <= 0:
                    # Finish already
                    break
                # Forward to the next level connect graph
                bc_next_connect_g = connect_graph[bc_lv_idx-1]
                cur_cluster_nids = torch.nonzero(bc_next_connect_g.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
                bc_next_connect_g.nodes[cur_cluster_nids].data['hie_anchor_next_state'] = bc_g.ndata['hie_anchor_next_state']
                bc_next_connect_g.nodes[cur_cluster_nids].data['hie_anchor_template_state'] = bc_g.ndata['hie_anchor_template_state']
                bc_next_connect_g.nodes[cur_cluster_nids].data['hie_pred_dg_mat'] = bc_g.ndata['hie_pred_dg_mat']
                bc_next_connect_g.nodes[cur_cluster_nids].data['hie_pred_dg'] = bc_g.ndata['hie_pred_dg']
            # Update the covariance for the points
            if self.forward_last_layer and lv_idx <= 0:
                input_graph[0].ndata['hie_pred_dg_mat'] = lv_out_g.ndata['pred_dg_mat']
                input_graph[0].ndata['hie_pred_dg'] = lv_out_g.ndata['pred_dg']
            if i == 1:
                skeleton_dict = {
                    'F':lv_out_g.ndata['pred_dg_mat'],
                    'G':lv_out_g.ndata['pred_dg']
                }
            outF_lv_list = [input_graph[0].ndata['hie_pred_dg_mat']] + outF_lv_list
            outDG_lv_list = [input_graph[0].ndata['hie_pred_dg']] + outDG_lv_list
        
        # Update the covariance for the points
        input_graph[0].ndata['cur_cov'] = apply_cov_rotations_batched(input_graph[0].ndata['cur_cov'], outF_lv_list, inverse=False)
        ## Here this rot need transpose, since the eq is R^T d; the code only Rd, thus transpose here
        pred_rot = get_shs_rotations_batched(outDG_lv_list, inverse=True)
        # Render loss
        if self.share_weight:
            decode_head = self.decode_head
        else:
            decode_head = self.decode_head[-1]
        
        pred_cov = input_graph[0].ndata['cur_cov']
        pred_pos = input_graph[0].ndata['cur_state'] if not self.forward_last_layer else input_graph[0].ndata['pred_pos']
        scene_rot = None
        if not self.render_mov_only:
            r_gaussians = gaussians
            # Merge cov3D and pos
            scene_cov3D = pred_cov[num_controller_points:, ...]
            scene_pcs = pred_pos[num_controller_points:, ...]
            
            if self.use_rotation:
                scene_rot = torch.eye(pred_rot[num_controller_points:, ...].shape[-1]).to(pred_rot).expand(scene_pcs.shape[0], -1, -1).clone()
                scene_rot = pred_rot[num_controller_points:, ...]
        else:
            # Only apply to training and testing
            assert cln_gaussian is not None
            r_gaussians = cln_gaussian
            scene_cov3D = pred_cov[num_controller_points:, ...]
            scene_pcs = pred_pos[num_controller_points:, ...]
            # scene_cov3D = scene_cov3D[cln_mask, ...]
            # scene_pcs = scene_pcs[cln_mask, ...]

            if self.use_rotation:
                scene_rot = torch.eye(pred_rot[num_controller_points:, ...].shape[-1]).to(pred_rot).expand(scene_pcs.shape[0], -1, -1).clone()
                scene_rot = pred_rot[num_controller_points:, ...]
        
        scene_opacity = r_gaussians.get_opacity
        if opacity_scalar is not None and not self.render_mov_only:
            fg_op = scene_opacity
            fg_op *= opacity_scalar
            scene_opacity = fg_op

        # Only per-camera need to form as list
        results = multi_apply(
            decode_head.pre_render,
            cam_list,
            gt_label if gt_label is not None else [None]*len(cam_list), # This one is for debug
            white_bg_list if white_bg_list is not None else [None]*len(cam_list),
            scene_gaussian=r_gaussians, cov3D_precomp=scene_cov3D, pos=scene_pcs, opacity=scene_opacity, rotation=scene_rot, return_tuple=True)
        img_list = results[0]
        depth_list = results[1]
        return pred_pos, pred_cov, outF_lv_list, skeleton_dict, img_list, depth_list, losses
    
    def save_gaussian(self, scene_name, frame_idx, pred_pos, pred_cov):
        real_frame_idx = frame_idx #* self.frame_gap
        save_dir = os.path.join(self.data_dir, scene_name, 'pred_stage2')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'frame_{real_frame_idx}.pth')

        torch.save({
            'pred_pos': pred_pos,
            'pred_cov': pred_cov
        }, save_path)
    
    def update_gaussian(self, scene_name, frame_idx, pred_pos, pred_cov, prev_pos=None):
        real_frame_idx = frame_idx #(frame_idx+1) * self.frame_gap
        pos_leaf = pred_pos.detach().clone().requires_grad_(True)
        cov_leaf = pred_cov.detach().clone().requires_grad_(True)
        self.scene_init_pos[f"{scene_name}_frame_{real_frame_idx}"] = pos_leaf
        self.scene_init_cov[f"{scene_name}_frame_{real_frame_idx}"] = cov_leaf
        if prev_pos is not None:
            prev_pos_leaf = prev_pos.detach().clone().requires_grad_(True)
            self.scene_init_prev_pos[f"{scene_name}_frame_{real_frame_idx}"] = prev_pos_leaf
        # self.scene_init_pos[f"{scene_name}_frame_{real_frame_idx}"] = pred_pos
        # self.scene_init_cov[f"{scene_name}_frame_{real_frame_idx}"] = pred_cov
        return

    def encode_decode_render_only(self, input_graph, connect_graph, gaussians, cam_list, scene_cov3D, scene_pcs, num_controller_points, register_norm=False, gt_label=None, rollout_size=1, cln_mask=None, cln_gaussian=None, white_bg_list=None, opacity_scalar=None):
        """Render only version of encode_decode - skips all physics simulation and only renders current state images.
        
        Same inputs as encode_decode, but only performs rendering without any computation.
        
        Returns:
            img_list: rendered images for each camera
        """
        # Skip all physics simulation computation
        # Directly use the provided scene_cov3D and scene_pcs for rendering
        
        # Determine which gaussian model to use for rendering
        if not self.render_mov_only:
            r_gaussians = gaussians
            # Use the provided scene_cov3D and scene_pcs directly (they should already be sliced appropriately)
            scene_cov3D_render = scene_cov3D
            scene_pcs_render = scene_pcs
            
            if self.use_rotation:
                # For render-only, we can use identity rotation or skip rotation
                scene_rot = torch.eye(scene_cov3D_render.shape[-1]).to(scene_cov3D_render).expand(scene_pcs_render.shape[0], -1, -1).clone()
        else:
            # Only render moving target object
            assert cln_gaussian is not None
            r_gaussians = cln_gaussian
            scene_cov3D_render = scene_cov3D
            scene_pcs_render = scene_pcs
            
            if self.use_rotation:
                scene_rot = torch.eye(scene_cov3D_render.shape[-1]).to(scene_cov3D_render).expand(scene_pcs_render.shape[0], -1, -1).clone()
        
        # Get scene opacity
        scene_opacity = r_gaussians.get_opacity
        if opacity_scalar is not None and not self.render_mov_only:
            fg_op = scene_opacity
            fg_op *= opacity_scalar
            scene_opacity = fg_op

        # Render images directly using current state
        img_list = multi_apply(
            self.decode_head.pre_render,
            cam_list,
            [None] * len(cam_list),  # gt_label not needed for render-only
            white_bg_list if white_bg_list is not None else [None] * len(cam_list),
            scene_gaussian=r_gaussians, 
            cov3D_precomp=scene_cov3D_render, 
            pos=scene_pcs_render, 
            opacity=scene_opacity, 
            rotation=scene_rot if self.use_rotation else None, 
            return_tuple=True
        )[0]

        return img_list
    

    def _encode_decode_train(self, img_list, gt_label, controller_img_mask=None, bbox=None):
        """Run forward function and calculate loss for decode head in
        training.
        img_list [n_cam, 3, H, W]
        gt_label [n_cam, 3, H, W]
        controller_img_mask [n_cam, 1, H, W]
        bbox [n_cam, 4] , [x_min, y_min, x_max, y_max]
        """
        losses = dict()
        assert gt_label is not None
        # Render loss
        if self.share_weight:
            decode_head = self.decode_head
        else:
            decode_head = self.decode_head[-1]
        render_loss = decode_head.forward_train(img_list, gt_label, controller_img_mask=controller_img_mask, bbox=bbox)
        losses.update(add_prefix(render_loss, 'decode'))
        return losses
    
    def _encode_decode_train_embodied(self, img_list, gt_label):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        assert gt_label is not None
        # Render loss
        if self.share_weight:
            decode_head = self.decode_head
        else:
            decode_head = self.decode_head[-1]
        embodied_loss = decode_head.forward_train_embodied(img_list, gt_label)
        losses.update(add_prefix(embodied_loss, 'decode'))
        return losses
    
    def _encode_decode_train_selfsup(self, pred_pos, cur_state, prev_state, base_graph, dg_list, num_iter):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        # Render loss
        if self.share_weight:
            decode_head = self.decode_head
        else:
            decode_head = self.decode_head[-1]
        selfsup_loss = decode_head.forward_train_selfsup(pred_pos, cur_state, prev_state, base_graph, dg_list, num_iter)
        losses.update(add_prefix(selfsup_loss, 'decode'))
        return losses

    def _encode_decode_train_static(self, pred_pos, gt_label):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        assert gt_label is not None
        # Render loss
        if self.share_weight:
            decode_head = self.decode_head
        else:
            decode_head = self.decode_head[-1]
        static_loss = decode_head.forward_train_static(pred_pos, gt_label)
        losses.update(add_prefix(static_loss, 'decode'))
        return losses

    def _encode_decode_test(self, img_list, gt_label):
        """Run forward function and calculate loss for decode head in
        inference."""
        # label = gt_label['vertices'] if gt_label is not None else None
        # In this case, the behavior is the same as the parent class
        if self.share_weight:
            decode_head = self.decode_head
        else:
            decode_head = self.decode_head[-1]
        logits = decode_head.forward_test(img_list=img_list, gt_label=gt_label, test_cfg=self.test_cfg)
        return logits

    def _rollout_steps(self, num_epoch, num_iter, cfg):
        step_initial = cfg.get('step_initial', None)
        step_increase_interval = cfg.get('step_increase_interval', None)
        step_increase_magnitude = cfg.get('step_increase_magnitude', None)
        max_rollout_steps = cfg.get('max_rollout_step', None)
        by_epoch = cfg.get('by_epoch', None)

        assert step_increase_magnitude is not None
        assert step_initial is not None
        assert by_epoch is not None
        assert max_rollout_steps is not None and max_rollout_steps > 0, "Should be at least 1"
        assert step_increase_interval is not None

        if by_epoch:
            counter = num_epoch
        else:
            counter = num_iter
        # Set current predictions of frames
        if step_increase_interval == 0:
            rollout_size = max_rollout_steps
        else:
            rollout_size = (counter // step_increase_interval) * step_increase_magnitude + step_initial
            rollout_size = min(rollout_size, max_rollout_steps)

        # TODO: no need this first
        # if self.is_est_vel:
        #     rollout_size = 1

        return rollout_size, max_rollout_steps
    
    def set_dataloader_info(self, dataloader):
        self.iters_per_epoch = len(dataloader)
        print(f"Set dataloader info: iters_per_epoch: {self.iters_per_epoch}")

    def _calculate_global_step(self, num_epoch, num_iter):
        # self.iters_per_epoch = 0 if self.iters_per_epoch is None else self.iters_per_epoch
        # global_step = num_epoch * self.iters_per_epoch + num_iter
        # self.iters_per_epoch = max(self.iters_per_epoch, num_iter)
        self.global_step += 1
        return self.global_step

    def forward_train(self, inputs, gt_label, num_epoch=0, num_iter=0, **kwargs):
        """Forward function for training.

        Args:
            inputs:
                img: n_cam, bs==1, 3, H, W
            gt_label: n_cam(for one seq, multiple camera), n_frame, 3, H, W

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # st_time = time.time()
        # This one is only for controlling the EM step
        self.num_iter, self.num_epoch = num_iter+1, num_epoch+1
        # This one input the current num_epoch and num_iter
        rollout_size, _ = self._rollout_steps(num_epoch, num_iter, self.train_cfg)
        # random_rs = np.random.rand() < 1/(rollout_size+1)
        # static_pred = (random_rs or self.static_loss) and self.opt_sim

        losses = defaultdict(list)
        acc_dict = defaultdict(list)
        
        # Init camera given the camera info
        ## Move the camera to the corresponding cuda device
        device = inputs['img'][0].device
        for i in range(len(inputs['cam'])):
            inputs['cam'][i].to_device(device)
        
        seq_idx = kwargs['meta']['seq_idx']
        assert seq_idx.shape[0] == 1 and seq_idx.shape[1] == 1
        seq_idx = seq_idx[0, 0].item()

        scene_name = kwargs['meta']['scene_name']
        assert scene_name in self.gs_scene_dict.keys()
        scene_gaussian = self.gs_scene_dict[scene_name]
        gs_aligned_frame = int(inputs['gs_aligned_frame'])
        # scene_render_gaussian = None if not self.render_mov_only else self.gs_scene_dict_render[scene_name]aa

        p2c_mapping_list = [[p2c[0].squeeze(0).to(torch.int64), p2c[1].squeeze(0).to(torch.int64)] for p2c in inputs['p2c_mapping']]

        controller_trajectory = inputs['controller_trajectory'].squeeze(0).to(torch.float32)
        controller_img_mask_list = inputs['controller_img_mask_list']
        
        # Every loop finish, need to recover the gaussian model with original rest state for next loop
        # original_mov_state = scene_gaussian.get_xyz
        # original_mov_cov = scene_gaussian.get_covariance() # TODO: here is not correct
        original_mov_state = self.scene_init_pos[f"{scene_name}_frame_{gs_aligned_frame}"]
        original_mov_prev_state = self.scene_init_prev_pos[f"{scene_name}_frame_{gs_aligned_frame}"]
        original_mov_cov = self.scene_init_cov[f"{scene_name}_frame_{gs_aligned_frame}"]
        # check required_grad

        original_controller_state = controller_trajectory[0]
        original_mov_scaling = scene_gaussian.get_scaling
        
        assert scene_name in self._scene_attr_dict.keys()
        attr_name = self._scene_attr_dict[scene_name]
        assert attr_name in self.scene_attr.keys()
        attr = self.scene_attr[attr_name]
        N = original_mov_state.shape[0]
        attr = attr.expand(N, -1)
        diag_volume = torch.prod(original_mov_scaling*inputs['volume_scalar'].squeeze(0), dim=-1, keepdim=True)
        
        cur_state = original_mov_state
        prev_state = original_mov_prev_state
        cur_cov = original_mov_cov # No need .clone()
        external_forces = inputs['external'].squeeze(0)

        num_object_points = original_mov_state.shape[0]
        num_controller_points = p2c_mapping_list[0][0].shape[0]
        controller_mask = torch.zeros((num_controller_points+num_object_points, 1), dtype=torch.float32).to(device)
        controller_mask[:num_controller_points] = 1.

        num_frame = min(gt_label[0].shape[1]-1, rollout_size)
        gt_label_idx_offset = 1

        for frame_idx in range(num_frame):
            #TODO; remove prev_state, add controller_trajectory / original_controller_state
            controller_cur_state = controller_trajectory[max(frame_idx+1, 0)].detach()
            controller_prev_state = controller_trajectory[max(frame_idx, 0)].detach()
           
            if not (frame_idx == 0 or frame_idx == -1):
                del input_graph, connect_graph, pred_pos, pred_cov, pred_img_list
            #     # prev_state = original_mov_state.detach()
            #     pass
            # else:
            #     del input_graph, connect_graph, pred_pos, pred_cov, pred_img_list
            
            input_graph, connect_graph = self._preprocess(
                prev_state, cur_state, original_mov_state, controller_prev_state, controller_cur_state, original_controller_state, attr, diag_volume, cur_cov,
                external_forces,
                p2c_mapping=p2c_mapping_list,
                pin_mask=controller_mask)
            # Forward: predicting the velocity or positions
            cur_label_idx = frame_idx + gt_label_idx_offset
            cur_label = [gt[0, cur_label_idx] for gt in gt_label]
            bbox = inputs['bbox'][0, :, cur_label_idx]
            controller_img_mask = [controller_img_mask[0, cur_label_idx] for controller_img_mask in controller_img_mask_list]
            # mask_imgs = []
            # TODO
            pred_pos, pred_cov, pred_dg_list, knowledge_dict, pred_img_list, pred_depth_list, losses_i = self.encode_decode(
                input_graph, connect_graph,
                scene_gaussian, inputs['cam'],
                original_mov_cov.detach().clone(), original_mov_state.detach().clone(), # For rendering, better not change this two variable
                num_controller_points, gt_label=cur_label, register_norm=True, # TODO: only opt the sim?
                rollout_size=rollout_size, # For checkpointing
                # cln_mask=cln_mask, 
                cln_gaussian=scene_gaussian,
                white_bg_list=inputs['white_bg'] if 'white_bg' in inputs.keys() else None
                )

            #TODO 
            img_loss = self._encode_decode_train(pred_img_list, cur_label, controller_img_mask, bbox)
            losses_i.update(img_loss)


            if self.flag_update_gaussian_train and frame_idx != 0 and frame_idx % self.frame_gap == 0:
                self.update_gaussian(scene_name, gs_aligned_frame + frame_idx, pred_pos[num_controller_points:], pred_cov[num_controller_points:], cur_state)
            if self.flag_update_gaussian and frame_idx != 0 and frame_idx % self.frame_gap == 0:
                self.flag_update = True

            # Update the state pointer and rollout to next frame
            if not self.accumulate_gradient:
                prev_state = cur_state.detach()
                cur_state = pred_pos[num_controller_points:].detach()
                # cur_cov = pred_cov[num_controller_points:].detach()
            else:
                prev_state = cur_state
                cur_state = pred_pos[num_controller_points:]
                # cur_cov = pred_cov[num_controller_points:]


            # Merge loss
            for key in losses_i.keys():
                if key.startswith('decode.loss'):
                    # Sum loss
                    losses[f'{key}_{scene_name}'].append(losses_i[key])
                else:
                    acc_key = f'{key}_{scene_name}'
                    acc_key = f"{acc_key}_frame{frame_idx}"
                    # Avg acc
                    # if losses_i[key] > 0:
                    if losses_i[key].numel() == 1:
                        acc_dict[acc_key].append(losses_i[key].detach().clone())
                    else:
                        acc_dict[acc_key].append(losses_i[key].detach())

        # reduce losses
        losses_rst = dict()
        for key, val in losses.items():
            if len(val) == 0:
                continue
            agg_func = torch.sum
            if self.avg_loss:
                agg_func = torch.mean
            losses_rst[key] = agg_func(torch.stack(val))
        for key, val in acc_dict.items():
            losses_rst[key] = torch.mean(torch.stack(val))
        
        # Align keys
        to_pad_static = True
        static_key = f'decode.loss_mse_static_{scene_name}'
        for key in losses_rst.keys():
            if 'static' in key:
                assert static_key == key
                to_pad_static = False
        if to_pad_static:
            # This one may lead to wrong acc l2 on frame 0. Need to check
            assert static_key not in losses_rst.keys()
            losses_rst[static_key] = torch.zeros(1).to(device)[0]
        
        # loss_key_list = ['momentum', 'render', 'static', 'spatial', 'acc']
        # losses_dict = defaultdict(list)
        # if self.tb_logger is not None:
        #     global_step = self._calculate_global_step(num_epoch, num_iter)
        #     for key in loss_key_list:
        #         # losses = {k:v for k, v in losses_rst.items() if key in k}
        #         for k, v in losses_rst.items():
        #             if key in k:
        #                 losses_dict[k].append(v)
        #     self.tb_logger.log_losses(losses_dict, global_step, f'train/{key}')


        return losses_rst

    def inference(self, inputs, gt_label=None, is_training=False, **kwargs):
        """Inference with slide/whole style.
        """
        # Can do something using self.test_cfg
        output = self.encode_decode(inputs, gt_label=gt_label, is_training=is_training, **kwargs)
        
        return output

    def evaluate(self, pred, gt_label, meta_info, **kwargs):
        pass

    def _merge_acc(self, pred_rst_dict):
        # Merge acc; pred cannot merge and no use actually
        merged_rst = dict(acc=dict())
        # Use the latest as the final output; Since this only useful when test, while the length is 1, thus no influence.
        # But the acc need to agg all
        pred_acc = pred_rst_dict.pop('acc')
        pred_step = len(pred_acc)
        laststep_rst_dict = dict()
        for key, acc_val in pred_acc[-1].items():
            laststep_rst_dict[f'{key}_step{pred_step}'] = acc_val
        merged_rst['acc'] = laststep_rst_dict
        for key, val in pred_rst_dict.items():
            merged_rst[key] = val[-1] # Only choose the last one during validation; Test is ok cuz only one step

        if torch.onnx.is_in_onnx_export():
            return merged_rst
        merged_rst = to_numpy_detach(merged_rst)
        return merged_rst


    def simple_test(self,
                    inputs, gt_label=None,
                    prev_state=None, cur_state=None, cur_cov=None, pred_frame_idx=None, zero_init=False, **kwargs):
        """Simple test with single image.
        
            pred_frame_idx: start from 1, given 0

        """
        assert cur_cov == None
        # This one input the current num_epoch and num_iter
        # rollout_size, _ = self._rollout_steps(0, 0, self.test_cfg)
        rollout_size = 1
        # assert rollout_size == 1 rollout_size==1 is test not validating
        acc_dict = defaultdict(list)
        
        # Init camera given the camera info
        ## Move the camera to the corresponding cuda device
        device = inputs['img'][0].device
        for i in range(len(inputs['cam'])):
            inputs['cam'][i].to_device(device)
        
        seq_idx = kwargs['meta']['seq_idx']
        assert seq_idx.shape[0] == 1 and seq_idx.shape[1] == 1
        seq_idx = seq_idx[0, 0].item()

        scene_name = kwargs['meta']['scene_name']
        assert scene_name in self.gs_scene_dict.keys()
        scene_gaussian = self.gs_scene_dict[scene_name]
        gs_aligned_frame = int(inputs['gs_aligned_frame'])
        # scene_render_gaussian = None if not self.render_mov_only else self.gs_scene_dict_render[scene_name]
        # Related scene mask
        p2c_mapping_list = [[p2c[0].squeeze(0).to(torch.int64), p2c[1].squeeze(0).to(torch.int64)] for p2c in inputs['p2c_mapping']]
        ## controller points
        controller_trajectory = inputs['controller_trajectory'].squeeze(0).to(torch.float32)

        if pred_frame_idx is None:
            pred_frame_idx = 0
        pred_frame_idx += 1

        # Every loop finish, need to recover the gaussian model with original rest state for next loop
        # original_mov_state = scene_gaussian.get_xyz
        # original_mov_cov = scene_gaussian.get_covariance() # TODO: here is not correct
        template_frame = gs_aligned_frame
        reset_from_template = pred_frame_idx == 1
        if self.test_rollout_mode == 'continuous':
            prev_real_frame_idx = gs_aligned_frame + pred_frame_idx - 1
            if self.frame_gap > 0:
                template_offset = ((prev_real_frame_idx - gs_aligned_frame) // self.frame_gap) * self.frame_gap
                template_frame = gs_aligned_frame + template_offset
            reset_from_template = reset_from_template or (
                template_frame != gs_aligned_frame and prev_real_frame_idx == template_frame)

        template_key = f"{scene_name}_frame_{template_frame}"
        if template_key not in self.scene_init_pos:
            raise KeyError(
                f"Missing {template_key} for {self.test_rollout_mode} rollout. "
                "Continuous rollout only initializes frame_0 from the raw Gaussian; "
                "later template frames must be produced by online update_gaussian().")
        original_mov_state = self.scene_init_pos[template_key]
        original_mov_prev_state = self.scene_init_prev_pos[template_key]
        original_mov_cov = self.scene_init_cov[template_key]

        controller_template_idx = template_frame - gs_aligned_frame
        original_controller_state = controller_trajectory[controller_template_idx]
        original_mov_scaling = scene_gaussian.get_scaling
        

        assert scene_name in self._scene_attr_dict.keys()
        attr_name = self._scene_attr_dict[scene_name]
        assert attr_name in self.scene_attr.keys()
        attr = self.scene_attr[attr_name]
        N = original_mov_state.shape[0]
        attr = attr.expand(N, -1)
        diag_volume = torch.prod(original_mov_scaling*inputs['volume_scalar'].squeeze(0), dim=-1, keepdim=True)

        if reset_from_template:
            cur_state = original_mov_state
            prev_state = original_mov_prev_state
        cur_cov = original_mov_cov
        external_forces = inputs['external'].squeeze(0)

        num_object_points = original_mov_state.shape[0]
        num_controller_points = p2c_mapping_list[0][0].shape[0]
        controller_mask = torch.zeros((num_controller_points+num_object_points, 1), dtype=torch.float32).to(device)
        controller_mask[:num_controller_points] = 1.

        # Prepare to sim
        ## Omit the first one, the first is known
        ## When testing, gt_label may be None
        num_frame = min(gt_label[0].shape[1]-1, rollout_size) if gt_label is not None else rollout_size

        rst_dict = defaultdict(list)
        for frame_idx_ in range(num_frame):
            frame_idx = frame_idx_ + pred_frame_idx - 1
            start_time = time.time()

            controller_cur_state = controller_trajectory[frame_idx+1].detach()
            controller_prev_state = controller_trajectory[frame_idx].detach()
            # if frame_idx == 0:
            #     prev_state = original_mov_state.detach()
            
            # Build graph for computing
            input_graph, connect_graph = self._preprocess(
                prev_state, cur_state, original_mov_state, controller_prev_state, controller_cur_state, original_controller_state, attr, diag_volume, cur_cov,
                external_forces,
                p2c_mapping=p2c_mapping_list,
                pin_mask=controller_mask)
            preprocess_time = time.time()
            # Forward: predicting the velocity or positions
            cur_label_idx = pred_frame_idx if pred_frame_idx is not None else frame_idx+1
            cur_label = [gt[0, cur_label_idx] for gt in gt_label] # Currently all have gt_label
            cur_img_list = self.encode_decode_render_only(
                input_graph, connect_graph,
                scene_gaussian, inputs['cam'],
                original_mov_cov.detach().clone(), original_mov_state.detach().clone(), # For rendering, better not change this two variable
                num_controller_points, gt_label=cur_label, register_norm=False,
                # cln_mask=cln_mask, 
                cln_gaussian=scene_gaussian,
                white_bg_list=inputs['white_bg'] if 'white_bg' in inputs.keys() else None,
                opacity_scalar=inputs.get('opacity_scalar', None))
            pred_pos, pred_cov, pred_dg_list,knowledge_dict,  pred_img_list, pred_depth_list, losses_i = self.encode_decode(
                input_graph, connect_graph,
                scene_gaussian, inputs['cam'],
                original_mov_cov.detach().clone(), original_mov_state.detach().clone(), # For rendering, better not change this two variable
                num_controller_points, gt_label=cur_label, register_norm=False,
                # cln_mask=cln_mask, 
                cln_gaussian=scene_gaussian,
                white_bg_list=inputs['white_bg'] if 'white_bg' in inputs.keys() else None,
                opacity_scalar=inputs.get('opacity_scalar', None))

            img_loss = self._encode_decode_test(pred_img_list, cur_label)
            img_loss['acc'].update(losses_i)
            rst_dict['acc'].append(img_loss['acc'])
            rst_dict['pred_pos'].append(pred_pos[num_controller_points:].detach().clone())
            rst_dict['pred_cov'].append(pred_cov[num_controller_points:].detach().clone())
            rst_dict['cur_state'].append(cur_state.detach().clone())
            rst_dict['pred_img_list'].append(pred_img_list)
            rst_dict['pred_depth_list'].append(pred_depth_list)
            rst_dict['cur_img_list'].append(cur_img_list)
            # Update the state pointer and rollout to next frame
            if not self.accumulate_gradient:
                prev_state = cur_state.detach()
                cur_state = pred_pos[num_controller_points:].detach()
                # cur_cov = pred_cov.detach()
            else:
                prev_state = cur_state
                cur_state = pred_pos[num_controller_points:]
                # cur_cov = pred_cov

            real_frame_idx = gs_aligned_frame + pred_frame_idx
            if self.flag_save_gaussian and pred_frame_idx != 0 and real_frame_idx % self.frame_gap == 0:
                self.save_gaussian(scene_name, real_frame_idx, pred_pos[num_controller_points:], pred_cov[num_controller_points:])
            # if self.flag_update_gaussian and self.flag_update and pred_frame_idx != 0 and real_frame_idx % self.frame_gap == 0:
            update_online_gaussian = self.flag_update_gaussian or self.test_rollout_mode == 'continuous'
            if update_online_gaussian and pred_frame_idx != 0 and real_frame_idx % self.frame_gap == 0:
                self.update_gaussian(scene_name, real_frame_idx, pred_pos[num_controller_points:], pred_cov[num_controller_points:], cur_state)

            end_time = time.time()

        return_dict = self._merge_acc(rst_dict)
        return_dict['time'] = [start_time, preprocess_time, end_time]
        return return_dict

    def _em_scheme(self, num_epoch, num_iter, cfg):
        step_increase_interval = cfg.get('step_increase_interval', None)
        by_epoch = cfg.get('by_epoch', None)
        # Apply during the interval before the rollout step increase
        em_step_ratio = cfg.get('em_step_ratio', None)

        assert by_epoch is not None
        assert step_increase_interval is not None
        assert em_step_ratio is not None

        # Here add one is for next one, cuz this func first then counter is increased in the training
        if by_epoch:
            counter = num_epoch + 1
        else:
            counter = num_iter + 1
        
        is_est = False
        if counter % step_increase_interval <= step_increase_interval * em_step_ratio:
            is_est = True
        return is_est
    
    def _freeze_stages(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return

    def _train_stages(self, model):
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        return
    
    def train(self, mode=True):
        """Set module status before forward computation.
        Args:
            mode (bool): Whether it is train_mode or test_mode
        """
        # This one will make all params to be trainable
        super(GsSimulatorEmbodiedS2, self).train(mode)
        if not self.opt_sim:
            if isinstance(self.backbone, nn.ModuleList):
                for i in range(len(self.backbone)):
                    self._freeze_stages(self.backbone[i])
            else:
                self._freeze_stages(self.backbone)
            if isinstance(self.decode_head, nn.ModuleList):
                for i in range(len(self.decode_head)):
                    self._freeze_stages(self.decode_head[i])
            else:
                self._freeze_stages(self.decode_head)
        if not self.opt_vel:
            for key in self.scene_attr.keys():
                self.scene_attr[key].requires_grad = False
                for seq_key in self.scene_initn1_pos[key].keys():
                    self.scene_initn1_pos[key][seq_key].requires_grad = False
                if self.pred_vel:
                    for seq_key in self.scene_init0_pos[key].keys():
                        self.scene_init0_pos[key][seq_key].requires_grad = False
        return

    def aug_test(self, **kwargs):
        """
            Refer mmsegmentation
        """
        raise NotImplementedError("aug_test is not implemented")
