"""Frozen real windows and evidence using the existing Dataset/processor/rendering."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from scripts.dataloader.dataset import Touch4DDataset
from scripts.dataloader.collate import collate_fn
from scripts.data_process import DataProcessor
from scripts.training.train import GS_CAMERAS,to_device,atomic_json,copy_initial
from scripts.training.gs_decoder import build_initial_anchor_field
from scripts.experiments.fixed_window.fixed_fit import prediction_metrics
from .conditions import hand_inputs,sanitize

WINDOWS=((1,85),(2,73))
WEIGHTS=dict(chamfer_weight=1.,rgb_weight=1.,alpha_weight=.1,scale_weight=.01)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dataset(root):
    return Touch4DDataset('001-rope','train',root,horizon=15,with_depth=True,
                         depth_cameras=GS_CAMERAS['001-rope'])

def batches(ds,device='cpu'):
    out=[]
    for window in WINDOWS:
        raw=collate_fn([ds[ds.windows.index(window)]])
        b=DataProcessor('001-rope')(to_device(raw,device))
        b['inputs']=sanitize(hand_inputs(b['inputs'],b['meta']['episode_id'],b['meta']['scene']))
        out.append(b)
    return out

def audit(root,path):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    ds=dataset(root);processor=DataProcessor('001-rope');rows=[];hashes={}
    for (ep,start),b in zip(WINDOWS,batches(ds)):
        ids=b['meta']['frame_ids'][0].tolist()
        assert ids==list(range(start,start+16))
        x,y=b['inputs'],b['targets'];assert len(y['target_xyz'])==15
        valid=y['target_depth_valid'].flatten(-2).any(-1)
        if not valid.all():raise ValueError('Missing trusted camera/horizon supervision')
        assert x['operation_hands'].tolist()==[[False,True]]
        assert x['controller_mask'][:,:,1].all() and not x['controller_mask'][:,:,0].any()
        field=build_initial_anchor_field(x['gs_xyz'],x['gs_mask'],128,16,1.)
        folder=ds.episodes[ep].path
        files=list(folder.glob('*.json'))+list(folder.glob('*.npz'))
        files += [folder/'gs'/f'frame_{f:06d}.npz' for f in ids]
        for p in files:hashes[str(p)]=sha(p)
        # Dataset has already validated depth/core binding and camera timestamps.
        depths=ds.depth_episodes[ep].path
        for p in list(depths.glob('*.json'))+list(depths.glob('*.npz')):hashes[str(p)]=sha(p)
        rows.append(dict(episode=ep,start=start,frames=ids,targets=ids[1:],
            condition_frames=ids[:-1],operation_hands=x['operation_hands'].tolist(),
            counts=[int(x['gs_mask'].sum())]+[int(v.sum()) for v in y['target_mask']],
            camera_ids=b['meta']['depth_camera_ids'],valid_pixels=y['target_depth_valid'].sum((-1,-2)).tolist(),
            sigma0_normalized=field.bandwidth.tolist(),sigma_tool_normalized=(2*field.bandwidth).tolist(),
            sigma_tool_mm=(2*field.bandwidth*processor.state_dict()['scale']*1000).tolist(),
            target_depth_valid_sha256=hashlib.sha256(y['target_depth_valid'].numpy().tobytes()).hexdigest(),
            observation_timestamp_us=b['meta']['observation_timestamp_us'].tolist(),
            tactile_age_us=x['tactile_age_us'].tolist(),finite=True))
    atomic_json(path,dict(status='frozen_before_training',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        windows=rows,scene='001-rope',horizon=15,split='train',delta=1,seed=42,updates=400,
        evaluation_interval=20,weights=WEIGHTS,processor=processor.state_dict(),source_file_sha256=hashes,
        selection='User specified historical P2 training windows; no replacement or prediction-based selection',
        limitations='Independent reconstructed GT sets, no cross-frame Gaussian IDs; valid render pixels do not certify absence of mask/reconstruction defects'))
    return rows

@torch.no_grad()
def metrics(pred,b,processor,renderer):
    out=prediction_metrics(pred,b,processor,renderer,512)
    keep=b['inputs']['gs_mask'][0];x=b['inputs']['gs_xyz'][0,keep];unit=processor.state_dict()['scale']*1000
    out['motion']=[]
    for k in range(15):
        p=pred['xyz'][0,k,keep];target=b['targets']['target_xyz'][k][0,b['targets']['target_mask'][k][0]]
        delta=p-x;mean=delta.mean(0);gt=target.mean(0)-x.mean(0)
        out['motion'].append(dict(rms_mm=float(delta.square().sum(-1).mean().sqrt()*unit),
            residual_rms_mm=float((delta-mean).square().sum(-1).mean().sqrt()*unit),
            mean_displacement_mm=(mean*unit).tolist(),gt_set_centroid_delta_mm=(gt*unit).tolist(),
            centroid_error_mm=float((mean-gt).norm()*unit),
            direction_cosine=float(torch.nn.functional.cosine_similarity(mean[None],gt[None]).item()) if mean.norm()>1e-9 and gt.norm()>1e-9 else None))
    return out

@torch.no_grad()
def evaluate(model,bs,processor,renderers,perturb=False):
    model.eval();rows=[];predictions=[]
    for b,renderer in zip(bs,renderers):
        pred=model(b['inputs']);predictions.append(pred)
        row=metrics(pred,b,processor,renderer)
        if perturb:
            row['perturbations']={}
            for mode in ('freeze_action','zero_tactile'):
                x=dict(b['inputs'])
                if mode=='freeze_action':
                    for key in ('action_state','action_valid','controller_points','controller_mask'):
                        x[key]=x[key][:,:1].expand_as(x[key]).clone()
                else:x['tactile']=torch.zeros_like(x['tactile'])
                alt=model(x);m=metrics(alt,b,processor,renderer)
                d=(alt['xyz']-pred['xyz'])[0,:,x['gs_mask'][0]]*processor.state_dict()['scale']*1000
                m['output_difference_rms_mm']=d.square().sum(-1).mean(-1).sqrt().tolist()
                m['delta_cd_mm']=(np.array(m['cd_mm'])-row['cd_mm']).tolist();row['perturbations'][mode]=m
        rows.append(row)
    return rows,predictions

@torch.no_grad()
def export(directory,pred,b,processor,renderer,reference=False):
    from scripts.training.visualize import png_bytes
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    rendered={'pred':renderer(pred,b,processor)}
    if reference:rendered.update(gt=renderer.render_targets(b,processor),copy=renderer(copy_initial(b['inputs'],'gs_geometry'),b,processor))
    for label,r in rendered.items():
        for typ in ('rgb','alpha'):
            images=r[typ][0].cpu().clamp(0,1).numpy()
            for h in range(15):
                for cam in range(3):
                    im=np.round(images[h,cam]*255).astype(np.uint8)
                    if im.ndim==2:im=np.repeat(im[...,None],3,-1)
                    (directory/f'h{h+1:02d}-c{cam}-{typ}-{label}.png').write_bytes(png_bytes(im))
    x=b['inputs'];keep=x['gs_mask'][0];scale=processor.state_dict()['scale'];center=np.array(processor.state_dict()['center'])
    arrays=dict(prediction_xyz_m=pred['xyz'][0,:,keep].cpu().numpy()*scale+center,
        initial_xyz_m=x['gs_xyz'][0,keep].cpu().numpy()*scale+center,frame_ids=b['meta']['frame_ids'][0].cpu().numpy())
    for k in range(15):arrays[f'gt_h{k+1}_xyz_m']=b['targets']['target_xyz'][k][0,b['targets']['target_mask'][k][0]].cpu().numpy()*scale+center
    arrays['initial_log_scales_m']=x['gs_attr'][0,keep,:3].cpu().numpy()+np.log(scale)
    arrays['initial_quaternions']=x['gs_attr'][0,keep,3:7].cpu().numpy()
    np.savez_compressed(directory/'geometry.npz',**arrays)
