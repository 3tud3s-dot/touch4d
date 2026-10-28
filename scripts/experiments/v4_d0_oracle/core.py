"""Only D is learnable. Production fixed_field/interpolation/loss are reused."""
import torch
from scripts.training.gs_decoder import interpolate_anchor_displacements
from scripts.training.model import masked, initial_knn
from scripts.training.losses import chamfer_samples, render_samples
from .regions import rms_mm


def one_horizon(batch,h):
    y=batch['targets'];out={}
    for k,v in y.items():
        if k in ('target_xyz','target_attr','target_mask'):out[k]=[v[h-1]]
        elif k in ('target_depth','target_depth_valid','target_lengths'):out[k]=v[:,h-1:h]
        else:out[k]=v
    return dict(batch,targets=out)


def decode(x,field,d):
    xyz=masked(x['gs_xyz']+interpolate_anchor_displacements(d,field,x['gs_mask']),x['gs_mask'])
    p={'xyz':xyz[:,None]}
    for key,a,b in [('log_scales',0,3),('quaternions',3,7),('opacity_logits',7,8),('sh_dc',8,11)]:
        p[key]=masked(x['gs_attr'][...,a:b],x['gs_mask'])[:,None]
    return p


def loss_parts(p,b,processor,renderer,target,render=True):
    cd=chamfer_samples(p['xyz'],b).mean()
    if not render:return cd,dict(cd=cd)
    images=renderer(p,b,processor);parts={'cd':cd}
    for k in ('rgb','alpha'):
        vals,available,_,_=render_samples(images[k],target[k],b['targets']['target_depth_valid'])
        if not available.all():raise ValueError('Missing render supervision')
        parts[k]=vals.mean()
    parts['joint']=cd+parts['rgb']+.1*parts['alpha']
    return parts['joint'],parts


def edges(x):
    ids,valid=initial_knn(x['gs_xyz'],x['gs_mask'],16)
    valid=valid[0]&x['gs_mask'][0,:,None]&(ids[0]!=torch.arange(ids.shape[1],device=ids.device)[:,None])
    dist=(x['gs_xyz'][0,:,None]-x['gs_xyz'][0,ids[0]]).norm(dim=-1)
    valid &= dist>1e-8
    return ids[0],valid,dist


@torch.no_grad()
def structure(x,pos,edge,scale):
    ids,valid,d0=edge;d1=(pos[:,None]-pos[ids]).norm(dim=-1);ratio=d1[valid]/d0[valid]
    error=(d1-d0)*scale*1000
    distortion=(d1/d0.clamp_min(1e-8)-1).abs().masked_fill(~valid,0)
    per_point=distortion.sum(-1)/valid.sum(-1).clamp_min(1)
    return dict(edge_count=int(valid.sum()),ratio_mean=float(ratio.mean()),ratio_p05=float(torch.quantile(ratio,.05)),ratio_p95=float(torch.quantile(ratio,.95)),distortion_p95=float(torch.quantile((ratio-1).abs(),.95)),neighbor_distance_change_rms_mm=rms_mm(error[valid][:,None],1)),per_point
