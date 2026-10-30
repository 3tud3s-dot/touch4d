"""Original objective/rendering with per-window, per-horizon and camera evidence."""
import math
import numpy as np
import torch
from scripts.training.rendering import GaussianRenderer
from scripts.training.v4_runtime import v4_objective
from scripts.training.losses import nearest_distances
from scripts.experiments.v4_g5_signed_local.diagnostics import finite_tree, stats, write_png
from .geometry import covariance, LOG_SCALE_BOUND

METRICS=('cd_mm','rgb','alpha','joint')
TOLERANCE=dict(cd_mm=1e-4,rgb=1e-6,alpha=1e-6,joint=1e-6)
REFERENCE=dict(cd_mm=2.4026261481544693,rgb=.07762423427493284,
               alpha=.07769889489117295,joint=.10069386053548167)


class CachedRenderer:
    """Cache detached GT for one batch only; unchanged original renderer and mask."""
    def __init__(self):
        self.inner=GaussianRenderer(image_size=(720,1280));self.clear()
    def clear(self):self.target=None;self.rendered=None;self.batch=None
    def render_targets(self,batch,processor):
        if self.batch is not batch:
            self.clear();self.batch=batch
        if self.target is None:
            self.target=self.inner.render_targets(batch,processor);finite_tree(self.target)
        return self.target
    def __call__(self,prediction,batch,processor):
        self.rendered=self.inner(prediction,batch,processor);finite_tree(self.rendered)
        return self.rendered


@torch.no_grad()
def equality(original,candidate,gs_mask,*,all_attributes=False):
    result={}
    for key in original:
        result[key+'_exact']=bool(torch.equal(original[key],candidate[key]))
        valid=gs_mask[:,None,:,None].expand_as(original[key])
        result[key+'_max_abs']=float((original[key]-candidate[key])[valid].abs().max())
    required=original.keys() if all_attributes else ('xyz','opacity_logits','sh_dc')
    if not all(result[k+'_exact'] for k in required):
        raise AssertionError('Baseline/isolation equality failed: '+str(result))
    return result


def assert_metrics_equal(a,b):
    if any(abs(a[k]-b[k])>TOLERANCE[k] for k in METRICS):
        raise AssertionError('Frozen metric tolerance exceeded: '+str({k:b[k]-a[k] for k in METRICS}))


def summarize(rows,mode):
    selected=[r for r in rows if r['mode']==mode and r['split']=='val']
    if len(selected)!=100 or sum(r['episode']==3 for r in selected)!=55 or sum(r['episode']==6 for r in selected)!=45:
        raise AssertionError('Full Val100 required')
    output=[]
    for ep in (3,6):
        group=[r for r in selected if r['episode']==ep]
        output.append(dict(mode=mode,episode=str(ep),windows=len(group),
                           **{k:float(np.mean([r[k] for r in group])) for k in METRICS}))
    output.append(dict(mode=mode,episode='episode_equal',windows=100,
                       **{k:sum(r[k] for r in output)/2 for k in METRICS}))
    return output


def objective_record(pred,batch,processor,renderer,*,mode,split,episode,start,curves=True):
    loss,metrics=v4_objective(pred,batch,processor,renderer)
    finite_tree((loss,metrics,pred))
    if not metrics['render_available'].all() or not metrics['render_time_available'].all():
        raise ValueError('Incomplete rendering supervision')
    scale=processor.state_dict()['scale']*1000
    row=dict(mode=mode,split=split,episode=episode,start=start,cd_mm=float(metrics['geometry'].mean())*scale,
        rgb=float(metrics['rgb'].detach().mean()),alpha=float(metrics['alpha'].detach().mean()),joint=float(loss.detach()),
        scale_regularization_diagnostic_only=float(metrics['scale_regularization'].detach().mean()))
    horizons=[];cameras=[]
    if curves:
        with torch.no_grad():
            x,y=batch['inputs'],batch['targets']
            for h in range(pred['xyz'].shape[1]):
                p=pred['xyz'][0,h,x['gs_mask'][0]];t=y['target_xyz'][h][0,y['target_mask'][h][0]]
                cd=float((nearest_distances(p,t).mean()+nearest_distances(t,p).mean())*.5)
                rgb=float(metrics['rgb_by_horizon'][0,h]);alpha=float(metrics['alpha_by_horizon'][0,h])
                horizons.append(dict(mode=mode,split=split,episode=episode,start=start,horizon=h+1,
                    cd_mm=cd*scale,rgb=rgb,alpha=alpha,joint=cd+rgb+.1*alpha))
            valid=y['target_depth_valid'];counts=valid.sum((-1,-2));available=counts>0
            for key in ('rgb','alpha'):
                diff=(renderer.rendered[key].detach()-renderer.target[key]).abs()
                if key=='rgb':diff=diff.mean(-1)
                per_view=torch.where(valid,diff,torch.zeros_like(diff)).sum((-1,-2))/counts.clamp_min(1)
                for v in range(3):
                    cameras.append(dict(mode=mode,split=split,episode=episode,start=start,camera=v,
                        metric=key,value=float(per_view[0,:,v][available[0,:,v]].mean())))
    return loss,row,horizons,cameras


@torch.no_grad()
def geometry_rows(pred,motion,geo,batch,*,mode,split,episode,start):
    mask=batch['inputs']['gs_mask'][0];rows=[]
    for h in range(pred['xyz'].shape[1]):
        logs=pred['log_scales'][0,h,mask];q=pred['quaternions'][0,h,mask]
        ratio=(logs-motion['log_scales'][0,h,mask]).exp()
        omega=geo['rotation_vector'][0,h,mask]
        angle=omega.norm(dim=-1)*180/math.pi
        cov=covariance(logs,q);eig=torch.linalg.eigvalsh(cov.double())
        if not torch.isfinite(cov).all() or not (eig>0).all():raise FloatingPointError('Covariance not positive definite')
        if not (ratio>=.8-1e-6).all() or not (ratio<=1.25+1e-6).all() or float(angle.max())>30+1e-5:
            raise AssertionError('Frozen geometry safety bound exceeded')
        row=dict(mode=mode,split=split,episode=episode,start=start,horizon=h+1,
            covariance_min_eigenvalue=float(eig.min()),quaternion_norm_max_error=float((q.norm(dim=-1)-1).abs().max()),
            log_scale_saturation_fraction=float((geo['log_scale_delta'][0,h,mask].abs()>=.99*LOG_SCALE_BOUND).float().mean()),
            rotation_angle_saturation_fraction=float((angle>=29.7).float().mean()))
        for name,v in (('scale_ratio',ratio),('rotation_deg',angle)):
            row.update({name+'_'+k:val for k,val in stats(v).items()})
        rows.append(row)
    return rows


def save_images(renderer,out,label,*,save_gt=False):
    for h in (5,15):
        for camera in range(3):
            for key in ('rgb','alpha'):
                for mode,images in [('prediction',renderer.rendered)]+([('gt',renderer.target)] if save_gt else []):
                    array=images[key][0,h-1,camera].detach().clamp(0,1).cpu().numpy()
                    if key=='alpha':array=np.repeat(array[...,None],3,-1)
                    # Original resolution is retained; presentation uses at most 3 columns.
                    write_png(out/f'{label}_{mode}_h{h}_camera{camera}_{key}.png',(array*255).round().astype('uint8'))
