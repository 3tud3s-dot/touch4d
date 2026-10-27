"""Reuse P2 data, geometry, renderer and loss; explicit causal interventions."""
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.dataset import Touch4DDataset
from scripts.dataloader.collate import collate_fn
from scripts.training.train import GS_CAMERAS, to_device, copy_initial
from scripts.training.losses import nearest_distances
from scripts.experiments.fixed_window.fixed_fit import prediction_metrics
from .conditions import ToolPredictor, hand_inputs, sanitize

ARCH = dict(width=128,heads=4,layers=2,controller_features=7,horizon=5,
    prediction_mode='autoregressive',output_mode='gs_geometry',gs_local_attention='knn',
    gs_local_k=16,gs_decoder='anchors',anchor_count=128,anchor_pool_k=16,
    anchor_bandwidth=1.,gs_structure='anchor_attention')
WEIGHTS = dict(chamfer_weight=1.,rgb_weight=1.,alpha_weight=.1,scale_weight=.01)
VERSION = 'condition-identifiability-v1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def state_hash(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def dataset(root,depth=True):
    return Touch4DDataset('001-rope','train',root,horizon=5,with_depth=depth,
                         depth_cameras=GS_CAMERAS['001-rope'] if depth else None)

def batch_at(ds,ep,start,device='cpu'):
    if (ep,start) not in ds.windows:raise ValueError('Only valid train windows allowed')
    raw=collate_fn([ds[ds.windows.index((ep,start))]])
    batch=DataProcessor('001-rope')(to_device(raw,device))
    batch['inputs']=sanitize(hand_inputs(batch['inputs'],batch['meta']['episode_id'],batch['meta']['scene']))
    return batch

def synthetic_pair(base,scale):
    """Identical GS/time/tactile; only action differs. One-step 2mm z translation.

    Synthetic GS targets and full-image render validity are constructed offline.
    Depth is a placeholder, never a supervised term. Real data is never edited.
    """
    output=[]
    for sign in (-1.,1.):
        b=copy.deepcopy(base);x=b['inputs'];y=b['targets']
        h=x['action_state'].shape[1]
        offset=torch.arange(1,h+1,device=x['gs_xyz'].device,dtype=x['gs_xyz'].dtype)*(.002/scale)*sign
        x['action_state']=x['action_state'][:,:1].repeat(1,h,1,1)
        x['action_valid']=x['action_valid'][:,:1].repeat(1,h,1)
        x['controller_points']=x['controller_points'][:,:1].repeat(1,h,1,1,1)
        x['controller_mask']=x['controller_mask'][:,:1].repeat(1,h,1,1)
        x['action_state'][...,2]+=offset[None,:,None]
        x['controller_points'][...,2]+=offset[None,:,None,None]
        x['tactile']=torch.zeros_like(x['tactile'])
        x['valid_mask']=x['valid_mask'][:,:1].repeat(1,h,1,1,1)
        x['tactile_spatial_mask']=x['tactile_spatial_mask'][:,:1].repeat(1,h,1,1,1)
        x['tactile_age_us']=torch.zeros_like(x['tactile_age_us'])
        x['tactile_xyz']=x['tactile_xyz'][:,:1].repeat(1,h,1,1,1,1)
        b['inputs']=sanitize(x)
        y['target_xyz']=[];y['target_attr']=[];y['target_mask']=[]
        for k in range(h):
            target=x['gs_xyz'].clone();target[...,2]+=offset[k]
            target=torch.where(x['gs_mask'][...,None],target,0.)
            y['target_xyz'].append(target);y['target_attr'].append(x['gs_attr'].clone());y['target_mask'].append(x['gs_mask'].clone())
        if 'target_lengths' in y:y['target_lengths']=[x['gs_mask'].sum(-1) for _ in range(h)]
        if 'target_depth' in y:
            y['target_depth']=torch.zeros_like(y['target_depth'])
            y['target_depth_valid']=torch.ones_like(y['target_depth_valid'])
        b['meta']['synthetic']=dict(sign=sign,step_mm=2.,depth_not_ground_truth=True,
            target='rigid z translation from initial GS; attributes inherited')
        output.append(b)
    return output

ACTION_KEYS=('action_state','action_valid','controller_points','controller_mask')
TACTILE_KEYS=('tactile','valid_mask','tactile_spatial_mask','tactile_age_us','tactile_xyz')
def intervene(x,mode,other=None):
    y=dict(x)
    if mode=='freeze_action':
        for key in ACTION_KEYS:y[key]=x[key][:,:1].expand_as(x[key]).clone()
    elif mode=='zero_tactile':y['tactile']=torch.zeros_like(x['tactile'])
    elif mode in ('swap_action','swap_tactile','swap_all'):
        if other is None or not torch.equal(x['operation_hands'],other['operation_hands']):
            raise ValueError('Matched hand mode required for condition swap')
        keys=ACTION_KEYS if mode=='swap_action' else TACTILE_KEYS if mode=='swap_tactile' else ACTION_KEYS+TACTILE_KEYS
        for key in keys:
            if x[key].shape!=other[key].shape:raise ValueError('Condition shape mismatch')
            y[key]=other[key].clone()
    elif mode!='normal':raise ValueError(mode)
    return sanitize(y)

@torch.no_grad()
def metrics(pred,b,processor,renderer):
    m=prediction_metrics(pred,b,processor,renderer,512)
    x=b['inputs'];keep=x['gs_mask'][0];initial=x['gs_xyz'][0,keep];units=processor.state_dict()['scale']*1000
    m['motion']=[]
    for k in range(5):
        p=pred['xyz'][0,k,keep];t=b['targets']['target_xyz'][k][0,b['targets']['target_mask'][k][0]]
        delta=p-initial;mean=delta.mean(0);target=t.mean(0)-initial.mean(0)
        cosine=float(torch.nn.functional.cosine_similarity(mean[None],target[None]).item()) if mean.norm()>1e-9 and target.norm()>1e-9 else None
        cp=p-p.mean(0);ct=t-t.mean(0)
        m['motion'].append(dict(displacement_rms_mm=float(delta.square().sum(-1).mean().sqrt()*units),
            residual_rms_mm=float((delta-mean).square().sum(-1).mean().sqrt()*units),
            mean_vector_mm=(mean*units).tolist(),gt_set_centroid_vector_mm=(target*units).tolist(),
            centroid_error_mm=float((mean-target).norm()*units),direction_cosine=cosine,
            centered_cd_mm=float((nearest_distances(cp,ct).mean()+nearest_distances(ct,cp).mean())*.5*units)))
    return m

@torch.no_grad()
def evaluate(model,batches,processor,renderers):
    model.eval();out=[];preds=[]
    for i,b in enumerate(batches):
        renderer=renderers[i];x=b['inputs'];normal=model(x);preds.append(normal)
        row={'normal':metrics(normal,b,processor,renderer),'copy':metrics(copy_initial(x,'gs_geometry'),b,processor,renderer)}
        for mode in ('freeze_action','zero_tactile','swap_action','swap_tactile','swap_all'):
            p=model(intervene(x,mode,batches[1-i]['inputs']));r=metrics(p,b,processor,renderer)
            r['delta_cd_mm']=(np.array(r['cd_mm'])-row['normal']['cd_mm']).tolist()
            d=(p['xyz']-normal['xyz'])[0,:,x['gs_mask'][0]]*processor.state_dict()['scale']*1000
            r['output_difference_rms_mm']=d.square().sum(-1).mean(-1).sqrt().tolist()
            if mode=='swap_action' and 'synthetic' in b['meta']:
                r['matched_swapped_target']=metrics(p,batches[1-i],processor,renderers[1-i])
            row[mode]=r
        out.append(row)
    if 'synthetic' in batches[0]['meta']:
        keep=batches[0]['inputs']['gs_mask'][0];d=(preds[1]['xyz']-preds[0]['xyz'])[0,:,keep]
        separation=(d.square().sum(-1).mean(-1).sqrt()*processor.state_dict()['scale']*1000).tolist()
        for row in out:row['pair_prediction_separation_rms_mm']=separation
    return out,preds

@torch.no_grad()
def export(directory,pred,b,processor,renderer):
    from scripts.training.visualize import png_bytes
    d=Path(directory);d.mkdir(parents=True,exist_ok=False)
    renders={'gt':renderer.render_targets(b,processor),'pred':renderer(pred,b,processor),
             'copy':renderer(copy_initial(b['inputs'],'gs_geometry'),b,processor)}
    for label,values in renders.items():
        for mode in ('rgb','alpha'):
            images=values[mode][0].cpu().clamp(0,1).numpy()
            for h in range(5):
                for camera in range(images.shape[1]):
                    image=np.round(images[h,camera]*255).astype(np.uint8)
                    if image.ndim==2:image=np.repeat(image[...,None],3,-1)
                    (d/f'h{h+1:02d}-c{camera}-{mode}-{label}.png').write_bytes(png_bytes(image))
    x=b['inputs'];keep=x['gs_mask'][0];scale=processor.state_dict()['scale'];center=np.array(processor.state_dict()['center'])
    arrays={'initial_xyz_m':x['gs_xyz'][0,keep].cpu().numpy()*scale+center,
            'prediction_xyz_m':pred['xyz'][0,:,keep].cpu().numpy()*scale+center,
            'frame_ids':b['meta']['frame_ids'][0].cpu().numpy()}
    for k in range(5):arrays[f'gt_h{k+1}_xyz_m']=b['targets']['target_xyz'][k][0,b['targets']['target_mask'][k][0]].cpu().numpy()*scale+center
    np.savez_compressed(d/'geometry.npz',**arrays)
    return dict(folder=d.name,frames=arrays['frame_ids'].tolist(),horizon=5,cameras=3)
