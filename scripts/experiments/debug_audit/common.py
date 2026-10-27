"""Shared diagnostic measurements using the accepted data and metric functions."""
from pathlib import Path
import numpy as np
import torch
from scripts.experiments.spatial_dynamics.runner import dataset, batch_at, perturb
from scripts.experiments.spatial_dynamics.metrics import enriched_metrics
from scripts.training.train import copy_initial

CASES = [('train',2,73),('val',3,115),('train',1,85)]


def get_batch(root, processor, episode=2, start=73, horizon=5, device='cuda'):
    split='train' if episode in (1,2,4,5,7,8) else 'val'
    ds=dataset(root,split,horizon)
    if episode not in ds.episodes or start<0 or start+horizon>=ds.episodes[episode].frames:
        raise ValueError('Window outside one permitted episode')
    row=dict(episode=episode,start=start,frame_ids=list(range(start,start+horizon+1)))
    return batch_at(ds,row,processor,device)


@torch.no_grad()
def measurements(pred,batch,processor,renderer,trace=None):
    result=enriched_metrics(pred,batch,processor,renderer,trace)
    units=processor.state_dict()['scale']*1000
    keep=batch['inputs']['gs_mask'][0]
    delta=pred['xyz'][0,:,keep]-batch['inputs']['gs_xyz'][0,keep]
    mean=delta.mean(1,keepdim=True)
    result['nonrigid_residual_rms_mm']=((delta-mean).square().sum(-1).mean(-1).sqrt()*units).tolist()
    result['displacement_mean_vector_mm']=(mean[:,0]*units).tolist()
    result['displacement_max_mm']=(delta.norm(dim=-1).amax(-1)*units).tolist()
    result['mean_cd_mm']=float(np.mean(result['cd_mm']))
    if trace:
        result['anchors']=[]
        for row in trace:
            d=row['displacement'][0,row['fixed_mask'][0]]*units
            result['anchors'].append(dict(mean_vector_mm=d.mean(0).tolist(),
                rms_mm=float(d.square().sum(-1).mean().sqrt()),max_mm=float(d.norm(dim=-1).max()),
                residual_rms_mm=float((d-d.mean(0)).square().sum(-1).mean().sqrt())))
    return result


@torch.no_grad()
def evaluate_conditions(model,batch,processor,renderer):
    model.eval();normal=model(batch['inputs'])
    result={'model':measurements(normal,batch,processor,renderer,model.trace)}
    keep=batch['inputs']['gs_mask'][0];units=processor.state_dict()['scale']*1000
    for mode in ('copy_initial','freeze_action','zero_tactile'):
        pred=copy_initial(batch['inputs'],'gs_geometry') if mode=='copy_initial' else model(perturb(batch['inputs'],mode))
        values=measurements(pred,batch,processor,renderer)
        values['delta_cd_mm']=(np.array(values['cd_mm'])-np.array(result['model']['cd_mm'])).tolist()
        values['output_change_mean_mm']=((pred['xyz'][0,:,keep]-normal['xyz'][0,:,keep]).norm(dim=-1).mean(-1)*units).tolist()
        result[mode]=values
    return result,normal


@torch.no_grad()
def export_evidence(directory,pred,batch,processor,renderer):
    """Small offline three-camera evidence, all five distinct states, no CDN."""
    from PIL import Image
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    copy=copy_initial(batch['inputs'],'gs_geometry')
    rendered={'gt':renderer.render_targets(batch,processor),'pred':renderer(pred,batch,processor),
              'copy':renderer(copy,batch,processor)}
    for label,values in rendered.items():
        for mode in ('rgb','alpha'):
            image=values[mode][0].detach().cpu().clamp(0,1).numpy()
            for h in range(len(image)):
                for view in range(image.shape[1]):
                    frame=np.round(image[h,view]*255).astype(np.uint8)
                    Image.fromarray(frame).save(directory/f'h{h+1:02d}-c{view}-{mode}-{label}.png')
    keep=batch['inputs']['gs_mask'][0]
    units=processor.state_dict()['scale']
    center=np.asarray(processor.state_dict()['center'])
    arrays={'initial_xyz_m':batch['inputs']['gs_xyz'][0,keep].cpu().numpy()*units+center,
            'prediction_xyz_m':pred['xyz'][0,:,keep].cpu().numpy()*units+center,
            'frame_ids':batch['meta']['frame_ids'][0].cpu().numpy()}
    for h in range(pred['xyz'].shape[1]):
        mask=batch['targets']['target_mask'][h][0]
        arrays[f'gt_h{h+1}_xyz_m']=batch['targets']['target_xyz'][h][0,mask].cpu().numpy()*units+center
    np.savez_compressed(directory/'geometry.npz',**arrays)
    # Independent center-set panels, common axes; no target point correspondence.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    all_points=np.concatenate([arrays['initial_xyz_m'],arrays['prediction_xyz_m'].reshape(-1,3),
                               *[arrays[f'gt_h{h+1}_xyz_m'] for h in range(pred['xyz'].shape[1])]])*1000
    low,high=all_points.min(0),all_points.max(0);margin=np.maximum((high-low)*.05,1.)
    for h in range(pred['xyz'].shape[1]):
        fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
        for col,(label,points) in enumerate([('GT',arrays[f'gt_h{h+1}_xyz_m']),
                      ('Prediction',arrays['prediction_xyz_m'][h]),('Copy initial',arrays['initial_xyz_m'])]):
            for row,(a,b) in enumerate([(0,2),(0,1)]):
                ax=axes[row,col];ax.scatter(points[:,a]*1000,points[:,b]*1000,s=1,alpha=.6)
                ax.set(xlim=(low[a]-margin[a],high[a]+margin[a]),ylim=(low[b]-margin[b],high[b]+margin[b]),
                       xlabel='XYZ'[a]+' (mm)',ylabel='XYZ'[b]+' (mm)',title=f'{label} h{h+1}')
                ax.set_aspect('equal')
        fig.savefig(directory/f'h{h+1:02d}-centers.png',dpi=120);plt.close(fig)
    return dict(folder=directory.name,horizon=pred['xyz'].shape[1],cameras=rendered['gt']['alpha'].shape[2],
                frame_ids=arrays['frame_ids'].tolist(),units='world metres')
