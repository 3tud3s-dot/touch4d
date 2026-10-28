"""Read-only post-training G4/G0/Copy diagnostics on the two specified windows."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.training.model import initial_knn
from scripts.training.losses import nearest_distances
from scripts.training.v4_model import V4Stage1Predictor
from scripts.training.v4_runtime import load_checkpoint,prepare_batch,v4_objective
from scripts.training.train import copy_initial,make_renderer,to_device,atomic_json
from scripts.experiments.full_training.v4_epochs import loader
from .model import G4GlobalMotionPredictor
from .protocol import configuration
from .checkpoint import load_for_evaluation
from .smoke import sha,trace_numbers


def fixed_regions(x):
    keep=x['gs_mask'][0];ids=keep.nonzero().flatten()
    cp=x['controller_points'][0,0].reshape(60,3)
    mask=x['controller_mask'][0,0].reshape(60)&x['operation_hands'][0].repeat_interleave(30)&torch.isfinite(cp).all(-1)
    if not mask.any():raise ValueError('No initial valid operating controller')
    d=torch.cdist(x['gs_xyz'][0,ids],cp[mask],compute_mode='donot_use_mm_for_euclid_dist').min(-1).values
    order=d.argsort(stable=True);n=max(1,len(ids)//4)
    return dict(all=ids,near=ids[order[:n]],middle=ids[order[n:-n]],far=ids[order[-n:]])


def motion_metrics(pred,x,scale):
    """Initial distance quartiles; 16 non-self fixed neighbors, not material strain."""
    groups=fixed_regions(x);initial=x['gs_xyz'][0];factor=scale*1000
    neighbors,valid=initial_knn(x['gs_xyz'],x['gs_mask'],17)
    neighbors,valid=neighbors[0],valid[0]
    edge=initial[:,None]-initial[neighbors];length=edge.norm(dim=-1)
    edge_valid=valid & (neighbors!=torch.arange(len(initial),device=initial.device)[:,None]) & (length>1e-8)
    rows=[]
    for h,position in enumerate(pred['xyz'][0],1):
        delta=position-initial;global_mean=delta[groups['all']].mean(0)
        distance=(position[:,None]-position[neighbors]).norm(dim=-1)
        for name,ids in groups.items():
            if not len(ids):continue
            d=delta[ids];ratios=(distance/length.clamp_min(1e-8))[ids][edge_valid[ids]]
            rows.append(dict(horizon=h,region=name,mean_xyz_mm=(d.mean(0)*factor).tolist(),
                displacement_rms_mm=float(d.square().sum(-1).mean().sqrt())*factor,
                global_translation_removed_rms_mm=float((d-global_mean).square().sum(-1).mean().sqrt())*factor,
                neighbor_ratio_mean=float(ratios.mean()) if ratios.numel() else None,
                neighbor_distortion_p95=float(torch.quantile((ratios-1).abs(),.95)) if ratios.numel() else None))
    return rows


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('g4-checkpoint','g0-checkpoint','root','depth-root','output','expected-sha'):p.add_argument('--'+k,required=True)
    a=p.parse_args();source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source!=a.expected_sha or subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Source drift')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Requires authorized Slurm CUDA inference')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    hashes={p:sha(p) for p in (a.g4_checkpoint,a.g0_checkpoint)}
    processor=DataProcessor('001-rope');g4=G4GlobalMotionPredictor(horizon=15);g0=V4Stage1Predictor(horizon=15)
    s4=load_for_evaluation(a.g4_checkpoint,g4,processor.state_dict())
    s0=load_checkpoint(a.g0_checkpoint,g0,processor_state=processor.state_dict())
    g4.cuda().eval();g0.cuda().eval();c=configuration(a.root,a.depth_root,source);renderer=make_renderer(c)
    reports={};table=[]
    for split,ep,start in [('train',7,105),('val',3,135)]:
        dl=loader(c,split);ds=dl.dataset
        raw=collate_fn([ds[ds.windows.index((ep,start))]])
        batch=to_device(prepare_batch(raw,processor,g4),'cuda');x=batch['inputs']
        pred,trace=g4(x,return_trace=True)
        outputs=dict(g4=pred,g0=g0(x),copy_initial=copy_initial(x,'gs_geometry'))
        window={};groups=fixed_regions(x);factor=processor.state_dict()['scale']*1000
        for label,value in outputs.items():
            loss,m=v4_objective(value,batch,processor,renderer)
            if not torch.isfinite(loss):raise ValueError('Nonfinite evaluation')
            rows=motion_metrics(value,x,processor.state_dict()['scale'])
            for row in rows:
                h=row['horizon']-1;pos=value['xyz'][0,h];gt=batch['targets']['target_xyz'][h][0,batch['targets']['target_mask'][h][0]]
                ids=groups[row['region']]
                row['pred_to_gt_mm']=float(nearest_distances(pos[ids],gt,512).mean())*factor
                allids=groups['all']
                if row['region']=='all':row['cd_mm']=float((nearest_distances(pos[allids],gt,512).mean()+nearest_distances(gt,pos[allids],512).mean())*.5)*factor
                row.update(scene='001-rope',episode=ep,start=start,model=label,
                           rgb=float(m['rgb_by_horizon'][0,h]),alpha=float(m['alpha_by_horizon'][0,h]))
                table.append(row)
            window[label]=dict(joint=float(loss),metrics={k:v.detach().cpu().tolist() for k,v in m.items()},motion=rows)
        window['g4_contributions']=trace_numbers(trace,processor.state_dict()['scale'])
        window['frame_ids']=raw['meta']['frame_ids'].tolist();reports[f'ep{ep}_start{start}']=window
    if any(sha(p)!=h for p,h in hashes.items()):raise AssertionError('Checkpoint changed')
    atomic_json(out/'results.json',dict(source_sha=source,checkpoints=hashes,
        g4_completed_epochs=s4['ledger']['completed_epochs'],g0_successful_updates=s0['successful_steps'],windows=reports,
        notes='Same evaluation only; different checkpoint training budgets must be reported. GT has no material ID. Region error is one-way. No Oracle labels.'))
    keys=sorted(set().union(*(r.keys() for r in table)))
    with (out/'motion.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(table)


if __name__=='__main__':main()
