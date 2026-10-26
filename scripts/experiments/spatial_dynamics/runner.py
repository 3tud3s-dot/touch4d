"""Frozen cross-episode, successful-update screen; never the epoch/fixed400 runner.

freeze is a CPU data-contract audit. smoke/run require a Slurm CUDA allocation.
All paths are explicit; fresh run directories only; no automatic resume/retry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

import numpy as np
import torch
from scripts.dataloader.dataset import Touch4DDataset
from scripts.dataloader.collate import collate_fn
from scripts.data_process import DataProcessor
from scripts.training.model import PositionPredictor
from scripts.training.train import (atomic_json, atomic_checkpoint, to_device, make_renderer,
                                    copy_initial, gs_render_settings, GS_CAMERAS)
from scripts.training.losses import objective
from scripts.experiments.fixed_window.fixed_fit import prediction_metrics
from .model import SpatialPredictor, CANDIDATES, local_penalty, initial_knn
from .conditions import HAND_POLICY, hand_inputs
from .metrics import enriched_metrics

ARCH = dict(width=128, heads=4, layers=2, controller_features=7, prediction_mode='autoregressive',
            output_mode='gs_geometry', gs_local_attention='knn', gs_local_k=16, gs_decoder='anchors',
            anchor_count=128, anchor_pool_k=16, anchor_bandwidth=1., gs_structure='anchor_attention', horizon=5)
WEIGHTS = dict(chamfer_weight=1.,rgb_weight=1.,alpha_weight=.1,scale_weight=.01)
VERSION = 'touch4d-spatial-dynamics-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def paired_model(spatial, seed=42, decoder='D0', feedback='F0', regularization=0., **architecture):
    torch.manual_seed(seed)
    baseline=SpatialPredictor(spatial='S0',**architecture)
    reference=baseline.state_dict()
    torch.manual_seed(seed)
    model=SpatialPredictor(spatial=spatial,decoder=decoder,feedback=feedback,regularization=regularization,**architecture)
    shared={k:tensor_sha(v) for k,v in model.state_dict().items() if k in reference and v.shape==reference[k].shape}
    for k in shared:
        if not torch.equal(model.state_dict()[k],reference[k]):raise AssertionError('Initialization drift: '+k)
    return model,dict(shared_sha256=shared,shared_count=len(shared),parameter_count=sum(p.numel() for p in model.parameters()))


def dataset(root,split,horizon=5):
    return Touch4DDataset('001-rope',split,root,with_depth=True,depth_cameras=GS_CAMERAS['001-rope'],horizon=horizon,cache_episodes=8)


def update_order(count,seed=42):
    rng=random.Random(seed);order=[]
    while len(order)<count:
        block=list(range(24));rng.shuffle(block);order.extend(block)
    return order[:count]


def freeze(root,output,previous):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    old=json.loads(Path(previous).read_text())
    for path,digest in old['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Source changed: '+path)
    if update_order(200)!=old['update_order']:raise ValueError('Sample order drift')
    windows={};cases=[(1,85),(2,73),(3,65),(3,115),(3,150),(6,110)]
    for h in (5,15):
        windows[str(h)]={}
        for split in ('train','val'):
            ds=dataset(root,split,h)
            rows=[dict(episode=r['episode'],start=r['start']) for r in old['windows'][split]]
            if h==15:rows += [dict(episode=e,start=s,case=True) for e,s in cases if e in ds.episodes]
            for row in rows:
                e,start=row['episode'],row['start']
                if (e,start) not in ds.windows:
                    # Purposeful cases may not lie on the default validation stride.
                    if start<0 or start+h>=ds.episodes[e].frames:raise ValueError('Illegal H15 window')
                    ds.windows=ds.windows+((e,start),)
                sample=ds[ds.windows.index((e,start))];batch=collate_fn([sample]);DataProcessor('001-rope')._validate_batch(batch)
                if not batch['targets']['target_depth_valid'].flatten(-2).any(-1).all():raise ValueError('Missing view supervision')
                row.update(frame_ids=batch['meta']['frame_ids'][0].tolist(),hands=['right'] if e<5 else ['left','right'],
                    timestamps_us=ds.episodes[e].timestamps[start:start+h+1].tolist(),
                    initial_points=int(batch['inputs']['gs_mask'].sum()),target_points=batch['targets']['target_lengths'][0].tolist(),
                    depth_valid_pixels=batch['targets']['target_depth_valid'].flatten(-2).sum(-1)[0].tolist())
                from scripts.experiments.fixed_window.fixed_fit import digest as tensor_digest
                row['data_digest']=tensor_digest(batch)
            windows[str(h)][split]=[r for r in rows if not r.get('case')]
            windows[str(h)].setdefault('cases',[]).extend(r for r in rows if r.get('case'))
    value=dict(version=VERSION,source_manifest_sha256=sha(previous),source_sha256=old['source_sha256'],windows=windows,
        hand_policy=HAND_POLICY,weights=WEIGHTS,cameras=GS_CAMERAS['001-rope'],base_architecture=ARCH,
        selection='same pre-existing 24 train/8 validation starts; separately validated H15, no model scoring',
        optimizer=dict(type='AdamW',lr=1e-4,weight_decay=1e-4,gradient_clip=1.))
    atomic_json(output,value);return value


def read_manifest(path):
    value=json.loads(Path(path).read_text())
    if value['version']!=VERSION or value['weights']!=WEIGHTS:raise ValueError('Protocol mismatch')
    for path,digest in value['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Data drift: '+path)
    return value


def save_checkpoint(path, model, optimizer, processor, manifest_sha, updates, history):
    atomic_checkpoint(path,dict(version=VERSION,config=dict(model.config),model=model.state_dict(),
        optimizer=optimizer.state_dict(),processor=processor.state_dict(),manifest_sha256=manifest_sha,
        completed_updates=updates,history=history,weights=WEIGHTS,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))


def restore(path, model, optimizer, processor, manifest_sha):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if (saved['version']!=VERSION or saved['config']!=model.config or saved['weights']!=WEIGHTS
            or saved['processor']!=processor.state_dict() or saved['manifest_sha256']!=manifest_sha):
        raise ValueError('Incompatible condition checkpoint; refusing restore')
    model.load_state_dict(saved['model'],strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'])
    if saved['cuda_rng'] is not None: torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved['completed_updates']


def batch_at(ds, row, processor, device):
    key=(row['episode'],row['start'])
    if key not in ds.windows:ds.windows=ds.windows+(key,)
    index=ds.windows.index(key)
    batch=processor(to_device(collate_fn([ds[index]]),device))
    if batch['meta']['frame_ids'][0].tolist()!=row['frame_ids']:
        raise ValueError('Actual frame IDs differ from manifest')
    batch['inputs']=hand_inputs(batch['inputs'],batch['meta']['episode_id'],batch['meta']['scene'])
    return batch


def perturb(inputs, mode):
    x=dict(inputs)
    if mode=='freeze_action':
        # Keep masks/current tactile; freeze pose AND controller trajectory consistently.
        for value, mask in [('action_state','action_valid'),('controller_points','controller_mask')]:
            keep=x[mask] & x[mask][:,:1]
            x[mask]=keep
            x[value]=torch.where(keep[...,None],x[value][:,:1].expand_as(x[value]),0.)
    elif mode=='zero_tactile':
        x['tactile']=torch.zeros_like(x['tactile'])
    return x


def summarize(rows):
    episodes={}
    for ep in sorted({r['episode'] for r in rows}):
        group=[r for r in rows if r['episode']==ep]
        per={k:np.mean([r[k] for r in group],axis=0).tolist() for k in ('cd_mm','rgb','alpha')}
        per.update(mean_cd_mm=float(np.mean(per['cd_mm'])),h1_cd_mm=per['cd_mm'][0],
                   h2_h5_cd_mm=float(np.mean(per['cd_mm'][1:])),windows=len(group),
                   hand_mode='single' if ep<5 else 'dual')
        per['joint']=per['mean_cd_mm']/(1000*DataProcessor('001-rope').state_dict()['scale'])+float(np.mean(per['rgb']))+.1*float(np.mean(per['alpha']))
        episodes[str(ep)]=per
    return dict(episodes=episodes,**{k:np.mean([r[k] for r in episodes.values()],axis=0).tolist()
        for k in ('cd_mm','rgb','alpha','mean_cd_mm','h1_cd_mm','h2_h5_cd_mm','joint')})


@torch.no_grad()
def evaluate(model, ds, rows, processor, renderer):
    model.eval(); results={k:[] for k in ('model','copy_initial','freeze_action','zero_tactile')}
    for row in rows:
        batch=batch_at(ds,row,processor,'cuda')
        normal=model(batch['inputs'])
        for name in results:
            pred=(normal if name=='model' else copy_initial(batch['inputs'],'gs_geometry') if name=='copy_initial'
                  else model(perturb(batch['inputs'],name)))
            metrics=enriched_metrics(pred,batch,processor,renderer,model.trace if name=='model' else None)
            metrics.update(episode=row['episode'],start=row['start'],frame_ids=row['frame_ids'])
            delta=(pred['xyz']-batch['inputs']['gs_xyz'][:,None]).norm(dim=-1)
            metrics['mean_displacement_mm']=(delta[0,:,batch['inputs']['gs_mask'][0]].mean(-1)*processor.state_dict()['scale']*1000).tolist()
            metrics['prediction_change_mm']=((pred['xyz']-normal['xyz']).norm(dim=-1)[0,:,batch['inputs']['gs_mask'][0]].mean(-1)*processor.state_dict()['scale']*1000).tolist()
            results[name].append(metrics)
        del batch,normal,pred
    output={k:dict(windows=v,summary=summarize(v)) for k,v in results.items()}
    for name in ('freeze_action','zero_tactile'):
        output[name]['delta_cd_mm']=(np.array(output[name]['summary']['cd_mm'])-np.array(output['model']['summary']['cd_mm'])).tolist()
    output['interpretation']='Copy-initial CD is GT set change, not per-Gaussian ground-truth motion. Positive perturbation delta CD means real conditions helped this comparison. Short-budget validation only.'
    return output


def run(args):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('CUDA allocation required')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if commit!=args.expected_sha or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise RuntimeError('Execution source drift')
    manifest=read_manifest(args.manifest); digest=sha(args.manifest)
    run=Path(args.output);run.mkdir(parents=True,exist_ok=False)
    try:
        random.seed(args.seed);np.random.seed(args.seed);torch.set_num_threads(4)
        model,pairing=paired_model(args.mode,seed=args.seed,decoder=args.decoder,feedback=args.feedback,regularization=args.regularization,**dict(ARCH,horizon=args.horizon));model=model.cuda()
        processor=DataProcessor('001-rope')
        renderer=make_renderer(dict(output_mode='gs_geometry',render_settings=gs_render_settings()))
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        ds=dataset(args.root,'train',args.horizon)
        atomic_json(run/'config.json',dict(source_sha=commit,model=model.config,weights=WEIGHTS,
            manifest_sha256=digest,manifest=manifest,mode=args.mode,seed=args.seed,budget=args.updates,slurm_job_id=os.environ['SLURM_JOB_ID'],started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),smoke=args.command=='smoke',
            torch_version=torch.__version__,device=torch.cuda.get_device_name(),pairing=pairing))
        updates=2 if args.command=='smoke' else args.updates
        windows=manifest['windows'][str(args.horizon)];order=update_order(updates,args.seed)
        evaluations={};best=float('inf')
        history=[];torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        gradient_max={}
        for step in range(updates):
            row=windows['train'][order[step]]
            batch=batch_at(ds,row,processor,'cuda');model.train();optimizer.zero_grad(set_to_none=True)
            prediction=model(batch['inputs'])
            loss,values=objective(prediction,batch,processor,renderer,0.,512,**WEIGHTS)
            if not values['render_time_available'].all(): raise ValueError('Missing supervised horizon; no skipped updates')
            r1=local_penalty(prediction['xyz'],batch['inputs']['gs_xyz'],batch['inputs']['gs_mask'],initial_knn(batch['inputs']['gs_xyz'],batch['inputs']['gs_mask'],16)) if args.regularization else loss*0
            loss=loss+args.regularization*r1
            loss.backward()
            for name,parameter in model.named_parameters():
                if parameter.grad is not None:
                    prefix=name.split('.')[0]
                    gradient_max[prefix]=max(gradient_max.get(prefix,0.),float(parameter.grad.detach().abs().max()))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()): raise FloatingPointError('Nonfinite weights')
            history.append(dict(update=step+1,episode=row['episode'],start=row['start'],
                loss=float(loss.detach()),r1=float(r1.detach()),gradient_norm=float(norm),
                **{k:float(values[k].detach().mean()) for k in ('geometry','rgb','alpha','scale_regularization')}))
            del prediction,loss,values,batch,r1
            model.trace=[]
            if (step+1)%20==0 or step+1==updates:
                atomic_json(run/'updates.json',history)
                save_checkpoint(run/'last.pt',model,optimizer,processor,digest,step+1,history)
                print(json.dumps(history[-1]),flush=True)
            milestones=([200,400] if args.updates==400 else [400,800,1200])
            if args.command=='run' and step+1 in milestones:
                metrics=evaluate(model,dataset(args.root,'val',args.horizon),windows['val'],processor,renderer)
                atomic_json(run/f'metrics_{step+1}.json',metrics)
                evaluations[str(step+1)]=metrics['model']['summary']
                save_checkpoint(run/f'checkpoint_{step+1}.pt',model,optimizer,processor,digest,step+1,history)
                if metrics['model']['summary']['mean_cd_mm']<best:
                    best=metrics['model']['summary']['mean_cd_mm']
                    save_checkpoint(run/'best.pt',model,optimizer,processor,digest,step+1,history)
                model.trace=[]

        torch.cuda.synchronize()
        resources=dict(allocated_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                       elapsed_seconds=time.monotonic()-started,successful_updates=updates)
        required=['controller_encoder','position_head','state_encoder']
        required+=['pressure_encoder','fusion'] if updates>2 else []
        if args.mode not in ('S0','S2-C0'):required+=['spatial_bias']
        if args.decoder in ('D1','D2'):required+=['deformation_head']
        if any(gradient_max.get(k,0.)<=0 for k in required):
            raise RuntimeError(f'Missing trained gradient path: {gradient_max}')
        # Restore into same instance, compare exact tensors/prediction without another optimizer step.
        batch=batch_at(ds,windows['train'][0],processor,'cuda');model.eval()
        with torch.no_grad(): before=model(batch['inputs'])
        assert restore(run/'last.pt',model,optimizer,processor,digest)==updates
        with torch.no_grad(): after=model(batch['inputs'])
        assert all(torch.equal(before[k],after[k]) for k in before),'Checkpoint output changed'
        del batch,before,after
        if args.command=='run':
            atomic_json(run/'metrics.json',metrics)
        atomic_json(run/'status.json',dict(status='complete',source_sha=commit,resources=resources,checkpoint_restore_exact=True,
            gradient_max=gradient_max,evaluations=evaluations,checkpoint_sha256=sha(run/'last.pt'),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),scientific_evaluation=args.command=='run'))
    except BaseException:
        atomic_json(run/'status.json',dict(status='failed',source_sha=commit,error=traceback.format_exc()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['freeze','smoke','run'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--mode',choices=CANDIDATES)
    parser.add_argument('--decoder',choices=['D0','D1','D2'],default='D0')
    parser.add_argument('--feedback',choices=['F0','F1'],default='F0')
    parser.add_argument('--regularization',type=float,choices=[0.,.01],default=0.)
    parser.add_argument('--horizon',type=int,choices=[5,15],default=5)
    parser.add_argument('--seed',type=int,choices=[42,43],default=42)
    parser.add_argument('--updates',type=int,choices=[400,1200],default=400)
    parser.add_argument('--previous-manifest')
    parser.add_argument('--expected-sha')
    args=parser.parse_args()
    if args.command=='freeze': freeze(args.root,args.output,args.previous_manifest)
    else:
        if not all((args.manifest,args.mode,args.expected_sha)):parser.error('mode, manifest and SHA required')
        run(args)


if __name__=='__main__': main()
