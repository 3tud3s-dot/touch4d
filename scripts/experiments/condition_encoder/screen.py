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
from .model import ConditionPredictor, MODES, HAND_POLICY, hand_inputs

ARCH = dict(width=128, heads=4, layers=2, controller_features=7, prediction_mode='autoregressive',
            output_mode='gs_geometry', gs_local_attention='knn', gs_local_k=16, gs_decoder='anchors',
            anchor_count=128, anchor_pool_k=16, anchor_bandwidth=1., gs_structure='anchor_attention', horizon=5)
WEIGHTS = dict(chamfer_weight=1.,rgb_weight=1.,alpha_weight=.1,scale_weight=.01)
VERSION = 'touch4d-condition-screen-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def paired_model(mode, **architecture):
    # Construct each baseline with identical RNG; added modules occur at the end.
    torch.manual_seed(42)
    baseline = PositionPredictor(**architecture)
    reference = baseline.state_dict()
    torch.manual_seed(42)
    model = ConditionPredictor(condition_mode=mode, **architecture)
    shared = {k: tensor_sha(v) for k,v in model.state_dict().items()
              if k in reference and v.shape == reference[k].shape}
    for key in shared:
        if not torch.equal(model.state_dict()[key],reference[key]):
            raise AssertionError(f'Unpaired common parameter: {key}')
    return model, dict(shared_sha256=shared, shared_count=len(shared),
                       parameter_count=sum(p.numel() for p in model.parameters()))


def dataset(root, split):
    return Touch4DDataset('001-rope',split,root,with_depth=True,
                          depth_cameras=GS_CAMERAS['001-rope'],horizon=5,cache_episodes=8)


def select(windows, count=4):
    """Temporal quartile midpoints of eligible starts, no output/target scoring."""
    result=[]
    for ep in sorted({e for e,_ in windows}):
        candidates=[w for w in windows if w[0]==ep]
        if len(candidates)<count:
            raise ValueError(f'Insufficient eligible windows: episode {ep}')
        result.extend(candidates[min(len(candidates)-1,int((i+.5)*len(candidates)/count))] for i in range(count))
    if len(set(result)) != len(result):
        raise ValueError('Duplicate selected window')
    return result


def freeze(root, output):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    selected={}; source={}
    for split, expected in [('train',[1,2,4,5,7,8]),('val',[3,6])]:
        ds=dataset(root,split)
        windows=select(ds.windows)
        if sorted({e for e,_ in windows})!=expected: raise ValueError('Split drift')
        selected[split]=[dict(episode=e,start=s,frame_ids=list(range(s,s+6)),
            timestamps_us=ds.episodes[e].timestamps[s:s+6].tolist(),hands=['right'] if e<5 else ['left','right'])
            for e,s in windows]
        # Read each selected sample once on CPU before any GPU job is submitted.
        # This audits real NPZ/HDF5 contracts and per-horizon supervised regions.
        for row in selected[split]:
            sample=ds[ds.windows.index((row['episode'],row['start']))]
            batch=collate_fn([sample])
            DataProcessor('001-rope')._validate_batch(batch)
            if batch['meta']['frame_ids'][0].tolist()!=row['frame_ids']:
                raise ValueError('Manifest frame binding mismatch')
            depth=batch['targets']['target_depth_valid']
            if not depth.flatten(-2).any(-1).all():
                raise ValueError('Selected camera/horizon has no valid supervision')
            row['initial_points']=int(batch['inputs']['gs_mask'].sum())
            row['target_points']=batch['targets']['target_lengths'][0].tolist()
            row['depth_valid_pixels']=depth.flatten(-2).sum(-1)[0].tolist()
            del sample,batch
        for ep in expected:
            for directory in (ds.episodes[ep].path, ds.depth_episodes[ep].path):
                for name in ('metadata.json','timestamps.npz','controller.npz','tactile.npz','camera_parameters.npz'):
                    path=directory/name
                    if path.is_file(): source[str(path)]=sha(path)
    rng=random.Random(42); order=[]
    while len(order)<200:
        block=list(range(len(selected['train'])));rng.shuffle(block);order.extend(block)
    value=dict(version=VERSION,scene='001-rope',horizon=5,seed=42,hand_policy=HAND_POLICY,
        windows=selected,update_order=order[:200],source_sha256=source,
        selection='4 temporal quartile-midpoint eligible starts per episode; frozen before predictions',
        architecture=ARCH,weights=WEIGHTS,cameras=GS_CAMERAS['001-rope'],
        optimizer=dict(type='AdamW',lr=1e-4,weight_decay=1e-4,gradient_clip=1.),
        evaluation='Only final update 200; per-window/per-episode/horizon; equal episode means; no test split')
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(output,value)
    return value


def read_manifest(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if (value['version']!=VERSION or value['architecture']!=ARCH or value['weights']!=WEIGHTS
            or value['hand_policy']!=HAND_POLICY or len(value['update_order'])!=200):
        raise ValueError('Frozen protocol mismatch')
    for path,digest in value['source_sha256'].items():
        if sha(path)!=digest: raise ValueError(f'Data source drift: {path}')
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
    index=ds.windows.index((row['episode'],row['start']))
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
            metrics=prediction_metrics(pred,batch,processor,renderer,512)
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
        random.seed(42);np.random.seed(42);torch.set_num_threads(4)
        model,pairing=paired_model(args.mode,**ARCH);model=model.cuda()
        processor=DataProcessor('001-rope')
        renderer=make_renderer(dict(output_mode='gs_geometry',render_settings=gs_render_settings()))
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        ds=dataset(args.root,'train')
        atomic_json(run/'config.json',dict(source_sha=commit,model=model.config,weights=WEIGHTS,
            manifest_sha256=digest,manifest=manifest,mode=args.mode,smoke=args.command=='smoke',
            torch_version=torch.__version__,device=torch.cuda.get_device_name(),pairing=pairing))
        updates=10 if args.command=='smoke' else 200
        history=[];torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        gradient_max={}
        for step in range(updates):
            row=manifest['windows']['train'][manifest['update_order'][step]]
            batch=batch_at(ds,row,processor,'cuda');model.train();optimizer.zero_grad(set_to_none=True)
            prediction=model(batch['inputs'])
            loss,values=objective(prediction,batch,processor,renderer,0.,512,**WEIGHTS)
            if not values['render_time_available'].all(): raise ValueError('Missing supervised horizon; no skipped updates')
            loss.backward()
            for name,parameter in model.named_parameters():
                if parameter.grad is not None:
                    prefix=name.split('.')[0]
                    gradient_max[prefix]=max(gradient_max.get(prefix,0.),float(parameter.grad.detach().abs().max()))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()): raise FloatingPointError('Nonfinite weights')
            history.append(dict(update=step+1,episode=row['episode'],start=row['start'],
                loss=float(loss.detach()),gradient_norm=float(norm),
                **{k:float(values[k].detach().mean()) for k in ('geometry','rgb','alpha','scale_regularization')}))
            del prediction,loss,values,batch
            if (step+1)%20==0 or step+1==updates:
                atomic_json(run/'updates.json',history)
                save_checkpoint(run/'last.pt',model,optimizer,processor,digest,step+1,history)
                print(json.dumps(history[-1]),flush=True)
        torch.cuda.synchronize()
        resources=dict(allocated_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                       elapsed_seconds=time.monotonic()-started,successful_updates=updates)
        required=['controller_encoder','position_head','state_encoder']
        required+=['pressure_encoder','fusion'] if args.mode in ('T2','T3') else ['tactile_encoder']
        if any(gradient_max.get(k,0.)<=0 for k in required):
            raise RuntimeError(f'Missing trained gradient path: {gradient_max}')
        # Restore into same instance, compare exact tensors/prediction without another optimizer step.
        batch=batch_at(ds,manifest['windows']['train'][0],processor,'cuda');model.eval()
        with torch.no_grad(): before=model(batch['inputs'])
        assert restore(run/'last.pt',model,optimizer,processor,digest)==updates
        with torch.no_grad(): after=model(batch['inputs'])
        assert all(torch.equal(before[k],after[k]) for k in before),'Checkpoint output changed'
        del batch,before,after
        if args.command=='run':
            metrics=evaluate(model,dataset(args.root,'val'),manifest['windows']['val'],processor,renderer)
            atomic_json(run/'metrics.json',metrics)
        atomic_json(run/'status.json',dict(status='complete',source_sha=commit,resources=resources,checkpoint_restore_exact=True,
            gradient_max=gradient_max,scientific_evaluation=args.command=='run'))
    except BaseException:
        atomic_json(run/'status.json',dict(status='failed',source_sha=commit,error=traceback.format_exc()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['freeze','smoke','run'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--mode',choices=MODES)
    parser.add_argument('--expected-sha')
    args=parser.parse_args()
    if args.command=='freeze': freeze(args.root,args.output)
    else:
        if not all((args.manifest,args.mode,args.expected_sha)):parser.error('mode, manifest and SHA required')
        run(args)


if __name__=='__main__': main()
