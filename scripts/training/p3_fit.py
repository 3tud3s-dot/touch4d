"""Frozen P3-v0 four-arm, 400-update fits using existing data/loss/render code.

Every optimizer update has an atomic durable last.pt. A resume starts after that
recorded update; an interrupted boundary evaluation is repeated without another
optimizer step. Evaluations and descriptive lowest-CD selection occur every 20.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from types import SimpleNamespace

import torch

from ..dataloader.dataset import Touch4DDataset, read_gs, GS_ATTR_FIELDS
from ..dataloader.collate import collate_fn
from ..data_process import DataProcessor
from .fixed_fit import (WINDOWS, WEIGHTS, digest, manifest, select_windows, TargetCache,
                        evaluate_fixed, candidate, better)
from .model import PositionPredictor, initial_knn
from .p3_model import P3Predictor
from .train import (PROJECT, MODEL_KEYS, GS_CAMERAS, GS_CONVENTION, gs_render_settings,
                    atomic_checkpoint, atomic_json, make_renderer, to_device)

VERSION = 'touch4d-p3-fit-v0'
VARIANTS = dict(M0=(1, False), M1=(1, False), M2=(2, False), M3=(2, True))
AUDIT = PROJECT/'audit/p3-two-frame-h10-v0'
P3_KEYS = (*MODEL_KEYS, 'architecture', 'history', 'template_refinement', 'scale_bound', 'opacity_bound')


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): result.update(block)
    return result.hexdigest()


def training_config(scene, variant, audit_path=AUDIT/'stage0_audit.json'):
    if scene not in WINDOWS or variant not in VARIANTS:
        raise ValueError('Unsupported frozen scene/variant')
    history, refine = VARIANTS[variant]
    config=dict(scene=scene,variant=variant,history=history,template_refinement=refine,
        architecture='p2-cumulative-v7' if variant=='M0' else 'p3-one-step-v0',
        scale_bound=1.3,opacity_bound=.5,horizon=10,width=128,heads=4,layers=2,
        controller_features=7,prediction_mode='autoregressive',output_mode='gs_geometry',
        gs_local_attention='knn',gs_local_k=16,gs_decoder='anchors',gs_structure='anchor_attention',
        anchor_count=128,anchor_pool_k=16,anchor_bandwidth=1.,seed=42,batch_size=1,
        lr=1e-4,weight_decay=1e-4,gradient_clip=1.,chamfer_chunk=512,depth_weight=0.,**WEIGHTS,
        root=str(PROJECT/'datasets/touch4d-v1'),depth_root=str(PROJECT/'datasets/touch4d-depth-v1'),
        depth_cameras=GS_CAMERAS[scene],render_settings=gs_render_settings(),gs_convention=GS_CONVENTION,
        fixed_windows=[list(w) for w in WINDOWS[scene]],audit_path=str(Path(audit_path).resolve()),
        audit_sha256=sha(audit_path),updates=400,eval_every=20,optimizer='AdamW',
        protocol='p3-fixed400-v0',causal_conditions='prefix through t+k-1; online tactile',
        position_reference='initial t' if variant=='M0' else 'current predicted state',
        attribute_reference='observed t',source_sha256=digest({str(p.relative_to(PROJECT)):sha(p)
            for folder in ('training','dataloader','data_process')
            for p in sorted((PROJECT/'scripts'/folder).glob('*.py')) if not p.name.startswith('test_')}))
    return config


def construct(config):
    variant=config.get('variant')
    if variant not in VARIANTS or (config.get('history'),config.get('template_refinement')) != VARIANTS[variant]:
        raise ValueError('Variant/history/refinement configuration conflict')
    if variant=='M0':
        if config.get('architecture') != 'p2-cumulative-v7': raise ValueError('Wrong M0 architecture')
        return PositionPredictor(**{k:config[k] for k in MODEL_KEYS})
    return P3Predictor(**{k:config[k] for k in P3_KEYS})


def current_only(raw):
    return dict(inputs={k:v for k,v in raw['inputs'].items() if not k.startswith('gs_prev_')},
                targets=raw['targets'],meta={k:v for k,v in raw['meta'].items() if not k.startswith('history_')})


def load_raws(config):
    document=json.loads(Path(config['audit_path']).read_text())
    if (sha(config['audit_path'])!=config['audit_sha256'] or document.get('version')!='p3-window-audit-v0'
            or document.get('horizon')!=config['horizon'] or document.get('baseline_commit')!='fdd4013c7f166daae12379f4ee020b4da5724965'):
        raise ValueError('P3 frozen audit provenance mismatch')
    ds=Touch4DDataset(config['scene'],'train',config['root'],horizon=config['horizon'],history=config['history'],
        with_depth=True,depth_root=config['depth_root'],depth_cameras=config['depth_cameras'])
    raws=select_windows(ds,config['scene'],config['fixed_windows'])
    recorded=document['windows'][config['scene']]
    if len(recorded)!=2: raise ValueError('Expected two audited windows')
    for raw,row in zip(raws,recorded):
        actual=manifest([current_only(raw)],config)[0]
        if any(actual[k]!=row[k] for k in actual): raise ValueError('Current/target data differs from frozen Stage0')
        ep,start=row['episode_id'],row['start_frame']
        if row['history_frame_ids']!=[start-1,start]: raise ValueError('Bad frozen history')
        for frame,expected in row['gs_file_sha256'].items():
            if sha(ds.episodes[ep].path/'gs'/f'frame_{int(frame):06d}.npz')!=expected:
                raise ValueError('Historical/target GS source changed since freeze')
    return raws,ds,recorded


def read_state(path, expected_config=None):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('version')!=VERSION: raise ValueError('Expected explicit P3 fit checkpoint version')
    cfg=state['config']; n=state['completed_updates']; history=state['history']
    if expected_config is not None and cfg!=expected_config: raise ValueError('Resume config/architecture/H/data/source mismatch')
    if type(n) is not int or not 0<=n<=400: raise ValueError('Invalid completed optimizer update')
    DataProcessor.from_state_dict(state['processor'])
    rows=[r['update'] for r in history]
    expected=list(range(20,n+1,20))
    if rows!=expected and not (n>0 and n%20==0 and rows==expected[:-1]):
        raise ValueError('Incomplete checkpoint evaluation history')
    if state['processor']['scene']!=cfg['scene']: raise ValueError('Processor scene mismatch')
    # Constructor validates all mode fields; strict state loading rejects omitted modules.
    with torch.random.fork_rng(devices=[]):
        probe=construct(cfg); probe.load_state_dict(state['model'],strict=True)
    return state


def recover_best(run, state):
    path=Path(run)/'lowest_cd.pt'; recorded=state.get('best_candidate')
    if not path.is_file():
        if recorded is not None: raise ValueError('Recorded lowest-CD checkpoint is missing')
        return None
    saved=read_state(path,state['config'])
    actual=saved['best_candidate']
    if (saved['manifest']!=state['manifest'] or actual is None
            or actual['update']!=saved['completed_updates']
            or actual['update']>state['completed_updates']
            or (recorded is not None and actual!=recorded and not better(actual,recorded,None))):
        raise ValueError('Inconsistent or newer best; refusing unsafe restore')
    return actual


def checkpoint(model, optimizer, config, evidence, completed, history, best, commit, runtime):
    return dict(version=VERSION,config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),
        processor=DataProcessor(config['scene']).state_dict(),manifest=evidence,completed_updates=completed,
        history=history,best_candidate=best,commit=commit,runtime=runtime,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        python_rng=random.getstate())


def restore(state,model,optimizer,config):
    if state['config']!=config: raise ValueError('Cannot change model/H/protocol during resume')
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng'])
    if state['cuda_rng'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])


def remaining_updates(completed):
    if type(completed) is not int or not 0 <= completed <= 400:
        raise ValueError('Invalid durable update counter')
    return range(completed+1,401)


def attribute_metrics(prediction,batch):
    keep=batch['inputs']['gs_mask'][0];initial=batch['inputs']['gs_attr'][0,keep]
    scales=(prediction['log_scales'][0,:,keep]-initial[None,:,:3]).detach()
    opacity=(prediction['opacity_logits'][0,:,keep]-initial[None,:,7:8]).detach()
    def quantiles(v):return torch.quantile(v.flatten().double(),v.new_tensor([0,.05,.5,.95,1.],dtype=torch.float64)).tolist()
    return dict(scale_ratio=[quantiles(v.exp()) for v in scales],opacity_logit_delta=[quantiles(v) for v in opacity],
                scale_regularization_by_horizon=scales.square().mean((-1,-2)).tolist())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True,choices=tuple(WINDOWS));p.add_argument('--variant',required=True,choices=tuple(VARIANTS))
    p.add_argument('--resume',action='store_true');p.add_argument('--export-only',action='store_true')
    args=p.parse_args(argv)
    if not torch.cuda.is_available():raise RuntimeError('P3 fits require a Slurm CUDA allocation')
    config=training_config(args.scene,args.variant)
    raws,ds,evidence=load_raws(config)
    run=PROJECT/'runs'/args.scene/f'p3-two-frame-h10-v0-{args.variant}'
    torch.manual_seed(42);random.seed(42)
    model=construct(config).cuda();config['initial_model_sha256']=digest(model.state_dict())
    shared={k:v for k,v in model.state_dict().items() if not k.startswith(('temporal.','refiner.'))}
    config['shared_initial_sha256']=digest(shared)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    processor=DataProcessor(args.scene);batches=[processor(to_device(raw,'cuda')) for raw in raws]
    renderer=TargetCache(make_renderer(config));neighbors=[initial_knn(b['inputs']['gs_xyz'],b['inputs']['gs_mask'],16) for b in batches]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
    history=[];best=None;start=0;previous_seconds=0.
    if args.resume or args.export_only:
        state=read_state(run/'last.pt',config)
        if state['manifest']!=evidence:raise ValueError('Resume data changed')
        best=recover_best(run,state);restore(state,model,optimizer,config)
        start=state['completed_updates'];history=state['history'];previous_seconds=state['runtime']['seconds']
    else:
        run.mkdir(parents=True,exist_ok=False)
        atomic_json(run/'config.json',dict(training=config,manifest=evidence,commit=commit,command=__import__('sys').argv))
    if args.export_only and start!=400:raise ValueError('Export requires completed 400 updates')
    torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
    runtime=lambda:dict(seconds=previous_seconds+time.perf_counter()-began,
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    def save(n):return checkpoint(model,optimizer,config,evidence,n,history,best,commit,runtime())
    def evaluate(n,last_loss=None,grad_norm=None):
        nonlocal best
        metrics=evaluate_fixed({'model':model},batches,processor,renderer,config,neighbors)
        with torch.no_grad():
            for raw,batch in zip(raws,batches):
                key=f"{int(raw['meta']['episode_id'][0])}:{int(raw['meta']['frame_ids'][0,0])}"
                metrics['model']['windows'][key]['attributes']=attribute_metrics(model(batch['inputs']),batch)
        row=candidate(metrics,None,n);improved=better(row,best,None)
        if improved:best=row
        history.append(dict(update=n,window_updates=[n//2,n//2],metrics=metrics,candidate=row,
                            last_train_loss=last_loss,grad_norm_before_clip=grad_norm))
        state=save(n)
        atomic_checkpoint(run/f'step_{n:06d}.pt',state)
        if improved:atomic_checkpoint(run/'lowest_cd.pt',state)
        atomic_checkpoint(run/'last.pt',state)
        atomic_json(run/'metrics.json',history)
        print(json.dumps(dict(scene=args.scene,variant=args.variant,update=n,cd_mm=row['rank'][0])),flush=True)
    if start and start%20==0 and (not history or history[-1]['update']!=start):evaluate(start)
    from .losses import objective
    if start==0:atomic_checkpoint(run/'last.pt',save(0))
    for update in remaining_updates(start):
        batch=batches[(update-1)%2];model.train();optimizer.zero_grad(set_to_none=True)
        loss,values=objective(model(batch['inputs']),batch,processor,renderer,0.,config['chamfer_chunk'],**WEIGHTS)
        if not values['render_time_available'].all():raise ValueError('Missing supervised time')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        atomic_checkpoint(run/'last.pt',save(update))
        if update%20==0:evaluate(update,float(loss.detach()),float(norm))
    atomic_checkpoint(run/'last_400.pt',save(400))
    from .visualize import export_gaussians
    summary=dict(scene=args.scene,variant=args.variant,completed_updates=400,commit=commit,
                 checkpoints={},runtime=runtime(),manifest=evidence)
    for tag in ('lowest_cd','last_400'):
        state=read_state(run/(tag+'.pt'),config);model.load_state_dict(state['model']);model.eval()
        metrics=evaluate_fixed({'model':model},batches,processor,renderer,config,neighbors)
        summary['checkpoints'][tag]=dict(update=state['completed_updates'],metrics=metrics)
        with torch.no_grad():
            for (ep,start_frame),raw,batch in zip(WINDOWS[args.scene],raws,batches):
                directory=run/'visualizations'/tag/f'episode-{ep}-start-{start_frame}'
                if directory.exists():
                    if not (directory/'complete.json').is_file():raise ValueError(f'Incomplete export requires manual preservation: {directory}')
                    exported=json.loads((directory/'complete.json').read_text())
                    if exported.get('checkpoint_sha256') != sha(run/(tag+'.pt')) or exported.get('horizon') != 10:
                        raise ValueError(f'Export/checkpoint mismatch; preserve existing output: {directory}')
                    continue
                export_gaussians(SimpleNamespace(output=str(directory),checkpoint=str(run/(tag+'.pt')),
                    episode=ep,start=start_frame,split='train'),config,raw,batch,model(batch['inputs']),processor)
                # Same source reader; historical geometry is diagnostic, never a target query.
                import numpy as np
                previous=read_gs(ds.episodes[ep],start_frame-1)
                np.savez_compressed(directory/'history.npz',xyz=previous['xyz'],
                    attr=np.concatenate([previous[k] for k in GS_ATTR_FIELDS],-1),frame_id=start_frame-1,
                    timestamp_us=previous['timestamp_us'])
                atomic_json(directory/'complete.json',dict(checkpoint_sha256=sha(run/(tag+'.pt')),horizon=10))
    summary['runtime']=runtime();atomic_json(run/'fit_summary.json',summary);atomic_json(run/'runtime.json',summary['runtime'])
    return 0

if __name__=='__main__':raise SystemExit(main())
