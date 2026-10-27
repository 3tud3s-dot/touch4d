"""Frozen debug: one GPU diagnosis then two paired single-window runs, never resume."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
import numpy as np
import torch
from scripts.data_process import DataProcessor
from scripts.training.train import atomic_json,atomic_checkpoint,make_renderer,gs_render_settings
from scripts.training.losses import objective
from scripts.experiments.spatial_dynamics.runner import ARCH,WEIGHTS,paired_model,sha,tensor_sha
from scripts.experiments.spatial_dynamics.visualize import load_research,condition_prefix
from .common import CASES,get_batch,evaluate_conditions,export_evidence

VERSION='touch4d-debug-audit-v1'
BASE='400b4c389325efe18d4d9a54eb5412e722ab22cc'
OLD_ROOT='/data1/userdata/tcweng/experiments/touch4d/p3-p4-spatial-dynamics-v1/runs'
CHECKPOINTS={f'{mode}-{kind}':f'{OLD_ROOT}/p3p4-{mode}-D0-F0-h5-u1200-s42-3556608/{kind}.pt'
             for mode in ('S0','S2-C1') for kind in ('last','best')}


def weights(mode):
    if mode not in ('E1','E2'):raise ValueError('Only preregistered E1/E2')
    return dict(WEIGHTS) if mode=='E2' else dict(chamfer_weight=1.,rgb_weight=0.,alpha_weight=0.,scale_weight=0.)


def parameter_stats(model,before=None):
    groups={}
    for name,p in model.named_parameters():
        g=groups.setdefault(name.split('.')[0],dict(parameter_sq=0.,gradient_sq=0.,update_sq=0.,parameters=0))
        reference=before[name] if before is not None else p.detach()
        g['parameter_sq']+=float(reference.double().square().sum());g['parameters']+=p.numel()
        if p.grad is not None:g['gradient_sq']+=float(p.grad.detach().double().square().sum())
        if before is not None:g['update_sq']+=float((p.detach()-before[name]).double().square().sum())
    return {name:dict(parameter_norm=g['parameter_sq']**.5,gradient_norm=g['gradient_sq']**.5,
            update_norm=g['update_sq']**.5,relative_update=(g['update_sq']/g['parameter_sq'])**.5 if g['parameter_sq'] else None,
            parameters=g['parameters']) for name,g in groups.items()}


def save(path,model,opt,processor,step,mode,history,source):
    atomic_checkpoint(path,dict(version=VERSION,config=model.config,model=model.state_dict(),
        optimizer=opt.state_dict(),processor=processor.state_dict(),completed_updates=step,
        weights=weights(mode),mode=mode,source_sha=source,history=history,
        window=dict(scene='001-rope',episode=2,start=73,horizon=5),seed=42))


def diagnose(args,out):
    from .gradients import audit_gradients
    from .oracle import run_oracle
    old_hash={key:sha(path) for key,path in CHECKPOINTS.items()}
    atomic_json(out/'historical_hashes_before.json',old_hash)
    records=[];prefix=[]
    for label,path in CHECKPOINTS.items():
        model,_,_,state=load_research(path,5);model=model.cuda().eval()
        processor=DataProcessor.from_state_dict(state['processor'])
        renderer=make_renderer(dict(output_mode='gs_geometry',render_settings=gs_render_settings()))
        for split,ep,start in CASES:
            batch=get_batch(args.root,processor,ep,start)
            metrics,pred=evaluate_conditions(model,batch,processor,renderer)
            evidence=export_evidence(out/f'{label}-ep{ep}-s{start}',pred,batch,processor,renderer)
            records.append(dict(label=label,checkpoint=path,checkpoint_sha256=old_hash[label],
                split=split,episode=ep,start=start,completed_updates=state['completed_updates'],
                model_config=state['config'],metrics=metrics,evidence=evidence))
            del pred,batch;model.trace=[]
        if label=='S0-last':
            batch=get_batch(args.root,processor)
            audit=audit_gradients(model,batch,processor,renderer,state['optimizer'])
            atomic_json(out/'gradient_audit.json',audit)
            del batch
        del model,state;torch.cuda.empty_cache()
    atomic_json(out/'checkpoint_results.json',records)
    # Exact already-established H5 prefix check; no H15 training or loss claim.
    for label in ('S0-last','S2-C1-last'):
        full,short,warning,state=load_research(CHECKPOINTS[label],15)
        full=full.cuda();short=short.cuda();processor=DataProcessor.from_state_dict(state['processor'])
        batch=get_batch(args.root,processor,2,73,15)
        with torch.no_grad():
            a=full(batch['inputs']);b=short(condition_prefix(batch['inputs'],5))
        exact=all(torch.equal(a[k][:,:5],b[k]) for k in a)
        if not exact:raise AssertionError('Historical H5 prefix changed')
        prefix.append(dict(label=label,exact=exact,policy=warning))
        del a,b,batch,full,short,state;torch.cuda.empty_cache()
    atomic_json(out/'prefix.json',prefix)
    processor=DataProcessor('001-rope');batch=get_batch(args.root,processor)
    renderer=make_renderer(dict(output_mode='gs_geometry',render_settings=gs_render_settings()))
    oracle={}
    for full in (False,True):
        name='full' if full else 'cd'
        oracle[name]=run_oracle(batch,processor,renderer,out/f'oracle-{name}',updates=400,lr=1e-3,full_loss=full)
        for filename,label in [('prediction_400.pt','final'),('centroid_translation.pt','centroid')]:
            saved=torch.load(out/f'oracle-{name}'/filename,map_location='cuda',weights_only=False)
            oracle[name].setdefault('visuals',{})[label]=export_evidence(
                out/f'oracle-{name}-{label}',saved['prediction'],batch,processor,renderer)
            del saved
        torch.cuda.empty_cache()
    atomic_json(out/'oracle_results.json',oracle)
    after={key:sha(path) for key,path in CHECKPOINTS.items()}
    if old_hash!=after:raise RuntimeError('Historical checkpoint changed')
    atomic_json(out/'historical_hashes_after.json',after)
    return dict(checkpoint_count=len(records),prefix_exact=True,historical_hashes_unchanged=True)


def fit(args,out,source):
    if args.mode not in ('E1','E2'):raise ValueError('fit requires E1/E2')
    model,pairing=paired_model('S0',seed=42,**ARCH);model=model.cuda()
    processor=DataProcessor('001-rope');batch=get_batch(args.root,processor)
    renderer=make_renderer(dict(output_mode='gs_geometry',render_settings=gs_render_settings()))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    ids=[id(p) for p in model.parameters() if p.requires_grad]
    optimizer_ids=[id(p) for g in opt.param_groups for p in g['params']]
    if set(ids)!=set(optimizer_ids) or len(ids)!=len(optimizer_ids):raise AssertionError('Optimizer missing/duplicated parameter')
    config=dict(source_sha=source,base_sha=BASE,mode=args.mode,weights=weights(args.mode),model=model.config,
                seed=42,updates=400,optimizer=dict(type='AdamW',lr=1e-4,weight_decay=1e-4,clip_norm=1.),
                scene='001-rope',episode=2,start=73,frames=list(range(73,79)),split='train',
                pairing=pairing,slurm_job_id=os.environ['SLURM_JOB_ID'],processor=processor.state_dict(),
                all_optimizer_parameters=True,initial_state_sha256={k:tensor_sha(v) for k,v in model.state_dict().items()})
    atomic_json(out/'config.json',config)
    history=[];evaluations={};updates=[]
    for step in range(401):
        if step in (0,100,200,400):
            metrics,pred=evaluate_conditions(model,batch,processor,renderer)
            evidence=export_evidence(out/f'update-{step:03d}',pred,batch,processor,renderer)
            evaluations[str(step)]=dict(metrics=metrics,evidence=evidence)
            save(out/f'checkpoint_{step}.pt',model,opt,processor,step,args.mode,history,source)
            atomic_json(out/'evaluations.json',evaluations)
            print(json.dumps(dict(update=step,mean_cd_mm=metrics['model']['mean_cd_mm'],
                                  copy_cd_mm=metrics['copy_initial']['mean_cd_mm'])),flush=True)
            del pred;model.trace=[]
        if step==400:break
        model.train();opt.zero_grad(set_to_none=True)
        pred=model(batch['inputs'])
        loss,values=objective(pred,batch,processor,renderer,0.,512,**weights(args.mode))
        if not values['render_time_available'].all():raise ValueError('Unobserved target time')
        loss.backward()
        sample=step in (0,1,99,199,399)
        raw=parameter_stats(model) if sample else None
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        clipped=parameter_stats(model) if sample else None
        before={n:p.detach().clone() for n,p in model.named_parameters()} if sample else None
        opt.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite parameter')
        if sample:updates.append(dict(update=step+1,raw=raw,clipped=clipped,actual=parameter_stats(model,before)))
        history.append(dict(update=step+1,loss=float(loss.detach()),raw_global_gradient_norm=float(norm),
            **{k:float(values[k].detach().mean()) for k in ('geometry','rgb','alpha','scale_regularization')}))
        del pred,loss,values,before;model.trace=[]
        if (step+1)%20==0:atomic_json(out/'updates.json',history)
    atomic_json(out/'parameter_updates.json',updates)
    save(out/'last_400.pt',model,opt,processor,400,args.mode,history,source)
    saved=torch.load(out/'last_400.pt',map_location='cpu',weights_only=False)
    with torch.no_grad():before=model(batch['inputs'])
    model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
    with torch.no_grad():after=model(batch['inputs'])
    if not all(torch.equal(before[k],after[k]) for k in before):raise AssertionError('Checkpoint output changed')
    return dict(successful_updates=400,checkpoint_restore_exact=True,checkpoint_sha256=sha(out/'last_400.pt'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('diagnose','fit'));p.add_argument('--mode',choices=('E1','E2'))
    p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--expected-sha',required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ['SLURM_JOB_ID']=='30903' or not torch.cuda.is_available():
        raise RuntimeError('A new dedicated Slurm CUDA allocation is required; user job is forbidden')
    if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one allocated GPU required')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source!=a.expected_sha or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise RuntimeError('Execution worktree drift')
    # No old implementation change is permitted for this diagnostic branch.
    changed=subprocess.check_output(['git','diff','--name-only',BASE,'HEAD','--','scripts/training','scripts/dataloader',
        'scripts/data_process','scripts/experiments/spatial_dynamics'],text=True).strip()
    if changed:raise RuntimeError('Frozen implementation changed: '+changed)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    atomic_json(out/'execution.json',dict(source_sha=source,base_sha=BASE,argv=os.sys.argv,
        job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),protected_user_job='30903'))
    try:
        value=diagnose(a,out) if a.command=='diagnose' else fit(a,out,source)
        torch.cuda.synchronize()
        atomic_json(out/'status.json',dict(status='complete',source_sha=source,job_id=os.environ['SLURM_JOB_ID'],
            elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),**value))
    except BaseException:
        atomic_json(out/'status.json',dict(status='failed',source_sha=source,job_id=os.environ['SLURM_JOB_ID'],error=traceback.format_exc()))
        raise


if __name__=='__main__':main()
