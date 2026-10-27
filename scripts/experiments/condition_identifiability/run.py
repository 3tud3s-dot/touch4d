"""One smoke gate, then two bounded 400-update experiments. Never auto-resume."""
import argparse
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
from scripts.experiments.fixed_window.fixed_fit import TargetCache
from .common import ARCH,WEIGHTS,VERSION,ToolPredictor,dataset,batch_at,synthetic_pair,state_hash,sha,evaluate,export

def fresh(device='cuda'):
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    return ToolPredictor(**ARCH).to(device)

def groups(model,before=None):
    values={}
    for name,p in model.named_parameters():
        q=values.setdefault(name.split('.')[0],[0.,0.,0.])
        base=p.detach() if before is None else before[name]
        q[0]+=float(base.double().square().sum())
        if p.grad is not None:q[1]+=float(p.grad.detach().double().square().sum())
        if before is not None:q[2]+=float((p.detach()-base).double().square().sum())
    return {k:dict(parameter_norm=v[0]**.5,gradient_norm=v[1]**.5,update_norm=v[2]**.5,
                   relative_update=(v[2]/v[0])**.5 if v[0] else None) for k,v in values.items()}

def load_checkpoint(path,model,opt,mode):
    s=torch.load(path,map_location='cpu',weights_only=False)
    if s['version']!=VERSION or s['mode']!=mode or s['config']!=model.config:raise ValueError('Checkpoint configuration conflict')
    model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer'])
    return s

def batch_pair(ds,audit,mode,processor):
    if mode=='synthetic':
        row=audit['synthetic_base'];return synthetic_pair(batch_at(ds,row['episode'],row['start'],'cuda'),processor.state_dict()['scale'])
    return [batch_at(ds,w['episode'],w['start'],'cuda') for w in audit['windows']]

def step(model,opt,b,processor,renderer,record):
    model.train();opt.zero_grad(set_to_none=True);pred=model(b['inputs'])
    loss,v=objective(pred,b,processor,renderer,0.,512,**WEIGHTS)
    if not v['render_time_available'].all():raise ValueError('Missing valid target horizon')
    loss.backward();raw=groups(model) if record else None
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    clipped=groups(model) if record else None
    before={n:p.detach().clone() for n,p in model.named_parameters()} if record else None
    opt.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite parameters')
    stats=dict(raw=raw,clipped=clipped,actual=groups(model,before)) if record else None
    values=dict(loss=float(loss.detach()),raw_global_gradient_norm=float(norm),
                **{k:float(v[k].detach().mean()) for k in ('geometry','rgb','alpha','scale_regularization')})
    return values,stats

def train(args,out,audit,source,smoke=False):
    processor=DataProcessor('001-rope');ds=dataset(args.root)
    modes=('synthetic','real') if smoke else (args.mode,)
    results={}
    for mode in modes:
        batches=batch_pair(ds,audit,mode,processor)
        model=fresh();initial=state_hash(model);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        params=[id(p) for p in model.parameters() if p.requires_grad]
        actual=[id(p) for g in opt.param_groups for p in g['params']]
        assert len(params)==len(actual) and set(params)==set(actual)
        renderers=[TargetCache(make_renderer(dict(output_mode='gs_geometry',render_settings=gs_render_settings()))) for _ in batches]
        if smoke:
            rows=[]
            for n in range(4):
                values,g=step(model,opt,batches[n%2],processor,renderers[n%2],True)
                rows.append(dict(update=n+1,case=n%2,metrics=values,gradients=g))
            with torch.no_grad():pred=model(batches[0]['inputs'])
            path=out/(mode+'-smoke.pt');atomic_checkpoint(path,dict(version=VERSION,mode=mode,config=model.config,model=model.state_dict(),optimizer=opt.state_dict()))
            load_checkpoint(path,model,opt,mode)
            with torch.no_grad():restored=model(batches[0]['inputs'])
            assert all(torch.equal(pred[k],restored[k]) for k in pred)
            results[mode]=dict(updates=rows,restore_exact=True,initial_state_sha256=initial)
        else:
            smoke_record=json.loads(Path(args.smoke).read_text())
            if smoke_record.get('status')!='complete' or smoke_record['source_sha']!=source:raise ValueError('Smoke must pass on identical source')
            config=dict(source_sha=source,mode=mode,model=model.config,weights=WEIGHTS,seed=42,updates=400,
                per_case_updates=200,optimizer=dict(type='AdamW',lr=1e-4,weight_decay=1e-4,clip_norm=1.),
                initial_state_sha256=initial,manifest_sha256=sha(args.audit),processor=processor.state_dict(),
                window_manifest=audit,slurm_job_id=os.environ['SLURM_JOB_ID'])
            atomic_json(out/'config.json',config)
            history=[];gradients=[];evaluations={}
            for n in range(401):
                if n in (0,100,200,400):
                    measures,predictions=evaluate(model,batches,processor,renderers)
                    evidence=[]
                    if n in (0,400):
                        for i,(pred,b,renderer) in enumerate(zip(predictions,batches,renderers)):
                            evidence.append(export(out/f'u{n:03d}-case{i}',pred,b,processor,renderer))
                    evaluations[str(n)]=dict(cases=measures,evidence=evidence)
                    atomic_json(out/'evaluations.json',evaluations)
                    atomic_checkpoint(out/f'checkpoint_{n}.pt',dict(version=VERSION,mode=mode,config=model.config,
                        model=model.state_dict(),optimizer=opt.state_dict(),completed_updates=n,source_sha=source))
                    print(json.dumps(dict(mode=mode,update=n,cd_mm=[r['normal']['mean_cd_mm'] for r in measures])),flush=True)
                    del predictions
                if n==400:break
                values,g=step(model,opt,batches[n%2],processor,renderers[n%2],n in (0,1,2,3,99,199,399))
                history.append(dict(update=n+1,case=n%2,**values))
                if g:gradients.append(dict(update=n+1,case=n%2,**g))
                if (n+1)%20==0:atomic_json(out/'updates.json',history)
            atomic_json(out/'gradients.json',gradients)
            with torch.no_grad():pred=model(batches[0]['inputs'])
            load_checkpoint(out/'checkpoint_400.pt',model,opt,mode)
            with torch.no_grad():restored=model(batches[0]['inputs'])
            exact=all(torch.equal(pred[k],restored[k]) for k in pred)
            if not exact:raise AssertionError('Checkpoint prediction mismatch')
            results[mode]=dict(successful_updates=400,per_case_updates=[200,200],restore_exact=exact,
                checkpoint_sha256=sha(out/'checkpoint_400.pt'),initial_state_sha256=initial)
        del model,opt,batches,renderers;torch.cuda.empty_cache()
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('smoke','synthetic','real'),required=True)
    for name in ('root','audit','output','expected-sha'):p.add_argument('--'+name,required=True)
    p.add_argument('--smoke');a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    job=os.environ.get('SLURM_JOB_ID')
    if not job or job=='30903' or not torch.cuda.is_available():raise RuntimeError('Requires separate authorized Slurm GPU')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source!=a.expected_sha or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Source drift')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(4)
    audit=json.loads(Path(a.audit).read_text());status=dict(source_sha=source,job_id=job,gpu=torch.cuda.get_device_name(),mode=a.mode)
    try:
        for path,digest in audit['source_file_sha256'].items():
            if sha(path)!=digest:raise ValueError('Frozen data changed: '+path)
        results=train(a,out,audit,source,a.mode=='smoke')
        status.update(status='complete',results=results,elapsed_seconds=time.monotonic()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception:
        status.update(status='failed',traceback=traceback.format_exc());raise
    finally:atomic_json(out/'status.json',status)

if __name__=='__main__':main()
