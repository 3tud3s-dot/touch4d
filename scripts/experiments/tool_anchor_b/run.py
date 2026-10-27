"""CPU manifest, CUDA smoke, then one GPU sequential paired fixed400 runs."""
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
from scripts.data_process import DataProcessor
from scripts.training.train import atomic_json,atomic_checkpoint,make_renderer,gs_render_settings,copy_initial
from scripts.training.losses import objective
from scripts.experiments.fixed_window.fixed_fit import TargetCache
from .model import ARCH,AnchorBPredictor
from .conditions import ToolPredictor
from .data import WEIGHTS,dataset,batches,audit,sha,metrics,evaluate,export

VERSION='tool-anchor-b-h15-v1'

def stable_gradient_norm(value):
    """Measure tiny FP32 AR gradients in FP64; do not square them in FP32."""
    return float(value.detach().double().norm())

def gradient_probe(model,inputs,multiplier=1.):
    """No optimizer update. Scalar amplification diagnoses FP32 underflow only.

    Report norms in original loss units. A missing .grad means this probe did not reach the retained output.
    A present zero may be underflow OR a true zero derivative; a tiny nonzero
    derivative proves response, not useful long-range learning.
    """
    if not np.isfinite(multiplier) or multiplier<=0:raise ValueError('Invalid diagnostic multiplier')
    captured=[]
    def hook(module,inputs,value):value.retain_grad();captured.append(value)
    handle=model.position_head.register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    try:
        pred=model(inputs);(pred['xyz'][:,-1].square().sum()*multiplier).backward()
    finally:handle.remove()
    return dict(loss_multiplier=multiplier,connected=[v.grad is not None for v in captured],
        norms_float64=[stable_gradient_norm(v.grad)/multiplier if v.grad is not None else None for v in captured],
        norms_float32=[float(v.grad.norm())/multiplier if v.grad is not None else None for v in captured],
        nonzero_elements=[int(v.grad.count_nonzero()) if v.grad is not None else 0 for v in captured],
        finite=[bool(torch.isfinite(v.grad).all()) if v.grad is not None else None for v in captured],
        classification=['not_reached_by_probe' if v.grad is None else
            'nonfinite' if not torch.isfinite(v.grad).all() else
            'present_nonzero' if v.grad.count_nonzero() else 'present_zero_ambiguous' for v in captured])

def fresh(mode,device='cpu',config=None):
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if mode not in ('base','candidate'):raise ValueError(mode)
    model=(ToolPredictor if mode=='base' else AnchorBPredictor)(**(ARCH if config is None else config))
    model.config.update(experiment=VERSION,experiment_mode=mode)
    return model.to(device)

def digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def pairing():
    a=fresh('base');b=fresh('candidate');sa,sb=a.state_dict(),b.state_dict()
    assert set(sa).issubset(sb)
    for k,v in sa.items():assert torch.equal(v,sb[k]),k
    return dict(common_state_sha256=digest(sa),common_exact=True,
        base_parameters=sum(p.numel() for p in a.parameters()),candidate_parameters=sum(p.numel() for p in b.parameters()),
        added_parameter_names=[k for k in sb if k not in sa])

def groups(model,before=None):
    rows={}
    for name,p in model.named_parameters():
        key='.'.join(name.split('.')[:2]) if name.startswith('tool_attention.') else name.split('.')[0]
        r=rows.setdefault(key,[0.,0.,0.]);base=p.detach() if before is None else before[name]
        r[0]+=float(base.double().square().sum())
        if p.grad is not None:r[1]+=float(p.grad.detach().double().square().sum())
        if before is not None:r[2]+=float((p.detach()-base).double().square().sum())
    return {k:dict(parameter_norm=v[0]**.5,gradient_norm=v[1]**.5,update_norm=v[2]**.5,
                   relative_update=(v[2]/v[0])**.5 if v[0] else None) for k,v in rows.items()}

def payload(model,opt,mode,step,source):
    return dict(version=VERSION,mode=mode,config=model.config,model=model.state_dict(),optimizer=opt.state_dict(),
                completed_updates=step,source_sha=source,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore(path,model,opt,mode):
    s=torch.load(path,map_location='cpu',weights_only=False)
    if s['version']!=VERSION or s['mode']!=mode or s['config']!=model.config:raise ValueError('Checkpoint mode/configuration conflict')
    model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);return s

def step(model,opt,b,processor,renderer,record=False):
    model.train();opt.zero_grad(set_to_none=True);pred=model(b['inputs'])
    loss,v=objective(pred,b,processor,renderer,0.,512,**WEIGHTS)
    if not torch.isfinite(loss) or not v['render_time_available'].all():raise FloatingPointError('Invalid loss/supervision')
    loss.backward();raw=groups(model) if record else None
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    clipped=groups(model) if record else None
    before={n:p.detach().clone() for n,p in model.named_parameters()} if record else None
    opt.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite updated parameters')
    g=dict(raw=raw,clipped=clipped,actual=groups(model,before)) if record else None
    row=dict(loss=float(loss.detach()),raw_global_gradient_norm=float(norm),
        **{k:float(v[k].detach().mean()) for k in ('geometry','rgb','alpha','scale_regularization')})
    return row,g

def smoke(out,bs,processor,renderers,source):
    results={}
    for mode in ('base','candidate'):
        torch.cuda.reset_peak_memory_stats();t=time.monotonic();model=fresh(mode,'cuda');opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        updates=[]
        for n in range(3):
            values,grad=step(model,opt,bs[n%2],processor,renderers[n%2],True)
            updates.append(dict(update=n+1,**values,gradients=grad))
            atomic_json(out/f'{mode}-updates.json',updates)
        # Preserve actual unamplified gradients, including numerical zeros. The
        # second fixed diagnostic changes no model weight or optimizer state.
        state_before=digest(model.state_dict());optimizer_steps_before=[int(v['step']) for v in opt.state.values()]
        unscaled=gradient_probe(model,bs[0]['inputs'])
        atomic_json(out/f'{mode}-gradient-chain.json',dict(unscaled=unscaled))
        amplified=gradient_probe(model,bs[0]['inputs'],1e16)
        if digest(model.state_dict())!=state_before or optimizer_steps_before!=[int(v['step']) for v in opt.state.values()]:
            raise AssertionError('Diagnostic changed training state')
        atomic_json(out/f'{mode}-gradient-chain.json',dict(unscaled=unscaled,amplified=amplified,model_unchanged=True,optimizer_steps_unchanged=True,
            interpretation='Amplified derivative is a connectivity diagnostic only; unscaled zeros and tiny responses remain learning limitations.'))
        cross=amplified['norms_float64']
        if len(cross)!=15 or not all(unscaled['connected']) or not all(amplified['connected']):raise AssertionError('Disconnected AR gradient')
        if not all(unscaled['finite']) or not all(amplified['finite']) or not all(v>0 and np.isfinite(v) for v in cross):raise AssertionError('AR derivative not verified even with fixed diagnostic amplification')
        if mode=='candidate':
            for name in ('tool_attention.bias','tool_attention.value','tool_attention.motion_fusion','gs_encoder','position_head'):
                value=sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(name) and p.grad is not None)
                if not value>0:raise AssertionError('Missing branch gradient: '+name)
        with torch.no_grad():saved=model(bs[0]['inputs'])
        path=out/f'{mode}.pt';atomic_checkpoint(path,payload(model,opt,mode,3,source))
        restored=fresh(mode,'cuda');restored_opt=torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=1e-4)
        checkpoint=restore(path,restored,restored_opt,mode)
        with torch.no_grad():loaded=restored(bs[0]['inputs'])
        assert all(torch.equal(saved[k],loaded[k]) for k in saved)
        # Paired fourth smoke update verifies optimizer moments/step restoration.
        # Neither branch is reused by formal training, which starts from seed42.
        next_live,_=step(model,opt,bs[1],processor,renderers[1])
        next_restored,_=step(restored,restored_opt,bs[1],processor,renderers[1])
        assert next_live==next_restored
        assert digest(model.state_dict())==digest(restored.state_dict())
        for key,state in opt.state_dict()['state'].items():
            for name,value in state.items():
                other=restored_opt.state_dict()['state'][key][name]
                assert torch.equal(value,other) if torch.is_tensor(value) else value==other
        results[mode]=dict(updates=updates,cross_step_gradient_probe=dict(unscaled=unscaled,amplified=amplified),restore_exact=True,
            restore_next_update_exact=True,checkpoint_sha256=sha(path),checkpoint_completed_updates=checkpoint['completed_updates'],
            next_update=next_live,model_unchanged_by_probe=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-t)
        atomic_json(out/f'{mode}-smoke.json',results[mode])
        del model,opt,restored,restored_opt,saved,loaded,checkpoint;torch.cuda.empty_cache()
    return results

def train(mode,out,bs,processor,renderers,source,manifest):
    out.mkdir(exist_ok=False);model=fresh(mode,'cuda');opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    assert {id(p) for p in model.parameters()}=={id(p) for g in opt.param_groups for p in g['params']}
    atomic_json(out/'config.json',dict(mode=mode,model=model.config,source_sha=source,seed=42,updates=400,
        optimizer=dict(type='AdamW',lr=1e-4,weight_decay=1e-4,clip_norm=1.),weights=WEIGHTS,
        manifest=manifest,initial_state_sha256=digest(model.state_dict()),renderer=gs_render_settings()))
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();history=[];evaluations={};gradients=[];best=float('inf');best_step=None
    for n in range(401):
        if n%20==0:
            rows,pred=evaluate(model,bs,processor,renderers);mean=sum(r['mean_cd_mm'] for r in rows)/2
            evaluations[str(n)]=dict(cases=rows,mean_cd_mm=mean)
            atomic_json(out/'evaluations.json',evaluations)
            if mean<best:
                best,best_step=mean,n;atomic_checkpoint(out/'best.pt',payload(model,opt,mode,n,source))
            if n==400:atomic_checkpoint(out/'last_400.pt',payload(model,opt,mode,n,source))
            print(json.dumps(dict(mode=mode,update=n,cd_mm=mean,best_update=best_step)),flush=True)
            del pred
        if n==400:break
        row,g=step(model,opt,bs[n%2],processor,renderers[n%2],n in (0,1,2,3) or (n+1)%20==0)
        history.append(dict(update=n+1,case=n%2,**row))
        if g:gradients.append(dict(update=n+1,case=n%2,**g))
        if (n+1)%20==0:atomic_json(out/'updates.json',history);atomic_json(out/'gradients.json',gradients)
    fit_seconds=time.monotonic()-start;peak_a=torch.cuda.max_memory_allocated();peak_r=torch.cuda.max_memory_reserved()
    with torch.no_grad():expected=model(bs[0]['inputs'])
    restore(out/'last_400.pt',model,opt,mode)
    with torch.no_grad():actual=model(bs[0]['inputs'])
    assert all(torch.equal(expected[k],actual[k]) for k in expected)
    final={}
    for label in ('best','last_400'):
        ck=restore(out/f'{label}.pt',model,opt,mode);rows,pred=evaluate(model,bs,processor,renderers,True)
        for i,(p,b,r) in enumerate(zip(pred,bs,renderers)):
            export(out/f'{label}-case{i}',p,b,processor,r,reference=mode=='base' and label=='best')
        final[label]=dict(update=ck['completed_updates'],cases=rows,mean_cd_mm=sum(r['mean_cd_mm'] for r in rows)/2,
            checkpoint=str(out/f'{label}.pt'),sha256=sha(out/f'{label}.pt'))
    atomic_json(out/'final.json',final)
    atomic_json(out/'status.json',dict(status='complete',source_sha=source,successful_updates=400,per_case_updates=[200,200],
        best_update=best_step,restore_exact=True,fit_and_evaluation_seconds=fit_seconds,total_seconds=time.monotonic()-start,
        peak_allocated_bytes=peak_a,peak_reserved_bytes=peak_r))
    del model,opt;torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('audit','smoke','paired'),required=True)
    for name in ('root','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--manifest');parser.add_argument('--expected-sha');parser.add_argument('--smoke');args=parser.parse_args()
    torch.set_num_threads(4)
    if args.stage=='audit':print(json.dumps(audit(args.root,args.output)));return
    if not os.environ.get('SLURM_JOB_ID') or os.environ['SLURM_JOB_ID']=='30903' or not torch.cuda.is_available():raise RuntimeError('Separate Slurm GPU required')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source!=args.expected_sha or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Source drift')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);status=dict(source_sha=source,job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name())
    try:
        manifest=json.loads(Path(args.manifest).read_text())
        for p,d in manifest['source_file_sha256'].items():
            if sha(p)!=d:raise ValueError('Frozen source data changed: '+p)
        ds=dataset(args.root);bs=batches(ds,'cuda');processor=DataProcessor('001-rope')
        for b,row in zip(bs,manifest['windows']):
            if hashlib.sha256(b['targets']['target_depth_valid'].cpu().numpy().tobytes()).hexdigest()!=row['target_depth_valid_sha256']:raise ValueError('Depth supervision changed')
        renderers=[TargetCache(make_renderer(dict(output_mode='gs_geometry',render_settings=gs_render_settings()))) for _ in bs]
        atomic_json(out/'pairing.json',pairing())
        if args.stage=='smoke':status['results']=smoke(out,bs,processor,renderers,source)
        else:
            check=json.loads(Path(args.smoke).read_text())
            if check.get('status')!='complete' or check['source_sha']!=source:raise ValueError('Identical-SHA smoke must pass')
            copies=[metrics(copy_initial(b['inputs'],'gs_geometry'),b,processor,r) for b,r in zip(bs,renderers)]
            atomic_json(out/'copy.json',copies)
            for mode in ('base','candidate'):train(mode,out/mode,bs,processor,renderers,source,manifest)
        status['status']='complete'
    except Exception:
        status.update(status='failed',traceback=traceback.format_exc());raise
    finally:atomic_json(out/'status.json',status)

if __name__=='__main__':main()
