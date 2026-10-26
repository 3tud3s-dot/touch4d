"""Real-data CUDA gates for the frozen P3 four-arm experiment. No training run."""
import copy
import gc
import json
from pathlib import Path
import subprocess
import time
import types
import sys
from io import BytesIO
import torch
from .p3_fit import (AUDIT, PROJECT, VARIANTS, training_config, construct, load_raws,
                     checkpoint, restore, sha, current_only)
from .train import MODEL_KEYS, to_device, atomic_json, make_renderer, read_checkpoint
from .fixed_fit import TargetCache, WEIGHTS, digest
from .model import PositionPredictor
from .losses import objective, nearest_distances
from ..data_process import DataProcessor


def stats(g):
    if g is None:return dict(connected=False,finite=False,nonzero=False,l2=None)
    v=g.detach().double()
    return dict(connected=True,finite=bool(torch.isfinite(v).all()),nonzero=bool(v.abs().max()>0),l2=float(v.norm()))


def terminal_cd(pred,batch):
    x=pred['xyz'][0,-1,batch['inputs']['gs_mask'][0]]
    y=batch['targets']['target_xyz'][-1][0,batch['targets']['target_mask'][-1][0]]
    return (nearest_distances(x,y).mean()+nearest_distances(y,x).mean())*.5


def old_module(path,label):
    ns=types.ModuleType(label);ns.__package__='scripts.training'
    sys.modules[label]=ns
    code=subprocess.check_output(['git','show','fdd4013:'+path],cwd=PROJECT,text=True)
    exec(compile(code,path,'exec'),ns.__dict__)
    return ns


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA allocation required')
    torch.set_num_threads(4)
    result=dict(version='p3-acceptance-v0',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        device=torch.cuda.get_device_name(),passed=False,initialization=[],cuda=[],legacy=[],preserved=[])
    frozen=json.loads((AUDIT/'stage0_audit.json').read_text())
    for item in frozen['checkpoints']:
        assert sha(PROJECT/item['path'])==item['sha256'],'Frozen checkpoint changed'
        result['preserved'].append(item)
    for scene in ('001-rope','051-cube'):
        processor=DataProcessor(scene)
        c=training_config(scene,'M3');raws,_,_=load_raws(c)
        batches=[processor(to_device(r,'cuda')) for r in raws]
        # All shared weights and initial predictions are paired exactly.
        reference=None;shared=None
        for variant in VARIANTS:
            cfg=training_config(scene,variant);torch.manual_seed(42);model=construct(cfg).cuda()
            h=digest({k:v for k,v in model.state_dict().items() if not k.startswith(('temporal.','refiner.'))})
            if shared is None:shared=h
            assert h==shared
            with torch.no_grad():out=model(batches[0]['inputs'])
            if reference is None:reference={k:v.detach().clone() for k,v in out.items()}
            for k in out:torch.testing.assert_close(reference[k],out[k],atol=0,rtol=0)
            result['initialization'].append(dict(scene=scene,variant=variant,shared_sha256=h,outputs_equal=True))
            del model,out
        del reference
        for variant in VARIANTS:
            cfg=training_config(scene,variant);torch.manual_seed(42);model=construct(cfg).cuda()
            # Discarded probe: activate zero-initialized routes to test all gradients.
            with torch.no_grad():
                model.position_head[-1].weight.normal_(std=.002)
                model.gs_structure.output.weight.normal_(std=.002)
                if variant in ('M2','M3'):model.temporal.output.weight.normal_(std=.002)
                if variant=='M3':
                    model.refiner.scale_head[-1].weight.normal_(std=.002)
                    model.refiner.opacity_head[-1].weight.normal_(std=.002)
            for wi,batch in enumerate(batches):
                x=dict(batch['inputs'])
                if variant in ('M2','M3'):
                    x['gs_prev_xyz']=x['gs_prev_xyz'].detach().clone().requires_grad_()
                    x['gs_prev_attr']=x['gs_prev_attr'].detach().clone().requires_grad_()
                heads=[];hook=model.position_head.register_forward_hook(lambda module,args,out:heads.append(out))
                renderer=TargetCache(make_renderer(cfg));torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
                if variant=='M0':pred=model(x);states=None
                else:pred,states=model(x,return_states=True)
                hook.remove();assert len(heads)==10
                loss,parts=objective(pred,batch,processor,renderer,0.,512,**WEIGHTS)
                terminal=terminal_cd(pred,batch)
                final_grad=[stats(g) for g in torch.autograd.grad(terminal,heads,retain_graph=True,allow_unused=True)]
                assert all(g['connected'] and g['finite'] for g in final_grad)
                all_grad=[stats(g) for g in torch.autograd.grad(loss,heads,retain_graph=True)]
                assert all(g['finite'] and g['nonzero'] for g in all_grad)
                state_grad=[];history_grad=[];render_grad={}
                if states is not None:
                    state_grad=[stats(g) for g in torch.autograd.grad(terminal,[s['xyz'] for s in states[:-1]],retain_graph=True)]
                    assert all(g['finite'] and g['nonzero'] for g in state_grad)
                if variant in ('M2','M3'):
                    history_grad=[stats(g) for g in torch.autograd.grad(loss,(x['gs_prev_xyz'],x['gs_prev_attr']),retain_graph=True)]
                    assert all(g['finite'] and g['nonzero'] for g in history_grad)
                params=[model.position_head[-1].weight]
                if variant=='M3':params += [model.refiner.scale_head[-1].weight,model.refiner.opacity_head[-1].weight]
                for component in ('rgb','alpha'):
                    render_grad[component]=[stats(g) for g in torch.autograd.grad(parts[component].mean(),params,retain_graph=True)]
                    assert all(g['finite'] and g['nonzero'] for g in render_grad[component])
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                torch.cuda.synchronize()
                row=dict(scene=scene,variant=variant,window=wi,frames=batch['meta']['frame_ids'][0].tolist(),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,seconds=time.perf_counter()-began,
                    final_cd_to_heads=final_grad,total_to_heads=all_grad,state_chain=state_grad,
                    historical_gradients=history_grad,render_gradients=render_grad,loss=float(loss.detach()))
                with torch.no_grad():
                    changed=dict(x)
                    for key in ('action_state','controller_points','tactile'):
                        changed[key]=x[key].clone();changed[key][:,5:]+=.2
                    changed['target_xyz']=torch.full((1,2,3),float('nan'),device='cuda')
                    altered=model(changed)
                    for k in pred:torch.testing.assert_close(pred[k][:,:5],altered[k][:,:5],atol=0,rtol=0)
                    for k,sl in (('quaternions',slice(3,7)),('sh_dc',slice(8,11))):
                        torch.testing.assert_close(pred[k],x['gs_attr'][...,sl][:,None].expand_as(pred[k]),atol=0,rtol=0)
                    if variant=='M3':
                        assert float((pred['log_scales']-x['gs_attr'][:,None,:,:3]).abs().max())<=.262365
                        assert float((pred['opacity_logits']-x['gs_attr'][:,None,:,7:8]).abs().max())<=.500001
                row.update(causal_prefix_exact=True,finite=True,attributes_valid=True)
                result['cuda'].append(row)
                model.zero_grad(set_to_none=True)
                del pred,parts,loss,terminal,heads,renderer,altered,states,x,changed
                gc.collect();torch.cuda.empty_cache()
            # Actual serialization, optimizer state, and identical next update.
            opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
            def simple_step(m,o):
                o.zero_grad(set_to_none=True);p=m(batches[0]['inputs'])
                v=p['xyz'].square().mean()+p['log_scales'].square().mean()+p['opacity_logits'].square().mean()
                v.backward();o.step()
            simple_step(model,opt)
            stream=BytesIO();torch.save(checkpoint(model,opt,cfg,[],1,[],None,result['commit'],dict(seconds=0.)),stream)
            stream.seek(0);state=torch.load(stream,map_location='cpu',weights_only=False)
            restored=construct(cfg).cuda();ropt=torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=1e-4)
            restore(state,restored,ropt,cfg);simple_step(model,opt);simple_step(restored,ropt)
            assert digest(model.state_dict())==digest(restored.state_dict())
            assert digest(opt.state_dict())==digest(ropt.state_dict())
            for row in result['cuda']:
                if row['scene']==scene and row['variant']==variant:row['checkpoint_next_update_exact']=True
            del model,opt,restored,ropt,state,stream;gc.collect();torch.cuda.empty_cache()
        del batches,raws;gc.collect();torch.cuda.empty_cache()
    # Compare real old H5/H15 frozen P2 against the exact baseline source.
    old=old_module('scripts/training/model.py','scripts.training._p3_legacy_model')
    old_structure=old_module('scripts/training/gs_structure.py','scripts.training._p3_legacy_structure')
    old.StructuralGSBlock=old_structure.StructuralGSBlock
    from .fixed_fit import fixed_batches
    for scene in ('001-rope','051-cube'):
        for h,run in ((5,'structural-gs-h15-v1-p2-h5'),(15,'structural-gs-h15-v1-p2-h15')):
            path=PROJECT/'runs'/scene/run/'lowest_cd.pt';state=read_checkpoint(path);cfg=state['config']
            raw=fixed_batches(cfg)[0];processor=DataProcessor(scene);batch=processor(to_device(raw,'cuda'))
            a=old.PositionPredictor(**{k:cfg[k] for k in MODEL_KEYS}).cuda()
            b=PositionPredictor(**{k:cfg[k] for k in MODEL_KEYS}).cuda()
            a.load_state_dict(state['model']);b.load_state_dict(state['model']);renderer=TargetCache(make_renderer(cfg))
            with torch.no_grad():
                pa,pb=a(batch['inputs']),b(batch['inputs'])
                for k in pa:torch.testing.assert_close(pa[k],pb[k],atol=0,rtol=0)
                la,ma=objective(pa,batch,processor,renderer,0.,512,**WEIGHTS)
                lb,mb=objective(pb,batch,processor,renderer,0.,512,**WEIGHTS)
                torch.testing.assert_close(la,lb,atol=0,rtol=0)
                for k in ma:torch.testing.assert_close(ma[k],mb[k],atol=0,rtol=0)
            result['legacy'].append(dict(scene=scene,horizon=h,checkpoint=str(path),outputs_and_losses_exact=True))
            del a,b,pa,pb,renderer,batch,raw,state;gc.collect();torch.cuda.empty_cache()
    result['passed']=True;atomic_json(AUDIT/'acceptance.json',result)
    print(json.dumps(dict(passed=True,cuda_probes=len(result['cuda']),legacy=len(result['legacy']))),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
