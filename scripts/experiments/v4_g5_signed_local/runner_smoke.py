"""Smoke-only integration of the unchanged formal runner. Never a formal Epoch."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.training.train import atomic_json, to_device
from scripts.training.v4_runtime import prepare_batch, v4_objective
from scripts.experiments.v4_g4_global_motion.smoke import same_tree
from . import formal_checkpoint as formal
from .protocol import configuration, preflight, paired_initialization, fresh_model
from .train import perform_epoch
from .monitor import EpochMonitor, ObservedPredictor
from .checkpoint import save_checkpoint, load_checkpoint
from .smoke import source_preflight, AuditedRenderer, metric_numbers
from .diagnostics import finite_tree, parameter_evidence, state_hash, sha256

BASE = '03fce059c6d6cf2f547c06cdff8434c8af2ffce6'
WINDOWS = (('train', 7, 105), ('val', 3, 135), ('val', 6, 110))
FROZEN = ('model.py','motion.py','propagation.py','train.py','monitor.py',
          'formal_checkpoint.py','protocol.py','evaluate.py','train.sbatch')


def guard_source(expected):
    import subprocess
    source, graphs = source_preflight(expected)
    for name in FROZEN:
        path = 'scripts/experiments/v4_g5_signed_local/' + name
        if subprocess.check_output(['git','show',BASE+':'+path]) != subprocess.check_output(['git','show','HEAD:'+path]):
            raise ValueError('Frozen formal/model implementation changed: '+path)
    return source, graphs


def smoke_output(path):
    out = Path(path)
    if not out.is_absolute() or out.is_symlink(): raise ValueError('Absolute non-symlink output required')
    out = out.resolve()
    if out.parent.name != 'v4-g5-runner-integration-smoke-v1' or out.name == 'run-001':
        raise ValueError('Only separate runner Smoke namespace allowed; never formal run-001')
    if out.exists(): raise FileExistsError(out)
    return out


def load_raw(raw_loaders, processor):
    result = {}; manifest = {}; model = fresh_model('direct')
    for split, ep, start in WINDOWS:
        ds = raw_loaders[split].dataset
        raw = collate_fn([ds[ds.windows.index((ep,start))]])
        batch = prepare_batch(raw,processor,model)
        ids = raw['meta']['frame_ids'].tolist()
        if ids != [list(range(start,start+16))]: raise AssertionError('Wrong H15 frames')
        x,y = batch['inputs'],batch['targets']
        if x['condition_frame_ids'].tolist() != [list(range(start,start+15))]: raise AssertionError('Action alignment')
        if x['operation_hands'].tolist() != [[ep>=5,True]]: raise AssertionError('Wrong Hand Mask')
        if tuple(y['target_depth'].shape) != (1,15,3,720,1280): raise AssertionError('Original 3-camera supervision required')
        if not all(v.any() for v in y['target_mask']) or not y['target_depth_valid'].all():
            # Per-pixel invalid depth is legal; require supervision in every target/view.
            if not all(v.any() for v in y['target_mask']) or not y['target_depth_valid'].flatten(3).any(-1).all():
                raise AssertionError('Missing target/view supervision')
        key = f'ep{ep}_start{start}'; result[key] = raw
        manifest[key] = dict(split=split,frames=ids[0],conditions=list(range(start,start+15)),
            targets=list(range(start+1,start+16)),operation_hands=x['operation_hands'].tolist(),
            gs_count=int(x['gs_mask'].sum()),input_hash=state_hash({k:v for k,v in x.items() if isinstance(v,torch.Tensor)}))
        del batch
    return result, manifest


def equivalence(model,batch,processor,renderer):
    """No optimizer; use active (nonzero) local branch after the two real steps."""
    before = state_hash(model.state_dict()); outputs=[]; losses=[]; gradients=[]
    for observed in (False,True):
        model.zero_grad(set_to_none=True)
        runner = ObservedPredictor(model,EpochMonitor(processor.state_dict()['scale'],hasattr(model,'propagation'))) if observed else model
        runner.train(); pred = runner(batch['inputs']); loss,_ = v4_objective(pred,batch,processor,renderer)
        loss.backward(); finite_tree((pred,loss))
        outputs.append({k:v.detach().cpu() for k,v in pred.items()}); losses.append(float(loss.detach()))
        gradients.append({n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in model.named_parameters()})
        finite_tree(gradients[-1]); del pred,loss
    prediction_exact = same_tree(outputs[0],outputs[1]); gradient_exact = same_tree(gradients[0],gradients[1])
    max_gradient_error = 0.
    for name,a in gradients[0].items():
        b=gradients[1][name]
        if (a is None)!=(b is None): raise AssertionError('Observer gradient connectivity changed')
        if a is not None:
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)
            max_gradient_error=max(max_gradient_error,float((a-b).abs().max()))
    for name in outputs[0]: torch.testing.assert_close(outputs[0][name],outputs[1][name],rtol=1e-6,atol=1e-7)
    if abs(losses[0]-losses[1])>1e-7: raise AssertionError('Observer changed Loss')
    if state_hash(model.state_dict())!=before: raise AssertionError('Equivalence check changed weights')
    model.zero_grad(set_to_none=True)
    return dict(prediction_bit_exact=prediction_exact,gradient_bit_exact=gradient_exact,losses=losses,
                max_gradient_absolute_error=max_gradient_error,passed=True,
                gradient_close=True,gradient_tolerance=dict(rtol=1e-5,atol=1e-7),
                optimizer_updates=0,active_local_projection=bool(model.direct_motion.mapping[-1].weight.count_nonzero()))


class UpdateAudit:
    """Observe the real runner's clip and optimizer; never replace their math."""
    def __init__(self,model,opt): self.model=model;self.opt=opt;self.records=[];self.clips=0
    def clip(self,original,parameters,max_norm,*args,**kwargs):
        if max_norm!=1.: raise AssertionError('Clip norm drift')
        self.raw={n:None if p.grad is None else p.grad.detach().clone() for n,p in self.model.named_parameters()}
        finite_tree(self.raw); norm=original(parameters,max_norm,*args,**kwargs)
        self.norm=float(norm); self.clips+=1
        return norm
    def before(self,opt,args,kwargs):
        if len(self.records)>=2: raise AssertionError('Smoke exceeded 2 optimizer updates')
        self.parameters={n:p.detach().clone() for n,p in self.model.named_parameters()}
    def after(self,opt,args,kwargs):
        finite_tree((self.model.state_dict(),opt.state_dict()))
        self.records.append(dict(update=len(self.records)+1,raw_gradient_norm=self.norm,clipped=self.norm>1,
            groups=parameter_evidence(self.model,self.parameters,self.raw)))
        del self.parameters,self.raw

    def objective(self,original,*args,**kwargs):
        result=original(*args,**kwargs)
        if self.model.training:
            loss,metrics=result
            self.losses.append(metric_numbers(loss,metrics,self.processor))
        return result


def run_variant(variant,args,out,processor,raw,data,pair,renderer,source,graphs):
    directory=out/variant;directory.mkdir(); model=fresh_model(variant).cuda()
    initial_hash=state_hash(model.state_dict())
    if initial_hash!=pair['variant_model_hashes'][variant]: raise AssertionError('Fresh initialization drift')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    c=configuration(args.root,args.depth_root,source,variant)
    identity=dict(run_id=str((directory/'formal_initial_boundary_probe').resolve()),variant=variant,source_sha=source,
        training=c,model=model.config,processor=processor.state_dict(),data=data,
        environment=args.environment,graph_files_sha256=graphs,initialization=pair)
    boundary=Path(identity['run_id']);boundary.mkdir()
    formal.save_boundary(boundary,model,opt,identity,formal.empty_ledger())
    ledger=formal.restore(boundary,model,opt,identity)
    if ledger!=formal.empty_ledger() or state_hash(model.state_dict())!=initial_hash: raise AssertionError('Initial boundary restore failed')
    audit=UpdateAudit(model,opt);audit.losses=[];audit.processor=processor
    pre=opt.register_step_pre_hook(audit.before);post=opt.register_step_post_hook(audit.after)
    val_evidence={}; phase={'name':'train'}
    def val_guard(module,inputs):
        if not module.training and phase['name']!='val':
            phase['name']='val';val_evidence.update(before_hash=state_hash(model.state_dict()),steps_before=len(audit.records))
    handle=model.register_forward_pre_hook(val_guard)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();clock=time.perf_counter()
    original=torch.nn.utils.clip_grad_norm_
    from scripts.training.train import objective as original_objective
    try:
        with patch('torch.nn.utils.clip_grad_norm_',side_effect=lambda *a,**k:audit.clip(original,*a,**k)), \
             patch('scripts.training.train.objective',side_effect=lambda *a,**k:audit.objective(original_objective,*a,**k)):
            row=perform_epoch(model,opt,processor,renderer,c,'cuda',
                [raw['ep7_start105'],raw['ep7_start105']],[raw['ep3_start135'],raw['ep6_start110']],
                [[7,105],[7,105]],[[3,135],[6,110]])
    finally:pre.remove();post.remove();handle.remove()
    torch.cuda.synchronize();seconds=time.perf_counter()-clock;peak=torch.cuda.max_memory_allocated()
    atomic_json(directory/'runner_evidence.json',dict(runner=row,updates=audit.records,loss_by_visit=audit.losses,
        clip_calls=audit.clips,wall_seconds=seconds,peak_cuda_bytes=peak))
    if row['successful_updates']!=2 or row['optimizer_update_flags']!=[True,True] or audit.clips!=2:
        raise AssertionError('Runner visit/update/clip accounting failed')
    if row['window_order']!=[[7,105],[7,105]] or row['validation_window_order']!=[[3,135],[6,110]]:
        raise AssertionError('Window visitation drift')
    val_evidence.update(after_hash=state_hash(model.state_dict()),steps_after=len(audit.records))
    if val_evidence['before_hash']!=val_evidence['after_hash'] or val_evidence['steps_before']!=val_evidence['steps_after']:
        raise AssertionError('Validation changed weights or stepped optimizer')
    for record in audit.records:
        group=record['groups']['direct_output']
        if group['raw_gradient_norm']<=0 or group['actual_update_norm']<=0: raise AssertionError('Direct projection inactive')
    if variant=='propagate':
        score=audit.records[1]['groups']['propagation_score']
        if score['raw_gradient_norm']<=0 or score['update_minus_adamw_decay_norm']<=0: raise AssertionError('A2A score inactive on update2')
    partial=dict(row,epoch=1)
    try: formal.commit_epoch(formal.empty_ledger(),partial,data['manifest'])
    except ValueError: partial_rejected=True
    else: raise AssertionError('Smoke partial Epoch accepted by formal selection')
    if list(boundary.glob('epoch_*.pt')): raise AssertionError('Smoke manufactured completed formal Epoch')
    batch=to_device(prepare_batch(raw['ep7_start105'],processor,model),'cuda')
    observer=equivalence(model,batch,processor,renderer)
    model.eval()
    with torch.no_grad(): expected=model(batch['inputs'])
    checkpoint=directory/'runner_smoke.pt'
    save_checkpoint(checkpoint,model,opt,source_sha=source,processor_state=processor.state_dict(),successful_steps=2)
    restored=fresh_model(variant).cuda();restored_opt=torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=1e-4)
    state=load_checkpoint(checkpoint,restored,restored_opt,processor_state=processor.state_dict(),source_sha=source)
    if not same_tree(model.state_dict(),restored.state_dict()) or not same_tree(opt.state_dict(),restored_opt.state_dict()):
        raise AssertionError('Strict Smoke checkpoint state restore differs')
    restored.eval()
    with torch.no_grad(): actual=restored(batch['inputs'])
    if not same_tree(expected,actual): raise AssertionError('Restored CUDA prediction differs')
    try: formal.read(checkpoint,identity)
    except ValueError: smoke_rejected=True
    else: raise AssertionError('Formal loader accepted Smoke format')
    result=dict(passed=True,variant=variant,initial_hash=initial_hash,successful_updates=2,visited_train_windows=2,
        visited_val_windows=2,updates=audit.records,training_loss_by_visit=audit.losses,runner=row,validation_readonly=val_evidence,observer_equivalence=observer,
        wall_seconds=seconds,peak_cuda_bytes=peak,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        checkpoint_prediction_bit_exact=True,checkpoint_optimizer_exact=True,partial_formal_epoch_rejected=partial_rejected,
        smoke_format_rejected_by_formal=smoke_rejected,formal_boundary_probe_completed_epochs=0,
        note='Two visits are a smoke-only partial Epoch, never a formal 1514/100 Epoch; formal configuration unchanged.')
    finite_tree(result);atomic_json(directory/'results.json',result)
    del model,opt,restored,restored_opt,batch,actual,expected,state;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','depth-root','g0-reference','output','expected-sha'):p.add_argument('--'+key,required=True)
    args=p.parse_args();source,graphs=guard_source(args.expected_sha);out=smoke_output(args.output)
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('Requires authorized Slurm CUDA GPU')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import gsplat
    args.environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gsplat=gsplat.__version__,gpu=torch.cuda.get_device_name(),tf32=False)
    reference=json.loads((Path(args.g0_reference)/'config.json').read_text())
    if args.environment['cuda']!='12.8' or any(args.environment[k]!=reference[k] for k in ('torch','gsplat','gpu')):
        raise ValueError('Frozen environment mismatch')
    c=configuration(args.root,args.depth_root,source,'direct');loaders,data=preflight(c,args.g0_reference)
    pair=paired_initialization();processor=DataProcessor('001-rope');raw,manifest=load_raw(loaders,processor)
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'provenance.json',dict(source_sha=source,job_id=os.environ['SLURM_JOB_ID'],environment=args.environment,
        model_graph_hashes=graphs,initialization=pair,full_manifest=data,smoke_windows=manifest,command=__import__('sys').argv))
    renderer=AuditedRenderer(out);results={}
    try:
        for variant in ('direct','propagate'):
            print('FORMAL RUNNER INTEGRATION',variant,flush=True)
            results[variant]=run_variant(variant,args,out,processor,raw,data,pair,renderer,source,graphs)
        guard_source(args.expected_sha)
        atomic_json(out/'results.json',dict(passed=True,source_sha=source,job_id=os.environ['SLURM_JOB_ID'],
            variants=results,renderer_finite_calls=renderer.finite_calls,formal_training_started=False))
        print('RUNNER INTEGRATION PASS',flush=True)
    except BaseException:
        atomic_json(out/'failure.json',dict(traceback=traceback.format_exc(),completed_variants=list(results),
            source_sha=source,job_id=os.environ['SLURM_JOB_ID']))
        raise


if __name__=='__main__':main()
