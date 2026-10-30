"""Exactly four fresh real-runner updates per variant; never a formal initializer."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.experiments.v4_g4_global_motion.checkpoint import atomic_save
from scripts.experiments.v4_g4_global_motion.smoke import same_tree
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.diagnostics import finite_tree,sha256,state_hash
from scripts.training.train import atomic_json
from .protocol import (VARIANTS,fresh_model,paired_initialization,configuration,preflight,source_guard,
    resource_guard,environment,configure_runtime,calibration_read,SMOKE_ORDER)
from .runner import CachedRenderer,prepared,assert_real_batch,frozen_attributes,perform_epoch,plain
from .arap import combined_regularizers

VERSION='touch4d-g5-arap-iso-four-update-smoke-v1'
VAL_WINDOWS=((3,135),(6,110))
REGULARIZER_RECOMPUTE_TOLERANCE=dict(atol=1e-7,rtol=1e-5)


def _cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def max_rss_bytes():
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    except ImportError:
        return None


def checkpoint_identity(model,processor,source,calibration_sha):
    return dict(version=VERSION,variant=model.experiment_variant,source_sha=source,
        model_config=copy.deepcopy(model.config),processor=processor.state_dict(),calibration_sha256=calibration_sha)


def _check_optimizer(model,optimizer,serialized=None):
    params=[p for group in optimizer.param_groups for p in group['params']]
    if [id(p) for p in params]!=[id(p) for p in model.parameters()]:raise ValueError('Smoke optimizer order/coverage differs')
    state=optimizer.state_dict() if serialized is None else serialized
    expected=dict(lr=1e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8,amsgrad=False,
        maximize=False,capturable=False,differentiable=False,foreach=None,fused=None)
    groups=state.get('param_groups',[])
    if len(groups)!=1 or groups[0]['params']!=list(range(len(params))) or any(groups[0].get(k)!=v for k,v in expected.items()):
        raise ValueError('Smoke requires the original complete AdamW configuration')
    for index,value in state.get('state',{}).items():
        if type(index) is not int or not 0<=index<len(params):raise ValueError('Unknown Smoke optimizer state parameter')
        for name in ('exp_avg','exp_avg_sq'):
            if name not in value or value[name].shape!=params[index].shape:raise ValueError('Smoke optimizer tensor shape differs')


def save_smoke(path,model,optimizer,identity,updates):
    if type(updates) is not int or updates not in (0,4):raise ValueError('Only initial or completed four-update Smoke snapshots allowed')
    if identity.get('version')!=VERSION or identity.get('variant')!=model.experiment_variant or identity.get('model_config')!=model.config:
        raise ValueError('Smoke identity does not match actual architecture')
    _check_optimizer(model,optimizer)
    state=dict(identity=copy.deepcopy(identity),successful_updates=updates,
               model=_cpu_copy(model.state_dict()),optimizer=_cpu_copy(optimizer.state_dict()))
    finite_tree(state)
    atomic_save(Path(path),state)
    return state


def load_smoke(path,model,optimizer,identity):
    if identity.get('version')!=VERSION or identity.get('variant')!=model.experiment_variant or identity.get('model_config')!=model.config:
        raise ValueError('Smoke restore identity does not match actual architecture')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(state,dict) or state.get('identity')!=identity or type(state.get('successful_updates')) is not int or state['successful_updates']!=4:
        raise ValueError('Wrong variant/source/calibration/processor or incomplete Smoke snapshot')
    finite_tree(state)
    _check_optimizer(model,optimizer,state['optimizer'])
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    return state


def gradient_update_groups(model,before):
    groups=dict(motion='motion.',cross='cross.',conditions='conditions.',global_motion='global_motion.',local_motion='direct_motion.')
    result={}
    for label,prefix in groups.items():
        gradient=update=0.;with_gradient=0
        for name,p in model.named_parameters():
            if not name.startswith(prefix):continue
            if p.grad is not None:
                gradient+=float(p.grad.detach().double().square().sum());with_gradient+=1
            update+=float((p.detach()-before[name]).double().square().sum())
        result[label]=dict(clipped_gradient_norm=gradient**.5,actual_update_norm=update**.5,parameters_with_gradient=with_gradient)
    return result


def _load_raw(raw,split,episode,start):
    dataset=raw[split].dataset
    if (episode,start) not in dataset.windows:raise ValueError('Prescribed window absent from '+split)
    return collate_fn([dataset[dataset.windows.index((episode,start))]])


def _assert_groups_learned(updates):
    for group in ('motion','cross','conditions','global_motion','local_motion'):
        if not any(row['groups'][group]['clipped_gradient_norm']>0 and row['groups'][group]['actual_update_norm']>0 for row in updates):
            raise AssertionError('No finite gradient AND real parameter update in '+group)
    for term in ('arap','iso'):
        norms=[row[term+'_xyz_gradient_norm'] for row in updates if row['successful_update'] in (3,4)]
        if not norms or not all(__import__('math').isfinite(v) for v in norms) or not any(v>0 for v in norms):
            raise AssertionError('Regularizer never has a finite nonzero XYZ gradient in updates3/4: '+term)


def run(args):
    configure_runtime();source,files=source_guard(args.expected_sha)
    calibration=calibration_read(args.calibration);calibration_sha=sha256(args.calibration)
    if calibration['source_sha']!=source:raise ValueError('Smoke must use the committed calibration source')
    resource=resource_guard(require_idle=True);env=environment();out=Path(args.output)
    if not out.is_absolute() or out.exists():raise FileExistsError('New absolute Smoke output directory required')
    config=configuration(args.root,args.depth_root,source,args.variant,calibration['lambda_arap'],calibration['lambda_iso'])
    raw,data=preflight(config,args.g0_reference)
    if calibration['manifest_hash']!=data['manifest_hash']:raise ValueError('Smoke and calibration Manifest differ')
    processor=DataProcessor('001-rope');initialization=paired_initialization(False);model=fresh_model(args.variant)
    if state_hash(model.state_dict())!=initialization['variant_model_hashes'][args.variant]:raise AssertionError('Fresh common initialization differs')
    model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    identity=checkpoint_identity(model,processor,source,calibration_sha)
    out.mkdir(parents=True,exist_ok=False)
    args._created_output=True
    train=[_load_raw(raw,'train',*window) for window in SMOKE_ORDER]
    validation=[_load_raw(raw,'val',*window) for window in VAL_WINDOWS]
    renderer=CachedRenderer();started=time.perf_counter()
    batch=prepared(train[1],processor,model,'cuda');assert_real_batch(batch,7,105)
    reference=G5DirectPredictor(horizon=15,prediction_interval=1).cuda()
    reference.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():expected=reference(batch['inputs']);actual=model(batch['inputs'])
    if not same_tree(expected,actual):raise AssertionError('Real H15 original G5 equivalence failed')
    del batch,reference,expected,actual;torch.cuda.empty_cache()
    save_smoke(out/'smoke_init.pt',model,optimizer,identity,0)
    updates=[];train_records=[];validation_records=[];val_hash=[];last_prediction={}
    def capture(module,inputs,output):
        frozen_attributes(output,inputs[0]);last_prediction.clear();last_prediction.update(output)
        if not module.training and not val_hash:val_hash.append(state_hash(module.state_dict()))
    def pending(visit,successful,inflight):
        args._updates=successful
        atomic_json(out/'progress.json',dict(visited=visit,successful_updates=successful,update_in_flight=inflight,source_sha=source))
    def observe(visit,current,opt,before,parts,graph,batch):
        finite_tree((last_prediction,current.state_dict(),opt.state_dict(),[p.grad for p in current.parameters()]))
        xyz=last_prediction['xyz'].detach().requires_grad_(True)
        arap,iso,stats=combined_regularizers(xyz,batch['inputs']['gs_xyz'],graph,capture_solver_failures=True)
        if stats['kabsch_fallback_count'].sum()>0:
            from .runner import SolverFallbackError
            raise SolverFallbackError(stats)
        ga=torch.autograd.grad(arap,xyz,retain_graph=True)[0];gi=torch.autograd.grad(iso,xyz)[0]
        finite_tree((ga,gi,arap,iso))
        differences={}
        for name,value in (('arap',arap),('iso',iso)):
            reference=parts['loss_'+name+'_unweighted'].detach()
            differences[name]=float((value.detach()-reference).abs())
            torch.testing.assert_close(value.detach(),reference,**REGULARIZER_RECOMPUTE_TOLERANCE)
        updates.append(dict(successful_update=len(updates)+1,visit=visit,groups=gradient_update_groups(current,before),
            loss_parts={k:float(v.detach()) for k,v in parts.items()},graph=graph.stats[0],regularizer_statistics=plain(stats),
            arap_xyz_gradient_norm=float(ga.double().norm()),iso_xyz_gradient_norm=float(gi.double().norm()),
            regularizer_recompute_absolute_difference=differences))
        atomic_json(out/'updates.json',updates)
    hook=model.register_forward_hook(capture);torch.cuda.reset_peak_memory_stats()
    try:
        row=perform_epoch(model,optimizer,processor,renderer,config,'cuda',train,validation,
            [list(w) for w in SMOKE_ORDER],[list(w) for w in VAL_WINDOWS],record=train_records.append,
            validation_record=validation_records.append,pending=pending,step_observer=observe)
    finally:hook.remove()
    if row['successful_updates']!=4 or row['optimizer_update_flags']!=[True]*4 or len(updates)!=4:raise AssertionError('Exactly four successful Smoke updates required')
    _assert_groups_learned(updates)
    if val_hash!=[state_hash(model.state_dict())]:raise AssertionError('Frozen validation changed model')
    snapshot=out/'smoke_4updates.pt';save_smoke(snapshot,model,optimizer,identity,4)
    restored=fresh_model(args.variant).cuda().eval();restored_opt=torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=1e-4)
    load_smoke(snapshot,restored,restored_opt,identity)
    if not same_tree(optimizer.state_dict(),restored_opt.state_dict()):raise AssertionError('Strict optimizer restore differs')
    model.eval()
    for raw_batch,(ep,start) in zip(validation,VAL_WINDOWS):
        batch=prepared(raw_batch,processor,model,'cuda');assert_real_batch(batch,ep,start)
        with torch.no_grad():a=model(batch['inputs']);b=restored(batch['inputs'])
        if not same_tree(a,b):raise AssertionError('Restored H15 predictions differ')
    torch.cuda.synchronize();source_guard(source)
    result=dict(passed=True,source_sha=source,source_files=files,variant=args.variant,
        calibration_sha256=calibration_sha,lambda_arap=config['lambda_arap'],lambda_iso=config['lambda_iso'],successful_updates=4,
        manifest_hash=data['manifest_hash'],epoch_order_hashes=data['epoch_order_hashes'],initialization=initialization,
        resource=resource,environment=env,command=sys.argv,pid=os.getpid(),seconds=time.perf_counter()-started,
        peak_memory_allocated=torch.cuda.max_memory_allocated(),peak_memory_reserved=torch.cuda.max_memory_reserved(),process_max_rss=max_rss_bytes(),
        initial_real_h15_exact=True,attributes_11d_unchanged=True,validation_weights_unchanged=True,
        strict_prediction_restore_exact=True,strict_optimizer_restore_exact=True,
        regularizer_recompute_tolerance=REGULARIZER_RECOMPUTE_TOLERANCE,train_records=train_records,
        validation_records=validation_records,runner_metrics=row,updates=updates,checkpoint_path=str(snapshot.resolve()),checkpoint_sha256=sha256(snapshot),
        note='Exactly four diagnostic updates. Discard this model and optimizer before formal paired fresh training.')
    atomic_json(out/'completion.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','depth-root','g0-reference','variant','calibration','output','expected-sha'):
        parser.add_argument('--'+name,required=True,choices=VARIANTS if name=='variant' else None)
    args=parser.parse_args()
    try:
        result=run(args);print(json.dumps({k:result[k] for k in ('passed','variant','successful_updates','seconds','peak_memory_allocated','peak_memory_reserved')}),flush=True)
    except BaseException as error:
        out=Path(args.output)
        if getattr(args,'_created_output',False) and out.is_dir():atomic_json(out/'failure.json',dict(source_sha=args.expected_sha,variant=args.variant,
            successful_updates=getattr(args,'_updates',0),oom=isinstance(error,torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower(),
            error_type=type(error).__name__,solver_diagnostics=getattr(error,'diagnostics',None),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
