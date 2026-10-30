"""Bounded A/B real-runner engineering Smoke and read-only concurrency probe.

Each ordinary invocation performs exactly the four prescribed diagnostic
updates and discards its state. Probe mode restores its OWN successful Smoke
snapshot, performs one forward/backward, and NEVER calls optimizer.step().
"""
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
from .model import VARIANTS,fresh_model,paired_initialization
from .protocol import (configuration,preflight,source_guard,resource_guard,environment,
                       configure_runtime,calibration_read,SMOKE_ORDER)
from .runner import CachedRenderer,prepared,assert_real_batch,frozen_attributes,losses,perform_epoch
from .structure import structure_loss

VERSION='touch4d-struct-state-four-update-smoke-v1'
VAL_WINDOWS=((3,135),(6,110))


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
    return dict(version=VERSION,variant=model.variant,source_sha=source,
        model_config=copy.deepcopy(model.config),processor=processor.state_dict(),calibration_sha256=calibration_sha)


def save_smoke(path,model,optimizer,identity,updates):
    if updates not in (0,4):raise ValueError('Only initial or completed four-update Smoke snapshots allowed')
    if identity.get('version')!=VERSION or identity.get('variant')!=model.variant or identity.get('model_config')!=model.config:
        raise ValueError('Smoke identity does not match actual architecture')
    params=[p for g in optimizer.param_groups for p in g['params']]
    if [id(p) for p in params]!=[id(p) for p in model.parameters()]:
        raise ValueError('Smoke optimizer must cover all parameters in model order')
    state=dict(identity=copy.deepcopy(identity),successful_updates=updates,
               model=_cpu_copy(model.state_dict()),optimizer=_cpu_copy(optimizer.state_dict()))
    finite_tree(state)
    atomic_save(Path(path),state)
    return state


def load_smoke(path,model,optimizer,identity):
    if identity.get('version')!=VERSION or identity.get('variant')!=model.variant or identity.get('model_config')!=model.config:
        raise ValueError('Smoke restore identity does not match actual architecture')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(state,dict) or state.get('identity')!=identity or state.get('successful_updates')!=4:
        raise ValueError('Wrong variant/source/calibration/processor or incomplete Smoke snapshot')
    finite_tree(state)
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    return state


def gradient_update_groups(model,before):
    groups=dict(motion='motion.',cross='cross.',conditions='conditions.',global_motion='global_motion.',local_motion='direct_motion.')
    if model.has_state:
        groups.update(state_readout='state_memory.readout.',gru='state_memory.cell.',state_encoder='state_memory.encoder.')
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


def wait_for_peers(path,expected_source,timeout=60.):
    if not path:raise ValueError('Concurrent probe requires --peer-registry')
    start=time.monotonic()
    while time.monotonic()-start<timeout:
        try:
            data=json.loads(Path(path).read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            time.sleep(.2);continue
        pids=data.get('pids',[])
        if data.get('source_sha')!=expected_source or data.get('parent_pid')!=os.getppid():
            raise ValueError('Peer registry source or launcher parent mismatch')
        if not isinstance(pids,list) or any(type(p) is not int or p<=0 for p in pids):
            raise ValueError('Invalid peer registry PIDs')
        if os.getpid() not in pids:
            time.sleep(.2);continue
        return data
    raise TimeoutError('No matching concurrency peer registry within 60 seconds')


def probe_barrier(peers,out,variant,source,ready,timeout=180.):
    """Hold BOTH full render/autograd graphs live until launcher checks VRAM."""
    barrier=Path(peers.get('barrier',''))
    if not barrier.is_absolute() or not barrier.is_dir():
        raise ValueError('Probe requires the launcher-owned absolute barrier directory')
    outputs=peers.get('outputs',{})
    if variant not in outputs or Path(outputs[variant]).resolve()!=Path(out).resolve():
        raise ValueError('Probe output does not match its registered variant')
    payload=dict(ready,source_sha=source,variant=variant,pid=os.getpid(),parent_pid=os.getppid())
    atomic_json(Path(out)/'ready.json',payload)
    marker=barrier/(variant+'-ready.json')
    if marker.exists():raise FileExistsError('Probe readiness marker already exists')
    atomic_json(marker,payload)
    started=time.monotonic()
    while time.monotonic()-started<timeout:
        try:release=json.loads((barrier/'go.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            time.sleep(.2);continue
        if release.get('source_sha')!=source or release.get('parent_pid')!=os.getppid():
            raise ValueError('Probe release source/parent identity mismatch')
        if release.get('abort'):
            raise RuntimeError('Concurrent probe rejected by launcher safety check: '+str(release.get('reason')))
        if release.get('go') is True:return release
        time.sleep(.2)
    raise TimeoutError('Launcher did not release both live probe graphs within 180 seconds')


def _load_raw(raw,split,episode,start):
    dataset=raw[split].dataset
    if (episode,start) not in dataset.windows:raise ValueError('Prescribed window absent from '+split)
    return collate_fn([dataset[dataset.windows.index((episode,start))]])


def _assert_groups_learned(updates,has_state):
    for group in ('motion','cross','conditions','global_motion','local_motion'):
        if not any(row['groups'][group]['clipped_gradient_norm']>0 and row['groups'][group]['actual_update_norm']>0 for row in updates):
            raise AssertionError('No finite gradient AND real update in '+group)
    if has_state:
        readout_steps=[r['successful_update'] for r in updates if r['groups']['state_readout']['clipped_gradient_norm']>0 and r['groups']['state_readout']['actual_update_norm']>0]
        if not readout_steps:raise AssertionError('State readout failed to learn')
        for group in ('gru','state_encoder'):
            if not any(r['successful_update']>min(readout_steps) and r['groups'][group]['clipped_gradient_norm']>0 and r['groups'][group]['actual_update_norm']>0 for r in updates):
                raise AssertionError('State module failed after readout activated: '+group)


def run(args):
    configure_runtime();source,files=source_guard(args.expected_sha)
    calibration=calibration_read(args.calibration);calibration_sha=sha256(args.calibration)
    peers=wait_for_peers(args.peer_registry,source) if args.probe else None
    if args.probe and not args.smoke_checkpoint:raise ValueError('Probe needs its own completed Smoke snapshot')
    if not args.probe and (args.peer_registry or args.smoke_checkpoint):raise ValueError('Training Smoke cannot load weights or share GPU')
    resource=resource_guard(require_idle=True,allowed_pids=peers['pids'] if peers else ())
    env=environment();out=Path(args.output)
    if not out.is_absolute() or out.exists():raise FileExistsError('New absolute Smoke/probe output directory required')
    out.mkdir(parents=True,exist_ok=False)
    config=configuration(args.root,args.depth_root,source,args.variant,calibration['lambda_struct'])
    raw,data=preflight(config,args.g0_reference);processor=DataProcessor('001-rope')
    initialization=paired_initialization(False)
    model=fresh_model(args.variant)
    if state_hash(model.state_dict())!=initialization['variant_model_hashes'][args.variant]:
        raise AssertionError('Fresh common initialization differs')
    model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    identity=checkpoint_identity(model,processor,source,calibration_sha)
    if args.probe:
        # The concurrency probe needs one complete real batch, not four unused
        # raw windows whose depth tensors would inflate the CPU memory budget.
        train=[_load_raw(raw,'train',7,105)];validation=[]
    else:
        cache={w:_load_raw(raw,'train',*w) for w in dict.fromkeys(SMOKE_ORDER)}
        train=[cache[w] for w in SMOKE_ORDER]
        validation=[_load_raw(raw,'val',ep,start) for ep,start in VAL_WINDOWS]
    args._updates=0
    started=time.perf_counter();renderer=CachedRenderer()

    if args.probe:
        state=load_smoke(args.smoke_checkpoint,model,optimizer,identity)
        before_model=state_hash(model.state_dict());before_opt=_cpu_copy(optimizer.state_dict())
        batch=prepared(train[0],processor,model,'cuda');assert_real_batch(batch,7,105)
        model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        pred=model(batch['inputs']);frozen_attributes(pred,batch['inputs'])
        total,values,parts,graph=losses(pred,batch,processor,renderer,config)
        torch.cuda.synchronize()
        release=probe_barrier(peers,out,args.variant,source,
            dict(peak_memory_allocated=torch.cuda.max_memory_allocated(),
                 peak_memory_reserved=torch.cuda.max_memory_reserved(),process_max_rss=max_rss_bytes()))
        total.backward()  # Deliberately no optimizer.step in probe mode.
        finite_tree((pred,total,model.state_dict(),optimizer.state_dict(),[p.grad for p in model.parameters()]))
        if state_hash(model.state_dict())!=before_model or not same_tree(before_opt,optimizer.state_dict()):
            raise AssertionError('Read-only memory probe changed model/optimizer')
        torch.cuda.synchronize()
        result=dict(passed=True,probe=True,source_sha=source,variant=args.variant,calibration_sha256=calibration_sha,
            successful_updates=0,checkpoint_updates=state['successful_updates'],checkpoint_sha256=sha256(args.smoke_checkpoint),
            initialization=initialization,resource=resource,peer_registry=peers,barrier_release=release,environment=env,
            peak_memory_allocated=torch.cuda.max_memory_allocated(),peak_memory_reserved=torch.cuda.max_memory_reserved(),
            process_max_rss=max_rss_bytes(),seconds=time.perf_counter()-started,weights_unchanged=True,
            optimizer_unchanged=True,loss_parts={k:float(v.detach()) for k,v in parts.items()},command=sys.argv,pid=os.getpid())
        atomic_json(out/'completion.json',result);return result

    # A/B fresh zero-state equivalence on a real, three-view H15 batch.
    batch=prepared(train[1],processor,model,'cuda');assert_real_batch(batch,7,105)
    reference=G5DirectPredictor(horizon=15,prediction_interval=1).cuda()
    reference.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('state_memory.')},strict=True)
    with torch.no_grad():
        expected=reference(batch['inputs']);actual=model(batch['inputs'])
    if not same_tree(expected,actual):raise AssertionError('Real H15 fresh initialization equality failed')
    del batch,reference,expected,actual
    torch.cuda.empty_cache()
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
        # An independent output-space derivative checks the fixed GS graph,
        # without another network backward or another optimizer update.
        xyz=last_prediction['xyz'].detach().requires_grad_(True)
        local_loss=structure_loss(xyz,graph)
        g=torch.autograd.grad(local_loss,xyz)[0]
        finite_tree((g,local_loss))
        if not torch.equal(local_loss.detach(),parts['loss_struct_unweighted'].detach()):
            raise AssertionError('Final-GS structural objective changed')
        groups=gradient_update_groups(current,before)
        updates.append(dict(successful_update=len(updates)+1,visit=visit,groups=groups,
            loss_parts={k:float(v.detach()) for k,v in parts.items()},graph=graph.stats[0],
            structure_xyz_gradient_norm=float(g.double().norm()),
            weighted_structure_xyz_gradient_norm=float(g.double().norm())*config['lambda_struct']))
        atomic_json(out/'updates.json',updates)
    hook=model.register_forward_hook(capture)
    torch.cuda.reset_peak_memory_stats()
    try:
        row=perform_epoch(model,optimizer,processor,renderer,config,'cuda',train,validation,
            [list(w) for w in SMOKE_ORDER],[list(w) for w in VAL_WINDOWS],
            record=train_records.append,validation_record=validation_records.append,pending=pending,step_observer=observe)
    finally:hook.remove()
    if row['successful_updates']!=4 or row['optimizer_update_flags']!=[True]*4 or len(updates)!=4:
        raise AssertionError('Exactly four successful diagnostic updates required')
    _assert_groups_learned(updates,model.has_state)
    if val_hash!=[state_hash(model.state_dict())]:raise AssertionError('Frozen validation changed weights/buffers')
    snapshot=out/'smoke_4updates.pt';save_smoke(snapshot,model,optimizer,identity,4)
    restored=fresh_model(args.variant).cuda().eval()
    restored_opt=torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=1e-4)
    load_smoke(snapshot,restored,restored_opt,identity)
    if not same_tree(optimizer.state_dict(),restored_opt.state_dict()):raise AssertionError('Strict optimizer restore differs')
    model.eval()
    for raw_batch,(ep,start) in zip(validation,VAL_WINDOWS):
        batch=prepared(raw_batch,processor,model,'cuda');assert_real_batch(batch,ep,start)
        with torch.no_grad():a=model(batch['inputs']);b=restored(batch['inputs'])
        if not same_tree(a,b):raise AssertionError('Strict restored H15 prediction differs')
    torch.cuda.synchronize()
    source_guard(source)
    result=dict(passed=True,probe=False,source_sha=source,source_files=files,variant=args.variant,
        calibration_sha256=calibration_sha,lambda_struct=config['lambda_struct'],successful_updates=4,
        initialization=initialization,resource=resource,environment=env,command=sys.argv,pid=os.getpid(),
        seconds=time.perf_counter()-started,peak_memory_allocated=torch.cuda.max_memory_allocated(),
        peak_memory_reserved=torch.cuda.max_memory_reserved(),process_max_rss=max_rss_bytes(),
        initial_real_h15_exact=True,attributes_11d_unchanged=True,validation_weights_unchanged=True,
        strict_prediction_restore_exact=True,strict_optimizer_restore_exact=True,
        train_records=train_records,validation_records=validation_records,runner_metrics=row,updates=updates,
        checkpoint_path=str(snapshot.resolve()),checkpoint_sha256=sha256(snapshot),
        note='Four-update Smoke only, not formal model selection. Its model/optimizer must be discarded before fresh paired eight-epoch training.')
    atomic_json(out/'completion.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','depth-root','g0-reference','variant','calibration','output','expected-sha'):
        parser.add_argument('--'+name,required=True,choices=VARIANTS if name=='variant' else None)
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--smoke-checkpoint');parser.add_argument('--peer-registry')
    args=parser.parse_args()
    try:
        result=run(args)
        print(json.dumps({k:result[k] for k in ('passed','probe','variant','successful_updates','seconds','peak_memory_allocated','peak_memory_reserved')}),flush=True)
    except BaseException as exc:
        out=Path(args.output)
        if out.is_dir():
            atomic_json(out/'failure.json',dict(source_sha=args.expected_sha,variant=args.variant,
                successful_updates=getattr(args,'_updates',0),probe=args.probe,
                oom=isinstance(exc,torch.cuda.OutOfMemoryError) or 'out of memory' in str(exc).lower(),
                error_type=type(exc).__name__,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
