"""One canonical worker per variant; fresh paired start, eight full Epochs."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import torch
from scripts.data_process import DataProcessor
from scripts.training.train import atomic_json
from scripts.experiments.full_training.v4_epochs import loader
from scripts.experiments.v4_g5_signed_local.diagnostics import sha256,state_hash,finite_tree
from .model import fresh_model,paired_initialization
from .protocol import configuration,preflight,source_guard,calibration_read,resource_guard,environment,output_check,configure_runtime,graph_review_read
from .runner import perform_epoch,CachedRenderer,JsonLines,WindowBoundaryStop
from . import checkpoint as cp


def peer_ids(path,source):
    if not path:return []
    until=time.monotonic()+60
    while not Path(path).is_file():
        if time.monotonic()>until:raise RuntimeError('Peer registry not published; fail closed')
        time.sleep(.2)
    d=json.loads(Path(path).read_text())
    if d.get('source_sha')!=source or d.get('parent_pid')!=os.getppid() or os.getpid() not in d.get('pids',[]):
        raise ValueError('Peer registry identity differs')
    for pid in d['pids']:
        st=Path(f'/proc/{pid}/status').read_text()
        if int(next(v for v in st.splitlines() if v.startswith('Uid:')).split()[1])!=os.getuid():
            raise ValueError('Peer PID is not owned by current user')
    return d['pids']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','depth-root','g0-reference','variant','output','common','expected-sha'):p.add_argument('--'+key,required=True)
    p.add_argument('--calibration');p.add_argument('--resume',action='store_true');p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--peer-registry')
    a=p.parse_args();stop={'requested':False}
    signal.signal(signal.SIGUSR1,lambda signum,frame:stop.update(requested=True))
    configure_runtime();source,files=source_guard(a.expected_sha)
    common=Path(a.common);prefile=common/'engineering'/'preflight.json'
    weight=.01 if a.preflight_only else calibration_read(a.calibration)['lambda_struct']
    config=configuration(a.root,a.depth_root,source,a.variant,weight)
    out=output_check(a.output,a.variant,resume=a.resume)
    raw,data=preflight(config,a.g0_reference);processor=DataProcessor('001-rope')
    if a.preflight_only:
        if a.resume:raise ValueError('Preflight cannot resume')
        initial=paired_initialization(True)
        result=dict(passed=True,source_sha=source,source_files=files,data=data,initialization=initial,
            configs={v:configuration(a.root,a.depth_root,source,v,.01) for v in initial['variant_model_hashes']},
            processor=processor.state_dict(),optimizer_created=False,successful_updates=0)
        prefile.parent.mkdir(parents=True,exist_ok=True)
        if prefile.exists():
            old=json.loads(prefile.read_text())
            if old!=result:raise ValueError('Existing paired preflight differs')
        else:atomic_json(prefile,result)
        print(json.dumps(dict(preflight=True,train=1514,val=100,initialization=initial)),flush=True)
        return
    evidence=json.loads(prefile.read_text())
    if not evidence['passed'] or evidence['source_sha']!=source or evidence['data']!=data or evidence['processor']!=processor.state_dict():
        raise ValueError('Formal source/data/processor differ from paired preflight')
    initial=evidence['initialization'];calibration_sha=sha256(a.calibration)
    smoke=json.loads((common/'smoke'/a.variant/'completion.json').read_text())
    if not smoke.get('passed') or smoke.get('successful_updates')!=4 or smoke.get('source_sha')!=source or smoke.get('variant')!=a.variant or smoke.get('calibration_sha256')!=calibration_sha:
        raise ValueError('Completed same-source four-update Smoke required')
    graph_review_read(common,a.calibration)
    if stop['requested']:
        target=common/'worker_stops';target.mkdir(exist_ok=True)
        atomic_json(target/f'{os.getpid()}.json',dict(pid=os.getpid(),source_sha=source,variant=a.variant,reason='Safe pre-GPU stop'))
        return
    peers=peer_ids(a.peer_registry,source);resource=resource_guard(require_idle=True,allowed_pids=peers);env=environment()
    model=fresh_model(a.variant)
    if state_hash(model.state_dict())!=initial['variant_model_hashes'][a.variant]:raise ValueError('Fresh shared initialization mismatch')
    model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    if not a.resume:out.mkdir(parents=True,exist_ok=False)
    import fcntl
    with (out/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(run_id=str(out),variant=a.variant,source_sha=source,training=config,model=model.config,
            processor=processor.state_dict(),data=data,environment=env,initialization=initial,calibration_sha256=calibration_sha)
        ledger=cp.empty_ledger();rollback={};begin=time.monotonic()
        try:
            if a.resume:ledger=cp.restore(out,model,optimizer,identity,rollback_info=rollback)
            else:
                atomic_json(out/'protocol.json',dict(identity=identity,source_files=files,
                    calibration=str(Path(a.calibration).resolve()),calibration_execution_source=calibration_read(a.calibration)['source_sha'],
                    initialization='Fresh paired seed42 G5; no historical or Smoke checkpoint load',
                    selection='Original complete-Val100 episode-equal Joint; structure excluded',
                    resume='Immutable complete Epoch only; interrupted attempted updates disclosed separately',cuda_bit_reproducibility_claim=False))
                cp.save_boundary(out,model,optimizer,identity,ledger)
            atomic_json(out/'metrics.json',ledger)
            atomic_json(out/f'launch-{os.getpid()}.json',dict(command=sys.argv,pid=os.getpid(),resource=resource,
                resume=a.resume,rollback=rollback,committed_epochs=ledger['completed_epochs']))
            renderer=CachedRenderer();(out/'logs').mkdir(exist_ok=True)
            for epoch in range(ledger['completed_epochs'],8):
                source_guard(source)
                train=raw['train'] if epoch==0 else loader(config,'train',epoch)
                keys=data['manifest']['train']['window_keys'];order=[keys[i] for i in data['epoch_orders'][epoch]]
                attempt=1+len(list((out/'logs').glob(f'epoch_{epoch+1:02d}-attempt-*-train.jsonl')))
                stem=f'epoch_{epoch+1:02d}-attempt-{attempt:02d}'
                trainlog=JsonLines(out/'logs'/(stem+'-train.jsonl'))
                vallog=JsonLines(out/'logs'/(stem+'-val.jsonl'))
                structlog=JsonLines(out/'logs'/(stem+'-structure.jsonl'))
                counter=out/'logs'/(stem+'-counter.json')
                atomic_json(counter,dict(epoch=epoch+1,attempt=attempt,visited=0,successful_updates=0,pending_update=False))
                def pending(visit,updates,is_pending):
                    atomic_json(counter,dict(epoch=epoch+1,attempt=attempt,visited=visit,successful_updates=updates,
                        pending_update=is_pending,note='If pending at interruption, one additional executed optimizer step is unknown; reported as a range.'))
                def progress(visit,updates,last):
                    atomic_json(out/'progress.json',dict(epoch=epoch+1,attempt=attempt,visited_in_epoch=visit,
                        updates_in_epoch=updates,committed_updates=ledger['successful_updates'],last=last,seconds=time.monotonic()-begin))
                    print(json.dumps(dict(variant=a.variant,epoch=epoch+1,visited=visit,successful_updates=updates,joint=last['joint'])),flush=True)
                start=time.monotonic();torch.cuda.reset_peak_memory_stats()
                try:
                    row=perform_epoch(model,optimizer,processor,renderer,config,'cuda',train,raw['val'],order,
                        data['manifest']['val']['window_keys'],record=trainlog,validation_record=vallog,
                        structure_record=structlog,pending=pending,progress=progress,stop_requested=lambda:stop['requested'])
                finally:
                    trainlog.close();vallog.close();structlog.close()
                finite_tree((model.state_dict(),optimizer.state_dict()))
                row.update(epoch=epoch+1,attempt=attempt,seconds=time.monotonic()-start,
                    peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved(),
                    train_log=str(trainlog.path),validation_log=str(vallog.path),structure_log=str(structlog.path))
                source_guard(source);candidate=cp.commit_epoch(ledger,row,data['manifest'])
                cp.save_boundary(out,model,optimizer,identity,candidate);ledger=candidate
                atomic_json(out/'metrics.json',ledger)
                print(json.dumps(dict(completed_epoch=epoch+1,variant=a.variant,successful_updates=ledger['successful_updates'],
                    best_epoch=ledger['best_epoch'],joint=row['validation']['model']['objective'])),flush=True)
                if stop['requested']:
                    atomic_json(out/'boundary_stop.json',dict(completed_epochs=ledger['completed_epochs'],source_sha=source,pid=os.getpid(),variant=a.variant));return
            atomic_json(out/'completion.json',dict(completed_epochs=8,variant=a.variant,source_sha=source,
                successful_updates=ledger['successful_updates'],best_epoch=ledger['best_epoch'],best_joint=ledger['best_score'],
                seconds=time.monotonic()-begin,checkpoint_sha256={q.name:sha256(q) for q in out.glob('*.pt')}))
        except WindowBoundaryStop as e:
            atomic_json(out/'boundary_stop.json',dict(source_sha=source,variant=a.variant,pid=os.getpid(),
                committed_epochs=ledger['completed_epochs'],reason=str(e),attempted_epoch_replayed=True))
            return
        except BaseException:
            atomic_json(out/f'failure-{os.getpid()}.json',dict(source_sha=source,variant=a.variant,
                committed_epochs=ledger['completed_epochs'],traceback=traceback.format_exc(),
                note='Immutable checkpoints are authoritative; attempted update counters remain in logs for rollback accounting.'))
            raise


if __name__=='__main__':main()
