"""Fresh joint G5/GEO eight-epoch training, using the original G4/G5 runner."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
from scripts.data_process import DataProcessor
from scripts.training.train import atomic_json, make_renderer
from scripts.experiments.full_training.v4_epochs import loader
from scripts.experiments.v4_g4_global_motion.train import perform_epoch as original_epoch
from scripts.experiments.v4_g5_signed_local.train import VisitedStream
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash, sha256
from scripts.experiments.v4_g5_geometry.smoke import assigned_gpu_idle
from .model import fresh_model, initialization
from .protocol import configuration, preflight, source_guard, output_check
from . import checkpoint as cp


def configure_runtime():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def environment():
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Allocated Slurm CUDA required')
    info=dict(torch=torch.__version__,cuda=torch.version.cuda,gsplat=importlib.metadata.version('gsplat'),
              gpu=torch.cuda.get_device_name(),tf32=False)
    if info['torch']!='2.7.1+cu128' or info['cuda']!='12.8' or info['gsplat']!='1.4.0' or '5090' not in info['gpu']:
        raise RuntimeError('Original G5 CUDA/gsplat environment required')
    return info


def perform_epoch(model, optimizer, processor, renderer, config, device, train, val, train_order, val_order, progress=None):
    visits=VisitedStream(train)
    def after_step(opt,args,kwargs):
        if not visits.flags or visits.flags[-1]:raise AssertionError('Bad visit/update association')
        visits.flags[-1]=True
        if progress is not None and len(visits.keys)%100==0:
            progress(len(visits.keys),sum(visits.flags))
    hook=optimizer.register_step_post_hook(after_step)
    try:
        row=original_epoch(model,optimizer,processor,renderer,config,device,visits,val,train_order,val_order)
    finally:
        hook.remove()
    if row['window_order']!=visits.keys or row['successful_updates']!=sum(visits.flags):
        raise AssertionError('Window/update accounting drift')
    row.update(visited_train_windows=len(visits.keys),visited_val_windows=len(row['validation_window_order']),
               optimizer_update_flags=visits.flags,skipped_optimizer_updates=len(visits.flags)-sum(visits.flags))
    return row


def make_identity(root,model,processor,config,data,env,initial):
    return dict(run_id=str(Path(root).resolve()),source_sha=config['source_sha'],training=config,
        model=model.config,geometry_config=model.geometry_config,processor=processor.state_dict(),data=data,
        environment=env,initialization=initial,optimizer_parameter_names=[n for n,_ in model.named_parameters()])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','depth-root','g0-reference','output','expected-sha'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--smoke-evidence')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();configure_runtime()
    source,files=source_guard(args.expected_sha)
    config=configuration(args.root,args.depth_root,source)
    root=output_check(args.output,resume=args.resume)
    raw,data=preflight(config,args.g0_reference)
    model=fresh_model();initial=initialization(model);processor=DataProcessor('001-rope')
    if args.preflight_only:
        if args.resume:raise ValueError('Preflight is not resume')
        print(json.dumps(dict(passed=True,train=1514,val=100,config=config,initialization=initial,
            orders=data['epoch_order_hashes'],manifest_hash=data['manifest_hash'],optimizer_created=False)),flush=True)
        return
    if not args.smoke_evidence:raise ValueError('Completed joint GPU Smoke required')
    smoke_path=Path(args.smoke_evidence);smoke=json.loads(smoke_path.read_text())
    if not smoke.get('passed') or smoke.get('source_sha')!=source or smoke.get('successful_updates')!=2 or smoke.get('initialization')!=initial:
        raise ValueError('Wrong joint Smoke evidence/source/fresh initialization')
    resource=assigned_gpu_idle();env=environment()
    model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    if not args.resume:root.mkdir(parents=True,exist_ok=False)
    import fcntl
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=make_identity(root,model,processor,config,data,env,initial)
        ledger=cp.empty_ledger();begin=time.perf_counter()
        try:
            if args.resume:ledger=cp.restore(root,model,optimizer,identity)
            else:
                atomic_json(root/'protocol.json',dict(identity=identity,source_files=files,
                    smoke_path=str(smoke_path.resolve()),smoke_sha256=sha256(smoke_path),
                    initialization='Entire G5 + Geometry Head freshly initialized; no trained/Smoke checkpoint load',
                    gradient_contract='All G5 and GEO trainable; features/displacements retain gradient; geometry never feeds AR',
                    resume='Complete immutable Epoch boundaries only; uncommitted Epoch replays',cuda_bit_reproducibility_claim=False))
                cp.save_boundary(root,model,optimizer,identity,ledger)
            atomic_json(root/f"launch-{os.environ['SLURM_JOB_ID']}.json",dict(command=sys.argv,job=os.environ['SLURM_JOB_ID'],
                step=os.environ.get('SLURM_STEP_ID'),resource=resource,resume=args.resume,completed_epochs=ledger['completed_epochs']))
            renderer=make_renderer(config)
            for epoch in range(ledger['completed_epochs'],8):
                source_guard(source)
                train=raw['train'] if epoch==0 else loader(config,'train',epoch)
                keys=data['manifest']['train']['window_keys'];order=[keys[i] for i in data['epoch_orders'][epoch]]
                def progress(visits,updates):
                    atomic_json(root/'progress.json',dict(epoch=epoch+1,visited_in_epoch=visits,updates_in_epoch=updates,
                        committed_updates=ledger['successful_updates'],seconds=time.perf_counter()-begin))
                    print(json.dumps(dict(epoch=epoch+1,visits=visits,updates=updates)),flush=True)
                start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                row=perform_epoch(model,optimizer,processor,renderer,config,'cuda',train,raw['val'],order,data['manifest']['val']['window_keys'],progress)
                row.update(epoch=epoch+1,seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                    backbone_state_hash=state_hash(model.backbone.state_dict()),head_state_hash=state_hash(model.head.state_dict()))
                source_guard(source);candidate=cp.commit_epoch(ledger,row,data['manifest'])
                cp.save_boundary(root,model,optimizer,identity,candidate);ledger=candidate
                atomic_json(root/'metrics.json',ledger)
                print(json.dumps(dict(completed_epoch=epoch+1,successful_updates=ledger['successful_updates'],
                    best_epoch=ledger['best_epoch'],joint=row['validation']['model']['objective'])),flush=True)
            atomic_json(root/'completion.json',dict(completed_epochs=8,successful_updates=ledger['successful_updates'],
                best_epoch=ledger['best_epoch'],best_joint=ledger['best_score'],seconds=time.perf_counter()-begin,
                checkpoint_sha256={p.name:sha256(p) for p in root.glob('*.pt')}))
        except BaseException:
            atomic_json(root/f"failure-{os.environ['SLURM_JOB_ID']}.json",dict(traceback=traceback.format_exc(),
                committed_epochs=ledger['completed_epochs'],note='Retain logs and rollback uncommitted Epoch only'))
            raise


if __name__=='__main__':main()
