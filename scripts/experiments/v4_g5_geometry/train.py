"""Eight head-only epochs on frozen G5 E8; original G4/G0 runner and Loss."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch
from scripts.data_process import DataProcessor
from scripts.training.train import atomic_json,make_renderer
from scripts.experiments.full_training.v4_epochs import loader
from scripts.experiments.v4_g4_global_motion.train import perform_epoch as original_perform_epoch
from scripts.experiments.v4_g5_signed_local.protocol import configuration,preflight
from scripts.experiments.v4_g5_signed_local.train import VisitedStream
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.formal_checkpoint import load_for_evaluation
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash,sha256,finite_tree
from .model import GeometryAdapter
from .smoke import BASE,BACKBONE_HASH,source_guard,assigned_gpu_idle
from .formal_checkpoint import empty_ledger,commit_epoch,save_boundary,restore


def perform_epoch(model,opt,processor,renderer,config,device,train,val,train_order,val_order,progress=None):
    visits=VisitedStream(train);updates=[];geometry={'train':[],'val':[]}
    before_hash=state_hash(model.backbone.state_dict())
    def pre(opt,args,kwargs):
        if not visits.flags or visits.flags[-1]:raise AssertionError('Unexpected update visit')
        before.clear();before.update({n:p.detach().clone() for n,p in model.head.named_parameters()})
    def post(opt,args,kwargs):
        visits.flags[-1]=True
        grad=sum(float(p.grad.double().square().sum()) for p in model.head.parameters() if p.grad is not None)**.5
        update=sum(float((p.detach()-before[n]).double().square().sum()) for n,p in model.head.named_parameters())**.5
        updates.append(dict(visit=len(visits.keys),window=visits.keys[-1],clipped_gradient_norm=grad,parameter_update_norm=update))
        if progress is not None and len(visits.keys)%100==0:progress(len(visits.keys),len(updates))
    def capture(module,args,output):
        with torch.no_grad():
            keep=args[0]['gs_mask'][:,None].expand_as(output['xyz'][...,0])
            initial=args[0]['gs_attr'][:,None,:,:3]
            delta=(output['log_scales']-initial)[keep]
            q=output['quaternions'][keep];q0=args[0]['gs_attr'][:,None,:,3:7].expand_as(output['quaternions'])[keep]
            cosine=(q*q0).sum(-1).abs()/(q.norm(dim=-1)*q0.norm(dim=-1)).clamp_min(1e-20)
            angle=2*torch.acos(cosine.clamp(0,1))*180/np.pi
            geometry['train' if module.training else 'val'].append(dict(scale_ratio_mean=float(delta.exp().mean()),
                scale_ratio_min=float(delta.exp().min()),scale_ratio_max=float(delta.exp().max()),
                rotation_deg_mean=float(angle.mean()),rotation_deg_max=float(angle.max())))
    before={};a=opt.register_step_pre_hook(pre);b=opt.register_step_post_hook(post);c=model.register_forward_hook(capture)
    try:
        row=original_perform_epoch(model,opt,processor,renderer,config,device,visits,val,train_order,val_order)
    finally:a.remove();b.remove();c.remove()
    if state_hash(model.backbone.state_dict())!=before_hash:raise AssertionError('Frozen backbone changed in Epoch')
    if row['window_order']!=visits.keys or row['successful_updates']!=sum(visits.flags):raise AssertionError('Update ledger drift')
    row.update(visited_train_windows=len(visits.keys),visited_val_windows=len(row['validation_window_order']),
        optimizer_update_flags=visits.flags,head_updates=updates,geometry_diagnostics=geometry,
        frozen_backbone_state_hash=before_hash)
    return row


def initialize(backbone):
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    model=GeometryAdapter(backbone)
    if any(p.count_nonzero() for p in model.head[-1].parameters()):raise AssertionError('Formal head must start from zero output')
    return model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','depth-root','g0-reference','checkpoint','output','expected-sha','smoke-evidence'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();source,files=source_guard(args.expected_sha)
    if args.preflight_only and args.resume:raise ValueError('Preflight cannot resume')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha256(args.checkpoint)!=BACKBONE_HASH:raise ValueError('Wrong frozen G5 Best')
    evidence=json.loads(Path(args.smoke_evidence).read_text())
    if not evidence.get('engineering_passed') or not evidence.get('xyz_isolation_passed') or evidence.get('successful_updates')!=4:
        raise ValueError('Completed four-update engineering Smoke required')
    # A real rendering regression is disclosed, never converted to an engineering
    # equality pass. User separately authorized proceeding after autonomous review.
    config=configuration(args.root,args.depth_root,source,'direct');raw,data=preflight(config,args.g0_reference)
    processor=DataProcessor('001-rope');backbone=G5DirectPredictor(horizon=15,prediction_interval=1)
    frozen=load_for_evaluation(args.checkpoint,backbone,processor.state_dict(),source_sha=BASE,manifest=data['manifest']);del frozen
    model=initialize(backbone);backbone_hash=state_hash(backbone.state_dict());head_hash=state_hash(model.head.state_dict())
    root=Path(args.output).resolve()
    if root.parent.name!='v4-g5-geometry-decoder-v1-training' or not Path(args.output).is_absolute():raise ValueError('Wrong formal output namespace')
    if args.resume:
        if not (root/'initial.pt').is_file():raise ValueError('Missing durable initial snapshot')
    elif root.exists():raise FileExistsError(root)
    if args.preflight_only:
        print(json.dumps(dict(passed=True,train=1514,val=100,head_initial_hash=head_hash,
            backbone_hash=backbone_hash,geometry_config=model.geometry_config,orders=data['epoch_order_hashes'],
            smoke_render_nonregression=evidence.get('render_nonregression'),optimizer_created=False)),flush=True)
        return
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm CUDA required')
    resource=assigned_gpu_idle()
    environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gsplat=importlib.metadata.version('gsplat'),
        gpu=torch.cuda.get_device_name(),tf32=False)
    if environment['torch']!='2.7.1+cu128' or environment['cuda']!='12.8' or environment['gsplat']!='1.4.0' or '5090' not in environment['gpu']:
        raise RuntimeError('Frozen runtime differs')
    model.cuda();opt=torch.optim.AdamW(model.head.parameters(),lr=1e-4,weight_decay=1e-4)
    if not args.resume:root.mkdir(parents=True,exist_ok=False)
    import fcntl
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(run_id=str(root),source_sha=source,training=config,data=data,processor=processor.state_dict(),
            geometry_config=model.geometry_config,backbone_state_hash=backbone_hash,backbone_checkpoint_sha256=BACKBONE_HASH,
            backbone_source_sha=BASE,initial_head_hash=head_hash,environment=environment)
        ledger=empty_ledger();start=time.perf_counter()
        try:
            if args.resume:ledger=restore(root,model,opt,identity)
            else:
                atomic_json(root/'protocol.json',dict(identity=identity,source_files=files,smoke_evidence=str(Path(args.smoke_evidence).resolve()),
                    smoke_evidence_sha256=sha256(args.smoke_evidence),smoke_render_nonregression=evidence.get('render_nonregression'),
                    initialization='Frozen G5 E8 backbone + fresh seed42 zero-output Geometry Head; no Smoke weight load',
                    resume='Only immutable completed Epochs; interrupted Epoch rolls back and replays',cuda_bit_reproducibility_claim=False))
                save_boundary(root,model,opt,identity,ledger)
            atomic_json(root/f"launch-{os.environ['SLURM_JOB_ID']}.json",dict(command=sys.argv,job=os.environ['SLURM_JOB_ID'],
                step=os.environ.get('SLURM_STEP_ID'),resource=resource,resume=args.resume,completed_epochs=ledger['completed_epochs']))
            renderer=make_renderer(config)
            for epoch in range(ledger['completed_epochs'],8):
                source_guard(source);train=raw['train'] if epoch==0 else loader(config,'train',epoch)
                keys=data['manifest']['train']['window_keys'];order=[keys[i] for i in data['epoch_orders'][epoch]]
                def progress(visits,updates):
                    atomic_json(root/'progress.json',dict(epoch=epoch+1,visited_in_epoch=visits,updates_in_epoch=updates,
                        committed_updates=ledger['successful_updates'],seconds=time.perf_counter()-start))
                    print(json.dumps(dict(epoch=epoch+1,visited=visits,updates=updates)),flush=True)
                begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                row=perform_epoch(model,opt,processor,renderer,config,'cuda',train,raw['val'],order,data['manifest']['val']['window_keys'],progress)
                row.update(epoch=epoch+1,seconds=time.perf_counter()-begin,peak_cuda_bytes=torch.cuda.max_memory_allocated())
                source_guard(source);candidate=commit_epoch(ledger,row,data['manifest'])
                save_boundary(root,model,opt,identity,candidate);ledger=candidate;atomic_json(root/'metrics.json',ledger)
                print(json.dumps(dict(completed_epoch=epoch+1,updates=ledger['successful_updates'],joint=row['validation']['model']['objective'])),flush=True)
            if sha256(args.checkpoint)!=BACKBONE_HASH:raise AssertionError('Original G5 Checkpoint changed')
            atomic_json(root/'completion.json',dict(completed_epochs=8,successful_updates=ledger['successful_updates'],
                best_epoch=ledger['best_epoch'],best_joint=ledger['best_score'],backbone_unchanged=True,total_seconds=time.perf_counter()-start,
                checkpoint_hashes={p.name:sha256(p) for p in root.glob('*.pt')}))
        except BaseException:
            atomic_json(root/f"failure-{os.environ['SLURM_JOB_ID']}.json",dict(traceback=traceback.format_exc(),
                committed_epoch=ledger['completed_epochs'],note='Preserve incomplete Epoch logs; resume only immutable boundary'))
            raise


if __name__=='__main__':main()
