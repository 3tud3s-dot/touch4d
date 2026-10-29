"""Independent fresh paired G5 H15 formal entry. --preflight-only never trains."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import traceback
import torch
from scripts.data_process import DataProcessor
from scripts.training.train import make_renderer,atomic_json
from scripts.experiments.full_training.v4_epochs import loader
from scripts.experiments.v4_g4_global_motion.train import perform_epoch as g4_perform_epoch
from .protocol import configuration,preflight,paired_initialization,fresh_model,output_check
from .formal_checkpoint import empty_ledger,save_boundary,restore,commit_epoch
from .monitor import EpochMonitor,ObservedPredictor
from .smoke import source_preflight


class VisitedStream:
    def __init__(self,raw):self.raw=raw;self.keys=[];self.flags=[]
    def __iter__(self):
        for batch in self.raw:
            self.keys.append([int(batch['meta']['episode_id'][0]),int(batch['meta']['frame_ids'][0,0])])
            self.flags.append(False)
            yield batch


def perform_epoch(model,optimizer,processor,renderer,config,device,raw_train,raw_val,expected_train,expected_val):
    visits=VisitedStream(raw_train);monitor=EpochMonitor(processor.state_dict()['scale'],config['variant']=='propagate')
    observed=ObservedPredictor(model,monitor)
    def step_done(opt,args,kwargs):
        if not visits.flags or visits.flags[-1]:raise ValueError('Optimizer step not associated with exactly one visit')
        visits.flags[-1]=True
    hook=optimizer.register_step_post_hook(step_done)
    try:
        row=g4_perform_epoch(observed,optimizer,processor,renderer,config,device,visits,raw_val,expected_train,expected_val)
    finally:hook.remove()
    if row['window_order']!=visits.keys or sum(visits.flags)!=row['successful_updates']:raise ValueError('Visit/update accounting mismatch')
    row.update(visited_train_windows=len(visits.keys),visited_val_windows=len(row['validation_window_order']),
               optimizer_update_flags=visits.flags,skipped_optimizer_updates=len(visits.flags)-sum(visits.flags),
               g5_diagnostics=monitor.summary())
    return row


def write_new_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('direct','propagate'),required=True)
    for name in ('root','depth-root','g0-reference','output','expected-sha'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--preflight-report')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.preflight_only and (args.resume or not args.preflight_report):parser.error('Preflight requires --preflight-report and forbids --resume')
    source,graphs=source_preflight(args.expected_sha)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    c=configuration(args.root,args.depth_root,source,args.variant)
    out,storage=output_check(args.output,args.variant,resume=args.resume,protected=(args.root,args.depth_root,args.g0_reference))
    raw,data=preflight(c,args.g0_reference)
    initialization=paired_initialization(verify_outputs=True);processor=DataProcessor('001-rope')
    if args.preflight_only:
        write_new_json(args.preflight_report,dict(passed=True,config=c,data=data,initialization=initialization,
            storage=storage,graph_files_sha256=graphs,processor=processor.state_dict(),training_started=False,optimizer_updates=0))
        print('PREFLIGHT ONLY',args.variant,1514,100);return
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Requires explicitly approved Slurm GPU training')
    import gsplat
    environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gsplat=gsplat.__version__,gpu=torch.cuda.get_device_name(),tf32=False)
    reference=json.loads((Path(args.g0_reference)/'config.json').read_text())
    if environment['cuda']!='12.8' or any(environment[k]!=reference[k] for k in ('torch','gsplat','gpu')):
        raise ValueError('Runtime differs from frozen G0/G4')
    model=fresh_model(args.variant).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    if {id(p) for group in optimizer.param_groups for p in group['params']}!={id(p) for p in model.parameters()}:
        raise AssertionError('Optimizer parameter coverage differs')
    if not args.resume:out.mkdir(parents=True,exist_ok=False)
    import fcntl
    with (out/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(run_id=str(out),variant=args.variant,source_sha=source,training=c,model=model.config,
            processor=processor.state_dict(),data=data,environment=environment,graph_files_sha256=graphs,initialization=initialization)
        renderer=make_renderer(c);ledger=empty_ledger()
        try:
            if args.resume:ledger=restore(out,model,optimizer,identity)
            else:
                write_new_json(out/'protocol.json',dict(identity=identity,initialization='fresh seed42 only; no G4/Smoke load',
                    resume='complete immutable Epoch only; interrupted Epoch replayed, logs retained',cuda_bit_reproducibility_claim=False))
                save_boundary(out,model,optimizer,identity,ledger)
            atomic_json(out/'metrics.json',ledger)
            write_new_json(out/('launch-'+os.environ['SLURM_JOB_ID']+'.json'),dict(command=__import__('sys').argv,resume=args.resume,
                source_sha=source,completed_epochs=ledger['completed_epochs']))
            for epoch in range(ledger['completed_epochs'],8):
                source_preflight(args.expected_sha)
                train=raw['train'] if epoch==0 else loader(c,'train',epoch)
                keys=data['manifest']['train']['window_keys'];expected=[keys[i] for i in data['epoch_orders'][epoch]]
                row=perform_epoch(model,optimizer,processor,renderer,c,'cuda',train,raw['val'],expected,data['manifest']['val']['window_keys'])
                row['epoch']=epoch+1
                source_preflight(args.expected_sha)
                candidate=commit_epoch(ledger,row,data['manifest'])
                save_boundary(out,model,optimizer,identity,candidate);ledger=candidate
                atomic_json(out/'metrics.json',ledger)
                print(json.dumps(dict(epoch=epoch+1,successful_updates=ledger['successful_updates'],visited=ledger['visited_train_windows'],
                    joint=row['validation']['model']['objective'])),flush=True)
            atomic_json(out/'status.json',dict(state='COMPLETED',source_sha=source,completed_epochs=8,
                successful_updates=ledger['successful_updates'],visited_train_windows=ledger['visited_train_windows']))
        except BaseException:
            atomic_json(out/('failure-'+os.environ['SLURM_JOB_ID']+'.json'),dict(traceback=traceback.format_exc(),
                in_memory_committed_epoch=ledger['completed_epochs'],
                note='Immutable Epoch files are authority and may have advanced before alias/metrics publication; uncommitted visits/updates are not cumulative'))
            raise


if __name__=='__main__':main()
