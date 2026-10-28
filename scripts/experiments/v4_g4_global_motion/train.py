"""Fresh G4 eight-epoch H15 entry; preflight-only is CPU and never trains."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import traceback
import torch
from scripts.data_process import DataProcessor
from scripts.experiments.full_training.v4_epochs import loader,PreparedLoader,PreparedProcessor
from scripts.training.train import train_epoch,evaluate_loader,make_renderer,atomic_json
from .protocol import configuration,preflight,fresh_model,CheckedStream,full_validation_score
from .checkpoint import save_boundary,restore,finite


class ContributionLog:
    """Detached per-horizon diagnostics; never used by loss or selection."""
    def __init__(self,scale):self.scale=scale*1000;self.rows={}
    def __call__(self,module,args,output):
        step=int(args[1])+1;delta,trace=output
        with torch.no_grad():
            hands=trace['hand_contribution'].detach().double().cpu()*self.scale
            shift=delta.detach().double().cpu()*self.scale
            finite(hands);finite(shift)
            valid=trace['hand_valid'].detach().cpu()
            if torch.count_nonzero(hands[~valid]):raise ValueError('Masked hand leaked')
            torch.testing.assert_close(shift,hands.sum(1),atol=1e-5,rtol=1e-5)
            row=self.rows.setdefault(step,dict(samples=0,valid_hands=torch.zeros(2,dtype=torch.int64),
                hand_sum=torch.zeros(2,3,dtype=torch.float64),delta_sum=torch.zeros(3,dtype=torch.float64),
                hand_sq=torch.zeros(2,dtype=torch.float64),delta_sq=0.))
            row['samples']+=len(shift);row['valid_hands']+=valid.sum(0)
            row['hand_sum']+=hands.sum(0);row['delta_sum']+=shift.sum(0)
            row['hand_sq']+=hands.square().sum(-1).sum(0);row['delta_sq']+=float(shift.square().sum())
    def summary(self):
        return {str(k):dict(samples=v['samples'],valid_hands=v['valid_hands'].tolist(),
                 hand_mean_xyz_mm=(v['hand_sum']/v['samples']).tolist(),
                 global_mean_xyz_mm=(v['delta_sum']/v['samples']).tolist(),
                 hand_rms_mm=(v['hand_sq']/v['samples']).sqrt().tolist(),
                 global_rms_mm=(v['delta_sq']/v['samples'])**.5)
                for k,v in sorted(self.rows.items())}


def perform_epoch(model,optimizer,processor,renderer,config,device,raw_train,raw_val,expected_train,expected_val):
    counts={'updates':0};grad={};contributions={};stream=CheckedStream(raw_train,expected_train)
    def after_step(opt,args,kwargs):
        finite(model.state_dict());finite(opt.state_dict());counts['updates']+=1
        for name,p in model.global_motion.named_parameters():
            if p.grad is not None:
                finite(p.grad);grad[name]=grad.get(name,0.)+float(p.grad.detach().norm())
    hook=optimizer.register_step_post_hook(after_step)
    log=ContributionLog(processor.state_dict()['scale']);capture=model.global_motion.register_forward_hook(log)
    try:
        training=train_epoch(model,optimizer,PreparedLoader(stream,processor,model),PreparedProcessor(processor),
                             renderer,config,device)
    finally:hook.remove();capture.remove()
    contributions['train']=log.summary()
    val_stream=CheckedStream(raw_val,expected_val)
    log=ContributionLog(processor.state_dict()['scale']);capture=model.global_motion.register_forward_hook(log)
    try:
        validation=evaluate_loader(model,PreparedLoader(val_stream,processor,model),PreparedProcessor(processor),
                                   renderer,config,device)
    finally:capture.remove()
    contributions['val']=log.summary()
    return dict(training=training,validation=validation,successful_updates=counts['updates'],
                window_order=stream.seen,validation_window_order=val_stream.seen,
                contributions=contributions,clipped_gradient_norm_sum=grad,
                partial_train=False,partial_validation=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','depth-root','g0-reference','output','expected-sha'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Resume latest durable complete epoch in the SAME output directory')
    args=parser.parse_args()
    if args.preflight_only and args.resume:parser.error('Preflight cannot resume')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source!=args.expected_sha or subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Source must be clean and pinned')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    c=configuration(args.root,args.depth_root,source)
    raw_loaders,data=preflight(c,args.g0_reference)
    model,initialization=fresh_model();processor=DataProcessor('001-rope')
    out=Path(args.output).resolve()
    if args.preflight_only:
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'preflight.json',dict(passed=True,config=c,data=data,initialization=initialization,
            processor=processor.state_dict(),torch=torch.__version__,training_started=False))
        print('PREFLIGHT ONLY',len(data['manifest']['train']['window_keys']),len(data['manifest']['val']['window_keys']))
        return
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Formal training requires a Slurm GPU allocation')
    import gsplat
    environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gsplat=gsplat.__version__,
                     gpu=torch.cuda.get_device_name(),tf32=False)
    g0_env=json.loads((Path(args.g0_reference)/'config.json').read_text())
    if any(environment[k]!=g0_env[k] for k in ('torch','gsplat','gpu')):
        raise ValueError('G0/G4 runtime differs; explicitly audit before comparing')
    if args.resume:
        if not out.is_dir():raise ValueError('Resume directory missing')
    else:out.mkdir(parents=True,exist_ok=False)
    # Linux OS lock releases on process exit; stale files do not authorize concurrent runs.
    import fcntl
    with (out/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(run_id=str(out),source_sha=source,training=c,model=model.config,
                      processor=processor.state_dict(),data=data,environment=environment)
        model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
        renderer=make_renderer(c)
        ledger=dict(completed_epochs=0,successful_updates=0,best_score=None,best_epoch=None,history=[])
        try:
            if args.resume:ledger=restore(out,model,optimizer,identity)
            else:
                atomic_json(out/'protocol.json',dict(identity=identity,initialization=initialization,
                    resume_policy='complete epoch only; uncommitted epoch is rolled back and replayed',
                    no_formal_g0_or_smoke_weight_loading=True))
                save_boundary(out,model,optimizer,identity,ledger)
            atomic_json(out/('launch-'+os.environ['SLURM_JOB_ID']+'.json'),dict(command=__import__('sys').argv,
                         source_sha=source,resume=args.resume,completed_epochs=ledger['completed_epochs']))
            for ep in range(ledger['completed_epochs'],8):
                raw_train=raw_loaders['train'] if ep==0 else loader(c,'train',ep)
                keys=data['manifest']['train']['window_keys'];order=[keys[i] for i in data['epoch_orders'][ep]]
                row=perform_epoch(model,optimizer,processor,renderer,c,'cuda',raw_train,raw_loaders['val'],
                                  order,data['manifest']['val']['window_keys'])
                row['epoch']=ep+1
                score=full_validation_score(row['validation'],data['manifest'])
                ledger['history'].append(row);ledger['completed_epochs']=ep+1
                ledger['successful_updates']+=row['successful_updates']
                if ledger['best_score'] is None or score<ledger['best_score']:
                    ledger['best_score']=score;ledger['best_epoch']=ep+1
                save_boundary(out,model,optimizer,identity,ledger)
                atomic_json(out/'metrics.json',ledger)
                print(json.dumps(dict(epoch=ep+1,updates=ledger['successful_updates'],joint=score)),flush=True)
            atomic_json(out/'metrics.json',ledger)
            atomic_json(out/'status.json',dict(state='COMPLETED',completed_epochs=8,
                        successful_updates=ledger['successful_updates'],source_sha=source))
        except BaseException:
            atomic_json(out/('failure-'+os.environ['SLURM_JOB_ID']+'.json'),dict(traceback=traceback.format_exc(),
                         note='Only immutable completed epoch files are resume authority; in-memory epoch may be uncommitted'))
            raise


if __name__=='__main__':main()
