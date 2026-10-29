"""Direct only: resume exact Last8 into an isolated eight-additional-Epoch Run."""
import argparse
import copy
import json
import os
from pathlib import Path
import traceback
import torch
from scripts.data_process import DataProcessor
from scripts.training.train import make_renderer,atomic_json
from scripts.experiments.full_training.v4_epochs import loader
from scripts.experiments.v4_g4_global_motion.protocol import epoch_order,digest
from scripts.experiments.v4_g4_global_motion.checkpoint import restore_rng
from .protocol import configuration,preflight,probe_order,fresh_model
from .train import perform_epoch
from .formal_checkpoint import commit_epoch
from .diagnostics import state_hash,sha256
from .runner_smoke import guard_source
from . import continuation_checkpoint as cp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','depth-root','g0-reference','parent','parent-sha256','output','expected-sha'):p.add_argument('--'+key,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--preflight-only',action='store_true');p.add_argument('--preflight-report')
    a=p.parse_args()
    if a.preflight_only and (a.resume or not a.preflight_report):p.error('Preflight requires report, forbids resume')
    source,graphs=guard_source(a.expected_sha);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(a.output).resolve();parent_path=Path(a.parent).resolve()
    if not Path(a.output).is_absolute() or out.parent.name!='v4-g5-direct-e16-continuation-v1' or Path(a.output).is_symlink():
        raise ValueError('Separate absolute continuation output required')
    for protected in (Path.cwd(),Path(a.root).resolve(),Path(a.depth_root).resolve(),parent_path.parent):
        if out==protected or protected in out.parents:raise ValueError('Output overlaps protected source/data/parent')
    if a.resume:
        if not (out/'initial.pt').exists():raise ValueError('No continuation origin')
    elif out.exists():raise FileExistsError(out)
    c=cp.training_config(a.root,a.depth_root,source)
    raw,data=preflight(configuration(a.root,a.depth_root,source,'direct'),a.g0_reference)
    processor=DataProcessor('001-rope')
    parent=cp.load_parent(parent_path,a.parent_sha256,processor.state_dict(),data['manifest'])
    expected_parent=configuration(a.root,a.depth_root,cp.PARENT_SHA,'direct')
    if parent['identity']['training']!=expected_parent:raise ValueError('Parent frozen config differs')
    if parent['identity']['data']['epoch_orders']!=data['epoch_orders']:raise ValueError('Parent eight-Epoch order drift')
    data=copy.deepcopy(data)
    data['epoch_orders']=[epoch_order(data['manifest']['train']['windows'],i) for i in range(16)]
    data['epoch_order_hashes']=[digest(v) for v in data['epoch_orders']]
    for i in range(8,16):
        if probe_order(loader(c,'train',i))!=data['epoch_orders'][i]:raise ValueError('Actual continuation shuffle mismatch')
    data['observed_epoch_order_hashes']=data['epoch_order_hashes']
    evidence=dict(passed=True,source_sha=source,parent=str(parent_path),parent_sha256=a.parent_sha256,
        output=str(out),training=c,data=data,parent_completed_epochs=8,parent_successful_updates=parent['ledger']['successful_updates'],
        additional_epochs=8,epoch_seed_sequence=list(range(50,58)),initialization='restore Last8 model+Optimizer+all RNG; no reset',
        graph_files_sha256=graphs,parent_model_hash=state_hash(parent['model']),
        parent_optimizer_state_entries=len(parent['optimizer']['state']),
        parent_optimizer_steps=sorted(set(float(s['step']) for s in parent['optimizer']['state'].values() if 'step' in s)),
        parent_rng_components=list(parent['rng']),parent_cuda_rng_states=len(parent['rng']['cuda'] or []))
    if a.preflight_only:
        with Path(a.preflight_report).open('x') as f:json.dump(evidence,f,indent=2,allow_nan=False)
        print('CONTINUATION PREFLIGHT PASS');return
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Requires approved Slurm GPU')
    import gsplat
    env=dict(torch=torch.__version__,cuda=torch.version.cuda,gsplat=gsplat.__version__,gpu=torch.cuda.get_device_name(),tf32=False)
    if env!=parent['identity']['environment']:raise ValueError('Continuation environment differs')
    model=fresh_model('direct').cuda();opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    model.load_state_dict(parent['model'],strict=True);opt.load_state_dict(parent['optimizer'])
    if any(g['lr']!=c['lr'] or g['weight_decay']!=c['weight_decay'] for g in opt.param_groups):raise ValueError('Optimizer hyperparameters differ')
    identity=dict(run_id=str(out),variant='direct',source_sha=source,training=c,model=model.config,processor=processor.state_dict(),
        data=data,environment=env,parent=str(parent_path),parent_sha256=a.parent_sha256,parent_source_sha=cp.PARENT_SHA,
        parent_ledger=copy.deepcopy(parent['ledger']),parent_model_hash=state_hash(parent['model']),graph_files_sha256=graphs)
    renderer=make_renderer(c)
    if not a.resume:out.mkdir(parents=True,exist_ok=False)
    import fcntl
    with (out/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if a.resume:ledger=cp.restore(out,model,opt,identity)
            else:
                restore_rng(parent['rng']);ledger=copy.deepcopy(parent['ledger'])
                atomic_json(out/'protocol.json',dict(identity=identity,scope='Independent Epoch9-16 continuation; original eight-Epoch results untouched'))
                cp.save(out,model,opt,identity,ledger)
            del parent
            atomic_json(out/'metrics.json',ledger)
            atomic_json(out/('launch-'+os.environ['SLURM_JOB_ID']+'.json'),dict(command=__import__('sys').argv,source_sha=source,resume=a.resume,start_epoch=ledger['completed_epochs']))
            for i in range(ledger['completed_epochs'],16):
                guard_source(a.expected_sha)
                order=[data['manifest']['train']['window_keys'][j] for j in data['epoch_orders'][i]]
                row=perform_epoch(model,opt,processor,renderer,c,'cuda',loader(c,'train',i),raw['val'],order,data['manifest']['val']['window_keys'])
                row['epoch']=i+1;guard_source(a.expected_sha)
                candidate=commit_epoch(ledger,row,data['manifest']);cp.save(out,model,opt,identity,candidate);ledger=candidate
                atomic_json(out/'metrics.json',ledger)
                print(json.dumps(dict(epoch=i+1,successful_updates=ledger['successful_updates'],additional_successful_updates=ledger['successful_updates']-identity['parent_ledger']['successful_updates'],joint=row['validation']['model']['objective'])),flush=True)
            if sha256(parent_path)!=a.parent_sha256:raise ValueError('Original parent checkpoint changed')
            atomic_json(out/'status.json',dict(state='COMPLETED',completed_epochs=16,additional_epochs=8,
                successful_updates=ledger['successful_updates'],additional_successful_updates=ledger['successful_updates']-identity['parent_ledger']['successful_updates'],source_sha=source))
        except BaseException:
            atomic_json(out/('failure-'+os.environ['SLURM_JOB_ID']+'.json'),dict(traceback=traceback.format_exc(),note='Only complete immutable Epoch boundaries can resume; no auto retry'))
            raise


if __name__=='__main__':main()
