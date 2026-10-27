"""One allocated GPU, two paired B=2 geometry-only runs, 400 updates each."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
import numpy as np
import torch

from scripts.training.v4_runtime import build_model, save_checkpoint, load_checkpoint
from scripts.training.losses import chamfer_samples, nearest_distances
from scripts.training.train import atomic_json, to_device
from scripts.experiments.fixed_window.fixed_fit import digest
from .data import load_initial, make_pair, normalize, action_variant

BASE = '991713649850c557caa58c6c7cd62cf4096e6ed6'
VERSION = 'v4-paired-action-identifiability-v1'
GROUPS = {
    'motion': lambda n:n.startswith('motion'),
    'controller': lambda n:n.startswith('conditions.controller'),
    'hand': lambda n:n.startswith('conditions.hand'),
    'attention_score': lambda n:n.startswith('cross.') and any(s in n for s in ('.geometry_score','.pairwise.','.q.','.k.')),
    'gs_encoder': lambda n:n.startswith(('gs_encoder','local')),
}


def stats(model, before=None):
    result = {}
    norm = lambda t: float(t.detach().double().square().sum().sqrt())
    for label, predicate in GROUPS.items():
        ps = [(n,p) for n,p in model.named_parameters() if predicate(n)]
        row = dict(gradient_norm=sum(norm(p.grad)**2 for _,p in ps if p.grad is not None)**.5,
                   missing_gradients=sum(p.grad is None for _,p in ps), parameter_norm=sum(norm(p)**2 for _,p in ps)**.5)
        if before is not None:
            row['update_norm'] = sum(norm(p-before[n])**2 for n,p in ps)**.5
            row['relative_update'] = row['update_norm']/max(sum(norm(before[n])**2 for n,_ in ps)**.5,1e-30)
        result[label] = row
    return result


def update(model, opt, batch):
    opt.zero_grad(set_to_none=True)
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    pred = model(batch['inputs'])
    losses = chamfer_samples(pred['xyz'], batch, 512)
    loss = losses.mean()  # Both A and B contribute on EVERY successful update.
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite loss/gradient before optimizer step')
    raw = stats(model)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    clipped = stats(model)
    opt.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()): raise FloatingPointError('Nonfinite parameter')
    return dict(loss=float(loss.detach()), branch_cd_normalized=losses.detach().tolist(), gradient_norm=grad_norm,
                clipped=grad_norm>1., raw_gradients=raw, clipped_gradients=clipped, actual_updates=stats(model,before))


@torch.no_grad()
def evaluate(model, batch, processor, affected):
    pred, trace = model(batch['inputs'],return_trace=True)
    scale_mm = processor.state_dict()['scale']*1000
    xyz = pred['xyz']; initial=batch['inputs']['gs_xyz']; targets=torch.stack(batch['targets']['target_xyz'],1)
    cd=[]
    for b in range(2):
        frames=[]
        for h in range(10):
            p=xyz[b,h,batch['inputs']['gs_mask'][b]]; t=targets[b,h,batch['targets']['target_mask'][h][b]]
            frames.append(float((nearest_distances(p,t).mean()+nearest_distances(t,p).mean())*.5)*scale_mm)
        cd.append(frames)
    delta=(xyz-initial[:,None])*scale_mm; target_delta=(targets-initial[:,None])*scale_mm
    rms=lambda v:v.square().sum(-1).mean(-1).sqrt()
    z=delta[:,:,affected,2].mean(-1); target_z=target_delta[:,:,affected,2].mean(-1)
    att=[]
    for step in trace['steps']:
        a=step['alpha']
        att.append([dict(null_by_head=a[b,trace['field'].anchor_mask[b],:,-1].mean(0).tolist(),
                         controllers_by_head=a[b,trace['field'].anchor_mask[b],:,:60].mean(0).tolist()) for b in range(2)])
    result=dict(cd_mm=cd, branch_mean_cd_mm=[sum(v)/10 for v in cd], mean_cd_mm=sum(sum(v) for v in cd)/20,
                predicted_separation_mm=rms((xyz[0]-xyz[1])*scale_mm).tolist(),
                target_separation_mm=rms((targets[0]-targets[1])*scale_mm).tolist(),
                affected_signed_z_mm=z.tolist(), target_affected_signed_z_mm=target_z.tolist(),
                affected_signed_separation_z_mm=(z[0]-z[1]).tolist(),
                target_affected_signed_separation_z_mm=(target_z[0]-target_z[1]).tolist(),
                motion_rms_mm=rms(delta).tolist(), target_motion_rms_mm=rms(target_delta).tolist(),
                paired_point_rmse_mm=rms((xyz-targets)*scale_mm).tolist(),
                first_prediction_equal=bool(torch.equal(xyz[0,0],xyz[1,0])), attention=att)
    if not result['first_prediction_equal']: raise AssertionError('First identical conditions produced different states')
    return result, pred


def checkpoint(path, model,opt,n,config,processor,ledger):
    save_checkpoint(path,model,opt,n,source_sha=config['source_sha'],processor_state=processor.state_dict(),
                    run_state=ledger,replace_existing=path.exists())


def export_npz(path,pred,batch,processor):
    initial=processor.denormalize_batch(batch)
    scale=processor.state_dict()['scale']; center=np.array(processor.state_dict()['center'])
    np.savez_compressed(path, initial_xyz=initial['inputs']['gs_xyz'].cpu().numpy(),
                        initial_attr=initial['inputs']['gs_attr'].cpu().numpy(),
                        predicted_xyz=pred['xyz'].detach().cpu().numpy()*scale+center,
                        target_xyz=np.stack([t.cpu().numpy() for t in initial['targets']['target_xyz']],1),
                        frame_ids=initial['meta']['frame_ids'].cpu().numpy())


def cuda_smoke(directory,batch,processor,affected,config):
    directory.mkdir()
    torch.manual_seed(42); model=build_model('v4_stage1').cuda(); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    records=[]
    for _ in range(3): records.append(update(model,opt,batch))
    # A zero-initialized motion head legitimately blocks upstream gradient on update1.
    for g in GROUPS:
        if not any(r['raw_gradients'][g]['gradient_norm']>0 for r in records[1:]):
            raise AssertionError(f'No nondegenerate CUDA gradient for {g}')
    metrics,pred=evaluate(model,batch,processor,affected)
    p,t=model(batch['inputs'],return_trace=True)
    gradients=torch.autograd.grad(p['xyz'][:,-1].square().mean(),[s['state'] for s in t['steps']],allow_unused=True)
    if any(g is None or not torch.isfinite(g).all() for g in gradients): raise AssertionError('AR graph missing/nonfinite')
    cp=directory/'smoke.pt'
    save_checkpoint(cp,model,opt,3,source_sha=config['source_sha'],processor_state=processor.state_dict(),run_state={'run_id':str(directory),'experiment':config})
    clone=build_model('v4_stage1').cuda(); co=torch.optim.AdamW(clone.parameters(),lr=1e-4,weight_decay=1e-4)
    state=load_checkpoint(cp,clone,co,processor_state=processor.state_dict()); assert state['successful_steps']==3
    with torch.no_grad(): torch.testing.assert_close(clone(batch['inputs'])['xyz'],pred['xyz'],rtol=0,atol=0)
    # Verify one subsequent update after restore, including optimizer moments.
    a=update(model,opt,batch); b=update(clone,co,batch)
    for k,v in model.state_dict().items(): torch.testing.assert_close(v,clone.state_dict()[k],rtol=0,atol=0)
    report=dict(passed=True,updates=records,shape=list(pred['xyz'].shape),checkpoint_restore_exact=True,
                next_optimizer_update_exact=True,ar_state_gradients=[dict(is_none=False,norm=float(g.double().norm())) for g in gradients],
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,python=platform.python_version(),
                allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved())
    atomic_json(directory/'acceptance.json',report)
    print('CUDA_SMOKE_PASSED',flush=True)


def train(directory,mode,batch,processor,affected,config,expected_initial_hash=None):
    directory.mkdir()
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model=build_model('v4_stage1').cuda(); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    initial_hash=digest(model.state_dict())
    if expected_initial_hash is not None and initial_hash!=expected_initial_hash: raise AssertionError('Unpaired initialization')
    training_batch=action_variant(batch,'real' if mode=='conditioned' else 'static')
    ledger=dict(run_id=str(directory.resolve()),experiment_version=VERSION,config=config,mode=mode,
                model_config=model.config,initial_parameter_sha256=initial_hash,updates=[],evaluations=[],best_update=0)
    begin=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for n in range(401):
        if n%20==0:
            row,pred=evaluate(model,training_batch,processor,affected); row['update']=n
            improved=not ledger['evaluations'] or row['mean_cd_mm']<min(r['mean_cd_mm'] for r in ledger['evaluations'])
            ledger['evaluations'].append(row)
            if improved: ledger['best_update']=n
            if n==0: checkpoint(directory/'initial.pt',model,opt,n,config,processor,ledger)
            if improved: checkpoint(directory/'best_diagnostic.pt',model,opt,n,config,processor,ledger)
            checkpoint(directory/'last.pt',model,opt,n,config,processor,ledger)
            atomic_json(directory/'metrics.json',ledger)
            print(json.dumps(dict(mode=mode,update=n,cd_A=row['branch_mean_cd_mm'][0],cd_B=row['branch_mean_cd_mm'][1],separation_h10=row['predicted_separation_mm'][-1])),flush=True)
        if n==400: break
        started=time.perf_counter(); record=update(model,opt,training_batch)
        record.update(update=n+1,seconds=time.perf_counter()-started); ledger['updates'].append(record)
    export_npz(directory/'last_predictions.npz',pred,training_batch,processor)
    load_checkpoint(directory/'best_diagnostic.pt',model,processor_state=processor.state_dict())
    best,pred=evaluate(model,training_batch,processor,affected)
    export_npz(directory/'best_predictions.npz',pred,training_batch,processor)
    ledger['perturbations']={}
    # Probe both models with the same real / static / half / swapped inputs.
    real_metrics,real_pred=evaluate(model,batch,processor,affected)
    for label in ('real','static','half','swap'):
        altered=action_variant(batch,label)
        m,pr=evaluate(model,altered,processor,affected)
        m['output_change_rms_mm']=((pr['xyz']-real_pred['xyz'])*processor.state_dict()['scale']*1000).square().sum(-1).mean(-1).sqrt().tolist()
        if label=='swap':
            error=float((pr['xyz']-real_pred['xyz'].flip(0)).abs().max())
            if error>1e-7: raise AssertionError('Action swap inconsistency')
            m['swap_max_abs_normalized']=error
            m['cd_against_swapped_gt_mm']=list(reversed(real_metrics['branch_mean_cd_mm']))
        ledger['perturbations'][label]=m
    ledger.update(successful_updates=400,elapsed_seconds=time.perf_counter()-begin,
                  allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved(),
                  checkpoints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.pt')})
    atomic_json(directory/'metrics.json',ledger)
    with (directory/'training.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['update','loss','CD_A_normalized','CD_B_normalized','gradient_norm','clipped','seconds'])
        for r in ledger['updates']: writer.writerow([r['update'],r['loss'],*r['branch_cd_normalized'],r['gradient_norm'],r['clipped'],r['seconds']])
    return ledger


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True); parser.add_argument('--output',required=True); parser.add_argument('--expected-sha',required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('Allocated Slurm CUDA required')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if sha!=args.expected_sha or subprocess.check_output(['git','status','--porcelain'],text=True).strip(): raise RuntimeError('Source drift')
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        config=dict(version=VERSION,source_sha=sha,base_sha=BASE,seed=42,batch=2,horizon=10,prediction_interval=1,
                    optimizer='AdamW',lr=1e-4,weight_decay=1e-4,clip_norm=1.,updates_per_model=400,
                    diagnostic_loss='mean(CD_A, CD_B), equal H10 unsquared symmetric CD in original normalization',
                    formal_v4_loss_unchanged=True,selection='lowest mean A/B CD among updates0,20,...400',
                    modes=['conditioned','unconditioned'],job_id=os.environ['SLURM_JOB_ID'])
        initial,source=load_initial(args.root)
        raw,weights,affected,manifest=make_pair(initial,source['timestamps_us'],source)
        batch,processor=normalize(raw,build_model('v4_stage1'))
        config['data_sha256']=manifest['data_sha256']
        atomic_json(out/'config.json',config); atomic_json(out/'data_manifest.json',manifest)
        torch.save(dict(raw=raw,weights=weights,affected=affected),out/'frozen_data.pt')
        batch=to_device(batch,'cuda'); affected=affected.cuda()
        cuda_smoke(out/'smoke',batch,processor,affected,config)
        torch.cuda.empty_cache()
        results={}
        results['conditioned']=train(out/'conditioned','conditioned',batch,processor,affected,config)
        torch.cuda.empty_cache()
        results['unconditioned']=train(out/'unconditioned','unconditioned',batch,processor,affected,config,
                                       results['conditioned']['initial_parameter_sha256'])
        atomic_json(out/'results.json',results)
        from .report import write_report
        write_report(out)
        atomic_json(out/'status.json',dict(state='COMPLETED',source_sha=sha,successful_updates={'conditioned':400,'unconditioned':400}))
    except BaseException:
        atomic_json(out/'failure.json',dict(source_sha=sha,traceback=traceback.format_exc()))
        raise


if __name__=='__main__': main()
