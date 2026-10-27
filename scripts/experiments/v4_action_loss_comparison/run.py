"""Two fresh 400-update runs, identical V1 frozen data; only the loss changes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
import torch
from scripts.experiments.v4_action_identifiability.data import normalize, action_variant
from scripts.experiments.v4_action_identifiability.run import stats, evaluate, export_npz, GROUPS
from scripts.experiments.fixed_window.fixed_fit import digest
from scripts.training.losses import chamfer_samples
from scripts.training.train import atomic_json, to_device
from scripts.training.v4_runtime import build_model, save_checkpoint, load_checkpoint

BASE='c9a74d2e9a2beae0f4d1b5a9afb76d94a9d74854'
FROZEN_SHA='b6f8c936006e59c3196f3aee461bc42a5ead7ef6ff5b465501da40645fc87ab7'


def mse_by_horizon(xyz,batch):
    """[B,H]: sum XYZ squares, mean valid points. No nearest-neighbor matching."""
    keep=batch['inputs']['gs_mask']
    rows=[]
    for h,(target,valid) in enumerate(zip(batch['targets']['target_xyz'],batch['targets']['target_mask'])):
        if target.shape!=xyz[:,h].shape or not torch.equal(valid,keep):
            raise ValueError('Correspondence MSE needs the identical template point order and mask')
        if not keep.any(1).all(): raise ValueError('Empty correspondence sample')
        # Remove invalid values BEFORE subtraction/squaring so padding cannot poison gradients.
        error=torch.where(keep[...,None],xyz[:,h],0)-torch.where(keep[...,None],target,0)
        rows.append(error.square().sum(-1).sum(-1)/keep.sum(-1))
    if len(rows)!=xyz.shape[1]: raise ValueError('Target horizon mismatch')
    return torch.stack(rows,1)


def loss_samples(pred,batch,mode):
    if mode=='cd': return chamfer_samples(pred['xyz'],batch,512)
    if mode=='mse': return mse_by_horizon(pred['xyz'],batch).mean(1)
    raise ValueError(mode)


def update(model,opt,batch,mode):
    opt.zero_grad(set_to_none=True)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    pred=model(batch['inputs']); branch=loss_samples(pred,batch,mode);loss=branch.mean()
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite loss/gradient')
    raw=stats(model)
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    clipped=stats(model);opt.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()): raise FloatingPointError('Nonfinite parameter')
    return dict(loss=float(loss.detach()),raw_loss=float(loss.detach()),loss_multiplier=1.,
                branch_raw_loss=branch.detach().tolist(),gradient_norm=norm,clipped=norm>1.,
                raw_gradients=raw,clipped_gradients=clipped,actual_updates=stats(model,before))


@torch.no_grad()
def measure(model,batch,processor,affected,mode):
    row,pred=evaluate(model,batch,processor,affected)
    mse=mse_by_horizon(pred['xyz'],batch);scale_mm=processor.state_dict()['scale']*1000
    row.update(mse_normalized_by_horizon=mse.tolist(),branch_mse_normalized=mse.mean(1).tolist(),
               branch_rmse_mm=(mse.mean(1).sqrt()*scale_mm).tolist(),rmse_mm_by_horizon=(mse.sqrt()*scale_mm).tolist(),
               raw_mse_normalized=float(mse.mean()),raw_cd_normalized=row['mean_cd_mm']/scale_mm,
               training_loss=float(mse.mean()) if mode=='mse' else row['mean_cd_mm']/scale_mm)
    # Preserve per-head sign/magnitude, separately over all and affected-region anchors.
    _,tr=model(batch['inputs'],return_trace=True)
    field=tr['field'];rows=[]
    for step in tr['steps']:
        branches=[]
        for b in range(len(batch['inputs']['gs_xyz'])):
            subsets={'all':field.anchor_mask[b], 'affected':affected[field.anchor_indices[b]]&field.anchor_mask[b]}
            result={}
            for name,mask in subsets.items():
                if not mask.any(): result[name]=dict(count=0);continue
                alpha=step['alpha'][b,mask];motion=step['tool_motion'][b,mask].reshape(-1,4,3)*scale_mm
                result[name]=dict(count=int(mask.sum()),null_by_head=alpha[...,-1].mean(0).tolist(),
                    controller_mass_by_head=alpha[...,:60].sum(-1).mean(0).tolist(),
                    motion_xyz_mm_by_head=motion.mean(0).tolist(),
                    cumulative_control_z_mm=float(step['cumulative_d'][b,mask,2].mean()*scale_mm))
            branches.append(result)
        rows.append(branches)
    row['head_motion']=rows
    return row,pred


def read_frozen(v1):
    v1=Path(v1)
    frozen=v1/'frozen_data.pt'
    file_sha=hashlib.sha256(frozen.read_bytes()).hexdigest()
    if file_sha!=FROZEN_SHA: raise ValueError('V1 frozen-data file hash changed')
    state=torch.load(frozen,map_location='cpu',weights_only=False)
    manifest=json.loads((v1/'data_manifest.json').read_text())
    v1_config=json.loads((v1/'config.json').read_text())
    metrics=json.loads((v1/'conditioned/metrics.json').read_text())
    if v1_config['source_sha']!=BASE or digest(state['raw'])!=manifest['data_sha256']:
        raise ValueError('V1 source/data identity failed')
    if hashlib.sha256(state['weights'].numpy().tobytes()).hexdigest()!=manifest['weights_sha256']:
        raise ValueError('V1 deformation weights changed')
    raw=state['raw']
    changed=[k for k,v in raw['inputs'].items() if isinstance(v,torch.Tensor) and not torch.equal(v[0],v[1])]
    if set(changed)!={'action_state','controller_points'}: raise ValueError('Non-action branch identity')
    hashes=dict(frozen_file_sha256=file_sha,raw_sha256=digest(raw),
        input_sha256=digest(raw['inputs']),target_sha256=digest(raw['targets']),
        mask_sha256=digest({k:v for k,v in raw['inputs'].items() if 'mask' in k or 'valid' in k or k=='operation_hands'}),
        field_hashes={k:digest(v) for k,v in raw['inputs'].items()},
        v1_initial_model_sha256=metrics['initial_parameter_sha256'],
        v1_checkpoint_hashes=metrics['checkpoints'],source_manifest=manifest)
    return state,hashes


def save(path,model,opt,n,processor,ledger):
    save_checkpoint(path,model,opt,n,source_sha=ledger['config']['source_sha'],processor_state=processor.state_dict(),
                    run_state=ledger,replace_existing=path.exists())


def train(out,mode,batch,processor,affected,config,initial_hash):
    out.mkdir()
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=build_model('v4_stage1').cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    if digest(model.state_dict())!=initial_hash: raise AssertionError('Unpaired model initialization')
    ledger=dict(run_id=str(out),config=config,mode=mode,model_config=model.config,initial_parameter_sha256=initial_hash,
                updates=[],evaluations=[],best_update=0)
    begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for n in range(401):
        if n%20==0:
            row,pred=measure(model,batch,processor,affected,mode);row['update']=n
            improved=not ledger['evaluations'] or row['training_loss']<min(v['training_loss'] for v in ledger['evaluations'])
            ledger['evaluations'].append(row)
            if n==0:save(out/'initial.pt',model,opt,n,processor,ledger)
            if improved:ledger['best_update']=n;save(out/'lowest_loss.pt',model,opt,n,processor,ledger)
            save(out/'last.pt',model,opt,n,processor,ledger)
            atomic_json(out/'metrics.json',ledger)
            print(json.dumps(dict(mode=mode,update=n,loss=row['training_loss'],cd=row['branch_mean_cd_mm'],rmse=row['branch_rmse_mm'],near_z=[v[-1] for v in row['affected_signed_z_mm']])),flush=True)
        if n==400:break
        start=time.perf_counter();r=update(model,opt,batch,mode);r.update(update=n+1,seconds=time.perf_counter()-start);ledger['updates'].append(r)
    export_npz(out/'last_predictions.npz',pred,batch,processor)
    ledger['probes']={}
    for label,cp in [('last','last.pt'),('best','lowest_loss.pt')]:
        state=load_checkpoint(out/cp,model,processor_state=processor.state_dict())
        real,pred=measure(model,batch,processor,affected,mode)
        if label=='best':export_npz(out/'best_predictions.npz',pred,batch,processor)
        probes={}
        for name in ('real','static','half','swap'):
            if name=='real':r,pr=real,pred
            else:r,pr=measure(model,action_variant(batch,name),processor,affected,mode)
            r['output_change_rms_mm']=((pr['xyz']-pred['xyz'])*processor.state_dict()['scale']*1000).square().sum(-1).mean(-1).sqrt().tolist()
            if name=='swap':
                r['swap_max_abs_normalized']=float((pr['xyz']-pred['xyz'].flip(0)).abs().max())
                if r['swap_max_abs_normalized']>1e-7:raise AssertionError('Swap mismatch')
            probes[name]=r
        ledger['probes'][label]=dict(update=state['successful_steps'],conditions=probes)
    ledger.update(successful_updates=400,elapsed_seconds=time.perf_counter()-begin,
        allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved(),
        checkpoint_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.pt')})
    atomic_json(out/'metrics.json',ledger)
    with (out/'training.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['update','raw_loss','multiplier','training_loss','branch_A_raw_loss','branch_B_raw_loss','gradient_norm','clipped','seconds'])
        for r in ledger['updates']:w.writerow([r['update'],r['raw_loss'],1,r['loss'],*r['branch_raw_loss'],r['gradient_norm'],r['clipped'],r['seconds']])
    return ledger


def smoke(out,batch,processor,config,initial_hash):
    out.mkdir();reports={}
    for mode in ('cd','mse'):
        torch.manual_seed(42);model=build_model('v4_stage1').cuda()
        assert digest(model.state_dict())==initial_hash
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        rows=[update(model,opt,batch,mode) for _ in range(3)]
        if any(not any(r['raw_gradients'][g]['gradient_norm']>0 for r in rows[1:]) for g in GROUPS):
            raise AssertionError('Missing nondegenerate gradient')
        ledger=dict(run_id=str(out/mode),config=config)
        save(out/f'{mode}.pt',model,opt,3,processor,ledger)
        clone=build_model('v4_stage1').cuda();co=torch.optim.AdamW(clone.parameters(),lr=1e-4,weight_decay=1e-4)
        load_checkpoint(out/f'{mode}.pt',clone,co,processor_state=processor.state_dict())
        with torch.no_grad():torch.testing.assert_close(model(batch['inputs'])['xyz'],clone(batch['inputs'])['xyz'],rtol=0,atol=0)
        update(model,opt,batch,mode);update(clone,co,batch,mode)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,clone.state_dict()[k],rtol=0,atol=0)
        reports[mode]=dict(passed=True,updates=rows,checkpoint_and_next_step_exact=True)
    atomic_json(out/'acceptance.json',dict(passed=True,groups=reports,gpu=torch.cuda.get_device_name(),torch=torch.__version__,python=platform.python_version()))
    print('PAIRED_CUDA_SMOKE_PASSED',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('v1','output','expected-sha'):p.add_argument('--'+k,required=True)
    args=p.parse_args()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if sha!=args.expected_sha or subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Source drift')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm GPU required')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        frozen,hashes=read_frozen(args.v1)
        torch.manual_seed(42);model=build_model('v4_stage1');initial_hash=digest(model.state_dict())
        if initial_hash!=hashes['v1_initial_model_sha256']:raise AssertionError('V1 initialization differs')
        batch,processor=normalize(frozen['raw'],model)
        hashes['normalized_inputs_sha256']=digest(batch['inputs']);hashes['normalized_targets_sha256']=digest(batch['targets'])
        config=dict(source_sha=sha,base_sha=BASE,v1=str(Path(args.v1).resolve()),horizon=10,prediction_interval=1,seed=42,
            batch_size=2,optimizer='AdamW',lr=1e-4,weight_decay=1e-4,clip_norm=1.,updates=400,eval_every=20,
            mse_multiplier=1.,scaling_rule='none: fixed coefficient 1 before training',mse='sum XYZ squares, mean valid points, mean horizons, mean branches; normalized coordinates',
            cd='V1 unsquared symmetric Chamfer',selection='minimum own training objective over 0,20,...,400',job_id=os.environ['SLURM_JOB_ID'])
        atomic_json(out/'config.json',config);atomic_json(out/'data_identity.json',hashes)
        batch=to_device(batch,'cuda');affected=frozen['affected'].cuda()
        smoke(out/'smoke',batch,processor,config,initial_hash);torch.cuda.empty_cache()
        results={}
        for mode in ('cd','mse'):
            results[mode]=train(out/mode,mode,batch,processor,affected,config,initial_hash);torch.cuda.empty_cache()
        # Verify immutable V1 frozen data and checkpoints AFTER this experiment.
        if read_frozen(args.v1)[1]!= {k:v for k,v in hashes.items() if not k.startswith('normalized_')}:raise AssertionError('V1 evidence changed')
        for name,value in hashes['v1_checkpoint_hashes'].items():
            if hashlib.sha256((Path(args.v1)/'conditioned'/name).read_bytes()).hexdigest()!=value:raise AssertionError('V1 checkpoint changed')
        previous=json.loads((Path(args.v1)/'conditioned/metrics.json').read_text())
        differences=[abs(a['mean_cd_mm']-b['mean_cd_mm']) for a,b in zip(previous['evaluations'],results['cd']['evaluations'])]
        atomic_json(out/'cd_reproduction.json',dict(max_mean_cd_difference_mm=max(differences),differences_mm=differences,
             note='Same initialization/data/optimizer; report numeric reproduction, never adjust results to match.'))
        atomic_json(out/'results.json',results)
        from .report import write_report
        write_report(out)
        atomic_json(out/'status.json',dict(state='COMPLETED',updates={'cd':400,'mse':400},source_sha=sha))
    except BaseException:
        atomic_json(out/'failure.json',dict(traceback=traceback.format_exc(),source_sha=sha));raise


if __name__=='__main__':main()
