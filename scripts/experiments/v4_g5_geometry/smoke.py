"""Gated Val100 equivalence -> exactly four head updates -> fixed Val100 audit."""
import argparse
import copy
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.training.train import atomic_json,to_device
from scripts.training.v4_runtime import prepare_batch
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.protocol import configuration,preflight
from scripts.experiments.v4_g5_signed_local.formal_checkpoint import load_for_evaluation
from scripts.experiments.v4_g5_signed_local.diagnostics import sha256,state_hash,finite_tree,write_csv
from scripts.experiments.v4_g4_global_motion.smoke import same_tree
from .model import GeometryAdapter
from . import checkpoint
from .evaluation import (CachedRenderer,equality,assert_metrics_equal,summarize,objective_record,
    geometry_rows,save_images,REFERENCE,TOLERANCE)

BASE='f978e0a29b6f0f0b6097c40684b949797a2f5d7b'
BACKBONE_HASH='067984c4c286306094efeff2f28ed57e3dcdd0c333972ecc6feb4b982ee3a40c'
ORDER=((2,141),(7,105),(2,141),(7,105))
PICTURES={(3,135),(3,245),(6,110),(7,105)}


def assigned_gpu_idle():
    # Slurm's physical index can differ from the isolated NVML index.
    visible=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True)
    uuids=[v.strip() for v in visible.splitlines() if v.strip()]
    if len(uuids)!=1 or not uuids[0].startswith('GPU-'):
        raise RuntimeError('Ambiguous single-GPU allocation namespace')
    processes=subprocess.check_output(['nvidia-smi','-i',uuids[0],'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    foreign=[int(v.strip()) for v in processes.splitlines() if v.strip() and int(v.strip())!=os.getpid()]
    if foreign:raise RuntimeError('Allocated GPU has existing compute processes; preserving them: '+str(foreign))
    return dict(uuid=uuids[0],foreign_compute_pids=foreign)


def source_guard(expected):
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if sha!=expected or subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Clean committed pinned Worktree required')
    changed=subprocess.check_output(['git','diff','--name-only',BASE,'HEAD'],text=True).splitlines()
    if any(not (p.startswith('scripts/experiments/v4_g5_geometry/') or p.startswith('scripts/tests/training/test_v4_g5_geometry')) for p in changed):
        raise ValueError('Unexpected changes outside independent geometry adapter: '+str(changed))
    return sha,{p:sha256(p) for p in changed}


def prepared(raw,processor,model,ep,start):
    batch=prepare_batch(raw,processor,model)
    x,y=batch['inputs'],batch['targets']
    if batch['meta']['frame_ids'].tolist()!=[list(range(start,start+16))]:raise AssertionError('Wrong H15 frames')
    if x['condition_frame_ids'].tolist()!=[list(range(start,start+15))]:raise AssertionError('Wrong causal action times')
    if x['operation_hands'].tolist()!=[[ep>=5,True]]:raise AssertionError('Wrong hand policy')
    if tuple(y['target_depth'].shape)!=(1,15,3,720,1280):raise AssertionError('Wrong renderer camera resolution')
    if not y['target_depth_valid'].flatten(3).any(-1).all():raise AssertionError('Missing camera/target observations')
    return to_device(batch,'cuda')


def flush(out,tables):
    for name,rows in tables.items():write_csv(out/'results'/f'{name}.csv',rows)


@torch.no_grad()
def evaluate_pass(stage,raw,adapter,processor,renderer,out,tables,baseline):
    adapter.eval();counts={3:0,6:0}
    for index,raw_batch in enumerate(raw):
        ep=int(raw_batch['meta']['episode_id'][0]);start=int(raw_batch['meta']['frame_ids'][0,0])
        if ep not in counts:raise AssertionError('Test/train leaked into validation')
        counts[ep]+=1;batch=prepared(raw_batch,processor,adapter,ep,start);x=batch['inputs']
        original=adapter.backbone(x)
        motion,trace,features=adapter.motion_forward(x)
        equality(original,motion,x['gs_mask'],all_attributes=True)
        modes=(('original',None),('geometry_off','geometry_off'),('geometry_on_zero','geometry_on_zero')) if stage=='baseline' else (
            ('geometry_4updates','geometry_on'),('geometry_off_after','geometry_off'))
        for label,mode in modes:
            pred,geo=(original,{}) if mode is None else adapter.render_from_motion(motion,trace,features,x['gs_mask'],mode)
            match=equality(original,pred,x['gs_mask'],all_attributes=label!='geometry_4updates')
            _,row,horizons,cameras=objective_record(pred,batch,processor,renderer,mode=label,split='val',episode=ep,start=start)
            tables['window_metrics'].append(row);tables['horizons'].extend(horizons);tables['camera_metrics'].extend(cameras)
            if label=='original':baseline[ep,start]=dict(row)
            elif label!='geometry_4updates':assert_metrics_equal(baseline[ep,start],row)
            elif abs(row['cd_mm']-baseline[ep,start]['cd_mm'])>TOLERANCE['cd_mm']:raise AssertionError('Center CD changed')
            tables['baseline_equivalence'].append(dict(stage=stage,mode=label,episode=ep,start=start,**match,
                **{'delta_'+k:row[k]-baseline[ep,start][k] for k in REFERENCE}))
            if geo:
                tables['geometry_statistics'].extend(geometry_rows(pred,motion,geo,batch,mode=label,split='val',episode=ep,start=start))
            if (ep,start) in PICTURES and label in ('original','geometry_on_zero','geometry_4updates'):
                save_images(renderer,out/'images',f'ep{ep}_start{start}_{label}',save_gt=label=='original')
                if geo:
                    np.savez_compressed(out/'figures'/f'ep{ep}_start{start}_{label}_attributes.npz',
                        xyz=processor.denormalize_xyz(original['xyz'],x['gs_mask'][:,None].expand(-1,15,-1)).cpu().numpy(),
                        mask=x['gs_mask'].cpu().numpy(),log_delta=geo['log_scale_delta'].cpu().numpy(),
                        rotation_vector=geo['rotation_vector'].cpu().numpy())
            del pred,geo
        renderer.clear();del batch,original,motion,trace,features,raw_batch
        if (index+1)%10==0:
            flush(out,tables);atomic_json(out/'status.json',dict(stage=stage,completed_windows=index+1,updates=0 if stage=='baseline' else 4))
            print(json.dumps(dict(stage=stage,completed_windows=index+1)),flush=True)
    if counts!={3:55,6:45}:raise AssertionError('Incomplete Val100')
    modes=['original','geometry_off','geometry_on_zero'] if stage=='baseline' else ['geometry_4updates','geometry_off_after']
    for mode in modes:
        rows=summarize(tables['window_metrics'],mode);tables['val_summary'].extend(rows)
        if stage=='baseline':assert_metrics_equal(REFERENCE,rows[-1])
    flush(out,tables)


def gradients(model,before):
    result={}
    for group,names in [('scale_output',('4.weight','4.bias')),('rotation_output',('4.weight','4.bias')),
                        ('hidden',('0.weight','0.bias','2.weight','2.bias'))]:
        grad=update=0.
        for name,p in model.head.named_parameters():
            if name not in names:continue
            selection=slice(0,3) if group=='scale_output' else slice(3,6) if group=='rotation_output' else slice(None)
            if p.grad is None:raise AssertionError('Missing geometry gradient '+name)
            grad+=float(p.grad[selection].double().square().sum())
            update+=float((p.detach()-before[name])[selection].double().square().sum())
        result[group]=dict(gradient_norm=grad**.5,update_norm=update**.5)
    return result


def train_four(raw,adapter,processor,renderer,out,tables,identity,backbone_hash):
    # There is no resume/extra-step CLI: a failed run is never silently restarted.
    opt=torch.optim.AdamW(adapter.head.parameters(),lr=1e-4,weight_decay=1e-4)
    checkpoint.load(out/'checkpoints'/'geo_init.pt',adapter,identity=identity)
    if {id(p) for g in opt.param_groups for p in g['params']}!={id(p) for p in adapter.head.parameters()}:
        raise AssertionError('Optimizer is not head-only')
    for step,(ep,start) in enumerate(ORDER,1):
        ds=raw.dataset;raw_batch=collate_fn([ds[ds.windows.index((ep,start))]])
        batch=prepared(raw_batch,processor,adapter,ep,start);x=batch['inputs'];adapter.train()
        motion,trace,features=adapter.motion_forward(x)
        opt.zero_grad(set_to_none=True);before={n:p.detach().clone() for n,p in adapter.head.named_parameters()}
        begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        pred,geo=adapter.render_from_motion(motion,trace,features,x['gs_mask'])
        equality(motion,pred,x['gs_mask'])
        loss,row,_,_=objective_record(pred,batch,processor,renderer,mode=f'update{step}_before',split='train',episode=ep,start=start,curves=False)
        loss.backward();finite_tree([p.grad for p in adapter.head.parameters()]);raw_grad=gradients(adapter,before)
        norm=float(torch.nn.utils.clip_grad_norm_(adapter.head.parameters(),1.))
        opt.step()  # The only optimizer.step in this bounded four-update runner.
        atomic_json(out/'update_counter.json',dict(successful_updates=step,order=ORDER[:step]))
        finite_tree((adapter.state_dict(),opt.state_dict()))
        evidence=gradients(adapter,before)
        for group in evidence:evidence[group]['raw_gradient_norm']=raw_grad[group]['gradient_norm']
        for group in ('scale_output','rotation_output'):
            if evidence[group]['raw_gradient_norm']<=0 or evidence[group]['update_norm']<=0:
                raise AssertionError('Inactive real gsplat '+group)
        if step>1 and evidence['hidden']['raw_gradient_norm']<=0:raise AssertionError('Geometry hidden layers inactive after seed update')
        if state_hash(adapter.backbone.state_dict())!=backbone_hash or any(p.grad is not None for p in adapter.backbone.parameters()):
            raise AssertionError('Frozen backbone changed')
        del loss,pred,geo;renderer.rendered=None
        with torch.no_grad():
            after,geo=adapter.render_from_motion(motion,trace,features,x['gs_mask'])
            equality(motion,after,x['gs_mask'])
            _,post,_,_=objective_record(after,batch,processor,renderer,mode=f'update{step}_after',split='train',episode=ep,start=start,curves=False)
            if post['cd_mm']!=row['cd_mm']:raise AssertionError('CD changed after head update')
            tables['geometry_statistics'].extend(geometry_rows(after,motion,geo,batch,mode=f'update{step}',split='train',episode=ep,start=start))
        torch.cuda.synchronize()
        record=dict(update=step,episode=ep,start=start,raw_gradient_norm=norm,clip_norm=1.,clipped=norm>1,
            seconds=time.perf_counter()-begin,peak_cuda_bytes=torch.cuda.max_memory_allocated(),backbone_state_hash=backbone_hash,
            **{'before_'+k:row[k] for k in REFERENCE},**{'after_'+k:post[k] for k in REFERENCE})
        for group,values in evidence.items():record.update({group+'_'+k:v for k,v in values.items()})
        tables['smoke_updates'].append(record);flush(out,tables);print(json.dumps(record),flush=True)
        renderer.clear();del batch,motion,trace,features,after,geo,before;gc.collect()
    checkpoint.save(out/'checkpoints'/'geo_smoke_4updates.pt',adapter,opt,identity=identity,updates=4)
    # Strictly reload both model and optimizer, then verify predictions on ep7.
    saved_head=copy.deepcopy(adapter.head.state_dict());saved_opt=copy.deepcopy(opt.state_dict())
    batch=prepared(collate_fn([raw.dataset[raw.dataset.windows.index((7,105))]]),processor,adapter,7,105)
    adapter.eval()
    with torch.no_grad():expected=adapter(batch['inputs'])
    restored=checkpoint.load(out/'checkpoints'/'geo_smoke_4updates.pt',adapter,opt,identity=identity)
    with torch.no_grad():actual=adapter(batch['inputs'])
    if not same_tree(expected,actual) or not same_tree(saved_head,adapter.head.state_dict()) or not same_tree(saved_opt,opt.state_dict()):
        raise AssertionError('Checkpoint strict restore failed')
    atomic_json(out/'results'/'checkpoint_restore.json',dict(passed=True,updates=restored['updates'],prediction_bit_exact=True,optimizer_exact=True))
    del opt,batch,expected,actual,saved_head,saved_opt


@torch.no_grad()
def train_picture(raw,adapter,processor,renderer,out,tables,init_head):
    ds=raw.dataset;ep,start=7,105;batch=prepared(collate_fn([ds[ds.windows.index((ep,start))]]),processor,adapter,ep,start)
    motion,tr,features=adapter.motion_forward(batch['inputs']);final=copy.deepcopy(adapter.head.state_dict())
    for mode in ('original','geometry_on_zero','geometry_4updates'):
        if mode=='original':pred,geo=motion,{}
        else:
            adapter.head.load_state_dict(init_head if mode=='geometry_on_zero' else final,strict=True)
            pred,geo=adapter.render_from_motion(motion,tr,features,batch['inputs']['gs_mask'],
                                              'geometry_on_zero' if mode=='geometry_on_zero' else 'geometry_on')
        equality(motion,pred,batch['inputs']['gs_mask'],all_attributes=mode!='geometry_4updates')
        _,row,horizons,cameras=objective_record(pred,batch,processor,renderer,mode=mode,split='train',episode=ep,start=start)
        tables['window_metrics'].append(row);tables['horizons'].extend(horizons);tables['camera_metrics'].extend(cameras)
        save_images(renderer,out/'images',f'ep{ep}_start{start}_{mode}',save_gt=mode=='original')
        if geo:
            tables['geometry_statistics'].extend(geometry_rows(pred,motion,geo,batch,mode=mode,split='train',episode=ep,start=start))
    adapter.head.load_state_dict(final,strict=True);renderer.clear();flush(out,tables)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','depth-root','g0-reference','checkpoint','output','expected-sha'):p.add_argument('--'+key,required=True)
    args=p.parse_args();out=Path(args.output)
    if not out.is_absolute() or out.exists() or out.parent.name!='v4-g5-geometry-decoder-v1':raise ValueError('New independent absolute output required')
    source,files=source_guard(args.expected_sha)
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Authorized Slurm GPU required')
    resource=assigned_gpu_idle()
    if sha256(args.checkpoint)!=BACKBONE_HASH:raise ValueError('Original checkpoint hash mismatch')
    out.mkdir(parents=True)
    for name in ('results','figures','images','logs','checkpoints'): (out/name).mkdir()
    tables={k:[] for k in ('window_metrics','horizons','camera_metrics','baseline_equivalence','geometry_statistics','val_summary','smoke_updates')}
    start_time=time.perf_counter()
    try:
        torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        config=configuration(args.root,args.depth_root,source,'direct');raw,data=preflight(config,args.g0_reference)
        if any(list(v) not in data['manifest']['train']['window_keys'] for v in ORDER):raise AssertionError('Requested update window not in Train manifest')
        processor=DataProcessor('001-rope');backbone=G5DirectPredictor(horizon=15,prediction_interval=1)
        state=load_for_evaluation(args.checkpoint,backbone,processor.state_dict(),source_sha=BASE,manifest=data['manifest'])
        if state['ledger']['best_epoch']!=8:raise AssertionError('Expected G5 Direct E8 Best')
        backbone_hash=state_hash(backbone.state_dict());del state
        torch.manual_seed(42);adapter=GeometryAdapter(backbone).cuda();adapter.eval()
        init_head=copy.deepcopy(adapter.head.state_dict())
        identity=dict(source_sha=source,backbone_source_sha=BASE,backbone_checkpoint=str(Path(args.checkpoint).resolve()),
            backbone_checkpoint_sha256=BACKBONE_HASH,processor=processor.state_dict(),run_id=str(out.resolve()),seed=42)
        environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
            gsplat=importlib.metadata.version('gsplat'),gpu=torch.cuda.get_device_name(),
            job_id=os.environ.get('SLURM_JOB_ID'),step_id=os.environ.get('SLURM_STEP_ID'),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        if (environment['torch']!='2.7.1+cu128' or environment['cuda']!='12.8'
                or environment['gsplat']!='1.4.0' or '5090' not in environment['gpu']):
            raise RuntimeError('Runtime differs from frozen CUDA/gsplat protocol: '+str(environment))
        environment.update(tf32=False,allocation_resource=resource)
        provenance=dict(identity=identity,environment=environment,command=sys.argv,source_files=files,
            manifest=data,geometry_config=adapter.geometry_config,tolerances=TOLERANCE,reference=REFERENCE,
            optimizer=dict(name='AdamW',lr=1e-4,weight_decay=1e-4,clip_norm=1.,head_only=True),update_order=ORDER,
            parameters=dict(backbone=sum(p.numel() for p in backbone.parameters()),head=sum(p.numel() for p in adapter.head.parameters())),
            original_backbone_state_hash=backbone_hash,loss='CD + RGB + 0.1 Alpha; scale_weight=0 depth_weight=0')
        atomic_json(out/'provenance.json',provenance)
        checkpoint.save(out/'checkpoints'/'geo_init.pt',adapter,None,identity=identity,updates=0)
        renderer=CachedRenderer();baseline={}
        evaluate_pass('baseline',raw['val'],adapter,processor,renderer,out,tables,baseline)
        atomic_json(out/'results'/'baseline_gate.json',dict(passed=True,complete_val_windows=100,tolerances=TOLERANCE))
        train_four(raw['train'],adapter,processor,renderer,out,tables,identity,backbone_hash)
        evaluate_pass('post',raw['val'],adapter,processor,renderer,out,tables,baseline)
        train_picture(raw['train'],adapter,processor,renderer,out,tables,init_head)
        final_hash=state_hash(backbone.state_dict())
        if final_hash!=backbone_hash or sha256(args.checkpoint)!=BACKBONE_HASH:raise AssertionError('Original backbone/checkpoint changed')
        paired=[]
        for r in tables['window_metrics']:
            if r['split']=='val' and r['mode']=='geometry_4updates':
                b=baseline[r['episode'],r['start']]
                paired.append(dict(episode=r['episode'],start=r['start'],**{'delta_'+k:r[k]-b[k] for k in REFERENCE}))
        write_csv(out/'results'/'paired_window_effects.csv',paired)
        original=next(r for r in tables['val_summary'] if r['mode']=='original' and r['episode']=='episode_equal')
        final=next(r for r in tables['val_summary'] if r['mode']=='geometry_4updates' and r['episode']=='episode_equal')
        result=dict(engineering_passed=True,xyz_isolation_passed=True,successful_updates=4,
            render_nonregression=all(final[k]<=original[k]+TOLERANCE[k] for k in REFERENCE),
            original=original,final=final,total_seconds=time.perf_counter()-start_time,backbone_hash_unchanged=True,
            checkpoint_hash_unchanged=True,checkpoints={p.name:dict(path=str(p),sha256=sha256(p)) for p in (out/'checkpoints').glob('*.pt')})
        atomic_json(out/'completion.json',result);atomic_json(out/'status.json',dict(stage='complete',updates=4))
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        flush(out,tables);atomic_json(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
