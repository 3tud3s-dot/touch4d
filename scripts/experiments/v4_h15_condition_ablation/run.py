"""Frozen H15 inference only: no backward, optimizer, parameter updates or test split."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.experiments.full_training.v4_epochs import loader
from scripts.experiments.fixed_window.fixed_fit import digest, geometry_diagnostics
from scripts.training.v4_runtime import build_model, prepare_batch, load_checkpoint
from scripts.training.train import to_device, copy_initial, make_renderer
from scripts.training.losses import nearest_distances, render_samples
from scripts.training.visualize import png_bytes
from .interventions import observe, altered_inputs, fixed_regions, rms_mm

BASE_SHA='60b7e6044a139a92bf7fdad29c262bde437ae2f9'
AUDIT_ROOT=Path('/data1/userdata/tcweng/experiments/touch4d/v4-001-h15-best-visual-v1')
CHECKPOINT_SHA='1fa35ddcc4c62ca359d8491d77bdd7888152a795ffae49b07a8480e190ccd05f'
WINDOWS=[('train',7,105),('val',3,135),('val',6,110)]
MODES=('normal','freeze_action','zero_tool')
LANDMARKS=(1,5,10,15)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(1024*1024),b''):h.update(v)
    return h.hexdigest()


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value


def save_tensor_bundle(path,value):
    # Complete original batch, including depth/masks, not a new Dataset.
    with gzip.open(path,'xb',compresslevel=6) as f:torch.save(cpu(value),f)


def metadata():
    return dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        command=[sys.executable,*sys.argv],python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        job=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name() if torch.cuda.is_initialized() else None,
        cwd=str(Path.cwd()),no_optimizer=True,no_backward=True)


def checked_model():
    provenance=json.loads((AUDIT_ROOT/'checkpoint_provenance.json').read_text(encoding='utf8'))
    checkpoint=AUDIT_ROOT/'best_snapshot.pt'
    if sha(checkpoint)!=CHECKPOINT_SHA or provenance['sha256']!=CHECKPOINT_SHA:raise ValueError('Frozen checkpoint hash mismatch')
    processor=DataProcessor('001-rope');model=build_model('v4_stage1',horizon=15,prediction_interval=1).eval()
    state=load_checkpoint(checkpoint,model,processor_state=processor.state_dict())
    if state['source_sha']!=BASE_SHA or state['run_state']['completed_epochs']!=6 or state['successful_steps']!=9084:
        raise ValueError('Wrong epoch/source/updates')
    config=json.loads((AUDIT_ROOT/'training_config.json').read_text(encoding='utf8'))['training']
    if config!=state['run_state']['config']:raise ValueError('Training config not equal checkpoint ledger')
    model.requires_grad_(False)
    return model,processor,state,config,dict(snapshot=str(checkpoint),sha256=CHECKPOINT_SHA,
        original_provenance=provenance,source_sha=state['source_sha'],config=state['config'],
        processor=state['processor'],epoch=6,successful_updates=9084,selection='best validation joint loss')


def batch_for(ds,key,processor,model):
    raw=collate_fn([ds[ds.windows.index(key)]])
    if raw['meta']['frame_ids'][0].tolist()!=list(range(key[1],key[1]+16)):raise ValueError('H15 frame mismatch')
    prepared=prepare_batch(raw,processor,model)
    if not prepared['targets']['target_depth_valid'].flatten(3).any(-1).all():raise ValueError('Missing camera supervision')
    for h in range(15):
        y=prepared['targets'];q=y['target_xyz'][h][0,y['target_mask'][h][0]]
        if not len(q) or not torch.isfinite(q).all():raise ValueError('Invalid target set')
    return raw,prepared


def preflight(root):
    if (root/'status/preflight.json').exists():raise FileExistsError('Preflight already attempted')
    for d in ['status','provenance','logs','windows','figures','source']:(root/d).mkdir(parents=True,exist_ok=True)
    m,p,s,c,ref=checked_model();datasets={k:loader(c,k).dataset for k in ('train','val')}
    for split,ds in datasets.items():
        if [list(k) for k in ds.windows]!=s['run_state']['manifest'][split]['window_keys']:raise ValueError('Split drift')
    manifest=[]
    for split,ep,start in WINDOWS:
        raw,b=batch_for(datasets[split],(ep,start),p,m)
        manifest.append(dict(split=split,episode=ep,start=start,frame_ids=raw['meta']['frame_ids'][0].tolist(),
            condition_frame_ids=b['inputs']['condition_frame_ids'][0].tolist(),
            timestamps_us=raw['meta']['observation_timestamp_us'][0].tolist(),
            initial_count=int(raw['inputs']['gs_mask'].sum()),target_counts=raw['targets']['target_lengths'][0].tolist(),
            operation_hands=raw['inputs']['operation_hands'][0].tolist(),raw_hash=digest(raw),prepared_hash=digest(b)))
    write(root/'provenance/checkpoint.json',ref)
    write(root/'provenance/protocol.json',dict(windows=manifest,training_config=c,base_sha=BASE_SHA,
        intervention_source='0934d8c488ce1da86ddc91a95300a3bb36ffd736/core.py observe, altered_inputs, fixed_regions, rms_mm',
        near_far='Initial valid GS distance to active valid Controller at t; bottom/top 25%, stable sort, fixed for all modes/horizons',
        local_error='Directed predicted fixed-subset to complete GT nearest distance; NOT regional symmetric CD or GT point identity',
        freeze_action='Repeat initial action_state/controller_points; action_valid/controller_mask intersect initial validity; tactile/age/other masks unchanged',
        zero_tool='Temporary forward pre-hook replaces motion input columns 128:140 only; full AR rerun; other current-step fields untouched',
        selected_horizons=LANDMARKS,no_test=True,no_optimizer=True))
    write(root/'status/preflight.json',dict(passed=True,windows=3,execution=metadata()))


def geometric_metrics(pred,batch,normal,regions,scale):
    keep=batch['inputs']['gs_mask'][0];initial=batch['inputs']['gs_xyz'][0]
    diagnostics=geometry_diagnostics(pred,batch,scale)
    rows=[]
    for h in range(pred['xyz'].shape[1]):
        pos=pred['xyz'][0,h];gt=batch['targets']['target_xyz'][h][0,batch['targets']['target_mask'][h][0]]
        p2q=nearest_distances(pos[keep],gt,512)
        cd=(p2q.mean()+nearest_distances(gt,pos[keep],512).mean())*.5*scale*1000
        row=dict(horizon=h+1,cd_mm=float(cd),**diagnostics[h])
        for name,ids in regions.items():
            row[name+'_motion_rms_mm']=rms_mm(pos[ids]-initial[ids],scale*1000)
            row[name+'_change_from_normal_mm']=rms_mm(pos[ids]-normal['xyz'][0,h,ids],scale*1000)
            row[name+'_pred_to_gt_mm']=float(nearest_distances(pos[ids],gt,512).mean())*scale*1000
        rows.append(row)
    return rows


def save_images(directory,mode,images):
    for h in LANDMARKS:
        for v in range(3):
            for kind in ('rgb','alpha'):
                data=images[kind][0,h-1,v].detach().cpu().numpy()
                if data.ndim==2:data=np.repeat(data[...,None],3,-1)
                (directory/f'{mode}_h{h:02}_v{v}_{kind}.png').write_bytes(png_bytes(np.rint(np.clip(data,0,1)*255).astype(np.uint8)))


@torch.inference_mode()
def infer(root):
    if not json.loads((root/'status/preflight.json').read_text(encoding='utf8'))['passed']:raise ValueError('Preflight incomplete')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm CUDA required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,processor,state,config,ref=checked_model();model=model.cuda();before=digest(model.state_dict())
    datasets={s:loader(config,s).dataset for s in ('train','val')};renderer=make_renderer(config)
    frozen=json.loads((root/'provenance/protocol.json').read_text(encoding='utf8'))['windows']
    historical=json.loads((AUDIT_ROOT/'window_metrics.json').read_text(encoding='utf8'));records=[]
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    for split,ep,start in WINDOWS:
        directory=root/'windows'/f'ep{ep}_start{start}';directory.mkdir(exist_ok=False);images_dir=directory/'images';images_dir.mkdir()
        raw,prepared=batch_for(datasets[split],(ep,start),processor,model)
        fm=next(v for v in frozen if v['episode']==ep and v['start']==start)
        if digest(raw)!=fm['raw_hash'] or digest(prepared)!=fm['prepared_hash']:raise ValueError('Data drift')
        save_tensor_bundle(directory/'raw_batch.pt.gz',raw)
        batch=to_device(prepared,'cuda');original_input_hash=digest(batch['inputs'])
        plain=model(batch['inputs'])
        with observe(model) as captured:normal,trace=model(batch['inputs'],return_trace=True)
        if any(not torch.equal(plain[k],normal[k]) for k in plain):raise ValueError('Normal instrumentation changed output')
        regions,_=fixed_regions(batch,trace['field']);target=renderer.render_targets(batch,processor)
        save_images(images_dir,'gt',target)
        all_predictions={};all_traces={};rows=[];mode_meta={}
        for mode in (*MODES,'copy'):
            if mode=='normal':pred,tt,cap=normal,trace,captured
            elif mode=='copy':pred=copy_initial(batch['inputs'],'gs_geometry');tt=cap=None
            else:
                x=altered_inputs(batch['inputs'],'freeze_action') if mode=='freeze_action' else batch['inputs']
                with observe(model,zero_tool=mode=='zero_tool') as cap:pred,tt=model(x,return_trace=True)
                save_tensor_bundle(directory/(mode+'_inputs.pt.gz'),x)
            if any(not torch.isfinite(v).all() or v.requires_grad for v in pred.values()):raise ValueError('Nonfinite/gradient-enabled frozen output')
            render=renderer(pred,batch,processor);save_images(images_dir,mode,render)
            metrics=geometric_metrics(pred,batch,normal,regions,processor.state_dict()['scale'])
            for kind in ('rgb','alpha'):
                _,available,per_h,valid_h=render_samples(render[kind],target[kind],batch['targets']['target_depth_valid'])
                if not available.all() or not valid_h.all():raise ValueError('Missing rendered horizon')
                for h,row in enumerate(metrics):row[kind+'_mae']=float(per_h[0,h])
                # Independent per-camera MAE under the unchanged target mask.
                for h,row in enumerate(metrics):
                    mask=batch['targets']['target_depth_valid'][0,h]
                    diff=(render[kind][0,h]-target[kind][0,h]).abs()
                    if diff.ndim==4:diff=diff.mean(-1)
                    row[kind+'_mae_by_camera']=[float(diff[v][mask[v]].mean()) for v in range(3)]
            rows.extend(dict(mode=mode,**r) for r in metrics)
            all_predictions[mode]=cpu(pred)
            if tt is not None:
                all_traces[mode]=cpu(dict(steps=tt['steps'],motion_inputs=cap['motion_input'],anchor_features=cap['anchor_features']))
                if mode=='zero_tool' and any(v[...,128:140].count_nonzero() for v in cap['motion_input']):raise ValueError('Zero-tool failed')
            mode_meta[mode]=dict(mean_cd_mm=float(np.mean([r['cd_mm'] for r in metrics])),mean_rgb=float(np.mean([r['rgb_mae'] for r in metrics])),mean_alpha=float(np.mean([r['alpha_mae'] for r in metrics])))
            del render
        old=next(r for r in historical if r['episode']==ep and r['start']==start and r['split']==split)
        comparisons={}
        for name,oldname,tol in [('mean_cd_mm','mean_cd_mm',1e-4),('mean_rgb','mean_rgb',1e-6),('mean_alpha','mean_alpha',1e-6)]:
            error=abs(mode_meta['normal'][name]-old['model'][oldname]);comparisons[name]=dict(error=error,tolerance=tol,passed=error<=tol)
        if not all(v['passed'] for v in comparisons.values()):raise ValueError('Normal differs from historical frozen checkpoint inference')
        if digest(batch['inputs'])!=original_input_hash or digest(model.state_dict())!=before:raise ValueError('Inputs/parameters mutated')
        save_tensor_bundle(directory/'outputs_and_trace.pt.gz',dict(predictions=all_predictions,traces=all_traces,processor=processor.state_dict()))
        # Portable arrays in world metres. Keep all 15 outputs, not just rendered landmarks.
        keep=raw['inputs']['gs_mask'][0];arrays=dict(initial=raw['inputs']['gs_xyz'][0,keep].numpy(),initial_attributes=raw['inputs']['gs_attr'][0,keep].numpy(),
            frame_ids=raw['meta']['frame_ids'][0].numpy(),near_indices=regions['near'].cpu().numpy(),far_indices=regions['far'].cpu().numpy())
        valid_ids=keep.nonzero().flatten().tolist();lookup={v:i for i,v in enumerate(valid_ids)}
        arrays['near_indices']=np.array([lookup[int(i)] for i in arrays['near_indices']]);arrays['far_indices']=np.array([lookup[int(i)] for i in arrays['far_indices']])
        for mode,p in all_predictions.items():arrays[mode]=processor.denormalize_xyz(p['xyz'],raw['inputs']['gs_mask'][:,None].expand(-1,15,-1))[0,:,keep].numpy()
        for h in range(15):
            km=raw['targets']['target_mask'][h][0];arrays[f'gt_{h+1}']=raw['targets']['target_xyz'][h][0,km].numpy()
        np.savez_compressed(directory/'geometry.npz',**arrays)
        record=dict(split=split,episode=ep,start=start,frame_ids=fm['frame_ids'],metrics=rows,summary=mode_meta,normal_regression=comparisons,
            input_sha256=original_input_hash,state_sha256=before,hook_identity_bit_exact=True)
        write(directory/'metrics.json',record);records.append(record)
        print(json.dumps(dict(event='window_completed',episode=ep,start=start,summary=mode_meta)),flush=True)
        del batch,raw,prepared,normal,trace,plain,target,all_predictions,all_traces;torch.cuda.empty_cache()
    if sha(ref['snapshot'])!=ref['sha256'] or digest(model.state_dict())!=before:raise ValueError('Historical checkpoint/state changed')
    write(root/'results.json',records)
    return dict(passed=True,windows=3,conditions=3,optimizer_updates=0,checkpoint_after_sha256=sha(ref['snapshot']),state_sha256=before,
        seconds=time.monotonic()-started,allocated_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved(),execution=metadata())


def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['preflight','infer'],required=True);a.add_argument('--root',type=Path,required=True);a.add_argument('--expected-sha',required=True);args=a.parse_args()
    if metadata()['source_sha']!=args.expected_sha or subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Source not clean/pinned')
    status=args.root/'status'/f'{args.stage}.json'
    if status.exists():raise FileExistsError('Do not overwrite previous attempt')
    try:
        if args.stage=='preflight':preflight(args.root)
        else:write(status,infer(args.root))
    except Exception:
        write(status,dict(passed=False,error=traceback.format_exc(),execution=metadata()));raise

if __name__=='__main__':main()
