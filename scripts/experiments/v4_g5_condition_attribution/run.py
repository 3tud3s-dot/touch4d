"""Frozen condition audit on Val100, with input-only activity preregistration."""
import argparse
import gc
import hashlib
import importlib.metadata
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
from scripts.training.train import GaussianMetrics,copy_initial,make_renderer,to_device
from scripts.training.v4_runtime import prepare_batch,v4_objective
from scripts.experiments.v4_g4_global_motion.protocol import configuration,preflight
from scripts.experiments.v4_g4_global_motion.evaluate import fixed_regions
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.formal_checkpoint import load_for_evaluation
from scripts.experiments.v4_g5_signed_local.diagnostics import sha256,state_hash,finite_tree,write_png
from scripts.experiments.v4_g5_motion_attribution.run import (
    BASE,CHECKPOINT_HASH,LOCKED,CASES,PICTURES,write_json,csv_write,CachedRenderer,horizon_rows)
from scripts.experiments.v4_g5_motion_attribution.metrics import aggregate,rms
from scripts.experiments.v4_g5_attention_routing_audit.run import spatial_rows,check_render_layout
from scripts.experiments.v4_g5_attention_routing_audit.diagnostics import branch_rows
from scripts.experiments.v4_g5_cross_attention_audit.diagnostics import same_history_regions,feature_rms
from .interventions import MODES,CONTENT_MODES,DIRECT_MODES,KV_MODES,CHANNELS,SELECTED,intervention,replay_step,one_frame
from .activity import activity_rows,freeze_groups
from .diagnostics import cross_rows,assert_preserved_inputs


def ensure_assigned_gpu_idle():
    # This cluster's Slurm step exposes physical4 as NVML/CUDA device0, and may
    # omit SLURM_JOB_GPUS. Never pass a physical Slurm index to remapped NVML.
    visible=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True)
    uuids=[v.strip() for v in visible.splitlines() if v.strip()]
    if len(uuids)!=1 or not uuids[0].startswith('GPU-'):
        raise RuntimeError('Cannot unambiguously identify assigned single-GPU NVML namespace')
    assigned=uuids[0]
    output=subprocess.check_output(['nvidia-smi','-i',assigned,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    foreign=[int(v.strip()) for v in output.splitlines() if v.strip() and int(v.strip())!=os.getpid()]
    if foreign:raise RuntimeError('Assigned GPU already has another compute process; preserve it: '+str(foreign))
    return dict(nvml_uuid=assigned,slurm_job_gpus=os.environ.get('SLURM_JOB_GPUS'),
                slurm_step_gpus=os.environ.get('SLURM_STEP_GPUS'),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),foreign_compute_pids=foreign)


def save_images(directory,mode,rendered):
    """Small fixed H15 set only; no new renderer or image-dependent selection."""
    for camera in range(3):
        for channel in ('rgb','alpha'):
            image=rendered[channel][0,14,camera].cpu().numpy()
            if image.ndim==2:image=np.repeat(image[...,None],3,-1)
            write_png(directory/f'h15_v{camera}_{channel}_{mode}.png',np.rint(np.clip(image,0,1)*255).astype(np.uint8))


def _assert_kv_contract(changed,normal,channel):
    for row,ref in zip(changed['condition_cross_calls'],normal):
        if channel=='FixedRoutingChangedValue':
            torch.testing.assert_close(row['alpha_used'],ref['alpha_used'],rtol=0,atol=0)
            torch.testing.assert_close(row['k'],ref['k'],rtol=0,atol=0)
        elif channel=='ChangedKeyNormalValue':torch.testing.assert_close(row['v'],ref['v'],rtol=0,atol=0)


@torch.inference_mode()
def smoke(model,loaders,processor,renderer):
    checks=[]
    for ep,start in ((3,135),(6,110)):
        ds=loaders['val'].dataset
        batch=to_device(prepare_batch(collate_fn([ds[ds.windows.index((ep,start))]]),processor,model),'cuda')
        x=batch['inputs'];before=state_hash(x);bare=model(x)
        with intervention(model) as captures:normal,trace=model(x,return_trace=True)
        for key in bare:torch.testing.assert_close(bare[key],normal[key],rtol=0,atol=0)
        target=renderer.render_targets(batch,processor);finite_tree(target);check_render_layout(target)
        for mode in MODES:
            with intervention(model,mode):pred,other=model(x,return_trace=True)
            finite_tree((pred,other))
            loss,metrics=v4_objective(pred,batch,processor,CachedRenderer(renderer,target));finite_tree((loss,metrics))
            if not bool(metrics['render_time_available'].all()):raise AssertionError('Pilot supervision incomplete')
            checks.append(dict(episode=ep,start=start,mode=mode,joint=float(loss),finite=True))
        for k in (0,4,9,14):
            normal_replay=replay_step(model,x,trace,captures,k)
            torch.testing.assert_close(normal_replay['xyz'],normal['xyz'][:,k],rtol=0,atol=0)
            for mode in MODES[1:]:
                changed=replay_step(model,x,trace,captures,k,mode)
                with intervention(model,mode,only_step=k) as rows:actual=model(x)
                torch.testing.assert_close(actual['xyz'][:,:k],normal['xyz'][:,:k],rtol=0,atol=0)
                torch.testing.assert_close(actual['xyz'][:,k],changed['xyz'],rtol=0,atol=0)
                assert_preserved_inputs(rows[k],captures[k],mode)
                if mode in KV_MODES:
                    for channel in CHANNELS[1:]:
                        c=replay_step(model,x,trace,captures,k,mode,channel);finite_tree(c)
                        _assert_kv_contract(c,captures[k]['condition_cross_calls'],channel)
        if state_hash(x)!=before:raise AssertionError('Pilot raw inputs changed')
    return dict(normal_bit_exact=True,all_modes_finite=True,only_step_replay_exact=True,
        individual_path_checks=True,kv_contracts=True,windows=[[3,135],[6,110]],rows=checks,optimizer_updates=0)


def single_effects(model,batch,normal,trace,captures,normal_rows,renderer,target,processor,scope,verify):
    records={k:[] for k in ('kv_attribution','kv_regions','cross_mechanisms','intervention_checks')}
    x=batch['inputs'];scale=processor.state_dict()['scale'];keep=x['gs_mask'][0]
    for h in SELECTED:
        k=h-1;ref=replay_step(model,x,trace,captures,k)
        torch.testing.assert_close(ref['xyz'],normal['xyz'][:,k],rtol=0,atol=0)
        view,original=one_frame(batch,normal,k);target_one={name:v[:,k:k+1] for name,v in target.items()}
        gt=batch['targets']['target_xyz'][k][0,batch['targets']['target_mask'][k][0]]
        for mode in MODES[1:]:
            channels=CHANNELS if mode in KV_MODES else CHANNELS[:1]
            for channel in channels:
                changed=replay_step(model,x,trace,captures,k,mode,channel)
                _assert_kv_contract(changed,captures[k]['condition_cross_calls'],channel)
                prediction=dict(original,xyz=changed['xyz'][:,None])
                loss,metrics=v4_objective(prediction,view,processor,CachedRenderer(renderer,target_one))
                finite_tree((prediction,changed,loss,metrics))
                if not bool(metrics['render_time_available'].all()):raise AssertionError('K/V supervision incomplete')
                row=dict(**scope,mode=mode,channel=channel,history='fixed_Normal',horizon=h,
                    cd_mm=float(metrics['geometry'][0])*scale*1000,rgb=float(metrics['rgb'][0]),alpha=float(metrics['alpha'][0]),joint=float(loss),
                    change_rms_mm=rms((changed['xyz']-ref['xyz'])[0,keep]*scale*1000),
                    query_change_feature_rms=feature_rms((changed['query']-ref['query'])[trace['field'].anchor_mask]))
                for name in ('cd_mm','rgb','alpha','joint'):row['delta_'+name]=row[name]-normal_rows[k][name]
                records['kv_attribution'].append(row)
                region_scope=dict(**scope,mode=mode,channel=channel,horizon=h,history='fixed_Normal')
                records['kv_regions'].extend(same_history_regions(x,trace,dict(trace['steps'][k],_query=ref['query']),changed,gt,scale,region_scope))
                records['cross_mechanisms'].extend(cross_rows(x,trace['field'],changed['condition_cross_calls'],captures[k]['condition_cross_calls'],
                    dict(**scope,channel=channel),mode,'fixed_Normal',h))
                if verify and channel=='EndToEnd':
                    with intervention(model,mode,only_step=k):actual=model(x)
                    torch.testing.assert_close(actual['xyz'][:,:k],normal['xyz'][:,:k],rtol=0,atol=0)
                    torch.testing.assert_close(actual['xyz'][:,k],changed['xyz'],rtol=0,atol=0)
                    records['intervention_checks'].append(dict(**scope,mode=mode,horizon=h,prefix_bit_exact=True,current_bit_exact=True))
    return records


@torch.inference_mode()
def execute(args,out):
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source!=args.expected_sha or subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Requires clean frozen source')
    if subprocess.check_output(['git','diff',BASE,'--',*LOCKED],text=True).strip():raise ValueError('Production source changed')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Requires authorized Slurm GPU')
    if args.allocation_id!=os.environ['SLURM_JOB_ID']:raise RuntimeError('Incorrect authorized allocation')
    if torch.cuda.device_count()!=1 or '5090' not in torch.cuda.get_device_name(0):raise RuntimeError('Expected one RTX5090')
    if torch.__version__!='2.7.1+cu128' or torch.version.cuda!='12.8' or importlib.metadata.version('gsplat')!='1.4.0':raise RuntimeError('Frozen environment changed')
    if sha256(args.checkpoint)!=CHECKPOINT_HASH:raise ValueError('Checkpoint hash differs')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    processor=DataProcessor('001-rope');config=configuration(args.root,args.depth_root,BASE)
    loaders,data=preflight(config,args.reference)
    model=G5DirectPredictor(horizon=15,prediction_interval=1)
    state=load_for_evaluation(args.checkpoint,model,processor.state_dict(),source_sha=BASE,manifest=data['manifest'])
    ledger=state['ledger']
    if ledger['completed_epochs']!=8 or ledger['successful_updates']!=12112:raise ValueError('Wrong Epoch8 Best')
    expected=ledger['history'][-1]['validation']['model'];del state,ledger
    model.eval().requires_grad_(False);digest=state_hash(model.state_dict())
    provenance=dict(training_source_sha=BASE,diagnostic_source_sha=source,checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=CHECKPOINT_HASH,
        model_hash_before=digest,job_id=os.environ['SLURM_JOB_ID'],step_id=os.environ.get('SLURM_STEP_ID'),command=sys.argv,python=sys.version,
        torch=torch.__version__,cuda=torch.version.cuda,gsplat=importlib.metadata.version('gsplat'),gpu=torch.cuda.get_device_name(0),
        processor=processor.state_dict(),expected_normal_validation=expected,modes=list(MODES),kv_modes=list(KV_MODES),kv_channels=list(CHANNELS),
        source_parent='2100d95851c041b85343eb4ff5898fccbaf87f46',selected_epoch=8,selected_updates=12112,
        fixed_routing_readout='Normal layer Alpha for aggregation AND final tool12/mass4; no residual Key leakage',
        changed_key='original Key projection and pairwise content; original Normal Value and real geometric score',
        optimizer_created=False,backward_called=False,optimizer_updates=0,manifest=data)
    write_json(out/'provenance_initial.json',provenance);write_json(out/'manifest.json',data)
    # Freeze input-based strata before ANY model prediction or rendered loss.
    activity=[];seen=[]
    for raw in loaders['val']:
        ep=int(raw['meta']['episode_id'][0]);start=int(raw['meta']['frame_ids'][0,0]);seen.append([ep,start])
        batch=prepare_batch(raw,processor,model)
        activity.extend(activity_rows(model.conditions,batch['inputs'],dict(split='val',episode=ep,start=start)))
    if seen!=data['manifest']['val']['window_keys'] or len(seen)!=100:raise AssertionError('Activity Val100 count/order differs')
    groups=freeze_groups(activity);write_json(out/'results'/'condition_activity_groups.json',groups)
    csv_write(out/'results'/'condition_activity.csv',activity);write_json(out/'results'/'condition_activity.json',activity)
    provenance['activity_groups_sha256']=sha256(out/'results'/'condition_activity_groups.json')
    provenance['gpu_idle_preflight']=ensure_assigned_gpu_idle()
    model.cuda();renderer=make_renderer(config)
    write_json(out/'real_preflight.json',smoke(model,loaders,processor,renderer));gc.collect()
    tables={k:[] for k in ('windows','horizons','kv_attribution','kv_regions','cross_mechanisms','intervention_checks','branch_changes','structure','input_checks','direct_inputs','condition_content')}
    trackers={m:GaussianMetrics(processor.state_dict()['scale'],config) for m in (*MODES,'Copy')}
    group_by={(w['episode'],w['start']):w['activity_group'] for w in groups['windows']}
    clock=time.perf_counter();torch.cuda.reset_peak_memory_stats();seen=[];scale=processor.state_dict()['scale']

    def process(raw,split):
        offsets={k:len(v) for k,v in tables.items()};ep=int(raw['meta']['episode_id'][0]);start=int(raw['meta']['frame_ids'][0,0])
        scope=dict(split=split,episode=ep,start=start,activity_group=group_by.get((ep,start),'MECHANISM_CASE_ONLY'))
        if raw['meta']['frame_ids'].tolist()!=[list(range(start,start+16))]:raise AssertionError('Frame IDs changed')
        batch=to_device(prepare_batch(raw,processor,model),'cuda');x=batch['inputs'];keep=x['gs_mask'][0]
        if x['operation_hands'].tolist()!=[[ep>=5,True]]:raise AssertionError('Hand policy differs')
        input_digest=state_hash(x);target=renderer.render_targets(batch,processor);finite_tree(target);check_render_layout(target)
        with intervention(model) as captures:normal,trace=model(x,return_trace=True)
        finite_tree((normal,trace));cache=CachedRenderer(renderer,target)
        loss,metrics=v4_objective(normal,batch,processor,cache);finite_tree((loss,metrics))
        normal_rows=horizon_rows(normal,batch,metrics,normal,scale,'Normal',scope)
        if (ep,start) in PICTURES:
            directory=out/'images'/f'{split}_ep{ep}_s{start}';directory.mkdir()
            save_images(directory,'GT',target);save_images(directory,'Normal',cache.last)
        cache.last=None
        singles=single_effects(model,batch,normal,trace,captures,normal_rows,renderer,target,processor,scope,(ep,start) in CASES)
        for name,rows in singles.items():tables[name].extend(rows)
        if (ep,start) in CASES:
            center=x['gs_xyz'].new_tensor(processor.state_dict()['center'])
            arrays=dict(initial_world=(x['gs_xyz'][0,keep]*scale+center).cpu().numpy(),frame_ids=raw['meta']['frame_ids'][0].numpy(),
                valid_ids=keep.nonzero().flatten().cpu().numpy(),anchor_indices=trace['field'].anchor_indices[0].cpu().numpy(),
                anchor_valid=trace['field'].anchor_mask[0].cpu().numpy(),bandwidth_normalized=trace['field'].bandwidth.cpu().numpy())
            for key,value in x.items():arrays['input_'+key]=value.cpu().numpy()
            for name,ids in fixed_regions(x).items():arrays['region_'+name]=torch.searchsorted(keep.nonzero().flatten(),ids).cpu().numpy()
            for k in range(15):
                gt=batch['targets']['target_xyz'][k][0,batch['targets']['target_mask'][k][0]]
                arrays[f'gt_h{k+1:02}_world']=(gt*scale+center).cpu().numpy()
        for mode in (*MODES,'Copy'):
            if mode=='Normal':pred,mt,mc,m,current_loss=normal,trace,captures,metrics,loss
            else:
                if mode=='Copy':pred=copy_initial(x,'gs_geometry')
                else:
                    with intervention(model,mode) as mc:pred,mt=model(x,return_trace=True)
                    finite_tree((pred,mt))
                current_loss,m=v4_objective(pred,batch,processor,cache);finite_tree((current_loss,m))
                if (ep,start) in PICTURES:save_images(directory,mode,cache.last)
                cache.last=None
            for key in pred:
                if key!='xyz':torch.testing.assert_close(pred[key],normal[key],rtol=0,atol=0)
            if not bool(m['render_time_available'].all()):raise AssertionError('Incomplete three-camera supervision')
            if split=='val':trackers[mode].add(batch['meta']['episode_id'],m)
            tables['windows'].append(dict(**scope,mode=mode,cd_mm=float(m['geometry'][0])*scale*1000,rgb=float(m['rgb'][0]),alpha=float(m['alpha'][0]),joint=float(current_loss)))
            tables['horizons'].extend(normal_rows if mode=='Normal' else horizon_rows(pred,batch,m,normal,scale,mode,scope))
            tables['structure'].extend(dict(**scope,mode=mode,**row) for row in spatial_rows(pred,batch,scale))
            if mode!='Copy':
                tables['branch_changes'].extend(branch_rows(x,mt,trace,scale,scope,mode))
                for k,(raw_mode,raw_normal) in enumerate(zip(mc,captures)):
                    assert_preserved_inputs(raw_mode,raw_normal,mode)
                    content=raw_mode['condition_content'];valid=content['valid']
                    change=(content['effective_features13']-content['original_features13'])[valid]
                    tables['condition_content'].append(dict(**scope,mode=mode,horizon=k+1,
                        a0_input_change_feature_rms=feature_rms(change[:,:7]),
                        pressure_input_change_feature_rms=feature_rms(change[:,7:9]),
                        quality_input_change_feature_rms=feature_rms(change[:,9:13]),
                        hand_projection_change_feature_rms=feature_rms(content['effective_hand_projection']-content['original_hand_projection']),
                        controller_positions_and_masks_preserved=True,hand_encoder_tokens_preserved=True))
                    mechanism=cross_rows(x,trace['field'],raw_mode['condition_cross_calls'],raw_normal['condition_cross_calls'],
                        dict(**scope,channel='EndToEnd'),mode,'full_AR',k+1)
                    tables['cross_mechanisms'].extend(r for r in mechanism if r['region']=='all' or k+1 in SELECTED)
                    d=raw_mode['direct_intervention'];s=mt['steps'][k];anchor_keep=mt['field'].anchor_mask[0]
                    for hand in range(2):
                        basis=s['hand_basis12'][0,hand]*scale*1000
                        local_basis=s['anchor_hand_basis18'][0,anchor_keep,hand]*scale*1000
                        contribution=s['hand_contribution'][0,hand]*scale*1000
                        tables['direct_inputs'].append(dict(**scope,mode=mode,horizon=k+1,hand='left' if hand==0 else 'right',
                            hand_valid=bool(s['hand_valid'][0,hand]),hand_basis12_mm=basis.cpu().tolist(),
                            anchor_basis18_mean_mm=local_basis.mean(0).cpu().tolist(),
                            anchor_controller6_rms_mm=float(local_basis[:,12:].square().mean().sqrt()),
                            global_hand_contribution_xyz_mm=contribution.cpu().tolist(),
                            global_direct_hand_token_feature_rms=feature_rms(d['global_kwargs_effective']['hand_tokens'][0,hand]),
                            local_direct_hand_token_feature_rms=feature_rms(d['local_kwargs_effective']['hand_tokens'][0,hand])))
            if (ep,start) in CASES:arrays[mode+'_world']=(pred['xyz'][0,:,keep]*scale+center).cpu().numpy()
        if (ep,start) in CASES:np.savez_compressed(out/'arrays'/f'{split}_ep{ep}_s{start}.npz',**arrays)
        after=state_hash(x)
        if after!=input_digest:raise AssertionError('Input mutation detected')
        tables['input_checks'].append(dict(**scope,input_hash_before=input_digest,input_hash_after=after,
            operation_hands=x['operation_hands'].cpu().tolist(),frame_ids=raw['meta']['frame_ids'][0].tolist(),
            initial_valid_gs=int(keep.sum()),target_gs_counts=[int(m[0].sum()) for m in batch['targets']['target_mask']],
            preserved=True,optimizer_updates=0))
        if not all(not p.requires_grad and p.grad is None for p in model.parameters()):raise AssertionError('Unexpected gradient state')
        write_json(out/'windows'/f'{split}_ep{ep}_s{start}.json',{name:rows[offsets[name]:] for name,rows in tables.items()})
        del batch,normal,trace,captures,target,cache
        gc.collect()

    for raw in loaders['val']:
        seen.append([int(raw['meta']['episode_id'][0]),int(raw['meta']['frame_ids'][0,0])]);process(raw,'val')
        if len(seen)%10==0:print('VAL_COMPLETE',len(seen),'/100',round(time.perf_counter()-clock,1),'seconds',flush=True)
    if seen!=data['manifest']['val']['window_keys']:raise AssertionError('Inference order differs from frozen activity audit')
    ds=loaders['train'].dataset;process(collate_fn([ds[ds.windows.index((7,105))]]),'train')
    summary={m:t.summary() for m,t in trackers.items()}
    for mode,s in summary.items():
        if {ep:r['samples'] for ep,r in s['episodes'].items()}!={'3':55,'6':45}:raise AssertionError('Incomplete '+mode)
    for key,tol in (('geometry_mm',1e-4),('rgb',1e-6),('alpha',1e-6),('objective',1e-6)):
        if abs(summary['Normal'][key]-expected[key])>tol:raise AssertionError(('Normal reproduction',key))
    for name,rows in tables.items():write_json(out/'results'/f'{name}.json',rows);csv_write(out/'results'/f'{name}.csv',rows)
    write_json(out/'results'/'val_summary.json',summary);val=[]
    for mode,s in summary.items():
        for ep,row in [('episode_equal',s),*s['episodes'].items()]:
            val.append(dict(mode=mode,episode=ep,cd_mm=row['geometry_mm'],rgb=row['rgb'],alpha=row['alpha'],
                joint=s['objective'] if ep=='episode_equal' else row['supervised_geometry']+row['rgb']+.1*row['alpha'],samples=row.get('samples')))
    csv_write(out/'results'/'val_summary.csv',val)
    normal_by={(r['split'],r['episode'],r['start']):r for r in tables['windows'] if r['mode']=='Normal'}
    paired=[]
    for row in tables['windows']:
        if row['mode']=='Normal':continue
        n=normal_by[row['split'],row['episode'],row['start']]
        paired.append(dict(row,**{'delta_'+k:row[k]-n[k] for k in ('cd_mm','rgb','alpha','joint')}))
    csv_write(out/'results'/'paired_window_effects.csv',paired);write_json(out/'results'/'paired_window_effects.json',paired)
    aggregate_specs={
        'horizons':(('mode','horizon'),('cd_mm','rgb','alpha','joint','rms_mm','mean_x_mm','mean_y_mm','mean_z_mm','change_from_normal_rms_mm')),
        'kv_attribution':(('mode','channel','horizon'),('cd_mm','rgb','alpha','joint','delta_cd_mm','delta_rgb','delta_alpha','delta_joint','change_rms_mm','query_change_feature_rms')),
        'kv_regions':(('mode','channel','region','horizon'),('change_rms_mm','pred_to_gt_mm','normal_pred_to_gt_mm','base_change_gs_rms_mm','global_change_gs_rms_mm','local_change_gs_rms_mm')),
        'branch_changes':(('mode','space','branch','region','horizon'),('rms_mm','demeaned_rms_mm','mean_x_mm','mean_y_mm','mean_z_mm','change_rms_mm','change_demeaned_rms_mm')),
        'structure':(('mode','region','horizon'),('displacement_rms_mm','global_translation_removed_rms_mm','mean_x_mm','mean_y_mm','mean_z_mm','pred_to_gt_mm','neighbor_ratio_mean','neighbor_distortion_p95')),
        'cross_mechanisms':(('mode','history','channel','region','horizon','layer','head'),('key_change_feature_rms','value_change_feature_rms','null_mass','left_mass','right_mass','entropy_nats','attention_total_variation','query_change_feature_rms','aggregate_change_feature_rms','projected_change_feature_rms_joint_heads'))}
    for name,(keys,values) in aggregate_specs.items():
        rows=aggregate(tables[name],keys,values);write_json(out/'results'/f'{name}_val_summary.json',rows);csv_write(out/'results'/f'{name}_val_summary.csv',rows)
    grouped=aggregate(tables['windows'],('mode','activity_group'),('cd_mm','rgb','alpha','joint'))
    csv_write(out/'results'/'activity_group_summary.csv',grouped);write_json(out/'results'/'activity_group_summary.json',grouped)
    if state_hash(model.state_dict())!=digest or sha256(args.checkpoint)!=CHECKPOINT_HASH:raise AssertionError('Model or checkpoint changed')
    torch.cuda.synchronize()
    provenance.update(model_hash_after=state_hash(model.state_dict()),checkpoint_hash_after=sha256(args.checkpoint),val_seen=seen,
        elapsed_seconds=time.perf_counter()-clock,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        activity_frozen_before_any_prediction=True)
    write_json(out/'provenance.json',provenance)
    write_json(out/'completed.json',dict(state='INFERENCE_COMPLETED',source_sha=source,val100=True,mode_count=9,optimizer_updates=0))
    print('FROZEN_CONDITION_ATTRIBUTION_COMPLETED',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','depth-root','reference','checkpoint','output','expected-sha','allocation-id'):p.add_argument('--'+name,required=True)
    args=p.parse_args();out=Path(args.output).resolve()
    protected=[Path(v).resolve() for v in (args.root,args.depth_root,args.reference,Path.cwd())]
    if out.exists() or out.parent.name!='v4-g5-frozen-condition-attribution-v1' or any(out==v or v in out.parents for v in protected):
        raise ValueError('Requires new isolated condition-attribution Run')
    out.mkdir(parents=True,exist_ok=False)
    for name in ('results','arrays','images','windows','logs','figures'):(out/name).mkdir()
    try:execute(args,out)
    except BaseException:
        write_json(out/'failure.json',dict(traceback=traceback.format_exc(),command=sys.argv));raise


if __name__=='__main__':main()
