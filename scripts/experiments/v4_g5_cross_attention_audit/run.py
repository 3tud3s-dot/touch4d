"""Six-mode frozen Cross-Attention audit; uses original model/renderer/loss."""
import argparse
import gc
import importlib.metadata
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
from scripts.training.train import GaussianMetrics, copy_initial, make_renderer, to_device
from scripts.training.v4_runtime import prepare_batch, v4_objective
from scripts.experiments.v4_g4_global_motion.evaluate import fixed_regions, motion_metrics
from scripts.experiments.v4_g4_global_motion.protocol import configuration, preflight
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.formal_checkpoint import load_for_evaluation
from scripts.experiments.v4_g5_signed_local.diagnostics import sha256, state_hash, finite_tree
from scripts.experiments.v4_g5_motion_attribution.run import (
    BASE, CHECKPOINT_HASH, LOCKED, CASES, PICTURES, write_json, csv_write,
    CachedRenderer, horizon_rows, save_pictures)
from scripts.experiments.v4_g5_motion_attribution.metrics import aggregate, rms
from .interventions import MODES, SELECTED, metadata, intervention, replay_step, one_frame
from .diagnostics import attention_rows, same_history_regions, feature_rms


@torch.inference_mode()
def isolation_check(model, inputs):
    """Alter only Cross K/V/scores, leaving all direct Action/Tactile routes intact."""
    with intervention(model,'NoCrossInfo') as raw:
        first = model(inputs)
    handles=[]
    try:
        for block in model.cross:
            for module in (block.k,block.v):
                handles.append(module.register_forward_hook(lambda m,a,o:o.neg()+7))
            for score in block.pairwise:
                handles.append(score.register_forward_hook(lambda m,a,o:
                    o+torch.linspace(-1000,1000,o.shape[-2],device=o.device,dtype=o.dtype)[None,None,:,None]))
        with intervention(model,'NoCrossInfo') as changed:
            second=model(inputs)
    finally:
        for h in reversed(handles):h.remove()
    for key in first:torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
    delta=max(float((a['cross_calls'][-1]['output'][1]-b['cross_calls'][-1]['output'][1]).abs().max())
              for a,b in zip(raw,changed))
    if not delta>0:raise AssertionError('Isolation diagnostic did not actually alter Attention scores/weights')
    return dict(all_attributes_bit_exact=True,max_attention_weight_change=delta,
                intervention='Cross K/V outputs -x+7; pairwise scores fixed linear -1000..1000; no parameter write')


def single_effects(model,batch,normal,trace,captures,normal_rows,renderer,target,processor,scope,verify):
    rows,regions,checks=[],[],[]
    x,scale=batch['inputs'],processor.state_dict()['scale']
    for h in SELECTED:
        k=h-1
        base=replay_step(model,x,trace,captures,k,'Normal')
        torch.testing.assert_close(base['xyz'],normal['xyz'][:,k],rtol=0,atol=0)
        gt=batch['targets']['target_xyz'][k][0,batch['targets']['target_mask'][k][0]]
        view,original=one_frame(batch,normal,k)
        target_one={key:v[:,k:k+1] for key,v in target.items()}
        normal_step=dict(trace['steps'][k],_query=base['query'])
        for mode in MODES[1:]:
            changed=replay_step(model,x,trace,captures,k,mode)
            prediction=dict(original,xyz=changed['xyz'][:,None])
            loss,m=v4_objective(prediction,view,processor,CachedRenderer(renderer,target_one))
            finite_tree((changed,prediction,loss,m))
            if not bool(m['render_time_available'].all()):raise ValueError('Incomplete single-step supervision')
            row=dict(**scope,mode=mode,horizon=h,history='fixed_Normal',cd_mm=float(m['geometry'][0])*scale*1000,
                     rgb=float(m['rgb'][0]),alpha=float(m['alpha'][0]),joint=float(loss),
                     change_rms_mm=rms((changed['xyz']-base['xyz'])[0,x['gs_mask'][0]]*scale*1000),
                     query_change_feature_rms=feature_rms((changed['query']-base['query'])[trace['field'].anchor_mask]))
            for key in ('cd_mm','rgb','alpha','joint'):row['delta_'+key]=row[key]-normal_rows[k][key]
            rows.append(row)
            regions.extend(same_history_regions(x,trace,normal_step,changed,gt,scale,
                                               dict(**scope,mode=mode,horizon=h)))
            if verify:
                with intervention(model,mode,only_step=k):actual=model(x)
                torch.testing.assert_close(actual['xyz'][:,:k],normal['xyz'][:,:k],rtol=0,atol=0)
                torch.testing.assert_close(actual['xyz'][:,k],changed['xyz'],rtol=0,atol=0)
                checks.append(dict(**scope,mode=mode,horizon=h,prefix_bit_exact=True,current_bit_exact=True))
    return rows,regions,checks


@torch.inference_mode()
def execute(args,out):
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source!=args.expected_sha or subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Requires frozen clean diagnostic source')
    if subprocess.check_output(['git','diff',BASE,'--',*LOCKED],text=True).strip():
        raise ValueError('Original production/model files changed')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Requires Slurm GPU')
    if torch.cuda.device_count()!=1 or '5090' not in torch.cuda.get_device_name(0):raise RuntimeError('Expected one assigned RTX5090')
    if (torch.__version__!='2.7.1+cu128' or torch.version.cuda!='12.8' or importlib.metadata.version('gsplat')!='1.4.0'):
        raise RuntimeError('Frozen CUDA/gsplat environment differs')
    if sha256(args.checkpoint)!=CHECKPOINT_HASH:raise ValueError('Checkpoint hash mismatch')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    processor=DataProcessor('001-rope')
    config=configuration(args.root,args.depth_root,BASE)
    loaders,data=preflight(config,args.reference)
    model=G5DirectPredictor(horizon=15,prediction_interval=1)
    state=load_for_evaluation(args.checkpoint,model,processor.state_dict(),source_sha=BASE,manifest=data['manifest'])
    ledger=state['ledger']
    if ledger['completed_epochs']!=8 or ledger['successful_updates']!=12112:raise ValueError('Not frozen Epoch8 Best')
    expected=ledger['history'][-1]['validation']['model']
    del state,ledger
    model.cuda().eval().requires_grad_(False)
    digest=state_hash(model.state_dict())
    provenance=dict(training_source_sha=BASE,diagnostic_source_sha=source,
        checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=CHECKPOINT_HASH,
        expected_frozen_validation=expected,selected_epoch=8,selected_updates=12112,manifest=data,
        job_id=os.environ['SLURM_JOB_ID'],command=sys.argv,python=sys.version,
        torch=torch.__version__,cuda=torch.version.cuda,gsplat=importlib.metadata.version('gsplat'),
        gpu=torch.cuda.get_device_name(0),processor=processor.state_dict(),modes=list(MODES),
        mode_definitions=[metadata(m) for m in MODES],single_step_horizons=list(SELECTED),
        optimizer_created=False,backward_called=False,optimizer_updates=0,
        query_feature_units='dimensionless; RMS=sqrt(mean(components squared))',
        spatial_units='world mm via DataProcessor scale; independent GT point sets')
    write_json(out/'provenance_initial.json',provenance)
    write_json(out/'manifest.json',data)
    renderer=make_renderer(config);scale=processor.state_dict()['scale']
    pilot_ds=loaders['val'].dataset
    raw=collate_fn([pilot_ds[pilot_ds.windows.index((3,135))]])
    batch=to_device(prepare_batch(raw,processor,model),'cuda');x=batch['inputs']
    bare=model(x)
    with intervention(model) as captures:normal,trace=model(x,return_trace=True)
    for key in bare:torch.testing.assert_close(bare[key],normal[key],rtol=0,atol=0)
    smoke_modes=[]
    for mode in MODES:
        with intervention(model,mode):pred,mt=model(x,return_trace=True)
        finite_tree((pred,mt))
        if torch.count_nonzero(pred['xyz'][~x['gs_mask'][:,None].expand(-1,15,-1)]):
            raise AssertionError('Padding output leak')
        for key in pred:
            if key!='xyz':torch.testing.assert_close(pred[key],normal[key],rtol=0,atol=0)
        smoke_modes.append(dict(mode=mode,h15_finite=True,padding_zero=True,attributes_unchanged=True))
    for k in (0,4,9,14):
        for mode in MODES:
            replay=replay_step(model,x,trace,captures,k,mode)
            with intervention(model,mode,only_step=k):actual=model(x)
            torch.testing.assert_close(actual['xyz'][:,:k],normal['xyz'][:,:k],rtol=0,atol=0)
            torch.testing.assert_close(actual['xyz'][:,k],replay['xyz'],rtol=0,atol=0)
    isolate=isolation_check(model,x)
    target=renderer.render_targets(batch,processor)
    loss,metrics=v4_objective(normal,batch,processor,CachedRenderer(renderer,target))
    finite_tree((loss,metrics))
    if not bool(metrics['render_time_available'].all()):raise ValueError('Pilot real renderer incomplete')
    write_json(out/'real_preflight.json',dict(frame_ids=raw['meta']['frame_ids'].tolist(),normal_all_attributes_bit_exact=True,
        same_history_replay_bit_exact=True,smoke_modes=smoke_modes,no_cross_info_isolation=isolate,
        pilot_joint=float(loss),camera_count=batch['targets']['target_depth'].shape[2],
        resolution=list(batch['targets']['target_depth'].shape[-2:]),optimizer_updates=0))
    del raw,batch,x,bare,normal,trace,captures,pred,mt,replay,actual,target,loss,metrics
    gc.collect()
    tables={n:[] for n in ('windows','horizons','same_history','same_history_regions','attention_trace','structure','intervention_checks')}
    trackers={m:GaussianMetrics(scale,config) for m in (*MODES,'Copy')}
    seen=[];clock=time.perf_counter();torch.cuda.reset_peak_memory_stats()

    def process(raw,split):
        offsets={n:len(v) for n,v in tables.items()}
        ep=int(raw['meta']['episode_id'][0]);start=int(raw['meta']['frame_ids'][0,0]);scope=dict(split=split,episode=ep,start=start)
        if raw['meta']['frame_ids'].tolist()!=[list(range(start,start+16))]:raise ValueError('Frame alignment differs')
        batch=to_device(prepare_batch(raw,processor,model),'cuda');x=batch['inputs'];keep=x['gs_mask'][0]
        if x['operation_hands'].tolist()!=[[ep>=5,True]]:raise ValueError('Incorrect hand routing')
        case,picture=(ep,start) in CASES,(ep,start) in PICTURES
        target=renderer.render_targets(batch,processor);finite_tree(target)
        with intervention(model) as captures:normal,trace=model(x,return_trace=True)
        finite_tree((normal,trace))
        tables['attention_trace'].extend(attention_rows(x,trace,captures,scale,scope))
        cache=CachedRenderer(renderer,target);loss,metrics=v4_objective(normal,batch,processor,cache)
        finite_tree((loss,metrics))
        normal_rows=horizon_rows(normal,batch,metrics,normal,scale,'Normal',scope)
        if picture:
            directory=out/'images'/f'{split}_ep{ep}_s{start}';directory.mkdir()
            save_pictures(directory,'GT',target);save_pictures(directory,'Normal',cache.last)
        cache.last=None
        single,regions,checks=single_effects(model,batch,normal,trace,captures,normal_rows,renderer,target,processor,scope,case)
        tables['same_history'].extend(single);tables['same_history_regions'].extend(regions);tables['intervention_checks'].extend(checks)
        if case:
            center=x['gs_xyz'].new_tensor(processor.state_dict()['center'])
            arrays=dict(initial_world=(x['gs_xyz'][0,keep]*scale+center).cpu().numpy(),frame_ids=raw['meta']['frame_ids'][0].numpy(),
                        valid_ids=keep.nonzero().flatten().cpu().numpy(),anchor_valid=trace['field'].anchor_mask[0].cpu().numpy())
            for name,value in x.items():
                if isinstance(value,torch.Tensor):arrays['input_'+name]=value.cpu().numpy()
            for name,ids in fixed_regions(x).items():arrays['region_'+name]=torch.searchsorted(keep.nonzero().flatten(),ids).cpu().numpy()
            for k in range(15):
                gt=batch['targets']['target_xyz'][k][0,batch['targets']['target_mask'][k][0]]
                arrays[f'gt_h{k+1:02}_world']=(gt*scale+center).cpu().numpy()
        for mode in (*MODES,'Copy'):
            if mode=='Normal':pred,m,current_loss=normal,metrics,loss
            else:
                if mode=='Copy':pred=copy_initial(x,'gs_geometry')
                else:
                    with intervention(model,mode):pred=model(x)
                finite_tree(pred)
                for key in pred:
                    if key!='xyz':torch.testing.assert_close(pred[key],normal[key],rtol=0,atol=0)
                current_loss,m=v4_objective(pred,batch,processor,cache);finite_tree((current_loss,m))
                if picture:save_pictures(directory,mode,cache.last)
                cache.last=None
            if not bool(m['render_time_available'].all()):raise ValueError('Incomplete three-camera supervision')
            if split=='val':trackers[mode].add(batch['meta']['episode_id'],m)
            tables['windows'].append(dict(**scope,mode=mode,cd_mm=float(m['geometry'][0])*scale*1000,
                                         rgb=float(m['rgb'][0]),alpha=float(m['alpha'][0]),joint=float(current_loss)))
            tables['horizons'].extend(normal_rows if mode=='Normal' else horizon_rows(pred,batch,m,normal,scale,mode,scope))
            tables['structure'].extend(dict(**scope,mode=mode,**row) for row in motion_metrics(pred,x,scale))
            if case:arrays[mode+'_world']=(pred['xyz'][0,:,keep]*scale+center).cpu().numpy()
        if case:np.savez_compressed(out/'arrays'/f'{split}_ep{ep}_s{start}.npz',**arrays)
        write_json(out/'windows'/f'{split}_ep{ep}_s{start}.json',{n:rows[offsets[n]:] for n,rows in tables.items()})
        if not all(not p.requires_grad and p.grad is None for p in model.parameters()):raise AssertionError('Unexpected gradient state')
        del batch,normal,trace,captures,target,cache
        gc.collect()

    for raw in loaders['val']:
        seen.append([int(raw['meta']['episode_id'][0]),int(raw['meta']['frame_ids'][0,0])]);process(raw,'val')
        if len(seen)%10==0:print('VAL_COMPLETE',len(seen),'/100',round(time.perf_counter()-clock,1),'seconds',flush=True)
    if seen!=data['manifest']['val']['window_keys'] or len(seen)!=100:raise AssertionError('Val100 order/count differs')
    ds=loaders['train'].dataset;process(collate_fn([ds[ds.windows.index((7,105))]]),'train')
    summary={mode:tracker.summary() for mode,tracker in trackers.items()}
    for mode,s in summary.items():
        if {ep:r['samples'] for ep,r in s['episodes'].items()}!={'3':55,'6':45}:raise AssertionError('Incomplete Val '+mode)
    for key,tol in (('geometry_mm',1e-4),('rgb',1e-6),('alpha',1e-6),('objective',1e-6)):
        if abs(summary['Normal'][key]-expected[key])>tol:raise AssertionError(('Normal reproduction',key,summary['Normal'][key],expected[key]))
    for name,rows in tables.items():write_json(out/'results'/f'{name}.json',rows);csv_write(out/'results'/f'{name}.csv',rows)
    write_json(out/'results'/'val_summary.json',summary)
    val=[]
    for mode,s in summary.items():
        for ep,row in [('episode_equal',s),*s['episodes'].items()]:
            val.append(dict(mode=mode,episode=ep,cd_mm=row['geometry_mm'],rgb=row['rgb'],alpha=row['alpha'],
                            joint=s['objective'] if ep=='episode_equal' else row['supervised_geometry']+row['rgb']+.1*row['alpha'],samples=row.get('samples')))
    csv_write(out/'results'/'val_summary.csv',val)
    for name,keys,values in (
        ('horizons',('mode','horizon'),('cd_mm','rgb','alpha','joint','rms_mm','mean_x_mm','mean_y_mm','mean_z_mm','change_from_normal_rms_mm')),
        ('same_history',('mode','horizon'),('cd_mm','rgb','alpha','joint','delta_cd_mm','delta_rgb','delta_alpha','delta_joint','change_rms_mm','query_change_feature_rms')),
        ('attention_trace',('layer','head','region','horizon'),('query_feature_rms','value_update_feature_rms','update_to_query_ratio','controller_mass','null_mass','attention_entropy_nats','controller_conditional_entropy_nats','final_layer_history_mass','final_layer_tool_vector_rms_mm'))):
        rows=aggregate(tables[name],keys,values)
        write_json(out/'results'/f'{name}_val_summary.json',rows);csv_write(out/'results'/f'{name}_val_summary.csv',rows)
    if state_hash(model.state_dict())!=digest or sha256(args.checkpoint)!=CHECKPOINT_HASH:raise AssertionError('Weights/checkpoint changed')
    torch.cuda.synchronize()
    provenance.update(elapsed_seconds=time.perf_counter()-clock,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),model_hash_before=digest,model_hash_after=state_hash(model.state_dict()),
        checkpoint_hash_after=sha256(args.checkpoint),parameters_frozen=True,val_seen=seen)
    write_json(out/'provenance.json',provenance)
    write_json(out/'completed.json',dict(state='INFERENCE_COMPLETED',source_sha=source,val100=True,train_cases=1,optimizer_updates=0))
    print('FROZEN_CROSS_ATTENTION_COMPLETED',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','depth-root','reference','checkpoint','output','expected-sha'):p.add_argument('--'+name,required=True)
    args=p.parse_args();out=Path(args.output).resolve()
    protected=[Path(v).resolve() for v in (args.root,args.depth_root,args.reference,Path.cwd())]
    if out.exists() or out.parent.name!='v4-g5-frozen-cross-attention-audit-v1' or any(out==v or v in out.parents for v in protected):
        raise ValueError('Requires a new isolated Cross-Attention audit Run')
    out.mkdir(parents=True,exist_ok=False)
    for name in ('results','arrays','images','windows','logs','figures'):(out/name).mkdir()
    try:execute(args,out)
    except BaseException:
        write_json(out/'failure.json',dict(traceback=traceback.format_exc(),command=sys.argv));raise


if __name__=='__main__':main()
