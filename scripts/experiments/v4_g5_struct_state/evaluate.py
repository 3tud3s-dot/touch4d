"""Frozen full-Val100 acceptance; no optimizer, backward, or model selection."""
import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.training.train import atomic_json, copy_initial, metric_tracker
from scripts.training.losses import nearest_distances
from scripts.training.model import initial_knn
from scripts.experiments.v4_g4_global_motion.evaluate import fixed_regions, motion_metrics
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.formal_checkpoint import load_for_evaluation as load_historical
from scripts.experiments.v4_g5_signed_local.diagnostics import finite_tree, sha256, state_hash, write_png
from .model import fresh_model
from .protocol import (BASE, REFERENCE, REFERENCE_HASH, WINDOWS, configuration, preflight,
    source_guard, resource_guard, environment, configure_runtime, calibration_read)
from .runner import CachedRenderer, prepared, original_objective, metric_record, frozen_attributes, JsonLines, assert_real_batch
from .structure import build_structure_graph, structure_loss, structure_statistics
from .checkpoint import load_for_evaluation


LABELS = ('copy', 'historical_g5', 'a_best', 'a_last', 'b_best', 'b_last')
PICTURES = {(3,135),(3,245),(6,110),(7,105)}
PICTURE_LABELS = {'copy','historical_g5','a_best','b_best'}
SELECTED_HORIZONS = (1,5,10,15)
REFERENCE_METRICS = dict(cd_mm=2.402626, rgb=.077624, alpha=.077699, joint=.100694)


def write_csv(path, rows):
    path=Path(path)
    columns=sorted(set().union(*(row.keys() for row in rows))) if rows else ['no_records']
    with path.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def _mean(rows, name):
    values=[row[name] for row in rows if row.get(name) is not None]
    return sum(values)/len(values) if values else None


def aggregate_horizons(rows):
    """Windows average within episode, then episodes receive equal mass."""
    groups={}
    for row in rows:
        if row['split']!='val':continue
        key=(row['model'],row['region'],row['horizon'])
        groups.setdefault(key,{}).setdefault(row['episode'],[]).append(row)
    result=[]
    excluded={'model','split','episode','start','region','horizon','frame_id'}
    for (label,region,horizon),episodes in sorted(groups.items()):
        numeric=set().union(*(r.keys() for rr in episodes.values() for r in rr))-excluded
        summaries=[]
        for episode,values in sorted(episodes.items()):
            record=dict(model=label,episode=episode,region=region,horizon=horizon,windows=len(values),
                available_windows=sum(bool(r.get('region_available',True)) for r in values))
            record.update({name:_mean(values,name) for name in numeric});result.append(record);summaries.append(record)
        combined=dict(model=label,episode='episode_equal',region=region,horizon=horizon,
                      windows=sum(len(v) for v in episodes.values()),
                      available_windows=sum(row['available_windows'] for row in summaries),reduction='episode_equal_window_means')
        combined.update({name:_mean(summaries,name) for name in numeric});result.append(combined)
    return result


def paired_effects(rows):
    selected={(r['model'],r['episode'],r['start']):r for r in rows if r['split']=='val'}
    pairs=(('a_best','historical_g5'),('b_best','a_best'),('a_last','a_best'),('b_last','b_best'))
    output=[]
    for label,reference in pairs:
        for key,row in sorted(selected.items()):
            if key[0]!=label:continue
            baseline=selected[(reference,key[1],key[2])]
            values={f'delta_{name}':row[name]-baseline[name] for name in ('cd_mm','rgb','alpha','joint','loss_struct_unweighted')}
            output.append(dict(model=label,reference=reference,episode=key[1],start=key[2],
                improved_joint=values['delta_joint']<0,worsened_joint=values['delta_joint']>0,**values))
    return output


def _save_images(renderer,out,split,episode,start,label,save_gt=False):
    folder=out/'images'/f'{split}_ep{episode}_start{start}';folder.mkdir(parents=True,exist_ok=True)
    sources=[(label,renderer.rendered)]+([('gt',renderer.target)] if save_gt else [])
    for name,images in sources:
        for horizon in SELECTED_HORIZONS:
            for camera in range(3):
                for channel in ('rgb','alpha'):
                    path=folder/f'h{horizon:02d}_cam{camera}_{channel}_{name}.png'
                    if path.exists():raise FileExistsError(path)
                    array=images[channel][0,horizon-1,camera].detach().cpu().numpy()
                    if channel=='alpha':array=np.repeat(array[...,None],3,-1)
                    write_png(path,np.clip(array*255,0,255).round().astype(np.uint8))


def _world(prediction,x,processor):
    keep=x['gs_mask'][:,None].expand(-1,prediction.shape[1],-1)
    return processor.denormalize_xyz(prediction,keep)[0,:,x['gs_mask'][0]].detach().cpu().numpy()


def diagnostic_regions(x):
    """Missing initial Controllers make regions undefined, never remove a window."""
    try:
        return fixed_regions(x), True
    except ValueError as error:
        if str(error)!='No initial valid operating controller':raise
        ids=x['gs_mask'][0].nonzero().flatten();empty=ids[:0]
        return dict(all=ids,near=empty,middle=empty,far=empty),False


def _motion_rows(prediction,x,scale,groups,controller_regions_available):
    if controller_regions_available:
        rows=motion_metrics(prediction,x,scale)
    else:
        # All-GS diagnostics are still fully defined without any Controller.
        # Exactly the historical 16-other-NN proxy, without fabricating regions.
        keep=x['gs_mask'][0];initial=torch.where(keep[:,None],x['gs_xyz'][0],0.)
        neighbor_ids,valid=initial_knn(x['gs_xyz'],x['gs_mask'],17)
        neighbor_ids,valid=neighbor_ids[0],valid[0]
        length=(initial[:,None]-initial[neighbor_ids]).norm(dim=-1)
        edge_valid=valid & (neighbor_ids!=torch.arange(len(initial),device=initial.device)[:,None]) & (length>1e-8)
        rows=[];factor=scale*1000
        for h,position in enumerate(prediction['xyz'][0],1):
            delta=position[groups['all']]-initial[groups['all']];mean=delta.mean(0)
            distance=(position[:,None]-position[neighbor_ids]).norm(dim=-1)
            ratios=(distance/length.clamp_min(1e-8))[groups['all']][edge_valid[groups['all']]]
            rows.append(dict(horizon=h,region='all',mean_xyz_mm=(mean*factor).tolist(),
                displacement_rms_mm=float(delta.square().sum(-1).mean().sqrt())*factor,
                global_translation_removed_rms_mm=float((delta-mean).square().sum(-1).mean().sqrt())*factor,
                neighbor_ratio_mean=float(ratios.mean()) if ratios.numel() else None,
                neighbor_distortion_p95=float(torch.quantile((ratios-1).abs(),.95)) if ratios.numel() else None))
    existing={(row['horizon'],row['region']) for row in rows}
    for horizon in range(1,prediction['xyz'].shape[1]+1):
        for region,ids in groups.items():
            if (horizon,region) not in existing:
                rows.append(dict(horizon=horizon,region=region,mean_xyz_mm=None,
                    displacement_rms_mm=None,global_translation_removed_rms_mm=None,
                    neighbor_ratio_mean=None,neighbor_distortion_p95=None))
    for row in rows:
        row.update(region_available=bool(len(groups[row['region']])),region_count=len(groups[row['region']]),
                   controller_regions_available=controller_regions_available)
    return sorted(rows,key=lambda row:(row['horizon'],row['region']))


def observed_controller_records(x,meta,processor,split,episode,start):
    """Per-hand observations and endpoint-valid signed changes; never hand-average."""
    cp=x['controller_points'];hand=x['action_state'];operating=x['operation_hands']
    root_valid=x['action_valid'] & operating[:,None] & torch.isfinite(hand).all(-1)
    cross_valid=x['controller_mask'] & operating[:,None,:,None] & torch.isfinite(cp).all(-1)
    signed_valid=cross_valid & root_valid[...,None]
    cp_world=processor.denormalize_xyz(cp,cross_valid)[0].cpu().numpy()
    root_world=processor.denormalize_xyz(hand[...,:3],root_valid)[0].cpu().numpy()
    cp_mask=cross_valid[0].cpu().numpy();signed_mask=signed_valid[0].cpu().numpy()
    root_mask=root_valid[0].cpu().numpy();frames=x['condition_frame_ids'][0].cpu().numpy()
    targets=meta['frame_ids'][0,1:].cpu().numpy();times=meta.get('observation_timestamp_us')
    arrays=dict(observed_controller_world=cp_world,observed_controller_valid=cp_mask,
        observed_signed_controller_valid=signed_mask,
        observed_controller_mask_raw=x['controller_mask'][0].cpu().numpy(),
        observed_root_world=root_world,observed_root_valid=root_mask,
        observed_action_valid_raw=x['action_valid'][0].cpu().numpy(),
        observed_hand_state13_normalized=hand[0].cpu().numpy(),
        operation_hands=operating[0].cpu().numpy())
    cumulative=np.zeros((len(cp_world),2,3),dtype=cp_world.dtype);adjacent=np.zeros_like(cumulative)
    cumulative_valid=np.zeros((len(cp_world),2),dtype=bool);adjacent_valid=np.zeros_like(cumulative_valid)
    records=[]
    for h in range(len(cp_world)):
        for index,name in enumerate(('left','right')):
            pair=signed_mask[h,index] & signed_mask[0,index]
            prev=signed_mask[h,index] & signed_mask[h-1,index] if h else np.zeros(30,dtype=bool)
            cdelta=cp_world[h,index,pair]-cp_world[0,index,pair]
            adelta=cp_world[h,index,prev]-cp_world[h-1,index,prev] if h else np.empty((0,3))
            if len(cdelta):cumulative[h,index]=cdelta.mean(0);cumulative_valid[h,index]=True
            if len(adelta):adjacent[h,index]=adelta.mean(0);adjacent_valid[h,index]=True
            valid_root_cum=bool(root_mask[h,index] and root_mask[0,index])
            valid_root_adj=bool(h and root_mask[h,index] and root_mask[h-1,index])
            root_cum=(root_world[h,index]-root_world[0,index])*1000 if valid_root_cum else None
            root_adj=(root_world[h,index]-root_world[h-1,index])*1000 if valid_root_adj else None
            row=dict(split=split,episode=episode,start=start,hand=name,horizon=h+1,
                condition_frame_id=int(frames[h]),target_frame_id=int(targets[h]),
                operation_hand=bool(operating[0,index]),root_valid=bool(root_mask[h,index]),
                controller_valid_nodes=int(cp_mask[h,index].sum()),signed_controller_valid_nodes=int(signed_mask[h,index].sum()),
                cumulative_pair_nodes=int(pair.sum()),adjacent_pair_nodes=int(prev.sum()),
                root_cumulative_valid=valid_root_cum,root_adjacent_valid=valid_root_adj,
                controller_cumulative_rms_mm=float(np.sqrt(np.mean(np.sum(cdelta**2,-1))))*1000 if len(cdelta) else None,
                controller_adjacent_rms_mm=float(np.sqrt(np.mean(np.sum(adelta**2,-1))))*1000 if len(adelta) else None)
            if times is not None:
                row.update(condition_timestamp_us=int(times[0,h]),target_timestamp_us=int(times[0,h+1]))
            for axis,k in zip('xyz',range(3)):
                row[f'controller_cumulative_{axis}_mm']=float(cumulative[h,index,k])*1000 if len(cdelta) else None
                row[f'controller_adjacent_{axis}_mm']=float(adjacent[h,index,k])*1000 if len(adelta) else (0. if h==0 and root_mask[h,index] else None)
                row[f'root_cumulative_{axis}_mm']=float(root_cum[k]) if root_cum is not None else None
                row[f'root_adjacent_{axis}_mm']=float(root_adj[k]) if root_adj is not None else (0. if h==0 and root_mask[h,index] else None)
            records.append(row)
    arrays.update(observed_controller_cumulative_mean_world=cumulative,observed_controller_cumulative_valid=cumulative_valid,
                  observed_controller_adjacent_mean_world=adjacent,observed_controller_adjacent_valid=adjacent_valid)
    return arrays,records


def _horizon_rows(prediction,batch,processor,values,label,split,episode,start):
    x,y=batch['inputs'],batch['targets'];groups,available=diagnostic_regions(x);factor=processor.state_dict()['scale']*1000
    rows=_motion_rows(prediction,x,processor.state_dict()['scale'],groups,available)
    for h in range(prediction['xyz'].shape[1]):
        pos=prediction['xyz'][0,h];gt=y['target_xyz'][h][0,y['target_mask'][h][0]]
        distances=nearest_distances(pos[groups['all']],gt,512)
        reverse=nearest_distances(gt,pos[groups['all']],512)
        cd=float((distances.mean()+reverse.mean())*.5)*factor
        per_gs=pos.new_zeros(len(pos));per_gs[groups['all']]=distances
        shift=(gt.mean(0)-x['gs_xyz'][0,groups['all']].mean(0))*factor
        for row in rows:
            if row['horizon']!=h+1:continue
            vector=row.pop('mean_xyz_mm') or [None,None,None]
            row.update(model=label,split=split,episode=episode,start=start,
                frame_id=int(batch['meta']['frame_ids'][0,h+1]),
                **dict(zip(('mean_x_mm','mean_y_mm','mean_z_mm'),vector)),
                pred_to_gt_mm=float(per_gs[groups[row['region']]].mean())*factor if len(groups[row['region']]) else None,
                cd_mm=cd,rgb=float(values['rgb_by_horizon'][0,h]),alpha=float(values['alpha_by_horizon'][0,h]))
            row['joint']=cd/factor+row['rgb']+.1*row['alpha']
            if row['region']=='all':
                row.update(**{f'gt_centroid_shift_{axis}_mm':float(shift[k]) for k,axis in enumerate('xyz')})
    return rows


def _state_rows(normal,disabled,trace,disabled_trace,x,processor,metrics_normal,metrics_disabled,episode,start,split,
                normal_horizons,disabled_horizons):
    factor=processor.state_dict()['scale']*1000;regions,available=diagnostic_regions(x);field=trace['field'];valid=field.anchor_mask[0]
    rows=[]
    on_by_h={(row['horizon'],row['region']):row for row in normal_horizons}
    off_by_h={(row['horizon'],row['region']):row for row in disabled_horizons}
    for h,(step,off) in enumerate(zip(trace['steps'],disabled_trace['steps']),1):
        for region,ids in regions.items():
            difference=(normal['xyz'][0,h-1,ids]-disabled['xyz'][0,h-1,ids])*factor
            row=dict(episode=episode,start=start,split=split,horizon=h,region=region,
                region_available=bool(len(ids)),region_count=len(ids),controller_regions_available=available,
                gs_output_change_rms_mm=float(difference.square().sum(-1).mean().sqrt()) if len(ids) else None,
                state_z_feature_rms=float(step['state_z'][0,valid].square().mean().sqrt()),
                state_readout_feature_rms=float(step['state_readout'][0,valid].square().mean().sqrt()),
                disabled_history_state_z_feature_rms=float(off['state_z'][0,valid].square().mean().sqrt()),
                intervention='full_AR_state_readout_off_frozen_B_not_retrained_A')
            for name in ('cd_mm','rgb','alpha','joint'):
                row[f'window_delta_{name}_off_minus_normal']=metrics_disabled[name]-metrics_normal[name]
                on=on_by_h[h,region][name];off_value=off_by_h[h,region][name]
                row[f'normal_horizon_{name}']=on
                row[f'horizon_delta_{name}_off_minus_normal']=off_value-on
            for name,vector in (('base',step['base_cumulative_d']),('global',step['global_delta'][:,None]),('local',step['local_correction'])):
                selected=vector[0] if name=='global' else vector[0,valid]
                row[name+'_anchor_control_rms_mm']=float(selected.square().sum(-1).mean().sqrt())*factor
            rows.append(row)
    return rows


def _epoch_rows(runs,ledgers):
    rows=[]
    for label,root in runs.items():
        ledger=ledgers[label];cumulative=0
        for row in ledger['history']:
            cumulative+=row['successful_updates'];loss=row.get('loss_component_batch_means',{})
            record=dict(model=label,epoch=row['epoch'],successful_updates=row['successful_updates'],
                cumulative_successful_updates=cumulative,visited_train_windows=row['visited_train_windows'],
                visited_val_windows=row['visited_val_windows'])
            for phase,summary in (('train',row['training']),('val',row['validation']['model'])):
                for name,key in (('cd_mm','geometry_mm'),('rgb','rgb'),('alpha','alpha'),('joint','objective')):
                    record[phase+'_'+name]=summary.get(key)
            record.update({f'train_{name}_batch_mean':loss.get(name) for name in
                ('loss_original','loss_struct_unweighted','loss_struct_weighted','loss_total')})
            rows.append(record)
    return rows


@torch.no_grad()
def run(args):
    configure_runtime();source,files=source_guard(args.expected_sha)
    training_source=getattr(args,'training_source_sha',None) or source
    subprocess.check_call(['git','cat-file','-e',training_source+'^{commit}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    resource=resource_guard(require_idle=True);env=environment();begin=time.perf_counter()
    out=Path(args.output)
    if not out.is_absolute() or out.exists():raise FileExistsError('New absolute comparison directory required')
    calibration=calibration_read(args.calibration);calibration_hash=sha256(args.calibration)
    config=configuration(args.root,args.depth_root,source,'g5_struct',calibration['lambda_struct'])
    raw,data=preflight(config,args.g0_reference);processor=DataProcessor('001-rope')
    if sha256(REFERENCE)!=REFERENCE_HASH:raise ValueError('Historical G5 checkpoint SHA256 mismatch')
    history=G5DirectPredictor(horizon=15,prediction_interval=1)
    historical_state=load_historical(REFERENCE,history,processor.state_dict(),source_sha=BASE,manifest=data['manifest'])
    models={'historical_g5':history.cuda().eval().requires_grad_(False)}
    checkpoints={'historical_g5':dict(path=REFERENCE,sha256=REFERENCE_HASH,
        training_source_sha=historical_state['identity']['source_sha'],epoch=historical_state['ledger']['completed_epochs'],
        successful_updates=historical_state['ledger']['successful_updates'])}
    del historical_state
    runs={'a':Path(args.a_run).resolve(),'b':Path(args.b_run).resolve()};protocols={};ledgers={}
    for short,variant in (('a','g5_struct'),('b','g6_state_struct')):
        root=runs[short];protocol=json.loads((root/'protocol.json').read_text());identity=protocol['identity'];protocols[short]=protocol
        if identity['variant']!=variant or identity['calibration_sha256']!=calibration_hash:
            raise ValueError('Run variant/frozen calibration mismatch')
        expected_config=configuration(args.root,args.depth_root,training_source,variant,calibration['lambda_struct'])
        if identity['source_sha']!=training_source or identity['training']!=expected_config:
            raise ValueError('A/B training source, calibrated lambda or frozen configuration mismatch')
        for suffix in ('best','last'):
            label=f'{short}_{suffix}';model=fresh_model(variant)
            state=load_for_evaluation(root/f'{suffix}.pt',model,processor.state_dict(),
                source_sha=identity['source_sha'],manifest=data['manifest'],calibration_sha256=calibration_hash)
            if state['identity']!=identity:raise ValueError('Run protocol/checkpoint identity mismatch')
            if state['ledger']['completed_epochs']<1:raise ValueError('At least one complete Epoch required')
            if suffix=='last':ledgers[short]=state['ledger']
            models[label]=model.cuda().eval().requires_grad_(False)
            checkpoints[label]=dict(path=str(root/f'{suffix}.pt'),sha256=sha256(root/f'{suffix}.pt'),
                training_source_sha=identity['source_sha'],epoch=state['ledger']['completed_epochs'],
                successful_updates=state['ledger']['successful_updates'],best_epoch=state['ledger']['best_epoch'])
            del state
        durable=sorted(root.glob('epoch_*.pt'))
        if [p.name for p in durable]!=[f'epoch_{i:02d}.pt' for i in range(1,len(durable)+1)] or len(durable)!=checkpoints[short+'_last']['epoch']:
            raise ValueError('Recover incomplete/stale Last alias against durable Epoch chain before evaluation')
        if checkpoints[short+'_best']['epoch']!=ledgers[short]['best_epoch']:
            raise ValueError('Recover stale Best alias before evaluation')
        metrics=json.loads((root/'metrics.json').read_text())
        if metrics!=ledgers[short]:raise ValueError('Recover stale metrics.json against durable Last boundary before evaluation')
    if protocols['a']['identity']['initialization']!=protocols['b']['identity']['initialization']:
        raise ValueError('A/B did not share the verified initialization record')
    if protocols['a']['identity']['data']!=protocols['b']['identity']['data']:
        raise ValueError('A/B Manifest/window-order records differ')
    hashes={name:state_hash(model.state_dict()) for name,model in models.items()}
    out.mkdir(parents=True,exist_ok=False)
    for name in ('results','images','arrays','figures','logs'):(out/name).mkdir()
    design=Path(__file__).with_name('DESIGN_AND_PROTOCOL.md')
    if design.is_file():shutil.copyfile(design,out/design.name)
    renderer=CachedRenderer(capture=True)
    trackers={label:metric_tracker(processor,config) for label in LABELS}
    tables={name:[] for name in ('window_metrics','horizon_metrics','structure_summary','state_usage','controller_motion')}
    journal=JsonLines(out/'results'/'window_metrics.jsonl');horizon_journal=JsonLines(out/'results'/'horizon_metrics.jsonl')
    structure_journal=JsonLines(out/'results'/'structure_summary.jsonl')
    counts={3:0,6:0};representatives={(split,ep,start) for split,ep,start in WINDOWS}
    def one(raw_batch,split):
        ep=int(raw_batch['meta']['episode_id'][0]);start=int(raw_batch['meta']['frame_ids'][0,0])
        selected=(split,ep,start) in representatives
        batch=prepared(raw_batch,processor,models['a_best'],'cuda');assert_real_batch(batch,ep,start);x=batch['inputs']
        graph=build_structure_graph(x['gs_xyz'],x['gs_mask']);normal_b=normal_trace=None;normal_metrics=normal_horizons=None
        arrays={};groups,regions_available=diagnostic_regions(x);all_ids=groups['all'];is_picture=(ep,start) in PICTURES
        if selected:
            inverse=torch.zeros(len(x['gs_mask'][0]),dtype=torch.long,device=all_ids.device)
            inverse[all_ids]=torch.arange(len(all_ids),device=all_ids.device)
            arrays=dict(initial_world=processor.denormalize_xyz(x['gs_xyz'],x['gs_mask'])[0,all_ids].cpu().numpy(),
                initial_normalized=x['gs_xyz'][0,all_ids].cpu().numpy(),valid_gs_indices=all_ids.cpu().numpy(),
                structure_edges=inverse[graph.edges[0,graph.edge_mask[0]]].cpu().numpy(),
                structure_rest_lengths_normalized=graph.rest_lengths[0,graph.edge_mask[0]].cpu().numpy(),
                processor_scale=np.asarray(processor.state_dict()['scale']),
                frame_ids=batch['meta']['frame_ids'][0].cpu().numpy(),condition_frame_ids=x['condition_frame_ids'][0].cpu().numpy())
            for region,ids in groups.items():arrays['region_'+region]=inverse[ids].cpu().numpy()
            observations,controller_rows=observed_controller_records(x,batch['meta'],processor,split,ep,start)
            arrays.update(observations);tables['controller_motion'].extend(controller_rows)
            for h in range(15):
                target=batch['targets']['target_xyz'][h];mask=batch['targets']['target_mask'][h]
                arrays[f'target_world_h{h+1:02d}']=processor.denormalize_xyz(target,mask)[0,mask[0]].cpu().numpy()
        for label in LABELS:
            trace=None
            if label=='copy':prediction=copy_initial(x,'gs_geometry')
            elif label=='b_best' and selected:prediction,trace=models[label](x,return_trace=True)
            else:prediction=models[label](x)
            frozen_attributes(prediction,x);finite_tree(prediction)
            loss,values=original_objective(prediction,batch,processor,renderer,config)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite frozen evaluation')
            if split=='val':trackers[label].add(batch['meta']['episode_id'],values)
            metric=metric_record(values,processor)
            record=dict(model=label,split=split,episode=ep,start=start,
                **{k:metric[k] for k in ('cd_mm','rgb','alpha','joint')},
                loss_struct_unweighted=float(structure_loss(prediction['xyz'],graph)),graph_edges=graph.stats[0]['retained_edges'],
                controller_regions_available=regions_available)
            tables['window_metrics'].append(record);journal(record)
            rows=_horizon_rows(prediction,batch,processor,values,label,split,ep,start)
            tables['horizon_metrics'].extend(rows)
            for row in rows:horizon_journal(row)
            structure=[dict(row,model=label,split=split,episode=ep,start=start) for row in structure_statistics(prediction['xyz'],graph)]
            tables['structure_summary'].extend(structure)
            for row in structure:structure_journal(row)
            if selected and label in PICTURE_LABELS:arrays[label+'_world']=_world(prediction['xyz'],x,processor)
            if is_picture and label in PICTURE_LABELS:_save_images(renderer,out,split,ep,start,label,save_gt=label=='copy')
            if label=='b_best' and selected:normal_b,normal_trace,normal_metrics,normal_horizons=prediction,trace,record,rows
            else:del prediction,trace
        if selected:
            disabled,off_trace=models['b_best'](x,return_trace=True,state_readout_off=True)
            frozen_attributes(disabled,x);loss,values=original_objective(disabled,batch,processor,renderer,config)
            off_metrics=metric_record(values,processor)
            off_horizons=_horizon_rows(disabled,batch,processor,values,'b_best_state_readout_off',split,ep,start)
            tables['state_usage'].extend(_state_rows(normal_b,disabled,normal_trace,off_trace,x,processor,normal_metrics,off_metrics,ep,start,split,
                normal_horizons,off_horizons))
            arrays['b_best_state_readout_off_world']=_world(disabled['xyz'],x,processor)
            with (out/'arrays'/f'{split}_ep{ep}_start{start}.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
            del normal_b,normal_trace,disabled,off_trace
        renderer.clear();del batch,graph,arrays,raw_batch
        gc.collect()
    seen=[]
    try:
        for raw_batch in raw['val']:
            ep=int(raw_batch['meta']['episode_id'][0])
            if ep not in counts:raise ValueError('Non-validation episode in Val100')
            seen.append([ep,int(raw_batch['meta']['frame_ids'][0,0])])
            counts[ep]+=1;one(raw_batch,'val')
            if sum(counts.values())%10==0:
                atomic_json(out/'progress.json',dict(stage='frozen_val',windows=sum(counts.values()),counts=counts))
                print(json.dumps(dict(frozen_val_windows=sum(counts.values()))),flush=True)
        if counts!={3:55,6:45}:raise AssertionError('Incomplete Val100')
        if seen!=data['manifest']['val']['window_keys']:raise AssertionError('Actual Val100 order/identity differs from frozen Manifest')
        dataset=raw['train'].dataset
        one(collate_fn([dataset[dataset.windows.index((7,105))]]),'train')
    finally:
        journal.close();horizon_journal.close();structure_journal.close()
    summaries=[];full_summaries={}
    for label,tracker in trackers.items():
        summary=tracker.summary();full_summaries[label]=summary
        for episode,row in [(key,value) for key,value in summary['episodes'].items()]+[('episode_equal',summary)]:
            summaries.append(dict(model=label,episode=episode,samples=row.get('samples',100),
                cd_mm=row['geometry_mm'],rgb=row['rgb'],alpha=row['alpha'],
                joint=row.get('objective',row['supervised_geometry']+row['rgb']+.1*row['alpha'])))
    historical=next(row for row in summaries if row['model']=='historical_g5' and row['episode']=='episode_equal')
    for name,value in REFERENCE_METRICS.items():
        if abs(historical[name]-value)>(1e-4 if name=='cd_mm' else 1e-6):
            raise AssertionError('Historical G5 full Val100 not reproduced: '+name)
    tables['val_summary']=summaries;tables['horizon_summary']=aggregate_horizons(tables['horizon_metrics'])
    tables['paired_window_effects']=paired_effects(tables['window_metrics']);tables['epoch_metrics']=_epoch_rows(runs,ledgers)
    for name,rows in tables.items():write_csv(out/'results'/f'{name}.csv',rows)
    if len({(r['split'],r['episode'],r['start']) for r in tables['state_usage']})!=7:
        raise AssertionError('B state-off diagnostic must cover exactly seven fixed windows')
    for name,model in models.items():
        if state_hash(model.state_dict())!=hashes[name]:raise AssertionError('Frozen model changed: '+name)
    for row in checkpoints.values():
        if sha256(row['path'])!=row['sha256']:raise AssertionError('Checkpoint changed during evaluation')
    atomic_json(out/'checkpoint_provenance.json',checkpoints)
    atomic_json(out/'results'/'val_summary.json',full_summaries)
    run_records={short:dict(path=str(root),protocol=protocols[short],
        completion=json.loads((root/'completion.json').read_text()) if (root/'completion.json').is_file() else None,
        execution_records={p.name:json.loads(p.read_text()) for p in root.glob('*.json')
            if p.name not in ('protocol.json','metrics.json','completion.json')},
        evidence_files={p.name:sha256(p) for p in root.glob('*.json')}) for short,root in runs.items()}
    atomic_json(out/'provenance.json',dict(evaluation_source_sha=source,source_files=files,
        training_source_sha={short:protocol['identity']['source_sha'] for short,protocol in protocols.items()},
        checkpoints=checkpoints,calibration=calibration,calibration_sha256=calibration_hash,resource=resource,
        environment=env,command=sys.argv,runs=run_records,model_state_hashes=hashes,
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        evaluation_worktree=str(Path(__file__).resolve().parents[3]),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        seconds=time.perf_counter()-begin,optimizer_created=False,backward_called=False,successful_updates=0,
        weights_unchanged=True,val_counts=counts,representative_windows=WINDOWS,
        images_note='Independent future GT GS reconstruction renders; original 1280x720 three cameras; no GEO',
        geometry_note='Predicted GS identity only. GT has no material IDs. Fixed spatial graphs and 16NN distortion are structure proxies.',
        controller_note='Hk reads condition at t+k-1 and predicts GS target t+k. Per-hand signed changes preserve opposing hands; endpoint-valid masks; association is not physical causality.',
        controller_regions_unavailable_windows=[dict(episode=r['episode'],start=r['start']) for r in tables['window_metrics']
            if r['split']=='val' and r['model']=='copy' and not r['controller_regions_available']],
        state_off_note='Frozen B Best full AR intervention on seven windows only; not a retrained A or causal physical proof'))
    atomic_json(out/'artifact_hashes.json',{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()})
    atomic_json(out/'completion.json',dict(passed=True,val_windows=100,representative_windows=7,
        images=len(list((out/'images').rglob('*.png'))),source_sha=source,optimizer_updates=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','depth-root','g0-reference','a-run','b-run','calibration','output','expected-sha'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--training-source-sha',help='Explicit original training SHA only when a later evaluation/report-only commit is used')
    args=parser.parse_args()
    try:run(args)
    except BaseException:
        out=Path(args.output)
        if out.is_dir():atomic_json(out/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise


if __name__=='__main__':main()
