import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.geometry import hand_slots,action_conditions
from scripts.rebuild_dataset.rebuild import controller_arrays
from scripts.training.v4_runtime import prepare_batch
from scripts.training.v4_conditions import V4ConditionEncoder
from scripts.experiments.v4_motion_field.analyze import sha,write,csv_write
from .core import deltas,tool_aggregation,cosine,stats,frame_contract


def load(p):
    with gzip.open(p,'rb') as f:return torch.load(f,map_location='cpu',weights_only=False)


def error(a,b):return float(torch.max(torch.abs(a-b)))


def run(ablation,motion,root):
    if (root/'results/coordinate_time_audit.json').exists():raise FileExistsError('Audit already exists')
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    for d in ('results','figures','provenance','logs','fields'):(root/d).mkdir(parents=True,exist_ok=True)
    paths=[ablation/'provenance/checkpoint.json',ablation/'provenance/protocol.json',motion/'provenance/analysis.json']
    for ep,start in ((7,105),(3,135)):
        paths += [ablation/'windows'/f'ep{ep}_start{start}'/n for n in ('raw_batch.pt.gz','outputs_and_trace.pt.gz','zero_tool_inputs.pt.gz')]
        paths += list((root/'provenance'/f'episode_{ep}').glob('*'))
        paths += [motion/'fields'/f'ep{ep}_start{start}_h{h:02}.npz' for h in (5,10,15)]
    hashes=[dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for p in paths]
    prior={r['path']:r['sha256'] for r in json.loads((ablation/'file_manifest.json').read_text())}
    for p in paths:
        if p.is_relative_to(ablation):
            rel=p.relative_to(ablation).as_posix()
            if rel in prior and sha(p)!=prior[rel]:raise ValueError('Prior export hash mismatch')
    controllers=[];objects=[];directions=[];tools=[];timeline=[];checks=[];pathway=[]
    processor=DataProcessor('001-rope');scale=processor.state_dict()['scale']
    # Only deterministic features() is called. No learned encoder/network forward.
    encoder=V4ConditionEncoder(horizon=15).eval().requires_grad_(False)
    for ep,start in ((7,105),(3,135)):
        window=f'ep{ep}_start{start}';folder=ablation/'windows'/window
        raw=load(folder/'raw_batch.pt.gz')
        # Preserve the complete saved batch contract, including unused depth metadata.
        saved=load(folder/'outputs_and_trace.pt.gz');stored_input=load(folder/'zero_tool_inputs.pt.gz')
        if saved['processor']!=processor.state_dict():raise ValueError('Processor differs from trace')
        batch=prepare_batch(raw,processor,SimpleNamespace(config=dict(horizon=15,prediction_interval=1)))
        x=batch['inputs'];trace=saved['traces']['normal'];pred=saved['predictions']['normal']['xyz'][0]
        for k in x:
            if not torch.equal(x[k],stored_input[k]):raise ValueError('Reprocessed input differs: '+k)
        ids=raw['meta']['frame_ids'][0].numpy();cond,target=frame_contract(ids,start,15,1)
        timestamps=raw['meta']['observation_timestamp_us'][0].numpy()
        src=root/'provenance'/f'episode_{ep}';metadata=json.loads((src/'metadata.json').read_text())
        source=dict(np.load(src/'controller.npz'));time=dict(np.load(src/'timestamps.npz'));slots=hand_slots(metadata)
        frames=source['controller_points'].shape[0]
        state,av=action_conditions(source['actions'],source['T_worlds'],source['openings'],slots)
        rebuilt,rm,clipped=controller_arrays(source['T_worlds'],source['openings'],metadata['robot_slot_hands'])
        if not np.array_equal(rebuilt[cond],source['controller_points'][cond]) or not np.array_equal(rm[cond],source['controller_valid_mask'][cond]):raise ValueError('Controller world rebuild mismatch')
        for a,b in [(state[cond],raw['inputs']['action_state'][0].numpy()),(av[cond],raw['inputs']['action_valid'][0].numpy()),
                    (source['controller_points'][cond],raw['inputs']['controller_points'][0].numpy()),
                    (source['controller_valid_mask'][cond],raw['inputs']['controller_mask'][0].numpy()),
                    (time['observation_timestamp_us'][ids],timestamps),(time['tactile_age_us'][cond],raw['inputs']['tactile_age_us'][0].numpy())]:
            if not np.array_equal(a,b):raise ValueError('Source batch alignment mismatch')
        for stream in range(4):
            ts=time[f'raw_tactile_timestamp_us_{stream}'];ix=np.searchsorted(ts,timestamps[:-1],side='right')-1
            age=np.where(ix>=0,timestamps[:-1]-ts[np.maximum(ix,0)],-1)
            if not np.array_equal(age,time['tactile_age_us'][cond,stream]):raise ValueError('Tactile future alignment')
        cp=x['controller_points'][0].reshape(15,60,3);world=raw['inputs']['controller_points'][0].reshape(15,60,3)
        allowed=x['operation_hands'][0].repeat_interleave(30)
        valid=x['controller_mask'][0].reshape(15,60)&allowed&torch.isfinite(cp).all(-1)
        roundtrip=processor.denormalize_xyz(x['controller_points'],x['controller_mask'])
        max_roundtrip=error(roundtrip,raw['inputs']['controller_points'])*1000
        assert max_roundtrip<1e-4
        gs_keep=x['gs_mask'][0];initial=raw['inputs']['gs_xyz'][0,gs_keep].numpy()
        displacement=(pred[:,gs_keep]-x['gs_xyz'][0,gs_keep][None]).double().numpy()*scale
        basefield=dict(np.load(motion/'fields'/f'{window}_h05.npz'))
        # Earlier Oracle exported normalized->world float32, while raw is original world.
        initial_export_error_mm=float(np.max(np.linalg.norm(initial.astype(np.float64)-basefield['initial'],axis=-1)))*1000
        assert initial.shape==basefield['initial'].shape and initial_export_error_mm<1e-4
        near,far=basefield['near'],basefield['far'];middle=np.setdiff1d(np.arange(len(initial)),np.r_[near,far]);regions={'all':np.arange(len(initial)),'near':near,'middle':middle,'far':far}
        window_errors=[];cp_cumulative=[];cp_adjacent=[];tool_time=[]
        methods_by_h={}
        for h in (5,10,15):
            z=dict(np.load(motion/'fields'/f'{window}_h{h:02}.npz'))
            assert np.max(np.linalg.norm(displacement[h-1]-z['normal'],axis=-1))*1000<1e-4
            methods_by_h[h]={m:z[m] for m in z if any(m==f'{loss}_{init}_{snap}' for loss in ('cd','joint') for init in ('zero','normal') for snap in ('best','last'))}
        for j in range(15):
            h=j+1;t=trace['steps'][j];mi=trace['motion_inputs'][j][0]
            cu,cv,ad,am=deltas(cp,valid,j);wc,wcv,wa,wam=deltas(world,valid,j)
            assert torch.equal(cv,wcv) and torch.equal(am,wam)
            delta_error=max(error(cu*scale,wc),error(ad*scale,wa))*1000
            action,hv,features,fv,clean_cp=encoder.features(x,j)
            assert torch.equal(fv[0],valid[j]);assert torch.equal(am,t['adjacent_valid'][0])
            feature_error=error(features,t['features13'])
            assert feature_error<1e-6
            assert torch.equal(t['token_mask62'][0],torch.cat((hv[0],valid[j])))
            assert not t['tokens62'][0][~t['token_mask62'][0]].count_nonzero()
            alpha=t['alpha'][0];assert torch.max(abs(alpha.sum(-1)-1))<2e-6
            assert not alpha[...,:60][...,~valid[j]].count_nonzero()
            agg=tool_aggregation(alpha,ad)
            tool_error=max(error(agg.flatten(-2),t['tool_motion'][0]),error(agg.flatten(-2),mi[:,128:140]))
            assert tool_error<2e-7 and delta_error<1e-4
            assert error(mi[:,:128],trace['anchor_features'][j][0])<1e-6
            assert error(mi[:,140:143],t['anchor_history'][0])==0
            assert error(mi[:,143:146],t['previous_d'][0])==0
            assert error(mi[:,146:150],t['head_history_mass'][0])<1e-6
            assert torch.all(mi[:,150]==int(j>0))
            assert error(t['state'][0],pred[j])==0
            if j==0:assert not agg.count_nonzero() and not t['anchor_history'].count_nonzero()
            else:assert error(t['anchor_history'],t['anchor_xyz']-trace['steps'][j-1]['anchor_xyz'])<1e-6
            window_errors.append(dict(horizon=h,feature_error=feature_error,delta_error_mm=delta_error,tool_error_normalized=tool_error,
                 condition_frame=int(cond[j]),target_frame=int(target[j]),valid_nodes=int(valid[j].sum()),adjacent_valid_nodes=int(am.sum()),
                 action_valid=raw['inputs']['action_valid'][0,j].tolist(),effective_hands=hv[0].tolist()))
            timeline.append(dict(window=window,horizon=h,condition_frame=int(cond[j]),target_frame=int(target[j]),
                 condition_timestamp_us=int(timestamps[j]),target_timestamp_us=int(timestamps[j+1]),prediction_dt_ms=float(timestamps[j+1]-timestamps[j])/1000,
                 observed_adjacent_dt_ms=None if j==0 else float(timestamps[j]-timestamps[j-1])/1000))
            cp_cumulative.append(wc.numpy());cp_adjacent.append(wa.numpy());tool_time.append(agg.numpy()*scale)
            current_controllers=[]
            for hand,name in enumerate(('left','right')):
                for finger,idx in [('all',np.arange(hand*30,(hand+1)*30)),('finger0',np.arange(hand*30,hand*30+15)),('finger1',np.arange(hand*30+15,(hand+1)*30))]:
                    for kind,v,m in [('cumulative',wc,cv),('adjacent',wa,am)]:
                        index=idx[m[idx].numpy()];s=stats(v[index].numpy());row=dict(window=window,horizon=h,condition_frame=int(cond[j]),target_frame=int(target[j]),hand=name,finger=finger,kind=kind,**s)
                        controllers.append(row)
                        if len(index):current_controllers.append((row,v[index].double().numpy().mean(0)))
            method_fields={'normal':displacement[j],**methods_by_h.get(h,{})}
            for method,u in method_fields.items():
                for region,ix in regions.items():
                    s=stats(u[ix]);objects.append(dict(window=window,horizon=h,target_frame=int(target[j]),method=method,region=region,**s))
                    for cr,mean in current_controllers:
                        co=cosine(mean,u[ix].mean(0))
                        directions.append(dict(window=window,horizon=h,method=method,region=region,hand=cr['hand'],finger=cr['finger'],controller_kind=cr['kind'],
                             mean_direction_cosine=co,direction_defined=co is not None,controller_mean_norm_mm=float(np.linalg.norm(mean))*1000,
                             object_mean_norm_mm=float(np.linalg.norm(u[ix].mean(0)))*1000))
            for head in range(4):
                for hand,name in [(None,'combined_model_input'),(0,'left'),(1,'right')]:
                    sl=slice(0,60) if hand is None else slice(hand*30,(hand+1)*30)
                    contrib=torch.einsum('aj,jd->ad',alpha[:,head,sl],ad[sl])*scale
                    tools.append(dict(window=window,horizon=h,condition_frame=int(cond[j]),target_frame=int(target[j]),head=head,hand_contribution=name,
                        null_weight_mean=float(alpha[:,head,60].mean()),valid_history_mass_mean=float(t['head_history_mass'][0,:,head].mean()),**stats(contrib.numpy())))
            pathway.append(dict(window=window,horizon=h,features13_norm=float(features.norm()),tokens62_norm=float(t['tokens62'].norm()),
                 cross_output_norm=float(mi[:,:128].norm()),tool12_norm=float(mi[:,128:140].norm()),anchor_history_norm=float(mi[:,140:143].norm()),
                 previous_control_norm=float(mi[:,143:146].norm()),cumulative_control_norm=float(t['cumulative_d'].norm())))
        np.savez_compressed(root/'fields'/f'{window}.npz',initial=initial,normal_displacement=displacement,controller_world=world.numpy(),controller_valid=valid.numpy(),
              controller_cumulative=np.stack(cp_cumulative),controller_adjacent=np.stack(cp_adjacent),tool_motion=np.stack(tool_time),near=near,middle=middle,far=far,
              condition_frame_ids=cond,target_frame_ids=target,**{f'h{h}_{m}':u for h,d in methods_by_h.items() for m,u in d.items()})
        checks.append(dict(window=window,passed=True,robot_slot_hands=metadata['robot_slot_hands'],physical_hand_to_slot=slots,
            operation_hands=x['operation_hands'][0].tolist(),original_episode_frames=frames,gs_valid_count=int(gs_keep.sum()),gs_padding_count=int((~gs_keep).sum()),
            source_controller_rebuild='bitwise equal for all 15 observed frames',source_batch='action/controller/mask/timestamp/age exactly equal',
            normalized_input='all inputs bitwise equal to saved zero-tool input (intervention acts only at Motion Decoder)',
            roundtrip_error_mm=max_roundtrip,initial_export_roundtrip_error_mm=initial_export_error_mm,opening_clipped_counts=clipped[cond].sum(0).tolist(),
            prediction_dt_ms_min=float(np.diff(timestamps).min()/1000),prediction_dt_ms_max=float(np.diff(timestamps).max()/1000),steps=window_errors))
        del raw,saved,stored_input,batch
    for p,row in zip(paths,hashes):assert sha(p)==row['sha256']
    for name,rows in [('controller_motion',controllers),('object_motion',objects),('signed_direction_comparison',directions),('tool_motion_alignment',tools),('timeline',timeline),('pathway_values',pathway)]:csv_write(root/'results'/f'{name}.csv',rows)
    sources={}
    for file,patterns in {'scripts/training/v4_model.py':['cp_delta =','tool_motion =','motion_input =','current = next_xyz'],
        'scripts/training/v4_conditions.py':['geom =','valid =','features =','return masked'],
        'scripts/data_process/processor.py':['def _xyz','for field, mask','position ='],
        'scripts/dataloader/dataset.py':['take = slice','for frame in range(start +','"frame_ids":'],
        'scripts/dataloader/geometry.py':['def hand_slots','def action_conditions'],
        'scripts/rebuild_dataset/rebuild.py':['def controller_arrays','finger = np.array','world = ('],
        'scripts/training/v4_runtime.py':['def prepare_batch','condition_frame_ids']}.items():
        lines=Path(file).read_text(encoding='utf8').splitlines();sources[file]=dict(sha256=sha(file),locations=[dict(line=i+1,text=l.strip()) for i,l in enumerate(lines) if any(p in l for p in patterns)])
    write(root/'results/coordinate_time_audit.json',dict(passed=True,processor=processor.state_dict(),windows=checks,source_locations=sources,
        confirmed_implementation_bugs=[],limits=['External camera/robot-world physical calibration not independently measured','Reconstructed source temporal semantics checked; original device clock synchronization not independently remeasured'],
        design=['A0 cumulative from frame t; tool12 uses adjacent observed delta','H1 adjacent delta absent, exactly zero','delta is displacement, not velocity; not divided by dt','two hands never averaged in controller diagnostic','Oracle only H5/H10/H15, no interpolation']))
    write(root/'provenance/input_hashes.json',hashes)
    write(root/'provenance/execution.json',dict(command=[sys.executable,*sys.argv],source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),device='cpu',
        network_forward=False,features_only=True,optimizer=False,backward=False,checkpoint=json.loads((ablation/'provenance/checkpoint.json').read_text())))
    print(json.dumps(dict(passed=True,windows=2,controller_rows=len(controllers),object_rows=len(objects),direction_rows=len(directions))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ablation',type=Path,required=True);p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.ablation,a.motion,a.output)
