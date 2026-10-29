"""Frozen H15 inference: Val100 plus exactly Train ep7/start105. No optimizer."""
import argparse
import csv
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
from scripts.training.v4_model import V4Stage1Predictor
from scripts.training.v4_runtime import prepare_batch,load_checkpoint,v4_objective
from scripts.training.train import make_renderer,copy_initial,to_device,GaussianMetrics,atomic_json
from scripts.training.losses import nearest_distances
from scripts.experiments.full_training.v4_epochs import loader
from scripts.experiments.v4_g4_global_motion.model import G4GlobalMotionPredictor
from scripts.experiments.v4_g4_global_motion.checkpoint import load_for_evaluation
from scripts.experiments.v4_g4_global_motion.evaluate import motion_metrics
from scripts.experiments.v4_g4_global_motion.protocol import configuration,preflight
from scripts.experiments.v4_g4_global_motion.smoke import sha,finite_tree
from .interventions import MODES,intervention,direct_positions,model_digest

BASE='8cba442736b81d05e9785a5963345d3fe2c35131'
G4_HASH='240045e3ccb0a37bfcc954feabb2b61f4a1f828d63566eb856619ebe6335b92d'
G0_HASH='e349a3711c61afe25787b15861b72aba1b5ce7b7997128c1559dd8166ad12c5e'
ALL_MODES=MODES+('G0','Copy')


def csv_write(path,rows):
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


class CachedTargets:
    def __init__(self,renderer,batch,processor):
        self.renderer=renderer;self.target=renderer.render_targets(batch,processor);finite_tree(self.target)
    def render_targets(self,*args):return self.target
    def __call__(self,*args):
        value=self.renderer(*args);finite_tree(value);return value


def rms(x):return float(x.double().square().sum(-1).mean().sqrt())


def horizon_metrics(pred,batch,m,normal,scale,ep,start,mode):
    keep=batch['inputs']['gs_mask'][0];initial=batch['inputs']['gs_xyz'][0,keep];factor=scale*1000
    rows=[];cds=[]
    for h in range(15):
        pos=pred['xyz'][0,h,keep];gt=batch['targets']['target_xyz'][h][0,batch['targets']['target_mask'][h][0]]
        cd=(nearest_distances(pos,gt,512).mean()+nearest_distances(gt,pos,512).mean())*.5;cds.append(cd)
        delta=(pos-initial)*factor;change=(pos-normal['xyz'][0,h,keep])*factor
        mean=delta.mean(0);difference=change.mean(0)
        rows.append(dict(episode=ep,start=start,horizon=h+1,target_frame=start+h+1,condition_frame=start+h,mode=mode,
            cd_mm=float(cd)*factor,rgb=float(m['rgb_by_horizon'][0,h]),alpha=float(m['alpha_by_horizon'][0,h]),
            mean_x_mm=float(mean[0]),mean_y_mm=float(mean[1]),mean_z_mm=float(mean[2]),
            rms_mm=rms(delta),residual_rms_mm=rms(delta-mean),change_from_normal_rms_mm=rms(change),
            change_from_normal_x_mm=float(difference[0]),change_from_normal_y_mm=float(difference[1]),change_from_normal_z_mm=float(difference[2])))
    torch.testing.assert_close(torch.stack(cds).mean(),m['geometry'][0],rtol=1e-6,atol=1e-8)
    return rows


def direct_effects(g4,x,normal,trace,scale,ep,start,verify=False):
    keep=x['gs_mask'][0];factor=scale*1000;rows=[];checks=[]
    for h in (1,5,10,15):
        s=trace['steps'][h-1]
        torch.testing.assert_close(direct_positions(x,trace,h-1,'Normal'),normal['xyz'][:,h-1],rtol=0,atol=0)
        for mode in MODES[1:]:
            direct=direct_positions(x,trace,h-1,mode);difference=(normal['xyz'][:,h-1]-direct)[0,keep]
            removed=s['global_delta'][0] if mode=='NoGlobal' else s['hand_contribution'][0,0 if mode=='NoLeft' else 1]
            error=float((difference-removed).abs().max())*factor
            if error>1e-4:raise AssertionError(('Direct RBF is not uniform translation',error))
            if h==1 and torch.count_nonzero(difference):raise AssertionError('H1 branch must be zero')
            if verify:
                with intervention(g4,mode,only_step=h-1):actual=g4(x)
                torch.testing.assert_close(actual['xyz'][:,:h-1],normal['xyz'][:,:h-1],rtol=0,atol=0)
                torch.testing.assert_close(actual['xyz'][:,h-1],direct,rtol=0,atol=0)
                checks.append(dict(horizon=h,mode=mode,prefix_bit_exact=True,current_step_bit_exact=True))
            vector=difference.mean(0)*factor
            rows.append(dict(episode=ep,start=start,horizon=h,mode=mode,
                on_minus_off_x_mm=float(vector[0]),on_minus_off_y_mm=float(vector[1]),on_minus_off_z_mm=float(vector[2]),
                on_minus_off_rms_mm=rms(difference)*factor,uniform_translation_max_error_mm=error,
                left_valid=bool(s['hand_valid'][0,0]),right_valid=bool(s['hand_valid'][0,1])))
    return rows,checks


def traces(trace,scale,ep,start,mode):
    rows=[];factor=scale*1000
    for h,s in enumerate(trace['steps'],1):
        raw=s.get('raw_hand_contribution',s['hand_contribution']);effective=s['hand_contribution']
        if torch.count_nonzero(effective[~s['hand_valid']]):raise AssertionError('Masked hand leak')
        if h==1 and torch.count_nonzero(s['global_delta']):raise AssertionError('H1 global not zero')
        torch.testing.assert_close(effective.sum(1),s['global_delta'],rtol=0,atol=0)
        rows.append(dict(episode=ep,start=start,horizon=h,mode=mode,
            hand_valid=s['hand_valid'].tolist(),raw_hand_contribution_mm=(raw*factor).cpu().tolist(),
            effective_hand_contribution_mm=(effective*factor).cpu().tolist(),global_delta_mm=(s['global_delta']*factor).cpu().tolist(),
            hand_basis_mm=(s['hand_basis']*factor).cpu().tolist()))
    return rows


@torch.inference_mode()
def run(a,out):
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source!=a.expected_sha or subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Source drift')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Requires Slurm GPU')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    hashes={a.g4:sha(a.g4),a.g0:sha(a.g0)}
    if hashes[a.g4]!=G4_HASH or hashes[a.g0]!=G0_HASH:raise ValueError('Frozen checkpoint hash mismatch')
    proc=DataProcessor('001-rope');g4=G4GlobalMotionPredictor(horizon=15);g0=V4Stage1Predictor(horizon=15)
    s4=load_for_evaluation(a.g4,g4,proc.state_dict());s0=load_checkpoint(a.g0,g0,processor_state=proc.state_dict())
    if s4['identity']['source_sha']!=BASE:raise ValueError('G4 source mismatch')
    provenance=dict(g4_selected_epoch=s4['ledger']['completed_epochs'],g4_selected_updates=s4['ledger']['successful_updates'],
                    g0_selected_updates=s0['successful_steps'],g0_selected_epoch=(s0.get('run_state') or {}).get('completed_epochs'),
                    checkpoint_hashes=hashes,model_base_sha=BASE,diagnostic_sha=source,command=sys.argv,job_id=os.environ['SLURM_JOB_ID'])
    del s4,s0
    g4.cuda().eval().requires_grad_(False);g0.cuda().eval().requires_grad_(False)
    initial_weights={'g4':model_digest(g4),'g0':model_digest(g0)}
    c=configuration(a.root,a.depth_root,source);raw_loaders,manifest=preflight(c,a.reference)
    atomic_json(out/'manifest.json',manifest);renderer=make_renderer(c);scale=proc.state_dict()['scale']
    trackers={m:GaussianMetrics(scale,c) for m in ALL_MODES}
    window_rows=[];horizon_rows=[];direct_rows=[];trace_rows=[];local_rows=[];checks=[];seen=[]
    torch.cuda.reset_peak_memory_stats();clock=time.perf_counter()
    def process(raw,split):
        ep=int(raw['meta']['episode_id'][0]);start=int(raw['meta']['frame_ids'][0,0]);diagnostic=(ep,start) in ((3,135),(7,105))
        batch=to_device(prepare_batch(raw,proc,g4),'cuda');x=batch['inputs'];cache=CachedTargets(renderer,batch,proc)
        normal,normal_trace=g4(x,return_trace=True);finite_tree(normal);finite_tree(normal_trace)
        direct,verified=direct_effects(g4,x,normal,normal_trace,scale,ep,start,verify=diagnostic)
        direct_rows.extend(direct);checks.extend([dict(episode=ep,start=start,**r) for r in verified])
        if diagnostic:
            keep=x['gs_mask'][0];center=x['gs_xyz'].new_tensor(proc.state_dict()['center'])
            arrays=dict(initial=(x['gs_xyz'][0,keep]*scale+center).cpu().numpy(),valid_ids=keep.nonzero().flatten().cpu().numpy(),
                        frame_ids=raw['meta']['frame_ids'][0].numpy(),
                        controller_initial=(x['controller_points'][0,0].reshape(60,3)*scale+center).cpu().numpy())
            for h in range(15):arrays[f'gt_h{h+1:02d}']=(batch['targets']['target_xyz'][h][0,batch['targets']['target_mask'][h][0]]*scale+center).cpu().numpy()
        for mode in ALL_MODES:
            if mode=='Normal':pred,trace=normal,normal_trace
            elif mode in MODES:
                with intervention(g4,mode):pred,trace=g4(x,return_trace=True)
            elif mode=='G0':pred,trace=g0(x),None
            else:pred,trace=copy_initial(x,'gs_geometry'),None
            finite_tree(pred)
            if mode in MODES:
                if torch.count_nonzero(pred['xyz'][:,0]-normal['xyz'][:,0]):raise AssertionError('H1 ablation differs')
                if ep==3 and mode=='NoLeft':
                    for k in pred:torch.testing.assert_close(pred[k],normal[k],rtol=0,atol=0)
                if ep==3 and mode=='NoGlobal':no_global=pred
                if ep==3 and mode=='NoRight':
                    for k in pred:torch.testing.assert_close(pred[k],no_global[k],rtol=0,atol=0)
                trace_rows.extend(traces(trace,scale,ep,start,mode))
            loss,m=v4_objective(pred,batch,proc,cache);finite_tree(m)
            if not m['render_time_available'].all():raise ValueError('Incomplete supervision')
            if split=='val':trackers[mode].add(batch['meta']['episode_id'],m)
            window_rows.append(dict(split=split,episode=ep,start=start,mode=mode,joint=float(loss),cd_mm=float(m['geometry'][0])*scale*1000,
                                    rgb=float(m['rgb'][0]),alpha=float(m['alpha'][0])))
            horizon_rows.extend(horizon_metrics(pred,batch,m,normal,scale,ep,start,mode))
            if diagnostic:
                arrays[mode]=(pred['xyz'][0,:,keep]*scale+center).cpu().numpy()
                for r in motion_metrics(pred,x,scale):local_rows.append(dict(episode=ep,start=start,mode=mode,**r))
            del pred,trace,loss,m
        if diagnostic:
            # Shared G4 weights in original G0 class must reproduce NoGlobal,
            # but are NOT the original independently trained G0 checkpoint.
            shared=V4Stage1Predictor(horizon=15).cuda().eval().requires_grad_(False)
            shared.load_state_dict({k:v for k,v in g4.state_dict().items() if not k.startswith('global_motion.')},strict=True)
            with intervention(g4,'NoGlobal'):ng=g4(x)
            shared_output=shared(x)
            for k in ng:torch.testing.assert_close(shared_output[k],ng[k],rtol=0,atol=0)
            checks.append(dict(episode=ep,start=start,shared_g4_in_g0_class_equals_no_global=True))
            np.savez_compressed(out/'arrays'/f'ep{ep}_start{start}.npz',**arrays)
            del shared,shared_output,ng,arrays
        if not all(not p.requires_grad and p.grad is None for p in g4.parameters()):raise AssertionError('Unexpected gradient state')
    for raw in raw_loaders['val']:
        key=[int(raw['meta']['episode_id'][0]),int(raw['meta']['frame_ids'][0,0])];seen.append(key)
        process(raw,'val')
        if len(seen)%20==0:print('VAL',len(seen),'/100',flush=True)
    if seen!=manifest['manifest']['val']['window_keys'] or len(seen)!=100:raise AssertionError('Val traversal mismatch')
    ds=raw_loaders['train'].dataset;process(collate_fn([ds[ds.windows.index((7,105))]]),'train')
    summary={m:t.summary() for m,t in trackers.items()}
    for m in summary:
        if {k:r['samples'] for k,r in summary[m]['episodes'].items()}!={'3':55,'6':45}:raise AssertionError('Val counts')
    for mode,expected in [('Normal',.11899972560006486),('G0',.14134251669322076)]:
        if abs(summary[mode]['objective']-expected)>1e-6:raise AssertionError(('Frozen baseline reproduction drift',mode,summary[mode]['objective']))
    if initial_weights!={'g4':model_digest(g4),'g0':model_digest(g0)}:raise AssertionError('Weights changed')
    if any(sha(p)!=v for p,v in hashes.items()):raise AssertionError('Checkpoint changed')
    torch.cuda.synchronize()
    provenance.update(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        elapsed_seconds=time.perf_counter()-clock,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),model_hashes_before_after=initial_weights,
        checkpoint_hashes_unchanged=True,parameters_unchanged=True,optimizer_created=False,backward_called=False)
    import gsplat
    provenance['gsplat']=gsplat.__version__
    atomic_json(out/'provenance.json',provenance);atomic_json(out/'summary.json',summary)
    atomic_json(out/'single_step_checks.json',checks)
    for name,rows in [('windows',window_rows),('horizons',horizon_rows),('direct_effects',direct_rows),('hand_contributions',trace_rows),('local_motion',local_rows)]:
        atomic_json(out/'results'/f'{name}.json',rows);csv_write(out/'results'/f'{name}.csv',rows)
    atomic_json(out/'status.json',dict(state='COMPLETED',val_windows=100,diagnostic_train_windows=1,
                                     source_sha=source,model_base_sha=BASE,weights_unchanged=True))
    print('COMPLETED',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','depth-root','reference','g4','g0','output','expected-sha'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'results').mkdir();(out/'arrays').mkdir()
    try:run(a,out)
    except BaseException:
        atomic_json(out/'failure.json',dict(traceback=traceback.format_exc(),command=sys.argv,source_sha=a.expected_sha));raise


if __name__=='__main__':main()
