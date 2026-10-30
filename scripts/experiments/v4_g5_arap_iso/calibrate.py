"""One downward-only ARAP/ISO-weight calibration on two frozen train windows."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.training.train import atomic_json
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.formal_checkpoint import load_for_evaluation
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash,sha256,finite_tree
from .protocol import (BASE,REFERENCE,REFERENCE_HASH,TRAIN_WINDOWS,configuration,preflight,
    source_guard,resource_guard,environment,component_hashes,calibration_read,configure_runtime,calibrated_weights,RULE)
from .runner import CachedRenderer,prepared,original_objective,frozen_attributes,assert_real_batch
from .structure import build_structure_graph
from .arap import combined_regularizers
from .runner import diagnostic_plain,SolverFallbackError




def run(args):
    configure_runtime();source,files=source_guard(args.expected_sha)
    out=Path(args.output)
    if not out.is_absolute():raise ValueError('Absolute new calibration directory required')
    final=out/'calibration.json'
    if final.exists():
        d=calibration_read(final);print(json.dumps(dict(reused=True,lambda_arap=d['lambda_arap'],lambda_iso=d['lambda_iso'],source_sha=d['source_sha'])));return
    resource=resource_guard(require_idle=True);env=environment()
    if sha256(REFERENCE)!=REFERENCE_HASH:raise ValueError('Frozen reference checkpoint hash mismatch')
    config=configuration(args.root,args.depth_root,source,'g5_arap',.01,.001)
    raw,data=preflight(config,args.g0_reference);processor=DataProcessor('001-rope')
    model=G5DirectPredictor(horizon=15,prediction_interval=1)
    loaded=load_for_evaluation(REFERENCE,model,processor.state_dict(),source_sha=BASE,manifest=data['manifest']);del loaded
    model.eval().requires_grad_(False).cuda();before=state_hash(model.state_dict())
    identity=dict(reference_sha256=REFERENCE_HASH,reference_source_sha=BASE,components=component_hashes(),
                  manifest_hash=data['manifest_hash'],windows=[list(w) for w in TRAIN_WINDOWS],rule=RULE)
    journal=out/'calibration_partial.json';records=[]
    if out.exists() and not journal.exists():raise FileExistsError('Existing calibration directory is not an owned partial calibration')
    out.mkdir(parents=True,exist_ok=True)
    if journal.exists():
        old=json.loads(journal.read_text())
        if old['identity']!=identity:raise ValueError('Partial calibration identity/math changed')
        records=old['records']
        if [(r['episode'],r['start']) for r in records]!=list(TRAIN_WINDOWS)[:len(records)] or len(records)>2:
            raise ValueError('Invalid partial calibration order')
    args._owned_output=True
    renderer=CachedRenderer()
    for ep,start in TRAIN_WINDOWS[len(records):]:
        ds=raw['train'].dataset
        batch=prepared(collate_fn([ds[ds.windows.index((ep,start))]]),processor,model,'cuda');x=batch['inputs']
        assert_real_batch(batch,ep,start)
        if batch['meta']['frame_ids'].tolist()!=[list(range(start,start+16))] or x['condition_frame_ids'].tolist()!=[list(range(start,start+15))]:
            raise AssertionError('Real calibration time IDs differ')
        with torch.no_grad():original=model(x)
        frozen_attributes(original,x)
        xyz=original['xyz'].detach().requires_grad_(True);prediction=dict(original,xyz=xyz)
        graph=build_structure_graph(x['gs_xyz'],x['gs_mask'])
        joint,metrics=original_objective(prediction,batch,processor,renderer,config)
        grad_joint=torch.autograd.grad(joint,xyz)[0]
        arap,iso,statistics=combined_regularizers(xyz,x['gs_xyz'],graph,capture_solver_failures=True)
        if statistics['kabsch_fallback_count'].sum()>0:
            failure=SolverFallbackError(statistics)
            token=f'solver_failure_ep{ep}_start{start}_'+str(__import__('os').getpid())
            with (out/(token+'.npz')).open('xb') as stream:
                np.savez_compressed(stream,initial_normalized=x['gs_xyz'].detach().cpu().numpy(),
                    predicted_normalized=xyz.detach().cpu().numpy(),mask=x['gs_mask'].cpu().numpy(),
                    edges=graph.edges.cpu().numpy(),edge_mask=graph.edge_mask.cpu().numpy(),
                    rest_lengths=graph.rest_lengths.cpu().numpy())
            atomic_json(out/(token+'.json'),dict(episode=ep,start=start,source_sha=source,
                graph=graph.stats,solver_diagnostics=failure.diagnostics,
                losses=diagnostic_plain(dict(joint=joint,arap=arap,iso=iso)),
                note='Original fit failed; stopped before regularizer backward and before calibration weights.'))
            raise failure
        grad_arap=torch.autograd.grad(arap,xyz,retain_graph=True)[0]
        grad_iso=torch.autograd.grad(iso,xyz)[0]
        finite_tree((grad_joint,grad_arap,grad_iso,joint,arap,iso))
        gj=float(grad_joint.double().norm());ga=float(grad_arap.double().norm());gi=float(grad_iso.double().norm())
        row=dict(episode=ep,start=start,joint=float(joint.detach()),arap=float(arap.detach()),iso=float(iso.detach()),
            joint_gradient_norm=gj,arap_gradient_norm=ga,iso_gradient_norm=gi,
            candidate_arap_weight=.01,candidate_iso_weight=.001,
            graph=graph.stats[0],regularizer_statistics=diagnostic_plain(statistics),frame_ids=batch['meta']['frame_ids'].tolist(),
            condition_frame_ids=x['condition_frame_ids'].tolist())
        array=out/f'graph_ep{ep}_start{start}.npz'
        if array.exists():raise FileExistsError('Unexpected completed graph array without calibration record')
        with array.open('xb') as stream:
            np.savez_compressed(stream,initial_world=processor.denormalize_xyz(x['gs_xyz'],x['gs_mask'])[0].detach().cpu().numpy(),
                initial_normalized=x['gs_xyz'][0].detach().cpu().numpy(),mask=x['gs_mask'][0].cpu().numpy(),
                edges=graph.edges[0,graph.edge_mask[0]].cpu().numpy(),
                rest_lengths=graph.rest_lengths[0,graph.edge_mask[0]].cpu().numpy())
        row['graph_array_sha256']=sha256(array)
        records.append(row);atomic_json(journal,dict(identity=identity,source_sha=source,records=records))
        print(json.dumps(row,allow_nan=False),flush=True)
        renderer.clear();del batch,original,prediction,xyz,graph,joint,arap,iso,grad_joint,grad_arap,grad_iso,statistics
        gc.collect();torch.cuda.empty_cache()
    weights=calibrated_weights(records)
    if sha256(REFERENCE)!=REFERENCE_HASH or state_hash(model.state_dict())!=before:raise AssertionError('Read-only calibration changed reference')
    result=dict(passed=True,**identity,source_sha=source,source_files=files,reference_path=REFERENCE,
        **weights,records=records,resource=resource,environment=env,command=sys.argv,
        reference_state_hash=before,reference_unchanged=True,optimizer_created=False,successful_updates=0,
        note='XYZ output-space L2 gradient norms, normalized coordinates. Only these two train windows; no validation tuning.')
    atomic_json(final,result);calibration_read(final)
    print(json.dumps(dict(calibration_passed=True,**weights)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','depth-root','g0-reference','output','expected-sha'):p.add_argument('--'+name,required=True)
    args=p.parse_args()
    try:run(args)
    except BaseException as error:
        import traceback
        out=Path(args.output)
        if getattr(args,'_owned_output',False) and out.is_dir():atomic_json(out/('failure-'+str(__import__('os').getpid())+'.json'),dict(source_sha=args.expected_sha,
            traceback=traceback.format_exc(),solver_diagnostics=getattr(error,'diagnostics',None),
            note='No zero/nonfinite calibration gradient can pass silently; inspect retained graph/solver records.'))
        raise


if __name__=='__main__':main()
