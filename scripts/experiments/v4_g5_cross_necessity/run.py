"""Val100 frozen Base/CrossValue/ToolMotion factorial. Never creates an optimizer.

Diagnostic infrastructure adapted from audit 18704f4; production predictor,
Dataset, Processor, Renderer, loss and GaussianMetrics are reused unchanged.
"""
import argparse
import csv
import gc
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
from scripts.training.gs_decoder import interpolate_anchor_displacements
from scripts.training.losses import nearest_distances
from scripts.training.train import make_renderer, copy_initial, to_device, GaussianMetrics
from scripts.training.v4_runtime import prepare_batch, v4_objective
from scripts.experiments.v4_g4_global_motion.evaluate import motion_metrics, fixed_regions
from scripts.experiments.v4_g4_global_motion.protocol import configuration, preflight
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.formal_checkpoint import load_for_evaluation
from scripts.experiments.v4_g5_signed_local.diagnostics import (
    sha256, state_hash, finite_tree, write_png)
from .interventions import (MODES, FACTORIAL_MODES, SELECTED, metadata,
                           intervention, replay_step, one_frame, branch_parts)
from .responses import counterfactual_inputs, ResponseUnavailable
from .gates import evaluate_gate
from .metrics import rms, decompose, aggregate, ANGLE_EPS_MM

BASE = "f978e0a29b6f0f0b6097c40684b949797a2f5d7b"
CHECKPOINT_HASH = "067984c4c286306094efeff2f28ed57e3dcdd0c333972ecc6feb4b982ee3a40c"
CASES = {(3,135), (3,245), (6,110), (7,105), (3,25), (6,20), (6,200)}
PICTURES = CASES
LOCKED = (
    "scripts/experiments/v4_g5_signed_local",
    "scripts/experiments/v4_g4_global_motion",
    "scripts/training", "scripts/dataloader", "scripts/data_process",
)


def write_json(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def csv_write(path, rows):
    if not rows:
        raise ValueError("Refuse empty result table: "+str(path))
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v
                             for k,v in row.items()})


class CachedRenderer:
    def __init__(self, renderer, target):
        self.renderer, self.target, self.last = renderer, target, None
    def render_targets(self, *unused):
        return self.target
    def __call__(self, *args):
        self.last = self.renderer(*args)
        finite_tree(self.last)
        return self.last


def horizon_rows(prediction, batch, values, normal, scale, mode, scope):
    keep = batch["inputs"]["gs_mask"][0]
    initial = batch["inputs"]["gs_xyz"][0, keep]
    factor, rows = scale*1000, []
    for k in range(15):
        p = prediction["xyz"][0,k,keep]
        q = batch["targets"]["target_xyz"][k][0,batch["targets"]["target_mask"][k][0]]
        cd = (nearest_distances(p,q,512).mean()+nearest_distances(q,p,512).mean())*.5
        rgb, alpha = float(values["rgb_by_horizon"][0,k]), float(values["alpha_by_horizon"][0,k])
        d = (p-initial)*factor
        mean = d.mean(0)
        change = (p-normal["xyz"][0,k,keep])*factor
        rows.append(dict(**scope, mode=mode, horizon=k+1, condition_frame=scope["start"]+k,
            target_frame=scope["start"]+k+1, cd_mm=float(cd)*factor, rgb=rgb, alpha=alpha,
            joint=float(cd)+rgb+.1*alpha, rms_mm=rms(d), demeaned_rms_mm=rms(d-mean),
            mean_x_mm=float(mean[0]), mean_y_mm=float(mean[1]), mean_z_mm=float(mean[2]),
            change_from_normal_rms_mm=rms(change)))
    actual = sum(r["cd_mm"] for r in rows)/15
    if abs(actual-float(values["geometry"][0])*factor) > 1e-4:
        raise AssertionError("Per-horizon CD reduction differs from original loss")
    return rows


def structure_rows(prediction, batch, scale, scope, mode):
    """Original 16NN/region statistics plus one-way regional set error."""
    x = batch['inputs']
    groups = fixed_regions(x)
    valid_ids = x['gs_mask'][0].nonzero().flatten()
    error_by_horizon = []
    for h in range(15):
        gt = batch['targets']['target_xyz'][h][0,batch['targets']['target_mask'][h][0]]
        # All groups contain valid GS indices; padding distances are never used.
        errors = nearest_distances(prediction['xyz'][0,h,valid_ids],gt,512)*scale*1000
        error_by_horizon.append({region:float(errors[torch.searchsorted(valid_ids,ids)].mean())
                                 for region,ids in groups.items()})
    return [dict(**scope,mode=mode,**row,
                 pred_region_to_full_gt_mm=error_by_horizon[row['horizon']-1][row['region']])
            for row in motion_metrics(prediction,x,scale)]


def single_effects(model, batch, normal, trace, raw_context, normal_rows, renderer, target, processor, scope, verify):
    """Recompute current Cross and all dependent Heads on fixed Normal history."""
    rows, checks, regional = [], [], []
    x = batch["inputs"]
    keep, factor = x["gs_mask"][0], processor.state_dict()["scale"]*1000
    groups = fixed_regions(x)
    for h in SELECTED:
        index = h-1
        replay = replay_step(model,x,trace,raw_context,index,"Normal")
        torch.testing.assert_close(replay["xyz"],normal["xyz"][:,index],rtol=0,atol=0)
        view, original = one_frame(batch,normal,index)
        gt_image = {k:v[:,index:index+1] for k,v in target.items()}
        reference = normal_rows[index]
        q = batch["targets"]["target_xyz"][index][0,batch["targets"]["target_mask"][index][0]]
        normal_error = nearest_distances(normal["xyz"][0,index,keep],q,512)*factor
        compact = torch.full_like(keep,-1,dtype=torch.long)
        compact[keep] = torch.arange(int(keep.sum()),device=keep.device)
        # Reuse Loss only after actual XYZ bit-equality; no model result is assumed equivalent.
        cache_values = [(normal["xyz"][:,index],tuple(reference[k] for k in ("cd_mm","rgb","alpha","joint")),"Normal")]
        for mode in MODES[1:]:
            replay = replay_step(model,x,trace,raw_context,index,mode)
            pos = replay["xyz"]
            prediction = dict(original,xyz=pos[:,None])
            equivalent = next(((v,n) for p,v,n in cache_values if torch.equal(pos,p)),None)
            if equivalent is None:
                cache = CachedRenderer(renderer,gt_image)
                loss,m = v4_objective(prediction,view,processor,cache)
                finite_tree((loss,m))
                if not bool(m["render_time_available"].all()):
                    raise ValueError("Missing single-step render supervision")
                values = (float(m["geometry"][0])*factor,float(m["rgb"][0]),float(m["alpha"][0]),float(loss))
                cache_values.append((pos,values,mode)); rendered_from=mode
                del cache
            else:
                values,rendered_from=equivalent
            difference=(pos-normal["xyz"][:,index])[0,keep]*factor
            row=dict(**scope,horizon=h,mode=mode,history="fixed_Normal",
                     **dict(zip(("cd_mm","rgb","alpha","joint"),values)),
                     change_rms_mm=rms(difference),change_x_mm=float(difference[:,0].mean()),
                     change_y_mm=float(difference[:,1].mean()),change_z_mm=float(difference[:,2].mean()),
                     equal_geometry_render_reused_from=rendered_from)
            for key in ("cd_mm","rgb","alpha","joint"):
                row["delta_"+key]=row[key]-reference[key]
            row["cd_improved"] = row["delta_cd_mm"] < -1e-6
            row["cd_worsened"] = row["delta_cd_mm"] > 1e-6
            rows.append(row)
            error=nearest_distances(pos[0,keep],q,512)*factor
            delta=(pos[0]-x["gs_xyz"][0])*factor
            for region,ids in groups.items():
                cids=compact[ids]
                regional.append(dict(**scope,horizon=h,mode=mode,region=region,history="fixed_Normal",
                    pred_region_to_full_gt_mm=float(error[cids].mean()),
                    delta_region_to_full_gt_mm=float((error[cids]-normal_error[cids]).mean()),
                    point_error_improved_fraction=float(((error[cids]-normal_error[cids]) < -1e-6).double().mean()),
                    point_error_worsened_fraction=float(((error[cids]-normal_error[cids]) > 1e-6).double().mean()),
                    rms_mm=rms(delta[ids]),change_from_normal_rms_mm=rms((pos[0]-normal["xyz"][0,index])[ids]*factor)))
            if verify:
                with intervention(model,mode,only_step=index):
                    actual=model(x)
                torch.testing.assert_close(actual["xyz"][:,:index],normal["xyz"][:,:index],rtol=0,atol=0)
                torch.testing.assert_close(actual["xyz"][:,index],pos,rtol=0,atol=0)
                checks.append(dict(**scope,horizon=h,mode=mode,prefix_bit_exact=True,current_step_bit_exact=True))
    return rows,checks,regional


def response_effects(model,batch,normal,renderer,target,processor,scope):
    """Input-only counterfactuals; factual GT does NOT become counterfactual truth."""
    rows,arrays=[],{}
    x=batch["inputs"]; keep=x["gs_mask"][0]; factor=processor.state_dict()["scale"]*1000
    for branch in ("Normal","NoBase+NoAttentionPaths"):
        if branch=="Normal":reference=normal
        else:
            with intervention(model,branch):reference=model(x)
        # The repeated frozen input measures the actual numerical repeat floor.
        with intervention(model,branch):repeat=model(x)
        repeat_floor=rms((repeat["xyz"][0,:,keep]-reference["xyz"][0,:,keep]).reshape(-1,3)*factor)
        for mode in ("FreezeAction","HalfRootTranslation"):
            try:
                changed,construction=counterfactual_inputs(x,mode)
            except ResponseUnavailable as exc:
                # Missing initial support is a data limitation, not a successful
                # zero response and not permission to change the H1 contract.
                for h in range(15):
                    rows.append(dict(**scope,mode=branch,response=mode,horizon=h+1,
                        available=False,output_change_rms_mm=None,repeat_floor_mm=repeat_floor,
                        measurable_above_numeric_floor=False,mean_x_mm=None,mean_y_mm=None,
                        mean_z_mm=None,rms_mm=None,factual_gt_joint_only=None,
                        counterfactual_ground_truth_available=False,construction=exc.diagnostics))
                continue
            with intervention(model,branch):prediction=model(changed)
            finite_tree(prediction)
            torch.testing.assert_close(prediction["xyz"][:,0],reference["xyz"][:,0],rtol=0,atol=0)
            loss,m=v4_objective(prediction,dict(batch,inputs=changed),processor,CachedRenderer(renderer,target))
            finite_tree((loss,m))
            for h in range(15):
                d=(prediction["xyz"][0,h,keep]-x["gs_xyz"][0,keep])*factor
                diff=(prediction["xyz"][0,h,keep]-reference["xyz"][0,h,keep])*factor
                rows.append(dict(**scope,mode=branch,response=mode,horizon=h+1,
                    available=True,output_change_rms_mm=rms(diff),repeat_floor_mm=repeat_floor,
                    measurable_above_numeric_floor=rms(diff)>max(1e-6,10*repeat_floor),
                    mean_x_mm=float(d[:,0].mean()),mean_y_mm=float(d[:,1].mean()),mean_z_mm=float(d[:,2].mean()),
                    rms_mm=rms(d),factual_gt_joint_only=float(loss),counterfactual_ground_truth_available=False,
                    construction=construction))
            arrays[branch+"_"+mode+"_world"]=(prediction["xyz"][0,:,keep]*processor.state_dict()["scale"]+
                prediction["xyz"].new_tensor(processor.state_dict()["center"])).cpu().numpy()
    return rows,arrays


def save_pictures(directory, name, images):
    for key in ("rgb","alpha"):
        for h in SELECTED:
            for camera in range(3):
                value = images[key][0,h-1,camera].cpu().numpy()
                if value.ndim == 2:
                    value = np.repeat(value[...,None],3,-1)
                write_png(directory/f"h{h:02}_v{camera}_{key}_{name}.png",
                          np.rint(np.clip(value,0,1)*255).astype(np.uint8))


@torch.inference_mode()
def execute(args, out):
    source = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    if source != args.expected_sha or subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise ValueError("Requires frozen clean diagnostic source")
    if subprocess.check_output(["git","diff",BASE,"--",*LOCKED],text=True).strip():
        raise ValueError("Production/model files differ from frozen training source")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Requires an allocated Slurm GPU")
    if torch.cuda.device_count()!=1 or '5090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Requires one Slurm-visible RTX 5090; no other allocation may be reused')
    if (torch.__version__!='2.7.1+cu128' or torch.version.cuda!='12.8'
            or importlib.metadata.version('gsplat')!='1.4.0'):
        raise RuntimeError('Frozen CUDA/gsplat environment differs; do not silently replace it')
    if sha256(args.checkpoint) != CHECKPOINT_HASH:
        raise ValueError("Wrong frozen Epoch8 Best checkpoint")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = DataProcessor("001-rope")
    # Exact G4/G0 loader/manifest protocol; no truncated or filtered Val.
    config = configuration(args.root,args.depth_root,BASE)
    raw_loaders,data = preflight(config,args.reference)
    manifest = data["manifest"]
    model = G5DirectPredictor(horizon=15,prediction_interval=1)
    state = load_for_evaluation(args.checkpoint,model,processor.state_dict(),source_sha=BASE,manifest=manifest)
    if state["ledger"]["completed_epochs"] != 8 or state["ledger"]["successful_updates"] != 12112:
        raise ValueError("Expected Epoch8 Best with 12112 successful updates")
    expected = state["ledger"]["history"][-1]["validation"]["model"]
    model.cuda().eval().requires_grad_(False)
    digest = state_hash(model.state_dict())
    provenance = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=CHECKPOINT_HASH,
        training_source_sha=BASE, diagnostic_source_sha=source, selected_epoch=8, selected_updates=12112,
        expected_frozen_validation=expected, manifest=data, command=sys.argv, job_id=os.environ["SLURM_JOB_ID"],
        python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        gsplat=importlib.metadata.version("gsplat"),gpu=torch.cuda.get_device_name(),processor=processor.state_dict(),
        angle_epsilon_mm=ANGLE_EPS_MM,optimizer_created=False,backward_called=False,
        modes=list(MODES),mode_definitions=[metadata(mode) for mode in MODES],
        factorial_modes=list(FACTORIAL_MODES),single_step_horizons=list(SELECTED),
        parameter_count=sum(p.numel() for p in model.parameters()),
        full_factorial=True,strict_head_mass_closure=True,
        intervention_note="Full 2x2x2 + H2-only + strict mass closure; projection-only CrossValue suppression retains FFN; current-step replay recomputes dependent Heads")
    del state
    write_json(out/"provenance_initial.json",provenance)
    write_json(out/"manifest.json",data)
    renderer = make_renderer(config)
    scale = processor.state_dict()["scale"]
    # Fail early on the real trained weights before an expensive Val100 pass.
    pilot_ds = raw_loaders["val"].dataset
    pilot_raw = collate_fn([pilot_ds[pilot_ds.windows.index((3,135))]])
    pilot_batch = to_device(prepare_batch(pilot_raw,processor,model),"cuda")
    pilot_x = pilot_batch["inputs"]
    bare = model(pilot_x)
    with intervention(model):
        observed,pilot_trace = model(pilot_x,return_trace=True)
    for key in bare:
        torch.testing.assert_close(bare[key],observed[key],rtol=0,atol=0)
    with intervention(model,"Normal") as pilot_context:
        observed,pilot_trace=model(pilot_x,return_trace=True)
    for index in (0,4,9,14):
        for mode in MODES:
            replay=replay_step(model,pilot_x,pilot_trace,pilot_context,index,mode)
            with intervention(model,mode,only_step=index):changed=model(pilot_x)
            torch.testing.assert_close(changed["xyz"][:,:index],observed["xyz"][:,:index],rtol=0,atol=0)
            torch.testing.assert_close(changed["xyz"][:,index],replay["xyz"],rtol=0,atol=0)
    with intervention(model,"NoBase+NoAttentionPaths"):joint=model(pilot_x)
    perturbed=dict(pilot_x,tactile=pilot_x["tactile"]+123.)
    with intervention(model,"NoBase+NoAttentionPaths"):joint_changed=model(perturbed)
    torch.testing.assert_close(joint["xyz"],joint_changed["xyz"],rtol=0,atol=0)
    _,_,pilot_equivalence = decompose(pilot_x,pilot_trace,scale)
    pilot_view,pilot_pred = one_frame(pilot_batch,observed,4)
    pilot_loss,pilot_m = v4_objective(pilot_pred,pilot_view,processor,renderer)
    finite_tree((pilot_loss,pilot_m))
    if not bool(pilot_m["render_time_available"].all()):
        raise ValueError("Real H5 pilot render is incomplete")
    write_json(out/"real_preflight.json",dict(episode=3,start=135,all_attributes_bit_exact=True,
        cross_replay_matches_only_step=True,joint_has_no_tactile_output_path=True,source_sha=source,optimizer_updates=0,
        frame_ids=pilot_raw["meta"]["frame_ids"].tolist(),rbf_equivalence=pilot_equivalence,
        h5_joint=float(pilot_loss),h5_cd_mm=float(pilot_m["geometry"][0])*scale*1000,
        camera_count=pilot_view["targets"]["target_depth"].shape[2],
        image_height=pilot_view["targets"]["target_depth"].shape[3],
        image_width=pilot_view["targets"]["target_depth"].shape[4]))
    del pilot_batch,pilot_x,bare,observed,pilot_trace,changed,pilot_view,pilot_pred,pilot_m,pilot_loss
    del pilot_context,replay,joint,joint_changed,perturbed
    gc.collect()
    trackers = {name:GaussianMetrics(scale,config) for name in (*MODES,"Copy")}
    tables = {name:[] for name in ("windows","horizons","same_history","branch_fields","cancellation",
                                  "structure","equivalence","intervention_checks","same_history_regions","action_response")}
    seen = []
    clock = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def process(raw, split):
        offsets = {key:len(rows) for key,rows in tables.items()}
        ep = int(raw["meta"]["episode_id"][0])
        start = int(raw["meta"]["frame_ids"][0,0])
        scope = dict(split=split,episode=ep,start=start)
        if raw["meta"]["frame_ids"].tolist() != [list(range(start,start+16))]:
            raise ValueError("Wrong H15/interval1 frames")
        case, picture = (ep,start) in CASES, (ep,start) in PICTURES
        batch = to_device(prepare_batch(raw,processor,model),"cuda")
        x = batch["inputs"]
        if x["operation_hands"].tolist() != [[ep>=5,True]]:
            raise ValueError("Hand Mask mismatch")
        target = renderer.render_targets(batch,processor)
        finite_tree(target)
        with intervention(model,"Normal") as raw_context:
            normal,trace = model(x,return_trace=True)
        finite_tree((normal,trace))
        # H1 G4/G5 have zero observed signed motion; do NOT assume Base zero.
        for key in ("global_delta","local_correction"):
            if torch.count_nonzero(trace["steps"][0][key]):
                raise AssertionError("H1 signed branch is not zero")
        branch,cancel,equiv = decompose(x,trace,scale)
        for name,rows in (("branch_fields",branch),("cancellation",cancel),("equivalence",equiv)):
            tables[name].extend(dict(**scope,**row) for row in rows)
        cache = CachedRenderer(renderer,target)
        loss, normal_m = v4_objective(normal,batch,processor,cache)
        finite_tree((loss,normal_m))
        normal_rows = horizon_rows(normal,batch,normal_m,normal,scale,"Normal",scope)
        if picture:
            directory = out/"images"/f"{split}_ep{ep}_s{start}"
            directory.mkdir()
            save_pictures(directory,"GT",target)
            save_pictures(directory,"Normal",cache.last)
        cache.last = None
        direct,checks,regional = single_effects(model,batch,normal,trace,raw_context,normal_rows,renderer,target,processor,scope,case)
        tables["same_history_regions"].extend(regional)
        tables["same_history"].extend(direct)
        tables["intervention_checks"].extend(checks)
        if case:
            response,response_arrays = response_effects(model,batch,normal,renderer,target,processor,scope)
            tables["action_response"].extend(response)
            keep = x["gs_mask"][0]
            center = x["gs_xyz"].new_tensor(processor.state_dict()["center"])
            arrays = dict(initial_world=(x["gs_xyz"][0,keep]*scale+center).cpu().numpy(),
                frame_ids=raw["meta"]["frame_ids"][0].numpy(),valid_ids=keep.nonzero().flatten().cpu().numpy(),
                anchor_initial_world=(x["gs_xyz"][0,trace["field"].anchor_indices[0]]*scale+center).cpu().numpy(),
                anchor_valid=trace["field"].anchor_mask[0].cpu().numpy())
            arrays.update(response_arrays)
            for name,value in x.items():
                if isinstance(value,torch.Tensor):arrays["input_"+name]=value.cpu().numpy()
            for name,ids in fixed_regions(x).items():
                # Store indices into the compact VALID GS array.
                compact = torch.searchsorted(keep.nonzero().flatten(),ids)
                arrays["region_"+name] = compact.cpu().numpy()
            for h,s in enumerate(trace["steps"],1):
                parts = branch_parts(s,trace["field"])
                for name,part in parts.items():
                    arrays[f"anchor_{name}_h{h:02}"] = (part[0]*scale).cpu().numpy()
                    arrays[f"gs_{name}_h{h:02}"] = (interpolate_anchor_displacements(
                        part,trace["field"],keep[None])[0,keep]*scale).cpu().numpy()
                gt = batch["targets"]["target_xyz"][h-1][0,batch["targets"]["target_mask"][h-1][0]]
                arrays[f"gt_h{h:02}_world"] = (gt*scale+center).cpu().numpy()
        per_window = []
        base_off_references = {}
        for mode in (*MODES,"Copy"):
            if mode == "Normal":
                pred, m, current_loss = normal, normal_m, loss
            else:
                if mode == "Copy":
                    pred = copy_initial(x,"gs_geometry")
                else:
                    with intervention(model,mode):
                        pred = model(x)
                finite_tree(pred)
                if mode == "NoBase-H2+":
                    torch.testing.assert_close(pred["xyz"][:,0],normal["xyz"][:,0],rtol=0,atol=0)
                if mode in ("NoBase-All","NoBase+NoCrossValue","NoBase+NoToolMotion12","NoBase+NoAttentionPaths"):
                    torch.testing.assert_close(pred["xyz"][:,0],x["gs_xyz"],rtol=0,atol=0)
                if mode in ("NoBase-All","NoBase+NoCrossValue"):
                    base_off_references[mode] = pred['xyz']
                if mode in ("NoBase+NoToolMotion12","NoBase+NoAttentionPaths"):
                    reference_mode = ('NoBase-All' if mode=='NoBase+NoToolMotion12'
                                      else 'NoBase+NoCrossValue')
                    torch.testing.assert_close(pred['xyz'],base_off_references[reference_mode],rtol=0,atol=0)
                for key in pred:
                    if key != "xyz":
                        torch.testing.assert_close(pred[key],normal[key],rtol=0,atol=0)
                current_loss,m = v4_objective(pred,batch,processor,cache)
                finite_tree((current_loss,m))
                if picture:
                    save_pictures(directory,mode,cache.last)
                cache.last = None
            if not bool(m["render_time_available"].all()):
                raise ValueError("Incomplete three-camera supervision")
            if split == "val":
                trackers[mode].add(batch["meta"]["episode_id"],m)
            record = dict(**scope,mode=mode,cd_mm=float(m["geometry"][0])*scale*1000,
                          rgb=float(m["rgb"][0]),alpha=float(m["alpha"][0]),joint=float(current_loss))
            per_window.append(record)
            tables["windows"].append(record)
            tables["horizons"].extend(normal_rows if mode=="Normal" else
                                     horizon_rows(pred,batch,m,normal,scale,mode,scope))
            # Full Val16NN output is required, not just visually selected cases.
            tables["structure"].extend(structure_rows(pred,batch,scale,scope,mode))
            if case:
                arrays[mode+"_world"] = (pred["xyz"][0,:,keep]*scale+center).cpu().numpy()
        if case:
            np.savez_compressed(out/"arrays"/f"{split}_ep{ep}_s{start}.npz",**arrays)
        write_json(out/"windows"/f"{split}_ep{ep}_s{start}.json",dict(
            summary=per_window,
            evidence={name:rows[offsets[name]:] for name,rows in tables.items()}))
        if not all(not p.requires_grad and p.grad is None for p in model.parameters()):
            raise AssertionError("Unexpected parameter/gradient state")
        del batch,normal,trace,target,cache
        gc.collect()

    for raw in raw_loaders["val"]:
        key = [int(raw["meta"]["episode_id"][0]),int(raw["meta"]["frame_ids"][0,0])]
        seen.append(key)
        process(raw,"val")
        if len(seen)%10 == 0:
            print("VAL_COMPLETE",len(seen),"/100",round(time.perf_counter()-clock,1),"seconds",flush=True)
    if seen != manifest["val"]["window_keys"] or len(seen)!=100:
        raise AssertionError("Val100 order/count mismatch")
    ds = raw_loaders["train"].dataset
    process(collate_fn([ds[ds.windows.index((7,105))]]),"train")
    summary = {mode:tracker.summary() for mode,tracker in trackers.items()}
    for mode,values in summary.items():
        if {ep:r["samples"] for ep,r in values["episodes"].items()} != {"3":55,"6":45}:
            raise AssertionError("Incomplete Val100: "+mode)
    for key,tolerance in (("geometry_mm",1e-4),("objective",1e-6),("rgb",1e-6),("alpha",1e-6)):
        if abs(summary["Normal"][key]-expected[key]) > tolerance:
            raise AssertionError(("Frozen Normal reproduction failed",key,summary["Normal"][key],expected[key]))
    for key,rows in tables.items():
        write_json(out/"results"/f"{key}.json",rows)
        csv_write(out/"results"/f"{key}.csv",rows)
    write_json(out/"results"/"val_summary.json",summary)
    val_csv = []
    for mode,s in summary.items():
        for ep,row in [("episode_equal",s),*s["episodes"].items()]:
            record = {k:v for k,v in row.items() if k in ("geometry_mm","rgb","alpha","samples")}
            record["joint"] = (s["objective"] if ep=="episode_equal" else
                               row["supervised_geometry"]+row["rgb"]+.1*row["alpha"])
            val_csv.append(dict(mode=mode,episode=ep,**record))
    csv_write(out/"results"/"val_summary.csv",val_csv)
    horizon_summary = aggregate(tables["horizons"],("mode","horizon"),
        ("cd_mm","rgb","alpha","joint","rms_mm","mean_x_mm","mean_y_mm","mean_z_mm","change_from_normal_rms_mm"))
    write_json(out/"results"/"horizon_val_summary.json",horizon_summary)
    csv_write(out/"results"/"horizon_val_summary.csv",horizon_summary)
    aggregated = aggregate(tables["same_history"],("mode","horizon"),
                           ("cd_mm","rgb","alpha","joint","delta_cd_mm","delta_rgb","delta_alpha","delta_joint","change_rms_mm","cd_improved","cd_worsened"))
    write_json(out/"results"/"same_history_val_summary.json",aggregated)
    csv_write(out/"results"/"same_history_val_summary.csv",aggregated)
    aggregate_cancel = aggregate(tables["cancellation"],("space","region","horizon"),
        ("base_vs_rest_cos_mean","base_vs_rest_opposite_fraction","cancellation_index",
         "base_rms_mm","correction_rms_mm","final_rms_mm","base_to_final_rms",
         "base_larger_than_final_fraction","nearly_cancelled_fraction"))
    write_json(out/"results"/"cancellation_val_summary.json",aggregate_cancel)
    csv_write(out/"results"/"cancellation_val_summary.csv",aggregate_cancel)
    if state_hash(model.state_dict()) != digest or sha256(args.checkpoint)!=CHECKPOINT_HASH:
        raise AssertionError("Frozen weights or checkpoint changed")
    torch.cuda.synchronize()
    provenance.update(elapsed_seconds=time.perf_counter()-clock,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        model_hash_before=digest,model_hash_after=state_hash(model.state_dict()),
        checkpoint_hash_after=sha256(args.checkpoint),val_seen=seen,
        parameters_frozen=True,optimizer_updates=0)
    write_json(out/"provenance.json",provenance)
    write_json(out/"results"/"gate_preliminary.json",evaluate_gate(summary,[r for r in tables["horizons"] if r["split"]=="val"]))
    write_json(out/"completed.json",dict(state="INFERENCE_COMPLETED",val100=True,train_cases=1,
        checkpoint_unchanged=True,model_unchanged=True,source_sha=source,seconds=provenance["elapsed_seconds"]))
    print("FROZEN_NECESSITY_INFERENCE_COMPLETED",out,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root","depth-root","reference","checkpoint","output","expected-sha"):
        p.add_argument("--"+name,required=True)
    args = p.parse_args()
    out = Path(args.output).resolve()
    protected = [Path(v).resolve() for v in (args.root,args.depth_root,args.reference,Path.cwd())]
    if out.exists() or out.parent.name!="v4-g5-cross-necessity-v1" or any(
        out==v or v in out.parents for v in protected):
        raise ValueError("Requires a new isolated attribution Run directory")
    out.mkdir(parents=True,exist_ok=False)
    for name in ("results","arrays","images","windows","logs","figures"):
        (out/name).mkdir()
    try:
        execute(args,out)
    except BaseException:
        write_json(out/"failure.json",dict(traceback=traceback.format_exc(),command=sys.argv))
        raise


if __name__ == "__main__":
    main()
