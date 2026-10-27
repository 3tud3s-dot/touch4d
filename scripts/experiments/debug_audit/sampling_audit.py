"""CPU-only motion coverage of the original frozen 24 training windows.

No checkpoint, learned model, rasterizer or GPU is used. Additional prescribed
diagnostic cases are reported separately, never added to the training list.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.dataloader.dataset import Touch4DDataset
from scripts.dataloader.splits import SPLITS
from scripts.experiments.spatial_dynamics.conditions import hand_inputs, sanitize
from scripts.training.losses import chamfer_samples
from scripts.training.train import copy_initial


DIAGNOSTICS = (("train", 2, 73), ("train", 1, 85), ("val", 3, 115))


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def frozen_rows(manifest):
    rows = manifest["windows"]["5"]["train"]
    if len(rows) != 24:
        raise ValueError("Expected the original 24 frozen training windows")
    keys = [(row["episode"], row["start"]) for row in rows]
    if len(set(keys)) != 24:
        raise ValueError("Duplicate original training window")
    counts = {ep: sum(key[0] == ep for key in keys) for ep in SPLITS["001-rope"]["train"]}
    if set(ep for ep, _ in keys) != set(counts) or any(count != 4 for count in counts.values()):
        raise ValueError("Frozen list must have four windows from each of six training episodes")
    for row in rows:
        e, s = row["episode"], row["start"]
        if type(e) is not int or type(s) is not int or s < 0:
            raise ValueError("Invalid frozen episode/start")
        if row.get("frame_ids", list(range(s,s+6))) != list(range(s,s+6)):
            raise ValueError("Frozen H5 frame IDs are not consecutive")
    return rows


def quantiles(values):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(values):
        return dict(count=0, values=None)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite motion coverage statistic")
    levels = [0., .1, .25, .5, .75, .9, 1.]
    return dict(count=len(values), method="numpy.quantile linear; one summary per window",
                values={name: float(value) for name, value in zip(
                    ("min", "p10", "p25", "median", "p75", "p90", "max"), np.quantile(values, levels))})


@torch.no_grad()
def window_statistics(batch, processor, centered_cd=False):
    if any(v.device.type != "cpu" for v in batch["inputs"].values() if isinstance(v, torch.Tensor)):
        raise ValueError("Sampling audit must run on CPU")
    inputs = sanitize(hand_inputs(batch["inputs"], batch["meta"]["episode_id"], batch["meta"]["scene"]))
    initial_mask = inputs["gs_mask"][0]
    initial = inputs["gs_xyz"][0, initial_mask]
    prediction = copy_initial(inputs, "gs_geometry")["xyz"]
    if len(initial) == 0 or prediction.shape[:2] != (1, 5):
        raise ValueError("Sampling audit requires a nonempty, batch-one H5 window")
    units = float(processor.state_dict()["scale"])*1000
    cd, centered, centers, counts = [], [], [], [len(initial)]
    for h in range(5):
        xyz = batch["targets"]["target_xyz"][h]
        mask = batch["targets"]["target_mask"][h]
        target = xyz[0, mask[0]]
        if not len(target):
            raise ValueError("Empty independently reconstructed target GS")
        minimal = dict(inputs=inputs, targets=dict(target_xyz=[xyz], target_mask=[mask]))
        cd.append(float(chamfer_samples(prediction[:, h:h+1], minimal).mean()*units))
        centers.append(((target.mean(0)-initial.mean(0))*units).tolist())
        counts.append(len(target))
        if centered_cd:
            p = prediction[:,h:h+1] - initial.mean(0)
            t = torch.where(mask[...,None], xyz-target.mean(0), 0.)
            centered_batch = dict(inputs=inputs, targets=dict(target_xyz=[t], target_mask=[mask]))
            centered.append(float(chamfer_samples(p, centered_batch).mean()*units))
    valid = inputs["controller_mask"] & inputs["controller_mask"][:, :1]
    delta = torch.where(valid[...,None], inputs["controller_points"]-inputs["controller_points"][:,:1], 0.)
    norms = delta[0].norm(dim=-1)*units
    def summary(keep, values):
        selected = values[keep]
        return dict(count=int(keep.sum()), rms_mm=float(selected.square().mean().sqrt()) if len(selected) else None,
                    max_mm=float(selected.max()) if len(selected) else None)
    cp = [summary(valid[0,h], norms[h]) for h in range(5)]
    per_hand = [[summary(valid[0,h,hand], norms[h,hand]) for hand in range(2)] for h in range(5)]
    centroid_norm = np.linalg.norm(np.asarray(centers), axis=-1).tolist()
    return dict(copy_cd_mm=cd, mean_copy_cd_mm=float(np.mean(cd)), h5_copy_cd_mm=cd[-1],
                gt_independent_set_centroid_vector_mm=centers,
                gt_independent_set_centroid_norm_mm=centroid_norm,
                mean_gt_set_centroid_norm_mm=float(np.mean(centroid_norm)),
                h5_gt_set_centroid_norm_mm=centroid_norm[-1],
                centered_copy_cd_mm=centered if centered_cd else None,
                controller_cumulative=cp, controller_cumulative_by_hand=per_hand,
                condition_t4_controller_rms_mm=cp[-1]["rms_mm"],
                max_controller_displacement_mm=max((row["max_mm"] for row in cp if row["max_mm"] is not None), default=None),
                point_counts=counts, target_point_count_ratio=[count/counts[0] for count in counts[1:]],
                point_count_delta=[count-counts[0] for count in counts[1:]],
                operation_hands=inputs["operation_hands"][0].tolist(),
                condition_frames=batch["meta"]["frame_ids"][0,:-1].tolist(),
                target_frames=batch["meta"]["frame_ids"][0,1:].tolist())


def audit_row(ds, row, processor, hashes, centered_cd=False):
    ep, start = row["episode"], row["start"]
    if ep not in ds.episodes or start < 0 or start+5 >= ds.episodes[ep].frames:
        raise ValueError("Window does not belong to its declared split/episode")
    info = ds.episodes[ep]
    paths = [info.path/name for name in ("metadata.json", "timestamps.npz", "controller.npz", "tactile.npz")]
    paths += [info.path/"gs"/f"frame_{frame:06d}.npz" for frame in range(start,start+6)]
    for path in paths:
        if str(path) not in hashes:
            hashes[str(path)] = file_sha(path)
    key = (ep,start)
    if key not in ds.windows:
        # Only a temporary read index for an explicitly requested diagnostic;
        # the manifest and the original 24 training rows are never changed.
        ds.windows = ds.windows + (key,)
    raw = collate_fn([ds[ds.windows.index(key)]])
    frames = raw["meta"]["frame_ids"][0].tolist()
    if frames != list(range(start,start+6)):
        raise ValueError("Loaded frame IDs disagree with frozen start")
    if row.get("timestamps_us", raw["meta"]["observation_timestamp_us"][0].tolist()) != raw["meta"]["observation_timestamp_us"][0].tolist():
        raise ValueError("Observation timestamps changed from frozen manifest")
    result = window_statistics(processor(raw), processor, centered_cd)
    if row.get("initial_points", result["point_counts"][0]) != result["point_counts"][0]:
        raise ValueError("Initial GS count differs from manifest")
    if row.get("target_points", result["point_counts"][1:]) != result["point_counts"][1:]:
        raise ValueError("Target GS counts differ from manifest")
    return dict(split=ds.split, episode=ep, start=start, frame_ids=frames, **result)


def run(root, manifest_path, output, centered_cd=False):
    output, manifest_path = Path(output), Path(manifest_path)
    if output.exists():
        raise FileExistsError(output)
    manifest_hash = file_sha(manifest_path)
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = frozen_rows(source)
    keys = {(row["episode"], row["start"]) for row in rows}
    torch.set_num_threads(4)
    processor = DataProcessor("001-rope")
    datasets = {split: Touch4DDataset("001-rope", split, root, horizon=5,
                        with_depth=False, cache_episodes=8) for split in ("train", "val")}
    hashes = {}
    training = [audit_row(datasets["train"], row, processor, hashes, centered_cd) for row in rows]
    lookup = {(row["episode"],row["start"]):row for row in training}
    diagnostic = []
    for split, ep, start in DIAGNOSTICS:
        key = (ep,start)
        value = (dict(lookup[key]) if key in lookup else
                 audit_row(datasets[split], dict(episode=ep,start=start), processor, hashes, centered_cd))
        value.update(in_original_train24=key in keys, role="diagnostic_only; not newly selected training data")
        diagnostic.append(value)
    columns = ("mean_copy_cd_mm", "h5_copy_cd_mm", "mean_gt_set_centroid_norm_mm", "h5_gt_set_centroid_norm_mm",
               "condition_t4_controller_rms_mm", "max_controller_displacement_mm")
    distribution = {name:quantiles([row[name] for row in training]) for name in columns}
    for path, digest in hashes.items():
        if file_sha(path) != digest:
            raise RuntimeError("Read-only data changed during coverage audit: "+path)
    if file_sha(manifest_path) != manifest_hash:
        raise RuntimeError("Original manifest changed during coverage audit")
    result = dict(status="complete", device="cpu", threads=4, with_depth=False,
                  model_loaded=False, checkpoint_loaded=False, horizon=5,
                  source_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                  manifest=str(manifest_path.resolve()), manifest_sha256=manifest_hash,
                  original_train24=training, diagnostic_cases=diagnostic,
                  train_window_quantiles=distribution, processor=processor.state_dict(),
                  source_sha256=hashes, source_hashes_unchanged=True,
                  centered_copy_cd_computed=bool(centered_cd),
                  depth_digest_revalidated=False,
                  limitations=["Independent GS set centroid/Chamfer changes are not material-particle displacement.",
                    "Controller differences use only t..t+4; targets cover t+1..t+5. Hand masks follow declared episodes.",
                    "Quantiles are descriptive per-window summaries; overlapping frames/windows are not independent trials.",
                    "No training data was added or removed. These diagnostics cannot establish causal sampling failure.",
                    "Depth/render supervision is not loaded or revalidated in this CPU coverage audit."])
    with output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--centered-cd",action="store_true",help="Also compute independent centered-set Chamfer on CPU")
    args = p.parse_args()
    value = run(args.root,args.manifest,args.output,args.centered_cd)
    print(json.dumps(dict(status=value["status"],original_windows=len(value["original_train24"]),
                         diagnostics=len(value["diagnostic_cases"]),device="cpu")))


if __name__ == "__main__":
    main()
