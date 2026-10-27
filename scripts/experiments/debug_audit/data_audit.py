"""CPU-only data/condition audit; source data and checkpoints are never written.

python -B -m scripts.experiments.debug_audit.data_audit --root ROOT --output data_audit.json
The output must not already exist. This is a contract audit, not model inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from scripts.dataloader.dataset import Touch4DDataset, read_npz
from scripts.dataloader.collate import collate_fn
from scripts.data_process import DataProcessor
from scripts.data_process.check import check_processed_batch
from scripts.training.train import GS_CAMERAS
from scripts.experiments.spatial_dynamics.runner import ARCH
from scripts.experiments.spatial_dynamics.conditions import ToolPredictor, hand_inputs, sanitize


CASES = (("train", 2, 73), ("val", 3, 115), ("train", 1, 85))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def condition_evidence(inputs, model):
    """Inspect actual T3 input/quality routing; no trained response inference."""
    x = sanitize(inputs)
    mean, maximum, coverage, age, observed, max_observed = model.aggregate(x)
    tokens, keep = model.encode_conditions(x)
    b, h = x["action_state"].shape[:2]
    cp = x["controller_mask"].reshape(b, h, 4, 15)
    gate = cp & observed
    expected_fallback = x["valid_mask"].flatten(-2).any(-1) & ~gate.any(-1)
    assert torch.equal(keep[..., 62:], expected_fallback)
    assert torch.equal(keep[..., 2:62], cp.reshape(b, h, 60))
    assert torch.isfinite(tokens).all()
    assert (tokens[~keep] == 0).all()
    for sample in range(b):
        for hand in range(2):
            if not x["operation_hands"][sample, hand]:
                ids = [hand, *range(2 + hand * 30, 32 + hand * 30), 62 + hand * 2, 63 + hand * 2]
                assert not keep[sample, :, ids].any()
    features = model.controller_inputs(x)
    endpoints = x["controller_mask"] & x["controller_mask"][:, :1]
    assert torch.equal(features[..., 6].bool(), endpoints)
    expected = torch.where(endpoints[..., None], x["controller_points"] - x["controller_points"][:, :1], 0.)
    assert torch.equal(features[..., 3:6], expected)
    return {
        "operation_hands": x["operation_hands"].tolist(),
        "token_shape": list(tokens.shape), "token_valid_per_time": keep.sum(-1).tolist(),
        "action_valid": x["action_valid"].tolist(),
        "controller_valid_per_hand": x["controller_mask"].sum(-1).tolist(),
        "tactile_valid_per_finger": x["valid_mask"].flatten(-2).sum(-1).tolist(),
        "spatial_valid_per_finger": x["tactile_spatial_mask"].flatten(-2).sum(-1).tolist(),
        "tool_pressure_gate_per_finger": gate.sum(-1).tolist(),
        "fallback_valid": expected_fallback.tolist(),
        "mean_pressure_max_per_finger": mean.amax(-1).tolist(),
        "local_max_pressure_max_per_finger": maximum.amax(-1).tolist(),
        "coverage_mean_per_finger": coverage.mean(-1).tolist(),
        "normalized_age_per_finger": age.amax(-1).tolist(),
        "mean_observed_per_finger": observed.sum(-1).tolist(),
        "local_max_observed_per_finger": max_observed.sum(-1).tolist(),
        "valid_zero_taxels": int((x["valid_mask"] & (x["tactile"] == 0)).sum()),
        "positive_taxels": int((x["valid_mask"] & (x["tactile"] > 0)).sum()),
        "note": "Fresh CPU encoder only checks routing and masks; trained fusion/attention utilization is measured separately.",
    }


def audit_window(ds, episode, start, processor, encoder):
    horizon = ds.horizon
    info = ds.episodes[episode]
    if start < 0 or start + horizon >= info.frames:
        raise ValueError("Window crosses episode boundary")
    expected_frames = list(range(start, start + horizon + 1))
    paths = [info.path / name for name in ("metadata.json", "timestamps.npz", "controller.npz", "tactile.npz")]
    paths += [info.path / "gs" / f"frame_{frame:06d}.npz" for frame in expected_frames]
    if ds.with_depth:
        paths += [ds.depth_episodes[episode].path / "metadata.json", ds.depth_episodes[episode].path / "camera_parameters.npz"]
    hashes = {str(path): file_sha(path) for path in paths}
    key = (episode, start)
    if key not in ds.windows:
        ds.windows += (key,)
    sample = ds[ds.windows.index(key)]
    raw = collate_fn([sample])
    normalized = processor(raw)
    roundtrip = check_processed_batch(raw, normalized, processor)
    assert raw["meta"]["frame_ids"][0].tolist() == expected_frames
    assert np.array_equal(raw["meta"]["observation_timestamp_us"][0].numpy(), info.timestamps[start:start+horizon+1])
    # Independent source reads verify slices rather than relying only on loader metadata.
    core = read_npz(info.path / "controller.npz")
    tactile = read_npz(info.path / "tactile.npz")
    assert np.array_equal(raw["inputs"]["controller_points"][0].numpy(), core["controller_points"][start:start+horizon])
    assert np.array_equal(raw["inputs"]["tactile"][0].numpy(), tactile["tactile"][start:start+horizon])
    assert np.array_equal(raw["inputs"]["valid_mask"][0].numpy(), tactile["valid_mask"][start:start+horizon])
    counts = []
    for h, frame in enumerate(expected_frames):
        path = info.path / "gs" / f"frame_{frame:06d}.npz"
        disk = read_npz(path, ["xyz", "frame_id", "timestamp_us"])
        actual = raw["inputs"]["gs_xyz"][0] if h == 0 else raw["targets"]["target_xyz"][h-1][0]
        assert np.array_equal(actual.numpy(), disk["xyz"])
        assert int(disk["frame_id"]) == frame
        assert int(disk["timestamp_us"]) == int(info.timestamps[frame])
        counts.append(len(actual))
    x = hand_inputs(normalized["inputs"], normalized["meta"]["episode_id"], normalized["meta"]["scene"])
    condition = condition_evidence(x, encoder)
    sanitized = sanitize(x)
    physical_cp = raw["inputs"]["controller_points"]
    active = sanitized["controller_mask"]
    endpoints = active & active[:, :1]
    norm_delta = encoder.controller_inputs(sanitized)[..., 3:6]
    native_delta = torch.where(endpoints[..., None], physical_cp - physical_cp[:, :1], 0.)
    restored_delta = processor.denormalize_displacement(norm_delta, endpoints)
    torch.testing.assert_close(restored_delta, native_delta, atol=1e-7, rtol=1e-5)
    actions = raw["inputs"]["action_state"]
    for hand, slot in enumerate(info.slots):
        if slot is not None:
            expected = core["actions"][start:start+horizon, slot].reshape(horizon, 15)[:, :13].astype(np.float32)
            good = raw["inputs"]["action_valid"][0, :, hand].numpy()
            assert np.array_equal(actions[0, good, hand].numpy(), expected[good])
    initial = raw["inputs"]["gs_xyz"][0]
    centroid = initial.mean(0)
    centroid_changes = [(target[0].mean(0)-centroid).tolist() for target in raw["targets"]["target_xyz"]]
    depth = None
    if ds.with_depth:
        assert raw["targets"]["target_depth_valid"].flatten(-2).any(-1).all()
        depth = {"camera_ids": raw["meta"]["depth_camera_ids"][0],
                 "positive_pixels": raw["targets"]["target_depth_valid"].flatten(-2).sum(-1)[0].tolist(),
                 "target_shape": list(raw["targets"]["target_depth"].shape),
                 "units": "metres; cameras native world coordinates, not scene normalized"}
    for path, expected in hashes.items():
        if file_sha(path) != expected:
            raise AssertionError("Source changed during read-only audit: " + path)
    return {"split": ds.split, "episode": episode, "start": start, "horizon": horizon,
            "frame_ids": expected_frames, "condition_frames": expected_frames[:-1], "target_frames": expected_frames[1:],
            "timestamps_us": raw["meta"]["observation_timestamp_us"][0].tolist(),
            "tactile_age_us": raw["inputs"]["tactile_age_us"][0].tolist(),
            "physical_hand_to_slot": {"left": info.slots[0], "right": info.slots[1]},
            "point_counts": counts, "normalization": roundtrip, "conditions": condition,
            "controller_delta_roundtrip_max_m": float((restored_delta-native_delta).abs().max()),
            "controller_cumulative_displacement_mm_max": (native_delta.norm(dim=-1)*1000).amax(-1)[0].tolist(),
            "gt_independent_set_centroid_changes_m": centroid_changes,
            "depth": depth, "source_sha256": hashes,
            "limitations": "GT set centroid change is not material-particle motion; no cross-frame Gaussian ID correspondence is asserted."}


def run(root, output, with_depth=True):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    encoder = ToolPredictor(**ARCH).cpu().eval()
    processor = DataProcessor("001-rope")
    datasets = {split: Touch4DDataset("001-rope", split, root, horizon=5, with_depth=with_depth,
                    depth_cameras=GS_CAMERAS["001-rope"] if with_depth else None) for split in ("train", "val")}
    rows = [audit_window(datasets[split], episode, start, processor, encoder) for split, episode, start in CASES]
    for row in rows:
        for path, expected in row["source_sha256"].items():
            if file_sha(path) != expected:
                raise AssertionError("Source changed during read-only audit: " + path)
    result = {"status": "passed", "device": "cpu", "checkpoint_loaded": False,
              "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "processor": processor.state_dict(), "cases": rows,
              "timeline": "t..t+4 conditions -> t+1..t+5 targets; robot action/pose attached to the registered observation time, tactile latest-past age 0..75ms",
              "raw_robot_timestamp_limit": "Rebuilt data does not retain a separate robot acquisition time axis; audit verifies existing registered observation indexing, not independent sensor hardware synchronization.",
              "scientific_inference": "No deterministic data-contract failure in checked windows; does not rule out source reconstruction/measurement error or ineffective learned condition utilization."}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--without-depth", action="store_true")
    args = parser.parse_args()
    result = run(args.root, args.output, not args.without_depth)
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]), "device": "cpu"}))


if __name__ == "__main__":
    main()
