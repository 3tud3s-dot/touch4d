"""CPU data preflight for the frozen 008 episode_0 temporal training protocol."""

import argparse
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.data_process.processor import DataProcessor, _Parameters
from scripts.dataloader.collate import collate_fn
from scripts.dataloader.dataset import Touch4DDataset, read_gs
from .cloth_protocol import configuration as g5_configuration, fresh_model, probe_order

BASE_SHA = "f978e0a29b6f0f0b6097c40684b949797a2f5d7b"
SCENE = "008-pink-cloth"
EPISODE = 0
HORIZON = 15
FRAME_RANGES = MappingProxyType({"train": (113, 268), "val": (268, 307)})
# The first three cameras of the existing SoMA 008 Config C, for data preflight.
CAMERAS = ("brics-odroid-023_cam0", "brics-odroid-021_cam1", "brics-odroid-009_cam1")
EPISODE_METADATA_SHA = "e7179d622176bf0e41960699c425003c9bd9d6d7d0b66af486b3b7816cd124e1"
NORMALIZATION = _Parameters(
    (0.21795742779917135, 0.010968994921301998, 0.14623293373089752),
    0.20123094573217368,
)
HAND_POLICY = "008-ep0-reviewed-right-v1"


class Cloth008Processor(DataProcessor):
    """Use the existing transforms with 008 constants from training-window initial GS only."""

    _scene_parameters = MappingProxyType({SCENE: NORMALIZATION})


def configuration(root, depth_root, sha):
    config = g5_configuration(root, depth_root, sha, "direct")
    config.update(scene=SCENE, depth_cameras=list(CAMERAS), operation_hand_policy=HAND_POLICY,
                  episode=EPISODE, frame_ranges=dict(FRAME_RANGES), window_stride=1)
    return config


def make_dataset(config, split):
    lo, hi = FRAME_RANGES[split]
    dataset = Touch4DDataset(
        SCENE, split, config["root"], horizon=HORIZON, prediction_interval=1,
        with_depth=True, depth_root=config["depth_root"], depth_cameras=CAMERAS,
        episode_frame_range=(EPISODE, lo, hi),
    )
    info = dataset.episodes[EPISODE]
    if info.slots != (None, 0):
        raise ValueError("Expected reviewed right-hand-only metadata for 008 episode_0")
    metadata_hash = hashlib.sha256((info.path / "metadata.json").read_bytes()).hexdigest()
    if metadata_hash != EPISODE_METADATA_SHA:
        raise ValueError("008 episode_0 metadata differs from the reviewed normalization inputs")
    return dataset


def make_loader(dataset, epoch=0):
    return DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn,
                      shuffle=dataset.split == "train", drop_last=False,
                      generator=torch.Generator().manual_seed(42 + epoch))


def validate_normalization(dataset):
    if dataset.split != "train":
        raise ValueError("Normalization only uses training-window initial GS")
    info = dataset.episodes[EPISODE]
    files = json.loads((info.path / "metadata.json").read_text(encoding="utf-8"))["output_files"]
    point_sum = np.zeros(3)
    point_count = 0
    radii = []
    for _, frame in dataset.windows:
        rel = f"gs/frame_{frame:06d}.npz"
        if hashlib.sha256((info.path / rel).read_bytes()).hexdigest() != files[rel]:
            raise ValueError(f"Train initial GS hash differs at {frame}")
        xyz = read_gs(info, frame)["xyz"].astype(np.float64)
        point_sum += xyz.sum(0)
        point_count += len(xyz)
        radii.append(np.quantile(np.linalg.norm(xyz - xyz.mean(0), axis=1), .99))
    np.testing.assert_allclose(point_sum / point_count, NORMALIZATION.center, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.median(radii), NORMALIZATION.scale, rtol=0, atol=1e-12)
    return point_count


def prepare_batch(raw, processor, model):
    """Supply the reviewed one-hand flag to the original G5 model and normalize the batch."""
    meta, inputs = raw["meta"], raw["inputs"]
    ids = meta["frame_ids"]
    times = meta["observation_timestamp_us"]
    expected = ids[:, :1] + torch.arange(HORIZON + 1, device=ids.device)
    valid_ranges = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)
    for lo, hi in FRAME_RANGES.values():
        valid_ranges |= (ids[:, 0] >= lo) & (ids[:, -1] < hi)
    if (model.config["horizon"] != HORIZON or model.config["prediction_interval"] != 1
            or not torch.equal(ids, expected) or not valid_ranges.all()
            or not (times[:, 1:] > times[:, :-1]).all()
            or any(scene != SCENE for scene in meta["scene"])
            or not (meta["episode_id"] == EPISODE).all()):
        raise ValueError("Expected a consecutive H15 window from 008 episode_0")
    hands = torch.tensor([False, True], device=ids.device).expand(ids.size(0), 2)
    if not torch.equal(inputs["action_valid"], hands[:, None].expand(-1, HORIZON, -1)):
        raise ValueError("Action validity disagrees with the reviewed right-hand-only episode")
    batch = processor(raw)
    batch["inputs"]["operation_hands"] = hands.clone()
    batch["inputs"]["condition_frame_ids"] = ids[:, :-1].clone()
    batch["meta"]["prediction_interval"] = 1
    batch["meta"]["operation_hand_policy"] = HAND_POLICY
    return batch


def preflight(root, depth_root):
    torch.set_num_threads(4)
    config = configuration(root, depth_root, BASE_SHA)
    datasets = {split: make_dataset(config, split) for split in FRAME_RANGES}
    processor = Cloth008Processor(SCENE)
    fitted_points = validate_normalization(datasets["train"])
    model = fresh_model("direct")
    model.eval()
    result = {}
    for split, dataset in datasets.items():
        lo, hi = FRAME_RANGES[split]
        keys = tuple((EPISODE, start) for start in range(lo, hi - HORIZON))
        if dataset.windows != keys:
            raise ValueError(f"{split}: frame manifest differs")
        order = probe_order(make_loader(dataset))
        if sorted(order) != list(range(len(keys))) or (split == "val" and order != list(range(len(keys)))):
            raise ValueError(f"{split}: loader order differs")
        samples = []
        for index in (0, len(keys) - 1):
            raw = collate_fn([dataset[index]])
            start = keys[index][1]
            if raw["meta"]["frame_ids"].tolist() != [list(range(start, start + HORIZON + 1))]:
                raise ValueError("Collated frame IDs differ from manifest")
            depth_valid = raw["targets"]["target_depth_valid"]
            if (tuple(depth_valid.shape) != (1, HORIZON, len(CAMERAS), 720, 1280)
                    or not depth_valid.flatten(-2).any(-1).all()):
                raise ValueError("Missing depth support or incorrect original G5 render resolution")
            batch = prepare_batch(raw, processor, model)
            sample = {"first_frame": start, "last_frame": start + HORIZON,
                      "points": int(batch["inputs"]["gs_mask"].sum()),
                      "valid_tactile_values": int(batch["inputs"]["valid_mask"].sum())}
            if index == 0:
                with torch.inference_mode():
                    prediction = model(batch["inputs"])
                if (tuple(prediction["xyz"].shape) != (1, HORIZON, sample["points"], 3)
                        or any(not torch.isfinite(value).all() for value in prediction.values())):
                    raise ValueError("Unexpected real-data G5 CPU forward result")
                sample["cpu_h15_forward"] = True
            samples.append(sample)
        result[split] = {"windows": len(keys), "target_bounds": [lo+1, hi-1], "samples": samples}
    result.update(base_sha=BASE_SHA, processor=processor.state_dict(), normalization_points=fitted_points,
                  camera_preflight=list(CAMERAS), model_parameters=sum(p.numel() for p in model.parameters()),
                  optimizer_updates=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--depth-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(preflight(args.root, args.depth_root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
