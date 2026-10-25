"""Right-pad real GS points per batch, independently at each target time."""

import torch

from .splits import HORIZON, validate_horizon

FIXED_FIELDS = {
    "action_state": ((5, 2, 13), torch.float32),
    "action_valid": ((5, 2), torch.bool),
    "controller_points": ((5, 2, 30, 3), torch.float32),
    "controller_mask": ((5, 2, 30), torch.bool),
    "tactile": ((5, 4, 12, 32), torch.float32),
    "valid_mask": ((5, 4, 12, 32), torch.bool),
    "tactile_xyz": ((5, 4, 12, 32, 3), torch.float32),
    "tactile_spatial_mask": ((5, 4, 12, 32), torch.bool),
    "tactile_age_us": ((5, 4), torch.int64),
}
DEPTH_FIELDS = ("target_depth", "target_depth_valid", "camera_intrinsics", "camera_to_world")


def fixed_fields(horizon=HORIZON):
    """H-dependent shapes, retaining the public H=5 FIXED_FIELDS constant."""
    validate_horizon(horizon)
    return {key: ((horizon, *shape[1:]), dtype) for key, (shape, dtype) in FIXED_FIELDS.items()}


def batch_horizon(batch):
    """Infer H from existing metadata; no new sample/checkpoint fields required."""
    frames = batch["meta"]["frame_ids"]
    if (not isinstance(frames, torch.Tensor) or frames.ndim != 2
            or frames.dtype != torch.int64 or frames.shape[0] < 1):
        raise ValueError("frame_ids must be int64 [B,H+1]")
    return validate_horizon(frames.shape[1]-1)


def validate_target_attr_batch(batch):
    """Optional geometry targets share XYZ padding; legacy point-only batches work."""
    targets = batch["targets"]
    if "target_attr" not in targets:
        return
    horizon = batch_horizon(batch)
    attrs = targets["target_attr"]
    if not isinstance(attrs, list) or len(attrs) != horizon:
        raise ValueError(f"target_attr must be a list of {horizon} tensors")
    for step, attr in enumerate(attrs):
        xyz, mask = targets["target_xyz"][step], targets["target_mask"][step]
        if (not isinstance(attr, torch.Tensor) or attr.shape != (*xyz.shape[:-1], 11)
                or attr.dtype != torch.float32 or attr.device != xyz.device):
            raise ValueError(f"target_attr[{step}] must align with target_xyz as float32 [B,N,11]")
        if mask.shape != xyz.shape[:-1] or mask.dtype != torch.bool or mask.device != xyz.device:
            raise ValueError(f"target_mask[{step}] must align with target attributes")
        lengths = targets["target_lengths"][:, step].to(device=mask.device)
        if not torch.equal(mask.sum(1), lengths):
            raise ValueError(f"target_attr[{step}] lengths disagree with the target mask")
        if not torch.isfinite(attr).all() or torch.any(attr[~mask] != 0):
            raise ValueError(f"target_attr[{step}] must be finite with zero padding")
        quaternions = attr[..., 3:7][mask]
        norms = torch.linalg.vector_norm(quaternions.double(), dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0):
            raise ValueError(f"target_attr[{step}] requires normalized WXYZ quaternions")


def _has_depth(targets, meta):
    present = [key in targets for key in DEPTH_FIELDS] + ["depth_camera_ids" in meta]
    if any(present) and not all(present):
        raise ValueError("Depth requires all depth targets, calibration and depth_camera_ids together")
    return all(present)


def validate_depth_batch(batch):
    """Validate optional depth on CPU or CUDA; legacy batches remain unchanged.

    Depth is camera-Z in metres, with 0 invalid. Calibration remains in pixel /
    native world-metre coordinates, even when other batch coordinates normalize.
    Camera selection/order and resolution must agree throughout one batch.
    """
    targets, meta = batch["targets"], batch["meta"]
    if not _has_depth(targets, meta):
        return
    horizon = batch_horizon(batch)
    depth = targets["target_depth"]
    if (not isinstance(depth, torch.Tensor) or depth.ndim != 5 or depth.dtype != torch.float32
            or depth.shape[0] < 1 or depth.shape[1] != horizon or min(depth.shape[2:]) < 1):
        raise ValueError("target_depth must be nonempty float32 [B,H,V,height,width]")
    b, _, views, _, _ = depth.shape
    xyz = batch["inputs"]["gs_xyz"]
    if depth.device != xyz.device or len(meta["scene"]) != b or xyz.shape[0] != b:
        raise ValueError("Depth must match the GS batch size and device")
    specs = {"target_depth_valid": (depth.shape, torch.bool),
             "camera_intrinsics": ((b, views, 3, 3), torch.float32),
             "camera_to_world": ((b, views, 4, 4), torch.float32)}
    for key, (shape, dtype) in specs.items():
        value = targets[key]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype != dtype or value.device != depth.device):
            raise ValueError(f"{key} must have shape {tuple(shape)}, dtype {dtype} and device {depth.device}")
    ids = meta["depth_camera_ids"]
    if (not isinstance(ids, list) or len(ids) != b
            or any(not isinstance(row, list) or len(row) != views
                   or any(not isinstance(camera, str) or not camera for camera in row)
                   or len(set(row)) != views for row in ids)
            or any(row != ids[0] for row in ids)):
        raise ValueError("depth_camera_ids must list the same unique ordered camera names per sample")
    valid = targets["target_depth_valid"]
    if (not torch.isfinite(depth).all() or torch.any(depth < 0)
            or not torch.equal(valid, depth > 0)):
        raise ValueError("Depth must be finite nonnegative metres; positive values exactly match valid pixels")
    intrinsics = targets["camera_intrinsics"].to(dtype=torch.float64)
    transforms = targets["camera_to_world"].to(dtype=torch.float64)
    if not torch.isfinite(intrinsics).all() or not torch.isfinite(transforms).all():
        raise ValueError("Nonfinite depth calibration")
    row3 = intrinsics.new_tensor([0, 0, 1]).expand(b, views, 3)
    row4 = transforms.new_tensor([0, 0, 0, 1]).expand(b, views, 4)
    rotation = transforms[..., :3, :3]
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device).expand(b, views, 3, 3)
    if (torch.any(intrinsics[..., 0, 0] <= 0) or torch.any(intrinsics[..., 1, 1] <= 0)
            or not torch.allclose(intrinsics[..., 2, :], row3, atol=1e-6, rtol=0)
            or torch.any(torch.linalg.det(intrinsics) == 0)
            or not torch.allclose(transforms[..., 3, :], row4, atol=1e-6, rtol=0)
            or not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-5, rtol=0)
            or not torch.allclose(torch.linalg.det(rotation), rotation.new_ones((b, views)),
                                  atol=1e-5, rtol=0)):
        raise ValueError("Depth requires invertible intrinsics and proper rigid camera_to_world matrices")


def _tensor(value, shape, dtype, name):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            or value.dtype != dtype or value.device.type != "cpu"):
        raise ValueError(f"{name} must be a CPU {dtype} tensor of shape {shape}")
    return value


def _points(value, width, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or len(value) < 1:
        raise ValueError(f"{name} must contain at least one real point")
    return _tensor(value, (len(value), width), torch.float32, name)


def _pad_points(values):
    lengths = torch.tensor([len(value) for value in values], dtype=torch.int64)
    maximum = int(lengths.max())
    padded = values[0].new_zeros((len(values), maximum, values[0].shape[-1]))
    for batch_index, value in enumerate(values):
        padded[batch_index, :len(value)] = value
    mask = torch.arange(maximum)[None, :] < lengths[:, None]
    return padded, mask, lengths


def collate_fn(samples: list[dict]) -> dict:
    """True means real/valid. Model-specific causal masks are not built here."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    frames = samples[0]["meta"]["frame_ids"]
    if not isinstance(frames, torch.Tensor) or frames.ndim != 1:
        raise ValueError("Each frame_ids must be int64 [H+1]")
    horizon = validate_horizon(len(frames)-1)
    scenes = [sample["meta"]["scene"] for sample in samples]
    if any(not isinstance(scene, str) for scene in scenes) or len(set(scenes)) != 1:
        raise ValueError("V0 batches must belong to one scene")
    attr_presence = ["target_attr" in sample["targets"] for sample in samples]
    if any(attr_presence) and not all(attr_presence):
        raise ValueError("Cannot mix samples with and without target_attr in one batch")
    initial_xyz, initial_attr = [], []
    for sample in samples:
        inputs = sample["inputs"]
        xyz = _points(inputs["gs_xyz"], 3, "gs_xyz")
        attr = _tensor(inputs["gs_attr"], (len(xyz), 11), torch.float32, "gs_attr")
        initial_xyz.append(xyz)
        initial_attr.append(attr)
        if (not isinstance(sample["targets"]["target_xyz"], list)
                or len(sample["targets"]["target_xyz"]) != horizon):
            raise ValueError(f"target_xyz must contain exactly {horizon} future frames; cannot mix horizons")
        if all(attr_presence) and (not isinstance(sample["targets"]["target_attr"], list)
                                   or len(sample["targets"]["target_attr"]) != horizon):
            raise ValueError(f"target_attr must contain exactly {horizon} future frames")
        if type(sample["meta"]["episode_id"]) is not int:
            raise ValueError("episode_id must be an integer")
    xyz, mask, lengths = _pad_points(initial_xyz)
    attr, _, _ = _pad_points(initial_attr)
    inputs = {"gs_xyz": xyz, "gs_attr": attr, "gs_mask": mask, "gs_lengths": lengths}
    for key, (shape, dtype) in fixed_fields(horizon).items():
        inputs[key] = torch.stack([_tensor(s["inputs"][key], shape, dtype, key) for s in samples])
    target_xyz, target_mask, target_lengths, target_attr = [], [], [], []
    for step in range(horizon):
        points = [_points(s["targets"]["target_xyz"][step], 3, f"target_xyz[{step}]") for s in samples]
        xyz, mask, lengths = _pad_points(points)
        target_xyz.append(xyz)
        target_mask.append(mask)
        target_lengths.append(lengths)
        if all(attr_presence):
            attributes = [_tensor(s["targets"]["target_attr"][step], (len(p), 11), torch.float32,
                                  f"target_attr[{step}]") for s, p in zip(samples, points)]
            target_attr.append(_pad_points(attributes)[0])
    meta = {"scene": scenes, "episode_id": torch.tensor(
        [s["meta"]["episode_id"] for s in samples], dtype=torch.int64)}
    for key in ("frame_ids", "observation_timestamp_us"):
        meta[key] = torch.stack([_tensor(s["meta"][key], (horizon+1,), torch.int64, key) for s in samples])
    targets = {"target_xyz": target_xyz, "target_mask": target_mask,
               "target_lengths": torch.stack(target_lengths, dim=1)}
    if all(attr_presence):
        targets["target_attr"] = target_attr
    depth_presence = [_has_depth(sample["targets"], sample["meta"]) for sample in samples]
    if any(depth_presence):
        if not all(depth_presence):
            raise ValueError("Cannot mix samples with and without depth in one batch")
        first = samples[0]["targets"]["target_depth"]
        if (not isinstance(first, torch.Tensor) or first.ndim != 4
                or first.shape[0] != horizon or min(first.shape[1:]) < 1):
            raise ValueError("Each target_depth must have nonempty shape [H,V,height,width]")
        views = first.shape[1]
        specs = {"target_depth": (first.shape, torch.float32),
                 "target_depth_valid": (first.shape, torch.bool),
                 "camera_intrinsics": ((views, 3, 3), torch.float32),
                 "camera_to_world": ((views, 4, 4), torch.float32)}
        for key, (shape, dtype) in specs.items():
            targets[key] = torch.stack([_tensor(s["targets"][key], shape, dtype, key) for s in samples])
        meta["depth_camera_ids"] = [s["meta"]["depth_camera_ids"] for s in samples]
    batch = {"inputs": inputs, "targets": targets, "meta": meta}
    validate_target_attr_batch(batch)
    validate_depth_batch(batch)
    if any(depth_presence):
        # Keep independently owned metadata just like the stacked tensor arrays.
        meta["depth_camera_ids"] = [list(row) for row in meta["depth_camera_ids"]]
    return batch
