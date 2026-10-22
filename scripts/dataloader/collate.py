"""Right-pad real GS points per batch, independently at each target time."""

import torch

from .splits import HORIZON

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
    scenes = [sample["meta"]["scene"] for sample in samples]
    if any(not isinstance(scene, str) for scene in scenes) or len(set(scenes)) != 1:
        raise ValueError("V0 batches must belong to one scene")
    initial_xyz, initial_attr = [], []
    for sample in samples:
        inputs = sample["inputs"]
        xyz = _points(inputs["gs_xyz"], 3, "gs_xyz")
        attr = _tensor(inputs["gs_attr"], (len(xyz), 11), torch.float32, "gs_attr")
        initial_xyz.append(xyz)
        initial_attr.append(attr)
        if len(sample["targets"]["target_xyz"]) != HORIZON:
            raise ValueError("target_xyz must contain exactly five future frames")
        if type(sample["meta"]["episode_id"]) is not int:
            raise ValueError("episode_id must be an integer")
    xyz, mask, lengths = _pad_points(initial_xyz)
    attr, _, _ = _pad_points(initial_attr)
    inputs = {"gs_xyz": xyz, "gs_attr": attr, "gs_mask": mask, "gs_lengths": lengths}
    for key, (shape, dtype) in FIXED_FIELDS.items():
        inputs[key] = torch.stack([_tensor(s["inputs"][key], shape, dtype, key) for s in samples])
    target_xyz, target_mask, target_lengths = [], [], []
    for step in range(HORIZON):
        xyz, mask, lengths = _pad_points([
            _points(s["targets"]["target_xyz"][step], 3, f"target_xyz[{step}]") for s in samples])
        target_xyz.append(xyz)
        target_mask.append(mask)
        target_lengths.append(lengths)
    meta = {"scene": scenes, "episode_id": torch.tensor(
        [s["meta"]["episode_id"] for s in samples], dtype=torch.int64)}
    for key in ("frame_ids", "observation_timestamp_us"):
        meta[key] = torch.stack([_tensor(s["meta"][key], (6,), torch.int64, key) for s in samples])
    return {"inputs": inputs, "targets": {"target_xyz": target_xyz, "target_mask": target_mask,
                                          "target_lengths": torch.stack(target_lengths, dim=1)}, "meta": meta}
