"""Fixed train-only transforms from DATA_PROCESS_DESIGN.md; no data fitting.

    processor = DataProcessor("001-rope")
    normalized = processor(batch)  # same structure; independent tensor storage
    restored = processor.denormalize_batch(normalized)

Save processor.state_dict() alongside a checkpoint. For prediction [B,5,N,3],
pass gs_mask[:, None, :].expand(B,5,N) to denormalize_xyz/displacement.
"""

from dataclasses import dataclass
import math
from types import MappingProxyType

import torch

from ..dataloader.collate import FIXED_FIELDS, validate_depth_batch
from ..dataloader.splits import HORIZON


@dataclass(frozen=True)
class _Parameters:
    center: tuple[float, float, float]
    scale: float


_PARAMETERS = MappingProxyType({
    "001-rope": _Parameters((0.210834337570, 0.013140156253, 0.157411820790), 0.157037090510),
    "051-cube": _Parameters((0.236388585803, 0.023145916698, 0.103283534840), 0.042746614739),
})
SUPPORTED_SCENES = tuple(_PARAMETERS)
_VERSION = "touch4d-data-process-v1"
_TACTILE_SCALE = 100.0


def _clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: _clone_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(v) for v in value)
    return value  # batch metadata leaves are immutable strings/integers


def _require_tensor(value, shape, dtype, name, device=None):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            or value.dtype != dtype):
        raise ValueError(f"{name} must have shape {tuple(shape)} and dtype {dtype}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")


def _masked(values, mask):
    """Masks match scalar values or vector prefixes exactly; no implicit tiling."""
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.device != values.device:
        raise ValueError("mask must be bool and on the values' device")
    if mask.shape == values.shape:
        active = mask
    elif mask.shape == values.shape[:-1]:
        active = mask.unsqueeze(-1)
    else:
        raise ValueError("mask shape must match values or their vector prefix")
    return torch.where(active, values, torch.zeros((), dtype=values.dtype, device=values.device))


@dataclass(frozen=True)
class DataProcessor:
    """One fixed scene, CPU or CUDA tensors, and no mutation of input batches.

    Apply once to a raw DataLoader batch. The batch schema deliberately does not
    acquire a normalization flag; callers own the raw/normalized distinction.
    Values follow the loader's float32 contract; inverse prediction helpers also
    accept float64. Masks must explicitly match prediction point/time dimensions.

    Optional depth targets stay in metres and camera matrices stay in the native
    world frame. Before rendering a prediction against them, restore its XYZ
    with denormalize_xyz and its log_scales by adding log(state_dict()['scale']).
    Depth validity is independent of the point-padding masks.
    """

    scene: str

    def __post_init__(self):
        if not isinstance(self.scene, str) or self.scene not in _PARAMETERS:
            raise ValueError(f"Unsupported scene {self.scene!r}; choose {SUPPORTED_SCENES}")

    def state_dict(self):
        """JSON-safe constants to retain with an experiment/checkpoint."""
        params = _PARAMETERS[self.scene]
        return {"version": _VERSION, "scene": self.scene, "center": list(params.center),
                "scale": params.scale, "tactile_scale": _TACTILE_SCALE}

    @classmethod
    def from_state_dict(cls, state):
        """Reject incompatible saved constants instead of silently replacing them."""
        if not isinstance(state, dict):
            raise ValueError("Expected a saved data_process parameter dictionary")
        result = cls(state.get("scene"))
        if state != result.state_dict():
            raise ValueError("Saved normalization parameters do not match the current V0 protocol")
        return result

    def _validate_batch(self, batch):
        try:
            inputs, targets, meta = batch["inputs"], batch["targets"], batch["meta"]
            scenes = meta["scene"]
            if not isinstance(scenes, list) or not scenes or any(s != self.scene for s in scenes):
                raise ValueError(f"batch must contain only {self.scene}")
            b = len(scenes)
            xyz = inputs["gs_xyz"]
            if not isinstance(xyz, torch.Tensor) or xyz.ndim != 3 or xyz.shape[1] < 1:
                raise ValueError("gs_xyz must be a nonempty [B,N,3] tensor")
            n, device = xyz.shape[1], xyz.device
            _require_tensor(xyz, (b, n, 3), torch.float32, "gs_xyz")
            _require_tensor(inputs["gs_attr"], (b, n, 11), torch.float32, "gs_attr", device)
            _require_tensor(inputs["gs_mask"], (b, n), torch.bool, "gs_mask", device)
            _require_tensor(inputs["gs_lengths"], (b,), torch.int64, "gs_lengths")
            for key, (shape, dtype) in FIXED_FIELDS.items():
                _require_tensor(inputs[key], (b, *shape), dtype, key,
                                None if key == "tactile_age_us" else device)
            _require_tensor(targets["target_lengths"], (b, HORIZON), torch.int64, "target_lengths")
            for key in ("target_xyz", "target_mask"):
                if not isinstance(targets[key], list) or len(targets[key]) != HORIZON:
                    raise ValueError(f"{key} must be a list of five tensors")
            for k, points in enumerate(targets["target_xyz"]):
                if not isinstance(points, torch.Tensor) or points.ndim != 3 or points.shape[1] < 1:
                    raise ValueError(f"target_xyz[{k}] must be nonempty [B,M,3]")
                _require_tensor(points, (b, points.shape[1], 3), torch.float32, f"target_xyz[{k}]", device)
                _require_tensor(targets["target_mask"][k], (b, points.shape[1]), torch.bool,
                                f"target_mask[{k}]", device)
            _require_tensor(meta["episode_id"], (b,), torch.int64, "episode_id")
            for key in ("frame_ids", "observation_timestamp_us"):
                _require_tensor(meta[key], (b, HORIZON+1), torch.int64, key)
            validate_depth_batch(batch)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed DataLoader batch: {exc}") from exc

    def _xyz(self, values, mask, inverse=False, displacement=False):
        if (not isinstance(values, torch.Tensor) or values.ndim < 2 or values.shape[-1] != 3
                or values.dtype not in (torch.float32, torch.float64)):
            raise ValueError("coordinates must be float32/float64 vectors with final dimension 3")
        if not isinstance(mask, torch.Tensor) or mask.shape != values.shape[:-1]:
            raise ValueError("coordinate mask must match every point/time dimension explicitly")
        params = _PARAMETERS[self.scene]
        center = values.new_tensor(params.center)
        scale = values.new_tensor(params.scale)
        if displacement:
            result = values * scale if inverse else values / scale
        elif inverse:
            result = values * scale + center
        else:
            result = (values - center) / scale
        return _masked(result, mask)

    def denormalize_xyz(self, values, mask):
        """Convert predicted coordinates to world metres, re-zeroing padding."""
        return self._xyz(values, mask, inverse=True)

    def denormalize_displacement(self, values, mask):
        """Convert predicted displacements to metres; do not add the center."""
        return self._xyz(values, mask, inverse=True, displacement=True)

    def _transform_batch(self, batch, inverse):
        self._validate_batch(batch)
        result = _clone_tree(batch)
        inputs, targets = result["inputs"], result["targets"]
        params = _PARAMETERS[self.scene]
        for field, mask in (("gs_xyz", "gs_mask"), ("controller_points", "controller_mask"),
                            ("tactile_xyz", "tactile_spatial_mask")):
            inputs[field] = self._xyz(inputs[field], inputs[mask], inverse=inverse)
        attrs = inputs["gs_attr"]
        log_scale = attrs.new_tensor(math.log(params.scale))
        scales = attrs[..., :3] + log_scale if inverse else attrs[..., :3] - log_scale
        inputs["gs_attr"] = _masked(torch.cat((scales, attrs[..., 3:]), dim=-1), inputs["gs_mask"])
        actions, valid = inputs["action_state"], inputs["action_valid"]
        position = self._xyz(actions[..., :3], valid, inverse=inverse)
        scale = actions.new_tensor(params.scale)
        opening = actions[..., 12:] * scale if inverse else actions[..., 12:] / scale
        inputs["action_state"] = _masked(torch.cat((position, actions[..., 3:12], opening), dim=-1), valid)
        tactile = inputs["tactile"]
        tactile = tactile * _TACTILE_SCALE if inverse else tactile / _TACTILE_SCALE
        inputs["tactile"] = _masked(tactile, inputs["valid_mask"])
        targets["target_xyz"] = [self._xyz(xyz, mask, inverse=inverse)
                                  for xyz, mask in zip(targets["target_xyz"], targets["target_mask"])]
        return result

    def __call__(self, batch):
        return self._transform_batch(batch, inverse=False)

    def denormalize_batch(self, batch):
        return self._transform_batch(batch, inverse=True)
