"""Lazy GS loading and episode-local six-frame samples from touch4d-v1.

The root is datasets/touch4d-v1, not the raw Deform360 directory. Fixed-size
episode arrays are cached per Dataset/worker; GS frames are read on demand.
No source files, reconstructed files or Markdown are written by this module.
"""

from collections import OrderedDict
from dataclasses import dataclass
import json
import operator
from pathlib import Path
from zipfile import BadZipFile
from zlib import error as ZlibError

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import (OPENING_RANGE, RIGID_ATOL, action_conditions, hand_slots,
                       tactile_world_coordinates)
from .splits import HORIZON, episode_ids, window_starts

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "touch4d-v1"
GS_WIDTHS = {"xyz": 3, "log_scales": 3, "quaternions": 4,
             "opacity_logits": 1, "sh_dc": 3, "sh_rest": 45}


class DataFormatError(ValueError):
    """A required rebuilt file or one of its contracts is invalid."""


def read_npz(path: Path, keys: tuple | list | None = None) -> dict:
    """Close the archive immediately; never allow object arrays or pickle."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in (archive.files if keys is None else keys)}
    except (OSError, ValueError, KeyError, EOFError, BadZipFile, ZlibError) as exc:
        raise DataFormatError(f"{path}: cannot read required NPZ arrays: {exc}") from exc


def require_array(arrays, name, shape, dtype, path, *, finite=False):
    value = arrays.get(name)
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.dtype(dtype):
        actual = None if value is None else (value.shape, str(value.dtype))
        raise DataFormatError(f"{path}: {name} requires {shape}/{np.dtype(dtype)}, got {actual}")
    if finite and not np.isfinite(value).all():
        raise DataFormatError(f"{path}: nonfinite {name}")
    return value


def require_native_float(arrays, name, shape, path):
    value = arrays.get(name)
    if (not isinstance(value, np.ndarray) or value.shape != shape
            or value.dtype not in (np.dtype("float32"), np.dtype("float64"))):
        raise DataFormatError(f"{path}: {name} requires {shape} float32/float64")
    return value


@dataclass
class EpisodeInfo:
    episode_id: int
    path: Path
    frames: int
    slots: tuple
    timestamps: np.ndarray
    ages: np.ndarray
    point_counts: tuple


def _episode_info(path: Path, scene: str, episode: int) -> EpisodeInfo:
    metadata_path = path / "metadata.json"
    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataFormatError(f"{metadata_path}: {exc}") from exc
    if not isinstance(meta, dict) or meta.get("schema_version") != "touch4d-v1":
        raise DataFormatError(f"{metadata_path}: expected touch4d-v1 metadata")
    frames = meta.get("frames")
    if (meta.get("scene") != scene or meta.get("episode") != episode
            or type(frames) is not int or frames < 1):
        raise DataFormatError(f"{metadata_path}: inconsistent scene, episode or frame count")
    try:
        slots = hand_slots(meta)
    except ValueError as exc:
        raise DataFormatError(f"{metadata_path}: {exc}") from exc
    # Geometry and validity assumptions must be explicit; version 1.3/1.4 both qualify.
    expected = {"rigid_atol": RIGID_ATOL, "rigid_rtol": 0,
                "opening_clip_m": list(OPENING_RANGE), "crop_rows": [0, 12],
                "invalid_columns": [31], "row_flip": False, "max_age_us": 75000,
                "gs_coordinate_transform": "identity; native world metres",
                "quaternion_order": "WXYZ"}
    params = meta.get("parameters")
    if not isinstance(params, dict):
        raise DataFormatError(f"{metadata_path}: missing reconstruction parameters")
    for key, value in expected.items():
        if key not in params or params[key] != value:
            raise DataFormatError(f"{metadata_path}: unsupported parameter {key}={params.get(key)!r}")
    counts = meta.get("gs_point_counts")
    if (not isinstance(counts, list) or len(counts) != frames
            or any(type(n) is not int or n < 1 for n in counts)):
        raise DataFormatError(f"{metadata_path}: invalid gs_point_counts")
    for filename in ("tactile.npz", "controller.npz", "timestamps.npz"):
        if not (path / filename).is_file():
            raise DataFormatError(f"{path / filename}: required file is missing")
    for frame in range(frames):
        filename = path / "gs" / f"frame_{frame:06d}.npz"
        if not filename.is_file():
            raise DataFormatError(f"{filename}: required GS frame is missing")
    time_path = path / "timestamps.npz"
    arrays = read_npz(time_path)
    ts = require_array(arrays, "observation_timestamp_us", (frames,), np.int64, time_path)
    ages = require_array(arrays, "tactile_age_us", (frames, 4), np.int64, time_path)
    indices = require_array(arrays, "tactile_raw_index", (frames, 4), np.int64, time_path)
    if np.any(ts[1:] <= ts[:-1]):
        raise DataFormatError(f"{time_path}: observation timestamps must strictly increase")
    for stream in range(4):
        raw = arrays.get(f"raw_tactile_timestamp_us_{stream}")
        if (not isinstance(raw, np.ndarray) or raw.ndim != 1 or raw.dtype != np.int64
                or np.any(raw[1:] < raw[:-1])):
            raise DataFormatError(f"{time_path}: invalid raw timestamp stream {stream}")
        # Verify stored provenance, without changing reconstructed alignment or masks.
        latest = np.searchsorted(raw, ts, side="right") - 1
        expected_age = np.full(frames, -1, dtype=np.int64)
        past = latest >= 0
        expected_age[past] = ts[past] - raw[latest[past]]
        if not np.array_equal(indices[:, stream], latest) or not np.array_equal(ages[:, stream], expected_age):
            raise DataFormatError(f"{time_path}: raw indices/ages disagree with stream {stream} timeline")
    return EpisodeInfo(episode, path, frames, slots, ts, ages, tuple(counts))


def _episode_conditions(info: EpisodeInfo) -> dict:
    path, frames = info.path, info.frames
    tactile_path = path / "tactile.npz"
    arrays = read_npz(tactile_path)
    tactile = require_array(arrays, "tactile", (frames, 4, 12, 32), np.float32, tactile_path, finite=True)
    valid = require_array(arrays, "valid_mask", tactile.shape, bool, tactile_path)
    if np.any(tactile < 0) or np.any(tactile[~valid] != 0) or valid[..., 31].any():
        raise DataFormatError(f"{tactile_path}: invalid tactile values, zero fill or column 31 mask")
    time_valid = (info.ages >= 0) & (info.ages <= 75000)
    if np.any(valid & ~time_valid[..., None, None]):
        raise DataFormatError(f"{tactile_path}: valid pixels violate the saved 0..75 ms time mask")
    controller_path = path / "controller.npz"
    arrays = read_npz(controller_path)
    if "actions" not in arrays:
        raise DataFormatError(f"{controller_path}: legacy reconstruction lacks actions; rebuild/upgrade it first")
    robot_count = sum(slot is not None for slot in info.slots)
    transforms = require_native_float(arrays, "T_worlds", (frames, robot_count, 4, 4), controller_path)
    openings = require_native_float(arrays, "openings", (frames, robot_count), controller_path)
    actions = require_native_float(arrays, "actions", (frames, robot_count, 5, 3), controller_path)
    points = require_array(arrays, "controller_points", (frames, 2, 30, 3), np.float32, controller_path, finite=True)
    mask = require_array(arrays, "controller_valid_mask", (frames, 2, 30), bool, controller_path)
    clipped = require_array(arrays, "opening_clipped", (frames, 2), bool, controller_path)
    try:
        state, action_valid = action_conditions(actions, transforms, openings, info.slots)
    except ValueError as exc:
        raise DataFormatError(f"{controller_path}: {exc}") from exc
    if np.any(points[~mask] != 0) or np.any(mask & ~action_valid[..., None]):
        raise DataFormatError(f"{controller_path}: inconsistent controller validity or zero fill")
    for hand, slot in enumerate(info.slots):
        if slot is None and clipped[:, hand].any():
            raise DataFormatError(f"{controller_path}: missing hand marked as opening_clipped")
    return {"tactile": tactile, "valid_mask": valid, "action_state": state,
            "action_valid": action_valid, "controller_points": points, "controller_mask": mask,
            "T_worlds": transforms, "openings": openings}


def read_gs(info: EpisodeInfo, frame: int) -> dict:
    """Check the entire frame, including preserved SH, before selecting V0 fields."""
    path = info.path / "gs" / f"frame_{frame:06d}.npz"
    arrays = read_npz(path)
    frame_id = require_array(arrays, "frame_id", (), np.int64, path)
    timestamp = require_array(arrays, "timestamp_us", (), np.int64, path)
    if int(frame_id) != frame or int(timestamp) != int(info.timestamps[frame]):
        raise DataFormatError(f"{path}: frame_id/timestamp_us disagrees with the observation timeline")
    n = info.point_counts[frame]
    for field, width in GS_WIDTHS.items():
        require_array(arrays, field, (n, width), np.float32, path, finite=True)
    norm = np.linalg.norm(arrays["quaternions"].astype(np.float64), axis=-1)
    if np.any(norm == 0) or not np.allclose(norm, 1, atol=1e-5, rtol=0):
        raise DataFormatError(f"{path}: expected nonzero normalized WXYZ quaternions")
    return arrays


class Touch4DDataset(Dataset):
    """One scene/split only. Samples contain independent, CPU-owned torch tensors.

    stage='single' or 'all' controls training curriculum. Recreate the loader to
    change stages; validation/test always use 'all'. cache_episodes limits the
    number of fixed-array episodes in each worker (GS frames are not cached).
    """

    def __init__(self, scene: str, split: str, root=DEFAULT_ROOT, *, stage="all", cache_episodes=2):
        selected = episode_ids(scene, split, stage)
        if type(cache_episodes) is not int or cache_episodes < 1:
            raise ValueError("cache_episodes must be a positive integer")
        self.root = Path(root).expanduser().resolve()
        self.scene, self.split, self.stage = scene, split, stage
        self.cache_episodes = cache_episodes
        self.episode_ids = selected
        self.episodes = {}
        self.windows = []
        self._cache = OrderedDict()
        for episode in selected:
            info = _episode_info(self.root / scene / f"episode_{episode}", scene, episode)
            if stage == "single" and sum(s is not None for s in info.slots) != 1:
                raise DataFormatError(f"{info.path}: reviewed single-hand curriculum conflicts with metadata")
            self.episodes[episode] = info
            self.windows.extend((episode, start) for start in window_starts(info.frames, split))
        self.windows = tuple(self.windows)

    def __len__(self):
        return len(self.windows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()  # Do not pickle a warm cache to spawned workers.
        return state

    def __getitem__(self, index):
        index = operator.index(index)
        episode, start = self.windows[index]
        info = self.episodes[episode]
        if episode not in self._cache:
            self._cache[episode] = _episode_conditions(info)
            while len(self._cache) > self.cache_episodes:
                self._cache.popitem(last=False)
        self._cache.move_to_end(episode)
        conditions = self._cache[episode]
        take = slice(start, start + HORIZON)
        inputs = {key: torch.from_numpy(conditions[key][take].copy()) for key in (
            "action_state", "action_valid", "controller_points", "controller_mask", "tactile", "valid_mask")}
        xyz, spatial = tactile_world_coordinates(
            conditions["T_worlds"][take], conditions["openings"][take], info.slots,
            conditions["action_valid"][take], conditions["valid_mask"][take])
        inputs["tactile_xyz"] = torch.from_numpy(xyz)
        inputs["tactile_spatial_mask"] = torch.from_numpy(spatial)
        inputs["tactile_age_us"] = torch.from_numpy(info.ages[take].copy())
        initial = read_gs(info, start)
        inputs["gs_xyz"] = torch.from_numpy(initial["xyz"].copy())
        inputs["gs_attr"] = torch.from_numpy(np.concatenate(
            [initial[key] for key in ("log_scales", "quaternions", "opacity_logits", "sh_dc")], axis=-1))
        targets = [torch.from_numpy(read_gs(info, frame)["xyz"].copy())
                   for frame in range(start + 1, start + HORIZON + 1)]
        return {"inputs": inputs, "targets": {"target_xyz": targets}, "meta": {
            "scene": self.scene, "episode_id": episode,
            "frame_ids": torch.arange(start, start + HORIZON + 1, dtype=torch.int64),
            "observation_timestamp_us": torch.from_numpy(info.timestamps[start:start + HORIZON + 1].copy()),
        }}
