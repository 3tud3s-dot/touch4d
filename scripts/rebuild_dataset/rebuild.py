#!/usr/bin/env python3
"""Reconstruct an audited Deform360 scene according to DATASET_PROTOCOL.md.

Dependencies: Python >= 3.10, numpy, plyfile. Run reconstruction on the server.
Example (not executed automatically):
  python scripts/rebuild_dataset/rebuild.py --scene 001-rope
  python scripts/rebuild_dataset/rebuild.py --scene 008-pink-cloth

Defaults resolve relative to this project's root, not the current directory:
  --source-root datasets/deform360 (contains raw/ and processed/)
  --output-root datasets/touch4d-v1

Original files are read-only. No tar extraction, download, overwrite or cleanup.
An interrupted/failed episode stays at .episode_N.incomplete for inspection;
only a validated episode is renamed to episode_N. No training features added.
Published robot.npy uses a pickled dictionary; source-root must be trusted.
All output NPZ files contain ordinary numeric/bool arrays, never pickle objects.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
VERSION = "touch4d-v1"
SCRIPT_VERSION = "1.1.0"
# Reviewed physical identities, not an inference from episode number or R.
# New scenes/episodes require a reviewed entry before reconstruction.
SCENE_BINDINGS = {
    "001-rope": {
        0: ("right",), 1: ("right",), 2: ("right",), 3: ("right",), 4: ("right",),
        5: ("right", "left"), 6: ("right", "left"), 7: ("right", "left"),
        8: ("right", "left"), 9: ("right", "left"),
    },
    "008-pink-cloth": {
        0: ("right",), 1: ("right",), 2: ("right",), 3: ("right",), 4: ("right",),
        5: ("right", "left"), 6: ("right", "left"), 7: ("right", "left"),
        8: ("right", "left"), 9: ("right", "left"),
    },
}
PINK_REVIEW_FRAMES = {
    0: [113, 163, 213], 1: [113, 238, 364], 2: [103, 185, 268],
    3: [97, 264, 432], 4: [101, 243, 385], 5: [88, 212, 336],
    6: [133, 241, 350], 7: [93, 210, 327], 8: [107, 236, 366],
    9: [130, 207, 284],
}
STREAMS = ("tactilel_left", "tactilel_right", "tactiler_left", "tactiler_right")
HAND_ORDER = ("left", "right")
MAX_AGE_US = 75000  # Provisional, as specified in the protocol.
RIGID_TOL = 1e-6
OPENING_LIMITS = (0.04, 0.112)
GRID_ROWS = (0, 6, 11)
GRID_COLS = (0, 8, 16, 23, 31)
POINT_IDS = [f"{side}_r{r:02d}_c{c:02d}"
             for side in HAND_ORDER for r in GRID_ROWS for c in GRID_COLS]
GS_FIELDS = {
    "xyz": ("x", "y", "z"),
    "log_scales": ("scale_0", "scale_1", "scale_2"),
    "quaternions": ("rot_0", "rot_1", "rot_2", "rot_3"),
    "opacity_logits": ("opacity",),
    "sh_dc": tuple(f"f_dc_{i}" for i in range(3)),
    "sh_rest": tuple(f"f_rest_{i}" for i in range(45)),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_episodes(value):
    try:
        items = []
        for part in value.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                require(start <= end, "Reversed episode range")
                items.extend(range(start, end + 1))
            else:
                items.append(int(part))
        require(items and all(i >= 0 for i in items), "Episode IDs must be nonnegative")
        require(len(set(items)) == len(items), "Duplicate episode selection")
        return sorted(items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def hand_slots(scene, episode):
    require(scene in SCENE_BINDINGS, f"Scene has no reviewed hand bindings: {scene}")
    require(episode in SCENE_BINDINGS[scene], f"Unreviewed hand binding: {scene}/episode_{episode}")
    return list(SCENE_BINDINGS[scene][episode])


def binding_evidence(scene, episode):
    hand_slots(scene, episode)
    return {"camera": "brics-odroid-001_cam0",
            "method": "episode-specific RGB and projected robot geometry review; no inferred identity",
            "review_frames": PINK_REVIEW_FRAMES[episode] if scene == "008-pink-cloth" else [],
            "review_record": "008 representative-frame review" if scene == "008-pink-cloth"
                             else "001-rope previously confirmed episode bindings"}


def discover_episodes(directory):
    directory = Path(directory)
    require(directory.is_dir(), f"Scene directory missing: {directory}")
    found = {}
    for path in directory.iterdir():
        match = re.fullmatch(r"episode_(\d+)", path.name)
        if match and path.is_dir():
            episode = int(match[1])
            require(episode not in found, f"Duplicate padded/unpadded episode ID: {episode}")
            found[episode] = path
    require(found, f"No episode directories in {directory}")
    return dict(sorted(found.items()))


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    require(name and not p.is_absolute() and ".." not in p.parts
            and "\\" not in name and ":" not in name, f"Unsafe relative path: {name}")
    return p.as_posix()


class SourceReader:
    """Read loose files or exact regular members; never extract an archive."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.stack = ExitStack()
        self.archives = {}
        self.refs = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def archive(self, relative):
        relative = safe_relative(relative)
        if relative not in self.archives:
            path = self.root / relative
            stat = path.stat()
            tar = self.stack.enter_context(tarfile.open(path, "r:*"))
            members = {}
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = safe_relative(member.name)
                require(name not in members, f"Duplicate archive member: {relative}!{name}")
                members[name] = member
            self.archives[relative] = (tar, members, (stat.st_size, stat.st_mtime_ns))
        return self.archives[relative]

    def read(self, key, relative, member=None):
        relative = safe_relative(relative)
        path = self.root / relative
        before = path.stat()
        if member is None:
            payload = path.read_bytes()
        else:
            member = safe_relative(member)
            tar, members, signature = self.archive(relative)
            require(signature == (before.st_size, before.st_mtime_ns), f"Source changed: {path}")
            require(member in members, f"Missing member: {relative}!{member}")
            with tar.extractfile(members[member]) as stream:
                payload = stream.read()
            require(len(payload) == members[member].size, f"Truncated member: {member}")
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                f"Source changed while reading: {path}")
        ref = {"path": relative, "member": member, "bytes": len(payload),
               "sha256": hashlib.sha256(payload).hexdigest()}
        if key in self.refs:
            require(self.refs[key] == ref, f"Source changed: {key}")
        self.refs[key] = ref
        return payload


def timestamp_array(payload, *, strict, allow_empty=False):
    stamps = []
    for line in payload.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"frame_(\d+)_(\d+)(?:\s+512)?", line.strip())
        require(match is not None, f"Malformed timestamp row: {line[:100]}")
        timestamp, index = map(int, match.groups())
        require(index == len(stamps), "Timestamp source frame indices must start at 0 and be contiguous")
        require(0 <= timestamp <= np.iinfo(np.int64).max, "Timestamp outside int64 range")
        stamps.append(timestamp)
    require(allow_empty or stamps, "Empty observation timeline")
    ts = np.asarray(stamps, dtype=np.int64)
    require(np.all(np.diff(ts) > 0 if strict else np.diff(ts) >= 0), "Non-monotonic timestamps")
    return ts


def load_raw(payload):
    require(not payload.startswith(b"\x93NUMPY"), "Expected headerless raw float32, not a NumPy container")
    require(len(payload) % (16 * 32 * 4) == 0, "Raw tactile byte count is not a multiple of 2048")
    return np.frombuffer(payload, dtype="<f4").reshape(-1, 16, 32)


def load_median(payload):
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    require(isinstance(value, np.ndarray) and value.shape == (16, 32)
            and value.dtype.kind == "f", "Median must be a floating [16,32] array")
    return value


def load_robot(payload, frames, slots):
    # The trusted published format is a NumPy scalar containing a dictionary.
    value = np.load(io.BytesIO(payload), allow_pickle=True)
    require(isinstance(value, np.ndarray) and value.shape == (), "Unsupported robot.npy container")
    data = value.item()
    require(isinstance(data, dict) and {"T_worlds", "openings"} <= data.keys(), "Missing robot fields")
    t, o = np.asarray(data["T_worlds"]), np.asarray(data["openings"])
    original_shapes = {"T_worlds": list(t.shape), "openings": list(o.shape)}
    if t.ndim == 3:
        require(len(slots) == 1 and t.shape == (frames, 4, 4)
                and o.shape == (frames,), "Malformed single-hand arrays")
        t, o = t[:, None], o[:, None]
    require(t.shape == (frames, len(slots), 4, 4) and o.shape == (frames, len(slots)),
            "Robot shape/frame count does not match observations and audited hand binding")
    require(t.dtype.kind == "f" and o.dtype.kind == "f"
            and t.dtype.itemsize in (4, 8) and o.dtype.itemsize in (4, 8), "Unsupported robot dtype")
    if "bimanual" in data:
        require(bool(data["bimanual"]) == (len(slots) == 2), "bimanual disagrees with audited slots")
    return t, o, original_shapes


def select_raw(raw_ts, obs):
    index = (np.searchsorted(raw_ts, obs, side="right") - 1).astype(np.int64)
    age = np.full(len(obs), -1, dtype=np.int64)
    has = index >= 0
    age[has] = obs[has] - raw_ts[index[has]]
    valid = has & (age >= 0) & (age <= MAX_AGE_US)
    return index, age, valid


def tactile_arrays(raw, median, raw_ts, obs):
    require(len(raw) == len(raw_ts), "Raw tactile data/timestamp count mismatch")
    index, age, time_valid = select_raw(raw_ts, obs)
    values = np.zeros((len(obs), 12, 32), dtype=np.float32)
    valid = np.zeros(values.shape, dtype=bool)
    if len(raw):
        selected = raw[np.maximum(index, 0), :12].astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            delta = selected - median[:12].astype(np.float64)
            positive = np.maximum(delta, 0).astype(np.float32)
        valid = (time_valid[:, None, None] & np.isfinite(selected)
                 & np.isfinite(median[:12])[None] & np.isfinite(delta) & np.isfinite(positive))
        valid[:, :, 31] = False
        values = np.where(valid, positive, 0).astype(np.float32)
    return values, valid, index, age


def pose_validity(t, openings):
    finite = np.isfinite(t).all(axis=(-1, -2)) & np.isfinite(openings)
    safe = np.where(finite[..., None, None], t.astype(np.float64), np.eye(4))
    r = safe[..., :3, :3]
    with np.errstate(over="ignore", invalid="ignore"):
        orth = np.max(np.abs(np.swapaxes(r, -1, -2) @ r - np.eye(3)), axis=(-1, -2))
        det = np.abs(np.linalg.det(r) - 1)
        bottom = np.max(np.abs(safe[..., 3, :] - [0, 0, 0, 1]), axis=-1)
    return finite & (orth <= RIGID_TOL) & (det <= RIGID_TOL) & (bottom <= RIGID_TOL)


def controller_arrays(t, openings, slots):
    frames = len(t)
    points = np.zeros((frames, 2, 30, 3), dtype=np.float32)
    mask = np.zeros((frames, 2, 30), dtype=bool)
    clipped = np.zeros((frames, 2), dtype=bool)
    good = pose_validity(t, openings)
    for slot, hand in enumerate(slots):
        target = HAND_ORDER.index(hand)
        opening = openings[:, slot].astype(np.float64)
        clipped[:, target] = np.isfinite(opening) & ((opening < .04) | (opening > .112))
        q = .038 - .033 * (np.clip(np.where(np.isfinite(opening), opening, .04), .04, .112) - .04) / .072
        pose = np.where(good[:, slot, None, None], t[:, slot].astype(np.float64), np.eye(4))
        point = 0
        for sign, base in [(1, [-.04246242, .0835, .0097]), (-1, [.04246242, .0835, .0107])]:
            for row in GRID_ROWS:
                for col in GRID_COLS:
                    finger = np.array([sign * .007, -.056 - .002 * col, (.025 / 12) * (11.5 - row)])
                    root = np.broadcast_to(np.asarray(base) + finger * [1, -1, -1], (frames, 3)).copy()
                    root[:, 0] += sign * q
                    with np.errstate(over="ignore", invalid="ignore"):
                        world = ((pose[:, :3, :3] @ root[:, :, None])[:, :, 0] + pose[:, :3, 3]).astype(np.float32)
                    point_good = good[:, slot] & np.isfinite(world).all(axis=1)
                    points[:, target, point] = np.where(point_good[:, None], world, 0)
                    mask[:, target, point] = point_good
                    point += 1
    return points, mask, clipped


def gs_arrays(payload, frame, timestamp):
    from plyfile import PlyData

    ply = PlyData.read(io.BytesIO(payload))
    require("vertex" in ply, "PLY has no vertex element")
    vertex = ply["vertex"].data
    require(len(vertex) > 0, "Empty GS frame")
    arrays = {}
    for name, fields in GS_FIELDS.items():
        require(set(fields) <= set(vertex.dtype.names or ()), f"Missing PLY fields: {name}")
        for field in fields:
            require(vertex[field].dtype.kind == "f" and vertex[field].dtype.itemsize == 4,
                    f"Expected float32 PLY field: {field}")
        arrays[name] = np.column_stack([vertex[field] for field in fields]).astype(np.float32)
        require(np.isfinite(arrays[name]).all(), f"Non-finite GS field: {name}, frame {frame}")
    q = arrays["quaternions"].astype(np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    require(np.all(np.isfinite(norm) & (norm > 0)), f"Invalid quaternion norm, frame {frame}")
    arrays["quaternions"] = (q / norm).astype(np.float32)
    arrays["frame_id"] = np.asarray(frame, dtype=np.int64)
    arrays["timestamp_us"] = np.asarray(timestamp, dtype=np.int64)
    return arrays


def episode_layout(reader, scene, episode):
    directories = discover_episodes(reader.root / "processed" / scene)
    require(episode in directories, f"Missing source episode: {scene}/{episode}")
    prefix = directories[episode].relative_to(reader.root).as_posix()
    meta = json.loads(reader.read("episode_metadata", f"{prefix}/metadata.json").decode("utf-8"))
    cameras = meta.get("cameras")
    require(isinstance(cameras, list) and cameras and all(
        isinstance(c, str) and re.fullmatch(r"[A-Za-z0-9_-]+", c) for c in cameras),
        "Invalid camera list in episode metadata")
    require(len(cameras) == len(set(cameras)), "Duplicate camera names")
    camera_refs = {}
    for camera in sorted(cameras):
        name = f"{camera}/aligned_timestamps.txt"
        if (reader.root / prefix / name).is_file():
            camera_refs[camera] = (f"{prefix}/{name}", None)
        else:
            archive = f"{prefix}/cameras.tar"
            _, members, _ = reader.archive(archive)
            require(name in members, f"Missing {archive}!{name}")
            camera_refs[camera] = (archive, name)
    loose = reader.root / prefix / "splatfacto"
    if loose.is_dir():
        gs_refs = [(p.name, (f"{prefix}/splatfacto/{p.name}", None)) for p in loose.glob("*.ply")]
    else:
        archive = f"{prefix}/splatfacto.tar"
        _, members, _ = reader.archive(archive)
        gs_refs = [(PurePosixPath(name).name, (archive, name)) for name in members
                   if name.startswith("splatfacto/") and name.endswith(".ply")]
    by_frame = {}
    for name, ref in gs_refs:
        match = re.fullmatch(r"splat_(\d+)\.ply", name)
        require(match is not None, f"Unexpected GS filename: {name}")
        frame = int(match[1])
        require(frame not in by_frame, f"Duplicate GS frame {frame}")
        by_frame[frame] = ref
    require(by_frame, f"No GS frames for {scene}/episode_{episode}")
    return prefix, camera_refs, by_frame


def observation_timeline(reader, camera_refs):
    # All published cameras must agree; never use metadata's research subwindow.
    camera = sorted(camera_refs)[0]
    obs = None
    for name, ref in sorted(camera_refs.items()):
        ts = timestamp_array(reader.read(f"camera_timestamps_{name}", *ref), strict=True)
        if obs is None:
            obs = ts
        else:
            require(np.array_equal(obs, ts), f"Aligned camera timelines disagree: {name}")
    primary = timestamp_array(reader.read("observation", *camera_refs[camera]), strict=True)
    require(np.array_equal(primary, obs), "Observation source changed while reading")
    return camera, obs


def find_raw_sources(source_root, scene, obs):
    """Pair by exact stem and unique time-interval overlap, never sorted ordinal.

    A partial download may omit unrelated episodes without renumbering anything.
    Missing/ambiguous selected recordings and multiple median candidates fail.
    """
    selected = []
    for sensor_name in STREAMS:
        sensor = "brics-odroid_" + sensor_name
        directory = Path(source_root) / "raw" / scene / sensor
        require(directory.is_dir(), f"Missing tactile sensor directory: {directory}")
        medians = sorted(directory.glob("median_*.npy"))
        require(len(medians) == 1, f"Expected one unambiguous baseline in {directory}, found {len(medians)}")
        require(medians[0].is_file() and re.fullmatch(r"median_\d+\.npy", medians[0].name),
                f"Unexpected baseline filename: {medians[0]}")
        data = {p.stem: p for p in directory.glob("*.npy") if not p.name.startswith("median_")}
        times = {p.stem: p for p in directory.glob("*.txt")}
        require(data and data.keys() == times.keys(), f"Unpaired raw .npy/.txt files in {directory}")
        candidates = []
        for stem in sorted(data):
            require(re.fullmatch(re.escape(sensor) + r"_\d+", stem) is not None,
                    f"Unexpected raw recording name: {stem}")
            ts = timestamp_array(times[stem].read_bytes(), strict=False)
            if ts[0] <= obs[-1] and ts[-1] >= obs[0]:
                require(data[stem].stat().st_size == len(ts) * 16 * 32 * 4,
                        f"Raw byte/timestamp count mismatch: {data[stem]}")
                candidates.append((data[stem], times[stem], medians[0]))
        require(len(candidates) == 1,
                f"{scene}/{sensor}: expected one raw recording overlapping observation "
                f"[{obs[0]}, {obs[-1]}], found {len(candidates)}; cannot guess pairing")
        selected.append(tuple(p.relative_to(source_root).as_posix() for p in candidates[0]))
    return selected


def output_paths(source_root, output_root, scene, episode):
    target = output_root / scene / f"episode_{episode}"
    partial = target.with_name(f".episode_{episode}.incomplete")
    require(not (output_root / scene).is_symlink(), "Output scene directory must not be a symlink")
    for path in (target, partial):
        resolved = path.resolve()
        require(resolved.is_relative_to(output_root.resolve()), f"Output escapes destination: {path}")
        for source in (source_root / "raw", source_root / "processed"):
            require(not resolved.is_relative_to(source.resolve()), f"Output is inside source data: {path}")
        require(not path.exists() and not path.is_symlink(), f"Refusing existing/partial output: {path}")
    return target, partial


def preflight(source_root, output_root, scene, episodes):
    require(source_root.is_dir(), f"Source root missing: {source_root}")
    for source in (source_root / "raw", source_root / "processed"):
        require(source.is_dir(), f"Source directory missing: {source}")
    plans = {}
    seen_recordings = set()
    for ep in episodes:
        slots = hand_slots(scene, ep)
        output_paths(source_root, output_root, scene, ep)
        with SourceReader(source_root) as reader:
            prefix, camera_refs, gs_refs = episode_layout(reader, scene, ep)
            camera, obs = observation_timeline(reader, camera_refs)
            require(sorted(gs_refs) == list(range(len(obs))), f"GS gaps/extra frames in episode {ep}")
            load_robot(reader.read("robot", f"{prefix}/robot/robot.npy"), len(obs), slots)
            tactile_sources = find_raw_sources(source_root, scene, obs)
            for channel, paths in enumerate(tactile_sources):
                raw_path, time_path, median_path = paths
                require(raw_path not in seen_recordings, f"One raw recording matched multiple episodes: {raw_path}")
                seen_recordings.add(raw_path)
                raw = load_raw(reader.read(f"tactile_{channel}_raw", raw_path))
                ts = timestamp_array(reader.read(f"tactile_{channel}_timestamps", time_path), strict=False)
                require(len(raw) == len(ts), f"Raw/time count mismatch: {raw_path}")
                load_median(reader.read(f"tactile_{channel}_median", median_path))
            plans[ep] = {"prefix": prefix, "camera": camera, "camera_refs": camera_refs,
                         "gs_refs": gs_refs, "obs": obs, "tactile_sources": tactile_sources,
                         "source_refs": dict(reader.refs)}
    return plans


def write_npz(path, arrays):
    require(all(isinstance(v, np.ndarray) and v.dtype.kind in "biuf" for v in arrays.values()),
            "Output NPZ must contain numeric/bool arrays only")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def parameters():
    return {"max_age_us": MAX_AGE_US, "max_age_status": "provisional", "rigid_atol": RIGID_TOL,
            "rigid_rtol": 0, "opening_clip_m": list(OPENING_LIMITS), "surface_offset_m": .007,
            "crop_rows": [0, 12], "invalid_columns": [31], "row_flip": False,
            "tactile_formula": "maximum(raw - median, 0)", "tactile_subtraction_dtype": "float64",
            "tactile_output_dtype": "float32", "contact_threshold": None,
            "duplicate_timestamps": "last raw record wins", "sampling": "latest past, no interpolation",
            "gs_coordinate_transform": "identity; native world metres", "quaternion_order": "WXYZ",
            "sh_rest_order": "f_rest_0..44 unchanged; channel/basis layout pending",
            "gs_correspondence": "no fixed cross-frame identity"}


def rebuild_episode(source_root, output_root, scene, episode, plan):
    target, partial = output_paths(source_root, output_root, scene, episode)
    partial.mkdir(parents=True, exist_ok=False)
    (partial / "gs").mkdir()
    try:
        with SourceReader(source_root) as reader:
            prefix, camera_refs, gs_refs = episode_layout(reader, scene, episode)
            camera, obs = observation_timeline(reader, camera_refs)
            require(prefix == plan["prefix"] and camera_refs == plan["camera_refs"]
                    and gs_refs == plan["gs_refs"] and np.array_equal(obs, plan["obs"]),
                    "Episode layout/timeline changed after preflight")
            tactile_sources = find_raw_sources(source_root, scene, obs)
            require(tactile_sources == plan["tactile_sources"], "Raw discovery changed after preflight")
            frames = len(obs)
            require(sorted(gs_refs) == list(range(frames)), "GS must cover the full observation timeline")
            source_meta = json.loads(reader.read("episode_metadata", f"{prefix}/metadata.json").decode("utf-8"))
            require(camera in source_meta.get("cameras", []), "Observation camera absent in source metadata")
            slots = hand_slots(scene, episode)
            t, o, original_shapes = load_robot(reader.read("robot", f"{prefix}/robot/robot.npy"), frames, slots)
            points, controller_mask, clipped = controller_arrays(t, o, slots)
            write_npz(partial / "controller.npz", {"controller_points": points,
                "controller_valid_mask": controller_mask, "T_worlds": t, "openings": o,
                "opening_clipped": clipped})
            tactile = np.zeros((frames, 4, 12, 32), dtype=np.float32)
            valid = np.zeros(tactile.shape, dtype=bool)
            index = np.full((frames, 4), -1, dtype=np.int64)
            age = np.full_like(index, -1)
            timestamps = {"observation_timestamp_us": obs}
            for channel in range(4):
                raw_path, time_path, median_path = tactile_sources[channel]
                raw = load_raw(reader.read(f"tactile_{channel}_raw", raw_path))
                raw_ts = timestamp_array(reader.read(f"tactile_{channel}_timestamps", time_path), strict=False, allow_empty=True)
                median = load_median(reader.read(f"tactile_{channel}_median", median_path))
                values, accepted, ix, ages = tactile_arrays(raw, median, raw_ts, obs)
                tactile[:, channel], valid[:, channel] = values, accepted
                index[:, channel], age[:, channel] = ix, ages
                timestamps[f"raw_tactile_timestamp_us_{channel}"] = raw_ts
            timestamps.update(tactile_raw_index=index, tactile_age_us=age)
            write_npz(partial / "tactile.npz", {"tactile": tactile, "valid_mask": valid})
            write_npz(partial / "timestamps.npz", timestamps)
            counts = []
            for frame in range(frames):
                gs = gs_arrays(reader.read(f"gs_{frame}", *gs_refs[frame]), frame, obs[frame])
                counts.append(len(gs["xyz"]))
                write_npz(partial / "gs" / f"frame_{frame:06d}.npz", gs)
                if frame % 100 == 0:
                    print(f"episode {episode}: GS {frame + 1}/{frames}", flush=True)
            for key, ref in plan["source_refs"].items():
                require(reader.refs.get(key) == ref, f"Source changed after preflight: {key}")
            output_files = sorted(partial.rglob("*.npz"))
            protocol = PROJECT / "DATASET_PROTOCOL.md"
            metadata = {"schema_version": VERSION, "script_version": SCRIPT_VERSION,
                "scene": scene, "episode": episode, "frames": frames,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "observation_camera": camera, "observation_cameras": sorted(camera_refs),
                "source_episode_directory": prefix, "source_root": str(source_root),
                "stream_order": list(STREAMS), "hand_order": list(HAND_ORDER),
                "robot_slot_hands": slots,
                "physical_hand_to_robot_slot": {h: slots.index(h) if h in slots else None for h in HAND_ORDER},
                "binding_basis": binding_evidence(scene, episode),
                "controller_point_ids": POINT_IDS, "source_robot_shapes": original_shapes,
                "source_robot_dtypes": {"T_worlds": str(t.dtype), "openings": str(o.dtype)},
                "parameters": parameters(), "source_files": reader.refs,
                "gs_point_counts": counts,
                "protocol_sha256": hash_file(protocol) if protocol.is_file() else None,
                "scripts_sha256": {name: hash_file(Path(__file__).with_name(name))
                                   for name in ("rebuild.py", "validate.py")},
                "output_files": {p.relative_to(partial).as_posix(): hash_file(p) for p in output_files},
                "summary": {"tactile_valid_elements": int(valid.sum()),
                            "controller_valid_points": int(controller_mask.sum()),
                            "opening_clipped_hand_frames": int(clipped.sum())}}
            with (partial / "metadata.json").open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        # Numeric and metadata checks before publication; source bytes were just read.
        from validate import validate_episode
        validate_episode(partial, expected_scene=scene, source_root=source_root, verify_sources=False)
        require(not target.exists() and not target.is_symlink(), f"Output appeared during rebuild: {target}")
        partial.rename(target)
        print(f"Completed: {target}", flush=True)
    except Exception:
        print(f"Failed episode {episode}; partial output retained at {partial}", file=sys.stderr)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, choices=sorted(SCENE_BINDINGS), help="Full audited scene name")
    parser.add_argument("--source-root", type=Path, default=PROJECT / "datasets/deform360")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "datasets/touch4d-v1")
    parser.add_argument("--episodes", type=parse_episodes, metavar="0-9 or 0,5",
                        help="Optional subset; default: every discovered source episode")
    args = parser.parse_args(argv)
    require(os.name != "nt", "Reconstruction outputs belong on the server; do not run on local Windows")
    source_root, output_root = args.source_root.resolve(), args.output_root.resolve()
    # Resolve dependencies before creating any directories or partial outputs.
    from plyfile import PlyData
    from validate import validate_episode
    available = discover_episodes(source_root / "processed" / args.scene)
    episodes = args.episodes if args.episodes is not None else list(available)
    require(set(episodes) <= set(available), "Requested episodes are absent from the source scene")
    plans = preflight(source_root, output_root, args.scene, episodes)
    print(f"Source: {source_root}\nOutput: {output_root / args.scene}\nEpisodes: {episodes}", flush=True)
    for episode in episodes:
        rebuild_episode(source_root, output_root, args.scene, episode, plans[episode])
    return 0


if __name__ == "__main__":
    # Also suppress sibling-module bytecode when the validator imports helpers.
    sys.dont_write_bytecode = True
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, ImportError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
