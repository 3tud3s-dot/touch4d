"""Independent Touch4D depth-v1 extension; core V1 files remain read-only.

Build: python -B -m scripts.rebuild_dataset.depth --mode build --scene 008-pink-cloth --episodes 0,1
Check: python -B -m scripts.rebuild_dataset.depth --mode validate --scene 008-pink-cloth --episodes 0,1

Core reconstruction uses the separately retained production v1.3 scripts.
This module contains only the historical depth-v1 helpers from v1.4, without
changing depth units, camera convention, masks, source identity or core binding.
Depth H5 payloads are copied byte-for-byte, independently of core arrays.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import numpy as np

if __package__:
    from .rebuild import (PROJECT, VERSION, RIGID_TOL, SourceReader,
                          discover_episodes, hash_file, parse_episodes, require,
                          safe_relative, timestamp_array, write_npz)
else:
    from rebuild import (PROJECT, VERSION, RIGID_TOL, SourceReader,
                         discover_episodes, hash_file, parse_episodes, require,
                         safe_relative, timestamp_array, write_npz)

DEPTH_VERSION = "touch4d-depth-v1"
DEPTH_SCRIPT_VERSION = "1.0.0"
# Depth reads existing core bindings; it does not reconstruct these scenes.
# Preserve the historical depth addon scene/episode domain independently of v1.3.
DEPTH_SCENES = {scene: tuple(range(10)) for scene in
                ("001-rope", "008-pink-cloth", "051-cube")}

def require_h5py():
    """Depth is optional; ordinary reconstruction never imports h5py."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Depth mode requires h5py in the server environment; make it available before rebuilding or validating depth") from exc
    return h5py


def checked_path(path):
    """Reject linked components before resolving paths (including missing targets)."""
    path = Path(path).absolute()
    for component in (path, *path.parents):
        require(not component.is_symlink(), f"Refusing symbolic-link path: {component}")
        require(not getattr(component, "is_junction", lambda: False)(), f"Refusing junction: {component}")
    return path.resolve()


def depth_roots(source_root, core_root, depth_root):
    source_root, core_root, depth_root = map(checked_path, (source_root, core_root, depth_root))
    require(source_root.is_dir() and core_root.is_dir(), "Depth mode requires existing source and core roots")
    for protected in (source_root, core_root):
        require(not depth_root.is_relative_to(protected) and not protected.is_relative_to(depth_root),
                f"Depth root must be disjoint from core/source data: {depth_root}, {protected}")
    require(not depth_root.exists() or depth_root.is_dir(), f"Depth root is not a directory: {depth_root}")
    return source_root, core_root, depth_root


def read_core_depth_binding(directory, *, expected_scene=None, expected_episode=None):
    directory = checked_path(directory)
    require(directory.is_dir(), f"Core episode must already exist: {directory}")
    meta_path = checked_path(directory / "metadata.json")
    time_path = checked_path(directory / "timestamps.npz")
    require(meta_path.is_file() and time_path.is_file(), f"Missing core metadata/timestamps: {directory}")
    metadata_hash, timestamp_hash = hash_file(meta_path), hash_file(time_path)
    with meta_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    scene, episode, frames = metadata["scene"], metadata["episode"], metadata["frames"]
    require(metadata["schema_version"] == VERSION and scene in DEPTH_SCENES, "Unsupported core schema/scene")
    require(type(episode) is int and episode in DEPTH_SCENES[scene], "Unreviewed core episode")
    require(type(frames) is int and frames > 0, "Invalid core frame count")
    require(expected_scene is None or expected_scene == scene, "Core scene mismatch")
    require(expected_episode is None or expected_episode == episode, "Core episode mismatch")
    require(directory.name == f"episode_{episode}" and directory.parent.name == scene,
            "Core directory does not match scene/episode")
    cameras = metadata["observation_cameras"]
    require(isinstance(cameras, list) and cameras and all(isinstance(c, str) and
            re.fullmatch(r"[A-Za-z0-9_-]+", c) for c in cameras) and cameras == sorted(set(cameras)),
            "Invalid core camera list")
    require(metadata["output_files"].get("timestamps.npz") == timestamp_hash, "Core timestamp SHA-256 mismatch")
    with np.load(time_path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "Duplicate core timestamp fields")
        obs = archive["observation_timestamp_us"]
    require(obs.shape == (frames,) and obs.dtype == np.int64 and np.all(obs >= 0)
            and np.all(np.diff(obs) > 0), "Invalid core observation timeline")
    require(hash_file(meta_path) == metadata_hash and hash_file(time_path) == timestamp_hash,
            "Core binding changed while reading")
    return metadata, obs, {"metadata_sha256": metadata_hash, "timestamps_sha256": timestamp_hash}


def check_camera_arrays(intrinsics, camera_to_world, cameras):
    require(intrinsics.shape == (len(cameras), 3, 3) and intrinsics.dtype == np.float64,
            "intrinsics must be float64 [V,3,3]")
    require(camera_to_world.shape == (len(cameras), 4, 4) and camera_to_world.dtype == np.float64,
            "camera_to_world must be float64 [V,4,4]")
    require(np.isfinite(intrinsics).all() and np.isfinite(camera_to_world).all(), "Non-finite calibration")
    require(np.all(intrinsics[:, 0, 0] > 0) and np.all(intrinsics[:, 1, 1] > 0)
            and np.allclose(intrinsics[:, 2, :], [0, 0, 1], atol=1e-8, rtol=0)
            and np.all(np.abs(np.linalg.det(intrinsics)) > 1e-12), "Invalid camera intrinsics")
    rotation = camera_to_world[:, :3, :3]
    require(np.allclose(camera_to_world[:, 3, :], [0, 0, 0, 1], atol=RIGID_TOL, rtol=0)
            and np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=RIGID_TOL, rtol=0)
            and np.allclose(np.linalg.det(rotation), 1, atol=RIGID_TOL, rtol=0),
            "Extrinsics must be right-handed rigid camera-to-world transforms")


def read_depth_metadata(directory, core_directory, *, expected_scene=None, expected_episode=None):
    """Lightweight loader check: binding, manifest, calibration, timeline; no depth hashes/reads."""
    directory = checked_path(directory)
    core, obs, binding = read_core_depth_binding(core_directory, expected_scene=expected_scene,
                                                expected_episode=expected_episode)
    with checked_path(directory / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    scene, episode, frames, cameras = core["scene"], core["episode"], core["frames"], core["observation_cameras"]
    require(metadata["schema_version"] == DEPTH_VERSION and metadata["script_version"] == DEPTH_SCRIPT_VERSION,
            "Unsupported depth schema/script version")
    require(type(metadata["episode"]) is int and type(metadata["frames"]) is int,
            "Depth episode/frame count must be integers")
    require(metadata["scene"] == scene and metadata["episode"] == episode and metadata["frames"] == frames,
            "Depth/core scene, episode or frames mismatch")
    require(directory.parent.name == scene and directory.name in
            (f"episode_{episode}", f".episode_{episode}.incomplete"), "Depth directory scene/episode mismatch")
    require(metadata["cameras"] == cameras and metadata["core_binding"] == binding,
            "Stale depth extension: core binding or camera list changed")
    require(metadata["depth_unit"] == "mm" and type(metadata["invalid_value"]) is int
            and metadata["invalid_value"] == 0
            and metadata["depth_kind"] == "camera_z_expected"
            and metadata["extrinsics_convention"] == "camera_to_world"
            and metadata["coordinate_system"] == "native_world_metres",
            "Unsupported depth/calibration convention")
    require(all(type(metadata[key]) is int and metadata[key] > 0 for key in ("height", "width")),
            "Invalid depth image dimensions")
    expected_files = {"camera_parameters.npz", *(f"{c}/rendered_depth.h5" for c in cameras)}
    files = metadata["output_files"]
    require(isinstance(files, dict) and set(files) == expected_files and all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in files.values()),
        "Incomplete or invalid depth output manifest")
    actual_files = set()
    for path in directory.rglob("*"):
        checked_path(path)
        if path.is_file():
            actual_files.add(path.relative_to(directory).as_posix())
    require(actual_files == expected_files | {"metadata.json"}, "Missing or unexpected depth files")
    parameters_path = directory / "camera_parameters.npz"
    require(hash_file(parameters_path) == files["camera_parameters.npz"], "Camera parameter SHA-256 mismatch")
    with np.load(parameters_path, allow_pickle=False) as archive:
        require(len(archive.files) == 3 and set(archive.files) ==
                {"intrinsics", "camera_to_world", "observation_timestamp_us"}, "Bad camera parameter fields")
        intrinsics, extrinsics, timestamps = (archive[k] for k in
                                             ("intrinsics", "camera_to_world", "observation_timestamp_us"))
    check_camera_arrays(intrinsics, extrinsics, cameras)
    require(timestamps.dtype == np.int64 and np.array_equal(timestamps, obs), "Depth/core timeline mismatch")
    return metadata


class DepthSourceReader(SourceReader):
    """Strict archive access plus bounded-memory, byte-preserving depth copying."""

    def archive(self, relative):
        relative = safe_relative(relative)
        checked_path(self.root / relative)
        if relative not in self.archives:
            path = self.root / relative
            before = path.stat()
            tar = self.stack.enter_context(tarfile.open(path, "r:*"))
            members, seen = {}, set()
            for member in tar.getmembers():
                name = safe_relative(member.name)
                require(name not in seen, f"Duplicate archive member: {relative}!{name}")
                seen.add(name)
                require(member.isfile() or member.isdir(), f"Linked/special archive member: {relative}!{name}")
                if member.isfile():
                    members[name] = member
            self.archives[relative] = (tar, members, (before.st_size, before.st_mtime_ns))
        return self.archives[relative]

    def read(self, key, relative, member=None):
        checked_path(self.root / safe_relative(relative))
        return super().read(key, relative, member)

    def stream(self, key, relative, member=None, *, destination=None):
        relative = safe_relative(relative)
        path = checked_path(self.root / relative)
        before = path.stat()
        with ExitStack() as stack:
            if member is None:
                source = stack.enter_context(path.open("rb"))
                size = before.st_size
            else:
                member = safe_relative(member)
                tar, members, signature = self.archive(relative)
                require(signature == (before.st_size, before.st_mtime_ns), f"Source changed: {path}")
                require(member in members, f"Missing depth archive member: {member}")
                source = stack.enter_context(tar.extractfile(members[member]))
                size = members[member].size
            target = stack.enter_context(Path(destination).open("xb")) if destination is not None else None
            digest, count = hashlib.sha256(), 0
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
                count += len(block)
                if target is not None:
                    target.write(block)
        after = path.stat()
        require(count == size and (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                f"Source changed/truncated while streaming: {path}")
        ref = {"path": relative, "member": member, "bytes": count, "sha256": digest.hexdigest()}
        require(key not in self.refs or self.refs[key] == ref, f"Source changed: {key}")
        self.refs[key] = ref
        return ref


def depth_camera_source(reader, key, prefix, relative, recorded_refs):
    """Keep a recorded source location; discover only sources not yet recorded."""
    loose, archive = f"{prefix}/{relative}", f"{prefix}/cameras.tar"
    if key in recorded_refs:
        ref = recorded_refs[key]
        require(isinstance(ref, dict) and set(ref) == {"path", "member", "bytes", "sha256"},
                f"Malformed recorded depth source: {key}")
        require((ref["path"], ref["member"]) in ((loose, None), (archive, relative)),
                f"Wrong recorded depth source path: {key}")
        require(type(ref["bytes"]) is int and ref["bytes"] >= 0
                and isinstance(ref["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
                f"Invalid recorded depth source hash/size: {key}")
        path, member = ref["path"], ref["member"]
    else:
        path, member = (loose, None) if checked_path(reader.root / loose).is_file() else (archive, relative)
    # Missing or changed recorded sources fail; an unrecorded copy cannot replace them.
    source = checked_path(reader.root / path)
    require(source.is_file(), f"Missing depth source: {path}")
    if member is not None:
        _, members, _ = reader.archive(path)
        require(member in members, f"Missing depth archive member: {path}!{member}")
    return path, member


def depth_source_layout(reader, core, obs, *, source_refs=None):
    """New depth uses core timeline provenance; existing depth uses its own manifest."""
    scene, episode, cameras = core["scene"], core["episode"], core["observation_cameras"]
    core_refs = core.get("source_files", {})
    require(isinstance(core_refs, dict), "Malformed core source manifest")
    if source_refs is not None:
        expected = {"episode_metadata", "extrinsics", *(f"camera_timestamps_{c}" for c in cameras),
                    *(f"depth_{c}" for c in cameras)}
        require(isinstance(source_refs, dict) and set(source_refs) == expected,
                "Incomplete or unexpected recorded depth source manifest")
    prefix = safe_relative(core["source_episode_directory"])
    match = re.fullmatch(re.escape(f"processed/{scene}/episode_") + r"(\d+)", prefix)
    require(match is not None and int(match[1]) == episode, "Wrong core source episode directory")
    metadata = json.loads(reader.read("episode_metadata", f"{prefix}/metadata.json").decode("utf-8"))
    source_cameras = metadata["cameras"]
    require(isinstance(source_cameras, list) and sorted(source_cameras) == cameras, "Source/core cameras mismatch")
    wh = metadata["WH"]
    require(isinstance(wh, list) and len(wh) == 2 and all(type(x) is int and x > 0 for x in wh),
            "Invalid source WH")
    all_intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float64)
    require(all_intrinsics.shape == (len(cameras), 3, 3), "Bad source intrinsics dimensions")
    intrinsics = all_intrinsics[[source_cameras.index(c) for c in cameras]]
    payload = reader.read("extrinsics", f"raw/{scene}/calibration_refined/extrinsics.npy")
    # Like the published robot.npy, the trusted original calibration is a pickled dictionary.
    value = np.load(io.BytesIO(payload), allow_pickle=True)
    require(isinstance(value, np.ndarray) and value.shape == (), "Unsupported extrinsics container")
    extrinsics = value.item()
    require(isinstance(extrinsics, dict) and all(c in extrinsics for c in cameras), "Missing camera extrinsics")
    camera_to_world = np.asarray([extrinsics[c] for c in cameras], dtype=np.float64)
    check_camera_arrays(intrinsics, camera_to_world, cameras)
    depth_refs = {}
    for camera in cameras:
        for filename in ("aligned_timestamps.txt", "rendered_depth.h5"):
            relative = f"{camera}/{filename}"
            key = f"camera_timestamps_{camera}" if filename == "aligned_timestamps.txt" else f"depth_{camera}"
            recorded = source_refs if source_refs is not None else core_refs
            ref = depth_camera_source(reader, key, prefix, relative, recorded)
            if filename == "aligned_timestamps.txt":
                timeline = timestamp_array(reader.read(key, *ref), strict=True)
                require(np.array_equal(timeline, obs), f"Full published depth/core timeline differs: {camera}")
            else:
                depth_refs[camera] = ref
    for key in ("episode_metadata", *(f"camera_timestamps_{c}" for c in cameras)):
        if key in core_refs:
            require(reader.refs[key] == core_refs[key], f"Depth source differs from core provenance: {key}")
    if source_refs is not None:
        for key, ref in reader.refs.items():
            require(ref == source_refs[key], f"Depth source hashes/paths changed: {key}")
    return {"width": wh[0], "height": wh[1], "intrinsics": intrinsics,
            "camera_to_world": camera_to_world, "depth_refs": depth_refs}


def check_depth_h5(path, frames, height, width, *, read_pixels=True):
    h5py = require_h5py()
    with h5py.File(checked_path(path), "r") as archive:
        require(set(archive.keys()) == {"data"}, f"Expected one H5 data array: {path}")
        require(isinstance(archive.get("data", getlink=True), h5py.HardLink), f"External/soft H5 data link: {path}")
        data = archive["data"]
        require(isinstance(data, h5py.Dataset) and data.dtype == np.uint16
                and data.shape == (frames, height, width), f"Expected uint16 [F,H,W] depth: {path}")
        require(not data.is_virtual and not data.external, f"External/virtual H5 storage: {path}")
        if read_pixels:
            # Released videos are frame-chunked: one ~1.8 MB frame at a time avoids
            # repeatedly decompressing that chunk when it exceeds the HDF5 cache.
            for frame in range(frames):
                data[frame]


def depth_output_paths(depth_root, scene, episode, *, allow_existing=True):
    require(scene in DEPTH_SCENES and type(episode) is int and episode in DEPTH_SCENES[scene],
            "Unreviewed depth scene/episode")
    target = Path(depth_root) / scene / f"episode_{episode}"
    partial, backup = (target.with_name(f".episode_{episode}.{suffix}") for suffix in ("incomplete", "previous"))
    for path in (target, partial, backup):
        checked_path(path)
        require(not path.exists() or path.is_dir(), f"Depth output is not a directory: {path}")
    if target.exists():
        for path in target.rglob("*"):
            checked_path(path)
    require(not partial.exists(), f"Inspect existing unfinished depth output: {partial}")
    require(allow_existing or not target.exists(), f"Depth output already exists: {target}")
    require(not backup.exists() or target.exists(), f"Recover depth backup before rebuilding: {backup}")
    return target, partial, backup


def rebuild_depth_episode(source_root, core_root, depth_root, scene, episode, *, overwrite=False):
    """Copy only the depth extension. Never writes or fully revalidates core arrays."""
    require_h5py()
    source_root, core_root, depth_root = depth_roots(source_root, core_root, depth_root)
    core_directory = core_root / scene / f"episode_{episode}"
    core, obs, binding = read_core_depth_binding(core_directory, expected_scene=scene, expected_episode=episode)
    target, partial, backup = depth_output_paths(depth_root, scene, episode)
    if target.exists() and not overwrite:
        result = validate_depth_episode(target, core_directory=core_directory,
                    expected_scene=scene, expected_episode=episode, source_root=source_root, verify_sources=True)
        print(f"Skipped (full depth validation passed): {target}", flush=True)
        return {**result, "skipped": True}
    require(not backup.exists(), f"Inspect previous depth backup before overwriting: {backup}")
    old_identity = (target.stat().st_dev, target.stat().st_ino) if target.exists() else None
    with DepthSourceReader(source_root) as reader:
        plan = depth_source_layout(reader, core, obs)
        partial.mkdir(parents=True, exist_ok=False)
        try:
            output_files = {}
            for camera in core["observation_cameras"]:
                (partial / camera).mkdir()
                destination = partial / camera / "rendered_depth.h5"
                ref = reader.stream(f"depth_{camera}", *plan["depth_refs"][camera], destination=destination)
                check_depth_h5(destination, len(obs), plan["height"], plan["width"], read_pixels=False)
                output_files[f"{camera}/rendered_depth.h5"] = ref["sha256"]
            params = partial / "camera_parameters.npz"
            write_npz(params, {"intrinsics": plan["intrinsics"], "camera_to_world": plan["camera_to_world"],
                               "observation_timestamp_us": obs})
            output_files["camera_parameters.npz"] = hash_file(params)
            metadata = {"schema_version": DEPTH_VERSION, "script_version": DEPTH_SCRIPT_VERSION,
                "scene": scene, "episode": episode, "frames": len(obs), "cameras": core["observation_cameras"],
                "height": plan["height"], "width": plan["width"], "depth_unit": "mm", "invalid_value": 0,
                "depth_kind": "camera_z_expected", "extrinsics_convention": "camera_to_world",
                "coordinate_system": "native_world_metres", "core_binding": binding,
                "source_root": str(source_root), "source_episode_directory": core["source_episode_directory"],
                "source_files": reader.refs, "output_files": output_files,
                "created_utc": datetime.now(timezone.utc).isoformat()}
            with (partial / "metadata.json").open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            result = validate_depth_episode(partial, core_directory=core_directory,
                        expected_scene=scene, expected_episode=episode, source_root=source_root, verify_sources=True)
            checked_path(target)
            if old_identity is None:
                require(not target.exists(), f"Depth output appeared during copying: {target}")
                partial.rename(target)
            else:
                require(target.is_dir() and (target.stat().st_dev, target.stat().st_ino) == old_identity,
                        f"Depth output changed during copying: {target}")
                require(not backup.exists(), f"Depth backup appeared during copying: {backup}")
                target.rename(backup)
                try:
                    partial.rename(target)
                except Exception:
                    backup.rename(target)
                    raise
        except Exception:
            print(f"Depth supplement failed; core data untouched. Inspect {partial}, {target}, {backup}", file=sys.stderr)
            raise
    print(f"Completed depth only: {target}", flush=True)
    if old_identity is not None:
        try:
            shutil.rmtree(backup)
        except Exception as exc:
            print(f"WARNING: depth publication succeeded; old backup cleanup failed at {backup}: {exc}. "
                  "Validation and default skipping remain available.", file=sys.stderr, flush=True)
    return {**result, "skipped": False}


def run_depth_only(args):
    require_h5py()
    source_root, core_root, depth_root = depth_roots(args.source_root, args.output_root,
        args.depth_root if args.depth_root is not None else args.output_root.parent / DEPTH_VERSION)
    available = discover_episodes(core_root / args.scene)
    episodes = args.episodes if args.episodes is not None else list(available)
    require(set(episodes) <= set(available), "Depth mode cannot create missing core episodes; rebuild core separately")
    import fcntl
    scene_dir = depth_root / args.scene
    with ExitStack() as stack:
        # Core reconstruction locks this same directory exclusively. Hold a shared
        # lock through publication so a concurrently replaced core cannot bind stale depth.
        core_descriptor = os.open(core_root / args.scene, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, core_descriptor)
        try:
            fcntl.flock(core_descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"A core reconstruction is using {core_root / args.scene}") from exc
        # Preflight every selected episode before any mutation. No robot/GS/tactile reads.
        for episode in episodes:
            core, obs, _ = read_core_depth_binding(available[episode], expected_scene=args.scene, expected_episode=episode)
            target, _, _ = depth_output_paths(depth_root, args.scene, episode)
            source_refs = None
            if target.exists() and not args.overwrite:
                metadata = read_depth_metadata(target, available[episode], expected_scene=args.scene,
                                               expected_episode=episode)
                check_depth_sources_schema(metadata)
                source_refs = metadata["source_files"]
            with DepthSourceReader(source_root) as reader:
                depth_source_layout(reader, core, obs, source_refs=source_refs)
        scene_dir.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(scene_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, descriptor)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another depth supplement is using {scene_dir}") from exc
        for episode in episodes:
            rebuild_depth_episode(source_root, core_root, depth_root, args.scene, episode, overwrite=args.overwrite)
    return 0


def equal(actual, expected, name):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype,
            f"{name}: recomputed shape/dtype mismatch")
    require(np.array_equal(actual, expected, equal_nan=True), f"{name}: values differ from source/recomputation")


def check_source_ref(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "member", "bytes", "sha256"},
            "Malformed source reference")
    safe_relative(ref["path"])
    if ref["member"] is not None:
        safe_relative(ref["member"])
    require(isinstance(ref["bytes"], int) and ref["bytes"] >= 0, "Invalid source byte count")
    require(isinstance(ref["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
            "Invalid source SHA-256")


def check_depth_sources_schema(metadata):
    scene, episode, cameras = metadata["scene"], metadata["episode"], metadata["cameras"]
    prefix = safe_relative(metadata["source_episode_directory"])
    match = re.fullmatch(re.escape(f"processed/{scene}/episode_") + r"(\d+)", prefix)
    require(match is not None and int(match[1]) == episode, "Wrong depth source episode directory")
    refs = metadata["source_files"]
    expected = {"episode_metadata", "extrinsics", *(f"camera_timestamps_{c}" for c in cameras),
                *(f"depth_{c}" for c in cameras)}
    require(isinstance(refs, dict) and set(refs) == expected, "Incomplete or unexpected depth source manifest")
    for ref in refs.values():
        check_source_ref(ref)
    for key, path in (("episode_metadata", f"{prefix}/metadata.json"),
                      ("extrinsics", f"raw/{scene}/calibration_refined/extrinsics.npy")):
        require((refs[key]["path"], refs[key]["member"]) == (path, None), f"Wrong depth source: {key}")
    for camera in cameras:
        for key, filename in ((f"camera_timestamps_{camera}", "aligned_timestamps.txt"),
                              (f"depth_{camera}", "rendered_depth.h5")):
            relative = f"{camera}/{filename}"
            ref = refs[key]
            require((ref["path"], ref["member"]) in
                    ((f"{prefix}/{relative}", None), (f"{prefix}/cameras.tar", relative)),
                    f"Wrong depth source path: {key}")
        require(refs[f"depth_{camera}"]["sha256"] == metadata["output_files"][f"{camera}/rendered_depth.h5"],
                f"Depth must preserve exact source H5 bytes: {camera}")


def validate_depth_episode(directory, *, core_directory, expected_scene=None, expected_episode=None,
                           source_root=None, verify_sources=True):
    """Validate only the isolated extension and its small binding to existing core data."""
    require_h5py()
    directory, core_directory = map(checked_path, (directory, core_directory))
    require(not directory.is_relative_to(core_directory) and not core_directory.is_relative_to(directory),
            "Depth and core episode directories must be disjoint")
    metadata = read_depth_metadata(directory, core_directory,
                    expected_scene=expected_scene, expected_episode=expected_episode)
    check_depth_sources_schema(metadata)
    scene, episode, cameras, frames = (metadata[k] for k in ("scene", "episode", "cameras", "frames"))
    for relative, digest in metadata["output_files"].items():
        path = directory / relative
        require(hash_file(path) == digest, f"Depth output SHA-256 mismatch: {relative}")
        if relative.endswith(".h5"):
            require(path.stat().st_size == metadata["source_files"][f"depth_{Path(relative).parent.name}"]["bytes"],
                    f"Depth source/output byte count mismatch: {relative}")
            check_depth_h5(path, frames, metadata["height"], metadata["width"])
    if verify_sources:
        source_root = checked_path(source_root or metadata["source_root"])
        core, obs, binding = read_core_depth_binding(core_directory, expected_scene=scene,
                                                             expected_episode=episode)
        require(binding == metadata["core_binding"], "Core changed during depth verification")
        require(core["source_episode_directory"] == metadata["source_episode_directory"], "Core source directory changed")
        with DepthSourceReader(source_root) as reader:
            plan = depth_source_layout(reader, core, obs, source_refs=metadata["source_files"])
            require((plan["height"], plan["width"]) == (metadata["height"], metadata["width"]),
                    "Depth dimensions differ from source metadata")
            with np.load(directory / "camera_parameters.npz", allow_pickle=False) as params:
                equal(params["intrinsics"], plan["intrinsics"], "Depth source intrinsics")
                equal(params["camera_to_world"], plan["camera_to_world"], "Depth source camera_to_world")
            for camera in cameras:
                reader.stream(f"depth_{camera}", *plan["depth_refs"][camera])
            require(reader.refs == metadata["source_files"], "Depth source hashes/paths changed")
    # A core overwrite during validation must not leave a stale extension marked valid.
    _, _, binding = read_core_depth_binding(core_directory, expected_scene=scene, expected_episode=episode)
    require(binding == metadata["core_binding"], "Core changed during depth validation")
    return {"scene": scene, "episode": episode, "status": "PASS", "frames": frames,
            "cameras": len(cameras), "height": metadata["height"], "width": metadata["width"],
            "depth_only": True, "source_verification": "full" if verify_sources else "skipped (structure-only)",
            "source_files_verified": len(metadata["source_files"]) if verify_sources else 0}


def validate_depth_main(args):
    core_root = checked_path(args.data_root)
    depth_root = checked_path(args.depth_root or (core_root.parent / DEPTH_VERSION))
    require(not depth_root.is_relative_to(core_root) and not core_root.is_relative_to(depth_root),
            "Depth root must be disjoint from core root")
    available = discover_episodes(core_root / args.scene)
    episodes = args.episodes if args.episodes is not None else list(available)
    require(set(episodes) <= set(available), "Missing core episodes for depth validation")
    failed = False
    for episode in episodes:
        target = depth_root / args.scene / f"episode_{episode}"
        try:
            require(not target.with_name(f".episode_{episode}.incomplete").exists(), "Unpublished depth partial exists")
            result = validate_depth_episode(target, core_directory=available[episode], expected_scene=args.scene,
                                           expected_episode=episode, source_root=args.source_root,
                                           verify_sources=not args.structure_only)
            backup = target.with_name(f".episode_{episode}.previous")
            if backup.exists() or backup.is_symlink():
                print(f"WARNING: published depth validated; retained backup needs inspection: {backup}",
                      file=sys.stderr, flush=True)
        except (ValueError, OSError, KeyError, TypeError, ImportError, tarfile.TarError) as exc:
            failed = True
            result = {"scene": args.scene, "episode": episode, "depth_only": True, "status": "FAIL", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return int(failed)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode",choices=("build","validate"),default="build")
    parser.add_argument("--scene",required=True,choices=sorted(DEPTH_SCENES))
    parser.add_argument("--source-root",type=Path,default=PROJECT / "datasets/deform360")
    parser.add_argument("--core-root",type=Path,default=PROJECT / "datasets/touch4d-v1")
    parser.add_argument("--depth-root",type=Path)
    parser.add_argument("--episodes",type=parse_episodes)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--structure-only",action="store_true")
    args=parser.parse_args(argv)
    require(args.mode=="validate" or not args.structure_only,"--structure-only requires --mode validate")
    require(args.mode=="build" or not args.overwrite,"--overwrite requires --mode build")
    args.output_root=args.core_root
    args.data_root=args.core_root
    if args.mode=="validate":
        return validate_depth_main(args)
    require(os.name!="nt","Depth reconstruction outputs belong on the Linux server")
    return run_depth_only(args)


if __name__=="__main__":
    sys.dont_write_bytecode=True
    try:
        raise SystemExit(main())
    except (ValueError,OSError,KeyError,TypeError,ImportError,tarfile.TarError) as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        raise SystemExit(1)
