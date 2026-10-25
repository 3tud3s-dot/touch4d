"""Local, disk-free tests: python -B -m unittest scripts.dataloader.test_dataloader -v"""

from contextlib import ExitStack, contextmanager
import copy
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import pickle
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

try:
    import h5py
except ImportError:  # Existing point-only tests remain usable without the optional reader.
    h5py = None

from . import dataset as dataset_module
from .check import check_batch, main as check_main
from .collate import FIXED_FIELDS, batch_horizon, collate_fn, fixed_fields, validate_depth_batch
from .dataset import DataFormatError, Touch4DDataset
from .geometry import (HAND_ORDER, STREAM_ORDER, action_conditions, hand_slots,
                       pose_validity, tactile_world_coordinates)
from .loader import make_dataloader
from .splits import SPLITS, episode_ids, window_starts
from ..rebuild_dataset import rebuild as rebuild_module
from ..rebuild_dataset import validate as validate_module


def encoded_file(name, value):
    """Serialize test sources in memory, including bytes used by hash checks."""
    if isinstance(value, bytes):
        return value
    if str(name).endswith(".json"):
        return json.dumps(value, sort_keys=True).encode("utf-8")
    stream = BytesIO()
    np.savez(stream, **value)
    return stream.getvalue()


def depth_h5_bytes(values):
    stream = BytesIO()
    with h5py.File(stream, "w") as archive:
        archive.create_dataset("data", data=values, chunks=(1, *values.shape[1:]), compression="gzip")
    return stream.getvalue()


def metadata(scene="001-rope", episode=0, frames=12, hands=("right",)):
    return {"schema_version": "touch4d-v1", "script_version": "1.3.0", "scene": scene,
            "episode": episode, "frames": frames, "hand_order": list(HAND_ORDER),
            "stream_order": list(STREAM_ORDER), "robot_slot_hands": list(hands),
            "physical_hand_to_robot_slot": {h: hands.index(h) if h in hands else None for h in HAND_ORDER},
            "gs_point_counts": [1 + (episode + frame) % 4 for frame in range(frames)],
            "parameters": {"rigid_atol": 1e-6, "rigid_rtol": 0, "opening_clip_m": [0.04, 0.112],
                           "crop_rows": [0, 12], "invalid_columns": [31], "row_flip": False,
                           "max_age_us": 75000, "gs_coordinate_transform": "identity; native world metres",
                           "quaternion_order": "WXYZ"}}


def encode_actions(transforms, openings):
    values = np.zeros((*openings.shape, 5, 3), dtype=transforms.dtype)
    values[..., 0, :] = transforms[..., :3, 3]
    values[..., 1:4, :] = transforms[..., :3, :3]
    values[..., 4, 0] = openings
    return values


class MemoryReconstruction:
    """Supply real NPZ bytes to np.load without creating any filesystem fixtures."""

    def __init__(self, scene="001-rope", frames=12):
        self.root = Path("__memory_only_touch4d__").resolve()
        self.scene = scene
        self.files = {}
        for episode in range(10):
            self.add_episode(episode, frames)

    def path(self, episode, name):
        return self.root / self.scene / f"episode_{episode}" / name

    def get(self, episode, name):
        return self.files[self.path(episode, name)]

    def add_episode(self, episode, frames, hands=None):
        if hands is None:
            hands = ("right",) if episode < 5 else ("right", "left")
        meta = metadata(self.scene, episode, frames, hands)
        self.files[self.path(episode, "metadata.json")] = meta
        transforms = np.broadcast_to(np.eye(4), (frames, len(hands), 4, 4)).copy()
        for slot in range(len(hands)):
            transforms[:, slot, :3, 3] = np.stack((np.arange(frames) + 100 * slot,
                                                  np.full(frames, episode), np.zeros(frames)), axis=-1)
        openings = np.full((frames, len(hands)), 0.08, dtype=np.float64)
        points = np.zeros((frames, 2, 30, 3), dtype=np.float32)
        controller_mask = np.zeros((frames, 2, 30), dtype=bool)
        for hand, name in enumerate(HAND_ORDER):
            if name in hands:
                points[:, hand] = 10 * (hand + 1)
                controller_mask[:, hand] = True
        self.files[self.path(episode, "controller.npz")] = {
            "T_worlds": transforms, "openings": openings, "actions": encode_actions(transforms, openings),
            "controller_points": points, "controller_valid_mask": controller_mask,
            "opening_clipped": np.zeros((frames, 2), dtype=bool)}
        valid = np.ones((frames, 4, 12, 32), dtype=bool)
        valid[..., 31] = False
        self.files[self.path(episode, "tactile.npz")] = {
            "tactile": np.zeros(valid.shape, dtype=np.float32), "valid_mask": valid}
        ts = 1_700_000_000_000_000 + np.arange(frames, dtype=np.int64) * 100_000
        times = {"observation_timestamp_us": ts, "tactile_age_us": np.zeros((frames, 4), dtype=np.int64),
                 "tactile_raw_index": np.tile(np.arange(frames, dtype=np.int64)[:, None], (1, 4))}
        times.update({f"raw_tactile_timestamp_us_{stream}": ts.copy() for stream in range(4)})
        self.files[self.path(episode, "timestamps.npz")] = times
        for frame, n in enumerate(meta["gs_point_counts"]):
            xyz = np.zeros((n, 3), dtype=np.float32)
            xyz[:, 0], xyz[:, 1], xyz[:, 2] = episode, frame, np.arange(n)
            self.files[self.path(episode, f"gs/frame_{frame:06d}.npz")] = {
                "xyz": xyz, "log_scales": np.full((n, 3), -2, dtype=np.float32),
                "quaternions": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
                "opacity_logits": np.full((n, 1), 0.3, dtype=np.float32),
                "sh_dc": np.full((n, 3), 0.1, dtype=np.float32), "sh_rest": np.zeros((n, 45), dtype=np.float32),
                "frame_id": np.asarray(frame, dtype=np.int64), "timestamp_us": np.asarray(ts[frame])}

    @contextmanager
    def mounted(self):
        original_load = np.load

        def read_text(path, **kwargs):
            if path not in self.files:
                raise FileNotFoundError(str(path))
            return json.dumps(self.files[path])

        def load(path, **kwargs):
            if hasattr(path, "read"):
                return original_load(path, **kwargs)
            if Path(path) not in self.files:
                raise FileNotFoundError(str(path))
            values = self.files[Path(path)]
            stream = BytesIO()
            if isinstance(values, bytes):
                stream.write(values)
            else:
                np.savez(stream, **values)
            stream.seek(0)
            return original_load(stream, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "read_text", read_text))
            stack.enter_context(patch.object(Path, "is_file", lambda path: path in self.files))
            stack.enter_context(patch.object(dataset_module.np, "load", load))
            yield self


class MemoryBackedDataset(Touch4DDataset):
    """Spawn-safe test fixture: each worker mounts its own in-memory NPZ source."""

    def __init__(self, scene, split, root, **kwargs):
        self.memory_source = MemoryReconstruction(scene, frames=max(12, kwargs.get("horizon", 5)+7))
        with self.memory_source.mounted():
            super().__init__(scene, split, root, **kwargs)

    def __getitem__(self, index):
        with self.memory_source.mounted():
            return super().__getitem__(index)


class MemoryDepthReconstruction(MemoryReconstruction):
    """The core and its optional depth extension exist only as memory buffers."""

    cameras = ("brics-odroid-001_cam0", "brics-odroid-002_cam0", "brics-odroid-003_cam0")

    def __init__(self, scene="001-rope", frames=12):
        super().__init__(scene, frames)
        self.depth_root = self.root.parent / "touch4d-depth-v1"
        self.source_root = self.root.parent / "__memory_only_deform360__"
        for episode in range(10):
            self.add_depth(episode, frames)

    def depth_path(self, episode, name):
        return self.depth_root / self.scene / f"episode_{episode}" / name

    def digest(self, path):
        return hashlib.sha256(encoded_file(path, self.files[path])).hexdigest()

    def add_depth(self, episode, frames, cameras=None):
        cameras = self.cameras if cameras is None else tuple(cameras)
        parent = self.depth_path(episode, "metadata.json").parent
        for path in list(self.files):
            if parent in path.parents:
                del self.files[path]
        core = self.get(episode, "metadata.json")
        core["observation_camera"] = cameras[0]
        core["observation_cameras"] = list(cameras)
        core["source_episode_directory"] = f"processed/{self.scene}/episode_{episode}"
        core["output_files"] = {"timestamps.npz": self.digest(self.path(episode, "timestamps.npz"))}
        times = self.get(episode, "timestamps.npz")["observation_timestamp_us"]
        intrinsic = np.repeat(np.array([[[700., 0., 2.], [0., 710., 1.5], [0., 0., 1.]]]), len(cameras), axis=0)
        camera_to_world = np.repeat(np.eye(4, dtype=np.float64)[None], len(cameras), axis=0)
        for i in range(len(cameras)):
            intrinsic[i, 0, 0] += i
            camera_to_world[i, 0, 3] = i / 10
        self.files[self.depth_path(episode, "camera_parameters.npz")] = {
            "intrinsics": intrinsic, "camera_to_world": camera_to_world,
            "observation_timestamp_us": times.copy()}
        for i, camera in enumerate(cameras):
            values = np.broadcast_to((1000 + np.arange(frames) * 20 + 100 * i)[:, None, None],
                                     (frames, 3, 4)).astype(np.uint16).copy()
            values[:, 0, 0] = 0
            self.files[self.depth_path(episode, f"{camera}/rendered_depth.h5")] = depth_h5_bytes(values)
        names = ["camera_parameters.npz"] + [f"{camera}/rendered_depth.h5" for camera in cameras]
        prefix = core["source_episode_directory"]
        source_values = {
            "episode_metadata": (f"{prefix}/metadata.json", {
                "cameras": list(cameras), "WH": [4, 3], "intrinsics": intrinsic.tolist()}),
        }
        extrinsics = BytesIO()
        np.save(extrinsics, dict(zip(cameras, camera_to_world)), allow_pickle=True)
        source_values["extrinsics"] = (f"raw/{self.scene}/calibration_refined/extrinsics.npy", extrinsics.getvalue())
        for camera in cameras:
            source_values[f"camera_timestamps_{camera}"] = (f"{prefix}/{camera}/aligned_timestamps.txt",
                "\n".join(f"frame_{int(value)}_{i}" for i, value in enumerate(times)).encode("utf-8"))
            source_values[f"depth_{camera}"] = (f"{prefix}/{camera}/rendered_depth.h5",
                self.files[self.depth_path(episode, f"{camera}/rendered_depth.h5")])
        refs = {}
        for key, (relative, value) in source_values.items():
            path = self.source_root / relative
            self.files[path] = value
            payload = encoded_file(path, value)
            refs[key] = {"path": relative, "member": None, "bytes": len(payload),
                         "sha256": hashlib.sha256(payload).hexdigest()}
        # Production core metadata retains these references. Keep the fixture honest
        # so depth validation exercises timeline provenance, not only equal arrays.
        core["source_files"] = {key: copy.deepcopy(refs[key]) for key in
                                ("episode_metadata", *(f"camera_timestamps_{c}" for c in cameras))}
        core["source_files"]["observation"] = copy.deepcopy(refs[f"camera_timestamps_{cameras[0]}"])
        self.files[self.depth_path(episode, "metadata.json")] = {
            "schema_version": "touch4d-depth-v1", "script_version": "1.0.0",
            "scene": self.scene, "episode": episode,
            "frames": frames, "cameras": list(cameras), "height": 3, "width": 4,
            "depth_unit": "mm", "invalid_value": 0, "depth_kind": "camera_z_expected",
            "extrinsics_convention": "camera_to_world", "coordinate_system": "native_world_metres",
            "source_root": str(self.source_root),
            "source_episode_directory": core["source_episode_directory"],
            "core_binding": {"metadata_sha256": self.digest(self.path(episode, "metadata.json")),
                             "timestamps_sha256": self.digest(self.path(episode, "timestamps.npz"))},
            "output_files": {name: self.digest(self.depth_path(episode, name)) for name in names},
            "source_files": refs}

    @contextmanager
    def mounted(self):
        original_h5 = h5py.File

        def open_file(path, mode="r", *args, **kwargs):
            if mode not in ("r", "rb"):
                raise AssertionError(f"Test attempted a filesystem write: {path} ({mode})")
            if path not in self.files:
                raise FileNotFoundError(str(path))
            payload = encoded_file(path, self.files[path])
            return BytesIO(payload) if "b" in mode else StringIO(payload.decode("utf-8"))

        def open_h5(path, mode="r", *args, **kwargs):
            if isinstance(path, (Path, str)):
                path = Path(path)
                if path not in self.files:
                    raise FileNotFoundError(str(path))
                return original_h5(BytesIO(encoded_file(path, self.files[path])), mode, *args, **kwargs)
            return original_h5(path, mode, *args, **kwargs)

        def is_dir(path):
            return any(path in file.parents for file in self.files)

        original_stat = Path.stat

        def stat(path, *args, **kwargs):
            if path in self.files:
                return SimpleNamespace(st_size=len(encoded_file(path, self.files[path])),
                                       st_mtime_ns=1, st_dev=1, st_ino=1)
            return original_stat(path, *args, **kwargs)

        with super().mounted(), ExitStack() as stack:
            stack.enter_context(patch.object(Path, "open", open_file))
            stack.enter_context(patch.object(Path, "is_dir", is_dir))
            stack.enter_context(patch.object(Path, "exists", lambda path: path in self.files or is_dir(path)))
            stack.enter_context(patch.object(Path, "is_symlink", lambda path: False))
            stack.enter_context(patch.object(Path, "stat", stat))
            stack.enter_context(patch.object(Path, "rglob", lambda path, pattern: iter(
                file for file in self.files if path in file.parents)))
            stack.enter_context(patch.object(h5py, "File", open_h5))
            yield self


class SplitTests(unittest.TestCase):
    def test_reviewed_splits_and_curriculum(self):
        expected = {
            "001-rope": ((1, 2, 4, 5, 7, 8), (3, 6), (0, 9)),
            "008-pink-cloth": ((0, 2, 4, 5, 7, 9), (1, 6), (3, 8)),
            "051-cube": ((0, 2, 3, 5, 7, 9), (4, 8), (1, 6)),
        }
        for scene, groups in expected.items():
            self.assertEqual(tuple(episode_ids(scene, split) for split in ("train", "val", "test")), groups)
            self.assertEqual(sorted(x for group in groups for x in group), list(range(10)))
            self.assertEqual(episode_ids(scene, "train", "single"), groups[0][:3])
            with self.assertRaises(ValueError):
                episode_ids(scene, "val", "single")
        with self.assertRaises(ValueError):
            episode_ids("001", "train")

    def test_window_boundaries(self):
        self.assertEqual(list(window_starts(5, "train")), [])
        self.assertEqual(list(window_starts(6, "train")), [0])
        self.assertEqual(list(window_starts(12, "train")), list(range(7)))
        self.assertEqual(list(window_starts(12, "val")), [0, 5])
        self.assertEqual(list(window_starts(10, "test")), [0])
        self.assertEqual(list(window_starts(11, "test")), [0, 5])

    def test_configurable_horizon_keeps_split_strides_and_rejects_invalid_lengths(self):
        for horizon in (1, 5, 10, 15):
            self.assertEqual(list(window_starts(horizon, "train", horizon)), [])
            self.assertEqual(list(window_starts(horizon+1, "train", horizon)), [0])
            self.assertEqual(list(window_starts(horizon+7, "train", horizon)), list(range(7)))
            self.assertEqual(list(window_starts(horizon+7, "val", horizon)), [0, 5])
            self.assertEqual(list(window_starts(horizon+7, "test", horizon)), [0, 5])
        self.assertEqual(fixed_fields(5), FIXED_FIELDS)
        for bad in (0, -1, 1.5, True, "15", None):
            with self.subTest(horizon=bad), self.assertRaises(ValueError):
                window_starts(20, "train", bad)


class GeometryTests(unittest.TestCase):
    def test_metadata_binding_not_slot_number(self):
        self.assertEqual(hand_slots(metadata(hands=("right", "left"))), (1, 0))
        self.assertEqual(hand_slots(metadata(hands=("left", "right"))), (0, 1))
        self.assertEqual(hand_slots(metadata(hands=("left",))), (0, None))
        bad = metadata()
        bad["physical_hand_to_robot_slot"]["right"] = False
        with self.assertRaisesRegex(ValueError, "binding"):
            hand_slots(bad)
        bad["physical_hand_to_robot_slot"]["right"] = 1
        with self.assertRaises(ValueError):
            hand_slots(bad)

    def test_rigid_tolerance_and_unclipped_action(self):
        transforms = np.tile(np.eye(4), (8, 1, 1, 1))
        openings = np.full((8, 1), 0.2)
        transforms[1, 0, 0, 0] += 1e-7  # within orthogonality tolerance
        transforms[2, 0, 0, 0] += 1e-5
        transforms[3, 0, 0, 0] = -1  # orthogonal reflection is not a rigid rotation
        transforms[4, 0, 3, 0] = 2e-6
        transforms[5, 0, 0, 3] = np.nan
        openings[6, 0] = np.nan
        transforms[7, 0, 3, 0] = 0.5e-6
        values = encode_actions(transforms, openings)
        expected = [True, True, False, False, False, False, False, True]
        self.assertEqual(pose_validity(transforms, openings)[:, 0].tolist(), expected)
        state, mask = action_conditions(values, transforms, openings, (None, 0))
        self.assertEqual(mask[:, 1].tolist(), expected)
        self.assertFalse(mask[:, 0].any())
        self.assertTrue(np.all(state[~mask] == 0))
        self.assertAlmostEqual(float(state[0, 1, -1]), 0.2)
        np.testing.assert_array_equal(state[0, 1, 3:12], np.eye(3).reshape(-1))

    def test_tactile_geometry_orientation_rotation_and_clipping(self):
        transforms = np.tile(np.eye(4), (2, 1, 1, 1))
        transforms[:, 0, :3, 3] = [1, 2, 3]
        transforms[1, 0, :3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        openings = np.array([[0.01], [0.2]])  # clipped only for geometry
        valid = np.ones((2, 4, 12, 32), dtype=bool)
        valid[..., 31] = False
        poses = np.tile([False, True], (2, 1))
        xyz, spatial = tactile_world_coordinates(transforms, openings, (None, 0), poses, valid)
        np.testing.assert_allclose(xyz[0, 2, 0, 0], [1.00253758, 2.1395, 2.9862], atol=2e-7, rtol=0)
        np.testing.assert_allclose(xyz[0, 3, 0, 0], [0.99746242, 2.1395, 2.9872], atol=2e-7, rtol=0)
        np.testing.assert_allclose(xyz[0, 2, 11, 30] - xyz[0, 2, 0, 0], [0, 0.06, 0.022], atol=2e-7, rtol=0)
        np.testing.assert_allclose(xyz[1, 2, 0, 0], [0.96953758, 2.0138, 3.1395], atol=2e-7, rtol=0)
        self.assertFalse(spatial[:, :2].any())
        self.assertTrue(valid[:, :2, :, :31].all())  # absent pose does not remove 2D readings
        self.assertTrue(np.all(xyz[~spatial] == 0))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.data = MemoryReconstruction()

    def test_long_windows_read_distinct_future_frames_and_only_past_conditions(self):
        for horizon in (10, 15):
            for scene in SPLITS:
                source = MemoryReconstruction(scene, frames=horizon+7)
                with self.subTest(horizon=horizon, scene=scene), source.mounted():
                    ds = Touch4DDataset(scene, "train", source.root, horizon=horizon)
                    sample = ds[ds.windows.index((5, 2))]
                    self.assertEqual(ds.horizon, horizon)
                    self.assertEqual(len(ds), 42)
                    self.assertEqual(sample["meta"]["frame_ids"].tolist(), list(range(2, horizon+3)))
                    self.assertEqual(sample["inputs"]["action_state"][:, 0, 0].tolist(), list(range(102, horizon+102)))
                    self.assertEqual(sample["inputs"]["action_state"][:, 1, 0].tolist(), list(range(2, horizon+2)))
                    self.assertEqual([int(t[0, 1]) for t in sample["targets"]["target_xyz"]], list(range(3, horizon+3)))
                    self.assertEqual(set(sample["meta"]), {"scene", "episode_id", "frame_ids", "observation_timestamp_us"})
                    # Mutating t+H conditions and future GS cannot change the initial/condition inputs.
                    initial = copy.deepcopy(sample["inputs"])
                    source.get(5, "tactile.npz")["tactile"][horizon+2, :, :, :31] = 123
                    source.get(5, f"gs/frame_{horizon+2:06d}.npz")["xyz"] += .5
                    fresh = Touch4DDataset(scene, "train", source.root, horizon=horizon)
                    result = fresh[fresh.windows.index((5, 2))]
                    for key in initial:
                        self.assertTrue(torch.equal(initial[key], result["inputs"][key]), key)
                    self.assertFalse(torch.equal(sample["targets"]["target_xyz"][-1], result["targets"]["target_xyz"][-1]))
                    check_batch(collate_fn([sample, ds[-1]]))

    def test_h5_explicit_and_default_samples_are_identical(self):
        with self.data.mounted():
            default = Touch4DDataset(self.data.scene, "train", self.data.root)
            explicit = Touch4DDataset(self.data.scene, "train", self.data.root, horizon=5)
            self.assertEqual(default.windows, explicit.windows)
            for index in (0, -1):
                a, b = default[index], explicit[index]
                for group in ("inputs", "targets", "meta"):
                    self.assertEqual(a[group].keys(), b[group].keys())
                    for key, value in a[group].items():
                        other = b[group][key]
                        if isinstance(value, torch.Tensor):
                            self.assertTrue(torch.equal(value, other))
                        elif isinstance(value, list):
                            self.assertEqual(len(value), len(other))
                            self.assertTrue(all(torch.equal(x, y) for x, y in zip(value, other)))
                        else:
                            self.assertEqual(value, other)

    def test_sample_timing_attributes_and_controller_order(self):
        with self.data.mounted():
            ds = Touch4DDataset(self.data.scene, "train", self.data.root)
            sample = ds[ds.windows.index((5, 2))]
            x, target, meta = sample["inputs"], sample["targets"], sample["meta"]
            self.assertEqual(meta["frame_ids"].tolist(), list(range(2, 8)))
            self.assertEqual(x["action_state"][:, 0, 0].tolist(), list(range(102, 107)))
            self.assertEqual(x["action_state"][:, 1, 0].tolist(), list(range(2, 7)))
            self.assertTrue(torch.all(x["controller_points"][:, 0] == 10))
            self.assertTrue(torch.all(x["controller_points"][:, 1] == 20))
            self.assertTrue(torch.all(x["gs_attr"][:, :3] == -2))
            torch.testing.assert_close(x["gs_attr"][0, 3:7], torch.tensor([1., 0., 0., 0.]))
            self.assertAlmostEqual(x["gs_attr"][0, 7].item(), 0.3)
            self.assertEqual([int(t[0, 1]) for t in target["target_xyz"]], [3, 4, 5, 6, 7])
            self.assertEqual([len(t) for t in target["target_xyz"]], [1, 2, 3, 4, 1])
            self.assertEqual([tuple(t.shape) for t in target["target_attr"]],
                             [(1, 11), (2, 11), (3, 11), (4, 11), (1, 11)])
            self.assertNotIn("delta_xyz", target)
            self.assertNotIn("sh_rest", x)
            self.assertEqual(len(ds), 42)

    def test_future_attributes_follow_native_frame_and_row_order_without_rereading(self):
        # Unique attributes expose accidental copying of initial values or sorting
        # coordinates independently from attributes in independently rebuilt frames.
        for frame in range(1, 6):
            values = self.data.get(1, f"gs/frame_{frame:06d}.npz")
            n = len(values["xyz"])
            values["log_scales"][:, 0] = -frame - np.arange(n) / 10
            values["opacity_logits"][:, 0] = frame + np.arange(n) / 10
            values["sh_dc"][:, 1] = values["xyz"][:, 2] + frame
            permutation = np.arange(n)[::-1]
            for key in dataset_module.GS_WIDTHS:
                values[key] = values[key][permutation].copy()
        with self.data.mounted(), patch.object(dataset_module, "read_gs", wraps=dataset_module.read_gs) as reader:
            sample = Touch4DDataset(self.data.scene, "train", self.data.root)[0]
        self.assertEqual([call.args[1] for call in reader.call_args_list], list(range(6)))
        for k, (xyz, attr) in enumerate(zip(sample["targets"]["target_xyz"], sample["targets"]["target_attr"]), 1):
            source = self.data.get(1, f"gs/frame_{k:06d}.npz")
            torch.testing.assert_close(xyz, torch.from_numpy(source["xyz"]))
            expected = np.concatenate([source[key] for key in dataset_module.GS_ATTR_FIELDS], axis=-1)
            torch.testing.assert_close(attr, torch.from_numpy(expected))
            attr.fill_(99)
            self.assertFalse(np.any(source["log_scales"] == 99))

    def test_all_scenes_and_metadata_reversed_slots(self):
        for scene in SPLITS:
            data = MemoryReconstruction(scene)
            data.add_episode(5, 12, hands=("left", "right"))
            data.get(5, "metadata.json")["script_version"] = "1.4.0"
            with data.mounted():
                ds = Touch4DDataset(scene, "train", data.root)
                x = ds[ds.windows.index((5, 0))]["inputs"]
                self.assertEqual(x["action_state"][0, :, 0].tolist(), [0, 100])
                self.assertEqual(x["controller_points"][0, :, 0, 0].tolist(), [10, 20])

    def test_missing_hand_and_zero_response_remain_valid_in_2d(self):
        with self.data.mounted():
            ds = Touch4DDataset(self.data.scene, "train", self.data.root, stage="single")
            self.assertEqual(ds.episode_ids, (1, 2, 4))
            sample = ds[0]
            x = sample["inputs"]
            self.assertFalse(x["action_valid"][:, 0].any())
            self.assertFalse(x["controller_mask"][:, 0].any())
            self.assertTrue(x["valid_mask"][:, :2, :, :31].all())
            self.assertFalse(x["tactile_spatial_mask"][:, :2].any())
            self.assertTrue(x["tactile_spatial_mask"][:, 2:, :, :31].all())
            self.assertTrue(torch.all(x["tactile"] == 0))
            check_batch(collate_fn([sample]))

    def test_invalid_pose_is_masked_without_dropping_window(self):
        c = self.data.get(1, "controller.npz")
        c["T_worlds"][0, 0, 0, 0] = 2
        c["actions"] = encode_actions(c["T_worlds"], c["openings"])
        c["controller_valid_mask"][0] = False
        c["controller_points"][0] = 0
        # A different frame has no usable controller points, but its pose is valid.
        c["controller_valid_mask"][1] = False
        c["controller_points"][1] = 0
        with self.data.mounted():
            ds = Touch4DDataset(self.data.scene, "train", self.data.root)
            x = ds[0]["inputs"]
            self.assertEqual(len(ds), 42)
            self.assertFalse(x["action_valid"][0].any())
            self.assertFalse(x["tactile_spatial_mask"][0].any())
            self.assertTrue(x["valid_mask"][0, :, :, :31].all())
            self.assertTrue(x["action_valid"][1, 1])

    def test_saved_no_past_stale_and_duplicate_timestamps(self):
        time = self.data.get(1, "timestamps.npz")
        ts = time["observation_timestamp_us"]
        # No past sample on frame 0, then 99,999 us old samples.
        time["raw_tactile_timestamp_us_0"] = ts + 1
        time["tactile_raw_index"][:, 0] = np.arange(len(ts)) - 1
        time["tactile_age_us"][:, 0] = [-1] + [99_999] * (len(ts) - 1)
        self.data.get(1, "tactile.npz")["valid_mask"][:, 0] = False
        time["raw_tactile_timestamp_us_1"] = np.repeat(ts, 2)
        time["tactile_raw_index"][:, 1] = 2 * np.arange(len(ts)) + 1
        with self.data.mounted():
            sample = Touch4DDataset(self.data.scene, "train", self.data.root)[0]
            self.assertEqual(sample["inputs"]["tactile_age_us"][:, 0].tolist(), [-1] + [99_999] * 4)
            self.assertFalse(sample["inputs"]["valid_mask"][:, 0].any())

    def test_75_ms_boundary_is_inclusive_and_stale_mask_is_enforced(self):
        time = self.data.get(1, "timestamps.npz")
        ts = time["observation_timestamp_us"]
        for stream, age in ((2, 75000), (3, 75001)):
            time[f"raw_tactile_timestamp_us_{stream}"] = ts - age
            time["tactile_age_us"][:, stream] = age
        self.data.get(1, "tactile.npz")["valid_mask"][:, 3] = False
        with self.data.mounted():
            sample = Touch4DDataset(self.data.scene, "train", self.data.root)[0]
            self.assertTrue(sample["inputs"]["valid_mask"][:, 2, :, :31].all())
            self.assertFalse(sample["inputs"]["valid_mask"][:, 3].any())
            self.assertEqual(sample["inputs"]["tactile_age_us"][0, 3].item(), 75001)
            self.data.get(1, "tactile.npz")["valid_mask"][0, 3, 0, 0] = True
            with self.assertRaisesRegex(DataFormatError, "time mask"):
                Touch4DDataset(self.data.scene, "train", self.data.root)[0]

    def test_corrupt_or_missing_files_raise_with_path(self):
        cases = [
            ("missing", "frame_000002.npz", lambda d: d.files.pop(d.path(1, "gs/frame_000002.npz"))),
            ("timestamp", "frame_000002.npz", lambda d: d.get(1, "gs/frame_000002.npz").update(timestamp_us=np.asarray(0, dtype=np.int64))),
            ("nonfinite", "sh_rest", lambda d: d.get(1, "gs/frame_000002.npz")["sh_rest"].fill(np.nan)),
            ("quaternion", "quaternions", lambda d: d.get(1, "gs/frame_000002.npz")["quaternions"].fill(0)),
            ("legacy", "upgrade", lambda d: d.get(1, "controller.npz").pop("actions")),
            ("action mismatch", "actions disagree", lambda d: d.get(1, "controller.npz")["actions"].fill(0)),
            ("negative tactile", "invalid tactile", lambda d: d.get(1, "tactile.npz")["tactile"].fill(-1)),
            ("raw index", "timeline", lambda d: d.get(1, "timestamps.npz")["tactile_raw_index"].fill(0)),
            ("observation order", "strictly increase", lambda d: d.get(1, "timestamps.npz")["observation_timestamp_us"].fill(0)),
            ("bad NPZ", "controller.npz", lambda d: d.files.update({d.path(1, "controller.npz"): b"not an npz"})),
            ("broken ZIP", "controller.npz", lambda d: d.files.update({d.path(1, "controller.npz"): b"PK\x03\x04broken"})),
        ]
        for name, message, corrupt in cases:
            with self.subTest(name=name):
                data = MemoryReconstruction()
                corrupt(data)
                with data.mounted(), self.assertRaisesRegex(DataFormatError, message):
                    Touch4DDataset(data.scene, "train", data.root)[0]

    def test_cache_is_bounded_and_samples_do_not_mutate_source(self):
        with self.data.mounted():
            ds = Touch4DDataset(self.data.scene, "train", self.data.root, cache_episodes=1)
            sample = ds[0]
            sample["inputs"]["tactile"].fill_(123)
            sample["inputs"]["action_state"].fill_(123)
            sample["meta"]["observation_timestamp_us"].fill_(0)
            again = ds[0]
            self.assertTrue(torch.all(again["inputs"]["tactile"] == 0))
            self.assertEqual(again["inputs"]["action_state"][0, 1, 0].item(), 0)
            self.assertGreater(again["meta"]["observation_timestamp_us"][0].item(), 0)
            ds[ds.windows.index((2, 0))]
            self.assertEqual(list(ds._cache), [2])
            restored = pickle.loads(pickle.dumps(ds))
            self.assertFalse(restored._cache)
            self.assertEqual(restored.windows, ds.windows)


class CollateAndLoaderTests(unittest.TestCase):
    def test_h15_dynamic_padding_mixed_lengths_rejected_and_loader_passes_horizon(self):
        source = MemoryReconstruction(frames=22)
        with source.mounted():
            ds = Touch4DDataset(source.scene, "test", source.root, horizon=15)
            samples = [ds[0], ds[1]]
            batch = collate_fn(samples)
            self.assertEqual(batch_horizon(batch), 15)
            self.assertEqual(batch["meta"]["frame_ids"].shape, (2, 16))
            for h in range(15):
                lengths = [len(s["targets"]["target_xyz"][h]) for s in samples]
                self.assertEqual(batch["targets"]["target_xyz"][h].shape[1], max(lengths))
                self.assertEqual(batch["targets"]["target_lengths"][:, h].tolist(), lengths)
                self.assertTrue(torch.all(batch["targets"]["target_attr"][h][~batch["targets"]["target_mask"][h]] == 0))
            shorter = Touch4DDataset(source.scene, "test", source.root, horizon=5)[0]
            with self.assertRaises(ValueError):
                collate_fn([samples[0], shorter])
            broken = copy.deepcopy(samples[0])
            broken["inputs"]["tactile"] = broken["inputs"]["tactile"][:5]
            with self.assertRaises(ValueError):
                collate_fn([broken])
            loader = make_dataloader(source.scene, "test", root=source.root, horizon=15, batch_size=2)
            self.assertEqual(loader.dataset.windows, ds.windows)
            check_batch(next(iter(loader)))

    def test_variable_point_padding_is_by_lengths_not_coordinates(self):
        data = MemoryReconstruction()
        with data.mounted():
            ds = Touch4DDataset(data.scene, "test", data.root)
            a, b = ds[0], ds[1]
            batch = collate_fn([a, b])
            x, target = batch["inputs"], batch["targets"]
            self.assertEqual(x["gs_lengths"].tolist(), [1, 2])
            self.assertEqual(x["gs_mask"].tolist(), [[True, False], [True, True]])
            self.assertTrue(torch.all(x["gs_xyz"][0, 0] == 0))  # real origin is still valid
            self.assertTrue(torch.all(x["gs_attr"][0, 1] == 0))
            self.assertEqual(target["target_lengths"].tolist(), [[2, 3, 4, 1, 2], [3, 4, 1, 2, 3]])
            self.assertEqual([t.shape[1] for t in target["target_xyz"]], [3, 4, 4, 2, 3])
            for i, sample in enumerate((a, b)):
                torch.testing.assert_close(x["gs_xyz"][i, :len(sample["inputs"]["gs_xyz"])], sample["inputs"]["gs_xyz"])
                for k, values in enumerate(sample["targets"]["target_xyz"]):
                    torch.testing.assert_close(target["target_xyz"][k][i, :len(values)], values)
                    torch.testing.assert_close(target["target_attr"][k][i, :len(values)],
                                               sample["targets"]["target_attr"][k])
                    self.assertTrue(torch.all(target["target_attr"][k][i, len(values):] == 0))
            check_batch(batch)
            self.assertEqual(x["action_state"].shape, (2, 5, 2, 13))
            self.assertEqual(batch["meta"]["frame_ids"].dtype, torch.int64)
            b["meta"]["scene"] = "051-cube"
            with self.assertRaisesRegex(ValueError, "one scene"):
                collate_fn([a, b])
            with self.assertRaises(ValueError):
                collate_fn([])

    def test_legacy_targets_supported_but_mixed_or_misaligned_attributes_fail(self):
        data = MemoryReconstruction()
        with data.mounted():
            ds = Touch4DDataset(data.scene, "test", data.root)
            original = [ds[0], ds[1]]
        legacy = copy.deepcopy(original)
        for sample in legacy:
            sample["targets"].pop("target_attr")
        batch = collate_fn(legacy)
        self.assertNotIn("target_attr", batch["targets"])
        check_batch(batch)
        with self.assertRaisesRegex(ValueError, "mix samples"):
            collate_fn([original[0], legacy[1]])
        mutations = [
            lambda t: t["target_attr"].pop(),
            lambda t: t["target_attr"].__setitem__(0, t["target_attr"][0][:-1]),
            lambda t: t["target_attr"].__setitem__(0, t["target_attr"][0].double()),
            lambda t: t["target_attr"][0].fill_(float("nan")),
            lambda t: t["target_attr"][0][:, 3:7].zero_(),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                samples = copy.deepcopy(original)
                mutation(samples[0]["targets"])
                with self.assertRaisesRegex(ValueError, "target_attr"):
                    collate_fn(samples)

    def test_seeded_train_and_fixed_evaluation_keep_partial_batch(self):
        data = MemoryReconstruction()
        with data.mounted():
            train_a = make_dataloader(data.scene, "train", root=data.root, seed=42, batch_size=5)
            train_b = make_dataloader(data.scene, "train", root=data.root, seed=42, batch_size=5)
            order_a, order_b = list(train_a.sampler), list(train_b.sampler)
            self.assertEqual(order_a, order_b)
            self.assertEqual(sorted(order_a), list(range(42)))
            self.assertNotEqual(order_a, sorted(order_a))
            loader = make_dataloader(data.scene, "val", root=data.root, seed=99, batch_size=3)
            batches = list(loader)
            self.assertEqual([len(b["meta"]["scene"]) for b in batches], [3, 1])
            self.assertEqual(torch.cat([b["meta"]["episode_id"] for b in batches]).tolist(), [3, 3, 6, 6])
            self.assertEqual(torch.cat([b["meta"]["frame_ids"][:, 0] for b in batches]).tolist(), [0, 5, 0, 5])
            for batch in batches:
                check_batch(batch)

    def test_empty_windows_and_curriculum_mismatch_are_explicit(self):
        data = MemoryReconstruction(frames=5)
        with data.mounted():
            self.assertEqual(len(Touch4DDataset(data.scene, "train", data.root)), 0)
            with self.assertRaisesRegex(ValueError, "no complete"):
                make_dataloader(data.scene, "train", root=data.root)
        data.add_episode(1, 12, hands=("right", "left"))
        with data.mounted(), self.assertRaisesRegex(DataFormatError, "curriculum"):
            Touch4DDataset(data.scene, "train", data.root, stage="single")

    def test_spawned_worker_can_load_and_collate(self):
        root = Path("__memory_only_touch4d__").resolve()
        with patch("scripts.dataloader.loader.Touch4DDataset", MemoryBackedDataset):
            loader = make_dataloader("001-rope", "val", root=root, num_workers=1, batch_size=3)
            loader.multiprocessing_context = "spawn"
            batches = list(loader)
        self.assertEqual([len(b["meta"]["scene"]) for b in batches], [3, 1])
        for batch in batches:
            check_batch(batch)

    def test_read_only_check_cli(self):
        data = MemoryReconstruction()
        with data.mounted(), patch("sys.stdout", new_callable=StringIO) as output:
            check_main(["--scene", data.scene, "--root", str(data.root), "--split", "val",
                        "--max-batches", "0", "--batch-size", "3"])
            result = json.loads(output.getvalue())
        self.assertEqual(result["checked_samples"], 4)
        self.assertTrue(result["all_windows_checked"])
        self.assertEqual(result["status"], "passed")

    def test_h15_read_only_check_cli_preserves_val_stride(self):
        data = MemoryReconstruction(frames=22)
        with data.mounted(), patch("sys.stdout", new_callable=StringIO) as output:
            check_main(["--scene", data.scene, "--root", str(data.root), "--split", "val",
                        "--max-batches", "0", "--batch-size", "3", "--horizon", "15"])
            result = json.loads(output.getvalue())
        self.assertEqual(result["horizon"], 15)
        self.assertEqual(result["checked_samples"], 4)
        self.assertEqual(result["first_batch_shapes"]["action_state"], [3, 15, 2, 13])
        self.assertEqual(len(result["first_batch_shapes"]["target_xyz"]), 15)
        self.assertTrue(result["all_windows_checked"])


@unittest.skipIf(h5py is None, "Optional depth tests require h5py")
class DepthDatasetTests(unittest.TestCase):
    def test_h10_h15_depth_covers_each_future_frame_with_matching_calibration(self):
        for horizon in (10, 15):
            for scene in SPLITS:
                with self.subTest(scene=scene, horizon=horizon):
                    data = MemoryDepthReconstruction(scene, frames=horizon+7)
                    chosen = [data.cameras[2], data.cameras[0]]
                    with data.mounted():
                        ds = Touch4DDataset(scene, "train", data.root, with_depth=True,
                                            depth_cameras=chosen, horizon=horizon)
                        sample = ds[ds.windows.index((5, 2))]
                        target = sample["targets"]
                        frames = torch.arange(3, horizon+3, dtype=torch.float32)
                        self.assertEqual(target["target_depth"].shape, (horizon, 2, 3, 4))
                        torch.testing.assert_close(target["target_depth"][:, 0, 1, 1], (1200+20*frames)/1000)
                        torch.testing.assert_close(target["target_depth"][:, 1, 1, 1], (1000+20*frames)/1000)
                        self.assertFalse(torch.equal(target["target_depth"][4], target["target_depth"][-1]))
                        self.assertEqual(sample["meta"]["frame_ids"].tolist(), list(range(2, horizon+3)))
                        self.assertEqual(sample["meta"]["depth_camera_ids"], chosen)
                        torch.testing.assert_close(target["camera_intrinsics"][:, 0, 0], torch.tensor([702., 700.]))
                        self.assertFalse(target["target_depth_valid"][:, :, 0, 0].any())
                        check_batch(collate_fn([sample, ds[-1]]))

    def test_old_mode_does_not_open_or_require_depth(self):
        data = MemoryReconstruction()
        with data.mounted(), patch.object(h5py, "File", side_effect=AssertionError("Unexpected H5 read")):
            ds = Touch4DDataset(data.scene, "train", data.root)
            sample = ds[0]
            self.assertEqual(set(sample["targets"]), {"target_xyz", "target_attr"})
            self.assertNotIn("depth_camera_ids", sample["meta"])
            validate_depth_batch(collate_fn([sample]))

    def test_all_three_scenes_future_offsets_units_masks_and_camera_order(self):
        for scene in SPLITS:
            with self.subTest(scene=scene):
                data = MemoryDepthReconstruction(scene)
                chosen = [data.cameras[2], data.cameras[0]]
                with data.mounted():
                    ds = Touch4DDataset(scene, "train", data.root, with_depth=True, depth_cameras=chosen)
                    sample = ds[ds.windows.index((5, 2))]
                    target = sample["targets"]
                    self.assertEqual(sample["meta"]["frame_ids"].tolist(), list(range(2, 8)))
                    self.assertEqual(sample["meta"]["depth_camera_ids"], chosen)
                    self.assertEqual(target["target_depth"].shape, (5, 2, 3, 4))
                    self.assertEqual(target["target_depth"].dtype, torch.float32)
                    self.assertEqual(target["target_depth_valid"].dtype, torch.bool)
                    torch.testing.assert_close(target["target_depth"][:, 0, 1, 1],
                                               torch.tensor([1.26, 1.28, 1.30, 1.32, 1.34]))
                    torch.testing.assert_close(target["target_depth"][:, 1, 1, 1],
                                               torch.tensor([1.06, 1.08, 1.10, 1.12, 1.14]))
                    self.assertTrue(torch.equal(target["target_depth_valid"], target["target_depth"] > 0))
                    self.assertFalse(target["target_depth_valid"][:, :, 0, 0].any())
                    torch.testing.assert_close(target["camera_intrinsics"][:, 0, 0], torch.tensor([702., 700.]))
                    torch.testing.assert_close(target["camera_to_world"][:, 0, 3], torch.tensor([0.2, 0.0]))
                    self.assertEqual(ds.depth_root, data.depth_root)
                    check_batch(collate_fn([sample]))

    def test_default_camera_selection_uses_common_ordered_subset_and_lazy_reads(self):
        data = MemoryDepthReconstruction()
        data.add_depth(1, 12, cameras=data.cameras[1:])
        with data.mounted(), patch.object(h5py, "File", wraps=h5py.File) as reader:
            ds = Touch4DDataset(data.scene, "train", data.root, with_depth=True)
            self.assertEqual(tuple(ds.depth_cameras), data.cameras[1:])
            self.assertEqual(reader.call_count, 0)
            sample = ds[0]
            self.assertEqual(reader.call_count, 2)
            self.assertEqual(sample["meta"]["depth_camera_ids"], list(data.cameras[1:]))

    def test_depth_samples_are_independent_and_dataset_pickle_has_no_open_h5(self):
        data = MemoryDepthReconstruction()
        with data.mounted():
            ds = Touch4DDataset(data.scene, "train", data.root, with_depth=True)
            sample = ds[0]
            sample["targets"]["target_depth"].zero_()
            sample["targets"]["camera_to_world"].zero_()
            restored = pickle.loads(pickle.dumps(ds))
            again = restored[0]
            self.assertTrue(again["targets"]["target_depth_valid"].any())
            self.assertGreater(again["targets"]["target_depth"][0, 0, 1, 1].item(), 0)
            self.assertEqual(again["targets"]["camera_to_world"][0, 3, 3].item(), 1)

    def test_invalid_camera_options_and_missing_extension_fail_explicitly(self):
        data = MemoryDepthReconstruction()
        options = [
            {"depth_root": data.depth_root},
            {"with_depth": True, "depth_cameras": []},
            {"with_depth": True, "depth_cameras": [data.cameras[0]] * 2},
            {"with_depth": True, "depth_cameras": ["missing_camera"]},
        ]
        with data.mounted():
            for kwargs in options:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    Touch4DDataset(data.scene, "train", data.root, **kwargs)
        del data.files[data.depth_path(1, "metadata.json")]
        with data.mounted(), self.assertRaisesRegex(DataFormatError, "depth|metadata"):
            Touch4DDataset(data.scene, "train", data.root, with_depth=True)

    def test_depth_schema_binding_and_calibration_failures_are_detected(self):
        def parameters(data):
            return data.files[data.depth_path(1, "camera_parameters.npz")]

        def rehash_parameters(data):
            name = "camera_parameters.npz"
            data.files[data.depth_path(1, "metadata.json")]["output_files"][name] = data.digest(data.depth_path(1, name))

        def corrupt_timeline(data):
            parameters(data)["observation_timestamp_us"][2] += 1
            rehash_parameters(data)

        def corrupt_rotation(data):
            parameters(data)["camera_to_world"][0, 0, 0] = -1
            rehash_parameters(data)

        def corrupt_core_timeline(data):
            times = data.get(1, "timestamps.npz")
            for key in ("observation_timestamp_us", *(f"raw_tactile_timestamp_us_{i}" for i in range(4))):
                times[key] += 1

        cases = [
            ("stale core", "binding", lambda d: d.get(1, "metadata.json").update(review_note="changed")),
            ("changed core times", "timestamp SHA", corrupt_core_timeline),
            ("changed parameters", "SHA", lambda d: parameters(d)["intrinsics"].fill(0)),
            ("different timeline", "timeline", corrupt_timeline),
            ("reflection", "rigid", corrupt_rotation),
            ("unsupported units", "convention", lambda d: d.files[d.depth_path(1, "metadata.json")].update(depth_unit="m")),
            ("missing camera", "Missing|manifest", lambda d: d.files.pop(d.depth_path(1, f"{d.cameras[0]}/rendered_depth.h5"))),
        ]
        for name, message, corrupt in cases:
            with self.subTest(name=name):
                data = MemoryDepthReconstruction()
                corrupt(data)
                with data.mounted(), self.assertRaisesRegex(DataFormatError, message):
                    Touch4DDataset(data.scene, "train", data.root, with_depth=True)

    def test_invalid_h5_dtype_shape_and_payload_fail_on_read(self):
        cases = [
            depth_h5_bytes(np.ones((12, 3, 4), dtype=np.float32)),
            depth_h5_bytes(np.ones((11, 3, 4), dtype=np.uint16)),
            b"not an HDF5 file",
        ]
        for payload in cases:
            with self.subTest(payload_size=len(payload)):
                data = MemoryDepthReconstruction()
                data.files[data.depth_path(1, f"{data.cameras[0]}/rendered_depth.h5")] = payload
                with data.mounted():
                    ds = Touch4DDataset(data.scene, "train", data.root, with_depth=True)
                    with self.assertRaisesRegex(DataFormatError, "depth|uint16"):
                        ds[0]

    def test_collate_preserves_depth_and_rejects_mixed_or_partial_fields(self):
        data = MemoryDepthReconstruction()
        with data.mounted():
            ds = Touch4DDataset(data.scene, "test", data.root, with_depth=True)
            a, b = ds[0], ds[1]
            batch = collate_fn([a, b])
            self.assertEqual(batch["targets"]["target_depth"].shape, (2, 5, 2, 3, 4))
            self.assertEqual(batch["targets"]["camera_to_world"].shape, (2, 2, 4, 4))
            torch.testing.assert_close(batch["targets"]["target_depth"][1], b["targets"]["target_depth"])
            self.assertEqual(batch["meta"]["depth_camera_ids"], [list(data.cameras[:2])] * 2)
            check_batch(batch)
            old = Touch4DDataset(data.scene, "test", data.root)[0]
            with self.assertRaisesRegex(ValueError, "mix"):
                collate_fn([a, old])
            broken = copy.deepcopy(b)
            del broken["targets"]["target_depth_valid"]
            with self.assertRaisesRegex(ValueError, "together|requires"):
                collate_fn([a, broken])
            broken = copy.deepcopy(b)
            broken["meta"]["depth_camera_ids"].reverse()
            with self.assertRaisesRegex(ValueError, "camera"):
                collate_fn([a, broken])

    def test_batch_check_rejects_nonfinite_depth_bad_mask_and_nonrigid_calibration(self):
        data = MemoryDepthReconstruction()
        with data.mounted():
            sample = Touch4DDataset(data.scene, "train", data.root, with_depth=True)[0]
            batch = collate_fn([sample])
        for field, index, value in (
            ("target_depth", (0, 0, 0, 1, 1), float("nan")),
            ("target_depth", (0, 0, 0, 1, 1), -1),
            ("target_depth_valid", (0, 0, 0, 1, 1), False),
            ("camera_to_world", (0, 0, 0, 0), 2),
            ("camera_intrinsics", (0, 0, 0, 0), 0),
        ):
            with self.subTest(field=field, value=value):
                broken = copy.deepcopy(batch)
                broken["targets"][field][index] = value
                with self.assertRaises(ValueError):
                    validate_depth_batch(broken)

    def test_loader_and_read_only_cli_forward_depth_options(self):
        data = MemoryDepthReconstruction()
        with data.mounted():
            loader = make_dataloader(data.scene, "val", root=data.root, with_depth=True,
                                     depth_root=data.depth_root, depth_cameras=[data.cameras[2]], batch_size=3)
            batches = list(loader)
            self.assertEqual([len(b["meta"]["scene"]) for b in batches], [3, 1])
            self.assertEqual(batches[0]["targets"]["target_depth"].shape, (3, 5, 1, 3, 4))
            self.assertEqual(batches[0]["meta"]["depth_camera_ids"], [[data.cameras[2]]] * 3)
            with patch("sys.stdout", new_callable=StringIO) as output:
                check_main(["--scene", data.scene, "--root", str(data.root), "--split", "val",
                            "--max-batches", "1", "--with-depth", "--depth-root", str(data.depth_root),
                            "--depth-cameras", data.cameras[2]])
                result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "passed")


class DepthCommandTests(unittest.TestCase):
    def test_rebuild_depth_only_dispatch_never_enters_core_pipeline(self):
        for scene in SPLITS:
            with self.subTest(scene=scene), ExitStack() as stack:
                dispatch = stack.enter_context(patch.object(rebuild_module, "run_depth_only", return_value=0))
                stack.enter_context(patch.object(rebuild_module, "os", SimpleNamespace(name="posix")))
                core = [stack.enter_context(patch.object(rebuild_module, name,
                        side_effect=AssertionError(f"Core path called: {name}")))
                        for name in ("preflight", "rebuild_episode", "_validators", "discover_episodes")]
                stack.enter_context(patch.dict("sys.modules", {"plyfile": None}))
                self.assertEqual(rebuild_module.main(["--scene", scene, "--depth-only", "--episodes", "0,5", "--overwrite"]), 0)
                options = dispatch.call_args.args[0]
                self.assertEqual(options.scene, scene)
                self.assertEqual(options.episodes, [0, 5])
                self.assertTrue(options.overwrite)
                for mocked in core:
                    mocked.assert_not_called()

    def test_validate_depth_only_dispatch_never_enters_core_validation(self):
        with patch.object(validate_module, "validate_depth_main", return_value=0) as dispatch, \
             patch.object(validate_module, "validate_episode", side_effect=AssertionError("Core validation")) as core, \
             patch.object(validate_module, "discover_episodes", side_effect=AssertionError("Core discovery")):
            self.assertEqual(validate_module.main(["--scene", "008-pink-cloth", "--depth-only", "--structure-only"]), 0)
            self.assertTrue(dispatch.call_args.args[0].structure_only)
            core.assert_not_called()


@unittest.skipIf(h5py is None, "Optional depth tests require h5py")
class DepthValidationTests(unittest.TestCase):
    @staticmethod
    def archive_bytes(entries):
        stream = BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for name, payload, kind in entries:
                info = tarfile.TarInfo(name)
                info.type = kind
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    info.linkname = "elsewhere/rendered_depth.h5"
                if kind == tarfile.REGTYPE:
                    info.size = len(payload)
                    archive.addfile(info, BytesIO(payload))
                else:
                    archive.addfile(info)
        return stream.getvalue()

    @contextmanager
    def memory_tar_reader(self, data, relative, payload):
        """Let the real tar parser inspect archive bytes without opening disk files."""
        data.files[data.source_root / relative] = payload
        original_open = tarfile.open

        def open_tar(name=None, mode="r", **kwargs):
            if name is not None:
                path = Path(name)
                if path not in data.files:
                    raise AssertionError(f"Unexpected archive read: {path}")
                return original_open(fileobj=BytesIO(data.files[path]), mode=mode, **kwargs)
            return original_open(mode=mode, **kwargs)

        with data.mounted(), patch.object(tarfile, "open", side_effect=open_tar):
            with rebuild_module.DepthSourceReader(data.source_root) as reader:
                yield reader

    def recorded_camera_archive(self, data, episode=0):
        """Record tar provenance while keeping identical loose copies alongside it."""
        core = data.get(episode, "metadata.json")
        extension = data.files[data.depth_path(episode, "metadata.json")]
        relative = f"{core['source_episode_directory']}/cameras.tar"
        members = {}
        for camera in core["observation_cameras"]:
            for key, filename in ((f"camera_timestamps_{camera}", "aligned_timestamps.txt"),
                                  (f"depth_{camera}", "rendered_depth.h5")):
                member = f"{camera}/{filename}"
                old = extension["source_files"][key]
                members[member] = data.files[data.source_root / old["path"]]
                ref = {**old, "path": relative, "member": member}
                extension["source_files"][key] = ref
                if key.startswith("camera_timestamps_"):
                    core["source_files"][key] = copy.deepcopy(ref)
        core["source_files"]["observation"] = copy.deepcopy(
            core["source_files"][f"camera_timestamps_{core['observation_camera']}"])
        extension["core_binding"]["metadata_sha256"] = data.digest(data.path(episode, "metadata.json"))
        return relative, members

    def test_first_depth_layout_keeps_core_tar_timestamps_with_loose_copies(self):
        data = MemoryDepthReconstruction()
        relative, members = self.recorded_camera_archive(data)
        archive = self.archive_bytes([(name, payload, tarfile.REGTYPE) for name, payload in members.items()])
        core = data.get(0, "metadata.json")
        obs = data.get(0, "timestamps.npz")["observation_timestamp_us"]
        with self.memory_tar_reader(data, relative, archive) as reader:
            plan = rebuild_module.depth_source_layout(reader, core, obs)
            for camera in data.cameras:
                key = f"camera_timestamps_{camera}"
                self.assertEqual(reader.refs[key], core["source_files"][key])
                self.assertEqual(reader.refs[key]["member"], f"{camera}/aligned_timestamps.txt")
                # H5 has no core reference: new/overwrite discovery may use loose data.
                self.assertEqual(plan["depth_refs"][camera],
                    (f"{core['source_episode_directory']}/{camera}/rendered_depth.h5", None))

    def test_recorded_tar_depth_validates_and_skips_despite_added_loose_copies(self):
        data = MemoryDepthReconstruction()
        relative, members = self.recorded_camera_archive(data)
        archive = self.archive_bytes([(name, payload, tarfile.REGTYPE) for name, payload in members.items()])
        before = {path: data.digest(path) for path in data.files
                  if data.root in path.parents or data.depth_root in path.parents}
        with self.memory_tar_reader(data, relative, archive), patch("sys.stdout", new_callable=StringIO):
            result = validate_module.validate_depth_episode(data.depth_path(0, "metadata.json").parent,
                core_directory=data.path(0, "metadata.json").parent, source_root=data.source_root,
                verify_sources=True)
            self.assertEqual(result["status"], "PASS")
            # Exercise the real skip validator/path guards, unlike publication_sandbox.
            skipped = rebuild_module.rebuild_depth_episode(data.source_root, data.root, data.depth_root,
                                                           data.scene, 0)
            self.assertTrue(skipped["skipped"])
            self.assertEqual(skipped["source_verification"], "full")
        self.assertEqual({path: data.digest(path) for path in before}, before)

    def test_changed_recorded_tar_bytes_fail_even_with_correct_loose_copies(self):
        for filename in ("aligned_timestamps.txt", "rendered_depth.h5"):
            with self.subTest(filename=filename):
                data = MemoryDepthReconstruction()
                relative, members = self.recorded_camera_archive(data)
                member = f"{data.cameras[0]}/{filename}"
                members[member] += b"\n"  # Timestamp values stay equal; provenance bytes must still fail.
                archive = self.archive_bytes([(name, payload, tarfile.REGTYPE) for name, payload in members.items()])
                with self.memory_tar_reader(data, relative, archive):
                    with self.assertRaisesRegex(ValueError, "provenance|hashes|SHA-256"):
                        validate_module.validate_depth_episode(data.depth_path(0, "metadata.json").parent,
                            core_directory=data.path(0, "metadata.json").parent, source_root=data.source_root,
                            verify_sources=True)

    def test_missing_recorded_tar_source_or_member_never_falls_back_to_loose(self):
        for missing in ("archive", "timestamp_member", "depth_member"):
            with self.subTest(missing=missing):
                data = MemoryDepthReconstruction()
                relative, members = self.recorded_camera_archive(data)
                if missing != "archive":
                    filename = "aligned_timestamps.txt" if missing == "timestamp_member" else "rendered_depth.h5"
                    del members[f"{data.cameras[0]}/{filename}"]
                archive = self.archive_bytes([(name, payload, tarfile.REGTYPE) for name, payload in members.items()])
                with self.memory_tar_reader(data, relative, archive):
                    if missing == "archive":
                        del data.files[data.source_root / relative]
                    with self.assertRaisesRegex(ValueError, "Missing depth"):
                        validate_module.validate_depth_episode(data.depth_path(0, "metadata.json").parent,
                            core_directory=data.path(0, "metadata.json").parent, source_root=data.source_root,
                            verify_sources=True)

    def test_camera_tar_stream_preserves_selected_member_bytes_and_hash(self):
        data = MemoryDepthReconstruction()
        camera = data.cameras[0]
        member = f"{camera}/rendered_depth.h5"
        payload = data.files[data.depth_path(0, member)]
        relative = f"processed/{data.scene}/episode_0/cameras.tar"
        archive = self.archive_bytes([
            (camera, b"", tarfile.DIRTYPE),
            (member, payload, tarfile.REGTYPE),
            (f"{data.cameras[1]}/rendered_depth.h5", b"unselected bytes", tarfile.REGTYPE),
        ])
        destination = data.depth_path(0, "copied_depth.h5")
        written = {}

        class CapturedOutput(BytesIO):
            def close(self):
                if not self.closed:
                    written[destination] = self.getvalue()
                super().close()

        with self.memory_tar_reader(data, relative, archive) as reader:
            original_open = Path.open

            def open_file(path, mode="r", *args, **kwargs):
                if path == destination and mode == "xb":
                    return CapturedOutput()
                return original_open(path, mode, *args, **kwargs)

            with patch.object(Path, "open", open_file):
                ref = reader.stream(f"depth_{camera}", relative, member, destination=destination)
                again = reader.stream(f"depth_{camera}", relative, member)
            self.assertEqual(written, {destination: payload})
            self.assertEqual(ref, {"path": relative, "member": member, "bytes": len(payload),
                                   "sha256": hashlib.sha256(payload).hexdigest()})
            self.assertEqual(again, ref)
            self.assertEqual(reader.refs, {f"depth_{camera}": ref})
            self.assertEqual(len(reader.archives), 1)

    def test_camera_tar_rejects_traversal_duplicates_and_links_before_copy(self):
        member = "brics-odroid-001_cam0/rendered_depth.h5"
        cases = {
            "traversal": [("../escape.h5", b"bad", tarfile.REGTYPE)],
            "duplicate": [(member, b"first", tarfile.REGTYPE), (member, b"second", tarfile.REGTYPE)],
            "symlink": [(member, b"", tarfile.SYMTYPE)],
            "hardlink": [(member, b"", tarfile.LNKTYPE)],
        }
        for label, entries in cases.items():
            with self.subTest(label=label):
                data = MemoryDepthReconstruction()
                relative = f"processed/{data.scene}/episode_0/cameras.tar"
                with self.memory_tar_reader(data, relative, self.archive_bytes(entries)) as reader:
                    with self.assertRaises(ValueError):
                        reader.stream("depth", relative, member,
                                      destination=data.depth_path(0, "must_not_be_written.h5"))
                    self.assertEqual(reader.refs, {})

    def test_full_depth_validation_reads_only_depth_calibration_and_core_binding(self):
        for scene in SPLITS:
            with self.subTest(scene=scene):
                data = MemoryDepthReconstruction(scene)
                before = {path: data.digest(path) for path in data.files if data.root in path.parents}
                with data.mounted(), ExitStack() as stack:
                    for name in ("preflight", "rebuild_episode", "gs_arrays", "tactile_arrays", "load_robot"):
                        stack.enter_context(patch.object(rebuild_module, name, side_effect=AssertionError(name)))
                    stack.enter_context(patch.object(validate_module, "validate_episode", side_effect=AssertionError("core validator")))
                    result = validate_module.validate_depth_episode(data.depth_path(0, "metadata.json").parent,
                        core_directory=data.path(0, "metadata.json").parent, expected_scene=scene,
                        expected_episode=0, source_root=data.source_root, verify_sources=True)
                    self.assertEqual(result["status"], "PASS")
                    self.assertTrue(result["depth_only"])
                    self.assertEqual(result["source_verification"], "full")
                self.assertEqual({path: data.digest(path) for path in before}, before)

    def test_output_and_source_hashes_are_independently_verified(self):
        for corrupt_output in (True, False):
            with self.subTest(corrupt_output=corrupt_output):
                data = MemoryDepthReconstruction()
                relative = f"{data.cameras[0]}/rendered_depth.h5"
                path = (data.depth_path(0, relative) if corrupt_output else
                        data.source_root / f"processed/{data.scene}/episode_0/{relative}")
                data.files[path] += b"changed"
                with data.mounted(), self.assertRaisesRegex(ValueError, "SHA-256|source hashes"):
                    validate_module.validate_depth_episode(data.depth_path(0, "metadata.json").parent,
                        core_directory=data.path(0, "metadata.json").parent,
                        source_root=data.source_root, verify_sources=True)

    def test_structure_only_still_checks_h5_and_exact_copy_provenance(self):
        data = MemoryDepthReconstruction()
        meta = data.files[data.depth_path(0, "metadata.json")]
        meta["source_files"][f"depth_{data.cameras[0]}"]["sha256"] = "f" * 64
        with data.mounted(), self.assertRaisesRegex(ValueError, "exact source H5 bytes"):
            validate_module.validate_depth_episode(data.depth_path(0, "metadata.json").parent,
                core_directory=data.path(0, "metadata.json").parent, verify_sources=False)

    def test_depth_root_cannot_overlap_core_or_source(self):
        data = MemoryDepthReconstruction()
        with data.mounted():
            for protected in (data.root, data.source_root):
                with self.subTest(protected=protected), self.assertRaisesRegex(ValueError, "disjoint"):
                    rebuild_module.depth_roots(data.source_root, data.root, protected / "depth")

    @contextmanager
    def publication_sandbox(self, *, fail_publish=False, fail_cleanup=False):
        """Exercise publication decisions against an in-memory directory state."""
        base = Path("__memory_only_publication__").resolve()
        source, core, depth = (base / name for name in ("source", "core", "depth"))
        target = depth / "001-rope" / "episode_0"
        partial, backup = (target.with_name(f".episode_0.{suffix}") for suffix in ("incomplete", "previous"))
        directories = {target}
        renames, created, writes = [], [], []
        reader = MagicMock()
        reader.__enter__.return_value = reader
        reader.refs = {}
        camera = "brics-odroid-001_cam0"
        metadata = {"scene": "001-rope", "episode": 0, "observation_cameras": [camera],
                    "source_episode_directory": "processed/001-rope/episode_0"}
        plan = {"depth_refs": {camera: ("ignored", None)}, "height": 3, "width": 4,
                "intrinsics": np.eye(3)[None], "camera_to_world": np.eye(4)[None]}
        validator = MagicMock()
        validator.validate_depth_episode.return_value = {"status": "PASS", "depth_only": True}

        def mkdir(path, *args, **kwargs):
            if not path.is_relative_to(partial):
                raise AssertionError(f"Attempted to create outside the depth staging directory: {path}")
            created.append(path)
            directories.add(path)

        def record_write(path, *args, **kwargs):
            if not path.is_relative_to(partial):
                raise AssertionError(f"Attempted to write outside the depth staging directory: {path}")
            writes.append(path)

        def open_output(path, *args, **kwargs):
            record_write(path)
            return StringIO()

        def copy_depth(*args, destination=None, **kwargs):
            record_write(destination)
            return {"sha256": "a" * 64}

        reader.stream.side_effect = copy_depth

        def rename(path, destination):
            renames.append((path, destination))
            if fail_publish and path == partial and destination == target:
                raise OSError("publication rejected")
            directories.remove(path)
            directories.add(destination)
            return destination

        with ExitStack() as stack:
            stack.enter_context(patch.object(rebuild_module, "depth_roots", return_value=(source, core, depth)))
            stack.enter_context(patch.object(rebuild_module, "read_core_depth_binding",
                return_value=(metadata, np.arange(6, dtype=np.int64), {"metadata_sha256": "a" * 64,
                                                                       "timestamps_sha256": "b" * 64})))
            stack.enter_context(patch.object(rebuild_module, "depth_output_paths", return_value=(target, partial, backup)))
            stack.enter_context(patch.object(rebuild_module, "DepthSourceReader", return_value=reader))
            stack.enter_context(patch.object(rebuild_module, "depth_source_layout", return_value=plan))
            stack.enter_context(patch.object(rebuild_module, "check_depth_h5"))
            stack.enter_context(patch.object(rebuild_module, "write_npz", side_effect=record_write))
            stack.enter_context(patch.object(rebuild_module, "hash_file", return_value="a" * 64))
            stack.enter_context(patch.object(rebuild_module, "checked_path", side_effect=lambda p: p))
            stack.enter_context(patch.object(rebuild_module, "_validators", return_value=validator))
            stack.enter_context(patch.object(Path, "exists", lambda path: path in directories))
            stack.enter_context(patch.object(Path, "is_dir", lambda path: path in directories))
            stack.enter_context(patch.object(Path, "stat", lambda path: SimpleNamespace(st_dev=1, st_ino=9)))
            stack.enter_context(patch.object(Path, "mkdir", mkdir))
            stack.enter_context(patch.object(Path, "rename", rename))
            stack.enter_context(patch.object(Path, "open", open_output))
            cleanup = stack.enter_context(patch.object(rebuild_module.shutil, "rmtree",
                side_effect=OSError("cleanup rejected") if fail_cleanup else lambda path: directories.remove(path)))
            output = stack.enter_context(patch("sys.stdout", new_callable=StringIO))
            errors = stack.enter_context(patch("sys.stderr", new_callable=StringIO))
            yield SimpleNamespace(source=source, core=core, depth=depth, target=target, partial=partial,
                backup=backup, directories=directories, renames=renames, created=created, writes=writes,
                cleanup=cleanup, reader=reader, validator=validator, output=output, errors=errors)

    def test_default_skip_validates_without_copying_sources_or_creating_output(self):
        with self.publication_sandbox() as state:
            result = rebuild_module.rebuild_depth_episode(state.source, state.core, state.depth, "001-rope", 0)
            self.assertTrue(result["skipped"])
            self.assertEqual(state.created, [])
            self.assertEqual(state.writes, [])
            self.assertEqual(state.renames, [])
            state.reader.stream.assert_not_called()
            self.assertTrue(state.validator.validate_depth_episode.call_args.kwargs["verify_sources"])

    def test_overwrite_publication_failure_restores_old_directory(self):
        with self.publication_sandbox(fail_publish=True) as state:
            with self.assertRaisesRegex(OSError, "publication rejected"):
                rebuild_module.rebuild_depth_episode(state.source, state.core, state.depth, "001-rope", 0,
                                                     overwrite=True)
            self.assertIn(state.target, state.directories)
            self.assertNotIn(state.backup, state.directories)
            self.assertEqual(state.renames, [(state.target, state.backup), (state.partial, state.target),
                                             (state.backup, state.target)])
            state.cleanup.assert_not_called()
            self.assertTrue(all(path.is_relative_to(state.depth) for path in state.created))

    def test_cleanup_failure_does_not_fail_publication_or_next_default_skip(self):
        with self.publication_sandbox(fail_cleanup=True) as state:
            result = rebuild_module.rebuild_depth_episode(state.source, state.core, state.depth, "001-rope", 0,
                                                         overwrite=True)
            self.assertEqual(result["status"], "PASS")
            self.assertFalse(result["skipped"])
            self.assertIn(state.target, state.directories)
            self.assertIn(state.backup, state.directories)
            self.assertIn("WARNING", state.errors.getvalue())
            again = rebuild_module.rebuild_depth_episode(state.source, state.core, state.depth, "001-rope", 0)
            self.assertTrue(again["skipped"])
            self.assertEqual(state.validator.validate_depth_episode.call_count, 2)


if __name__ == "__main__":
    unittest.main()
