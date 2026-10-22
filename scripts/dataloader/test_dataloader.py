"""Local, disk-free tests: python -B -m unittest scripts.dataloader.test_dataloader -v"""

from contextlib import ExitStack, contextmanager
from io import BytesIO, StringIO
import json
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import dataset as dataset_module
from .check import check_batch, main as check_main
from .collate import collate_fn
from .dataset import DataFormatError, Touch4DDataset
from .geometry import (HAND_ORDER, STREAM_ORDER, action_conditions, hand_slots,
                       pose_validity, tactile_world_coordinates)
from .loader import make_dataloader
from .splits import SPLITS, episode_ids, window_starts


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
        self.memory_source = MemoryReconstruction(scene)
        with self.memory_source.mounted():
            super().__init__(scene, split, root, **kwargs)

    def __getitem__(self, index):
        with self.memory_source.mounted():
            return super().__getitem__(index)


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
            self.assertNotIn("delta_xyz", target)
            self.assertNotIn("sh_rest", x)
            self.assertEqual(len(ds), 42)

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
            check_batch(batch)
            self.assertEqual(x["action_state"].shape, (2, 5, 2, 13))
            self.assertEqual(batch["meta"]["frame_ids"].dtype, torch.int64)
            b["meta"]["scene"] = "051-cube"
            with self.assertRaisesRegex(ValueError, "one scene"):
                collate_fn([a, b])
            with self.assertRaises(ValueError):
                collate_fn([])

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


if __name__ == "__main__":
    unittest.main()
