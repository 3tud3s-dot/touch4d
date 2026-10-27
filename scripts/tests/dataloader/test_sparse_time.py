"""Sparse sampling is target indexing, not window stride or extra horizon."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from scripts.tests.dataloader.test_dataloader import MemoryReconstruction
from scripts.dataloader.dataset import Touch4DDataset, read_depth
from scripts.dataloader.collate import collate_fn
from scripts.dataloader.splits import window_starts
from scripts.dataloader.hand_policy import HAND_POLICY
from scripts.data_process import DataProcessor


class SparseTimeTests(unittest.TestCase):
    def test_exact_frames_conditions_targets_timestamps_and_tail(self):
        source = MemoryReconstruction(frames=53)
        with source.mounted():
            dataset = Touch4DDataset("001-rope", "train", source.root, prediction_interval=10,
                                     operation_hand_policy=HAND_POLICY)
            self.assertEqual(dataset.windows[:3], ((1, 0), (1, 1), (1, 2)))
            self.assertEqual(len(dataset), 6*3)
            sample = dataset[2]
            self.assertEqual(sample["meta"]["frame_ids"].tolist(), [2, 12, 22, 32, 42, 52])
            self.assertEqual(sample["inputs"]["action_state"][:, 1, 0].tolist(), [2, 12, 22, 32, 42])
            self.assertEqual([float(x[0, 1]) for x in sample["targets"]["target_xyz"]], [12, 22, 32, 42, 52])
            self.assertEqual(sample["inputs"]["operation_hands"].tolist(), [False, True])
            self.assertEqual(dataset[9]["inputs"]["operation_hands"].tolist(), [True, True])
            batch = collate_fn([sample, dataset[-1]])
        processed = DataProcessor("001-rope")(batch)
        torch.testing.assert_close(processed["meta"]["frame_ids"], batch["meta"]["frame_ids"])
        self.assertTrue(torch.all(processed["meta"]["observation_timestamp_us"].diff() == 1000000))
        self.assertEqual(processed["inputs"]["operation_hands"].tolist(), [[False, True], [True, True]])

    def test_default_interval_one_preserves_sample_exactly(self):
        source = MemoryReconstruction()
        with source.mounted():
            old = Touch4DDataset("001-rope", "train", source.root)
            explicit = Touch4DDataset("001-rope", "train", source.root, prediction_interval=1)
            self.assertEqual(old.windows, explicit.windows)
            a, b = old[0], explicit[0]
        self.assertEqual(a["meta"].keys(), b["meta"].keys())
        self.assertNotIn("prediction_interval", a["meta"])
        self.assertNotIn("operation_hands", a["inputs"])
        for key in a["inputs"]:
            torch.testing.assert_close(a["inputs"][key], b["inputs"][key], rtol=0, atol=0)

    def test_invalid_interval_mixed_samples_and_processor_contract(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                window_starts(53, "train", 5, value)
        self.assertEqual(list(window_starts(50, "train", 5, 10)), [])
        self.assertEqual(list(window_starts(51, "train", 5, 10)), [0])
        source = MemoryReconstruction(frames=51)
        with source.mounted():
            sparse = Touch4DDataset("001-rope", "train", source.root, prediction_interval=10)[0]
            dense = Touch4DDataset("001-rope", "train", source.root)[0]
        with self.assertRaises(ValueError):
            collate_fn([sparse, dense])
        batch = collate_fn([sparse])
        batch["meta"]["frame_ids"][0, -1] = 49
        with self.assertRaises(ValueError):
            DataProcessor("001-rope")(batch)

    def test_sparse_depth_slicing_and_units_without_optional_h5py(self):
        selections = []
        class Dataset:
            shape, dtype = (53, 2, 3), np.dtype("uint16")
            is_virtual, external = False, None
            def __getitem__(self, selection):
                selections.append(selection)
                return np.broadcast_to(np.arange(53, dtype=np.uint16)[:, None, None], self.shape)[selection].copy()
        class HardLink:
            pass
        class Archive:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def keys(self): return ["data"]
            def get(self, *_args, **_kwargs): return HardLink()
            def __getitem__(self, _): return Dataset()
        fake = SimpleNamespace(File=lambda *_: Archive(), HardLink=HardLink, Dataset=Dataset)
        info = SimpleNamespace(path=Path("unused"), frames=53, height=2, width=3, cameras=("cam",),
                               intrinsics=np.eye(3)[None], camera_to_world=np.eye(4)[None])
        with patch.dict("sys.modules", {"h5py": fake}):
            targets = read_depth(info, 2, ("cam",), 5, 10)
        self.assertEqual(selections, [slice(12, 62, 10)])
        torch.testing.assert_close(targets["target_depth"][:, 0, 0, 0], torch.tensor([.012, .022, .032, .042, .052]))
        self.assertTrue(targets["target_depth_valid"].all())

    def test_hand_label_mismatch_fails(self):
        source = MemoryReconstruction(frames=51)
        with source.mounted():
            sample = Touch4DDataset("001-rope", "train", source.root, prediction_interval=10,
                                    operation_hand_policy=HAND_POLICY)[0]
        sample["inputs"]["operation_hands"][0] = True
        with self.assertRaises(ValueError):
            collate_fn([sample])
