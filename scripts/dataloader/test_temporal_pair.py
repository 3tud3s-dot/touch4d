"""Disk-free temporal history and legacy-path regression tests."""
import copy
import unittest
import torch
from .test_dataloader import MemoryReconstruction
from .dataset import Touch4DDataset
from .collate import collate_fn
from .temporal_pair import validate_pair_batch
from ..data_process import DataProcessor
from ..training.fixed_fit import digest


class TemporalPairTests(unittest.TestCase):
    def test_history_one_unchanged_and_two_distinct_counts(self):
        with MemoryReconstruction(frames=18).mounted() as source:
            root = source.root
            old = Touch4DDataset('001-rope', 'train', root, horizon=10)
            one = Touch4DDataset('001-rope', 'train', root, horizon=10, history=1)
            two = Touch4DDataset('001-rope', 'train', root, horizon=10, history=2)
            self.assertEqual(digest(old[0]), digest(one[0]))
            self.assertTrue(all(s > 0 for _, s in two.windows))
            s = two[0]; t = int(s['meta']['frame_ids'][0])
            self.assertEqual(s['meta']['history_frame_ids'].tolist(), [t-1, t])
            self.assertEqual(s['meta']['frame_ids'].tolist(), list(range(t, t+11)))
            self.assertNotEqual(len(s['inputs']['gs_prev_xyz']), len(s['inputs']['gs_xyz']))
            self.assertEqual(len(s['targets']['target_xyz']), 10)

    def test_padding_normalization_and_bad_metadata(self):
        with MemoryReconstruction(frames=18).mounted() as source:
            ds = Touch4DDataset('001-rope', 'train', source.root, horizon=10, history=2)
            raw = collate_fn([ds[0], ds[1]])
            validate_pair_batch(raw)
            p = DataProcessor('001-rope'); normalized = p(raw); restored = p.denormalize_batch(normalized)
            for key in ('gs_xyz', 'gs_attr', 'gs_prev_xyz', 'gs_prev_attr'):
                torch.testing.assert_close(restored['inputs'][key], raw['inputs'][key], atol=1e-6, rtol=1e-6)
            bad = copy.deepcopy(raw); bad['meta']['history_frame_ids'][0, 0] -= 1
            with self.assertRaises(ValueError): p(bad)
            bad = copy.deepcopy(raw); bad['meta']['history_timestamp_us'][:, 0] = bad['meta']['history_timestamp_us'][:, 1]
            with self.assertRaises(ValueError): p(bad)

    def test_mixed_history_rejected(self):
        with MemoryReconstruction(frames=18).mounted() as source:
            a = Touch4DDataset('001-rope', 'train', source.root)
            b = Touch4DDataset('001-rope', 'train', source.root, history=2)
            with self.assertRaises(ValueError): collate_fn([a[1], b[0]])
            for invalid in (0, 3, True):
                with self.assertRaises(ValueError): Touch4DDataset('001-rope', 'train', source.root, history=invalid)

if __name__ == '__main__': unittest.main()
