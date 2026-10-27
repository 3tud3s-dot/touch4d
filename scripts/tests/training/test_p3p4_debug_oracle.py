"""CPU checks for the offline D0 capacity diagnostic, not model quality gates."""

import copy
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.data_process import DataProcessor
from scripts.experiments.debug_audit.oracle import (
    D0Oracle, centroid_translation_oracle, field_diagnostics, run_oracle,
)
from scripts.experiments.fixed_window.fixed_fit import TargetCache
from scripts.tests.training.test_fixed_fit import single_raw
from scripts.tests.training.test_training import fake_gaussian_rasterizer
from scripts.training.losses import chamfer_samples
from scripts.training.rendering import GaussianRenderer


class DebugOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(42)
        self.processor = DataProcessor("001-rope")
        self.batch = self.processor(single_raw(window=(2, 73)))
        self.renderer = TargetCache(GaussianRenderer(fake_gaussian_rasterizer))

    def test_zero_equals_copy_and_constant_translation_survives_interpolation(self):
        oracle = D0Oracle(self.batch["inputs"], 5)
        keep = oracle.keep
        zero = oracle()["xyz"]
        torch.testing.assert_close(zero, oracle.initial[:, None].expand_as(zero), rtol=0, atol=0)
        shift = torch.tensor([.02, -.01, .005])
        with torch.no_grad():
            oracle.delta.copy_(shift.expand_as(oracle.delta))
        xyz = oracle()["xyz"]
        for h in range(5):
            torch.testing.assert_close((xyz[:, h]-oracle.initial)[keep],
                                       shift.expand_as(oracle.initial)[keep], rtol=2e-5, atol=1e-7)
        field = field_diagnostics(oracle, self.processor.state_dict()["scale"])
        self.assertLess(field["row_sum_max_error"], 2e-6)
        self.assertEqual(field["padding_weight_abs_sum"], 0.)

    def test_geometry_is_fixed_but_every_horizon_anchor_has_a_live_gradient(self):
        oracle = D0Oracle(self.batch["inputs"], 5)
        self.assertFalse(oracle.field.weights.requires_grad)
        loss = chamfer_samples(oracle()["xyz"], self.batch, 2).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(oracle.delta.grad).all())
        for h in range(5):
            self.assertGreater(float(oracle.delta.grad[:, h].abs().sum()), 0.)
        self.assertTrue(torch.equal(oracle.delta.grad[~oracle.field.anchor_mask[:, None].expand(-1,5,-1)],
                                    torch.zeros_like(oracle.delta.grad[~oracle.field.anchor_mask[:, None].expand(-1,5,-1)])))

    def test_centroid_oracle_uses_independent_sets_not_point_rows(self):
        changed = copy.deepcopy(self.batch)
        for h in range(5):
            count = changed["targets"]["target_xyz"][h].shape[1]
            order = torch.arange(count-1, -1, -1)
            changed["targets"]["target_xyz"][h] = changed["targets"]["target_xyz"][h][:, order]
            changed["targets"]["target_mask"][h] = changed["targets"]["target_mask"][h][:, order]
        first, second = centroid_translation_oracle(self.batch), centroid_translation_oracle(changed)
        torch.testing.assert_close(first["xyz"], second["xyz"], rtol=1e-6, atol=1e-7)
        keep = self.batch["inputs"]["gs_mask"][0]
        for h in range(5):
            gt = self.batch["targets"]["target_xyz"][h][0, self.batch["targets"]["target_mask"][h][0]]
            torch.testing.assert_close(first["xyz"][0,h,keep].mean(0), gt.mean(0), atol=1e-6, rtol=1e-6)

    def test_invalid_padding_does_not_enter_field_or_prediction(self):
        inputs = copy.deepcopy(self.batch["inputs"])
        inputs["gs_mask"][:, -1] = False
        inputs["gs_xyz"][:, -1] = float("nan")
        inputs["gs_attr"][:, -1] = float("nan")
        oracle = D0Oracle(inputs, 5)
        prediction = oracle()
        self.assertTrue(all(torch.isfinite(v).all() for v in prediction.values()))
        self.assertEqual(float(prediction["xyz"][:, :, -1].detach().abs().sum()), 0.)

    def test_bounded_runner_writes_fresh_evidence_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "oracle"
            result = run_oracle(self.batch, self.processor, self.renderer, output, updates=2)
            self.assertEqual(result["successful_updates"], 2)
            self.assertEqual(result["config"]["weights"]["rgb_weight"], 0.)
            self.assertTrue((output / "prediction_2.pt").is_file())
            self.assertTrue((output / "oracle_results.json").is_file())
            with self.assertRaises(FileExistsError):
                run_oracle(self.batch, self.processor, self.renderer, output, updates=2)
            with self.assertRaises(ValueError):
                run_oracle(self.batch, self.processor, self.renderer, Path(temporary)/"bad", updates=401)


if __name__ == "__main__":
    unittest.main()
