"""G4 isolated architecture contracts; CPU tests are not real-data evidence."""
import copy
import io
import unittest

import torch

from scripts.data_process import DataProcessor
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.experiments.v4_g4_global_motion.model import (
    G4GlobalMotionPredictor)
from scripts.training.v4_model import V4Stage1Predictor
from scripts.training.v4_runtime import prepare_batch


class G4GlobalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(72)
        self.base = V4Stage1Predictor(horizon=5, prediction_interval=1)
        torch.manual_seed(72)
        self.g4 = G4GlobalMotionPredictor(horizon=5, prediction_interval=1)
        raw = synthetic_batch(points=40, horizon=5, prediction_interval=1)
        self.inputs = prepare_batch(raw, DataProcessor("001-rope"), self.base)["inputs"]
        # Make the base network's motion nontrivial, not a trivially copied state.
        with torch.no_grad():
            self.base.motion[-1].weight.normal_(std=.003)
            self.g4.motion[-1].weight.copy_(self.base.motion[-1].weight)

    def test_h15_zero_init_full_ar_matches_original(self):
        torch.manual_seed(127)
        base = V4Stage1Predictor(horizon=15, prediction_interval=1)
        torch.manual_seed(127)
        g4 = G4GlobalMotionPredictor(horizon=15, prediction_interval=1)
        with torch.no_grad():
            base.motion[-1].weight.normal_(std=.002)
            g4.motion[-1].weight.copy_(base.motion[-1].weight)
        raw = synthetic_batch(points=25, horizon=15, prediction_interval=1)
        x = prepare_batch(raw, DataProcessor("001-rope"), base)["inputs"]
        baseline = base(x)
        changed = g4(x)
        for key in baseline:
            torch.testing.assert_close(changed[key], baseline[key], rtol=0, atol=0)

    def test_initialization_and_shared_state_exact(self):
        for name, p in self.base.state_dict().items():
            torch.testing.assert_close(p, self.g4.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(torch.all(self.g4.global_motion.mapping[-1].weight == 0))
        self.assertTrue(torch.all(self.g4.global_motion.mapping[-1].bias == 0))
        self.assertNotEqual(self.base.config["architecture"], self.g4.config["architecture"])

    def test_zero_init_entire_rollout_matches_g0_bit_exact(self):
        baseline = self.base(self.inputs)
        prediction, trace = self.g4(self.inputs, return_trace=True)
        self.assertEqual(set(prediction), set(baseline))
        for key in baseline:
            torch.testing.assert_close(prediction[key], baseline[key], rtol=0, atol=0)
        for step in trace["steps"]:
            torch.testing.assert_close(step["cumulative_d"], step["base_cumulative_d"], rtol=0, atol=0)
            self.assertTrue(torch.all(step["global_delta"] == 0))
        self.assertEqual(trace["field"].anchor_mask.shape[-1], 128)

    def test_root_motion_signed_and_absent_hand_stays_zero(self):
        # At normalized H3, synthetic +X root displacement is positive.
        _, trace = self.g4(self.inputs, return_trace=True)
        basis = trace["steps"][2]["hand_basis"]
        self.assertTrue(torch.all(basis[:, 1, 0] > 0))
        self.assertTrue(torch.all(basis[:, 1, 3] > 0))
        self.assertTrue(torch.all(basis[0, 0] == 0))  # ep2 has no left hand
        self.assertTrue(torch.all(trace["steps"][0]["hand_basis"] == 0))
        # If the underlying signed root movement reverses, basis must reverse.
        x = copy.deepcopy(self.inputs)
        x["action_state"][:, 2, 1, :3] = (
            2 * x["action_state"][:, 0, 1, :3] - x["action_state"][:, 2, 1, :3])
        _, changed = self.g4(x, return_trace=True)
        torch.testing.assert_close(changed["steps"][2]["hand_basis"][:, 1, :3],
                                   -basis[:, 1, :3], rtol=0, atol=1e-6)

    def test_h2_exact_shared_translation_and_no_direct_shape_warp(self):
        # Make G4 output a constant B_{x,root_cumulative_x} = 0.75.
        with torch.no_grad():
            self.g4.global_motion.mapping[-1].bias[0] = .75
        baseline = self.base(self.inputs)["xyz"]
        pred, trace = self.g4(self.inputs, return_trace=True)
        pred = pred["xyz"]
        torch.testing.assert_close(pred[:, 0], baseline[:, 0], atol=0, rtol=0)
        delta = trace["steps"][1]["global_delta"]
        keep = self.inputs["gs_mask"]
        for b in range(len(keep)):
            valid = (pred[b, 1]-baseline[b, 1])[keep[b]]
            torch.testing.assert_close(valid, delta[b].expand_as(valid), rtol=3e-5, atol=5e-7)
            # A uniform extra shift cannot change inter-GS relative positions at H2.
            torch.testing.assert_close(valid-valid[:1], torch.zeros_like(valid),
                                       atol=5e-7, rtol=0)
        self.assertGreater(float(delta.abs().max()), 0)
        self.assertTrue(torch.isfinite(pred).all())

    def test_validity_missing_endpoints_and_zero_motion(self):
        with torch.no_grad():
            self.g4.global_motion.mapping[-1].bias[0] = 1.
        x = copy.deepcopy(self.inputs)
        x["action_valid"][:, 2, :] = False
        x["controller_mask"][:, 2, :] = False
        x["controller_points"][:, 2, :] = float("nan")
        x["action_state"][:, 2, :] = float("nan")
        _, trace = self.g4(x, return_trace=True)
        self.assertTrue(torch.all(trace["steps"][2]["global_delta"] == 0))
        self.assertTrue(torch.isfinite(trace["steps"][2]["state"]).all())
        x = copy.deepcopy(self.inputs)
        x["controller_mask"][:, :, :, :] = False
        # Missing controllers remove controller-based motion only; valid root
        # action still contributes its signed displacement.
        _, trace = self.g4(x, return_trace=True)
        basis = trace["steps"][2]["hand_basis"]
        self.assertTrue(torch.all(basis[..., 6:] == 0))
        self.assertGreater(float(basis[:, 1, 0].abs().max()), 0.)
        x = copy.deepcopy(self.inputs)
        x["controller_points"][:] = x["controller_points"][:, :1]
        x["action_state"][:] = x["action_state"][:, :1]
        _, trace = self.g4(x, return_trace=True)
        self.assertTrue(all((s["global_delta"] == 0).all() for s in trace["steps"]))

    def test_masked_hand_isolation_with_active_mapping(self):
        with torch.no_grad():
            self.g4.global_motion.mapping[-1].weight.normal_(std=.002)
        before = self.g4(self.inputs)
        x = copy.deepcopy(self.inputs)
        x["action_state"][0, :, 0] = float("nan")
        x["controller_points"][0, :, 0] = float("nan")
        x["tactile"][0, :, :2] = float("nan")
        after = self.g4(x)
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def test_internal_mapping_gradient_after_nonzero_projection(self):
        with torch.no_grad():
            self.g4.global_motion.mapping[-1].weight.normal_(std=.002)
        self.g4.zero_grad(set_to_none=True)
        pred = self.g4(self.inputs)["xyz"]
        (pred[:, -1] - self.inputs["gs_xyz"]).square().mean().backward()
        for name, parameter in self.g4.global_motion.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)

    def test_state_dict_roundtrip_and_original_strict_rejection(self):
        with torch.no_grad():
            self.g4.global_motion.mapping[-1].weight.normal_(std=.002)
        buffer = io.BytesIO()
        torch.save(dict(config=self.g4.config, model=self.g4.state_dict()), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = G4GlobalMotionPredictor(horizon=5, prediction_interval=1)
        self.assertEqual(saved["config"], restored.config)
        restored.load_state_dict(saved["model"], strict=True)
        before, after = self.g4(self.inputs), restored(self.inputs)
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            self.base.load_state_dict(saved["model"], strict=True)

    def test_future_causality_and_gradient_into_new_head(self):
        with torch.no_grad():
            self.g4.global_motion.mapping[-1].bias[0] = .5
        base = self.g4(self.inputs)["xyz"]
        x = copy.deepcopy(self.inputs)
        x["action_state"][:, 4, 1, 0] += .04
        x["controller_points"][:, 4, 1, :, 0] += .04
        changed = self.g4(x)["xyz"]
        torch.testing.assert_close(changed[:, :4], base[:, :4], rtol=0, atol=0)
        self.assertGreater(float((changed[:, -1]-base[:, -1]).abs().max()), 0.)
        # Reset zero init and verify first backward already reaches output map.
        with torch.no_grad():
            self.g4.global_motion.mapping[-1].weight.zero_()
            self.g4.global_motion.mapping[-1].bias.zero_()
        self.g4.zero_grad(set_to_none=True)
        pred = self.g4(self.inputs)["xyz"]
        loss = (pred[:, -1] - self.inputs["gs_xyz"]).square().mean()
        loss.backward()
        grad = self.g4.global_motion.mapping[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
