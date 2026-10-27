"""Behavioral contracts for V4; all gradient/causal tests activate the zero head."""
import copy
from pathlib import Path
import tempfile
import unittest
import torch

from scripts.data_process import DataProcessor
from scripts.experiments.v4_stage1.fixtures import synthetic_batch, analytic_rasterizer
from scripts.training.v4_model import V4Stage1Predictor, fixed_field
from scripts.training.v4_runtime import build_model, prepare_batch, v4_objective, save_checkpoint, load_checkpoint
from scripts.training.v4_attention import GeometryCrossBlock
from scripts.training.model import PositionPredictor
from scripts.training.losses import chamfer_samples
from scripts.training.rendering import GaussianRenderer


class V4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(11)
        self.model = V4Stage1Predictor(horizon=5, prediction_interval=10)
        self.processor = DataProcessor("001-rope")
        self.raw = synthetic_batch(points=64, horizon=5, prediction_interval=10)
        self.batch = prepare_batch(self.raw, self.processor, self.model)
        self.x = self.batch["inputs"]

    def activate(self):
        torch.nn.init.normal_(self.model.motion[-1].weight, std=.003)

    def test_feature13_a0_valid_zero_missing_and_nan(self):
        x = copy.deepcopy(self.x)
        x["tactile"].zero_()
        x["controller_points"][:, 2, 1, 3] += .1
        action, hm, features, mask, cp = self.model.conditions.features(x, 2)
        self.assertEqual(features.shape, (2, 60, 13))
        self.assertTrue(torch.all(features[0, :30] == 0))
        torch.testing.assert_close(features[:, 33, 3:6], cp[:, 33]-x["controller_points"][:, 0, 1, 3])
        self.assertTrue(torch.all(features[:, 30:, 7:9] == 0))
        self.assertTrue(torch.all(features[:, 30:, 11:] == 1))  # Valid zero is observed.
        x["valid_mask"][:, 2, 2] = False
        features = self.model.conditions.features(x, 2)[2]
        self.assertTrue(torch.all(features[:, 30:45, 7:] == 0))
        x["tactile"][:, 2, 3] = float("nan")
        x["controller_points"][:, 2, 1, 0] = float("nan")
        action, hm, features, mask, _ = self.model.conditions.features(x, 2)
        self.assertTrue(torch.isfinite(features).all())
        self.assertFalse(mask[:, 30].any())

    def test_local_max_partition_coverage_age_and_finger_isolation(self):
        x = copy.deepcopy(self.x)
        x["tactile"].zero_()
        x["tactile"][:, 1, 2, 2, 3] = 9.
        features = self.model.conditions.features(x, 1)[2]
        self.assertEqual((features[:, 30:45, 8] > 0).sum(-1).tolist(), [1, 1])
        self.assertTrue(torch.all(features[:, 45:, 7:9] == 0))
        torch.testing.assert_close(features[:, 30:, 9], torch.ones(2, 30))
        torch.testing.assert_close(features[:, 30:, 10], torch.full((2, 30), .16))
        x["tactile_age_us"][:, 1, 2] = 75001
        self.assertTrue(torch.all(self.model.conditions.features(x, 1)[2][:, 30:45, 7:] == 0))

    def test_initial_invalid_reference_cumulative_and_adjacent_flags(self):
        self.activate()
        x = copy.deepcopy(self.x)
        x["controller_mask"][:, 0, 1, 0] = False
        features = self.model.conditions.features(x, 2)[2]
        self.assertTrue(torch.all(features[:, 30, 3:7] == 0))
        _, trace = self.model(x, return_trace=True)
        self.assertFalse(trace["steps"][0]["adjacent_valid"].any())
        self.assertFalse(trace["steps"][1]["adjacent_valid"][:, 30].any())
        self.assertTrue(trace["steps"][2]["adjacent_valid"][:, 30].all())

    def test_masked_hand_data_cannot_leak_after_embeddings(self):
        self.activate()
        expected, trace = self.model(self.x, return_trace=True)
        x = copy.deepcopy(self.x)
        for key in ("action_state", "controller_points"):
            x[key][0, :, 0] = float("nan")
        for key in ("tactile", "tactile_xyz"):
            x[key][0, :, :2] = 1e10
        actual = self.model(x)
        torch.testing.assert_close(actual["xyz"], expected["xyz"], rtol=0, atol=0)
        self.assertEqual(trace["steps"][0]["token_mask62"].sum(-1).tolist(), [31, 62])
        alpha = trace["steps"][0]["alpha"]
        self.assertTrue(torch.all(alpha[0, ..., :30] == 0))

    def test_geometry_initial_score_null_and_learnable_ranking(self):
        block = GeometryCrossBlock()
        q, tools = torch.randn(1, 2, 128), torch.randn(1, 3, 128)
        xyz, cp = torch.zeros(1, 2, 3), torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]])
        am, tm = torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool)
        _, alpha = block(q, tools, xyz, cp, torch.ones(1), am, tm)
        expected = torch.tensor([0., -.5, -2., 0.]).softmax(0).expand(1, 2, 4, 4)
        torch.testing.assert_close(alpha, expected)
        with torch.no_grad(): block.geometry_score[:, -1] = .5
        _, changed = block(q, tools, xyz, cp, torch.ones(1), am, tm)
        self.assertTrue(torch.all(changed[..., 2] > changed[..., 0]))
        _, null = block(q, tools, xyz, cp, torch.ones(1), am, tm & False)
        self.assertTrue(torch.all(null[..., -1] == 1))
        self.assertTrue(torch.isfinite(null).all())

    def test_full_h5_128anchors_padding_inherited_attributes(self):
        self.activate()
        pred, trace = self.model(self.x, return_trace=True)
        self.assertEqual(pred["xyz"].shape, (2, 5, 64, 3))
        self.assertEqual(trace["field"].anchor_mask.shape, (2, 128))
        self.assertEqual(trace["field"].anchor_mask.sum(-1).tolist(), [64, 17])
        self.assertEqual(trace["steps"][-1]["alpha"].shape, (2, 128, 4, 61))
        torch.testing.assert_close(trace["field"].weights.sum(-1), self.x["gs_mask"].float())
        for step in trace["steps"]:
            torch.testing.assert_close(step["alpha"].sum(-1), torch.ones(2, 128, 4))
        for key, start, stop in (("log_scales", 0, 3), ("quaternions", 3, 7), ("opacity_logits", 7, 8), ("sh_dc", 8, 11)):
            torch.testing.assert_close(pred[key], self.x["gs_attr"][:, None, :, start:stop].expand(-1, 5, -1, -1), rtol=0, atol=0)
        self.assertTrue(torch.all(pred["xyz"][1, :, 17:] == 0))
        x = copy.deepcopy(self.x)
        x["gs_xyz"][1, 17:] = float("nan")
        x["gs_attr"][1, 17:] = float("nan")
        torch.testing.assert_close(self.model(x)["xyz"], pred["xyz"], rtol=0, atol=0)

    def test_actual_anchor_history_not_control_variables_or_increment(self):
        self.activate()
        pred, trace = self.model(self.x, return_trace=True)
        field = trace["field"]
        batch = torch.arange(2)[:, None]
        for k, step in enumerate(trace["steps"]):
            expected = self.x["gs_xyz"] + field.weights @ step["cumulative_d"]
            expected = expected.masked_fill(~self.x["gs_mask"][..., None], 0.)
            torch.testing.assert_close(pred["xyz"][:, k], expected)
            if k:
                previous_state = self.x["gs_xyz"] if k == 1 else trace["steps"][k-2]["state"]
                previous_anchor = previous_state[batch, field.anchor_indices]
                expected_history = (step["anchor_xyz"]-previous_anchor).masked_fill(~field.anchor_mask[..., None], 0.)
                torch.testing.assert_close(step["anchor_history"], expected_history)
                torch.testing.assert_close(step["previous_d"], trace["steps"][k-1]["cumulative_d"])
        actual = trace["steps"][1]["anchor_history"]
        self.assertFalse(torch.allclose(actual, trace["steps"][0]["cumulative_d"], atol=1e-8, rtol=0))

    def test_four_head_motion_not_averaged_or_renormalized(self):
        self.activate()
        _, trace = self.model(self.x, return_trace=True)
        step = trace["steps"][2]
        delta = self.x["controller_points"][:, 2].reshape(2, 60, 3)-self.x["controller_points"][:, 1].reshape(2, 60, 3)
        delta = delta.masked_fill(~step["adjacent_valid"][..., None], 0.)
        expected = torch.einsum("bahj,bjd->bahd", step["alpha"][..., :60], delta).flatten(-2)
        torch.testing.assert_close(step["tool_motion"], expected)
        self.assertEqual(expected.shape, (2, 128, 12))

    def test_causality_future_conditions_and_gt_never_read(self):
        self.activate()
        base = self.model(self.x)["xyz"]
        x = copy.deepcopy(self.x)
        x["controller_points"][:, 3:] += .3
        x["action_state"][:, 3:] += .2
        x["tactile"][:, 3:] += 7
        changed = self.model(x)["xyz"]
        torch.testing.assert_close(changed[:, :3], base[:, :3], rtol=0, atol=0)
        self.assertGreater(float((changed[:, 3:]-base[:, 3:]).abs().max().detach()), 0.)
        raw = copy.deepcopy(self.raw)
        for xyz in raw["targets"]["target_xyz"]: xyz += 100
        batch = prepare_batch(raw, self.processor, self.model)
        torch.testing.assert_close(self.model(batch["inputs"])["xyz"], base, rtol=0, atol=0)

    def test_gradient_all_paths_and_cross_step_state(self):
        self.activate()
        pred, trace = self.model(self.x, return_trace=True)
        late = pred["xyz"][:, -1].square().mean()
        nodes = [s["state"] for s in trace["steps"]] + [s["alpha"] for s in trace["steps"]]
        grads = torch.autograd.grad(late, nodes, retain_graph=True, allow_unused=True)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertTrue(all(torch.any(g != 0) for g in grads))
        loss = chamfer_samples(pred["xyz"], self.batch, 32).mean()
        loss.backward()
        for group in ("motion", "cross.0.q", "cross.1.geometry_score", "cross.1.pairwise",
                      "conditions.controller", "conditions.hand", "gs_encoder", "local"):
            grads = [p.grad for n, p in self.model.named_parameters() if n.startswith(group) and p.grad is not None]
            self.assertTrue(grads, group)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads), group)
            self.assertGreater(sum(float(g.double().square().sum()) for g in grads), 0., group)

    def test_all_invalid_tools_null_and_finite(self):
        self.activate()
        x = copy.deepcopy(self.x)
        x["action_valid"].zero_(); x["controller_mask"].zero_(); x["valid_mask"].zero_()
        for name in ("action_state", "controller_points", "tactile"): x[name].fill_(float("nan"))
        pred, trace = self.model(x, return_trace=True)
        self.assertTrue(torch.isfinite(pred["xyz"]).all())
        self.assertTrue(all(torch.all(s["alpha"][..., -1] == 1) for s in trace["steps"]))

    def test_original_full_loss_mock_backward_optimizer_step_and_zero_init(self):
        pred = self.model(self.x)
        torch.testing.assert_close(pred["xyz"], self.x["gs_xyz"][:, None].expand(-1, 5, -1, -1), rtol=0, atol=0)
        renderer = GaussianRenderer(analytic_rasterizer)
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        before = self.model.motion[-1].weight.detach().clone()
        loss, parts = v4_objective(pred, self.batch, self.processor, renderer, 32)
        expected = (parts["geometry"]+parts["rgb"]+.1*parts["alpha"]).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward(); opt.step()
        self.assertFalse(torch.equal(self.model.motion[-1].weight, before))
        self.assertTrue(torch.isfinite(self.model(self.x)["xyz"]).all())

    def test_checkpoint_strict_restore_and_no_overwrite(self):
        self.activate()
        opt = torch.optim.AdamW(self.model.parameters())
        loss = chamfer_samples(self.model(self.x)["xyz"], self.batch, 32).mean()
        loss.backward(); opt.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"state.pt"
            save_checkpoint(path, self.model, opt, 1, source_sha="test", processor_state=self.processor.state_dict())
            restored = V4Stage1Predictor(horizon=5, prediction_interval=10); optimizer = torch.optim.AdamW(restored.parameters())
            state = load_checkpoint(path, restored, optimizer, processor_state=self.processor.state_dict())
            self.assertEqual(state["successful_steps"], 1)
            torch.testing.assert_close(restored(self.x)["xyz"], self.model(self.x)["xyz"], rtol=0, atol=0)
            # Resume the NEXT optimizer step, retaining Adam moments/counts.
            for model, optimizer_ in ((self.model, opt), (restored, optimizer)):
                optimizer_.zero_grad(set_to_none=True)
                chamfer_samples(model(self.x)["xyz"], self.batch, 32).mean().backward()
                optimizer_.step()
            for key in self.model.state_dict():
                torch.testing.assert_close(restored.state_dict()[key], self.model.state_dict()[key], rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, self.model, opt, 1, source_sha="test", processor_state={})
            state["config"]["prediction_interval"] = 1
            bad = Path(directory)/"bad.pt"; torch.save(state, bad)
            with self.assertRaises(ValueError): load_checkpoint(bad, restored)
            state["config"]["prediction_interval"] = 10
            state["version"] = "touch4d-gs-v7"; torch.save(state, bad)
            with self.assertRaises(ValueError): load_checkpoint(bad, restored)

    def test_config_and_data_contract_fail_closed(self):
        for cfg in ({"horizon": 0}, {"prediction_interval": 0}, {"horizon": True}, {"prediction_interval": 1.5}):
            with self.assertRaises(ValueError): V4Stage1Predictor(**cfg)
        raw = copy.deepcopy(self.raw); raw["meta"]["frame_ids"][0, -1] += 1
        with self.assertRaises(ValueError): prepare_batch(raw, self.processor, self.model)
        x = copy.deepcopy(self.x); del x["condition_frame_ids"]
        with self.assertRaises(ValueError): self.model(x)

    def test_legacy_factory_preserves_default_parameters_and_outputs(self):
        torch.manual_seed(23); legacy = PositionPredictor()
        torch.manual_seed(23); via_factory = build_model()
        self.assertEqual(legacy.config, via_factory.config)
        for key in legacy.state_dict():
            torch.testing.assert_close(legacy.state_dict()[key], via_factory.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(legacy(self.x), via_factory(self.x), rtol=0, atol=0)
