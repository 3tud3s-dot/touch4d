"""CPU-only frozen-path contracts; no optimizer, backward or performance claim."""
import copy
import unittest

import torch

from scripts.data_process import DataProcessor
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor, G5PropagatePredictor
from scripts.experiments.v4_g5_cross_necessity.interventions import (
    MODES, MODE_CONFIGS, FACTORIAL_MODES, SELECTED, Mode, metadata, switches,
    intervention, capture_normal, replay_step, only_step_forward,
    branch_parts, combine, one_frame,
)
from scripts.training.model import masked
from scripts.training.v4_runtime import prepare_batch


class G5CrossInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, horizon=15):
        torch.manual_seed(470)
        model = G5DirectPredictor(horizon=horizon).eval().requires_grad_(False)
        with torch.no_grad():
            model.motion[-1].weight.normal_(std=.0002)
            model.global_motion.mapping[-1].weight.normal_(std=.0003)
            model.direct_motion.mapping[-1].weight.normal_(std=.0003)
            # Make scores content-dependent, not only the initial geometry prior.
            for block in model.cross:
                block.q.weight.normal_(std=.003)
                for score in block.pairwise:
                    score[-1].weight.normal_(std=.003)
        batch = prepare_batch(synthetic_batch(55, horizon, 1), DataProcessor("001-rope"), model)
        return model, batch

    def assert_prediction_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)

    def test_mode_table_factorial_and_explicit_mass_closure(self):
        self.assertEqual(len(MODES), 10)
        self.assertEqual(SELECTED, (1, 5, 10, 15))
        self.assertEqual(len({switches(name, 1)[:3] for name in FACTORIAL_MODES}), 8)
        self.assertFalse(metadata("NoCrossValue+NoToolMotion12")["close_mass"])
        self.assertTrue(metadata("NoAttentionPaths")["close_mass"])
        self.assertTrue(metadata("NoBase+NoAttentionPaths")["close_mass"])
        self.assertTrue(switches("NoBase-H2+", 0)[0])
        self.assertFalse(switches("NoBase-H2+", 1)[0])
        self.assertEqual(set(MODE_CONFIGS), set(MODES))
        for bad in ("unregistered", None, Mode(base=1)):
            with self.assertRaises(ValueError):
                switches(bad, 0)

    @torch.inference_mode()
    def test_normal_h15_outputs_branches_capture_and_weights_bit_exact(self):
        model, batch = self.fixture()
        before = state_hash(model.state_dict())
        expected, expected_trace = model(batch["inputs"], return_trace=True)
        actual, trace, raw = capture_normal(model, batch["inputs"])
        self.assert_prediction_equal(expected, actual)
        self.assertEqual(len(raw), 15)
        for index, row in enumerate(raw):
            step = trace["steps"][index]
            self.assertEqual(len(row["cross_calls"]), 2)
            self.assertEqual(len(row["cross_projection_raw"]), 2)
            self.assertEqual(row["motion_input"].shape[-1], 151)
            self.assertIs(row["motion_input"], row["effective_motion_input"])
            torch.testing.assert_close(masked(row["base"], trace["field"].anchor_mask),
                                       step["base_cumulative_d"], rtol=0, atol=0)
            torch.testing.assert_close(combine(branch_parts(step, trace["field"]), trace["field"]),
                                       step["cumulative_d"], rtol=0, atol=0)
            torch.testing.assert_close(step["state"], expected_trace["steps"][index]["state"],
                                       rtol=0, atol=0)
        self.assertEqual(before, state_hash(model.state_dict()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    @torch.inference_mode()
    def test_h15_replay_matches_only_step_rollouts_for_all_modes_and_horizons(self):
        model, batch = self.fixture()
        x = batch["inputs"]
        normal, trace, raw = capture_normal(model, x)
        before = state_hash(model.state_dict())
        for horizon in SELECTED:
            index = horizon-1
            for mode in MODES:
                with self.subTest(horizon=horizon, mode=mode):
                    replay = replay_step(model, x, trace, raw, index, mode)
                    actual = only_step_forward(model, x, mode, index)
                    torch.testing.assert_close(actual["xyz"][:, :index], normal["xyz"][:, :index],
                                               rtol=0, atol=0)
                    torch.testing.assert_close(replay["xyz"], actual["xyz"][:, index], rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(replay["xyz"]).all())
                    for key in normal:
                        if key != "xyz":
                            torch.testing.assert_close(actual[key], normal[key], rtol=0, atol=0)
        self.assertEqual(before, state_hash(model.state_dict()))

    @torch.inference_mode()
    def test_value_suppression_preserves_both_original_ffns_scores_and_masks(self):
        model, batch = self.fixture(5)
        x = batch["inputs"]
        with intervention(model, "NoCrossValue") as raw:
            actual = model(x)
        for row in raw:
            for block, call in zip(model.cross, row["cross_calls"]):
                q, _, _, _, _, anchor_valid, _ = call["args"]
                expected = masked(q+block.ff(block.ff_norm(q)), anchor_valid)
                query, alpha = call["output"]
                torch.testing.assert_close(query, expected, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(alpha).all())
                torch.testing.assert_close(alpha.sum(-1), torch.ones_like(alpha.sum(-1)),
                                           rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.isfinite(actual["xyz"]).all())

    @torch.inference_mode()
    def test_tool12_suppression_does_not_silently_close_head_mass(self):
        model, batch = self.fixture(5)
        with intervention(model, "NoToolMotion12") as raw:
            model(batch["inputs"])
        for row in raw:
            original, effective = row["motion_input"], row["effective_motion_input"]
            self.assertEqual(int(effective[..., 128:140].count_nonzero()), 0)
            torch.testing.assert_close(effective[..., :128], original[..., :128], rtol=0, atol=0)
            torch.testing.assert_close(effective[..., 140:], original[..., 140:], rtol=0, atol=0)
        self.assertGreater(int(raw[-1]["effective_motion_input"][..., 146:150].count_nonzero()), 0)
        with intervention(model, "NoAttentionPaths") as raw:
            model(batch["inputs"])
        self.assertTrue(all(not row["effective_motion_input"][..., 146:150].count_nonzero() for row in raw))

    @torch.inference_mode()
    def test_no_base_makes_tool_and_head_mass_toggles_exactly_redundant_h15(self):
        model, batch = self.fixture()
        x = batch["inputs"]
        for value in (False, True):
            outputs = []
            for tool, mass in ((True, False), (False, False), (False, True)):
                with intervention(model, Mode(base=False, cross_value=value,
                                               tool_motion=tool, close_mass=mass)):
                    outputs.append(model(x))
            for result in outputs[1:]:
                self.assert_prediction_equal(outputs[0], result)
        torch.testing.assert_close(outputs[0]["xyz"][:, 0], masked(x["gs_xyz"], x["gs_mask"]),
                                   rtol=0, atol=0)

    @torch.inference_mode()
    def test_no_base_h2_keeps_real_h1_and_only_suppresses_later_xyz_head(self):
        model, batch = self.fixture()
        normal = model(batch["inputs"])
        with intervention(model, "NoBase-H2+") as raw:
            actual = model(batch["inputs"])
        torch.testing.assert_close(actual["xyz"][:, 0], normal["xyz"][:, 0], rtol=0, atol=0)
        self.assertGreater(int(raw[0]["effective_base"].count_nonzero()), 0)
        self.assertTrue(all(not row["effective_base"].count_nonzero() for row in raw[1:]))
        self.assertTrue(any(row["base"].count_nonzero() for row in raw[1:]))

    @torch.inference_mode()
    def test_joint_closed_attention_has_no_remaining_tactile_path(self):
        model, batch = self.fixture()
        x = batch["inputs"]
        changed = copy.deepcopy(x)
        changed["tactile"] = torch.rand_like(changed["tactile"])*20
        changed["tactile_age_us"] = torch.zeros_like(changed["tactile_age_us"])
        with intervention(model, "NoBase+NoAttentionPaths"):
            expected = model(x)
        with intervention(model, "NoBase+NoAttentionPaths"):
            actual = model(changed)
        self.assert_prediction_equal(expected, actual)

    @torch.inference_mode()
    def test_single_and_double_hand_mask_padding_and_invalid_nan_do_not_leak(self):
        model, batch = self.fixture()
        x = batch["inputs"]
        self.assertEqual(x["operation_hands"].tolist(), [[False, True], [True, True]])
        changed = copy.deepcopy(x)
        changed["action_state"][0, :, 0] = float("nan")
        changed["controller_points"][0, :, 0] = float("nan")
        changed["tactile"][0, :, :2] = float("nan")
        changed["gs_xyz"][~changed["gs_mask"]] = float("nan")
        changed["gs_attr"][~changed["gs_mask"]] = float("nan")
        for mode in ("Normal", "NoCrossValue", "NoAttentionPaths", "NoBase+NoAttentionPaths"):
            with intervention(model, mode):
                expected = model(x)
            with intervention(model, mode):
                actual = model(changed)
            self.assert_prediction_equal(expected, actual)
            self.assertTrue(all(torch.isfinite(v).all() for v in actual.values()))
            self.assertEqual(int(actual["xyz"][~x["gs_mask"][:, None].expand(-1, 15, -1)].count_nonzero()), 0)

    @torch.inference_mode()
    def test_future_condition_changes_cannot_affect_the_prefix(self):
        model, batch = self.fixture()
        x = batch["inputs"]
        changed = copy.deepcopy(x)
        changed["action_state"][:, 10:, :, :3] += .03
        changed["controller_points"][:, 10:] += .03
        changed["tactile"][:, 10:] *= 7
        for mode in ("Normal", "NoAttentionPaths", "NoBase+NoAttentionPaths"):
            with intervention(model, mode):
                expected = model(x)
            with intervention(model, mode):
                actual = model(changed)
            torch.testing.assert_close(expected["xyz"][:, :10], actual["xyz"][:, :10], rtol=0, atol=0)

    def test_hooks_are_removed_after_exceptions_and_invalid_modes_rejected(self):
        model, batch = self.fixture(5)
        counts = lambda: [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
        before = counts()
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with intervention(model, "NoAttentionPaths"):
                raise RuntimeError("intentional")
        self.assertEqual(before, counts())
        for index in (-1, 5, True):
            with self.assertRaises(ValueError):
                with intervention(model, only_step=index):
                    pass
        with self.assertRaises(ValueError):
            with intervention(G5PropagatePredictor(horizon=5)):
                pass
        with self.assertRaises(ValueError):
            with intervention(model, "unknown"):
                pass
        self.assertEqual(before, counts())

    @torch.inference_mode()
    def test_replay_rejects_active_capture_and_wrong_context(self):
        model, batch = self.fixture(5)
        _, trace, raw = capture_normal(model, batch["inputs"])
        with self.assertRaises(ValueError):
            replay_step(model, batch["inputs"], trace, raw[:-1], 1)
        with intervention(model, "NoCrossValue") as intervened:
            _, altered_trace = model(batch["inputs"], return_trace=True)
        with self.assertRaisesRegex(ValueError, "Normal captures"):
            replay_step(model, batch["inputs"], altered_trace, intervened, 1)
        with intervention(model) as active:
            model(batch["inputs"])
            with self.assertRaises(ValueError):
                replay_step(model, batch["inputs"], trace, active, 1)

    @torch.inference_mode()
    def test_one_frame_is_only_an_aligned_loss_render_view(self):
        model, batch = self.fixture()
        prediction = model(batch["inputs"])
        view, pred = one_frame(batch, prediction, 14)
        self.assertIs(view["inputs"], batch["inputs"])
        self.assertEqual(pred["xyz"].shape[1], 1)
        self.assertEqual(len(view["targets"]["target_xyz"]), 1)
        torch.testing.assert_close(view["meta"]["frame_ids"], batch["meta"]["frame_ids"][:, 14:16])
        self.assertEqual(len(batch["targets"]["target_xyz"]), 15)
        self.assertEqual(batch["meta"]["frame_ids"].shape[1], 16)


if __name__ == "__main__":
    unittest.main()
