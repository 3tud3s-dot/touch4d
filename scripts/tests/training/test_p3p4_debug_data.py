"""CPU-only independent data-path audit checks; no real data/checkpoints/GPU."""
import copy
import unittest
from unittest.mock import patch

import torch

from scripts.dataloader.dataset import Touch4DDataset
from scripts.dataloader.collate import collate_fn
from scripts.data_process import DataProcessor
from scripts.tests.dataloader.test_dataloader import MemoryReconstruction
from scripts.tests.data_process.test_data_process import make_batch
from scripts.experiments.spatial_dynamics.conditions import ToolPredictor, hand_inputs, sanitize
from scripts.experiments.debug_audit.data_audit import audit_window, condition_evidence


class DebugDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(42)
        self.model = ToolPredictor(width=16, heads=4, layers=2, controller_features=7,
                                  horizon=5, prediction_mode="autoregressive", output_mode="gs_geometry",
                                  gs_local_attention="knn", gs_local_k=3, gs_decoder="anchors",
                                  anchor_count=4, anchor_pool_k=3, gs_structure="anchor_attention")
        self.processor = DataProcessor("001-rope")

    def inputs(self):
        batch = self.processor(make_batch())
        x = hand_inputs(batch["inputs"], torch.tensor([2, 5]), ["001-rope"]*2)
        for key in ("action_valid", "controller_mask", "valid_mask", "tactile_spatial_mask"):
            x[key].fill_(True)
        x["valid_mask"][..., 31] = False
        x["tactile_spatial_mask"][..., 31] = False
        x["tactile_age_us"].fill_(1000)
        x["tactile"].fill_(.5)
        return x

    def test_hand_policy_and_t3_observed_zero_keep_are_independent(self):
        x = self.inputs()
        x["tactile"].zero_()
        evidence = condition_evidence(x, self.model)
        self.assertEqual(evidence["operation_hands"], [[False, True], [True, True]])
        self.assertEqual(evidence["token_valid_per_time"], [[31]*5, [62]*5])
        self.assertEqual(evidence["tool_pressure_gate_per_finger"][0][0], [0, 0, 15, 15])
        self.assertGreater(evidence["valid_zero_taxels"], 0)
        self.assertEqual(evidence["positive_taxels"], 0)

    def test_fallback_only_pose_missing_cannot_reopen_absent_hand(self):
        x = self.inputs()
        x["controller_mask"].zero_()
        evidence = condition_evidence(x, self.model)
        self.assertEqual(evidence["fallback_valid"][0][0], [False, False, True, True])
        self.assertEqual(evidence["fallback_valid"][1][0], [True]*4)
        x["valid_mask"][1].zero_()
        evidence = condition_evidence(x, self.model)
        self.assertEqual(evidence["fallback_valid"][1][0], [False]*4)

    def test_cpu_audit_reads_distinct_initial_and_future_source_frames(self):
        source = MemoryReconstruction(frames=16)
        with source.mounted(), patch("scripts.experiments.debug_audit.data_audit.file_sha", return_value="memory-fixture"):
            ds = Touch4DDataset("001-rope", "train", source.root)
            # Fixture positions are large; forward/inverse contracts remain exact enough.
            result = audit_window(ds, 2, 3, self.processor, self.model)
        self.assertEqual(result["frame_ids"], [3, 4, 5, 6, 7, 8])
        self.assertEqual(result["condition_frames"], [3, 4, 5, 6, 7])
        self.assertEqual(result["target_frames"], [4, 5, 6, 7, 8])
        self.assertGreater(len(set(result["point_counts"])), 1)
        self.assertEqual(result["physical_hand_to_slot"], {"left": None, "right": 0})

    def test_future_target_changes_do_not_enter_condition_encoder(self):
        x = self.inputs()
        tokens, keep = self.model.encode_conditions(x)
        altered = dict(x, target_xyz=torch.full((2, 5, 9, 3), float("nan")))
        again, again_keep = self.model.encode_conditions(altered)
        self.assertTrue(torch.equal(tokens, again))
        self.assertTrue(torch.equal(keep, again_keep))
        later = copy.deepcopy(x)
        later["controller_points"][:, 3:] += 100
        later["tactile"][:, 3:] += 100
        prefix, _ = self.model.encode_conditions(later)
        self.assertTrue(torch.equal(tokens[:, :3], prefix[:, :3]))

    def test_pressure_features_enter_fusion_and_zero_init_only_blocks_initial_output(self):
        x = self.inputs()
        captured = []
        hook = self.model.fusion.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        initial_tokens = self.model.encode_conditions(x)[0]
        zero = copy.deepcopy(x); zero["tactile"].zero_()
        zero_tokens = self.model.encode_conditions(zero)[0]
        hook.remove()
        self.assertFalse(torch.equal(captured[0][:, :, 2:], captured[1][:, :, 2:]))
        self.assertTrue(torch.equal(initial_tokens, zero_tokens))
        with torch.no_grad():
            self.model.fusion[-1].weight.normal_(std=.05)
        trained_path = self.model.encode_conditions(x)[0]
        trained_zero = self.model.encode_conditions(zero)[0]
        self.assertGreater(float((trained_path-trained_zero).detach().abs().sum()), 0.)
        absent = [0, *range(2, 32), 62, 63]
        self.assertEqual(float(trained_path[0, :, absent].detach().abs().sum()), 0.)

    def test_age_cutoff_and_nan_clear_before_difference(self):
        x = self.inputs()
        x["tactile_age_us"][0, 1, 2] = 75001
        x["tactile"][0, 1, 2] = float("nan")
        x["controller_points"][0, 0, 1] = float("nan")
        clean = sanitize(x)
        self.assertFalse(clean["valid_mask"][0, 1, 2].any())
        features = self.model.controller_inputs(clean)
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue((features[0, :, 1, :, 3:] == 0).all())
        self.assertTrue(torch.isfinite(self.model.encode_conditions(x)[0]).all())


if __name__ == "__main__":
    unittest.main()
