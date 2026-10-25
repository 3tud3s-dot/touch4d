"""P2 structure features, masks, causal rollout and unchanged P1a contracts."""

import copy
import unittest
from unittest.mock import patch

import torch

from .gs_decoder import build_initial_anchor_field
from .gs_structure import StructuralAnchorLayer, StructuralGSBlock
from .model import PositionPredictor
from . import test_gs_decoder as anchor_tests
from .test_gs_decoder import model_config


class StructuralTests(unittest.TestCase):
    setUpClass = classmethod(anchor_tests.AnchorDecoderTests.setUpClass.__func__)
    tearDownClass = classmethod(anchor_tests.AnchorDecoderTests.tearDownClass.__func__)

    def setUp(self):
        anchor_tests.AnchorDecoderTests.setUp(self)
        self.model = PositionPredictor(**model_config(width=128, gs_structure="anchor_attention"))

    def activate(self):
        torch.nn.init.normal_(self.model.position_head[-1].weight, std=.02)
        torch.nn.init.normal_(self.model.gs_structure.output.weight, std=.02)

    # Exercise the learned branch when reusing P1a's causal, masked, gradient
    # and inherited-attribute contracts, rather than testing only a zero gate.
    test_causality_and_masks = anchor_tests.AnchorDecoderTests.test_five_step_causality_and_invalid_nan_masks
    test_missing_sample = anchor_tests.AnchorDecoderTests.test_all_missing_sample_remains_finite_and_does_not_mix_batch_geometry
    test_cross_step_chamfer = anchor_tests.AnchorDecoderTests.test_fifth_chamfer_reaches_every_previous_anchor_head_and_condition
    test_joint_loss = anchor_tests.AnchorDecoderTests.test_joint_loss_has_zero_scale_regularizer_and_finite_anchor_gradients
    test_initial_relative_constant_translation = anchor_tests.AnchorDecoderTests.test_zero_and_constant_head_are_initial_relative_not_accumulated

    def test_disabled_rng_and_zero_structural_initial_alignment_with_active_head(self):
        torch.manual_seed(42)
        baseline = PositionPredictor(**model_config(width=128))
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        explicit = PositionPredictor(**model_config(width=128, gs_structure="none"))
        self.assertIsNone(explicit.gs_structure)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(explicit.state_dict()[key], value, atol=0, rtol=0)
        torch.manual_seed(42)
        structured = PositionPredictor(**model_config(width=128, gs_structure="anchor_attention"))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(structured.state_dict()[key], value, atol=0, rtol=0)
        added = set(structured.state_dict()) - set(baseline.state_dict())
        self.assertTrue(added and all(key.startswith("gs_structure.") for key in added))
        self.assertEqual(len(structured.gs_structure.blocks), 2)
        self.assertEqual(structured.gs_structure.output.in_features, 128)
        self.assertTrue(all(block.heads == 4 for block in structured.gs_structure.blocks))
        self.assertEqual(int(structured.gs_structure.output.weight.count_nonzero()), 0)
        self.assertEqual(int(structured.gs_structure.output.bias.count_nonzero()), 0)
        torch.nn.init.normal_(baseline.position_head[-1].weight, std=.02)
        structured.position_head.load_state_dict(baseline.position_head.state_dict())
        before = baseline(self.batch["inputs"])
        after = structured(self.batch["inputs"])
        for key in before:
            torch.testing.assert_close(after[key], before[key], atol=0, rtol=0)

    def test_structure_reuses_initial_field_and_live_positions_before_conditions(self):
        self.activate()
        x = self.batch["inputs"]
        seen = []
        hook = self.model.gs_structure.register_forward_pre_hook(lambda module, args: seen.append(args))
        try:
            with patch("scripts.training.model.build_initial_anchor_field", wraps=build_initial_anchor_field) as build:
                prediction = self.model(x)
        finally:
            hook.remove()
        self.assertEqual(build.call_count, 1)
        self.assertEqual(len(seen), 5)
        self.assertTrue(all(args[3] is seen[0][3] for args in seen))
        self.assertIsNone(seen[0][3].metric)
        for h, args in enumerate(seen):
            expected = x["gs_xyz"] if h == 0 else prediction["xyz"][:, h-1]
            torch.testing.assert_close(args[1], expected, atol=0, rtol=0)
        self.assertFalse(torch.equal(seen[0][1], seen[-1][1]))
        # The first structure block sees only initial GS. Changing even t0
        # conditions may affect the head, but must not alter this first input.
        changed = copy.deepcopy(x)
        changed["action_state"] += 10.
        changed["controller_points"] -= 5.
        changed["tactile"] += 3.
        after = []
        hook = self.model.gs_structure.register_forward_pre_hook(lambda module, args: after.append(args))
        try:
            self.model(changed)
        finally:
            hook.remove()
        torch.testing.assert_close(after[0][0], seen[0][0], atol=0, rtol=0)

    def test_final_horizon_reaches_both_structural_layers_and_prior_geometry(self):
        self.activate()
        seen, output = [], []
        hooks = [self.model.gs_structure.register_forward_pre_hook(lambda module, args: seen.append(args[1])),
                 self.model.position_head.register_forward_hook(lambda module, args, value: output.append(value))]
        try:
            result = self.model(self.batch["inputs"])
        finally:
            for hook in hooks:
                hook.remove()
        loss = result["xyz"][:, -1].square().sum()
        gradients = torch.autograd.grad(loss, seen[1:]+output, retain_graph=True)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.)
        loss.backward()
        for block in self.model.gs_structure.blocks:
            for parameter in (block.qkv.weight, block.relative[0].weight,
                              block.relative[-1].weight, block.ff[-1].weight):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertGreater(float(self.model.gs_encoder[0].weight.grad.abs().sum()), 0.)

    def test_relative_attention_direction_translation_and_anchor_permutation(self):
        layer = StructuralAnchorLayer(128, 4)
        features = torch.randn(1, 6, 128, requires_grad=True)
        xyz = torch.randn(1, 6, 3, requires_grad=True)
        keep = torch.ones(1, 6, dtype=torch.bool)
        output = layer(features, xyz, keep)
        moved = layer(features, xyz + torch.tensor([1., -2., 3.]), keep)
        torch.testing.assert_close(moved, output, atol=5e-7, rtol=2e-6)
        flipped = layer(features, -xyz, keep)
        self.assertFalse(torch.allclose(flipped, output))
        order = torch.tensor([4, 2, 5, 0, 3, 1])
        shuffled = layer(features[:, order], xyz[:, order], keep[:, order])
        torch.testing.assert_close(shuffled, output[:, order], atol=5e-7, rtol=2e-6)
        output.square().sum().backward()
        self.assertGreater(float(xyz.grad.abs().sum()), 0.)
        self.assertGreater(float(features.grad.abs().sum()), 0.)

    def test_small_variable_empty_and_nan_padding_do_not_change_rbf(self):
        for n, counts in ((6, [6, 3, 1, 0]), (0, [0, 0]), (1, [1, 0])):
            xyz = torch.randn(len(counts), n, 3)
            keep = torch.arange(n)[None] < torch.tensor(counts)[:, None]
            xyz[~keep] = float("nan")
            features = torch.randn(len(counts), n, 128)
            features[~keep] = float("nan")
            features.requires_grad_()
            xyz.requires_grad_()
            field = build_initial_anchor_field(xyz, keep, 128, 2)
            frozen_weights = field.weights.clone()
            block = StructuralGSBlock(128, 4)
            torch.nn.init.normal_(block.output.weight, std=.02)
            result = block(features, xyz, keep, field)
            self.assertEqual(result.shape, features.shape)
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue((result[~keep] == 0).all())
            torch.testing.assert_close(field.weights, frozen_weights, atol=0, rtol=0)
            result.square().sum().backward()
            self.assertTrue(torch.isfinite(features.grad).all())
            self.assertTrue((features.grad[~keep] == 0).all())
            if xyz.grad is not None:
                self.assertTrue(torch.isfinite(xyz.grad).all())
                self.assertTrue((xyz.grad[~keep] == 0).all())

    def test_invalid_configuration_rejects_free_or_anisotropic_combinations(self):
        for changes in (dict(gs_structure="other"),
                        dict(gs_structure="anchor_attention", gs_decoder="free"),
                        dict(gs_structure="anchor_attention", gs_decoder="anchors_anisotropic")):
            with self.assertRaises(ValueError):
                PositionPredictor(**model_config(**changes))
        with self.assertRaises(ValueError):
            StructuralAnchorLayer(127, 4)

    def test_full_rollout_accepts_zero_or_one_gaussian(self):
        self.activate()
        for n in (0, 1):
            x = copy.deepcopy(self.batch["inputs"])
            for key in ("gs_xyz", "gs_attr", "gs_mask"):
                x[key] = x[key][:, :n]
            result = self.model(x)
            self.assertEqual(result["xyz"].shape, (len(x["gs_xyz"]), 5, n, 3))
            self.assertTrue(all(torch.isfinite(value).all() for value in result.values()))


if __name__ == "__main__":
    unittest.main()
