"""Disk-free checks for known-window diagnostics, not real CUDA evidence."""

import copy
from io import BytesIO
import unittest
from unittest.mock import patch

import torch

from .model import PositionPredictor
from .oracle import OracleLatent, OracleAnchor, ORACLE_VERSION
from .train import checkpoint_config


def inputs(count=7, valid=5):
    xyz = torch.arange(count * 3, dtype=torch.float32).reshape(1, count, 3) * .02
    keep = torch.arange(count)[None] < valid
    attr = torch.zeros(1, count, 11)
    attr[..., :3] = -4.
    attr[..., 3] = 1.
    xyz[~keep], attr[~keep] = float("nan"), float("nan")
    return dict(gs_xyz=xyz, gs_attr=attr, gs_mask=keep)


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.base = PositionPredictor(width=32, heads=4, layers=1,
            output_mode="gs_geometry", gs_decoder="anchors", gs_local_attention="knn",
            gs_local_k=3, anchor_count=4, anchor_pool_k=2)
        self.windows = [inputs(), inputs(3, 3)]
        self.masks = [x["gs_mask"] for x in self.windows]

    def active(self):
        model = OracleLatent(self.base, self.masks)
        torch.nn.init.normal_(model.base.position_head[-1].weight, std=.03)
        return model

    def test_zero_initialization_copy_and_attribute_inheritance(self):
        for cls in (OracleLatent, OracleAnchor):
            model = cls(self.base, self.masks)
            for index, x in enumerate(self.windows):
                pred = model(x, index)
                expected = torch.nan_to_num(x["gs_xyz"])[:, None].expand(-1, 5, -1, -1)
                torch.testing.assert_close(pred["xyz"], expected, atol=0, rtol=0)
                for key, columns in (("log_scales", slice(0, 3)), ("quaternions", slice(3, 7)),
                                     ("opacity_logits", slice(7, 8)), ("sh_dc", slice(8, 11))):
                    attr = torch.nan_to_num(x["gs_attr"])[..., columns][:, None].expand_as(pred[key])
                    torch.testing.assert_close(pred[key], attr, atol=0, rtol=0)
                self.assertTrue(all(torch.isfinite(v).all() for v in pred.values()))

    def test_conditions_and_targets_never_enter_oracle_forward(self):
        model = self.active()
        with patch.object(model.base, "encode_conditions", side_effect=AssertionError("condition leak")):
            actual = model(self.windows[0], 0)
            changed = dict(self.windows[0], action_state=torch.full((1, 5, 2, 13), float("nan")),
                           tactile=torch.randn(1, 5, 4, 12, 32), targets={"xyz": torch.randn(500, 3)})
            perturbed = model(changed, 0)
        for key in actual:
            torch.testing.assert_close(actual[key], perturbed[key], atol=0, rtol=0)
        used = {n for n, p in model.named_parameters() if p.requires_grad}
        for prefix in ("base.state_encoder", "base.controller_encoder", "base.tactile_encoder",
                       "base.blocks", "base.condition_time", "base.token_identity", "base.null_token"):
            self.assertFalse(any(n.startswith(prefix) for n in used), prefix)
        self.assertTrue(all(p.requires_grad for p in self.base.parameters()))
        self.assertTrue(any(n.startswith("base.gs_local") for n in used))

    def test_last_horizon_has_live_ar_and_encoder_gradients(self):
        model = self.active()
        received = []
        def retain(module, args):
            if args[0].requires_grad:
                args[0].retain_grad()
                received.append(args[0])
        hook = model.base.gs_encoder.register_forward_pre_hook(retain)
        pred = model(self.windows[0], 0)
        pred["xyz"][:, -1].square().sum().backward()
        hook.remove()
        self.assertEqual(len(received), 4)
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all()
                            and v.grad.abs().sum() > 0 for v in received))
        for h in range(5):
            self.assertGreater(float(model.variables[0].grad[h].abs().sum()), 0.)
        self.assertIsNone(model.variables[1].grad)
        self.assertGreater(float(model.base.gs_encoder[0].weight.grad.abs().sum()), 0.)

    def test_independent_window_adamw_state_is_not_advanced(self):
        model = self.active()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=.01, weight_decay=.1)
        for window in (0, 1):
            optimizer.zero_grad(set_to_none=True)
            model(self.windows[window], window)["xyz"].square().sum().backward()
            optimizer.step()
        parameter = model.variables[1]
        before = parameter.detach().clone()
        previous = copy.deepcopy(optimizer.state[parameter])
        optimizer.zero_grad(set_to_none=True)
        model(self.windows[0], 0)["xyz"].square().sum().backward()
        self.assertIsNone(parameter.grad)
        optimizer.step()
        torch.testing.assert_close(parameter, before, atol=0, rtol=0)
        for key in previous:
            torch.testing.assert_close(optimizer.state[parameter][key], previous[key], atol=0, rtol=0)

    def test_anchor_translation_is_cumulative_not_step_increment(self):
        model = OracleAnchor(self.base, self.masks)
        shifts = torch.arange(1, 6)[:, None] * torch.tensor([.01, -.02, .03])[None]
        with torch.no_grad():
            model.variables[0].copy_(shifts[:, None].expand_as(model.variables[0]))
        result = model(self.windows[0], 0)["xyz"]
        mask = self.windows[0]["gs_mask"][0]
        target = self.windows[0]["gs_xyz"][:, None, mask] + shifts[None, :, None]
        torch.testing.assert_close(result[:, :, mask], target)
        self.assertTrue((result[:, :, ~mask] == 0).all())
        result.square().sum().backward()
        self.assertGreater(float(model.variables[0].grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))

    def test_latent_head_also_references_initial_each_horizon(self):
        model = OracleLatent(self.base, self.masks)
        shifts = torch.arange(1, 6)[:, None] * torch.tensor([.01, -.02, .03])[None]
        count = 0
        def fixed(query):
            nonlocal count
            value = shifts[count].expand(*query.shape[:2], 3)
            count += 1
            return value
        with patch.object(model.base.position_head, "forward", side_effect=fixed):
            result = model(self.windows[0], 0)["xyz"]
        keep = self.masks[0][0]
        expected = self.windows[0]["gs_xyz"][:, None, keep] + shifts[None, :, None]
        torch.testing.assert_close(result[:, :, keep], expected)

    def test_invalid_and_empty_windows_remain_finite_and_masked(self):
        for x in (inputs(4, 0), inputs(0, 0), inputs(1, 1)):
            for cls in (OracleLatent, OracleAnchor):
                model = cls(self.base, [x["gs_mask"]])
                result = model(x, 0)
                self.assertEqual(result["xyz"].shape, (1, 5, x["gs_xyz"].shape[1], 3))
                self.assertTrue(all(torch.isfinite(v).all() for v in result.values()))
                self.assertTrue((result["xyz"][:, :, ~x["gs_mask"][0]] == 0).all())
        model = self.active()
        changed = copy.deepcopy(self.windows[0])
        changed["gs_mask"][0, 0] = False
        with self.assertRaises(ValueError):
            model(changed, 0)
        with self.assertRaises(ValueError):
            model(self.windows[0], 2)
        changed = copy.deepcopy(self.windows[0])
        changed["gs_xyz"][0, 0, 0] = float("inf")
        with self.assertRaises(ValueError):
            model(changed, 0)

    def test_diagnostic_checkpoint_roundtrip_optimizer_and_production_rejection(self):
        model = self.active()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        def step(module, opt):
            opt.zero_grad(set_to_none=True)
            module(self.windows[0], 0)["xyz"].square().sum().backward()
            opt.step()
        step(model, optimizer)
        stream = BytesIO()
        torch.save(model.diagnostic_checkpoint(optimizer, completed_updates=1), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=False)
        self.assertEqual(saved["version"], ORACLE_VERSION)
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint"):
            checkpoint_config(saved)
        restored = OracleLatent(self.base, self.masks)
        restored_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, restored.parameters()), lr=1e-4)
        self.assertEqual(restored.load_diagnostic_checkpoint(saved, restored_optimizer)["completed_updates"], 1)
        step(model, optimizer)
        step(restored, restored_optimizer)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            OracleAnchor(self.base, self.masks).load_diagnostic_checkpoint(saved)
        wrong = copy.deepcopy(saved)
        wrong["model"]["window_mask_0"][0] = False
        with self.assertRaisesRegex(ValueError, "mask mismatch"):
            restored.load_diagnostic_checkpoint(wrong)


if __name__ == "__main__":
    unittest.main()
