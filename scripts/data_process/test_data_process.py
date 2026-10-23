"""Disk-free tests: python -B -m unittest scripts.data_process.test_data_process -v"""

import copy
from io import StringIO
import json
import unittest
from unittest.mock import patch

import torch

from ..dataloader.check import check_batch
from ..dataloader.collate import collate_fn
from ..dataloader.dataset import Touch4DDataset
from ..dataloader.test_dataloader import MemoryReconstruction
from .check import check_processed_batch, main as check_main, require_unchanged
from .processor import DataProcessor, SUPPORTED_SCENES


def make_batch(scene="001-rope", split="train"):
    source = MemoryReconstruction(scene)
    with source.mounted():
        dataset = Touch4DDataset(scene, split, source.root)
        batch = collate_fn([dataset[0], dataset[-1]])
    # Physically sized coordinates, including a valid world-origin point.
    x = batch["inputs"]
    x["gs_xyz"] *= .01
    x["controller_points"] *= .01
    x["tactile_xyz"] *= .01
    x["action_state"][..., :3] *= .01
    for target in batch["targets"]["target_xyz"]:
        target *= .01
    x["gs_xyz"][0, 0] = 0
    # Raw opening outside geometry's clip interval must still be preserved.
    x["action_state"][0, 0, 1, 12] = .2
    # A nonzero 2D reading for the missing left hand must survive separately.
    x["tactile"][0, 0, 0, 0, 0] = 23
    x["tactile"][0, 0, 2, 0, :4] = torch.tensor([0., .5, 100., 226.])
    check_batch(batch)
    return batch


def add_depth_targets(batch):
    """Tiny metric images and static cameras; no files or renderer required."""
    b = len(batch["meta"]["scene"])
    depth = torch.full((b, 5, 2, 3, 4), .5, dtype=torch.float32)
    depth += torch.arange(5, dtype=torch.float32)[None, :, None, None, None] * .01
    depth[..., -1, -1] = 0
    k = torch.tensor([[100., 0., 1.5], [0., 110., 1.], [0., 0., 1.]])
    c2w = torch.eye(4).repeat(b, 2, 1, 1)
    c2w[..., :3, 3] = torch.tensor([.2, -.1, .3])
    batch["targets"].update(target_depth=depth, target_depth_valid=depth > 0,
                            camera_intrinsics=k.repeat(b, 2, 1, 1), camera_to_world=c2w)
    batch["meta"]["depth_camera_ids"] = [["cam_a", "cam_b"] for _ in range(b)]
    return batch


class ProcessorTests(unittest.TestCase):
    def test_depth_and_world_cameras_remain_metric_and_independent(self):
        for scene in SUPPORTED_SCENES:
            batch = add_depth_targets(make_batch(scene))
            saved = copy.deepcopy(batch)
            p = DataProcessor(scene)
            out = p(batch)
            check_processed_batch(batch, out, p)
            require_unchanged(saved, batch)
            for name in ("target_depth", "target_depth_valid", "camera_intrinsics", "camera_to_world"):
                self.assertTrue(torch.equal(out["targets"][name], batch["targets"][name]))
                self.assertNotEqual(out["targets"][name].data_ptr(), batch["targets"][name].data_ptr())
            self.assertEqual(out["meta"]["depth_camera_ids"], batch["meta"]["depth_camera_ids"])
            out["targets"]["target_depth"].fill_(9)
            out["meta"]["depth_camera_ids"][0][0] = "changed"
            require_unchanged(saved, batch)

    def test_restored_prediction_projects_into_metric_depth_camera(self):
        batch = add_depth_targets(make_batch())
        p = DataProcessor("001-rope")
        target = batch["targets"]
        k, c2w = target["camera_intrinsics"][0, 0], target["camera_to_world"][0, 0]
        pixel = torch.tensor([2., 1., 1.])
        camera_point = torch.linalg.solve(k, pixel) * target["target_depth"][0, 0, 0, 1, 2]
        world = c2w[:3, :3] @ camera_point + c2w[:3, 3]
        batch["inputs"]["gs_xyz"][0, 0] = world
        normalized = p(batch)
        predicted = normalized["inputs"]["gs_xyz"].clone().requires_grad_()
        restored = p.denormalize_xyz(predicted, normalized["inputs"]["gs_mask"])
        projected = torch.linalg.inv(c2w) @ torch.cat((restored[0, 0], torch.ones(1)))
        torch.testing.assert_close(projected[2], target["target_depth"][0, 0, 0, 1, 2])
        projected[2].backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertGreater(predicted.grad[0, 0].abs().sum().item(), 0)

    def test_depth_semantic_errors_fail_before_normalization(self):
        mutations = [
            lambda b: b["targets"].pop("target_depth_valid"),
            lambda b: b["targets"].update(target_depth=b["targets"]["target_depth"].double()),
            lambda b: b["targets"]["target_depth"].fill_(float("nan")),
            lambda b: b["targets"]["target_depth_valid"].fill_(True),
            lambda b: b["targets"]["camera_to_world"][..., 3, 3].fill_(0),
            lambda b: b["targets"]["camera_intrinsics"][..., 0, 0].fill_(0),
            lambda b: b["meta"].pop("depth_camera_ids"),
        ]
        for change in mutations:
            with self.subTest(mutation=change):
                batch = add_depth_targets(make_batch())
                change(batch)
                with self.assertRaises(ValueError):
                    DataProcessor("001-rope")(batch)

    def test_frozen_parameters_and_checkpoint_roundtrip(self):
        expected = {
            "001-rope": ([.210834337570, .013140156253, .157411820790], .157037090510),
            "051-cube": ([.236388585803, .023145916698, .103283534840], .042746614739),
        }
        self.assertEqual(SUPPORTED_SCENES, tuple(expected))
        for scene, (center, scale) in expected.items():
            p = DataProcessor(scene)
            state = p.state_dict()
            self.assertEqual(state["center"], center)
            self.assertEqual(state["scale"], scale)
            self.assertEqual(state["tactile_scale"], 100.)
            self.assertEqual(DataProcessor.from_state_dict(json.loads(json.dumps(state))), p)
            state["center"][0] += 1
            self.assertEqual(p.state_dict()["center"], center)
            with self.assertRaisesRegex(ValueError, "parameters"):
                DataProcessor.from_state_dict(state)
        for scene in ("008-pink-cloth", "001", "051", None):
            with self.assertRaises(ValueError):
                DataProcessor(scene)

    def test_shared_affine_transform_for_all_conditions_and_five_targets(self):
        for scene in SUPPORTED_SCENES:
            p = DataProcessor(scene)
            batch = make_batch(scene)
            center = torch.tensor(p.state_dict()["center"])
            scale = p.state_dict()["scale"]
            world = center + scale*torch.tensor([1., 2., 3.])
            x = batch["inputs"]
            x["gs_xyz"][0, 0] = world
            x["controller_points"][0, 0, 1, 0] = world
            x["tactile_xyz"][0, 0, 2, 0, 0] = world
            x["action_state"][0, 0, 1, :3] = world
            for target in batch["targets"]["target_xyz"]:
                target[0, 0] = world
            out = p(batch)
            y = out["inputs"]
            actual = [y["gs_xyz"][0, 0], y["controller_points"][0, 0, 1, 0],
                      y["tactile_xyz"][0, 0, 2, 0, 0], y["action_state"][0, 0, 1, :3]]
            actual += [target[0, 0] for target in out["targets"]["target_xyz"]]
            for value in actual:
                torch.testing.assert_close(value, torch.tensor([1., 2., 3.]), atol=1e-6, rtol=0)
            check_processed_batch(batch, out, p)

    def test_roundtrip_same_structure_dtypes_and_point_counts_for_both_scenes(self):
        for scene in SUPPORTED_SCENES:
            batch = make_batch(scene)
            p = DataProcessor(scene)
            out = p(batch)
            metrics = check_processed_batch(batch, out, p)
            self.assertLess(max(metrics["roundtrip_max_abs_errors"].values()), 2e-5)
            self.assertEqual([a.shape for a in out["targets"]["target_xyz"]],
                             [a.shape for a in batch["targets"]["target_xyz"]])
            self.assertTrue(torch.equal(out["inputs"]["gs_lengths"], batch["inputs"]["gs_lengths"]))

    def test_physical_scales_opening_and_unchanged_orientation_appearance(self):
        p = DataProcessor("051-cube")
        batch = make_batch("051-cube")
        x, y = batch["inputs"], p(batch)["inputs"]
        mask, scale = x["gs_mask"], p.state_dict()["scale"]
        torch.testing.assert_close(y["gs_attr"][..., :3][mask].exp(), x["gs_attr"][..., :3][mask].exp()/scale)
        self.assertTrue(torch.equal(y["gs_attr"][..., 3:], x["gs_attr"][..., 3:]))
        self.assertTrue(torch.equal(y["action_state"][..., 3:12], x["action_state"][..., 3:12]))
        self.assertAlmostEqual(y["action_state"][0, 0, 1, 12].item(), .2/scale, places=5)

    def test_sparse_tactile_zeros_ratios_no_clipping_and_independent_2d_mask(self):
        p = DataProcessor("001-rope")
        batch = make_batch()
        x, y = batch["inputs"], p(batch)["inputs"]
        torch.testing.assert_close(y["tactile"][0, 0, 2, 0, :4], torch.tensor([0., .005, 1., 2.26]))
        self.assertTrue(y["valid_mask"][0, 0, 2, 0, 0])
        self.assertFalse(y["tactile_spatial_mask"][0, 0, 0, 0, 0])
        self.assertTrue(y["valid_mask"][0, 0, 0, 0, 0])
        self.assertAlmostEqual(y["tactile"][0, 0, 0, 0, 0].item(), .23)
        self.assertTrue(torch.all(y["tactile_xyz"][0, :, :2] == 0))
        self.assertEqual(int((x["tactile"] > 0).sum()), int((y["tactile"] > 0).sum()))

    def test_real_origin_and_zero_normalized_location_are_both_valid(self):
        p = DataProcessor("001-rope")
        batch = make_batch()
        x = batch["inputs"]
        x["gs_xyz"][0, 0] = torch.tensor(p.state_dict()["center"])
        out = p(batch)
        self.assertTrue(out["inputs"]["gs_mask"][0, 0])
        self.assertTrue(torch.all(out["inputs"]["gs_xyz"][0, 0] == 0))
        recovered = p.denormalize_batch(out)
        torch.testing.assert_close(recovered["inputs"]["gs_xyz"][0, 0], x["gs_xyz"][0, 0])
        x["gs_xyz"][0, 0] = 0
        out = p(batch)
        self.assertTrue(out["inputs"]["gs_mask"][0, 0])
        self.assertTrue(torch.any(out["inputs"]["gs_xyz"][0, 0] != 0))

    def test_all_invalid_conditions_remain_zero_in_both_directions(self):
        batch = make_batch()
        x = batch["inputs"]
        for field, mask in (("action_state", "action_valid"), ("controller_points", "controller_mask"),
                            ("tactile_xyz", "tactile_spatial_mask"), ("tactile", "valid_mask")):
            x[field].zero_()
            x[mask].fill_(False)
        p = DataProcessor("001-rope")
        out = p(batch)
        check_processed_batch(batch, out, p)
        check_batch(p.denormalize_batch(out))
        self.assertTrue(torch.all(out["inputs"]["gs_attr"][~x["gs_mask"]] == 0))

    def test_input_storage_and_metadata_are_independent(self):
        batch = make_batch()
        saved = copy.deepcopy(batch)
        p = DataProcessor("001-rope")
        out = p(batch)
        require_unchanged(saved, batch)
        restored = p.denormalize_batch(out)
        out_saved = copy.deepcopy(out)
        restored["inputs"]["gs_xyz"].fill_(99)
        require_unchanged(out_saved, out)
        out["inputs"]["valid_mask"].fill_(False)
        out["meta"]["frame_ids"].fill_(-1)
        out["meta"]["scene"][0] = "changed"
        out["targets"]["target_xyz"][0].fill_(99)
        require_unchanged(saved, batch)

    def test_inverse_predictions_and_displacements_have_different_offsets(self):
        p = DataProcessor("051-cube")
        values = torch.zeros(2, 5, 3, 3, dtype=torch.float64)
        mask = torch.ones(2, 5, 3, dtype=torch.bool)
        mask[0, :, 2] = False
        xyz = p.denormalize_xyz(values, mask)
        delta = p.denormalize_displacement(values, mask)
        torch.testing.assert_close(xyz[mask], torch.tensor(p.state_dict()["center"], dtype=torch.float64).expand(int(mask.sum()), 3))
        self.assertTrue(torch.all(xyz[~mask] == 0) and torch.all(delta == 0))
        ones = p.denormalize_displacement(torch.ones_like(values), mask)
        torch.testing.assert_close(ones[mask], torch.full_like(ones[mask], p.state_dict()["scale"]))
        with self.assertRaisesRegex(ValueError, "dimension"):
            p.denormalize_xyz(values, mask[:, 0])

    def test_autograd_survives_cloning_and_masked_inverse(self):
        p = DataProcessor("001-rope")
        batch = make_batch()
        initial = batch["inputs"]["gs_xyz"].requires_grad_()
        normalized = p(batch)
        normalized["inputs"]["gs_xyz"].sum().backward()
        mask = batch["inputs"]["gs_mask"]
        expected = mask[..., None].expand_as(initial).float()/p.state_dict()["scale"]
        torch.testing.assert_close(initial.grad, expected)
        prediction = torch.ones(2, 5, 3, 3, requires_grad=True)
        valid = torch.ones(2, 5, 3, dtype=torch.bool)
        valid[0] = False
        p.denormalize_displacement(prediction, valid).sum().backward()
        torch.testing.assert_close(prediction.grad,
                                   valid[..., None].expand_as(prediction).float()*p.state_dict()["scale"])

    def test_fixed_parameters_for_train_val_test_and_batch_composition(self):
        for scene in SUPPORTED_SCENES:
            p = DataProcessor(scene)
            parameters = p.state_dict()
            for split in ("train", "val", "test"):
                batch = make_batch(scene, split)
                check_processed_batch(batch, p(batch), p)
                self.assertEqual(p.state_dict(), parameters)
            batch = make_batch(scene)
            a = p(batch)["inputs"]["gs_xyz"][0].clone()
            batch["inputs"]["gs_xyz"][1] *= 20
            torch.testing.assert_close(p(batch)["inputs"]["gs_xyz"][0], a, atol=0, rtol=0)

    def test_scene_shape_dtype_and_mask_mismatches_fail(self):
        p = DataProcessor("001-rope")
        mutations = [
            lambda b: b["meta"]["scene"].__setitem__(0, "051-cube"),
            lambda b: b["meta"].update(scene=[]),
            lambda b: b["targets"]["target_xyz"].pop(),
            lambda b: b["inputs"].update(gs_xyz=b["inputs"]["gs_xyz"].double()),
            lambda b: b["inputs"].update(gs_mask=b["inputs"]["gs_mask"].float()),
            lambda b: b["inputs"].update(action_state=b["inputs"]["action_state"][..., :12]),
            lambda b: b["inputs"].pop("tactile_spatial_mask"),
        ]
        for change in mutations:
            with self.subTest(mutation=change):
                batch = make_batch()
                change(batch)
                with self.assertRaises(ValueError):
                    p(batch)

    def test_acceptance_detects_incorrect_forward_and_mask_changes(self):
        p = DataProcessor("001-rope")
        batch = make_batch()
        out = p(batch)
        out["inputs"]["action_state"][0, 0, 1, 12] = .112/p.state_dict()["scale"]
        with self.assertRaises(AssertionError):
            check_processed_batch(batch, out, p)
        out = p(batch)
        out["inputs"]["gs_attr"][..., 7] += .01
        with self.assertRaises((AssertionError, ValueError)):
            check_processed_batch(batch, out, p)

    def test_acceptance_cli_uses_real_loader_pipeline_without_writes(self):
        for scene in SUPPORTED_SCENES:
            source = MemoryReconstruction(scene)
            with source.mounted(), patch("sys.stdout", new_callable=StringIO) as output:
                check_main(["--scene", scene, "--root", str(source.root), "--split", "val",
                            "--max-batches", "0", "--batch-size", "3"])
                report = json.loads(output.getvalue())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["normal_samples"], 4)
            self.assertEqual(report["checked_samples"], 6)
            self.assertEqual(report["checked_batches"], 3)
            self.assertEqual(report["boundary_batches"], 1)
            self.assertTrue(report["all_windows_checked"])


if __name__ == "__main__":
    unittest.main()
