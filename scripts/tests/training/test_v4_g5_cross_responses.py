"""CPU-only rigid action interventions; no optimizer or backward is used."""
import copy
import json
import unittest
import torch

from scripts.experiments.v4_g5_cross_necessity.responses import (
    RESPONSE_MODES, ResponseUnavailable, counterfactual_inputs)


def response_fixture(horizon=15):
    """Two samples: single right hand / both hands, with moving rigid tools."""
    dtype = torch.float64
    action = torch.zeros(2, horizon, 2, 13, dtype=dtype)
    controllers = torch.zeros(2, horizon, 2, 30, 3, dtype=dtype)
    rows, cols = torch.meshgrid(torch.arange(3), torch.arange(10), indexing="ij")
    local = torch.stack((rows.flatten()*.01, cols.flatten()*.01,
                         torch.full((30,), .03)), -1).to(dtype)
    for b in range(2):
        for h in range(horizon):
            for hand in range(2):
                sign = 1 if hand else -1
                angle = torch.tensor(.02*h*sign, dtype=dtype)
                rotation = torch.tensor([[angle.cos(), -angle.sin(), 0.],
                                         [angle.sin(), angle.cos(), 0.], [0., 0., 1.]], dtype=dtype)
                root = torch.tensor([.15+hand*.2+.001*h, -.03+.002*h*sign,
                                     .04+.003*h*sign], dtype=dtype)
                action[b,h,hand,:3] = root
                action[b,h,hand,3:12] = rotation.flatten()
                action[b,h,hand,12] = .08+.0001*h
                controllers[b,h,hand] = root+local@rotation.T
    sensor_center = controllers.reshape(2,horizon,4,15,3).mean(-2)
    tactile_xyz = sensor_center[...,None,None,:].expand(-1,-1,-1,12,32,-1).clone()
    valid = torch.ones(2,horizon,4,12,32,dtype=torch.bool)
    return dict(
        gs_xyz=torch.zeros(2,5,3,dtype=dtype), gs_mask=torch.ones(2,5,dtype=torch.bool),
        gs_attr=torch.zeros(2,5,11,dtype=dtype),
        action_state=action, action_valid=torch.ones(2,horizon,2,dtype=torch.bool),
        controller_points=controllers,
        controller_mask=torch.ones(2,horizon,2,30,dtype=torch.bool),
        operation_hands=torch.tensor([[False,True],[True,True]]),
        tactile=torch.zeros(2,horizon,4,12,32,dtype=dtype),
        valid_mask=valid, tactile_spatial_mask=valid.clone(), tactile_xyz=tactile_xyz,
        tactile_age_us=torch.full((2,horizon,4),12345,dtype=torch.int64),
        condition_frame_ids=torch.arange(horizon).expand(2,-1).clone()+135,
        arbitrary_metadata={"name":"not-a-model-identity", "values":[2,5]},
    )


class CounterfactualResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def assert_tree_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for key, value in first.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, second[key], rtol=0, atol=0, equal_nan=True)
            else:
                self.assertEqual(value, second[key])

    def test_both_modes_do_not_mutate_input_or_change_unrelated_fields(self):
        x = response_fixture()
        before = copy.deepcopy(x)
        changed = {"action_state","action_valid","controller_points","controller_mask","tactile_xyz"}
        for mode in RESPONSE_MODES:
            with self.subTest(mode=mode):
                result, stats = counterfactual_inputs(x,mode)
                self.assert_tree_equal(before,x)
                for key in x.keys()-changed:
                    if isinstance(x[key],torch.Tensor):
                        torch.testing.assert_close(result[key],x[key],rtol=0,atol=0)
                        self.assertNotEqual(result[key].data_ptr(),x[key].data_ptr())
                    else:
                        self.assertEqual(result[key],x[key])
                json.dumps(stats,allow_nan=False)
                self.assertFalse(stats["targets_read"])

    def test_freeze_reuses_initial_complete_rigid_state_and_sensor_geometry(self):
        x = response_fixture()
        out, stats = counterfactual_inputs(x,"FreezeAction")
        allowed = x["operation_hands"][:,None].expand_as(out["action_valid"])
        expected_action = x["action_state"][:,:1].expand_as(x["action_state"])
        torch.testing.assert_close(out["action_state"][allowed],expected_action[allowed],rtol=0,atol=0)
        cm = out["controller_mask"]
        expected_cp = x["controller_points"][:,:1].expand_as(x["controller_points"])
        torch.testing.assert_close(out["controller_points"][cm],expected_cp[cm],rtol=0,atol=0)
        allowed_t = x["operation_hands"][:,None].repeat_interleave(2,-1)[...,None,None]
        allowed_t = allowed_t.expand_as(x["tactile_spatial_mask"])
        expected_xyz = x["tactile_xyz"][:,:1].expand_as(x["tactile_xyz"])
        torch.testing.assert_close(out["tactile_xyz"][allowed_t],expected_xyz[allowed_t],rtol=0,atol=0)
        self.assertLess(stats["relative_root_controller_max_abs_error"],1e-15)

    def test_half_root_motion_preserves_rotation_opening_and_current_tool_shape(self):
        x = response_fixture()
        out, stats = counterfactual_inputs(x,"HalfRootTranslation")
        valid = out["action_valid"]
        first = x["action_state"][:,:1,:,:3]
        expected = first+.5*(x["action_state"][...,:3]-first)
        torch.testing.assert_close(out["action_state"][...,:3][valid],expected[valid],rtol=0,atol=1e-16)
        torch.testing.assert_close(out["action_state"][...,3:][valid],x["action_state"][...,3:][valid],rtol=0,atol=0)
        before_relative = x["controller_points"]-x["action_state"][...,:3].unsqueeze(-2)
        after_relative = out["controller_points"]-out["action_state"][...,:3].unsqueeze(-2)
        torch.testing.assert_close(before_relative[out["controller_mask"]],after_relative[out["controller_mask"]],rtol=0,atol=1e-16)
        self.assertLess(stats["relative_root_controller_max_abs_error"],1e-15)

    def test_half_sensor_xyz_uses_same_hand_translation_not_pressure_scaling(self):
        x = response_fixture()
        x["tactile"][:,2:] = .75
        out,_ = counterfactual_inputs(x,"HalfRootTranslation")
        shift = out["action_state"][...,:3]-x["action_state"][...,:3]
        shift = shift.repeat_interleave(2,2)[...,None,None,:]
        allowed = x["operation_hands"][:,None].repeat_interleave(2,-1)[...,None,None]
        allowed = allowed.expand_as(x["tactile_spatial_mask"])
        torch.testing.assert_close(out["tactile_xyz"][allowed],(x["tactile_xyz"]+shift)[allowed],rtol=0,atol=1e-16)
        torch.testing.assert_close(out["tactile"],x["tactile"],rtol=0,atol=0)

    def test_h1_exact_for_supported_single_and_dual_hand(self):
        for horizon in (1,15):
            for mode in RESPONSE_MODES:
                x = response_fixture(horizon)
                out, stats = counterfactual_inputs(x,mode)
                av = out["action_valid"][:,0]
                cm = out["controller_mask"][:,0]
                torch.testing.assert_close(out["action_state"][:,0][av],x["action_state"][:,0][av],rtol=0,atol=0)
                torch.testing.assert_close(out["controller_points"][:,0][cm],x["controller_points"][:,0][cm],rtol=0,atol=0)
                self.assertTrue(stats["initial_valid_action_bit_exact"])
                self.assertTrue(stats["initial_valid_controller_bit_exact"])

    def test_masks_do_not_revive_and_freeze_intersects_endpoints(self):
        x = response_fixture(5)
        x["action_valid"][1,0,0] = False
        # No initial observation is removed by Half when this Root and its
        # Controllers are already absent together.  Root-only absence with
        # observed Controllers is tested separately as NOT_APPLICABLE.
        x["controller_mask"][1,0,0] = False
        x["action_valid"][1,2,1] = False
        x["controller_mask"][1,0,1,4] = False
        x["controller_mask"][1,3,1,5] = False
        for mode in RESPONSE_MODES:
            out,stats = counterfactual_inputs(x,mode)
            self.assertFalse(bool((out["action_valid"] & ~x["action_valid"]).any()))
            self.assertFalse(bool((out["controller_mask"] & ~x["controller_mask"]).any()))
            self.assertFalse(bool(out["action_valid"][1,:,0].any()))
            if mode=="FreezeAction":
                self.assertFalse(bool(out["controller_mask"][1,:,1,4].any()))
            else:
                self.assertFalse(bool(out["controller_mask"][1,:,0].any()))
                self.assertTrue(bool(out["controller_mask"][1,1,1,4]))
            self.assertGreater(stats["root_endpoint_unavailable_hands"],0)

    def test_initial_root_missing_but_controllers_present_is_not_applicable(self):
        x = response_fixture(5)
        x["action_valid"][0,0,1] = False
        before = copy.deepcopy(x)
        with self.assertRaises(ResponseUnavailable) as raised:
            counterfactual_inputs(x,"HalfRootTranslation")
        self.assert_tree_equal(x,before)
        stats = raised.exception.diagnostics
        self.assertEqual(stats["status"],"NOT_APPLICABLE")
        self.assertFalse(stats["counterfactual_constructed"])
        self.assertTrue(stats["inputs_unchanged"])
        self.assertTrue(stats["h1_action_support_bit_exact"])
        self.assertFalse(stats["h1_controller_support_bit_exact"])
        self.assertEqual(stats["h1_controller_observations_before"]-
                         stats["h1_controller_observations_after"],30)
        self.assertEqual(len(stats["h1_controller_support_changes"]),30)
        json.dumps(stats,allow_nan=False)

    def test_nonfinite_initial_root_with_observed_controllers_is_not_applicable(self):
        x = response_fixture(3)
        x["action_state"][1,0,1,4] = float('nan')
        before = copy.deepcopy(x)
        with self.assertRaises(ResponseUnavailable) as raised:
            counterfactual_inputs(x,"HalfRootTranslation")
        self.assert_tree_equal(x,before)
        self.assertFalse(raised.exception.diagnostics["h1_controller_support_bit_exact"])

    def test_missing_nonoperating_hand_does_not_trigger_rejection(self):
        x = response_fixture(5)
        x["action_valid"][0,0,0] = False
        x["action_state"][0,0,0] = float('nan')
        # Left-hand Controllers may be present in the raw schema, but the
        # explicit episode operation mask excludes them from model support.
        self.assertTrue(bool(x["controller_mask"][0,0,0].all()))
        for mode in RESPONSE_MODES:
            out,stats = counterfactual_inputs(x,mode)
            self.assertEqual(stats["status"],"CONSTRUCTED")
            self.assertTrue(stats["h1_action_support_bit_exact"])
            self.assertTrue(stats["h1_controller_support_bit_exact"])
            self.assertTrue(bool(out["controller_mask"][0,0,1].all()))

    @torch.inference_mode()
    def test_missing_initial_root_freeze_preserves_real_h1_prediction(self):
        from scripts.data_process import DataProcessor
        from scripts.training.v4_runtime import prepare_batch
        from scripts.experiments.v4_stage1.fixtures import synthetic_batch
        from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
        torch.manual_seed(412)
        model = G5DirectPredictor(horizon=3).eval().requires_grad_(False)
        model.motion[-1].weight.normal_(std=.0005)
        model.global_motion.mapping[-1].weight.normal_(std=.0005)
        model.direct_motion.mapping[-1].weight.normal_(std=.0005)
        x = prepare_batch(synthetic_batch(24,3,1),DataProcessor("001-rope"),model)["inputs"]
        x["action_valid"][0,0,1] = False
        original = model(x)
        frozen,stats = counterfactual_inputs(x,"FreezeAction")
        self.assertTrue(stats["h1_controller_support_bit_exact"])
        initial_support = (x["controller_mask"][:,0] & x["operation_hands"][...,None]
                           & torch.isfinite(x["controller_points"][:,0]).all(-1))
        torch.testing.assert_close(frozen["controller_mask"][:,0],initial_support,rtol=0,atol=0)
        actual = model(frozen)
        for key in original:
            torch.testing.assert_close(actual[key][:,0],original[key][:,0],rtol=0,atol=0)
        with self.assertRaises(ResponseUnavailable):
            counterfactual_inputs(x,"HalfRootTranslation")

    def test_invalid_nan_inputs_are_sanitized_without_pressure_mask_change(self):
        x = response_fixture(5)
        x["action_state"][0,:,0] = float('nan')
        x["controller_points"][0,:,0] = float('nan')
        x["action_state"][1,2,1,4] = float('nan')
        x["controller_points"][1,3,1,7] = float('nan')
        x["tactile_xyz"][1,0,3,2,1] = float('nan')
        x["tactile_xyz"][1,4,3,2,2] = float('nan')
        for mode in RESPONSE_MODES:
            out,stats = counterfactual_inputs(x,mode)
            for key in ("action_state","controller_points","tactile_xyz"):
                self.assertTrue(bool(torch.isfinite(out[key]).all()))
            self.assertFalse(bool(out["action_valid"][0,:,0].any()))
            self.assertFalse(bool(out["controller_mask"][0,:,0].any()))
            self.assertFalse(bool(out["action_valid"][1,2,1]))
            for key in ("tactile","valid_mask","tactile_spatial_mask","tactile_age_us"):
                torch.testing.assert_close(out[key],x[key],rtol=0,atol=0)
            self.assertGreater(stats["tactile_xyz_geometry"]["unsupported_positions_after"],0)
            self.assertEqual(float(out["action_state"][0,:,0].abs().max()),0.)

    def test_future_action_changes_cannot_modify_earlier_counterfactuals(self):
        x = response_fixture(5)
        later = copy.deepcopy(x)
        later["action_state"][:,3:,:,:3] += .1
        later["controller_points"][:,3:] += .1
        later["tactile_xyz"][:,3:] += .1
        for mode in RESPONSE_MODES:
            a,_ = counterfactual_inputs(x,mode)
            b,_ = counterfactual_inputs(later,mode)
            for key in ("action_state","controller_points","action_valid","controller_mask","tactile_xyz"):
                torch.testing.assert_close(a[key][:,:3],b[key][:,:3],rtol=0,atol=0)

    def test_valid_zero_pressure_remains_valid_zero_not_missing(self):
        x = response_fixture(5)
        for mode in RESPONSE_MODES:
            out,_ = counterfactual_inputs(x,mode)
            self.assertTrue(torch.equal(out["tactile"],torch.zeros_like(out["tactile"])))
            self.assertTrue(torch.equal(out["valid_mask"],x["valid_mask"]))
            self.assertTrue(bool(out["valid_mask"].all()))
            self.assertTrue(torch.equal(out["tactile_age_us"],x["tactile_age_us"]))

    def test_optional_sensor_xyz_can_be_absent(self):
        x = response_fixture(3)
        del x["tactile_xyz"]
        for mode in RESPONSE_MODES:
            out,stats = counterfactual_inputs(x,mode)
            self.assertNotIn("tactile_xyz",out)
            self.assertFalse(stats["tactile_xyz_geometry"]["present"])

    @torch.inference_mode()
    def test_ancillary_world_xyz_changes_do_not_change_real_v4_model_output(self):
        from scripts.data_process import DataProcessor
        from scripts.training.v4_runtime import prepare_batch
        from scripts.experiments.v4_stage1.fixtures import synthetic_batch
        from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
        torch.manual_seed(412)
        model = G5DirectPredictor(horizon=3).eval().requires_grad_(False)
        # Nonzero diagnostic output layers: equality must not be a vacuous
        # consequence of all three zero-initialized displacement heads.
        model.motion[-1].weight.normal_(std=.0005)
        model.global_motion.mapping[-1].weight.normal_(std=.0005)
        model.direct_motion.mapping[-1].weight.normal_(std=.0005)
        x = prepare_batch(synthetic_batch(24,3,1),DataProcessor("001-rope"),model)["inputs"]
        for mode in RESPONSE_MODES:
            changed,_ = counterfactual_inputs(x,mode)
            retained = copy.deepcopy(changed)
            retained["tactile_xyz"] = x["tactile_xyz"].clone()
            self.assertFalse(torch.equal(changed["tactile_xyz"],retained["tactile_xyz"]))
            a,b = model(changed),model(retained)
            self.assertFalse(torch.equal(a["xyz"][:,0],x["gs_xyz"]))
            for key in a:
                torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_reject_unknown_modes_and_misaligned_schema(self):
        with self.assertRaises(ValueError):
            counterfactual_inputs(response_fixture(2),"zero_future_target")
        x = response_fixture(2)
        x["controller_points"] = x["controller_points"].reshape(2,2,60,3)
        with self.assertRaises(ValueError):
            counterfactual_inputs(x,"FreezeAction")
        x = response_fixture(2)
        x["controller_mask"] = x["controller_mask"].float()
        with self.assertRaises(ValueError):
            counterfactual_inputs(x,"FreezeAction")


if __name__ == '__main__':
    unittest.main()
