"""Frozen synthetic contracts: no optimizer and no backward in this test suite."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from scripts.data_process import DataProcessor
from scripts.training.gs_decoder import interpolate_anchor_displacements
from scripts.training.model import masked
from scripts.training.rendering import GaussianRenderer
from scripts.training.v4_runtime import prepare_batch, v4_objective
from scripts.experiments.v4_stage1.fixtures import synthetic_batch, analytic_rasterizer
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash
from scripts.experiments.v4_g5_motion_attribution.interventions import (
    MODES, intervention, switches, combine, branch_parts, same_history_position, one_frame)
from scripts.experiments.v4_g5_motion_attribution.metrics import (
    decompose, angle_stats, cancellation_stats, aggregate)


class FrozenMotionAttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def fixture(self, horizon=15):
        torch.manual_seed(470)
        model = G5DirectPredictor(horizon=horizon)
        with torch.no_grad():
            model.motion[-1].weight.normal_(std=.0002)
            model.global_motion.mapping[-1].weight.normal_(std=.0003)
            model.direct_motion.mapping[-1].weight.normal_(std=.0003)
        model.eval().requires_grad_(False)
        processor = DataProcessor("001-rope")
        batch = prepare_batch(synthetic_batch(55,horizon,1),processor,model)
        return model,processor,batch

    @torch.inference_mode()
    def test_normal_hooks_h15_all_attributes_bit_exact_and_weights_unchanged(self):
        model,_,batch = self.fixture()
        before = state_hash(model.state_dict())
        expected = model(batch["inputs"])
        with intervention(model) as raw:
            actual,trace = model(batch["inputs"],return_trace=True)
        self.assertEqual(len(raw),15)
        for key in expected:
            torch.testing.assert_close(expected[key],actual[key],rtol=0,atol=0)
        for s,r in zip(trace["steps"],raw):
            # The production caller masks the raw Motion MLP output. Padding
            # may have an MLP bias; it must not enter base_d or any GS field.
            torch.testing.assert_close(s["base_cumulative_d"],
                                       masked(r["base"],trace["field"].anchor_mask),rtol=0,atol=0)
            self.assertEqual(int(s["base_cumulative_d"][~trace["field"].anchor_mask].count_nonzero()),0)
            torch.testing.assert_close(s["global_delta"],r["global"],rtol=0,atol=0)
            torch.testing.assert_close(s["local_correction"],r["local"],rtol=0,atol=0)
        self.assertEqual(before,state_hash(model.state_dict()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    @torch.inference_mode()
    def test_all_modes_single_step_recomposition_exact_and_prefix_unchanged(self):
        model,_,batch = self.fixture()
        x = batch["inputs"]
        normal,trace = model(x,return_trace=True)
        for step in (0,4,9,14):
            for mode in MODES:
                with self.subTest(step=step,mode=mode):
                    with intervention(model,mode,only_step=step):
                        actual = model(x)
                    torch.testing.assert_close(actual["xyz"][:,:step],normal["xyz"][:,:step],rtol=0,atol=0)
                    direct = same_history_position(x,trace,step,mode)
                    torch.testing.assert_close(actual["xyz"][:,step],direct,rtol=0,atol=0)

    @torch.inference_mode()
    def test_no_global_does_not_zero_local_global_hand_context(self):
        model,_,batch = self.fixture(5)
        _,normal = model(batch["inputs"],return_trace=True)
        with intervention(model,"NoGlobal",only_step=3):
            _,other = model(batch["inputs"],return_trace=True)
        a,b = normal["steps"][3],other["steps"][3]
        self.assertEqual(int(b["global_delta"].count_nonzero()),0)
        self.assertGreater(float(a["hand_contribution"].abs().max()),0)
        for key in ("hand_contribution","local_correction","direct_seed_per_hand","base_cumulative_d"):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    @torch.inference_mode()
    def test_no_base_h2_preserves_h1_but_all_is_copy(self):
        model,_,batch = self.fixture(5)
        x = batch["inputs"]
        normal = model(x)
        with intervention(model,"NoBase-H2+"):
            later = model(x)
        with intervention(model,"NoBase-All"):
            all_off = model(x)
        torch.testing.assert_close(later["xyz"][:,0],normal["xyz"][:,0],rtol=0,atol=0)
        torch.testing.assert_close(all_off["xyz"][:,0],x["gs_xyz"],rtol=0,atol=0)
        self.assertFalse(torch.equal(normal["xyz"][:,0],all_off["xyz"][:,0]))
        for mode in ("NoGlobal","NoLocal","BaseOnly"):
            with intervention(model,mode):
                pred = model(x)
            torch.testing.assert_close(pred["xyz"][:,0],normal["xyz"][:,0],rtol=0,atol=0)

    @torch.inference_mode()
    def test_h1_zero_signed_branches_and_single_hand_mask(self):
        model,_,batch = self.fixture(5)
        _,trace = model(batch["inputs"],return_trace=True)
        self.assertEqual(int(trace["steps"][0]["global_delta"].count_nonzero()),0)
        self.assertEqual(int(trace["steps"][0]["local_correction"].count_nonzero()),0)
        for s in trace["steps"]:
            self.assertEqual(int(s["hand_contribution"][0,0].count_nonzero()),0)
            self.assertEqual(int(s["direct_seed_per_hand"][0,:,0].count_nonzero()),0)

    @torch.inference_mode()
    def test_rbf_is_linear_masked_and_global_is_uniform(self):
        model,_,batch = self.fixture(5)
        x = batch["inputs"]
        _,trace = model(x,return_trace=True)
        field = trace["field"]
        parts = branch_parts(trace["steps"][4],field)
        d = combine(parts,field)
        final = interpolate_anchor_displacements(d,field,x["gs_mask"])
        projected = [interpolate_anchor_displacements(p,field,x["gs_mask"]) for p in parts.values()]
        torch.testing.assert_close(sum(projected),final,rtol=1e-5,atol=1e-7)
        for b in range(2):
            keep = x["gs_mask"][b]
            expected = trace["steps"][4]["global_delta"][b].expand(int(keep.sum()),3)
            torch.testing.assert_close(projected[1][b,keep],expected,rtol=1e-5,atol=1e-7)
        for p in projected:
            self.assertEqual(int(p[~x["gs_mask"]].count_nonzero()),0)

    @torch.inference_mode()
    def test_branch_decomposition_separates_anchor_controls_from_actual_gs(self):
        model,processor,batch = self.fixture(5)
        x = {k:v[:1] for k,v in batch["inputs"].items()}
        _,trace = model(x,return_trace=True)
        branches,cancel,checks = decompose(x,trace,processor.state_dict()["scale"])
        self.assertEqual({r["space"] for r in branches},{"anchor_control","gs_actual"})
        self.assertEqual({r["branch"] for r in branches},{"base","global_","local","correction","final"})
        self.assertEqual(len(checks),5)
        self.assertTrue(all(r["xyz_bit_exact"] for r in checks))
        self.assertTrue(len(cancel)>0)

    def test_undefined_zero_cosines_are_none(self):
        stats = angle_stats(torch.zeros(3,3),torch.ones(3,3),"a_")
        self.assertEqual(stats["a_defined"],0)
        self.assertEqual(stats["a_undefined"],3)
        self.assertIsNone(stats["a_cos_mean"])
        self.assertIsNone(stats["a_opposite_fraction"])

    def test_known_opposite_fields_and_cancellation(self):
        base = torch.tensor([[10.,0,0],[10.,0,0]])
        rest = torch.tensor([[-9.,0,0],[-9.,0,0]])
        stats = cancellation_stats(base,rest,base+rest)
        self.assertAlmostEqual(stats["base_vs_rest_cos_mean"],-1)
        self.assertAlmostEqual(stats["cancellation_index"],1-1/19)
        self.assertAlmostEqual(stats["base_to_final_rms"],10)
        self.assertEqual(stats["base_vs_rest_opposite_fraction"],1.)

    def test_episode_equal_aggregation_not_window_or_point_weighted(self):
        rows = [dict(split="val",episode=3,mode="A",value=1.) for _ in range(55)]
        rows += [dict(split="val",episode=6,mode="A",value=3.) for _ in range(45)]
        rows += [dict(split="train",episode=7,mode="A",value=1000.)]
        got = aggregate(rows,("mode",),("value",))[0]
        self.assertEqual(got["value"],2.)
        self.assertEqual(got["episode_window_counts"],{"3":55,"6":45})

    @torch.inference_mode()
    def test_one_frame_real_loss_and_mock_renderer_match_full_time_reduction(self):
        model,processor,batch = self.fixture(5)
        prediction = model(batch["inputs"])
        renderer = GaussianRenderer(rasterizer=analytic_rasterizer)
        _,m = v4_objective(prediction,batch,processor,renderer)
        singles = []
        for h in range(5):
            view,pred = one_frame(batch,prediction,h)
            loss,metrics = v4_objective(pred,view,processor,renderer)
            singles.append(metrics["geometry"])
            torch.testing.assert_close(metrics["rgb"][:,None],m["rgb_by_horizon"][:,h:h+1],rtol=0,atol=0)
            self.assertEqual(len(view["targets"]["target_xyz"]),1)
            self.assertEqual(view["meta"]["frame_ids"].shape,(2,2))
        torch.testing.assert_close(torch.stack(singles).mean(0),m["geometry"],rtol=1e-6,atol=1e-8)

    @torch.inference_mode()
    def test_full_rollout_masks_and_attributes_and_no_future_leak(self):
        model,_,batch = self.fixture(5)
        x = batch["inputs"]
        changed = copy.deepcopy(x)
        changed["controller_points"][:,3:] += .2
        for mode in MODES:
            with intervention(model,mode):
                expected = model(x)
            with intervention(model,mode):
                actual = model(changed)
            torch.testing.assert_close(expected["xyz"][:,:3],actual["xyz"][:,:3],rtol=0,atol=0)
            self.assertTrue(torch.isfinite(actual["xyz"]).all())
            self.assertEqual(int(actual["xyz"][~x["gs_mask"][:,None].expand(-1,5,-1)].count_nonzero()),0)
            for name in ("log_scales","quaternions","opacity_logits","sh_dc"):
                torch.testing.assert_close(expected[name],actual[name],rtol=0,atol=0)

    def test_hooks_removed_after_failure_and_bad_mode_rejected(self):
        model,_,batch = self.fixture(5)
        with self.assertRaises(RuntimeError):
            with intervention(model,"NoLocal"):
                raise RuntimeError("synthetic interruption")
        self.assertEqual(len(model.motion._forward_hooks),0)
        self.assertEqual(len(model.global_motion._forward_hooks),0)
        self.assertEqual(len(model.direct_motion._forward_hooks),0)
        with self.assertRaises(ValueError):
            switches("NotAMode",0)
        with self.assertRaises(ValueError):
            with intervention(model,only_step=5):
                pass

    def test_report_from_isolated_synthetic_records_and_no_overwrite(self):
        # No synthetic result is written to any experiment output directory.
        from scripts.experiments.v4_g5_motion_attribution.report import build
        modes = (*MODES,"Copy")
        cases = (("val",3,135),("val",3,245),("val",6,110),("train",7,105),
                 ("val",3,25),("val",6,20),("val",6,200))
        episode = lambda count: dict(samples=count,geometry_mm=2.,supervised_geometry=.02,rgb=.1,alpha=.1)
        summary = {mode:dict(geometry_mm=2.,rgb=.1,alpha=.1,objective=.13,
                            episodes={"3":episode(55),"6":episode(45)}) for mode in modes}
        windows = [dict(split=split,episode=ep,start=start,mode=mode,cd_mm=2.,rgb=.1,alpha=.1,joint=.13)
                   for split,ep,start in cases for mode in modes]
        horizons = [dict(r,horizon=15,mean_x_mm=0.,mean_y_mm=0.,mean_z_mm=0.,rms_mm=0.) for r in windows]
        empty_names = ("same_history_val_summary","branch_fields","cancellation","cancellation_val_summary",
                       "structure","equivalence","intervention_checks")
        data = {name:[] for name in empty_names}
        data.update(val_summary=summary,windows=windows,horizons=horizons,horizon_val_summary=[])
        data["equivalence"] = [dict(projected_sum_max_error_mm=0.)]
        provenance = dict(training_source_sha="SYNTHETIC_TEST_ONLY",diagnostic_source_sha="TEST",
                          checkpoint="TEST_NO_CHECKPOINT",checkpoint_sha256="0"*64,job_id="TEST",
                          gpu="NO_GPU",torch="TEST",cuda="NONE",gsplat="MOCK",elapsed_seconds=0.,
                          peak_allocated_bytes=0,model_hash_before="unchanged",model_hash_after="unchanged")
        with tempfile.TemporaryDirectory(prefix="touch4d-g5-attribution-") as tmp:
            out=Path(tmp)
            (out/"results").mkdir();(out/"figures").mkdir()
            for name,value in data.items():
                (out/"results"/(name+".json")).write_text(json.dumps(value),encoding="utf8")
            (out/"provenance.json").write_text(json.dumps(provenance),encoding="utf8")
            (out/"completed.json").write_text("{}",encoding="utf8")
            with patch("scripts.experiments.v4_g5_motion_attribution.report.overview"), \
                 patch("scripts.experiments.v4_g5_motion_attribution.report.case_plots",return_value=[]):
                build(out)
                report=(out/"final_report.md").read_text(encoding="utf8")
                self.assertIn("SYNTHETIC_TEST_ONLY",report)
                self.assertIn("base_d",report)
                self.assertIn("未定义",report)
                self.assertIn("Q7",report)
                self.assertTrue((out/"artifact_manifest.json").is_file())
                with self.assertRaises(FileExistsError):
                    build(out)


if __name__ == "__main__":
    unittest.main()
