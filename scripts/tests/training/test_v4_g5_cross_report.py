"""Reporting fixtures are SYNTHETIC_TEST_ONLY, never real experiment evidence."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.experiments.v4_g5_cross_necessity import report
from scripts.experiments.v4_g5_cross_necessity.gates import evaluate_gate
from scripts.tests.training.test_v4_g5_cross_gates import fixture as gate_fixture


SYNTHETIC = "SYNTHETIC_TEST_ONLY"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6"
    "WiAAAAAASUVORK5CYII=")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)


def fixture(out, *, fail=True, unavailable=False):
    """Explicit synthetic Val100 metadata used only in a TemporaryDirectory."""
    out.mkdir(exist_ok=False)
    (out/"results").mkdir()
    (out/"arrays").mkdir()
    (out/"figures").mkdir()
    summary_two, raw = gate_fixture(candidate_cd=lambda ep, h: 1.04 if fail else 1.)
    starts = {
        3: [135, 245, 25, *range(300, 352)],
        6: [110, 20, 200, *range(400, 442)],
    }
    horizons, summary = [], {}
    for mode in (*report.MODES, "Copy"):
        source = report.JOINT if mode == report.JOINT else "Normal"
        summary[mode] = copy.deepcopy(summary_two[source])
        for row in raw:
            if row["mode"] != source:
                continue
            r = dict(row, mode=mode, start=starts[row["episode"]][row["start"]])
            r.update(rms_mm=.1*r["horizon"], demeaned_rms_mm=.05*r["horizon"],
                     mean_x_mm=.05, mean_y_mm=-.1, mean_z_mm=.1,
                     change_from_normal_rms_mm=.02)
            horizons.append(r)
        for h in range(1, 16):
            horizons.append(dict(split="train", episode=7, start=105, mode=mode, horizon=h,
                cd_mm=2., rgb=.08, alpha=.05, joint=.1, rms_mm=h*.1,
                mean_x_mm=.05, mean_y_mm=-.1, mean_z_mm=.1))
    same_history = []
    for row in horizons:
        if row["horizon"] not in (1, 5, 10, 15) or row["mode"] != "Normal":
            continue
        for mode in report.MODES[1:]:
            same_history.append(dict(row, mode=mode, history="fixed_Normal",
                delta_cd_mm=.01, delta_joint=.001, delta_rgb=.001, delta_alpha=.001,
                change_rms_mm=.03, cd_improved=False, cd_worsened=True))
    branch, cancellation, structure, regions, action = [], [], [], [], []
    geometry = {}
    for case in report.CASES:
        split, ep, start = case
        scope = dict(split=split, episode=ep, start=start)
        for h in range(1, 16):
            for space in ("anchor_control", "gs_actual"):
                for region in ("all", "near", "middle", "far"):
                    cancellation.append(dict(scope, horizon=h, space=space, region=region,
                        base_vs_rest_cos_mean=-.2, base_vs_rest_opposite_fraction=.6,
                        cancellation_index=.1, base_to_final_rms=1.2,
                        base_larger_than_final_fraction=.4, nearly_cancelled_fraction=.02))
                    for name in ("base", "global_", "local", "correction", "final"):
                        branch.append(dict(scope, horizon=h, space=space, region=region, branch=name,
                            mean_x_mm=.1, mean_y_mm=-.2, mean_z_mm=.3, rms_mm=.5,
                            norm_p95_mm=.8, signed_projection_mean_mm=.2, final_fraction_mean=.4))
            for mode in (*report.MODES, "Copy"):
                for region in ("all", "near", "middle", "far"):
                    structure.append(dict(scope, horizon=h, mode=mode, region=region,
                        displacement_rms_mm=.3, pred_region_to_full_gt_mm=1.8, neighbor_distortion_p95=.05))
            for mode in ("Normal", report.JOINT):
                for response in ("FreezeAction", "HalfRootTranslation"):
                    absent = unavailable and case == report.CASES[0] and mode == report.JOINT and response == "HalfRootTranslation"
                    action.append(dict(scope, horizon=h, mode=mode, response=response,
                        available=not absent, output_change_rms_mm=None if absent else .02*h,
                        repeat_floor_mm=0., measurable_above_numeric_floor=not absent,
                        mean_x_mm=None if absent else .01, mean_y_mm=None if absent else -.02,
                        mean_z_mm=None if absent else .03,
                        reason="SYNTHETIC_TEST_ONLY unavailable Root" if absent else None))
        initial = np.array([[.1, .01, .2], [.11, .02, .21], [.12, .03, .22], [.13, .04, .23]])
        arrays = dict(initial_world=initial, frame_ids=np.arange(start, start+16),
                      region_near=np.array([0]), region_middle=np.array([1, 2]), region_far=np.array([3]))
        for mode in (*report.MODES, "Copy"):
            arrays[mode+"_world"] = np.broadcast_to(initial, (15, 4, 3)).copy()
        for h in range(1, 16):
            arrays[f"gt_h{h:02}_world"] = initial+np.array([.0001*h, 0., 0.])
        np.savez_compressed(out/"arrays"/f"{report._key(case)}.npz", **arrays)
        geometry[report._key(case)] = arrays
    regions = [dict(row, region=region, pred_region_to_full_gt_mm=2.,
                    delta_region_to_full_gt_mm=.1, change_from_normal_rms_mm=.03)
               for row in same_history for region in ("all", "near", "middle", "far")]
    checks = [dict(split=case[0], episode=case[1], start=case[2], horizon=h, mode=mode,
                   prefix_bit_exact=True, current_step_bit_exact=True)
              for case in report.CASES for h in (1, 5, 10, 15) for mode in report.MODES[1:]]
    tables = dict(val_summary=summary, horizons=horizons, same_history=same_history,
                  branch_fields=branch, cancellation=cancellation, structure=structure,
                  action_response=action, same_history_regions=regions, intervention_checks=checks)
    gate = evaluate_gate(summary, [r for r in horizons if r["split"] == "val"])
    tables["gate_preliminary"] = gate
    for name, value in tables.items():
        write(out/"results"/f"{name}.json", value)
    p = dict(run_kind=SYNTHETIC, training_source_sha=report.BASE, diagnostic_source_sha=SYNTHETIC,
             checkpoint_sha256=report.CHECKPOINT_HASH, checkpoint_hash_after=report.CHECKPOINT_HASH,
             model_hash_before="synthetic_state", model_hash_after="synthetic_state",
             optimizer_created=False, backward_called=False, optimizer_updates=0,
             parameters_frozen=True, job_id=SYNTHETIC, gpu=SYNTHETIC,
             checkpoint="/SYNTHETIC_TEST_ONLY/not_a_real_checkpoint.pt",
             expected_frozen_validation=summary["Normal"], parameter_count=1,
             torch=SYNTHETIC, cuda=SYNTHETIC, gsplat=SYNTHETIC,
             elapsed_seconds=0., peak_allocated_bytes=0, command=[SYNTHETIC])
    write(out/"provenance.json", p)
    write(out/"completed.json", dict(state="INFERENCE_COMPLETED", val100=True,
                                    checkpoint_unchanged=True, model_unchanged=True, source_sha=SYNTHETIC))
    for relative in report.expected_images():
        target = out/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(PNG)
    return tables, geometry


def mock_plots(out, evidence):
    """Only output plumbing is tested: no fake geometry claim or GPU rendering."""
    for name in report.FIGURES:
        with (out/"figures"/name).open("xb") as stream:
            stream.write(PNG)


class FrozenCrossReportTests(unittest.TestCase):
    def test_build_explicit_synthetic_report_manifest_and_null_review(self):
        with tempfile.TemporaryDirectory(prefix=SYNTHETIC+"-") as directory:
            out = Path(directory)/SYNTHETIC
            fixture(out)
            before = {p:report._sha(p) for p in out.rglob("*") if p.is_file()}
            with patch.object(report, "_make_figures", side_effect=mock_plots) as plots:
                manifest = report.build(out)
            plots.assert_called_once()
            text = (out/"frozen_attribution_report.md").read_text(encoding="utf8")
            self.assertIn(SYNTHETIC, text)
            self.assertIn("FAIL_STAGE1", text)
            self.assertIn("阶段一停止", text)
            self.assertIn("没有触觉到输出", text)
            self.assertIn("4D head_mass", text)
            for mode in report.MODES:
                self.assertIn(mode, text)
            self.assertIn("固定 Normal", text)
            self.assertIn("不是原始照片", text)
            self.assertIn("完整AR预测区域→完整GT mm", text)
            self.assertIn("固定历史预测区域→完整GT mm", text)
            self.assertIn("机械地输出Copy Initial", text)
            self.assertIn("全部为True", text)
            self.assertFalse(manifest["launches_stage2"])
            self.assertFalse(manifest["independent_visual_review_completed"])
            self.assertEqual(manifest["run_kind"], SYNTHETIC)
            review = report._json(out/"review_template.json")
            self.assertTrue(all(review[name] is None for name in report.REVIEW_FLAGS))
            for item in manifest["files"]:
                self.assertEqual(report._sha(out/item["path"]), item["sha256"])
            for path, value in before.items():
                self.assertEqual(report._sha(path), value)
            self.assertFalse(list(out.rglob("*.html")))
            self.assertFalse(list(out.rglob("*.zip")))
            with self.assertRaises(FileExistsError):
                report.build(out)

    def test_numeric_pass_is_not_review_pass_or_autostart(self):
        with tempfile.TemporaryDirectory(prefix=SYNTHETIC+"-") as directory:
            out = Path(directory)/SYNTHETIC
            fixture(out, fail=False)
            with patch.object(report, "_make_figures", side_effect=mock_plots):
                manifest = report.build(out)
            self.assertEqual(manifest["preliminary_gate"], "INCONCLUSIVE_NO_AUTOSTART")
            self.assertFalse(manifest["launches_stage2"])
            text = (out/"frozen_attribution_report.md").read_text(encoding="utf8")
            self.assertIn("未完成独立视觉", text)
            self.assertIn("未执行", text)

    def test_unavailable_action_is_not_zero_sensitivity_or_fake_pass(self):
        with tempfile.TemporaryDirectory(prefix=SYNTHETIC+"-") as directory:
            out = Path(directory)/SYNTHETIC
            fixture(out, unavailable=True)
            with patch.object(report, "_make_figures", side_effect=mock_plots):
                report.build(out)
            text = (out/"frozen_attribution_report.md").read_text(encoding="utf8")
            self.assertIn("动作反事实不可用的案例/路径/动作组合：1", text)
            self.assertIn("不能把缺测当作零响应", text)
            self.assertFalse(report._response_available(dict(available=False, output_change_rms_mm=None)))
            self.assertFalse(report._response_available(dict(available=True, output_change_rms_mm=None)))
            self.assertTrue(report._response_available(dict(available=True, output_change_rms_mm=0.)))

    def test_refuse_existing_artifact_before_touching_inputs(self):
        with tempfile.TemporaryDirectory(prefix=SYNTHETIC+"-") as directory:
            out = Path(directory)
            old = out/"frozen_attribution_report.md"
            old.write_text("PRESERVED_USER_FILE", encoding="utf8")
            with patch.object(report, "_make_figures") as plots:
                with self.assertRaises(FileExistsError):
                    report.build(out)
            plots.assert_not_called()
            self.assertEqual(old.read_text(encoding="utf8"), "PRESERVED_USER_FILE")

    def test_reject_incomplete_run_changed_weights_stale_gate_and_missing_image(self):
        with tempfile.TemporaryDirectory(prefix=SYNTHETIC+"-") as directory:
            out = Path(directory)/SYNTHETIC
            fixture(out)
            variants = (
                ("completed.json", "val100", False),
                ("provenance.json", "checkpoint_hash_after", "changed"),
                ("provenance.json", "model_hash_after", "changed"),
                ("provenance.json", "optimizer_updates", 1),
                ("results/gate_preliminary.json", "status", "PASS_ELIGIBLE"),
            )
            for relative, key, value in variants:
                path = out/relative
                original = path.read_bytes()
                record = report._json(path)
                record[key] = value
                path.write_text(json.dumps(record), encoding="utf8")
                with self.subTest(relative=relative, key=key), patch.object(report, "_make_figures") as plots:
                    with self.assertRaises(ValueError):
                        report.build(out)
                    plots.assert_not_called()
                    self.assertFalse((out/"frozen_attribution_report.md").exists())
                path.write_bytes(original)
            image = out/report._image_refs()[0]
            image.unlink()
            with self.assertRaisesRegex(ValueError, "Missing saved rendering"):
                report.build(out)
            self.assertFalse((out/"artifact_manifest.json").exists())

    def test_fixed_world_bounds_include_all_modes_horizons_and_independent_gt(self):
        initial = np.array([[.1, .2, .3], [.11, .21, .31]])
        arrays = {"initial_world": initial}
        for mode in (*report.MODES, "Copy"):
            arrays[mode+"_world"] = np.broadcast_to(initial, (15, 2, 3)).copy()
        for h in range(1, 16):
            arrays[f"gt_h{h:02}_world"] = initial.copy()
        arrays["NoCrossValue_world"][0, 0, 0] = -.25
        arrays["gt_h01_world"][0, 2] = 1.1
        lower, upper = report._world_bounds(arrays)
        self.assertLess(lower[0], -250.)
        self.assertGreater(upper[2], 1100.)
        self.assertTrue(np.all(upper-lower > 0))

    def test_nonfinite_json_is_rejected_not_converted_to_zero(self):
        with tempfile.TemporaryDirectory(prefix=SYNTHETIC+"-") as directory:
            path = Path(directory)/"SYNTHETIC_TEST_ONLY.json"
            path.write_text('{"invalid": NaN}', encoding="utf8")
            with self.assertRaises(ValueError):
                report._json(path)

    def test_all_2016_renderings_checked_including_non_displayed_images(self):
        expected, displayed = report.expected_images(), set(report._image_refs())
        self.assertEqual(len(expected), 2016)
        self.assertEqual(len(set(expected)), 2016)
        self.assertTrue(displayed < set(expected))
        with tempfile.TemporaryDirectory(prefix=SYNTHETIC+"-") as directory:
            out = Path(directory)/SYNTHETIC
            fixture(out)
            hidden = next(relative for relative in expected if relative not in displayed)
            (out/hidden).unlink()
            with patch.object(report, "_make_figures") as plots:
                with self.assertRaisesRegex(ValueError, "Missing saved rendering"):
                    report.build(out)
                plots.assert_not_called()
            self.assertFalse((out/"frozen_attribution_report.md").exists())

    def test_exact_single_step_region_and_action_record_coverage(self):
        with tempfile.TemporaryDirectory(prefix=SYNTHETIC+"-") as directory:
            out = Path(directory)/SYNTHETIC
            fixture(out)
            evidence = report._read_inputs(out)
            self.assertEqual(evidence["coverage"]["normal_windows"], 101)
            self.assertEqual(evidence["coverage"]["same_history"], 3636)
            self.assertEqual(evidence["coverage"]["same_history_regions"], 14544)
            self.assertEqual(evidence["coverage"]["action_response"], 420)
            self.assertEqual(evidence["coverage"]["intervention_checks"], 252)
            for name in ("same_history", "same_history_regions", "action_response"):
                path = out/"results"/f"{name}.json"
                original = path.read_bytes()
                rows = report._json(path)
                variants = (
                    ("missing", rows[1:]),
                    ("duplicate", rows+[copy.deepcopy(rows[0])]),
                    ("extra", rows+[dict(rows[0], mode="SYNTHETIC_TEST_ONLY_UNEXPECTED")]),
                )
                for label, changed in variants:
                    path.write_text(json.dumps(changed), encoding="utf8")
                    with self.subTest(table=name, variant=label), patch.object(report, "_make_figures") as plots:
                        with self.assertRaisesRegex(ValueError, name+": exact coverage"):
                            report.build(out)
                        plots.assert_not_called()
                        self.assertFalse((out/"artifact_manifest.json").exists())
                path.write_bytes(original)

    def test_optional_intervention_checks_exact_keys_and_true_invariants(self):
        with tempfile.TemporaryDirectory(prefix=SYNTHETIC+"-") as directory:
            out = Path(directory)/SYNTHETIC
            fixture(out)
            path = out/"results"/"intervention_checks.json"
            original = path.read_bytes()
            rows = report._json(path)
            for label, changed in (
                ("missing", rows[1:]), ("duplicate", rows+[rows[0]]),
                ("prefix_failed", [dict(rows[0], prefix_bit_exact=False), *rows[1:]]),
                ("current_failed", [dict(rows[0], current_step_bit_exact=False), *rows[1:]]),
            ):
                path.write_text(json.dumps(changed), encoding="utf8")
                with self.subTest(variant=label), patch.object(report, "_make_figures") as plots:
                    with self.assertRaises(ValueError):
                        report.build(out)
                    plots.assert_not_called()
            path.write_bytes(original)
            path.unlink()
            evidence = report._read_inputs(out)
            self.assertIsNone(evidence["coverage"]["intervention_checks"])
            self.assertNotIn("intervention_checks", evidence)

    def test_empty_anchor_region_is_not_fabricated_or_required(self):
        with tempfile.TemporaryDirectory(prefix=SYNTHETIC+"-") as directory:
            out = Path(directory)/SYNTHETIC
            fixture(out)
            for name in ("branch_fields", "cancellation"):
                path = out/"results"/f"{name}.json"
                rows = report._json(path)
                rows = [row for row in rows if not (
                    row["space"] == "anchor_control" and row["region"] in ("near", "far"))]
                path.write_text(json.dumps(rows), encoding="utf8")
            evidence = report._read_inputs(out)
            text = report._report(out, evidence)
            self.assertIn("未定义/未记录", text)
            self.assertIn("不对Anchor近端/远端空分组", text)


if __name__ == "__main__":
    unittest.main()
