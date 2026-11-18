"""CPU-only gates: complete 55/45 window fixtures, no models or inference."""
import copy
import json
import unittest

from scripts.experiments.v4_g5_cross_necessity.gates import (
    EPISODE_COUNTS, JOINT_ABLATION, NORMAL, REVIEW_FLAGS, evaluate_gate,
)


def fixture(candidate_cd=None, candidate_joint=None, normal_cd=None):
    """Each update is a fixed whole paired Val100, never independent GS rows."""
    factor = .15703709051 * 1000.
    candidate_cd = candidate_cd or (lambda ep, h: 1.)
    candidate_joint = candidate_joint or (lambda ep, h: 1.)
    normal_cd = normal_cd or (lambda ep, h: 2.)
    rows = []
    for mode in (NORMAL, JOINT_ABLATION):
        for ep, count in EPISODE_COUNTS.items():
            for start in range(count):
                for h in range(1, 16):
                    cd = normal_cd(ep, h) * (candidate_cd(ep, h) if mode == JOINT_ABLATION else 1.)
                    joint = .1 * (candidate_joint(ep, h) if mode == JOINT_ABLATION else 1.)
                    alpha = .05
                    rgb = joint - cd / factor - .1 * alpha
                    assert rgb >= 0
                    rows.append(dict(mode=mode, episode=int(ep), start=start,
                                     split="val", horizon=h, cd_mm=cd,
                                     joint=joint, rgb=rgb, alpha=alpha))
    summary = {}
    for mode in (NORMAL, JOINT_ABLATION):
        episodes = {}
        for ep, count in EPISODE_COUNTS.items():
            selected = [r for r in rows if r["mode"] == mode and str(r["episode"]) == ep]
            avg = lambda key: sum(r[key] for r in selected) / len(selected)
            episodes[ep] = dict(samples=count, geometry_mm=avg("cd_mm"),
                                geometry=avg("cd_mm") / factor,
                                supervised_geometry=avg("cd_mm") / factor,
                                rgb=avg("rgb"), alpha=avg("alpha"))
        avg_ep = lambda key: sum(e[key] for e in episodes.values()) / 2.
        summary[mode] = dict(episodes=episodes, geometry_mm=avg_ep("geometry_mm"),
                             rgb=avg_ep("rgb"), alpha=avg_ep("alpha"),
                             objective=avg_ep("supervised_geometry") + avg_ep("rgb") + .1 * avg_ep("alpha"))
    return summary, rows


def reviewed():
    return dict({k: True for k in REVIEW_FLAGS}, evidence_paths=["figures/reviewed.png", "results/response.json"])


class FrozenCrossGateTests(unittest.TestCase):
    def test_complete_equal_models_require_explicit_review(self):
        summary, rows = fixture()
        result = evaluate_gate(summary, rows)
        self.assertEqual(result["status"], "INCONCLUSIVE_NO_AUTOSTART")
        self.assertTrue(result["hard_gate_passed"])
        self.assertTrue(result["conservative_numeric_passed"])
        self.assertFalse(result["eligible_for_stage2"])

    def test_complete_reviewed_evidence_eligible_and_json_safe(self):
        summary, rows = fixture()
        before = copy.deepcopy((summary, rows))
        result = evaluate_gate(summary, rows, reviewed())
        self.assertEqual(result["status"], "PASS_ELIGIBLE")
        self.assertTrue(result["eligible_for_stage2"])
        self.assertEqual(result["coverage"]["rows"], 3000)
        self.assertTrue(result["coverage"]["window_identities_validated"])
        self.assertEqual(result["coverage"]["horizon_counts"][NORMAL]["3"]["15"], 55)
        self.assertEqual(result["coverage"]["horizon_counts"][NORMAL]["6"]["15"], 45)
        self.assertEqual(len(result["checks"]["conservative_late_horizon"]), 36)
        self.assertEqual((summary, rows), before)
        json.dumps(result, allow_nan=False)

    def test_exact_hard_and_conservative_boundaries_pass(self):
        summary, rows = fixture(lambda ep, h: 1.03, lambda ep, h: 1.02)
        result = evaluate_gate(summary, rows, reviewed())
        self.assertEqual(result["status"], "PASS_ELIGIBLE")
        for check in result["checks"]["user_hard"]:
            self.assertAlmostEqual(check["ratio"], check["ratio_max"], places=12)

    def test_small_real_joint_excess_is_hard_failure_even_if_reviewed(self):
        summary, rows = fixture(candidate_joint=lambda ep, h: 1.020000001)
        result = evaluate_gate(summary, rows, reviewed())
        self.assertEqual(result["status"], "FAIL_STAGE1")
        self.assertFalse(result["eligible_for_stage2"])
        self.assertIn("user_whole_val:joint", result["reasons"])

    def test_cd_excess_hard_failure(self):
        summary, rows = fixture(candidate_cd=lambda ep, h: 1.030000001)
        result = evaluate_gate(summary, rows, reviewed())
        self.assertEqual(result["status"], "FAIL_STAGE1")
        self.assertIn("user_whole_val:cd_mm", result["reasons"])

    def test_one_episode_regression_cannot_hide_in_other_episode_improvement(self):
        ratio = lambda ep, h: 1.10 if ep == "3" else .80
        summary, rows = fixture(ratio, ratio)
        result = evaluate_gate(summary, rows, reviewed())
        self.assertTrue(result["hard_gate_passed"])
        self.assertEqual(result["status"], "INCONCLUSIVE_NO_AUTOSTART")
        self.assertFalse(result["conservative_numeric_passed"])
        self.assertIn("conservative_episode:joint:ep3", result["reasons"])

    def test_h15_regression_cannot_hide_in_early_horizon_improvement(self):
        ratio = lambda ep, h: 2. if h == 15 else .9
        summary, rows = fixture(ratio, ratio)
        result = evaluate_gate(summary, rows, reviewed())
        self.assertTrue(result["hard_gate_passed"])
        self.assertEqual(result["status"], "INCONCLUSIVE_NO_AUTOSTART")
        self.assertIn("conservative_late_episode_equal:cd_mm:H15", result["reasons"])

    def test_late_summary_is_episode_equal_not_55_45_window_weighted(self):
        base = lambda ep, h: 1. if ep == "3" else 3.
        ratio = lambda ep, h: 1.04 if ep == "3" else 2.95 / 3.
        summary, rows = fixture(ratio, normal_cd=base)
        result = evaluate_gate(summary, rows, reviewed())
        check = next(c for c in result["checks"]["conservative_late_horizon"]
                     if c["scope"] == "conservative_late_episode_equal"
                     and c["horizon"] == 10 and c["metric"] == "cd_mm")
        self.assertAlmostEqual(check["baseline"], 2., places=12)
        self.assertAlmostEqual(check["value"], (1.04 + 2.95) / 2., places=12)
        self.assertAlmostEqual(check["ratio"], (1.04 + 2.95) / 4., places=12)
        self.assertEqual(check["episode_window_counts"], {"3": 55, "6": 45})

    def test_each_review_flag_and_evidence_paths_must_be_explicit(self):
        summary, rows = fixture()
        for flag in REVIEW_FLAGS:
            for value in (None, False, "true", 1):
                review = reviewed()
                review[flag] = value
                with self.subTest(flag=flag, value=value):
                    result = evaluate_gate(summary, rows, review)
                    self.assertEqual(result["status"], "INCONCLUSIVE_NO_AUTOSTART")
        for paths in (None, [], [""], ["  "], "figures/reviewed.png", [3]):
            review = reviewed()
            review["evidence_paths"] = paths
            with self.subTest(paths=paths):
                self.assertFalse(evaluate_gate(summary, rows, review)["eligible_for_stage2"])

    def test_train_rows_are_rejected_rather_than_contaminating_val(self):
        summary, rows = fixture()
        rows.append(dict(rows[0], split="train", episode=7))
        with self.assertRaisesRegex(ValueError, "only explicit Val"):
            evaluate_gate(summary, rows, reviewed())

    def test_missing_horizon_or_window_is_rejected(self):
        summary, rows = fixture()
        with self.assertRaisesRegex(ValueError, "expected 55"):
            evaluate_gate(summary, rows[1:])
        rows = [r for r in rows if not (r["mode"] == NORMAL and r["horizon"] == 15)]
        with self.assertRaises(ValueError):
            evaluate_gate(summary, rows)

    def test_wrong_summary_count_and_additional_episode_are_rejected(self):
        summary, rows = fixture()
        summary[NORMAL]["episodes"]["3"]["samples"] = 54
        with self.assertRaisesRegex(ValueError, "55 Val windows"):
            evaluate_gate(summary, rows)
        summary, rows = fixture()
        summary[NORMAL]["episodes"]["7"] = copy.deepcopy(summary[NORMAL]["episodes"]["3"])
        with self.assertRaisesRegex(ValueError, "only Val episodes"):
            evaluate_gate(summary, rows)

    def test_nonfinite_negative_and_invalid_data_are_rejected(self):
        for value in (float("nan"), float("inf"), -1., True, "0.1"):
            summary, rows = fixture()
            rows[0]["joint"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate_gate(summary, rows)
        summary, rows = fixture()
        summary[NORMAL]["objective"] = float("nan")
        with self.assertRaises(ValueError):
            evaluate_gate(summary, rows)

    def test_duplicate_or_changed_window_identity_is_rejected(self):
        summary, rows = fixture()
        rows[1] = copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_gate(summary, rows)
        summary, rows = fixture()
        rows[1]["start"] = 1000
        with self.assertRaisesRegex(ValueError, "window identities differ"):
            evaluate_gate(summary, rows)

    def test_mixed_or_omitted_identities_are_not_silently_invented(self):
        summary, rows = fixture()
        del rows[0]["start"]
        with self.assertRaises(ValueError):
            evaluate_gate(summary, rows)
        for row in rows:
            row.pop("start", None)
        result = evaluate_gate(summary, rows, reviewed())
        self.assertEqual(result["status"], "PASS_ELIGIBLE")
        self.assertFalse(result["coverage"]["window_identities_validated"])

    def test_summary_must_match_raw_horizon_records_and_episode_equal_reduction(self):
        summary, rows = fixture()
        summary[JOINT_ABLATION]["geometry_mm"] *= .9
        with self.assertRaisesRegex(ValueError, "summary/raw-record mismatch"):
            evaluate_gate(summary, rows)
        summary, rows = fixture()
        for row in rows:
            if row["mode"] == JOINT_ABLATION:
                row["cd_mm"] *= .9
        with self.assertRaisesRegex(ValueError, "summary/raw-record mismatch"):
            evaluate_gate(summary, rows)

    def test_missing_required_mode_or_extra_partial_mode_is_rejected(self):
        summary, rows = fixture()
        del summary[JOINT_ABLATION]
        with self.assertRaisesRegex(ValueError, "summaries are required"):
            evaluate_gate(summary, rows)
        summary, rows = fixture()
        summary["Copy"] = copy.deepcopy(summary[NORMAL])
        with self.assertRaisesRegex(ValueError, "Copy"):
            evaluate_gate(summary, rows)

    def test_zero_baseline_never_generates_nan_ratio(self):
        summary, rows = fixture()
        for row in rows:
            row["cd_mm"] = 0.
        for mode in summary:
            summary[mode]["geometry_mm"] = 0.
            for ep in summary[mode]["episodes"].values():
                ep["geometry_mm"] = 0.
        with self.assertRaisesRegex(ValueError, "positive"):
            evaluate_gate(summary, rows)


if __name__ == "__main__":
    unittest.main()
