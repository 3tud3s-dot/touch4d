"""Pre-registered, conservative eligibility checks for frozen G5 ablations.

This module makes no inference and never launches Stage 2. ``PASS_ELIGIBLE``
means that a separate caller may consider the user's conditional next stage;
it is not proof that attention is useless or that a retrained model will work.
The episode/late-horizon guards are deliberately stricter sufficient conditions
for automatic progression, distinct from the user's two whole-Val thresholds.
"""
from collections import defaultdict
from collections.abc import Mapping
import math
from numbers import Integral, Real


NORMAL = "Normal"
JOINT_ABLATION = "NoBase+NoAttentionPaths"
EPISODE_COUNTS = {"3": 55, "6": 45}
HORIZONS = tuple(range(1, 16))
LATE_HORIZONS = tuple(range(10, 16))
MAX_RATIOS = {"joint": 1.02, "cd_mm": 1.03}
REVIEW_FLAGS = (
    "visual_no_new_severe_failures",
    "structure_no_new_severe_failures",
    "action_response_measurable",
    "action_response_semantics_valid",
    "causal_prefix_valid",
)
# FP arithmetic at an exact ratio boundary is not a scientific tolerance.
BOUNDARY_RTOL = 1e-12
# These match the pre-existing frozen-inference reproduction checks. They only
# check that a summary and its own raw horizon records describe the same run.
REPRODUCTION_ATOL = {"cd_mm": 1e-4, "rgb": 1e-6, "alpha": 1e-6, "joint": 1e-6}


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label}: expected a numeric value")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f"{label}: expected a finite {'positive' if positive else 'nonnegative'} value")
    return result


def _integer(value, label, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label}: expected an integer >= {minimum}")
    return int(value)


def _episode_metrics(values, label):
    if not isinstance(values, Mapping):
        raise ValueError(f"{label}: missing episode summary")
    result = {k: _number(values.get(source), f"{label}.{source}")
              for k, source in (("cd_mm", "geometry_mm"), ("rgb", "rgb"), ("alpha", "alpha"))}
    if "joint" in values:
        joint = values["joint"]
    elif "objective" in values:
        joint = values["objective"]
    else:
        # Do not add geometry_mm to RGB. G5's joint uses normalized CD.
        normalized = values.get("supervised_geometry", values.get("geometry"))
        joint = (_number(normalized, f"{label}.normalized_cd")
                 + result["rgb"] + .1 * result["alpha"])
    result["joint"] = _number(joint, f"{label}.joint")
    return result


def _mean(rows, key):
    return math.fsum(r[key] for r in rows) / len(rows)


def _compatible(actual, expected, metric, label):
    if abs(actual - expected) > REPRODUCTION_ATOL[metric]:
        raise ValueError(f"{label}: summary/raw-record mismatch ({actual} vs {expected})")


def _comparison(scope, metric, baseline, value, counts, *, horizon=None, episode=None):
    baseline = _number(baseline, f"{scope}.{metric}.baseline", positive=True)
    value = _number(value, f"{scope}.{metric}.value")
    maximum = MAX_RATIOS[metric]
    bound = baseline * maximum
    ratio = value / baseline
    if not math.isfinite(bound) or not math.isfinite(ratio):
        raise ValueError(f"{scope}.{metric}: unrepresentable comparison")
    passed = value <= bound or math.isclose(value, bound, rel_tol=BOUNDARY_RTOL, abs_tol=0.)
    row = dict(scope=scope, metric=metric, baseline=baseline, value=value,
               ratio=ratio, ratio_max=maximum, limit=bound, passed=passed,
               episode_window_counts=dict(counts))
    if horizon is not None:
        row["horizon"] = horizon
    if episode is not None:
        row["episode"] = episode
    return row


def evaluate_gate(summary, horizon_rows, review=None):
    """Evaluate eligibility without silently accepting incomplete evidence.

    ``summary`` maps mode to the original GaussianMetrics summary. All supplied
    modes must have complete ep3=55/ep6=45 Val H1--H15 records. Input rows must
    already be Val-only; passing Train rows is an error, not a hidden filter.
    If ``start`` is supplied, all rows need it and exact window identities are
    checked across horizons and modes. Review flags must be literal ``True``.
    Evidence paths are recorded assertions; this function does not authenticate
    a review, inspect figures, or invent counterfactual ground truth.
    """
    if not isinstance(summary, Mapping) or not all(isinstance(k, str) for k in summary):
        raise ValueError("summary must map mode names to GaussianMetrics summaries")
    if NORMAL not in summary or JOINT_ABLATION not in summary:
        raise ValueError("Normal and NoBase+NoAttentionPaths summaries are required")
    if not isinstance(horizon_rows, (list, tuple)) or not horizon_rows:
        raise ValueError("nonempty complete Val horizon rows are required")
    if review is not None and not isinstance(review, Mapping):
        raise ValueError("review must be a mapping or None")

    parsed = {}
    for mode, values in summary.items():
        if not isinstance(values, Mapping) or not isinstance(values.get("episodes"), Mapping):
            raise ValueError(f"{mode}: missing episodes")
        episodes = {str(k): v for k, v in values["episodes"].items()}
        if len(episodes) != len(values["episodes"]) or set(episodes) != set(EPISODE_COUNTS):
            raise ValueError(f"{mode}: expected only Val episodes 3 and 6")
        aggregate = {k: _number(values.get(source), f"{mode}.{source}")
                     for k, source in (("cd_mm", "geometry_mm"), ("rgb", "rgb"),
                                       ("alpha", "alpha"), ("joint", "objective"))}
        per_episode = {}
        for ep, count in EPISODE_COUNTS.items():
            row = episodes[ep]
            if not isinstance(row, Mapping) or _integer(row.get("samples"), f"{mode}.ep{ep}.samples") != count:
                raise ValueError(f"{mode}.ep{ep}: expected {count} Val windows")
            per_episode[ep] = _episode_metrics(row, f"{mode}.ep{ep}")
        for metric in aggregate:
            expected = math.fsum(v[metric] for v in per_episode.values()) / len(per_episode)
            _compatible(aggregate[metric], expected, metric, f"{mode}.episode_equal.{metric}")
        parsed[mode] = dict(all=aggregate, episodes=per_episode)

    has_identity = any(isinstance(r, Mapping) and "start" in r for r in horizon_rows)
    buckets = defaultdict(list)
    identities = defaultdict(set)
    for index, row in enumerate(horizon_rows):
        if not isinstance(row, Mapping) or row.get("split") != "val":
            raise ValueError(f"row {index}: only explicit Val rows are allowed")
        mode, ep = row.get("mode"), str(row.get("episode"))
        h = _integer(row.get("horizon"), f"row {index}.horizon", minimum=1)
        if mode not in parsed or ep not in EPISODE_COUNTS or h not in HORIZONS:
            raise ValueError(f"row {index}: unexpected mode, episode or horizon")
        clean = {k: _number(row.get(k), f"row {index}.{k}")
                 for k in ("cd_mm", "rgb", "alpha", "joint")}
        bucket = (mode, ep, h)
        if has_identity:
            start = _integer(row.get("start"), f"row {index}.start")
            if start in identities[bucket]:
                raise ValueError(f"row {index}: duplicate window/horizon")
            identities[bucket].add(start)
        buckets[bucket].append(clean)

    coverage = {}
    for mode in parsed:
        coverage[mode] = {}
        for ep, count in EPISODE_COUNTS.items():
            coverage[mode][ep] = {}
            for h in HORIZONS:
                bucket = (mode, ep, h)
                observed = len(buckets[bucket])
                if observed != count:
                    raise ValueError(f"{mode}.ep{ep}.H{h}: {observed} records, expected {count}")
                coverage[mode][ep][str(h)] = observed
                if has_identity and identities[bucket] != identities[(NORMAL, ep, 1)]:
                    raise ValueError(f"{mode}.ep{ep}.H{h}: window identities differ")
            for metric in ("cd_mm", "rgb", "alpha", "joint"):
                actual = math.fsum(_mean(buckets[(mode, ep, h)], metric) for h in HORIZONS) / len(HORIZONS)
                _compatible(actual, parsed[mode]["episodes"][ep][metric], metric,
                            f"{mode}.ep{ep}.{metric}")

    hard = [_comparison("user_whole_val", metric, parsed[NORMAL]["all"][metric],
                        parsed[JOINT_ABLATION]["all"][metric], EPISODE_COUNTS)
            for metric in MAX_RATIOS]
    episode_checks = [_comparison("conservative_episode", metric,
                                 parsed[NORMAL]["episodes"][ep][metric],
                                 parsed[JOINT_ABLATION]["episodes"][ep][metric], {ep: count}, episode=ep)
                      for ep, count in EPISODE_COUNTS.items() for metric in MAX_RATIOS]
    late = []
    for h in LATE_HORIZONS:
        for metric in MAX_RATIOS:
            for ep, count in EPISODE_COUNTS.items():
                late.append(_comparison("conservative_late_episode", metric,
                    _mean(buckets[(NORMAL, ep, h)], metric),
                    _mean(buckets[(JOINT_ABLATION, ep, h)], metric), {ep: count}, horizon=h, episode=ep))
            # Deliberately average two episode means, not all 100 windows.
            baseline = math.fsum(_mean(buckets[(NORMAL, ep, h)], metric)
                                 for ep in EPISODE_COUNTS) / len(EPISODE_COUNTS)
            value = math.fsum(_mean(buckets[(JOINT_ABLATION, ep, h)], metric)
                              for ep in EPISODE_COUNTS) / len(EPISODE_COUNTS)
            late.append(_comparison("conservative_late_episode_equal", metric,
                                    baseline, value, EPISODE_COUNTS, horizon=h))

    evidence = review or {}
    review_checks = [dict(name=flag, passed=evidence.get(flag) is True,
                          recorded_value=(evidence.get(flag) if isinstance(evidence.get(flag), bool) else None))
                     for flag in REVIEW_FLAGS]
    raw_paths = evidence.get("evidence_paths")
    paths_valid = (isinstance(raw_paths, (list, tuple)) and bool(raw_paths)
                   and all(isinstance(p, str) and bool(p.strip()) for p in raw_paths))
    paths = list(raw_paths) if paths_valid else []
    review_checks.append(dict(name="evidence_paths_nonempty", passed=bool(paths_valid), evidence_paths=paths))
    hard_ok = all(r["passed"] for r in hard)
    conservative_ok = all(r["passed"] for r in episode_checks + late)
    reviewed = all(r["passed"] for r in review_checks)
    if not hard_ok:
        status = "FAIL_STAGE1"
    elif not conservative_ok or not reviewed:
        status = "INCONCLUSIVE_NO_AUTOSTART"
    else:
        status = "PASS_ELIGIBLE"
    return dict(
        status=status, eligible_for_stage2=status == "PASS_ELIGIBLE",
        hard_gate_passed=hard_ok, conservative_numeric_passed=conservative_ok,
        review_passed=reviewed, baseline_mode=NORMAL, candidate_mode=JOINT_ABLATION,
        thresholds=dict(user_hard=dict(MAX_RATIOS, scope="whole Val100, episode equal"),
            conservative_auto_start_sufficient=dict(MAX_RATIOS,
                scope="each episode and every H10-H15, per episode plus episode equal",
                note="Stricter operational safeguards; not additional user-stated scientific thresholds."),
            floating_boundary_relative_tolerance=BOUNDARY_RTOL,
            summary_reproduction_absolute_tolerance=dict(REPRODUCTION_ATOL)),
        coverage=dict(modes=list(parsed), episode_windows=dict(EPISODE_COUNTS),
                      horizon_counts=coverage, rows=len(horizon_rows),
                      window_identities_validated=has_identity),
        checks=dict(user_hard=hard, conservative_episode=episode_checks,
                    conservative_late_horizon=late, review=review_checks),
        reasons=[r["scope"] + ":" + r["metric"]
                 + (":ep" + r["episode"] if "episode" in r else "")
                 + (":H" + str(r["horizon"]) if "horizon" in r else "")
                 for r in hard + episode_checks + late if not r["passed"]]
                + ["review:" + r["name"] for r in review_checks if not r["passed"]],
        interpretation=("Frozen checkpoint dependence only. Review flags assert evidence review, not "
                        "counterfactual GT, physical causality, or a requirement that robot/object "
                        "motion directions coincide. This function never launches a job."))
