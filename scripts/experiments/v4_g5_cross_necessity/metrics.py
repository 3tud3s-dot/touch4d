"""Read-only world statistics, adapted from frozen audit 18704f4.

Uses unchanged production fixed_regions/motion_metrics and RBF. No model code
or training logic is copied. Near-zero directions/projections are undefined.
"""
import math
import torch
from scripts.training.gs_decoder import interpolate_anchor_displacements
from scripts.experiments.v4_g4_global_motion.evaluate import fixed_regions, motion_metrics
from .interventions import branch_parts, combine

ANGLE_EPS_MM = 1e-6  # numerical definition only, never contact/motion gating


def rms(value):
    return float(value.double().square().sum(-1).mean().sqrt())


def vector_stats(value):
    if not len(value):
        return dict(count=0, mean_x_mm=None, mean_y_mm=None, mean_z_mm=None,
                    rms_mm=None, demeaned_rms_mm=None, norm_p95_mm=None)
    v = value.double()
    mean = v.mean(0)
    return dict(count=len(v), mean_x_mm=float(mean[0]), mean_y_mm=float(mean[1]),
                mean_z_mm=float(mean[2]), rms_mm=rms(v),
                demeaned_rms_mm=rms(v-mean),
                norm_p95_mm=float(torch.quantile(v.norm(dim=-1), .95)))


def angle_stats(first, second, prefix):
    a, b = first.double(), second.double()
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    valid = (na > ANGLE_EPS_MM) & (nb > ANGLE_EPS_MM)
    cosine = ((a[valid]*b[valid]).sum(-1)/(na[valid]*nb[valid])).clamp(-1, 1)
    return {prefix+"defined": int(valid.sum()), prefix+"undefined": int((~valid).sum()),
            prefix+"cos_mean": float(cosine.mean()) if len(cosine) else None,
            prefix+"opposite_fraction": float((cosine < 0).double().mean()) if len(cosine) else None}


def cancellation_stats(base, rest, final):
    b, r, f = (v.double() for v in (base, rest, final))
    bn, rn, fn = (v.norm(dim=-1) for v in (b, r, f))
    denominator = rms(b)+rms(r)
    return dict(
        **angle_stats(b, r, "base_vs_rest_"),
        **angle_stats(b, f, "base_vs_final_"),
        **angle_stats(r, f, "rest_vs_final_"),
        base_rms_mm=rms(b), correction_rms_mm=rms(r), final_rms_mm=rms(f),
        # RMS triangle-based cancellation INDEX, not mechanical energy.
        cancellation_index=1-rms(f)/denominator if denominator > ANGLE_EPS_MM else None,
        base_to_final_rms=rms(b)/rms(f) if rms(f) > ANGLE_EPS_MM else None,
        base_larger_than_final_fraction=float((bn > fn+ANGLE_EPS_MM).double().mean()),
        nearly_cancelled_fraction=float(((bn+rn > ANGLE_EPS_MM) &
                                        (fn < .1*(bn+rn))).double().mean()),
    )


def projection_stats(branch, final):
    """Signed projection onto final vector; undefined final is never zero-filled."""
    a, f = branch.double(), final.double()
    norm = f.norm(dim=-1)
    valid = norm > ANGLE_EPS_MM
    if not valid.any():
        return dict(projection_defined=0, projection_undefined=len(f),
                    signed_projection_mean_mm=None, final_fraction_mean=None)
    dot = (a[valid]*f[valid]).sum(-1)
    return dict(projection_defined=int(valid.sum()), projection_undefined=int((~valid).sum()),
                signed_projection_mean_mm=float((dot/norm[valid]).mean()),
                final_fraction_mean=float((dot/norm[valid].square()).mean()))


def decompose(inputs, trace, scale):
    """B=1 audit. Anchor controls and actual GS fields have separate records."""
    if len(inputs["gs_xyz"]) != 1:
        raise ValueError("Audit aggregation expects batch size one")
    field, factor = trace["field"], scale*1000
    regions = fixed_regions(inputs)
    anchor_ids = field.anchor_indices[0]
    anchor_keep = field.anchor_mask[0]
    anchor_regions = {"all": anchor_keep.nonzero().flatten()}
    for name, ids in regions.items():
        if name != "all":
            anchor_regions[name] = (torch.isin(anchor_ids, ids) & anchor_keep).nonzero().flatten()
    branch_rows, cancel_rows, equivalence = [], [], []
    for h, step in enumerate(trace["steps"], 1):
        parts = branch_parts(step, field)
        total = combine(parts, field)
        torch.testing.assert_close(total, step["cumulative_d"], rtol=0, atol=0)
        projected = {name: interpolate_anchor_displacements(value, field, inputs["gs_mask"])
                     for name, value in parts.items()}
        final_gs = interpolate_anchor_displacements(total, field, inputs["gs_mask"])
        error = float((sum(projected.values())-final_gs).abs().max())*factor
        if error > 1e-3:  # 1 micrometre, FP32 addition/interpolation roundoff
            raise AssertionError(("Nonlinear/mismatched RBF decomposition", error))
        reconstructed = inputs["gs_xyz"]+final_gs
        keep = inputs["gs_mask"]
        torch.testing.assert_close(reconstructed[keep], step["state"][keep], rtol=0, atol=0)
        equivalence.append(dict(horizon=h, anchor_sum_bit_exact=True, xyz_bit_exact=True,
                                projected_sum_max_error_mm=error))
        for space, fields, groups in (
            ("anchor_control", {**parts, "final": total}, anchor_regions),
            ("gs_actual", {**projected, "final": final_gs}, regions),
        ):
            fields = {name: value[0]*factor for name, value in fields.items()}
            fields["correction"] = fields["global_"]+fields["local"]
            for region, ids in groups.items():
                if not len(ids):
                    continue
                for name, value in fields.items():
                    branch_rows.append(dict(horizon=h, space=space, region=region, branch=name,
                        **vector_stats(value[ids]),
                        **projection_stats(value[ids], fields["final"][ids]),
                        **angle_stats(value[ids], fields["final"][ids], "vs_final_")))
                cancel_rows.append(dict(horizon=h, space=space, region=region,
                    **cancellation_stats(fields["base"][ids], fields["correction"][ids],
                                         fields["final"][ids])))
    return branch_rows, cancel_rows, equivalence


def aggregate(rows, group_keys, value_keys):
    """Windows equal within episode; episodes equal. Undefined values omitted
    WITH counts, never silently converted to zero or treated as independent GS.
    """
    groups = {}
    for row in rows:
        if row.get("split") != "val":
            continue
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, {}).setdefault(str(row["episode"]), []).append(row)
    result = []
    for key, episodes in sorted(groups.items()):
        values = dict(zip(group_keys, key))
        values["episode_window_counts"] = {ep:len(items) for ep,items in episodes.items()}
        for name in value_keys:
            means = []
            defined = {}
            for ep, items in episodes.items():
                numbers = [float(r[name]) for r in items if r.get(name) is not None]
                if any(not math.isfinite(v) for v in numbers):
                    raise FloatingPointError(name)
                defined[ep] = len(numbers)
                if numbers:
                    means.append(sum(numbers)/len(numbers))
            values[name] = sum(means)/len(means) if means else None
            values[name+"_defined_windows"] = defined
        result.append(values)
    return result
