"""Differentiable loss decomposition and temporary observation-only hooks.

All gradients are in the original normalized coordinate/parameter units.
The model parameters require gradients but are never updated.
"""
from contextlib import contextmanager
import copy
import math
import numpy as np
import torch

from scripts.training.losses import chamfer_samples, render_samples, prediction_horizon
from scripts.training.v4_runtime import v4_objective
from scripts.experiments.fixed_window.fixed_fit import digest

LOSSES = ('cd', 'rgb', 'alpha_weighted', 'joint')
MODULES = {
    'motion': lambda n: n.startswith('motion.'),
    'motion_last': lambda n: n.startswith('motion.2.'),
    'cross': lambda n: n.startswith('cross.'),
    'cross_geometry_score': lambda n: n.startswith('cross.') and n.endswith('geometry_score'),
    'cross_pairwise': lambda n: n.startswith('cross.') and '.pairwise.' in n,
    'conditions_controller': lambda n: n.startswith('conditions.controller.'),
    'conditions_hand': lambda n: n.startswith('conditions.hand.'),
    'gs_encoder': lambda n: n.startswith('gs_encoder.'),
    'local': lambda n: n.startswith('local.'),
}


def components(pred, batch, processor, renderer, chunk=512, with_time=False):
    """Exact V4 reduction, with live CD and weighted alpha tensors."""
    prediction_horizon(pred['xyz'], batch)
    valid = batch['targets']['target_depth_valid']
    with torch.no_grad():
        target = renderer.render_targets(batch, processor)
    rendered = renderer(pred, batch, processor)
    rgb, available, rgb_time, times = render_samples(rendered['rgb'], target['rgb'].detach(), valid)
    alpha, av, alpha_time, at = render_samples(rendered['alpha'], target['alpha'].detach(), valid)
    if not torch.equal(available, av) or not torch.equal(times, at):
        raise ValueError('Different RGB/alpha availability')
    cd = chamfer_samples(pred['xyz'], batch, chunk)
    zero = pred['xyz'].sum()*0.
    out = {k: v[available].mean() if available.any() else zero
           for k, v in [('cd', cd), ('rgb', rgb), ('alpha_weighted', .1*alpha)]}
    # Match production addition order before batch reduction.
    out['joint'] = (rgb+.1*alpha+cd)[available].mean() if available.any() else zero
    if not all(torch.isfinite(z).all() for z in out.values()):
        raise FloatingPointError('Nonfinite decomposed loss')
    if not with_time:
        return out
    time_values = {k: [] for k in LOSSES}
    with torch.no_grad():
        for h in range(pred['xyz'].shape[1]):
            pp, bb = single_horizon(pred, batch, h)
            cv = chamfer_samples(pp['xyz'], bb, chunk)
            if not times[:, h].any():
                for key in LOSSES: time_values[key].append(None)
                continue
            c = float(cv[times[:, h]].mean()); r = float(rgb_time[times[:, h], h].mean())
            a = float((.1*alpha_time[times[:, h], h]).mean())
            for key, v in [('cd', c), ('rgb', r), ('alpha_weighted', a), ('joint', c+r+a)]:time_values[key].append(v)
    return out, time_values


def finite_gradient(g):
    if g is not None and not torch.isfinite(g).all():
        raise FloatingPointError('Nonfinite gradient: stop affected diagnostic')
    return g


def stats(value):
    """Norm/RMS refer to scalar coordinates; vector RMS is separately named."""
    if value is None:
        return dict(connected=False, l2=None, scalar_rms=None, abs_p95=None,
                    max_abs=None, zero_fraction=None, elements=0)
    v = finite_gradient(value.detach()).double().cpu()
    if not v.numel():
        raise ValueError('Empty selected diagnostic tensor')
    flat = v.flatten()
    out = dict(connected=True, l2=float(flat.norm()), scalar_rms=float(flat.square().mean().sqrt()),
               abs_p95=float(torch.quantile(flat.abs(), .95)), max_abs=float(flat.abs().max()),
               zero_fraction=float((flat == 0).double().mean()), elements=flat.numel())
    if v.ndim >= 2:
        out['signed_mean_last_axis'] = v.reshape(-1, v.shape[-1]).mean(0).tolist()
    if v.ndim >= 2 and v.shape[-1] == 3:
        norms = v.norm(dim=-1).flatten()
        out.update(vector_rms=float(norms.square().mean().sqrt()), vector_p95=float(torch.quantile(norms, .95)),
                   mean_vector=v.reshape(-1, 3).mean(0).tolist())
    return out


def cosine(a, b):
    if a is None or b is None:
        return None
    a, b = a.double().flatten(), b.double().flatten()
    if a.shape != b.shape:
        raise ValueError('Cosine must use identical coordinates')
    den = a.norm()*b.norm()
    return None if float(den) == 0. else float((a@b/den).clamp(-1, 1))


def cosines(grads):
    pairs = [('cd', 'rgb'), ('cd', 'alpha_weighted'), ('rgb', 'alpha_weighted'),
             ('cd', 'joint'), ('rgb', 'joint'), ('alpha_weighted', 'joint')]
    return {a+'__'+b: cosine(grads.get(a), grads.get(b)) for a, b in pairs}


def point_angles(a, b):
    if a is None or b is None:
        return dict(valid_vectors=0)
    a, b = a.double().reshape(-1, 3), b.double().reshape(-1, 3)
    den = a.norm(dim=-1)*b.norm(dim=-1); mask = den > 0
    c = ((a*b).sum(-1)[mask]/den[mask]).clamp(-1, 1)
    if not len(c):
        return dict(valid_vectors=0)
    return dict(valid_vectors=len(c), negative_fraction=float((c < 0).double().mean()),
                cosine_quantiles=torch.quantile(c, torch.tensor([0., .05, .25, .5, .75, .95, 1.], dtype=c.dtype)).tolist(),
                bins=[-1., -.5, 0., .5, 1.], counts=torch.histc(c, bins=4, min=-1, max=1).int().tolist())


@contextmanager
def observe(model, zero_tool=False):
    """Hooks are removed even on error. Intervention only replaces [128:140]."""
    captured = dict(motion_input=[], anchor_features=[])
    def before_motion(module, args):
        value = args[0]
        if zero_tool:
            value = torch.cat((value[..., :128], torch.zeros_like(value[..., 128:140]), value[..., 140:]), -1)
        captured['motion_input'].append(value)
        return (value,) if zero_tool else None
    def after_cross(module, args, output):
        captured['anchor_features'].append(output[0])
    handles = [model.motion.register_forward_pre_hook(before_motion),
               model.cross[-1].register_forward_hook(after_cross)]
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def altered_inputs(inputs, mode):
    x = copy.deepcopy(inputs)
    if mode == 'freeze_action':
        for n in ('action_state', 'controller_points'):
            x[n] = x[n][:, :1].expand_as(x[n]).clone()
        for n in ('action_valid', 'controller_mask'):
            x[n] = x[n] & x[n][:, :1]
    elif mode == 'zero_tactile_pressure':
        x['tactile'].zero_()
    else:
        raise ValueError(mode)
    return x


def same_state(model, before):
    after = digest(model.state_dict())
    if after != before:
        raise RuntimeError('Frozen model state changed')
    if model.training:
        raise RuntimeError('Model left eval state')
    return after


def verify_equivalence(model, batch, processor, renderer, chunk=512):
    """Independent official/decomposed evaluations; compare all parameter VJPs."""
    model.eval(); before = digest(model.state_dict())
    ps = list(model.parameters())
    pred = model(batch['inputs'])
    original, _ = v4_objective(pred, batch, processor, renderer, chunk)
    ga = torch.autograd.grad(original, [pred['xyz']]+ps, allow_unused=True)
    ga = [None if g is None else finite_gradient(g).detach().cpu() for g in ga]
    value = float(original.detach()); reference = pred['xyz'].detach().clone()
    del original, pred
    with observe(model) as capture:
        pred, trace = model(batch['inputs'], return_trace=True)
    torch.testing.assert_close(pred['xyz'], reference, rtol=0, atol=0)
    losses = components(pred, batch, processor, renderer, chunk)
    torch.testing.assert_close(losses['joint'].detach().cpu(), torch.tensor(value), rtol=1e-6, atol=1e-7)
    gb = torch.autograd.grad(losses['joint'], [pred['xyz']]+ps, allow_unused=True)
    max_abs = 0.; max_rel_l2 = 0.; none_pairs = 0
    for a, b in zip(ga, gb):
        finite_gradient(b)
        if a is None or b is None:
            if (a is None) != (b is None):
                raise AssertionError('Gradient connectivity differs from original objective')
            none_pairs += 1; continue
        b = b.detach().cpu()
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        diff = (a.double()-b.double()); max_abs = max(max_abs, float(diff.abs().max()))
        max_rel_l2 = max(max_rel_l2, float(diff.norm()/a.double().norm().clamp_min(1e-30)))
    same_state(model, before)
    return dict(passed=True, original=value, decomposed=float(losses['joint'].detach()),
                max_gradient_abs_error=max_abs, max_gradient_relative_l2=max_rel_l2,
                missing_gradient_pairs=none_pairs, compared_tensors=len(ga),
                hooked_output_bit_exact=True, state_sha256=before,
                loss_tolerance=dict(rtol=1e-6, atol=1e-7), gradient_tolerance=dict(rtol=2e-5, atol=2e-6))


def fixed_regions(batch, field):
    x = batch['inputs']; keep = x['gs_mask'][0]
    cp = x['controller_points'][0, 0].reshape(60, 3)
    cm = x['controller_mask'][0, 0].reshape(60) & x['operation_hands'][0].repeat_interleave(30)
    cm = cm & torch.isfinite(cp).all(-1)
    if not cm.any():
        raise ValueError('No initial active controller for fixed distance groups')
    dist = torch.cdist(x['gs_xyz'][0, keep][None], cp[cm][None], compute_mode='donot_use_mm_for_euclid_dist')[0].min(-1).values
    ix = keep.nonzero().flatten(); order = dist.argsort(stable=True); n = max(1, len(order)//4)
    gs = dict(all=ix, near=ix[order[:n]], far=ix[order[-n:]])
    ai = field.anchor_indices[0, field.anchor_mask[0]]
    ad = torch.cdist(x['gs_xyz'][0, ai][None], cp[cm][None], compute_mode='donot_use_mm_for_euclid_dist')[0].min(-1).values
    valid_ai = field.anchor_mask[0].nonzero().flatten(); ao = ad.argsort(stable=True); n = max(1, len(ao)//4)
    anchors = dict(all=valid_ai, near=valid_ai[ao[:n]], far=valid_ai[ao[-n:]])
    return gs, anchors


def rms_mm(x, scale_mm):
    return float(x.detach().double().square().sum(-1).mean().sqrt())*scale_mm


def single_horizon(pred, batch, h):
    b = dict(batch); y = dict(batch['targets'])
    for key in ('target_xyz', 'target_mask', 'target_attr'):
        y[key] = [y[key][h]]
    for key in ('target_depth', 'target_depth_valid'):
        y[key] = y[key][:, h:h+1]
    b['targets'] = y
    return {k: v[:, h:h+1] for k, v in pred.items()}, b


def normalized_direction(value, mask):
    out = torch.where(mask[..., None], value, 0.)
    norm = out[mask].double().square().sum(-1).mean().sqrt()
    if not torch.isfinite(norm) or float(norm) == 0.:
        return None
    return out/norm.to(out.dtype)


def parameter_summary(named, gradients):
    result = {}
    for group, select in MODULES.items():
        ids = [i for i, (n, _) in enumerate(named) if select(n)]
        if not ids:
            raise ValueError('Requested parameter group does not exist: '+group)
        size = sum(named[i][1].numel() for i in ids)
        vectors = {}
        rows = {}
        for loss, gg in gradients.items():
            parts = [gg[i] for i in ids]
            missing = sum(named[i][1].numel() for i, g in zip(ids, parts) if g is None)
            vec = torch.cat([(torch.zeros(named[i][1].numel()) if g is None else g.flatten()) for i, g in zip(ids, parts)])
            vectors[loss] = vec
            rows[loss] = dict(**stats(vec), missing_elements=missing, missing_fraction=missing/size,
                              missing_tensors=sum(g is None for g in parts), tensors=len(ids))
            rows[loss]['connected'] = any(g is not None for g in parts)
        result[group] = dict(parameter_count=size, parameter_names=[named[i][0] for i in ids],
                             losses=rows, cosines=cosines(vectors))
    return result
