"""Controlled, reversible loss/gradient/AdamW diagnostics; no architecture change.

Call only inside the allocation owned by the audit when the model is on CUDA.
Every comparison starts with the same frozen parameters and optimizer moments.
Raw gradients, globally clipped gradients and actual AdamW deltas are distinct.
"""
import copy
import math

import torch

from scripts.training.losses import objective, chamfer_samples


FULL_WEIGHTS = dict(chamfer_weight=1., rgb_weight=1., alpha_weight=.1,
                    scale_weight=.01)
COMPONENTS = {
    "cd": dict(chamfer_weight=1., rgb_weight=0., alpha_weight=0., scale_weight=0.),
    "rgb": dict(chamfer_weight=0., rgb_weight=1., alpha_weight=0., scale_weight=0.),
    "alpha": dict(chamfer_weight=0., rgb_weight=0., alpha_weight=1., scale_weight=0.),
    "full": FULL_WEIGHTS,
}
CONDITION_PREFIXES = ("state_encoder", "controller_encoder", "pressure_encoder",
                      "fusion", "fallback_encoder", "tactile_encoder",
                      "token_identity", "condition_time")
LABELS = {"blocks": "condition_cross_attention",
          "gs_structure": "structural_attention",
          "position_head": "anchor_displacement_head"}


def _cpu_clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(v) for v in value)
    return copy.deepcopy(value)


def _groups(named):
    groups = {}
    for name in named:
        prefix = name.split(".")[0]
        groups.setdefault(LABELS.get(prefix, prefix), []).append(name)
    groups["condition_encoders_combined"] = [
        n for n in named if n.split(".")[0] in CONDITION_PREFIXES]
    groups["all_parameters"] = list(named)
    return groups


def _norm(values, names):
    return math.sqrt(sum(float(values[n].double().square().sum())
                         for n in names if values.get(n) is not None))


def cosine(left, right, names):
    """Missing or zero vectors have no defined angle: report null, not zero."""
    nl, nr = _norm(left, names), _norm(right, names)
    if nl == 0 or nr == 0:
        return None
    dot = sum(float((left[n].double() * right[n].double()).sum())
              for n in names if left.get(n) is not None and right.get(n) is not None)
    return max(-1., min(1., dot / (nl * nr)))


def parameter_groups(model):
    return _groups(dict(model.named_parameters()))


def _gradient_values(named):
    values = {}
    for name, parameter in named.items():
        grad = parameter.grad
        if grad is not None and not torch.isfinite(grad).all():
            raise FloatingPointError(f"Nonfinite gradient: {name}")
        values[name] = None if grad is None else grad.detach().cpu().clone()
    return values


def _prediction_change(before, after, keep, scale):
    difference = after["xyz"] - before["xyz"]
    # RMS of Euclidean per-point displacement, not RMS over scalar coordinates.
    return [float(difference[:, h][keep].square().sum(-1).mean().sqrt()) * scale * 1000
            for h in range(difference.shape[1])]


def audit_gradients(model, batch, processor, renderer, optimizer_state=None):
    """One reversible update per loss from a common checkpoint and moment state.

    Passing the checkpoint's optimizer state measures the next AdamW update with
    its existing moments. Without it, measurements use fresh zero moments. This
    difference is explicitly recorded and must not be conflated with raw-gradient
    directions. The supplied model's weights, buffers, grads, modes and RNG are
    restored even on failure. No checkpoint or dataset file is written.
    """
    named = dict(model.named_parameters())
    if not named or not all(p.requires_grad for p in named.values()):
        raise ValueError("Audit expects all model parameters trainable")
    device = next(iter(named.values())).device
    groups = _groups(named)
    initial = _cpu_clone(model.state_dict())
    initial_params = {name: initial[name] for name in named}
    original_grads = _gradient_values(named)
    original_modes = {name: module.training for name, module in model.named_modules()}
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    # Avoid retaining five-step diagnostic graphs after this function returns.
    original_trace = getattr(model, "trace", None)
    moments = _cpu_clone(optimizer_state) if optimizer_state is not None else None
    scale = float(processor.state_dict()["scale"])
    raw_vectors, clipped_vectors, delta_vectors = {}, {}, {}
    results = dict(
        optimizer="AdamW", lr=1e-4, weight_decay=1e-4, clip_norm=1.,
        optimizer_start="checkpoint_moments" if moments is not None else "fresh_zero_moments",
        units=dict(cd="normalized unsquared L2", rgb="unit-range RGB MAE",
                   alpha="unit-range alpha MAE", parameter_update="parameter native units",
                   output_change="mm"),
        parameter_group_definitions=groups, comparisons={},
        note="Gradient cosines describe this checkpoint/batch, not a general causal proof. "
             "AdamW update directions include restored moments and weight decay.",
    )
    try:
        for component, weights in COMPONENTS.items():
            model.load_state_dict(initial, strict=True)
            model.train()
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            if moments is not None:
                optimizer.load_state_dict(_cpu_clone(moments))
            if any(group["lr"] != 1e-4 or group["weight_decay"] != 1e-4
                   for group in optimizer.param_groups):
                raise ValueError("Frozen optimizer settings disagree with audited protocol")
            optimizer_ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
            if len(optimizer_ids) != len(named) or set(optimizer_ids) != {id(p) for p in named.values()}:
                raise ValueError("Optimizer does not cover every model parameter exactly once")
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["inputs"])
            before_xyz = {"xyz": prediction["xyz"].detach().clone()}
            loss, values = objective(prediction, batch, processor, renderer,
                                     0., 512, **weights)
            if not values["render_time_available"].all():
                raise ValueError("Missing target render supervision in audit batch")
            unweighted = {k: float(values[k].detach().mean())
                          for k in ("geometry", "rgb", "alpha", "scale_regularization")}
            loss.backward()
            raw = _gradient_values(named)
            preclip = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1., error_if_nonfinite=True))
            clipped = _gradient_values(named)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in named.values()):
                raise FloatingPointError("Nonfinite AdamW update")
            delta = {name: p.detach().cpu() - initial_params[name] for name, p in named.items()}
            modules = {}
            for group, names in groups.items():
                pn = _norm(initial_params, names)
                dn = _norm(delta, names)
                modules[group] = dict(
                    parameter_norm=pn, raw_gradient_norm=_norm(raw, names),
                    clipped_gradient_norm=_norm(clipped, names), actual_update_norm=dn,
                    relative_update_norm=dn / pn if pn else None,
                    parameter_count=sum(named[n].numel() for n in names),
                    tensors_with_gradient=sum(raw[n] is not None for n in names),
                    tensors_with_nonzero_gradient=sum(raw[n] is not None and bool(raw[n].any()) for n in names),
                    tensors_updated=sum(bool(delta[n].any()) for n in names))
            del prediction
            if hasattr(model, "trace"):
                model.trace = []
            with torch.no_grad():
                after = model(batch["inputs"])
                change = _prediction_change(before_xyz, after, batch["inputs"]["gs_mask"], scale)
                after_cd = float(chamfer_samples(after["xyz"], batch, 512).mean())
            results[component] = dict(
                loss=float(loss.detach()), weights=weights, unweighted_components=unweighted,
                global_gradient_norm_before_clip=preclip,
                global_gradient_norm_after_clip=_norm(clipped, list(named)),
                effective_clip_multiplier=min(1., 1. / (preclip + 1e-6)),
                modules=modules, prediction_change_mm_by_horizon=change,
                cd_before_mm=unweighted["geometry"] * scale * 1000,
                cd_after_one_update_mm=after_cd * scale * 1000)
            raw_vectors[component], clipped_vectors[component], delta_vectors[component] = raw, clipped, delta
            del after, before_xyz, loss, values, optimizer
            if hasattr(model, "trace"):
                model.trace = []
        # Include alpha's registered .1 weighting when measuring the combined
        # rendering objective; individual alpha gradients above are unweighted.
        render_vectors = {}
        for n in named:
            rgb, alpha = raw_vectors["rgb"][n], raw_vectors["alpha"][n]
            render_vectors[n] = (None if rgb is None and alpha is None else
                                 (torch.zeros_like(initial_params[n]) if rgb is None else rgb)
                                 + .1 * (torch.zeros_like(initial_params[n]) if alpha is None else alpha))
        raw_vectors["rgb_plus_0.1alpha"] = render_vectors
        for a, b in (("cd", "rgb"), ("cd", "alpha"), ("cd", "rgb_plus_0.1alpha"), ("cd", "full")):
            key = f"{a}_vs_{b}"
            results["comparisons"][key] = {
                group: dict(raw_gradient_cosine=cosine(raw_vectors[a], raw_vectors[b], names),
                            clipped_gradient_cosine=(cosine(clipped_vectors[a], clipped_vectors[b], names)
                                if b in clipped_vectors else None),
                            actual_adamw_update_cosine=(cosine(delta_vectors[a], delta_vectors[b], names)
                                if b in delta_vectors else None))
                for group, names in groups.items()}
    finally:
        model.load_state_dict(initial, strict=True)
        for name, parameter in named.items():
            grad = original_grads[name]
            parameter.grad = None if grad is None else grad.to(parameter.device)
        for name, module in model.named_modules():
            module.training = original_modes[name]
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        if hasattr(model, "trace"):
            model.trace = original_trace
    results["model_state_restored_exact"] = all(
        torch.equal(value.detach().cpu(), initial[name]) for name, value in model.state_dict().items())
    if not results["model_state_restored_exact"]:
        raise RuntimeError("Gradient audit changed its frozen model")
    return results
