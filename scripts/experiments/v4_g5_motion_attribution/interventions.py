"""Observe or suppress final branch outputs, never alter the production rollout.

NoGlobal preserves raw hand_contribution in the global module's trace: Direct
Mapping reads that context, so zeroing it would be a different intervention.
"""
from contextlib import contextmanager
import torch
from scripts.training.model import masked
from scripts.training.gs_decoder import interpolate_anchor_displacements

MODES = ("Normal", "NoBase-All", "NoBase-H2+", "NoGlobal", "NoLocal", "BaseOnly")
SELECTED = (1, 5, 10, 15)


def switches(mode, step):
    if mode not in MODES or type(step) is not int or step < 0:
        raise ValueError((mode, step))
    return (
        not (mode == "NoBase-All" or mode == "NoBase-H2+" and step > 0),
        mode not in ("NoGlobal", "BaseOnly"),
        mode not in ("NoLocal", "BaseOnly"),
    )


@contextmanager
def intervention(model, mode="Normal", only_step=None):
    """A single forward per context; capture raw outputs on that AR history.

    All upstream modules still execute. previous_d and current GS naturally use
    the effective output on later steps. only_step leaves the entire prefix
    unmodified, providing an independent check of same-history recomposition.
    """
    switches(mode, 0)
    if only_step is not None and (type(only_step) is not int or
                                  not 0 <= only_step < model.config["horizon"]):
        raise ValueError("Invalid intervention step")
    if model.use_propagation:
        raise ValueError("This audit is restricted to G5-Direct")
    rows = []

    def selected(step):
        return switches(mode if only_step is None or step == only_step else "Normal", step)

    def base_hook(module, args, value):
        step = len(rows)
        if step >= model.config["horizon"]:
            raise ValueError("Use a fresh context for each forward")
        rows.append(dict(step=step, base=value))
        return value if selected(step)[0] else torch.zeros_like(value)

    def global_hook(module, args, output):
        step = int(args[1])
        if step != len(rows)-1:
            raise AssertionError("Unexpected branch execution order")
        delta, trace = output
        rows[step]["global"] = delta
        # Preserve trace and its raw per-hand contributions for Local context.
        return (delta if selected(step)[1] else torch.zeros_like(delta)), trace

    def local_hook(module, args, output):
        seed, trace = output
        step = int(args[1])
        rows[step]["local"] = seed.sum(2)
        return (seed if selected(step)[2] else torch.zeros_like(seed)), trace

    handles = [
        model.motion.register_forward_hook(base_hook),
        model.global_motion.register_forward_hook(global_hook),
        model.direct_motion.register_forward_hook(local_hook),
    ]
    try:
        yield rows
    finally:
        for handle in handles:
            handle.remove()


def branch_parts(step, field):
    mask = field.anchor_mask
    return dict(
        base=masked(step["base_cumulative_d"], mask),
        global_=masked(step["global_delta"][:, None].expand_as(step["base_cumulative_d"]), mask),
        local=masked(step["local_correction"], mask),
    )


def combine(parts, field, mode="Normal", step=0):
    b, g, l = switches(mode, step)
    base = parts["base"] if b else torch.zeros_like(parts["base"])
    glob = parts["global_"] if g else torch.zeros_like(parts["global_"])
    local = parts["local"] if l else torch.zeros_like(parts["local"])
    # Preserve both production addition/masking operations and their order.
    return masked(masked(base+glob, field.anchor_mask)+local, field.anchor_mask)


def same_history_position(inputs, normal_trace, step, mode):
    field = normal_trace["field"]
    d = combine(branch_parts(normal_trace["steps"][step], field), field, mode, step)
    return masked(inputs["gs_xyz"]+interpolate_anchor_displacements(
        d, field, inputs["gs_mask"]), inputs["gs_mask"])


def one_frame(batch, prediction, index):
    """Loss/render view only; NEVER feed the sliced view to a predictor."""
    y = batch["targets"]
    targets = dict(y)
    for key in ("target_xyz", "target_mask", "target_attr"):
        if key in y:
            targets[key] = [y[key][index]]
    for key in ("target_depth", "target_depth_valid"):
        if key in y:
            targets[key] = y[key][:, index:index+1]
    meta = dict(batch["meta"])
    for key in ("frame_ids", "observation_timestamp_us"):
        if key in meta:
            meta[key] = meta[key][:, index:index+2]
    view = dict(batch, targets=targets, meta=meta)
    pred = {key: value[:, index:index+1] for key, value in prediction.items()}
    return view, pred
