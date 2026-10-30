"""Frozen G5-Direct interventions; original parameters and rollout are untouched.

Only Controller Value projections and the original Motion Head tool/mass
channels are intervened. All Base/global/local heads remain active. Six fixed
modes separate the three paths; no new training or architecture is introduced.

Hooks retain the original FFNs and score computations.  Captures retain tensor
references on their original device; no detach, CPU synchronization or optimizer
is introduced here.  Use one full forward per intervention context.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch

from scripts.training.gs_decoder import interpolate_anchor_displacements
from scripts.training.model import masked


@dataclass(frozen=True)
class Mode:
    cross_value: bool = True
    tool_motion: bool = True
    close_mass: bool = False


MODES = ("Normal", "NoCrossValue", "NoToolMotion12", "NoHeadMass4",
         "NoValue+NoTool12", "NoCrossInfo")
SELECTED = (1, 5, 10, 15)
MODE_CONFIGS = dict(zip(MODES, (
    Mode(), Mode(cross_value=False), Mode(tool_motion=False), Mode(close_mass=True),
    Mode(cross_value=False, tool_motion=False),
    Mode(cross_value=False, tool_motion=False, close_mass=True),
)))


def resolve_mode(mode):
    if isinstance(mode, Mode):
        if any(type(v) is not bool for v in asdict(mode).values()):
            raise ValueError("Mode switches must be bool")
        return mode
    if not isinstance(mode, str) or mode not in MODE_CONFIGS:
        raise ValueError(f"Unknown intervention mode: {mode!r}")
    return MODE_CONFIGS[mode]


def metadata(mode):
    return dict(name=mode if isinstance(mode, str) else "custom", **asdict(resolve_mode(mode)))


def switches(mode, step):
    """Return effective Base/Value/Tool switches and mass closure, zero-based step."""
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative zero-based integer")
    m = resolve_mode(mode)
    return (True, m.cross_value, m.tool_motion, m.close_mass)


def _check_model(model):
    if getattr(model, "use_propagation", None) is not False or len(model.cross) != 2:
        raise ValueError("This frozen audit requires the original two-Cross G5-Direct")


def _check_step(model, step):
    if type(step) is not int or not 0 <= step < model.config["horizon"]:
        raise ValueError("Invalid zero-based intervention step")


def _motion_input(value, tool, close_mass):
    if value.shape[-1] != 151:
        raise ValueError("Expected original 151D G0 Motion Head input")
    if tool and not close_mass:
        return value
    result = value.clone()
    if not tool:
        result[..., 128:140] = 0
    if close_mass:
        result[..., 146:150] = 0
    return result


@contextmanager
def intervention(model, mode="Normal", only_step=None):
    """Temporarily suppress paths and capture the executed rollout's raw context.

    ``rows[step]`` contains raw Base/global/local outputs, both Cross calls,
    original/effective Motion inputs, and global/direct args+kwargs.  A raw Base
    may contain padding bias; mask it before interpreting an Anchor field.
    ``only_step`` leaves the whole prefix untouched and changes only that step.
    Later autoregressive changes are not a same-history direct effect.
    """
    _check_model(model)
    resolve_mode(mode)
    if only_step is not None:
        _check_step(model, only_step)
    rows, handles = [], []

    def current():
        if not rows:
            raise RuntimeError("Cross/Head called before Condition Encoder")
        return rows[-1]

    def enabled():
        step = current()["step"]
        return switches(mode if only_step is None or step == only_step else "Normal", step)

    def conditions_pre(module, args):
        step = int(args[1])
        if step != len(rows) or step >= model.config["horizon"]:
            raise ValueError("Use a fresh intervention context for one full rollout")
        rows.append(dict(step=step, cross_calls=[], cross_projection_raw=[],
                         capture_mode=metadata(mode), capture_is_normal=resolve_mode(mode) == Mode()))

    handles.append(model.conditions.register_forward_pre_hook(conditions_pre))

    for index, block in enumerate(model.cross):
        def cross_pre(module, args, kwargs, index=index):
            row = current()
            if len(row["cross_calls"]) != index:
                raise AssertionError("Unexpected Cross execution order")
            row["cross_calls"].append(dict(args=args, kwargs=dict(kwargs)))

        def cross_post(module, args, kwargs, output, index=index):
            current()["cross_calls"][index]["output"] = output

        def projection(module, args, output):
            current()["cross_projection_raw"].append(output)
            return output if enabled()[1] else output * 0

        handles.append(block.register_forward_pre_hook(cross_pre, with_kwargs=True))
        handles.append(block.register_forward_hook(cross_post, with_kwargs=True))
        handles.append(block.projection.register_forward_hook(projection))

    def motion_pre(module, args):
        row = current()
        row["motion_input"] = args[0]
        _, _, tool, close_mass = enabled()
        effective = _motion_input(args[0], tool, close_mass)
        row["effective_motion_input"] = effective
        return (effective, *args[1:])

    def motion_post(module, args, output):
        row = current()
        row["base"] = output
        effective = output if enabled()[0] else output * 0
        row["effective_base"] = effective
        return effective

    def global_pre(module, args, kwargs):
        current()["global_call"] = dict(args=args, kwargs=dict(kwargs))

    def global_post(module, args, kwargs, output):
        current()["global"], current()["global_trace"] = output

    def direct_pre(module, args, kwargs):
        current()["direct_call"] = dict(args=args, kwargs=dict(kwargs))

    def direct_post(module, args, kwargs, output):
        seed, trace = output
        current().update(local=seed.sum(2), seed=seed, direct_trace=trace)

    handles.extend((model.motion.register_forward_pre_hook(motion_pre),
                    model.motion.register_forward_hook(motion_post),
                    model.global_motion.register_forward_pre_hook(global_pre, with_kwargs=True),
                    model.global_motion.register_forward_hook(global_post, with_kwargs=True),
                    model.direct_motion.register_forward_pre_hook(direct_pre, with_kwargs=True),
                    model.direct_motion.register_forward_hook(direct_post, with_kwargs=True)))
    try:
        yield rows
    finally:
        for handle in reversed(handles):
            handle.remove()


def capture_normal(model, inputs):
    with intervention(model) as rows:
        prediction, trace = model(inputs, return_trace=True)
    return prediction, trace, rows


@contextmanager
def _without_cross_value(model, enabled):
    handles = []
    if not enabled:
        for block in model.cross:
            handles.append(block.projection.register_forward_hook(
                lambda module, args, output: output * 0))
    try:
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


def replay_step(model, inputs, normal_trace, raw, step, mode="Normal"):
    """Recompute only one step on fixed Normal history, without GS re-encoding.

    Capture with ``intervention(model, 'Normal')`` first.  Reuse the first Cross
    input Query and all Normal history/geometry, recompute both Cross blocks,
    Alpha-based tool motion and mass, G0, G4 and G5.  In particular G4/global and
    G5/local outputs MUST NOT be reused when the Query is changed.

    Return a dict whose ``xyz`` is [B,N,3], plus raw/effective branch fields.
    No future targets, GT states, optimizer, detach or device transfer is used.
    """
    _check_model(model)
    _check_step(model, step)
    base, value, tool, close_mass = switches(mode, step)
    if len(raw) != model.config["horizon"] or len(normal_trace["steps"]) != len(raw):
        raise ValueError("Replay requires captures of one complete Normal rollout")
    if not all(row.get("capture_is_normal", False) for row in raw):
        raise ValueError("Replay requires Normal captures, not an intervened history")
    row = raw[step]
    if row["step"] != step or len(row["cross_calls"]) != 2:
        raise ValueError("Missing ordered Normal Cross captures")
    # A separate replay inside active capture hooks would corrupt that context.
    if model.conditions._forward_pre_hooks or model.motion._forward_hooks or model.motion._forward_pre_hooks:
        raise ValueError("Replay must run after the capture context has exited")
    query = row["cross_calls"][0]["args"][0]
    with _without_cross_value(model, value):
        for block, call in zip(model.cross, row["cross_calls"]):
            args = (query, *call["args"][1:])
            query, alpha = block(*args, **call["kwargs"])

    global_call = row["global_call"]
    gkwargs = dict(global_call["kwargs"], query=query)
    cp, valid = gkwargs["cp"], gkwargs["cp_valid"]
    adjacent = torch.zeros_like(valid) if step == 0 else valid & gkwargs["previous_valid"]
    delta = torch.zeros_like(cp) if step == 0 else masked(cp-gkwargs["previous_cp"], adjacent)
    weights = alpha[..., :60]
    tool_motion = torch.einsum("bahj,bjd->bahd", weights, delta).flatten(-2)
    head_mass = (weights*adjacent[:, None, None]).sum(-1)
    original = row["motion_input"]
    # Preserve the original history, previous cumulative displacement and flag.
    motion_input = torch.cat((query, tool_motion, original[..., 140:146],
                              head_mass, original[..., 150:151]), -1)
    effective_input = _motion_input(motion_input, tool, close_mass)
    field, keep = normal_trace["field"], inputs["gs_mask"]
    raw_base = model.motion(effective_input)
    base_d = masked(raw_base if base else raw_base*0, field.anchor_mask)
    global_delta, global_trace = model.global_motion(inputs, step, **gkwargs)
    dkwargs = dict(row["direct_call"]["kwargs"], query=query,
                  hand_basis12=global_trace["hand_basis"], hand_valid=global_trace["hand_valid"],
                  global_hand_contribution=global_trace["hand_contribution"])
    seed, direct_trace = model.direct_motion(inputs, step, **dkwargs)
    local = masked(seed.sum(2), field.anchor_mask)
    cumulative = masked(masked(base_d+global_delta[:, None], field.anchor_mask)+local,
                        field.anchor_mask)
    position = masked(masked(inputs["gs_xyz"], keep)+interpolate_anchor_displacements(
        cumulative, field, keep), keep)
    return dict(xyz=position, query=query, alpha=alpha, tool_motion=tool_motion,
                head_history_mass=head_mass, adjacent_valid=adjacent,
                motion_input=motion_input, effective_motion_input=effective_input,
                base=raw_base, base_cumulative_d=base_d, global_delta=global_delta,
                local_correction=local, direct_seed_per_hand=seed,
                cumulative_d=cumulative, previous_d=original[..., 143:146],
                global_trace=global_trace, direct_trace=direct_trace)


def same_history_position(model, inputs, normal_trace, raw, step, mode="Normal"):
    return replay_step(model, inputs, normal_trace, raw, step, mode)["xyz"]


def only_step_forward(model, inputs, mode, step, *, return_trace=False):
    with intervention(model, mode, only_step=step):
        return model(inputs, return_trace=return_trace)


def branch_parts(step, field):
    mask = field.anchor_mask
    base = step["base_cumulative_d"]
    return dict(base=masked(base, mask),
                global_=masked(step["global_delta"][:, None].expand_as(base), mask),
                local=masked(step["local_correction"], mask))


def combine(parts, field):
    """Exactly the production's two additions and masks, not a reassociation."""
    return masked(masked(parts["base"]+parts["global_"], field.anchor_mask)+parts["local"],
                  field.anchor_mask)


def one_frame(batch, prediction, index):
    """Loss/render view only; never feed this sliced batch into a predictor."""
    targets = dict(batch["targets"])
    for key in ("target_xyz", "target_mask", "target_attr"):
        if key in targets:
            targets[key] = [targets[key][index]]
    for key in ("target_depth", "target_depth_valid"):
        if key in targets:
            targets[key] = targets[key][:, index:index+1]
    meta = dict(batch["meta"])
    for key in ("frame_ids", "observation_timestamp_us"):
        if key in meta:
            meta[key] = meta[key][:, index:index+2]
    return dict(batch, targets=targets, meta=meta), {
        key: value[:, index:index+1] for key, value in prediction.items()}
