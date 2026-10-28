"""Exact audited interventions from 0934d8c; no model implementation changes."""
from contextlib import contextmanager
import copy
import torch

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
