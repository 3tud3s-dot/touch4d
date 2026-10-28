# Fixed initial regions identical to the audited H15 diagnostic.
import torch

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
