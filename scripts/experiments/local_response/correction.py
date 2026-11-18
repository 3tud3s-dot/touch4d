"""Experimental algebraic guard for a future Local-response head.

No learned head, graph builder, model integration, optimizer or loss is provided.
All displacements use the same units. W must be the original fixed RBF matrix.
Preserving the GS mean is not conservation of physical momentum.
"""
import math
import torch


def preserve_transport(local, candidate, weights, gs_valid, anchor_valid, *, max_ratio):
    """Return (corrected_local, applied_correction, per_batch_scale).

    local/candidate: [B,M,3]; weights: [B,N,M]; boolean masks [B,N]/[B,M].
    Center ONLY the candidate using GS-induced anchor masses, then bound its
    interpolated RMS by max_ratio * the original Local demeaned GS RMS.
    max_ratio is deliberately required, not an experimental hyperparameter
    selected here. Callers must provide a causal, independently reviewed head.
    Padding is zeroed before arithmetic; nonfinite valid inputs are errors.
    """
    if not math.isfinite(max_ratio) or max_ratio < 0:
        raise ValueError('max_ratio must be finite and nonnegative')
    if local.ndim != 3 or local.shape[-1] != 3 or candidate.shape != local.shape:
        raise ValueError('local/candidate must have matching [B,M,3] shapes')
    b, m, _ = local.shape
    if weights.ndim != 3 or weights.shape[0] != b or weights.shape[2] != m:
        raise ValueError('weights must have shape [B,N,M]')
    if gs_valid.shape != weights.shape[:2] or anchor_valid.shape != (b, m):
        raise ValueError('mask shapes do not match')
    if gs_valid.dtype != torch.bool or anchor_valid.dtype != torch.bool:
        raise ValueError('masks must be boolean')
    if not local.is_floating_point() or any(x.dtype != local.dtype for x in (candidate, weights)):
        raise ValueError('floating tensors must share dtype')
    if any(x.device != local.device for x in (candidate, weights, gs_valid, anchor_valid)):
        raise ValueError('all tensors must share device')
    for x in (local, candidate):
        if not torch.isfinite(x[anchor_valid]).all():
            raise ValueError('nonfinite valid anchor values')
    if not torch.isfinite(weights[gs_valid]).all() or (weights[gs_valid] < 0).any():
        raise ValueError('valid RBF rows must be finite and nonnegative')
    w = torch.where(gs_valid[..., None], weights, 0.)
    if (w.masked_select(~anchor_valid[:, None, :].expand_as(w)) != 0).any():
        raise ValueError('valid RBF rows reference padded anchors')
    if not torch.allclose(w.sum(-1)[gs_valid], torch.ones_like(w.sum(-1)[gs_valid]), atol=1e-6, rtol=1e-6):
        raise ValueError('valid RBF rows must sum to one')
    u = torch.where(anchor_valid[..., None], local, 0.)
    q = torch.where(anchor_valid[..., None], candidate, 0.)
    count = gs_valid.sum(-1).clamp_min(1).to(local.dtype)
    mass = w.sum(1) / count[:, None]
    # Divide by measured total mass, including floating-point row-sum error.
    mass = mass / mass.sum(-1, keepdim=True).clamp_min(torch.finfo(local.dtype).tiny)
    mean_q = (mass[..., None] * q).sum(1, keepdim=True)
    centered = torch.where(anchor_valid[..., None], q - mean_q, 0.)
    mean_u = (mass[..., None] * u).sum(1, keepdim=True)
    relative = w @ (u - mean_u)
    projected = w @ centered
    rms_u = torch.linalg.vector_norm(relative.flatten(1), dim=1) / count.sqrt()
    rms_q = torch.linalg.vector_norm(projected.flatten(1), dim=1) / count.sqrt()
    scale = (max_ratio * rms_u / rms_q.clamp_min(torch.finfo(local.dtype).tiny)).clamp(max=1.)
    scale = torch.where(gs_valid.any(-1), scale, 0.)
    applied = centered * scale[:, None, None]
    return u + applied, applied, scale
