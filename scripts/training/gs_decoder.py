"""Initial-state FPS anchors and a shared, smooth Gaussian displacement field.

This P1a decoder changes centers only. Its geometry is built from the valid
initial template once per rollout; it never reads target frames or transfers
covariance. The helpers have no trainable parameters.
"""

from dataclasses import dataclass, replace
import math

import torch


@dataclass
class InitialAnchorField:
    anchor_indices: torch.Tensor
    anchor_mask: torch.Tensor
    pool_indices: torch.Tensor
    pool_mask: torch.Tensor
    weights: torch.Tensor
    bandwidth: torch.Tensor
    metric: torch.Tensor = None


@torch.no_grad()
def build_initial_anchor_field(xyz, keep, anchor_count=128, pool_k=16,
                               bandwidth_factor=1.0):
    """Build deterministic FPS, fixed nearest-neighbor pools and dense RBF weights.

    The first FPS point is farthest from the valid centroid; exact distance ties
    select the lowest original index. Selected points are excluded explicitly,
    so coincident centers still give distinct anchors. K is capped by each
    sample's valid point count. Storage is O(B*N*K), without an N-by-N matrix.

    RBF bandwidth is the median nearest *other* anchor distance, multiplied by
    the configured factor, with a 1e-4 floor in normalized coordinates. A
    singleton/coincident field uses that same floor. Full-support Gaussian
    weights are normalized by a numerically stable softmax.
    """
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or keep.shape != xyz.shape[:2]:
        raise ValueError("Expected xyz [B,N,3] and keep [B,N]")
    if keep.dtype != torch.bool or not xyz.is_floating_point():
        raise ValueError("Anchor geometry requires floating XYZ and a Boolean mask")
    if type(anchor_count) is not int or anchor_count < 1:
        raise ValueError("anchor_count must be a positive integer")
    if type(pool_k) is not int or pool_k < 1:
        raise ValueError("pool_k must be a positive integer")
    if (isinstance(bandwidth_factor, bool) or not isinstance(bandwidth_factor, (int, float))
            or not math.isfinite(bandwidth_factor) or bandwidth_factor <= 0):
        raise ValueError("bandwidth_factor must be finite and positive")
    b, n, _ = xyz.shape
    slots, pool_slots = min(anchor_count, n), min(pool_k, n)
    indices = torch.zeros((b, slots), dtype=torch.long, device=xyz.device)
    anchor_mask = torch.zeros_like(indices, dtype=torch.bool)
    pools = torch.zeros((b, slots, pool_slots), dtype=torch.long, device=xyz.device)
    pool_mask = torch.zeros_like(pools, dtype=torch.bool)
    weights = xyz.new_zeros((b, n, slots))
    bandwidth = xyz.new_full((b,), 1e-4)
    for batch in range(b):
        ids = keep[batch].nonzero(as_tuple=True)[0]
        count = len(ids)
        if not count:
            continue
        # Float32 avoids low-precision squared-distance underflow. Keep float64
        # when supplied by numerical tests; masked padding never enters geometry.
        points = xyz[batch, ids]
        points = points.to(torch.float64 if points.dtype == torch.float64 else torch.float32)
        if not torch.isfinite(points).all():
            raise ValueError("Nonfinite valid GS coordinates in initial anchor field")
        used, pooled = min(anchor_count, count), min(pool_k, count)
        selected = torch.zeros(count, dtype=torch.bool, device=xyz.device)
        nearest = torch.full((count,), float("inf"), dtype=points.dtype, device=xyz.device)
        next_index = ((points - points.mean(0)) ** 2).sum(-1).argmax()
        local_indices = torch.empty(used, dtype=torch.long, device=xyz.device)
        for anchor in range(used):
            local_indices[anchor] = next_index
            selected[next_index] = True
            distance = ((points - points[next_index]) ** 2).sum(-1)
            nearest = torch.minimum(nearest, distance)
            next_index = nearest.masked_fill(selected, float("-inf")).argmax()
        indices[batch, :used] = ids[local_indices]
        anchor_mask[batch, :used] = True
        anchors = points[local_indices]
        # At most K*N distances, also reused to construct the interpolation field.
        distances = torch.cdist(anchors, points, compute_mode="donot_use_mm_for_euclid_dist")
        pools[batch, :used, 0] = ids[local_indices]
        pool_mask[batch, :used, :pooled] = True
        if pooled > 1:
            pool_distances = distances.clone()
            pool_distances[torch.arange(used, device=xyz.device), local_indices] = float("inf")
            # Stable sorting gives deterministic ties in the original valid order.
            nearest_ids = pool_distances.argsort(dim=-1, stable=True)[:, :pooled-1]
            pools[batch, :used, 1:pooled] = ids[nearest_ids]
        if used > 1:
            between = distances[:, local_indices].clone()
            between.fill_diagonal_(float("inf"))
            sigma = (between.amin(-1).median() * bandwidth_factor).clamp_min(1e-4)
        else:
            sigma = points.new_tensor(1e-4)
        bandwidth[batch] = sigma.to(xyz.dtype)
        logits = -.5 * (distances.T / sigma).square()
        weights[batch, ids, :used] = logits.softmax(-1).to(xyz.dtype)
    return InitialAnchorField(indices, anchor_mask, pools, pool_mask, weights, bandwidth)


def pool_anchor_features(features, field):
    """Mean existing GS query features over each anchor's fixed initial pool."""
    b, _, width = features.shape
    if field.pool_indices.shape[-1] == 0:
        return features.new_zeros((*field.anchor_indices.shape, width))
    batches = torch.arange(b, device=features.device)[:, None, None]
    gathered = features[batches, field.pool_indices]
    gathered = torch.where(field.pool_mask[..., None], gathered, torch.zeros_like(gathered))
    counts = field.pool_mask.sum(-1, keepdim=True).clamp_min(1)
    return gathered.sum(-2) / counts.to(features.dtype)


def bounded_metric(raw, bandwidth, anchor_mask):
    """M = sigma^-2 (I+A)^T(I+A), A=.5 R/max(1,||R||_F).

    The dimensionless eigenvalues lie in [.25, 2.25], so cond(M)<=9.
    Rescale before taking the norm to avoid overflow for large finite R.
    This is algebraically the requested parameterization, with a live gradient.
    """
    if raw.shape != (*anchor_mask.shape, 3, 3) or bandwidth.shape != raw.shape[:1]:
        raise ValueError("Expected raw [B,K,3,3], bandwidth [B] and anchor mask [B,K]")
    if anchor_mask.dtype != torch.bool:
        raise ValueError("Anchor mask must be Boolean")
    raw = torch.where(anchor_mask[..., None, None], raw, torch.zeros_like(raw))
    if not torch.isfinite(raw).all() or not torch.isfinite(bandwidth).all() or not (bandwidth > 0).all():
        raise ValueError("Nonfinite metric parameters or invalid bandwidth")
    # Compute metric geometry at least in float32, even under mixed precision.
    raw = raw.to(torch.float64 if raw.dtype == torch.float64 else torch.float32)
    scale = raw.abs().amax((-1, -2), keepdim=True).clamp_min(1.)
    unit = raw / scale
    denominator = torch.maximum(scale.reciprocal(), torch.linalg.vector_norm(unit, dim=(-1,-2), keepdim=True))
    a = .5 * unit / denominator
    transform = torch.eye(3, device=raw.device, dtype=raw.dtype) + a
    with torch.autocast(device_type=raw.device.type, enabled=False):
        metric = transform.transpose(-1,-2) @ transform
    return metric / bandwidth.to(raw.dtype)[:, None, None, None].square()


def anisotropic_anchor_field(xyz, keep, field, raw):
    """Initial-only learnable RBF; FPS/pools stay fixed, M and weights stay live."""
    metric = bounded_metric(raw, field.bandwidth, field.anchor_mask)
    xyz = torch.where(keep[..., None], xyz, torch.zeros_like(xyz)).to(metric.dtype)
    batches = torch.arange(len(xyz), device=xyz.device)[:, None]
    centers = xyz[batches, field.anchor_indices]
    offsets = xyz[:, :, None] - centers[:, None]
    with torch.autocast(device_type=xyz.device.type, enabled=False):
        distance = torch.einsum('bnki,bkij,bnkj->bnk', offsets, metric, offsets)
    logits = (-.5 * distance).masked_fill(~field.anchor_mask[:, None], float('-inf'))
    logits = torch.where(field.anchor_mask.any(-1)[:, None, None], logits, 0.)
    weights = logits.softmax(-1).masked_fill(~field.anchor_mask[:, None], 0.)
    weights = torch.where(keep[..., None], weights, torch.zeros_like(weights))
    return replace(field, weights=weights, metric=metric)


def interpolate_anchor_displacements(displacements, field, keep):
    """Spread cumulative anchor displacement to the initial template.

    In exact arithmetic, constant anchor displacement produces the same global
    translation, irrespective of FPS positions or RBF bandwidth. Invalid anchors
    and padded Gaussian rows are sanitized before and after interpolation.
    """
    displacements = torch.where(field.anchor_mask[..., None], displacements,
                                torch.zeros_like(displacements))
    delta = torch.bmm(field.weights.to(displacements.dtype), displacements)
    return torch.where(keep[..., None], delta, torch.zeros_like(delta))
