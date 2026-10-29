"""Fixed initial Anchor graph and one layer of per-hand signed-vector mixing.

This graph is an initial spatial neighborhood, not a material topology. In
particular, folded rope segments can become neighbors despite no material edge.
"""
import torch
from torch import nn


@torch.no_grad()
def build_anchor_graph(initial_xyz, anchor_valid):
    """Self first, followed by nearest 8 other valid rows; stable index ties.

    Returns indices/mask [B,A,min(9,A)]. Invalid queries have no neighbors;
    unused slots are masked and their placeholder indices have no meaning.
    """
    if (initial_xyz.ndim != 3 or initial_xyz.shape[-1] != 3
            or anchor_valid.shape != initial_xyz.shape[:2]
            or anchor_valid.dtype != torch.bool or initial_xyz.shape[1] < 1):
        raise ValueError('Expected initial XYZ [B,A,3] and bool mask [B,A]')
    if not torch.isfinite(initial_xyz[anchor_valid]).all():
        raise ValueError('Nonfinite valid initial Anchor')
    batch, anchors, _ = initial_xyz.shape
    slots = min(9, anchors)
    indices = torch.zeros(batch, anchors, slots, dtype=torch.long, device=initial_xyz.device)
    allowed = torch.zeros_like(indices, dtype=torch.bool)
    for b in range(batch):
        ids = anchor_valid[b].nonzero().flatten()  # ascending row index
        count = len(ids)
        if not count:
            continue
        indices[b, ids, 0] = ids
        allowed[b, ids, 0] = True
        used = min(8, count - 1)
        if not used:
            continue
        points = initial_xyz[b, ids]
        if points.dtype not in (torch.float32, torch.float64):
            points = points.float()
        distance = torch.cdist(points, points, compute_mode='donot_use_mm_for_euclid_dist')
        distance.fill_diagonal_(torch.inf)
        # Stable sorting retains ascending original row index for equal distance.
        order = distance.argsort(dim=-1, stable=True)[:, :used]
        indices[b, ids, 1:used + 1] = ids[order]
        allowed[b, ids, 1:used + 1] = True
    return indices, allowed


def _masked_softmax(scores, allowed):
    """An empty row is exactly zero, without a synthetic null neighbor."""
    valid_row = allowed.any(-1, keepdim=True)
    logits = scores.masked_fill(~allowed, -torch.inf)
    logits = torch.where(valid_row, logits, torch.zeros_like(logits))
    return torch.where(allowed, logits.softmax(-1), torch.zeros_like(logits))


class SignedAnchorPropagation(nn.Module):
    """One per-hand A2A, z + .5 sum_b P_ab (z_b-z_a); no output gate.

    P has shape [B,A,2,K], K<=9. Both hands share a score MLP but have separate
    token/basis inputs, neighborhoods and normalizations. Only signed XYZ seeds
    are propagated, rather than a hidden feature replacement.
    """
    mixing = .5

    def __init__(self, width=128):
        super().__init__()
        self.width = width
        self.score = nn.Sequential(
            nn.LayerNorm(3 * width + 8 + 18),
            nn.Linear(3 * width + 8 + 18, width), nn.GELU(),
            nn.Linear(width, 1, bias=False),
        )

    def forward(self, query, hand_tokens, basis18, direct_seed_per_hand,
                initial_xyz, current_xyz, anchor_valid, hand_valid,
                indices, neighbor_mask):
        batch, anchors, width = query.shape
        slots = indices.shape[-1]
        expected = {
            'hand_tokens': ((batch, 2, width), hand_tokens),
            'basis18': ((batch, anchors, 2, 18), basis18),
            'direct_seed_per_hand': ((batch, anchors, 2, 3), direct_seed_per_hand),
            'initial_xyz': ((batch, anchors, 3), initial_xyz),
            'current_xyz': ((batch, anchors, 3), current_xyz),
            'anchor_valid': ((batch, anchors), anchor_valid),
            'hand_valid': ((batch, 2), hand_valid),
            'indices': ((batch, anchors, slots), indices),
            'neighbor_mask': ((batch, anchors, slots), neighbor_mask),
        }
        if width != self.width or not 1 <= slots <= min(9, anchors):
            raise ValueError('Unexpected feature width or graph degree')
        for name, (shape, tensor) in expected.items():
            if tuple(tensor.shape) != shape:
                raise ValueError(f'{name} must have shape {shape}')
        if (indices.dtype != torch.long or any(t.dtype != torch.bool
                for t in (anchor_valid, hand_valid, neighbor_mask))):
            raise ValueError('Graph indices must be long and masks bool')
        if (indices < 0).any() or (indices >= anchors).any():
            raise ValueError('Graph index out of range')
        batches = torch.arange(batch, device=query.device)[:, None, None]
        references = anchor_valid[batches, indices]
        if (neighbor_mask & (~references | ~anchor_valid[..., None])).any():
            raise ValueError('Graph cannot connect padding Anchors')
        self_indices = torch.arange(anchors, device=query.device)[None].expand(batch, -1)
        if not torch.equal(neighbor_mask[..., 0], anchor_valid) or not torch.equal(
                indices[..., 0][anchor_valid], self_indices[anchor_valid]):
            raise ValueError('Every valid Anchor needs Self in graph slot zero')

        anchor_hand = anchor_valid[:, :, None] & hand_valid[:, None]
        q = torch.where(anchor_valid[..., None], query, 0.)
        tokens = torch.where(hand_valid[..., None], hand_tokens, 0.)
        basis = torch.where(anchor_hand[..., None], basis18, 0.)
        seed = torch.where(anchor_hand[..., None], direct_seed_per_hand, 0.)
        initial = torch.where(anchor_valid[..., None], initial_xyz, 0.)
        current = torch.where(anchor_valid[..., None], current_xyz, 0.)
        relative0 = initial[:, :, None] - initial[batches, indices]
        relative = current[:, :, None] - current[batches, indices]
        geometry = torch.cat((relative0, torch.linalg.vector_norm(relative0, dim=-1, keepdim=True),
                              relative, torch.linalg.vector_norm(relative, dim=-1, keepdim=True)), -1)
        qa = q[:, :, None, None].expand(-1, -1, 2, slots, -1)
        qb = q[batches, indices][:, :, None].expand(-1, -1, 2, -1, -1)
        hand = tokens[:, None, :, None].expand(-1, anchors, -1, slots, -1)
        geom = geometry[:, :, None].expand(-1, -1, 2, -1, -1)
        motion = basis[:, :, :, None].expand(-1, -1, -1, slots, -1)
        scores = self.score(torch.cat((qa, qb, hand, geom, motion), -1)).squeeze(-1)
        allowed = neighbor_mask[:, :, None] & anchor_hand[..., None]
        weights = _masked_softmax(scores, allowed)
        neighbor_seed = seed[batches, indices].permute(0, 1, 3, 2, 4)
        difference = neighbor_seed - seed[:, :, :, None]
        update = (weights[..., None] * difference).sum(-2)
        propagated = torch.where(anchor_hand[..., None], seed + self.mixing * update, 0.)
        if not torch.isfinite(propagated).all() or not torch.isfinite(weights).all():
            raise FloatingPointError('Nonfinite signed Anchor propagation')
        return propagated, weights
