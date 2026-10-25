"""P2 anchor-level structure features; the isotropic P1a field stays fixed.

The two attention layers describe relationships between existing FPS anchors.
Only their feature residual is interpolated back to Gaussian queries. They do
not change the initial neighborhoods, RBF weights or cumulative position head.
"""

import math

import torch
from torch import nn

from .gs_decoder import pool_anchor_features, interpolate_anchor_displacements


def _masked(value, keep):
    return torch.where(keep[..., None], value, torch.zeros_like(value))


class StructuralAnchorLayer(nn.Module):
    """Full attention across valid anchors with live, relative normalized XYZ."""

    def __init__(self, width=128, heads=4):
        super().__init__()
        if min(width, heads) < 1 or width % heads:
            raise ValueError("Structural width must be positive and divisible by heads")
        self.heads, self.head_dim = heads, width // heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.relative = nn.Sequential(nn.Linear(3, width), nn.GELU(),
                                      nn.Linear(width, heads + width))
        self.projection = nn.Linear(width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                nn.Linear(2 * width, width))

    def forward(self, features, xyz, keep):
        features, xyz = _masked(features, keep), _masked(xyz, keep)
        b, count, width = features.shape
        if count == 0:
            return features
        query, key, value = self.qkv(self.norm(features)).chunk(3, dim=-1)
        query = query.reshape(b, count, self.heads, self.head_dim)
        key, value = key.reshape_as(query), value.reshape_as(query)
        # Fixed anchor identities, but relative positions follow the live AR
        # prediction. No target geometry or material axis is supplied here.
        relative = self.relative(xyz[:, None] - xyz[:, :, None])
        bias, relative_value = relative[..., :self.heads], relative[..., self.heads:]
        logits = torch.einsum("bqhd,bkhd->bqkh", query, key) / math.sqrt(self.head_dim)
        logits = (logits + bias).masked_fill(~keep[:, None, :, None], float("-inf"))
        logits = torch.where(keep.any(-1)[:, None, None, None], logits, 0.)
        weights = logits.softmax(dim=2).masked_fill(~keep[:, None, :, None], 0.)
        values = value[:, None] + relative_value.reshape(b, count, count, self.heads, self.head_dim)
        update = (weights[..., None] * values).sum(2).reshape(b, count, width)
        features = _masked(features + self.projection(update), keep)
        return _masked(features + self.ff(self.ff_norm(features)), keep)


class StructuralGSBlock(nn.Module):
    """Two structural layers and a zero-initialized Gaussian-feature residual.

Formal paired experiments use width=128 and heads=4, matching the existing GS
features. The fixed initial RBF interpolates a learned feature update, not a
second position field. A single zero output projection preserves P1a behavior
at initialization while allowing the structural layers to become trainable.
"""

    def __init__(self, width=128, heads=4):
        super().__init__()
        self.blocks = nn.ModuleList(StructuralAnchorLayer(width, heads) for _ in range(2))
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, xyz, keep, field):
        if (features.ndim != 3 or xyz.shape != (*features.shape[:2], 3)
                or keep.shape != features.shape[:2] or keep.dtype != torch.bool):
            raise ValueError("Structural features need [B,N,D], XYZ [B,N,3] and Boolean keep [B,N]")
        features, xyz = _masked(features, keep), _masked(xyz, keep)
        anchors = pool_anchor_features(features, field)
        batch = torch.arange(len(features), device=features.device)[:, None]
        centers = _masked(xyz[batch, field.anchor_indices], field.anchor_mask)
        for block in self.blocks:
            anchors = block(anchors, centers, field.anchor_mask)
        residual = _masked(self.output(anchors), field.anchor_mask)
        residual = interpolate_anchor_displacements(residual, field, keep)
        return _masked(features + residual, keep)
