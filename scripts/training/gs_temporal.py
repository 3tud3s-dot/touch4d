"""P3 cross-frame anchor features, without point IDs or visibility assumptions."""
import math
import torch
from torch import nn


class TemporalAnchorFusion(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        if width < 1 or heads < 1 or width % heads:
            raise ValueError("Invalid temporal attention dimensions")
        self.heads, self.dim = heads, width//heads
        self.query_norm, self.memory_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2*width)
        self.relative = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width+heads))
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, current, previous, xyz_current, xyz_previous, current_mask, previous_mask):
        b, n, width = current.shape
        m = previous.shape[1]
        if (previous.shape != (b, m, width) or xyz_current.shape != (b, n, 3)
                or xyz_previous.shape != (b, m, 3) or current_mask.shape != (b, n)
                or previous_mask.shape != (b, m) or current_mask.dtype != torch.bool
                or previous_mask.dtype != torch.bool):
            raise ValueError("Temporal anchors require independent [B,K,D]/[B,K] masks")
        mask = lambda v, k: torch.where(k[..., None], v, torch.zeros_like(v))
        current, previous = mask(current, current_mask), mask(previous, previous_mask)
        xyz_current, xyz_previous = mask(xyz_current, current_mask), mask(xyz_previous, previous_mask)
        if not all(torch.isfinite(v).all() for v in (current, previous, xyz_current, xyz_previous)):
            raise FloatingPointError("Nonfinite valid temporal input")
        if n == 0 or m == 0:
            return torch.zeros_like(current)
        q = self.query(self.query_norm(current)).reshape(b, n, self.heads, self.dim)
        k, v = self.key_value(self.memory_norm(previous)).chunk(2, -1)
        k, v = k.reshape(b, m, self.heads, self.dim), v.reshape(b, m, self.heads, self.dim)
        relative = self.relative(xyz_previous[:, None]-xyz_current[:, :, None])
        logits = torch.einsum('bqhd,bkhd->bqkh', q, k)/math.sqrt(self.dim) + relative[..., :self.heads]
        logits = logits.masked_fill(~previous_mask[:, None, :, None], -float('inf'))
        logits = torch.where(previous_mask.any(-1)[:, None, None, None], logits, 0.)
        weights = logits.softmax(2).masked_fill(~previous_mask[:, None, :, None], 0.)
        values = v[:, None] + relative[..., self.heads:].reshape(b, n, m, self.heads, self.dim)
        update = self.output((weights[..., None]*values).sum(2).reshape(b, n, width))
        return mask(update, current_mask & previous_mask.any(-1)[:, None])
