"""Rotation-equivariant experimental edge response, not a cloth solver.

Edges must be admitted independently at initialization and held fixed. This
module neither builds spatial KNN nor certifies material connectivity. Inputs
are initial anchors and the *total* original motion proposal (Base+Global+Local),
not a velocity or future GT. The output is a candidate cumulative correction.
"""
import math
import torch
from torch import nn


class EdgeResponseHead(nn.Module):
    """Learn signed bounded edge gains from dimensionless length change.

    Positive gain contracts stretched edges / expands compressed edges; negative
    gain reverses that response. No restoring-force or convergence guarantee.
    Zero last layer preserves the old model while allowing nonzero first-step
    gradients for that layer on nonrigid inputs. Hidden layers learn afterwards.
    """
    def __init__(self, *, hidden_dim, min_length):
        super().__init__()
        if hidden_dim < 1 or not math.isfinite(min_length) or min_length <= 0:
            raise ValueError('positive hidden_dim and min_length required')
        self.min_length = float(min_length)
        self.net = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, reference, proposed, edges, admitted, anchor_valid):
        """[B,M,3] positions, [B,E,2] int64 edges, boolean [B,E]/[B,M].

        Admitted edges are unique undirected pairs. Masked edge indices may be
        -1; padded anchor NaNs are sanitized. Valid NaNs or collapsed admitted
        edges raise instead of silently changing the graph. Empty graphs yield
        zero and must not be reported by a caller as effective regularization.
        """
        if reference.ndim != 3 or reference.shape[-1] != 3 or proposed.shape != reference.shape:
            raise ValueError('positions must have matching [B,M,3] shapes')
        b, m, _ = reference.shape
        if m < 1 or edges.ndim != 3 or edges.shape[0] != b or edges.shape[-1] != 2:
            raise ValueError('at least one anchor and edges [B,E,2] required')
        if edges.dtype != torch.long or admitted.dtype != torch.bool or anchor_valid.dtype != torch.bool:
            raise ValueError('int64 edges and boolean masks required')
        if admitted.shape != edges.shape[:2] or anchor_valid.shape != (b, m):
            raise ValueError('mask shapes do not match')
        if not reference.is_floating_point() or proposed.dtype != reference.dtype:
            raise ValueError('positions must share floating dtype')
        if any(x.device != reference.device for x in (proposed, edges, admitted, anchor_valid)):
            raise ValueError('all inputs must share device')
        for p in (reference, proposed):
            if not torch.isfinite(p[anchor_valid]).all():
                raise ValueError('nonfinite valid positions')
        active = edges[admitted]
        if ((active < 0) | (active >= m)).any() or (active[:, 0] == active[:, 1]).any():
            raise ValueError('invalid endpoint or self edge')
        safe = torch.where(admitted[..., None], edges, 0)
        batch = torch.arange(b, device=reference.device)[:, None]
        if not (anchor_valid[batch, safe[..., 0]] & anchor_valid[batch, safe[..., 1]])[admitted].all():
            raise ValueError('admitted edge uses padded anchor')
        for n in range(b):
            pairs = edges[n, admitted[n]].sort(-1).values
            if len(torch.unique(pairs, dim=0)) != len(pairs):
                raise ValueError('duplicate undirected edge')
        ref = torch.where(anchor_valid[..., None], reference, 0.)
        pred = torch.where(anchor_valid[..., None], proposed, 0.)
        i, j = safe.unbind(-1)
        e0 = ref[batch, j] - ref[batch, i]
        e1 = pred[batch, j] - pred[batch, i]
        l0 = torch.linalg.vector_norm(e0, dim=-1)
        l1 = torch.linalg.vector_norm(e1, dim=-1)
        if ((l0[admitted] <= self.min_length) | (l1[admitted] <= self.min_length)).any():
            raise ValueError('degenerate admitted edge: keep evidence and stop')
        strain = (l1 - l0) / l0.clamp_min(self.min_length)
        gain = torch.tanh(self.net(strain[..., None]).squeeze(-1))
        message = gain[..., None] * (l1-l0)[..., None] * e1 / l1.clamp_min(self.min_length)[..., None]
        message = torch.where(admitted[..., None], message, 0.)
        output = torch.zeros_like(ref)
        output = output.scatter_add(1, i[..., None].expand(-1,-1,3), message)
        output = output.scatter_add(1, j[..., None].expand(-1,-1,3), -message)
        degree = torch.zeros_like(ref[..., 0])
        degree = degree.scatter_add(1, i, admitted.to(ref.dtype)).scatter_add(1, j, admitted.to(ref.dtype))
        # One denominator per graph retains endpoint antisymmetry. This is not
        # mass-weighted conservation; the GS-mean guard is applied downstream.
        output = output / degree.amax(-1).clamp_min(1)[:, None, None]
        if not torch.isfinite(output).all():
            raise ValueError('nonfinite candidate response')
        return output
