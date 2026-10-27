"""Learnable geometry-initialized scores, with a zero-value null option."""
import math
import torch
from torch import nn
from .model import masked


class GeometryCrossBlock(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.heads, self.dim = heads, width // heads
        self.query_norm, self.memory_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.q, self.k, self.v = (nn.Linear(width, width) for _ in range(3))
        nn.init.zeros_(self.q.weight)
        nn.init.zeros_(self.q.bias)
        self.geometry_score = nn.Parameter(torch.tensor([[0., 0., 0., -.5]]).repeat(heads, 1))
        self.pairwise = nn.ModuleList([
            nn.Sequential(nn.Linear(2*self.dim+4, 32), nn.GELU(), nn.Linear(32, 1))
            for _ in range(heads)])
        for mlp in self.pairwise:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
        self.projection = nn.Linear(width, width, bias=False)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))

    def forward(self, query, tools, anchor_xyz, controller_xyz, sigma, anchor_valid, tool_valid):
        b, a, width = query.shape
        n = tools.shape[1]
        query, tools = masked(query, anchor_valid), masked(tools, tool_valid)
        q = self.q(self.query_norm(query)).reshape(b, a, self.heads, self.dim)
        k = self.k(self.memory_norm(tools)).reshape(b, n, self.heads, self.dim)
        v = self.v(self.memory_norm(tools)).reshape(b, n, self.heads, self.dim)
        relative = (masked(anchor_xyz, anchor_valid)[:, :, None]
                    - masked(controller_xyz, tool_valid)[:, None]) / sigma[:, None, None, None]
        geometry = torch.cat((relative, relative.square().sum(-1, keepdim=True)), -1)
        scores = []
        for h in range(self.heads):
            qh = q[:, :, h, None].expand(-1, -1, n, -1)
            kh = k[:, None, :, h].expand(-1, a, -1, -1)
            score = (qh*kh).sum(-1)/math.sqrt(self.dim)
            score = score + (geometry*self.geometry_score[h]).sum(-1)
            score = score + self.pairwise[h](torch.cat((qh, kh, geometry), -1)).squeeze(-1)
            scores.append(score)
        scores = torch.stack(scores, 2)  # [B,A,heads,controllers]
        allowed = anchor_valid[:, :, None, None] & tool_valid[:, None, None]
        scores = scores.masked_fill(~allowed, -torch.inf)
        # Null logit/value are exactly zero; all-invalid attention is well defined.
        alpha = torch.cat((scores, scores.new_zeros(b, a, self.heads, 1)), -1).softmax(-1)
        update = torch.einsum("bahj,bjhd->bahd", alpha[..., :n], v).reshape(b, a, width)
        query = query + self.projection(update)
        query = masked(query + self.ff(self.ff_norm(query)), anchor_valid)
        return query, alpha
