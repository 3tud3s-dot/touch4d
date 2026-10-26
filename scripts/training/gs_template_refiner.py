"""Cumulative bounded template attributes, always relative to observed GS_t."""
import math
import torch
from torch import nn
from .gs_decoder import interpolate_anchor_displacements


class TemplateRefiner(nn.Module):
    SCALE_BOUND = math.log(1.3)
    OPACITY_BOUND = .5

    def __init__(self, width=128):
        super().__init__()
        self.scale_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
        self.opacity_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        for head in (self.scale_head, self.opacity_head):
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def forward(self, anchors, field, initial_attr, keep):
        scale = interpolate_anchor_displacements(self.scale_head(anchors), field, keep)
        opacity = interpolate_anchor_displacements(self.opacity_head(anchors), field, keep)
        result = torch.cat((initial_attr[..., :3]+self.SCALE_BOUND*scale.tanh(),
                            initial_attr[..., 3:7],
                            initial_attr[..., 7:8]+self.OPACITY_BOUND*opacity.tanh(),
                            initial_attr[..., 8:]), -1)
        return torch.where(keep[..., None], result, torch.zeros_like(result))
