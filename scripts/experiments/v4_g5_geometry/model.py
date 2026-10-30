"""Frozen original G5 forward plus a detached, output-only geometry adapter."""
import torch
from torch import nn
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from .geometry import adapt_attributes, LOG_SCALE_BOUND, ROT_COMPONENT_BOUND

ARCHITECTURE = 'touch4d-g5-geometry-output-adapter-v1'


class GeometryAdapter(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if type(backbone) is not G5DirectPredictor:
            raise TypeError('Requires the original G5DirectPredictor')
        self.backbone = backbone.requires_grad_(False).eval()
        self.head = nn.Sequential(nn.Linear(131, 128), nn.GELU(),
                                  nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 6))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.config = dict(backbone.config)
        self.geometry_config = dict(architecture=ARCHITECTURE, backbone=backbone.config,
            widths=[131,128,64,6], activation='GELU', log_scale_bound=LOG_SCALE_BOUND,
            rotation_component_bound=ROT_COMPONENT_BOUND, quaternion='WXYZ_left_world',
            reference='initial', ar_feedback=False, detached_geometry_inputs=True)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def motion_forward(self, inputs):
        """Read-only hook observes final Cross output; original AR loop is untouched."""
        queries = []
        handle = self.backbone.cross[-1].register_forward_hook(
            lambda module, args, output: queries.append(output[0].detach()))
        try:
            with torch.no_grad():
                motion, trace = self.backbone(inputs, return_trace=True)
        finally:
            handle.remove()
        if len(queries) != self.config['horizon']:
            raise RuntimeError('Unexpected final Cross invocation count')
        features = torch.stack(queries, 1)
        displacement = torch.stack([s['final_cumulative_anchor_displacement'].detach()
                                    for s in trace['steps']], 1)
        return motion, trace, torch.cat((features, displacement), -1).detach()

    def render_from_motion(self, motion, trace, features, gs_mask, mode='geometry_on'):
        if mode not in ('geometry_off','geometry_on_zero','geometry_on'):
            raise ValueError('Unknown geometry mode')
        if mode == 'geometry_off':
            return motion, {}
        if mode == 'geometry_on_zero' and any(p.detach().count_nonzero() for p in self.head[-1].parameters()):
            raise ValueError('geometry_on_zero requires the independent zero-head snapshot')
        valid = trace['field'].anchor_mask[:, None, :, None]
        features = torch.where(valid, features.detach(), torch.zeros_like(features))
        raw = self.head(features)
        render, geometry = adapt_attributes(raw, motion, trace['field'], gs_mask)
        geometry['raw_anchor_geometry'] = raw
        return render, geometry

    def forward(self, inputs, *, mode='geometry_on', return_trace=False):
        motion, trace, features = self.motion_forward(inputs)
        render, geometry = self.render_from_motion(motion, trace, features, inputs['gs_mask'], mode)
        if return_trace:
            return render, dict(motion=motion, motion_trace=trace, geometry=geometry,
                                detached_features=features)
        return render
