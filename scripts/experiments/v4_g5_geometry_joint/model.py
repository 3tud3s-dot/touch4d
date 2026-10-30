"""Joint gradients, unchanged G5 motion rollout, no geometry feedback into AR."""
import random
import numpy as np
import torch
from torch import nn
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_geometry.geometry import adapt_attributes, LOG_SCALE_BOUND, ROT_COMPONENT_BOUND
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash

ARCHITECTURE = 'v4_g5_direct_geometry_joint_scratch_v1'


class JointGeometryPredictor(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        self.backbone = G5DirectPredictor(horizon=15, prediction_interval=1) if backbone is None else backbone
        if type(self.backbone) is not G5DirectPredictor:
            raise TypeError('Original G5-Direct required')
        if any(not p.requires_grad for p in self.backbone.parameters()):
            raise ValueError('Joint experiment must not receive a frozen backbone')
        self.head = nn.Sequential(nn.Linear(131, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 6))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.config = dict(self.backbone.config, architecture=ARCHITECTURE, checkpoint_kind='g5_geo_joint_scratch_v1')
        self.geometry_config = dict(architecture=ARCHITECTURE, widths=[131,128,64,6], activation='GELU',
            log_scale_bound=LOG_SCALE_BOUND, rotation_component_bound=ROT_COMPONENT_BOUND,
            quaternion='WXYZ_left_world', reference='initial', ar_feedback=False,
            detached_geometry_inputs=False, backbone_trainable=True)

    @property
    def global_motion(self):
        return self.backbone.global_motion

    def forward(self, inputs, *, mode='geometry_on', return_trace=False):
        if mode not in ('geometry_on', 'geometry_off', 'geometry_on_zero'):
            raise ValueError('Unknown geometry mode')
        if mode == 'geometry_on_zero' and any(p.detach().count_nonzero() for p in self.head[-1].parameters()):
            raise ValueError('Zero mode requires zero output projection')
        queries = []
        # Capture the original final Cross output without detaching its graph.
        handle = self.backbone.cross[-1].register_forward_hook(
            lambda module, args, output: queries.append(output[0]))
        try:
            motion, trace = self.backbone(inputs, return_trace=True)
        finally:
            handle.remove()
        if len(queries) != self.config['horizon']:
            raise RuntimeError('Unexpected final Cross invocation count')
        features = torch.cat((torch.stack(queries, 1), torch.stack(
            [s['final_cumulative_anchor_displacement'] for s in trace['steps']], 1)), -1)
        features = torch.where(trace['field'].anchor_mask[:, None, :, None], features, torch.zeros_like(features))
        if mode == 'geometry_off':
            prediction, geometry = motion, {}
        else:
            raw = self.head(features)
            prediction, geometry = adapt_attributes(raw, motion, trace['field'], inputs['gs_mask'])
            geometry['raw_anchor_geometry'] = raw
        if return_trace:
            return prediction, dict(trace, motion=motion, geometry=geometry, geometry_features=features)
        return prediction


def fresh_model():
    """Match fresh seed42 G5 shared weights AND original zero GEO head init."""
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    backbone = G5DirectPredictor(horizon=15, prediction_interval=1)
    torch.manual_seed(42)  # Same head initialization as the prior head-only formal run.
    model = JointGeometryPredictor(backbone)
    return model


def initialization(model):
    return dict(protocol='fresh seed42 G5 and fresh zero-output GEO; no checkpoint loading',
        model_state_hash=state_hash(model.state_dict()), backbone_state_hash=state_hash(model.backbone.state_dict()),
        head_state_hash=state_hash(model.head.state_dict()), seed=42,
        backbone_parameters=sum(p.numel() for p in model.backbone.parameters()),
        geometry_parameters=sum(p.numel() for p in model.head.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
