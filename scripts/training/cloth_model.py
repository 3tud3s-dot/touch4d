"""Single retained cloth dynamics graph; no imports from archived experiment packages.

Initialization order, parameter names, cumulative displacement and all formulas
are preserved from the frozen Direct model. The legacy PositionPredictor stays
in scripts.training.model for historical checkpoint compatibility.
"""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from .model import LocalGSBlock, initial_knn, masked
from .gs_decoder import build_initial_anchor_field, pool_anchor_features, interpolate_anchor_displacements
from .cloth_conditions import V4ConditionEncoder, HAND_POLICY
from .cloth_attention import GeometryCrossBlock
from .cloth_motion import SignedGlobalTranslation, AnchorSignedDirect
from scripts.dataloader.splits import validate_horizon, validate_prediction_interval


def fixed_field(xyz, keep):
    field = build_initial_anchor_field(xyz, keep, 128, 16, 1.)
    missing = 128-field.anchor_mask.shape[1]
    if missing:
        field = replace(field, anchor_indices=F.pad(field.anchor_indices, (0, missing)),
                        anchor_mask=F.pad(field.anchor_mask, (0, missing)),
                        pool_indices=F.pad(field.pool_indices, (0, 0, 0, missing)),
                        pool_mask=F.pad(field.pool_mask, (0, 0, 0, missing)),
                        weights=F.pad(field.weights, (0, missing)))
    return field


class ClothDynamicsPredictor(nn.Module):
    """The retained 620,511-parameter, causal H15 Direct cloth model."""

    def __init__(self, horizon=15, prediction_interval=1):
        super().__init__()
        validate_horizon(horizon)
        validate_prediction_interval(prediction_interval)
        self.config = dict(architecture='v4_stage1', horizon=horizon, prediction_interval=prediction_interval, width=128, heads=4, layers=2, local_k=16, anchor_count=128, pool_k=16, anchor_bandwidth=1.0, sigma_geo_multiplier=1.0, hand_policy=HAND_POLICY, controller_features=13, tactile_sigma_m=0.01, history_flags='head_mass4_anchor1', output_mode='gs_geometry', displacement_reference='initial_cumulative', selection_metric='joint', loss_weights=dict(cd=1.0, rgb=1.0, alpha=0.1, scale=0.0, depth=0.0))
        self.gs_encoder = nn.Sequential(nn.Linear(14, 128), nn.GELU(), nn.Linear(128, 128))
        self.local = LocalGSBlock(128, 4)
        self.conditions = V4ConditionEncoder(horizon)
        self.pred_time = nn.Embedding(horizon, 128)
        self.cross = nn.ModuleList([GeometryCrossBlock() for _ in range(2)])
        self.motion = nn.Sequential(nn.Linear(151, 128), nn.GELU(), nn.Linear(128, 3))
        nn.init.zeros_(self.motion[-1].weight)
        nn.init.zeros_(self.motion[-1].bias)
        self.config = dict(self.config, architecture='v4_g4_global_motion_v1', global_motion='signed_per_hand_global_translation_v1')
        self.global_motion = SignedGlobalTranslation(width=128)
        propagate = False
        self.direct_motion = AnchorSignedDirect(width=128)
        self.use_propagation = propagate
        mode = 'propagate' if propagate else 'direct'
        self.config = dict(self.config, architecture=f'v4_g5_signed_local_{mode}_v1', local_motion='per_anchor_per_hand_18d_v1', local_zero_mean=False, local_mapping_width=128, local_bandwidth_multiplier=1.0, anchor_propagation=propagate, anchor_graph_other_neighbors=8 if propagate else 0, anchor_graph_self=propagate, propagation_layers=1 if propagate else 0, propagation_lambda=0.5 if propagate else 0.0, checkpoint_kind=f'g5_{mode}_v1')

    def forward(self, x, *, return_trace=False):
        h = self.config['horizon']
        frame_ids = x.get('condition_frame_ids')
        if frame_ids is None or frame_ids.dtype != torch.int64 or frame_ids.shape != (len(x['gs_xyz']), h) or (not torch.all(frame_ids[:, 1:] - frame_ids[:, :-1] == self.config['prediction_interval'])):
            raise ValueError('G5 condition_frame_ids must match configured H/interval')
        if any((x[key].shape[1] != h for key in ('action_state', 'controller_points', 'tactile'))):
            raise ValueError('G5 requires exactly H sampled conditions')
        keep = x['gs_mask']
        if keep.dtype != torch.bool or not keep.any(1).all():
            raise ValueError('Each sample requires valid GS')
        initial, attr = (masked(x['gs_xyz'], keep), masked(x['gs_attr'], keep))
        if not torch.isfinite(initial).all() or not torch.isfinite(attr).all():
            raise ValueError('Nonfinite initial GS')
        field = fixed_field(initial, keep)
        neighbors = initial_knn(initial, keep, 16)
        batches = torch.arange(len(initial), device=initial.device)[:, None]
        initial_anchors = masked(initial[batches, field.anchor_indices], field.anchor_mask)
        current = initial
        previous_anchor = initial_anchors
        previous_d = torch.zeros_like(initial_anchors)
        previous_cp = previous_valid = None
        states = []
        steps = []
        for step in range(h):
            features = masked(self.gs_encoder(torch.cat((current, attr), -1)), keep)
            features = self.local(features, current, keep, neighbors)
            query = masked(pool_anchor_features(features, field) + self.pred_time.weight[step], field.anchor_mask)
            anchors = masked(current[batches, field.anchor_indices], field.anchor_mask)
            tools, valid, cp, condition_trace = self.conditions(x, step)
            for block in self.cross:
                query, alpha = block(query, tools, anchors, cp, field.bandwidth, field.anchor_mask, valid)
            adjacent = torch.zeros_like(valid) if step == 0 else valid & previous_valid
            cp_delta = torch.zeros_like(cp) if step == 0 else masked(cp - previous_cp, adjacent)
            weights = alpha[..., :60]
            tool_motion = torch.einsum('bahj,bjd->bahd', weights, cp_delta).flatten(-2)
            history = torch.zeros_like(anchors) if step == 0 else anchors - previous_anchor
            head_mass = (weights * adjacent[:, None, None]).sum(-1)
            anchor_history_valid = field.anchor_mask[..., None].to(query.dtype) * (step > 0)
            motion_input = torch.cat((query, tool_motion, history, previous_d, head_mass, anchor_history_valid), -1)
            base_d = masked(self.motion(motion_input), field.anchor_mask)
            global_delta, global_trace = self.global_motion(x, step, query=query, anchor_valid=field.anchor_mask, hand_tokens=condition_trace['tokens62'][:, :2], cp=cp, cp_valid=valid, previous_cp=previous_cp, previous_valid=previous_valid)
            seed, direct_trace = self.direct_motion(x, step, query=query, anchor_xyz=anchors, anchor_valid=field.anchor_mask, hand_tokens=condition_trace['tokens62'][:, :2], hand_basis12=global_trace['hand_basis'], hand_valid=global_trace['hand_valid'], cp=cp, cp_valid=valid, previous_cp=previous_cp, previous_valid=previous_valid, anchor_history=history, previous_d=previous_d, global_hand_contribution=global_trace['hand_contribution'], sigma=field.bandwidth)
            local = masked(seed.sum(2), field.anchor_mask)
            g4_d = masked(base_d + global_delta[:, None], field.anchor_mask)
            d = masked(g4_d + local, field.anchor_mask)
            next_xyz = masked(initial + interpolate_anchor_displacements(d, field, keep), keep)
            if not torch.isfinite(next_xyz).all():
                raise FloatingPointError('Nonfinite G5 prediction')
            states.append(next_xyz)
            if return_trace:
                local_gs = interpolate_anchor_displacements(local, field, keep)
                count = keep.sum(1, keepdim=True).clamp_min(1).to(local_gs.dtype)
                mean = masked(local_gs, keep).sum(1) / count
                residual = masked(local_gs - mean[:, None], keep)
                rms = torch.linalg.vector_norm(residual, dim=(1, 2)) / count.squeeze(-1).sqrt()
                trace = dict(alpha=alpha, anchor_xyz=anchors, anchor_history=history, tool_motion=tool_motion, adjacent_valid=adjacent, head_history_mass=head_mass, base_cumulative_d=base_d, global_delta=global_delta, cumulative_d=d, previous_d=previous_d, state=next_xyz, g4_global_delta=global_delta, hand_basis12=global_trace['hand_basis'], direct_seed_per_hand=seed, local_correction=local, final_cumulative_anchor_displacement=d, local_gs_mean=mean, local_gs_demeaned_rms=rms, **global_trace, **condition_trace, **direct_trace)
                steps.append(trace)
            previous_anchor, previous_d = (anchors, d)
            previous_cp, previous_valid = (cp, valid)
            current = next_xyz
        prediction = {'xyz': torch.stack(states, 1)}
        for key, begin, end in (('log_scales', 0, 3), ('quaternions', 3, 7), ('opacity_logits', 7, 8), ('sh_dc', 8, 11)):
            prediction[key] = attr[:, None, :, begin:end].expand(-1, h, -1, -1)
        trace = {'steps': steps, 'field': field}
        return (prediction, trace) if return_trace else prediction


# Read compatibility for callers of the original state-dict architecture name.
G5DirectPredictor = ClothDynamicsPredictor
