"""Independent Stage 1 path: sparse causal conditions, per-head motion, fixed RBF.

No Structural Anchor Attention, fallback, future GS queries or attribute heads.
The legacy PositionPredictor and all its checkpoint formats remain untouched.
"""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F

from .model import LocalGSBlock, initial_knn, masked
from .gs_decoder import build_initial_anchor_field, pool_anchor_features, interpolate_anchor_displacements
from .v4_conditions import V4ConditionEncoder, HAND_POLICY
from .v4_attention import GeometryCrossBlock


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


class V4Stage1Predictor(nn.Module):
    def __init__(self, horizon=5, prediction_interval=10):
        super().__init__()
        if type(horizon) is not int or horizon != 5:
            raise ValueError("V4 Stage 1 is frozen to H=5")
        if type(prediction_interval) is not int or prediction_interval != 10:
            raise ValueError("V4 Stage 1 is frozen to prediction_interval=10")
        self.config = dict(architecture="v4_stage1", horizon=horizon, prediction_interval=prediction_interval,
                           width=128, heads=4, layers=2, local_k=16, anchor_count=128, pool_k=16,
                           anchor_bandwidth=1., sigma_geo_multiplier=1., hand_policy=HAND_POLICY,
                           controller_features=13, tactile_sigma_m=.01, history_flags="head_mass4_anchor1",
                           output_mode="gs_geometry", displacement_reference="initial_cumulative",
                           selection_metric="joint", loss_weights=dict(cd=1., rgb=1., alpha=.1, scale=0., depth=0.))
        self.gs_encoder = nn.Sequential(nn.Linear(14, 128), nn.GELU(), nn.Linear(128, 128))
        self.local = LocalGSBlock(128, 4)
        self.conditions = V4ConditionEncoder(horizon)
        self.pred_time = nn.Embedding(horizon, 128)
        self.cross = nn.ModuleList([GeometryCrossBlock() for _ in range(2)])
        self.motion = nn.Sequential(nn.Linear(151, 128), nn.GELU(), nn.Linear(128, 3))
        nn.init.zeros_(self.motion[-1].weight)
        nn.init.zeros_(self.motion[-1].bias)

    def forward(self, x, *, return_trace=False):
        h = self.config["horizon"]
        frame_ids = x.get("condition_frame_ids")
        if (frame_ids is None or frame_ids.dtype != torch.int64
                or frame_ids.shape != (len(x["gs_xyz"]), h)
                or not torch.all(frame_ids[:, 1:]-frame_ids[:, :-1] == self.config["prediction_interval"])):
            raise ValueError("V4 requires checked sparse condition_frame_ids [B,5] with interval 10")
        if any(x[key].shape[1] != h for key in ("action_state", "controller_points", "tactile")):
            raise ValueError("V4 requires exactly five sampled condition frames")
        keep = x["gs_mask"]
        if keep.dtype != torch.bool or not keep.any(1).all():
            raise ValueError("Each sample needs at least one valid GS")
        initial, attr = masked(x["gs_xyz"], keep), masked(x["gs_attr"], keep)
        if not torch.isfinite(initial).all() or not torch.isfinite(attr).all():
            raise ValueError("Nonfinite valid initial GS")
        field = fixed_field(initial, keep)
        neighbors = initial_knn(initial, keep, 16)
        batches = torch.arange(len(initial), device=initial.device)[:, None]
        current = initial
        previous_anchor = masked(current[batches, field.anchor_indices], field.anchor_mask)
        previous_d = torch.zeros_like(previous_anchor)
        previous_cp = previous_valid = None
        states, traces = [], []
        for step in range(h):
            features = masked(self.gs_encoder(torch.cat((current, attr), -1)), keep)
            features = self.local(features, current, keep, neighbors)
            query = masked(pool_anchor_features(features, field) + self.pred_time.weight[step], field.anchor_mask)
            anchors = masked(current[batches, field.anchor_indices], field.anchor_mask)
            tools, valid, cp, condition_trace = self.conditions(x, step)
            for block in self.cross:
                query, alpha = block(query, tools, anchors, cp, field.bandwidth,
                                     field.anchor_mask, valid)
            adjacent = torch.zeros_like(valid) if step == 0 else valid & previous_valid
            cp_delta = torch.zeros_like(cp) if step == 0 else masked(cp-previous_cp, adjacent)
            weights = alpha[..., :60]  # Do not average heads or renormalize away null.
            tool_motion = torch.einsum("bahj,bjd->bahd", weights, cp_delta).flatten(-2)
            history = torch.zeros_like(anchors) if step == 0 else anchors-previous_anchor
            head_mass = (weights*adjacent[:, None, None]).sum(-1)
            anchor_history_valid = field.anchor_mask[..., None].to(query.dtype) * (step > 0)
            motion_input = torch.cat((query, tool_motion, history, previous_d,
                                       head_mass, anchor_history_valid), -1)
            d = masked(self.motion(motion_input), field.anchor_mask)
            next_xyz = masked(initial + interpolate_anchor_displacements(d, field, keep), keep)
            if not torch.isfinite(next_xyz).all():
                raise FloatingPointError("Nonfinite V4 prediction")
            states.append(next_xyz)
            if return_trace:
                traces.append(dict(alpha=alpha, anchor_xyz=anchors, anchor_history=history,
                                   tool_motion=tool_motion, adjacent_valid=adjacent, head_history_mass=head_mass,
                                   cumulative_d=d, previous_d=previous_d, state=next_xyz, **condition_trace))
            previous_anchor, previous_d = anchors, d
            previous_cp, previous_valid = cp, valid
            current = next_xyz  # Full AR graph: never detach or replace with GT.
        prediction = {"xyz": torch.stack(states, 1)}
        for key, begin, end in (("log_scales", 0, 3), ("quaternions", 3, 7),
                                ("opacity_logits", 7, 8), ("sh_dc", 8, 11)):
            prediction[key] = attr[:, None, :, begin:end].expand(-1, h, -1, -1)
        return (prediction, {"steps": traces, "field": field}) if return_trace else prediction
