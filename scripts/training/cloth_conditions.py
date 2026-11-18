"""V4-only A0/T3 early fusion. Geometry follows the audited 10 mm T3 mapping.

Pressure is already scaled by DataProcessor; the kernel uses sensor-local metres.
Missing observations never become valid zero pressure; no fallback tokens exist.
"""
import torch
from torch import nn

from scripts.rebuild_dataset.rebuild import GRID_ROWS, GRID_COLS
from scripts.dataloader.hand_policy import HAND_POLICY, operation_hands
from scripts.dataloader.splits import validate_horizon
from .model import masked


def local_geometry():
    r, c = torch.meshgrid(torch.arange(12, dtype=torch.float64),
                          torch.arange(32, dtype=torch.float64), indexing="ij")
    taxels = torch.stack((torch.full_like(r, .007), -.056-.002*c, .0235-.002*r), -1).reshape(384, 3)
    r, c = torch.meshgrid(torch.tensor(GRID_ROWS, dtype=torch.float64),
                          torch.tensor(GRID_COLS, dtype=torch.float64), indexing="ij")
    points = torch.stack((torch.full_like(r, .007), -.056-.002*c,
                          (.025/12)*(11.5-r)), -1).reshape(15, 3)
    d2 = (points[:, None]-taxels[None]).square().sum(-1)
    roi = torch.arange(384) % 32 != 31
    partition = (d2.argmin(0)[None] == torch.arange(15)[:, None]) & roi
    return torch.exp(-d2/(2*.01**2)).float(), partition, roi


class V4ConditionEncoder(nn.Module):
    def __init__(self, horizon=10, width=128):
        super().__init__()
        self.horizon = validate_horizon(horizon)
        self.hand = nn.Sequential(nn.Linear(13, width), nn.GELU(), nn.Linear(width, width))
        self.controller = nn.Sequential(nn.Linear(13, width), nn.GELU(), nn.Linear(width, width))
        self.identity = nn.Embedding(62, width)
        self.time = nn.Embedding(horizon, width)
        self.hand_projection = nn.Linear(width, width, bias=False)
        self.fusion_norm = nn.LayerNorm(width)
        kernel, partition, roi = local_geometry()
        self.register_buffer("kernel", kernel)
        self.register_buffer("partition", partition)
        self.register_buffer("roi", roi)

    def features(self, x, step):
        """Read only the current sparse step and initial controller reference."""
        hands = x["operation_hands"]
        b = x["gs_xyz"].shape[0]
        if hands.dtype != torch.bool or hands.shape != (b, 2):
            raise ValueError("operation_hands must be bool [B,2]")
        action = x["action_state"][:, step]
        hand_valid = x["action_valid"][:, step] & hands & torch.isfinite(action).all(-1)
        action = masked(action, hand_valid)
        cp = x["controller_points"][:, step].reshape(b, 60, 3)
        cp0 = x["controller_points"][:, 0].reshape(b, 60, 3)
        allowed = hands.repeat_interleave(30, -1)
        valid = x["controller_mask"][:, step].reshape(b, 60) & allowed & torch.isfinite(cp).all(-1)
        valid0 = x["controller_mask"][:, 0].reshape(b, 60) & allowed & torch.isfinite(cp0).all(-1)
        cp, cp0 = masked(cp, valid), masked(cp0, valid0)
        endpoints = valid & valid0
        geom = torch.cat((cp, masked(cp-cp0, endpoints), endpoints[..., None].to(cp.dtype)), -1)
        pressure = x["tactile"][:, step].flatten(-2)
        age = x["tactile_age_us"][:, step]
        age_valid = torch.isfinite(age) & (age >= 0) & (age <= 75000)
        taxel_valid = (x["valid_mask"][:, step].flatten(-2)
                       & x["tactile_spatial_mask"][:, step].flatten(-2)
                       & hands.repeat_interleave(2, -1)[..., None]
                       & age_valid[..., None] & torch.isfinite(pressure) & self.roi)
        pressure = torch.where(taxel_valid, pressure, 0.)
        weights = taxel_valid[..., None, :].to(pressure.dtype) * self.kernel
        denominator = weights.sum(-1)
        mean_valid = denominator > 0
        mean = (weights * pressure[..., None, :]).sum(-1) / denominator.clamp_min(torch.finfo(pressure.dtype).tiny)
        local = taxel_valid[..., None, :] & self.partition
        max_valid = local.any(-1)
        maximum = pressure[..., None, :].expand_as(weights).masked_fill(~local, -torch.inf).amax(-1)
        maximum = torch.where(max_valid, maximum, 0.)
        coverage = local.sum(-1).to(pressure.dtype) / self.partition.sum(-1).clamp_min(1)
        safe_age = torch.where(age_valid, age, 0).to(pressure.dtype) / 75000.
        age_feature = torch.where(mean_valid, safe_age[..., None], 0.)
        tactile = torch.stack((mean, maximum, coverage, age_feature,
                               mean_valid.to(pressure.dtype), max_valid.to(pressure.dtype)), -1).reshape(b, 60, 6)
        features = masked(torch.cat((geom, tactile), -1), valid)
        return action, hand_valid, features, valid, cp

    def forward(self, x, step):
        action, hand_valid, features, valid, cp = self.features(x, step)
        identity = self.identity.weight
        time = self.time.weight[step]
        hand = masked(self.hand(action) + identity[:2] + time, hand_valid)
        controller = masked(self.controller(features) + identity[2:] + time, valid)
        fused = self.fusion_norm(controller + self.hand_projection(hand).repeat_interleave(30, 1))
        return masked(fused, valid), valid, cp, {
            "features13": features, "tokens62": torch.cat((hand, controller), 1),
            "token_mask62": torch.cat((hand_valid, valid), 1)}
