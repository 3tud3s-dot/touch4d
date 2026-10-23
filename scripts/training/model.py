"""Shared point MLPs and causal cross-attention; absolute normalized XYZ output."""

import torch
from torch import nn


def masked(value, mask):
    return torch.where(mask.unsqueeze(-1), value, torch.zeros_like(value))


class CrossBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, query, memory, valid):
        memory = self.memory_norm(memory)
        update, _ = self.attention(self.query_norm(query), memory, memory,
                                   key_padding_mask=~valid, need_weights=False)
        query = query + update
        return query + self.ff(self.ff_norm(query))


class PositionPredictor(nn.Module):
    """Each horizon sees only its own condition prefix; queries never mix horizons.

    Tokens per condition frame: two states, 60 controllers, four tactile fingers.
    Finger readings retain their fixed 12x32 order, continuous values, validity,
    spatial validity, and age. Valid zero readings are not discarded. Controllers
    provide the spatial context; no hard pixel-to-controller assignment is made.
    """

    def __init__(self, width=128, heads=4, layers=2):
        super().__init__()
        if min(width, heads, layers) < 1 or width % heads:
            raise ValueError("Positive dimensions required; width must be divisible by heads")
        self.config = dict(width=width, heads=heads, layers=layers)
        def mlp(n):
            return nn.Sequential(nn.Linear(n, width), nn.GELU(), nn.Linear(width, width))
        self.gs_encoder = mlp(14)
        self.state_encoder = mlp(13)
        self.controller_encoder = mlp(3)
        self.tactile_encoder = mlp(384 * 3 + 1)
        # Each controller ID encodes hand, finger and within-finger point index.
        self.token_identity = nn.Embedding(66, width)
        self.condition_time = nn.Embedding(5, width)
        self.prediction_time = nn.Embedding(5, width)
        self.null_token = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList(CrossBlock(width, heads) for _ in range(layers))
        self.position_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))

    def encode_conditions(self, x):
        b = x["gs_xyz"].shape[0]
        state = self.state_encoder(masked(x["action_state"], x["action_valid"]))
        controllers = self.controller_encoder(masked(x["controller_points"], x["controller_mask"]))
        controllers = controllers.reshape(b, 5, 60, -1)
        valid = x["valid_mask"]
        fingers_valid = valid.flatten(-2).any(-1)
        tactile = torch.where(valid, x["tactile"], torch.zeros_like(x["tactile"]))
        age = torch.where(fingers_valid, x["tactile_age_us"], 0).to(tactile.dtype) / 75000.
        features = torch.cat((tactile.flatten(-2), valid.flatten(-2).to(tactile.dtype),
                              x["tactile_spatial_mask"].flatten(-2).to(tactile.dtype),
                              age.unsqueeze(-1)), dim=-1)
        fingers = self.tactile_encoder(features)
        tokens = torch.cat((state, controllers, fingers), dim=2)
        keep = torch.cat((x["action_valid"], x["controller_mask"].reshape(b, 5, 60),
                          fingers_valid), dim=2)
        tokens = tokens + self.token_identity.weight[None, None] + self.condition_time.weight[None, :, None]
        return masked(tokens, keep), keep

    def forward(self, inputs):
        x = inputs
        keep = x["gs_mask"]
        initial = masked(torch.cat((x["gs_xyz"], x["gs_attr"]), -1), keep)
        features = self.gs_encoder(initial)
        tokens, token_valid = self.encode_conditions(x)
        b = features.shape[0]
        outputs = []
        for k in range(5):
            # No condition encoder combines times before this causal slicing.
            memory = tokens[:, :k+1].flatten(1, 2)
            valid = token_valid[:, :k+1].flatten(1, 2)
            memory = torch.cat((self.null_token.expand(b, -1, -1), memory), dim=1)
            valid = torch.cat((torch.ones((b, 1), dtype=torch.bool, device=valid.device), valid), dim=1)
            query = features + self.prediction_time.weight[k]
            for block in self.blocks:
                query = block(query, memory, valid)
            # Deliberately no addition of initial XYZ and no delta parameterization.
            outputs.append(masked(self.position_head(query), keep))
        return torch.stack(outputs, dim=1)
