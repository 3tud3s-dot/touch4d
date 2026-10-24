"""Causal five-step displacement prediction; return normalized XYZ."""

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
    """Each horizon reads only its condition prefix and prior predicted states.

    Tokens per condition frame: two states, 60 controllers, four tactile fingers.
    Finger readings retain their fixed 12x32 order, continuous values, validity,
    spatial validity, and age. Valid zero readings are not discarded. Controllers
    provide the spatial context; no hard pixel-to-controller assignment is made.
    """

    def __init__(self, width=128, heads=4, layers=2, controller_features=7,
                 prediction_mode="autoregressive"):
        super().__init__()
        if min(width, heads, layers) < 1 or width % heads:
            raise ValueError("Positive dimensions required; width must be divisible by heads")
        if type(controller_features) is not int or controller_features not in (3, 7):
            raise ValueError("controller_features must be 3 (legacy) or 7 (relative motion)")
        if prediction_mode not in ("independent", "autoregressive"):
            raise ValueError("prediction_mode must be independent or autoregressive")
        self.config = dict(width=width, heads=heads, layers=layers,
                           controller_features=controller_features, prediction_mode=prediction_mode)
        def mlp(n):
            return nn.Sequential(nn.Linear(n, width), nn.GELU(), nn.Linear(width, width))
        self.gs_encoder = mlp(14)
        self.state_encoder = mlp(13)
        self.controller_encoder = mlp(controller_features)
        self.tactile_encoder = mlp(384 * 3 + 1)
        # Each controller ID encodes hand, finger and within-finger point index.
        self.token_identity = nn.Embedding(66, width)
        self.condition_time = nn.Embedding(5, width)
        self.prediction_time = nn.Embedding(5, width)
        self.null_token = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList(CrossBlock(width, heads) for _ in range(layers))
        self.position_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
        # A zero displacement head copies the initial GS across all five steps.
        nn.init.zeros_(self.position_head[-1].weight)
        nn.init.zeros_(self.position_head[-1].bias)

    def controller_inputs(self, x):
        """Same-frame XYZ, optional displacement from t0 and endpoint validity.

        Sanitize operands before subtraction. A missing reference disables only
        the displacement; current valid coordinates still produce a token.
        Coordinates are already normalized, so differences need no rescaling.
        """
        valid = x["controller_mask"]
        coordinates = masked(x["controller_points"], valid)
        if self.config["controller_features"] == 3:
            return coordinates
        endpoints = valid & valid[:, :1]
        displacement = masked(coordinates - coordinates[:, :1], endpoints)
        return torch.cat((coordinates, displacement,
                          endpoints.unsqueeze(-1).to(coordinates.dtype)), dim=-1)

    def encode_conditions(self, x):
        b = x["gs_xyz"].shape[0]
        state = self.state_encoder(masked(x["action_state"], x["action_valid"]))
        controllers = self.controller_encoder(self.controller_inputs(x))
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
        autoregressive = self.config["prediction_mode"] == "autoregressive"
        current = initial[..., :3]
        features = None if autoregressive else self.gs_encoder(initial)
        tokens, token_valid = self.encode_conditions(x)
        b = initial.shape[0]
        outputs = []
        for k in range(5):
            if autoregressive:
                # Re-encode predicted positions with unchanged initial attributes.
                # Keep the full graph through every preceding step (no detach).
                features = self.gs_encoder(masked(torch.cat((current, initial[..., 3:]), -1), keep))
            # No condition encoder combines times before this causal slicing.
            memory = tokens[:, :k+1].flatten(1, 2)
            valid = token_valid[:, :k+1].flatten(1, 2)
            memory = torch.cat((self.null_token.expand(b, -1, -1), memory), dim=1)
            valid = torch.cat((torch.ones((b, 1), dtype=torch.bool, device=valid.device), valid), dim=1)
            query = features + self.prediction_time.weight[k]
            for block in self.blocks:
                query = block(query, memory, valid)
            reference = current if autoregressive else initial[..., :3]
            current = masked(reference + self.position_head(query), keep)
            outputs.append(current)
        return torch.stack(outputs, dim=1)
