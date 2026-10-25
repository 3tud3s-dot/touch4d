"""Causal five-step position or full Gaussian-geometry prediction."""

import math

import torch
from torch import nn

from .gs_decoder import (build_initial_anchor_field, pool_anchor_features,
                         interpolate_anchor_displacements)


def masked(value, mask):
    return torch.where(mask.unsqueeze(-1), value, torch.zeros_like(value))


def rotation_vector_to_quaternion(rotation):
    """World-axis rotation vector in radians to Hamilton WXYZ, stable at zero.

    sinc avoids dividing by the angle, so a zero-initialized rotation head still
    has a finite, nonzero derivative for each of its three output coordinates.
    """
    angle = torch.linalg.vector_norm(rotation, dim=-1, keepdim=True)
    return torch.cat((torch.cos(angle * .5),
                      rotation * (.5 * torch.sinc(angle / (2. * math.pi)))), dim=-1)


def quaternion_multiply(left, right):
    """Hamilton product in WXYZ order; a world-frame increment is the left input."""
    lw, lv = left[..., :1], left[..., 1:]
    rw, rv = right[..., :1], right[..., 1:]
    return torch.cat((lw * rw - (lv * rv).sum(-1, keepdim=True),
                      lw * rv + rw * lv + torch.linalg.cross(lv, rv, dim=-1)), dim=-1)


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


@torch.no_grad()
def initial_knn(xyz, keep, k=16, chunk=512):
    """Exact initial-state neighborhoods, including self; no padded/batch edges.

    Only indices are nondifferentiable. Distances are computed in query chunks,
    once per rollout, without storing a global attention/distance matrix.
    """
    if type(k) is not int or k < 1 or type(chunk) is not int or chunk < 1:
        raise ValueError("k and chunk must be positive integers")
    b, n, _ = xyz.shape
    slots = min(k, n)
    indices = torch.zeros((b, n, slots), dtype=torch.long, device=xyz.device)
    valid = torch.zeros_like(indices, dtype=torch.bool)
    for batch in range(b):
        ids = keep[batch].nonzero(as_tuple=True)[0]
        count = len(ids)
        if not count:
            continue
        points = xyz[batch, ids].float()
        if not torch.isfinite(points).all():
            raise ValueError("Nonfinite valid GS coordinates in initial neighborhood")
        used = min(k, count)
        indices[batch, ids, 0] = ids
        valid[batch, ids, :used] = True
        if used == 1:
            continue
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            distances = torch.cdist(points[start:stop], points,
                                    compute_mode="donot_use_mm_for_euclid_dist")
            # Force self into slot zero even when multiple centers coincide.
            distances[torch.arange(stop-start, device=xyz.device),
                      torch.arange(start, stop, device=xyz.device)] = float("inf")
            nearest = distances.topk(used-1, largest=False, sorted=True).indices
            indices[batch, ids[start:stop], 1:used] = ids[nearest]
    return indices, valid


class LocalGSBlock(nn.Module):
    """One shared local multihead attention layer with live relative geometry."""
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.relative = nn.Sequential(nn.Linear(3, width), nn.GELU(),
                                      nn.Linear(width, width + heads))
        self.projection = nn.Linear(width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, features, xyz, keep, neighbors):
        indices, valid = neighbors
        features, xyz = masked(features, keep), masked(xyz, keep)
        b, n, width = features.shape
        if n == 0:
            return features
        batch = torch.arange(b, device=features.device)[:, None, None]
        q, key, value = self.qkv(self.norm(features)).chunk(3, dim=-1)
        q = q.reshape(b, n, self.heads, self.head_dim)
        key = key[batch, indices].reshape(b, n, -1, self.heads, self.head_dim)
        value = value[batch, indices].reshape_as(key)
        # Gradients through the current (possibly predicted) coordinates remain live.
        position = self.relative(xyz[batch, indices] - xyz[:, :, None])
        bias, relative_value = position[..., :self.heads], position[..., self.heads:]
        logits = (q[:, :, None] * key).sum(-1) / math.sqrt(self.head_dim) + bias
        logits = logits.masked_fill(~valid[..., None], float("-inf"))
        # Empty padded queries must not produce softmax(-inf,...,-inf) NaNs.
        logits = torch.where(valid.any(-1)[:, :, None, None], logits, 0.)
        weights = logits.softmax(dim=2).masked_fill(~valid[..., None], 0.)
        update = (weights[..., None] * (value + relative_value.reshape_as(value))).sum(2)
        features = features + self.projection(update.reshape(b, n, width))
        return masked(features + self.ff(self.ff_norm(features)), keep)


class PositionPredictor(nn.Module):
    """Each horizon reads only its condition prefix and prior predicted states.

    Tokens per condition frame: two states, 60 controllers, four tactile fingers.
    Finger readings retain their fixed 12x32 order, continuous values, validity,
    spatial validity, and age. Valid zero readings are not discarded. Controllers
    provide the spatial context; no hard pixel-to-controller assignment is made.
    """

    def __init__(self, width=128, heads=4, layers=2, controller_features=7,
                 prediction_mode="autoregressive", output_mode="position",
                 gs_local_attention="none", gs_local_k=16, gs_decoder="free",
                 anchor_count=128, anchor_pool_k=16, anchor_bandwidth=1.0):
        super().__init__()
        if min(width, heads, layers) < 1 or width % heads:
            raise ValueError("Positive dimensions required; width must be divisible by heads")
        if type(controller_features) is not int or controller_features not in (3, 7):
            raise ValueError("controller_features must be 3 (legacy) or 7 (relative motion)")
        if prediction_mode not in ("independent", "autoregressive"):
            raise ValueError("prediction_mode must be independent or autoregressive")
        if output_mode not in ("position", "gs_geometry"):
            raise ValueError("output_mode must be position or gs_geometry")
        if output_mode == "gs_geometry" and prediction_mode != "autoregressive":
            raise ValueError("gs_geometry requires autoregressive prediction_mode")
        if gs_local_attention not in ("none", "knn"):
            raise ValueError("gs_local_attention must be none or knn")
        if type(gs_local_k) is not int or gs_local_k < 1:
            raise ValueError("gs_local_k must be a positive integer")
        if gs_decoder not in ("free", "anchors"):
            raise ValueError("gs_decoder must be free or anchors")
        if (gs_decoder == "anchors" and
                (prediction_mode != "autoregressive" or output_mode != "gs_geometry"
                 or gs_local_attention != "knn")):
            raise ValueError("anchors requires autoregressive gs_geometry and knn local attention")
        if type(anchor_count) is not int or anchor_count < 1:
            raise ValueError("anchor_count must be a positive integer")
        if type(anchor_pool_k) is not int or anchor_pool_k < 1:
            raise ValueError("anchor_pool_k must be a positive integer")
        if (isinstance(anchor_bandwidth, bool) or not isinstance(anchor_bandwidth, (int, float))
                or not math.isfinite(anchor_bandwidth) or anchor_bandwidth <= 0):
            raise ValueError("anchor_bandwidth must be finite and positive")
        self.config = dict(width=width, heads=heads, layers=layers,
                           controller_features=controller_features, prediction_mode=prediction_mode,
                           output_mode=output_mode, gs_local_attention=gs_local_attention,
                           gs_local_k=gs_local_k, gs_decoder=gs_decoder,
                           anchor_count=anchor_count, anchor_pool_k=anchor_pool_k,
                           anchor_bandwidth=anchor_bandwidth)
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
        if output_mode == "gs_geometry":
            self.scale_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
            self.rotation_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
            for head in (self.scale_head, self.rotation_head):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        # Disabled legacy path creates no parameters and consumes no extra RNG.
        self.gs_local = LocalGSBlock(width, heads) if gs_local_attention == "knn" else None
        if gs_decoder == "anchors":
            # Construct the free geometry heads before LocalGS, then remove them.
            # This preserves the P0 RNG stream and identical initial shared weights
            # for paired experiments without retaining unused trainable heads.
            del self.scale_head, self.rotation_head

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
        geometry = self.config["output_mode"] == "gs_geometry"
        anchors = self.config["gs_decoder"] == "anchors"
        current = initial[..., :3]
        current_scales, current_quats = initial[..., 3:6], initial[..., 6:10]
        features = None if autoregressive else self.gs_encoder(initial)
        neighbors = (initial_knn(current, keep, self.config["gs_local_k"])
                     if self.gs_local is not None else None)
        anchor_field = (build_initial_anchor_field(
            current, keep, self.config["anchor_count"], self.config["anchor_pool_k"],
            self.config["anchor_bandwidth"]) if anchors else None)
        if not autoregressive and self.gs_local is not None:
            features = self.gs_local(features, current, keep, neighbors)
        tokens, token_valid = self.encode_conditions(x)
        b = initial.shape[0]
        outputs = []
        scale_outputs, quaternion_outputs = [], []
        for k in range(5):
            if autoregressive:
                # GS mode feeds back updated geometry, with initial appearance.
                # Both modes keep the full graph through every preceding step.
                attrs = (torch.cat((current_scales, current_quats, initial[..., 10:]), -1)
                         if geometry else initial[..., 3:])
                features = self.gs_encoder(masked(torch.cat((current, attrs), -1), keep))
                if self.gs_local is not None:
                    features = self.gs_local(features, current, keep, neighbors)
            # No condition encoder combines times before this causal slicing.
            memory = tokens[:, :k+1].flatten(1, 2)
            valid = token_valid[:, :k+1].flatten(1, 2)
            memory = torch.cat((self.null_token.expand(b, -1, -1), memory), dim=1)
            valid = torch.cat((torch.ones((b, 1), dtype=torch.bool, device=valid.device), valid), dim=1)
            query = features + self.prediction_time.weight[k]
            for block in self.blocks:
                query = block(query, memory, valid)
            if anchors:
                # The head predicts total displacement from t0 at every horizon.
                # Previous predictions still feed the live AR encoder above; do
                # not add this cumulative field to the previous predicted center.
                anchor_delta = self.position_head(pool_anchor_features(query, anchor_field))
                delta = interpolate_anchor_displacements(anchor_delta, anchor_field, keep)
                current = masked(initial[..., :3] + delta, keep)
            else:
                reference = current if autoregressive else initial[..., :3]
                current = masked(reference + self.position_head(query), keep)
            outputs.append(current)
            if geometry:
                if not anchors:
                    current_scales = masked(current_scales + self.scale_head(query), keep)
                    # Rotate each Gaussian's axes in world coordinates. The center
                    # is translated above and is never rotated about the origin.
                    increment = rotation_vector_to_quaternion(self.rotation_head(query))
                    current_quats = quaternion_multiply(increment, current_quats)
                    current_quats = masked(torch.nn.functional.normalize(current_quats, dim=-1), keep)
                scale_outputs.append(current_scales)
                quaternion_outputs.append(current_quats)
        xyz = torch.stack(outputs, dim=1)
        if not geometry:
            return xyz
        return {"xyz": xyz, "log_scales": torch.stack(scale_outputs, dim=1),
                "quaternions": torch.stack(quaternion_outputs, dim=1),
                "opacity_logits": initial[..., 10:11][:, None].expand(-1, 5, -1, -1),
                "sh_dc": initial[..., 11:14][:, None].expand(-1, 5, -1, -1)}
