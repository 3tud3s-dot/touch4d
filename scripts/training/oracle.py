"""Diagnostic-only latent/anchor fitting with the unchanged isotropic P1a field.

These modules optimize variables attached to known windows. They are not causal
condition predictors and their checkpoints cannot be loaded as production models.
Targets are consumed by the ordinary external loss, never by this forward path.
"""

import copy
import random

import torch
from torch import nn

from .gs_decoder import (build_initial_anchor_field, pool_anchor_features,
                         interpolate_anchor_displacements)
from .model import masked, initial_knn


ORACLE_VERSION = "touch4d-oracle-v1"


class _Oracle(nn.Module):
    def __init__(self, base, window_masks, mode):
        super().__init__()
        config = base.config
        if (config.get("gs_decoder") != "anchors"
                or config.get("prediction_mode") != "autoregressive"
                or config.get("output_mode") != "gs_geometry"
                or config.get("gs_local_attention") != "knn"
                or config.get("gs_structure", "none") != "none"):
            raise ValueError("Oracle requires the isotropic P1a AR GS model")
        self.horizon = config.get("horizon", 5)
        if (type(self.horizon) is not int or self.horizon < 1
                or base.prediction_time.num_embeddings != self.horizon):
            raise ValueError("Oracle horizon must match the base time embedding")
        if mode not in ("latent", "anchor") or not len(window_masks):
            raise ValueError("Oracle needs a supported mode and at least one window")
        # Copy ownership: freezing unused condition modules must not affect a
        # caller's P1a baseline or its optimizer.
        self.base = copy.deepcopy(base)
        self.mode = mode
        self.config = dict(self.base.config, oracle_mode=mode, diagnostic_only=True)
        self.base.requires_grad_(False)
        if mode == "latent":
            for name in ("gs_encoder", "gs_local", "position_head", "prediction_time"):
                getattr(self.base, name).requires_grad_(True)
        reference = next(self.base.parameters())
        width = config["width"] if mode == "latent" else 3
        self.variables = nn.ParameterList()
        for index, value in enumerate(window_masks):
            if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
                raise ValueError("Each window mask must be a Boolean tensor")
            if value.ndim == 2 and value.shape[0] == 1:
                value = value[0]
            if value.ndim != 1:
                raise ValueError("Oracle fixed windows require masks [N] or [1,N]")
            keep = value.detach().to(reference.device).clone()
            self.register_buffer(f"window_mask_{index}", keep)
            slots = min(config["anchor_count"], len(keep))
            # Separate Parameters ensure an unselected window has grad=None;
            # AdamW must not advance its momentum or decay its latent values.
            self.variables.append(nn.Parameter(reference.new_zeros(self.horizon, slots, width)))

    def forward(self, inputs, window_id):
        if type(window_id) is not int or not 0 <= window_id < len(self.variables):
            raise ValueError("Unknown Oracle window_id")
        keep = inputs["gs_mask"]
        expected = getattr(self, f"window_mask_{window_id}")
        if (keep.dtype != torch.bool or keep.ndim != 2 or keep.shape != (1, len(expected))
                or not torch.equal(keep[0], expected)):
            raise ValueError("Oracle input mask differs from its registered fixed window")
        initial = masked(torch.cat((inputs["gs_xyz"], inputs["gs_attr"]), -1), keep)
        if initial.shape != (1, len(expected), 14) or not torch.isfinite(initial).all():
            raise ValueError("Oracle requires finite valid initial 14D GS features")
        config = self.base.config
        field = build_initial_anchor_field(initial[..., :3], keep, config["anchor_count"],
                                           config["anchor_pool_k"], config["anchor_bandwidth"])
        neighbors = (initial_knn(initial[..., :3], keep, config["gs_local_k"])
                     if self.mode == "latent" else None)
        current, outputs = initial[..., :3], []
        for h in range(self.horizon):
            value = self.variables[window_id][h][None]
            if self.mode == "latent":
                features = self.base.gs_encoder(masked(torch.cat((current, initial[..., 3:]), -1), keep))
                features = self.base.gs_local(features, current, keep, neighbors)
                query = (pool_anchor_features(features, field)
                         + self.base.prediction_time.weight[h] + value)
                displacement = self.base.position_head(query)
            else:
                displacement = value
            delta = interpolate_anchor_displacements(displacement, field, keep)
            # Each latent/anchor parameter predicts the TOTAL displacement from
            # the initial template. Latent mode preserves the live AR feedback.
            current = masked(initial[..., :3] + delta, keep)
            outputs.append(current)
        return dict(xyz=torch.stack(outputs, 1),
                    log_scales=initial[..., 3:6][:, None].expand(-1, self.horizon, -1, -1),
                    quaternions=initial[..., 6:10][:, None].expand(-1, self.horizon, -1, -1),
                    opacity_logits=initial[..., 10:11][:, None].expand(-1, self.horizon, -1, -1),
                    sh_dc=initial[..., 11:14][:, None].expand(-1, self.horizon, -1, -1))

    def diagnostic_checkpoint(self, optimizer=None, **metadata):
        """Explicitly incompatible with train.read_checkpoint/load_model."""
        return dict(version=ORACLE_VERSION, diagnostic_only=True, oracle_mode=self.mode,
                    config=copy.deepcopy(self.config), model=self.state_dict(),
                    optimizer=None if optimizer is None else optimizer.state_dict(),
                    metadata=metadata, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    python_rng=random.getstate())

    def load_diagnostic_checkpoint(self, state, optimizer=None, restore_rng=True):
        saved_config = dict(state.get("config", {}))
        # These disabled/default semantics predate the fields. Normalization is
        # read-only and must never turn a five-frame P1a Oracle into P2/H=15.
        for key, default in (("gs_structure", "none"), ("horizon", 5)):
            if key in self.config:
                saved_config.setdefault(key, default)
        if (state.get("version") != ORACLE_VERSION or state.get("diagnostic_only") is not True
                or state.get("oracle_mode") != self.mode or saved_config != self.config):
            raise ValueError("Oracle checkpoint mode/configuration mismatch")
        if optimizer is not None and state.get("optimizer") is None:
            raise ValueError("Oracle checkpoint has no optimizer state")
        values = state.get("model", {})
        for index in range(len(self.variables)):
            name = f"window_mask_{index}"
            saved = values.get(name)
            current = getattr(self, name)
            if not isinstance(saved, torch.Tensor) or not torch.equal(saved.cpu(), current.cpu()):
                raise ValueError("Oracle checkpoint fixed-window mask mismatch")
        self.load_state_dict(values, strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
        if restore_rng:
            torch.set_rng_state(state["torch_rng"].cpu())
            if state.get("cuda_rng") is not None and torch.cuda.is_available():
                if len(state["cuda_rng"]) != torch.cuda.device_count():
                    raise ValueError("Oracle RNG restore requires the same visible CUDA devices")
                torch.cuda.set_rng_state_all([state.cpu() for state in state["cuda_rng"]])
            random.setstate(state["python_rng"])
        return state.get("metadata", {})


class OracleLatent(_Oracle):
    """Known-window anchor latents replace the condition-fused query features."""
    def __init__(self, base, window_masks):
        super().__init__(base, window_masks, "latent")


class OracleAnchor(_Oracle):
    """Direct cumulative anchor displacements: same RBF function class, no encoder."""
    def __init__(self, base, window_masks):
        super().__init__(base, window_masks, "anchor")
