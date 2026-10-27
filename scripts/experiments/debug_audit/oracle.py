"""Offline D0 capacity audit: optimize anchor displacements, never a predictor.

Targets are permitted ONLY in this offline upper-bound diagnostic. This module
does not alter the production decoder, fit a condition encoder or claim physical
particle correspondence between independently reconstructed target GS frames.
"""

from pathlib import Path
import time

import torch
from torch import nn

from scripts.training.gs_decoder import (
    build_initial_anchor_field, interpolate_anchor_displacements,
)
from scripts.training.losses import nearest_distances, objective
from scripts.training.train import atomic_checkpoint, atomic_json, copy_initial
from scripts.experiments.fixed_window.fixed_fit import prediction_metrics


class D0Oracle(nn.Module):
    """Independent cumulative [B,H,A,3] variables over the frozen initial RBF."""

    def __init__(self, inputs, horizon):
        super().__init__()
        if type(horizon) is not int or horizon < 1:
            raise ValueError("Positive integer horizon required")
        keep = inputs["gs_mask"].detach().clone()
        if not keep.any(-1).all():
            raise ValueError("An Oracle sample must contain valid initial GS")
        self.register_buffer("keep", keep)
        self.register_buffer("initial", torch.where(
            keep[..., None], inputs["gs_xyz"], 0.).detach().clone())
        self.register_buffer("attributes", torch.where(
            keep[..., None], inputs["gs_attr"], 0.).detach().clone())
        self.field = build_initial_anchor_field(self.initial, keep, 128, 16, 1.)
        self.delta = nn.Parameter(self.initial.new_zeros(
            len(self.initial), horizon, self.field.anchor_indices.shape[1], 3))
        self.horizon = horizon

    def forward(self):
        xyz = torch.stack([
            self.initial + interpolate_anchor_displacements(self.delta[:, h], self.field, self.keep)
            for h in range(self.horizon)
        ], 1)
        expand = lambda value: value[:, None].expand(-1, self.horizon, -1, -1)
        return dict(xyz=xyz, log_scales=expand(self.attributes[..., :3]),
                    quaternions=expand(self.attributes[..., 3:7]),
                    opacity_logits=expand(self.attributes[..., 7:8]),
                    sh_dc=expand(self.attributes[..., 8:11]))


@torch.no_grad()
def centroid_translation_oracle(batch):
    """Translate X0 by independent-set centroid differences, not GT row deltas."""
    pred = copy_initial(batch["inputs"], "gs_geometry")
    xyz = pred["xyz"].clone()
    keep = batch["inputs"]["gs_mask"]
    for b in range(len(xyz)):
        if not keep[b].any():
            raise ValueError("Empty initial point set")
        center = batch["inputs"]["gs_xyz"][b, keep[b]].mean(0)
        for h in range(xyz.shape[1]):
            target = batch["targets"]["target_xyz"][h][b]
            target_keep = batch["targets"]["target_mask"][h][b]
            if not target_keep.any():
                raise ValueError("Empty Oracle target point set")
            shift = target[target_keep].mean(0) - center
            xyz[b, h] = torch.where(keep[b, :, None], xyz[b, h] + shift, 0.)
    pred["xyz"] = xyz
    return pred


@torch.no_grad()
def field_diagnostics(oracle, scale):
    field, keep = oracle.field, oracle.keep
    rows = field.weights.sum(-1)
    valid = rows[keep]
    return dict(anchor_count=field.anchor_mask.sum(-1).tolist(),
                bandwidth_mm=(field.bandwidth * float(scale) * 1000).tolist(),
                row_sum_min=float(valid.min()), row_sum_max=float(valid.max()),
                row_sum_max_error=float((valid-1).abs().max()),
                padding_weight_abs_sum=float(field.weights[~keep].abs().sum()),
                fixed_geometry_requires_grad=bool(field.weights.requires_grad),
                displacement_requires_grad=bool(oracle.delta.requires_grad))


@torch.no_grad()
def oracle_metrics(prediction, batch, processor, renderer, oracle=None):
    """Use the existing independent-set/render metric implementation verbatim."""
    if prediction["xyz"].shape[0] != 1:
        raise ValueError("This fixed-window audit requires batch_size=1")
    result = prediction_metrics(prediction, batch, processor, renderer, 512)
    units = float(processor.state_dict()["scale"]) * 1000
    keep = batch["inputs"]["gs_mask"][0]
    initial = batch["inputs"]["gs_xyz"][0, keep]
    motion = []
    for h in range(prediction["xyz"].shape[1]):
        points = prediction["xyz"][0, h, keep]
        target = batch["targets"]["target_xyz"][h][0, batch["targets"]["target_mask"][h][0]]
        delta = points - initial
        average = delta.mean(0)
        p = points - points.mean(0)
        t = target - target.mean(0)
        centered = .5 * (nearest_distances(p, t).mean() + nearest_distances(t, p).mean())
        row = dict(displacement_mean_mm=(average*units).tolist(),
                   displacement_rms_mm=float(delta.square().sum(-1).mean().sqrt()*units),
                   displacement_max_mm=float(delta.norm(dim=-1).max()*units),
                   nonrigid_residual_rms_mm=float((delta-average).square().sum(-1).mean().sqrt()*units),
                   centered_cd_mm=float(centered*units),
                   gt_centroid_motion_mm=((target.mean(0)-initial.mean(0))*units).tolist(),
                   centroid_error_mm=float((points.mean(0)-target.mean(0)).norm()*units))
        if oracle is not None:
            anchors = oracle.delta[0, h, oracle.field.anchor_mask[0]]
            row.update(anchor_mean_mm=(anchors.mean(0)*units).tolist(),
                       anchor_rms_mm=float(anchors.square().sum(-1).mean().sqrt()*units),
                       anchor_max_mm=float(anchors.norm(dim=-1).max()*units))
        motion.append(row)
    result["motion"] = motion
    return result


def _save_prediction(path, prediction, oracle=None):
    payload = dict(prediction={k: v.detach().cpu() for k, v in prediction.items()},
                   offline_oracle_only=True)
    if oracle is not None:
        payload.update(anchor_delta=oracle.delta.detach().cpu(),
                       anchor_indices=oracle.field.anchor_indices.cpu(),
                       anchor_mask=oracle.field.anchor_mask.cpu(),
                       weights=oracle.field.weights.cpu())
    atomic_checkpoint(path, payload)


def run_oracle(batch, processor, renderer, output, updates=400, lr=1e-3, full_loss=False):
    """Bounded zero-initialized D0 Oracle, with fresh-only output and no resume.

    AdamW lr=1e-3, weight_decay=0 and clip=1 are a frozen offline geometry
    protocol, not the network's paired optimizer settings. No tuning is done.
    Both modes use the original GS objective; all horizons must have trusted
    render observations even when RGB/alpha are assigned zero training weight.
    """
    if type(updates) is not int or updates < 1 or updates > 400:
        raise ValueError("Oracle budget must be 1..400 successful updates")
    if lr != 1e-3:
        raise ValueError("Frozen Oracle learning rate is 1e-3")
    if batch["inputs"]["gs_xyz"].shape[0] != 1:
        raise ValueError("Oracle requires one frozen training window")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    horizon = len(batch["targets"]["target_xyz"])
    oracle = D0Oracle(batch["inputs"], horizon)
    optimizer = torch.optim.AdamW(oracle.parameters(), lr=lr, weight_decay=0.)
    weights = dict(chamfer_weight=1., rgb_weight=float(full_loss),
                   alpha_weight=.1 if full_loss else 0., scale_weight=.01 if full_loss else 0.)
    config = dict(offline_oracle_only=True, horizon=horizon, updates=updates,
                  optimizer="AdamW", lr=lr, weight_decay=0., clip_norm=1.,
                  anchor_count=128, anchor_pool_k=16, bandwidth_factor=1.,
                  weights=weights, initial_displacements="zero",
                  window={k: v.detach().cpu().tolist() for k, v in batch.get("meta", {}).items()
                          if isinstance(v, torch.Tensor) and k in ("episode_id", "frame_ids")})
    atomic_json(output / "config.json", config)
    initial = copy_initial(batch["inputs"], "gs_geometry")
    centroid = centroid_translation_oracle(batch)
    baselines = dict(copy_initial=oracle_metrics(initial, batch, processor, renderer),
                     centroid_translation=oracle_metrics(centroid, batch, processor, renderer))
    _save_prediction(output / "copy_initial.pt", initial)
    _save_prediction(output / "centroid_translation.pt", centroid)
    field = field_diagnostics(oracle, processor.state_dict()["scale"])
    if field["row_sum_max_error"] > 2e-6 or field["padding_weight_abs_sum"] != 0:
        raise AssertionError("D0 interpolation failed partition-of-unity or mask check")
    milestones = sorted({0, min(100, updates), min(200, updates), updates})
    history, metrics = [], {}
    started = time.monotonic()
    for step in range(updates+1):
        if step in milestones:
            with torch.no_grad():
                prediction = oracle()
                metrics[str(step)] = oracle_metrics(prediction, batch, processor, renderer, oracle)
                _save_prediction(output / f"prediction_{step}.pt", prediction, oracle)
            atomic_json(output / "milestones.json", metrics)
        if step == updates:
            break
        optimizer.zero_grad(set_to_none=True)
        prediction = oracle()
        loss, values = objective(prediction, batch, processor, renderer, 0., 512, **weights)
        if not values["render_time_available"].all():
            raise ValueError("Oracle requires trusted observations at every horizon")
        loss.backward()
        if oracle.delta.grad is None or not torch.isfinite(oracle.delta.grad).all():
            raise FloatingPointError("Oracle displacement gradient is absent or nonfinite")
        if step == 0 and not oracle.delta.grad.abs().any():
            raise RuntimeError("Zero initial Oracle gradient; investigate before optimization")
        raw_norm = torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1., error_if_nonfinite=True)
        clipped_norm = oracle.delta.grad.norm().detach()
        before = oracle.delta.detach().clone()
        optimizer.step()
        if not torch.isfinite(oracle.delta).all():
            raise FloatingPointError("Nonfinite Oracle displacement")
        history.append(dict(update=step+1, loss=float(loss.detach()),
                            gradient_norm=float(raw_norm), clipped_gradient_norm=float(clipped_norm),
                            parameter_update_norm=float((oracle.delta.detach()-before).norm()),
                            **{k: float(values[k].detach().mean()) for k in
                               ("geometry", "rgb", "alpha", "scale_regularization")}))
        if (step+1) % 20 == 0 or step+1 == updates:
            atomic_json(output / "updates.json", history)
        del prediction, loss, values, before
    result = dict(config=config, baselines=baselines, field=field, milestones=metrics,
                  final=metrics[str(updates)], successful_updates=updates,
                  elapsed_seconds=time.monotonic()-started,
                  interpretation="Offline target-optimized capacity; not condition prediction or generalization")
    atomic_json(output / "oracle_results.json", result)
    return result
