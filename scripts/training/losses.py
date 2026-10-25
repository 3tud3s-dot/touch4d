"""Legacy geometry/depth losses and target-masked Gaussian RGB/alpha losses."""

import math
import torch


def nearest_distances(query, reference, chunk=512):
    """Exact nearest neighbours in bounded blocks; gradients through matched XYZ.

    Neighbour search itself is discrete. Keeping its distance matrices out of the
    graph avoids retaining quadratic memory for every frame of a five-frame loss.
    """
    if not len(query) or not len(reference) or chunk < 1:
        raise ValueError("Nonempty point sets and a positive chunk size are required")
    indices = []
    with torch.no_grad():
        for q in query.split(chunk):
            best = torch.full((len(q),), float("inf"), device=q.device)
            index = torch.zeros(len(q), dtype=torch.long, device=q.device)
            for offset in range(0, len(reference), chunk):
                # The default MM identity loses precision for close neighbours:
                # even identical dense sets can select a different point. Use
                # coordinate differences, independently of the block size.
                distances = torch.cdist(q, reference[offset:offset+chunk],
                                        compute_mode="donot_use_mm_for_euclid_dist")
                value, candidate = distances.min(dim=1)
                better = value < best
                index = torch.where(better, candidate + offset, index)
                best = torch.minimum(best, value)
            indices.append(index)
    nearest = reference[torch.cat(indices)]
    return torch.linalg.vector_norm(query - nearest, dim=-1)


def chamfer_samples(prediction, batch, chunk=512):
    """[B] equal averages across five times and the two unsquared L2 directions."""
    x, y = batch["inputs"], batch["targets"]
    expected = (x["gs_xyz"].shape[0], 5, x["gs_xyz"].shape[1], 3)
    if tuple(prediction.shape) != expected:
        raise ValueError(f"Expected prediction shape {expected}")
    losses = []
    for b in range(len(prediction)):
        frames = []
        for k in range(5):
            p = prediction[b, k, x["gs_mask"][b]]
            t = y["target_xyz"][k][b, y["target_mask"][k][b]].detach()
            frames.append((nearest_distances(p, t, chunk).mean() + nearest_distances(t, p, chunk).mean()) * .5)
        losses.append(torch.stack(frames).mean())
    return torch.stack(losses)


def depth_samples(prediction_m, target_m, valid):
    """Return [B] metre MAE and [B] availability, equal valid-view/time averaging.

    Only target validity chooses pixels. In particular an absent render (zero)
    remains an error wherever the target is valid. All-empty views/times do not
    contribute a zero score; an all-empty sample is marked unavailable.
    """
    if (prediction_m.shape != target_m.shape or valid.shape != target_m.shape
            or target_m.ndim != 5 or target_m.shape[1] != 5 or valid.dtype != torch.bool):
        raise ValueError("Expected matching [B,5,V,H,W] depth and boolean target mask")
    if not torch.isfinite(prediction_m).all():
        raise FloatingPointError("Nonfinite rendered depth")
    delta = torch.where(valid, prediction_m - target_m, torch.zeros_like(prediction_m)).abs()
    counts = valid.sum((-1, -2))
    views = counts > 0
    view_loss = delta.sum((-1, -2)) / counts.clamp_min(1)
    time_loss = (view_loss * views).sum(-1) / views.sum(-1).clamp_min(1)
    times = views.any(-1)
    loss = (time_loss * times).sum(-1) / times.sum(-1).clamp_min(1)
    return loss, times.any(-1)


def render_samples(prediction, target, valid):
    """L1 with equal pixel, view, time means; RGB channels are averaged first.

    Return sample loss, sample availability, per-time loss and time availability.
    The target mask alone chooses observations, including absent predictions.
    """
    if valid.dtype != torch.bool or valid.ndim != 5 or valid.shape[1] != 5:
        raise ValueError("Expected boolean [B,5,V,H,W] target render mask")
    if prediction.shape != target.shape or prediction.shape[:5] != valid.shape:
        raise ValueError("Rendered predictions, targets and target mask must match")
    if prediction.ndim not in (5, 6) or (prediction.ndim == 6 and prediction.shape[-1] != 3):
        raise ValueError("Rendered values must be alpha [B,5,V,H,W] or RGB with three channels")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Nonfinite rendered prediction or target")
    delta = (prediction - target).abs()
    if delta.ndim == 6:
        delta = delta.mean(-1)
    delta = torch.where(valid, delta, torch.zeros_like(delta))
    counts = valid.sum((-1, -2))
    views = counts > 0
    view_loss = delta.sum((-1, -2)) / counts.clamp_min(1)
    time_loss = (view_loss * views).sum(-1) / views.sum(-1).clamp_min(1)
    times = views.any(-1)
    samples = (time_loss * times).sum(-1) / times.sum(-1).clamp_min(1)
    return samples, times.any(-1), time_loss, times


def gs_objective(prediction, batch, processor, renderer, depth_weight=0., chunk=512,
                 rgb_weight=1., alpha_weight=.1, scale_weight=.01, chamfer_weight=0.):
    """RGB/alpha supervision with optional normalized, unsquared Chamfer.

    All-empty samples are excluded from every training component, including the
    scale regularizer and Chamfer. The training loop must skip an entirely empty batch, so
    AdamW decay/moments cannot move parameters without an observed objective.
    """
    if depth_weight != 0:
        raise ValueError("gs_geometry keeps depth diagnostic only; depth_weight must be zero")
    for name, weight in (("rgb_weight", rgb_weight), ("alpha_weight", alpha_weight),
                         ("scale_weight", scale_weight), ("chamfer_weight", chamfer_weight)):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    x, y = batch["inputs"], batch["targets"]
    real = x["gs_mask"][:, None].expand(-1, 5, -1)
    dimensions = {"xyz": 3, "log_scales": 3, "quaternions": 4, "opacity_logits": 1, "sh_dc": 3}
    for key, size in dimensions.items():
        if key not in prediction or tuple(prediction[key].shape) != (*real.shape, size):
            raise ValueError(f"Expected predicted {key} shape {(*real.shape, size)}")
        if not torch.isfinite(prediction[key][real]).all():
            raise FloatingPointError(f"Nonfinite predicted {key}; refusing to call the rasterizer")
    valid = y["target_depth_valid"]
    if (valid.dtype != torch.bool or valid.shape != y["target_depth"].shape
            or valid.ndim != 5 or valid.shape[1] != 5):
        raise ValueError("Expected matching depth and boolean [B,5,V,H,W] target mask")
    if not torch.isfinite(y["target_depth"]).all():
        raise FloatingPointError("Nonfinite target depth")
    # Never backpropagate through target GS or observations, even when supplied
    # by a custom renderer in a test or evaluation tool.
    with torch.no_grad():
        target = renderer.render_targets(batch, processor)
    rendered = renderer(prediction, batch, processor)
    rgb, available, rgb_time, times = render_samples(rendered["rgb"], target["rgb"].detach(), valid)
    alpha, alpha_available, alpha_time, alpha_times = render_samples(
        rendered["alpha"], target["alpha"].detach(), valid)
    if not torch.equal(available, alpha_available) or not torch.equal(times, alpha_times):
        raise RuntimeError("RGB and alpha must use exactly the same target observation mask")
    difference = torch.where(real.unsqueeze(-1),
                             prediction["log_scales"] - x["gs_attr"][:, None, :, :3],
                             torch.zeros_like(prediction["log_scales"]))
    scale_regularization = difference.square().sum((1, 2, 3)) / (real.sum((1, 2)) * 3).clamp_min(1)
    with torch.set_grad_enabled(torch.is_grad_enabled() and chamfer_weight > 0):
        geometry = chamfer_samples(prediction["xyz"], batch, chunk)
    data_score = rgb_weight * rgb + alpha_weight * alpha
    # Preserve the original zero-weight objective and its gradient graph exactly.
    if chamfer_weight > 0:
        data_score = data_score + chamfer_weight * geometry
    sample_loss = data_score + scale_weight * scale_regularization
    if available.any():
        loss = sample_loss[available].mean()
    else:
        loss = sum(prediction[key][real].sum() for key in ("xyz", "log_scales", "quaternions")) * 0.
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite Gaussian rendering objective")
    with torch.no_grad():
        depth, depth_available = depth_samples(rendered["depth"].detach(),
                                               y["target_depth"], valid)
    return loss, {"geometry": geometry.detach(), "depth_m": depth, "depth_available": depth_available,
                  "rgb": rgb, "alpha": alpha, "render_available": available,
                  "scale_regularization": scale_regularization, "data_score": data_score,
                  "rgb_by_horizon": rgb_time, "alpha_by_horizon": alpha_time,
                  "render_time_available": times}


def objective(prediction, batch, processor, renderer, depth_weight=0., chunk=512,
              rgb_weight=1., alpha_weight=.1, scale_weight=.01, chamfer_weight=0.):
    if isinstance(prediction, dict):
        return gs_objective(prediction, batch, processor, renderer, depth_weight, chunk,
                            rgb_weight, alpha_weight, scale_weight, chamfer_weight)
    if chamfer_weight != 0:
        raise ValueError("chamfer_weight is GS-only; position mode retains unit-weight Chamfer")
    if not math.isfinite(depth_weight) or depth_weight < 0:
        raise ValueError("depth_weight must be finite and nonnegative")
    real = batch["inputs"]["gs_mask"][:, None].expand(-1, 5, -1)
    if not torch.isfinite(prediction[real]).all():
        raise FloatingPointError("Nonfinite predicted position; refusing to call the rasterizer")
    geometry = chamfer_samples(prediction, batch, chunk)
    y = batch["targets"]
    # With zero weight, depth remains a reporting metric, with no rendering
    # graph retained and no depth contribution to the optimization objective.
    with torch.set_grad_enabled(torch.is_grad_enabled() and depth_weight > 0):
        rendered = renderer(prediction, batch, processor)
        depth, available = depth_samples(rendered, y["target_depth"], y["target_depth_valid"])
    scale = processor.state_dict()["scale"]
    # Empty depth samples do not dilute the depth component of a mixed batch.
    depth_mean = depth[available].mean() if available.any() else rendered.sum() * 0.
    loss = geometry.mean() if depth_weight == 0 else geometry.mean() + depth_weight * depth_mean / scale
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training objective")
    return loss, {"geometry": geometry, "depth_m": depth, "depth_available": available}
