"""Unordered point geometry and target-masked expected-depth losses."""

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
            t = y["target_xyz"][k][b, y["target_mask"][k][b]]
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


def objective(prediction, batch, processor, renderer, depth_weight=1., chunk=512):
    if not math.isfinite(depth_weight) or depth_weight < 0:
        raise ValueError("depth_weight must be finite and nonnegative")
    real = batch["inputs"]["gs_mask"][:, None].expand(-1, 5, -1)
    if not torch.isfinite(prediction[real]).all():
        raise FloatingPointError("Nonfinite predicted position; refusing to call the rasterizer")
    geometry = chamfer_samples(prediction, batch, chunk)
    rendered = renderer(prediction, batch, processor)
    y = batch["targets"]
    depth, available = depth_samples(rendered, y["target_depth"], y["target_depth_valid"])
    scale = processor.state_dict()["scale"]
    # Empty depth samples do not dilute the depth component of a mixed batch.
    depth_mean = depth[available].mean() if available.any() else rendered.sum() * 0.
    loss = geometry.mean() + depth_weight * depth_mean / scale
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training objective")
    return loss, {"geometry": geometry, "depth_m": depth, "depth_available": available}
