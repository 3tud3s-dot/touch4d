"""Same chunked nearest neighbours and 1/2 convention as production CD."""
import torch
from scripts.training.losses import nearest_distances, prediction_horizon


def squared_chamfer_samples(prediction, batch, chunk=512):
    horizon = prediction_horizon(prediction, batch)
    x, y = batch['inputs'], batch['targets']
    samples = []
    for b in range(len(prediction)):
        frames = []
        for k in range(horizon):
            p = prediction[b, k, x['gs_mask'][b]]
            t = y['target_xyz'][k][b, y['target_mask'][k][b]].detach()
            if not torch.isfinite(p).all() or not torch.isfinite(t).all():
                raise FloatingPointError('Nonfinite valid squared-Chamfer points')
            # Square EACH matched distance, not the reduced Chamfer scalar.
            frames.append(.5 * (nearest_distances(p,t,chunk).square().mean()
                                 + nearest_distances(t,p,chunk).square().mean()))
        samples.append(torch.stack(frames).mean())
    return torch.stack(samples)
