"""Pure image/world-geometry helpers; no legacy HTML viewer or training CLI."""
import struct
import zlib
import numpy as np
import torch
from .losses import prediction_horizon


def png_bytes(rgb):
    """Small RGB PNG encoder, so visualization needs no extra image packages."""
    height, width, _ = rgb.shape
    def chunk(tag, data):
        return struct.pack('!I', len(data)) + tag + data + struct.pack('!I', zlib.crc32(tag+data) & 0xffffffff)
    scanlines = b''.join(b'\0' + row.tobytes() for row in rgb.astype(np.uint8))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b''))


def ellipsoid_axes(log_scales, quaternions):
    """World-space one-sigma axes; normalize WXYZ exactly as the rasterizer does."""
    q = np.asarray(quaternions, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(norm <= 0):
        raise ValueError('Invalid ellipsoid quaternion')
    w, x, y, z = (q/norm).T
    rotation = np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                         2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                         2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)
    scales = np.exp(np.asarray(log_scales, dtype=np.float64))
    axes = np.swapaxes(rotation*scales[:, None, :], -1, -2)
    if not np.isfinite(axes).all() or np.any(scales <= 0):
        raise ValueError('Invalid ellipsoid axes')
    return axes


def geometry_data(arrays, limit=600):
    """All centers; deterministic template subset for ellipsoids/trajectories only."""
    horizon = len(arrays['predicted_xyz'])
    if horizon < 1 or len(arrays['frame_ids']) != horizon+1:
        raise ValueError('Geometry needs H predictions and H+1 original frame IDs')
    if (len(arrays['predicted_log_scales']) != horizon
            or len(arrays['predicted_quaternions']) != horizon
            or any(f'target_xyz_{h}' not in arrays or f'target_attr_{h}' not in arrays
                   for h in range(1, horizon+1))):
        raise ValueError('Missing Gaussian attributes or targets for a predicted horizon')
    def frame(xyz, attr):
        ids = np.linspace(0, len(xyz)-1, min(limit, len(xyz)), dtype=int)
        return dict(centers=xyz.round(6).tolist(), ids=ids.tolist(),
                    axes=ellipsoid_axes(attr[ids, :3], attr[ids, 3:7]).round(9).tolist())
    initial = frame(arrays['initial_xyz'], arrays['initial_attributes'])
    predicted, targets = [], []
    for h in range(horizon):
        attr = np.concatenate((arrays['predicted_log_scales'][h], arrays['predicted_quaternions'][h]), -1)
        predicted.append(frame(arrays['predicted_xyz'][h], attr))
        targets.append(frame(arrays[f'target_xyz_{h+1}'], arrays[f'target_attr_{h+1}']))
    return dict(initial=initial, prediction=predicted, target=targets,
                frame_ids=arrays['frame_ids'].tolist(), ellipsoid_limit=limit)


@torch.no_grad()
def world_gaussian_arrays(raw, batch, prediction, processor):
    """Extract the original world-space Gaussian arrays without rendering or file writes."""
    if isinstance(prediction, torch.Tensor):
        prediction = {'xyz': prediction}
    horizon = prediction_horizon(prediction['xyz'], batch)
    if raw['meta']['frame_ids'].shape[1] != horizon + 1:
        raise ValueError('Original frame IDs must match prediction H')
    keep = batch['inputs']['gs_mask'][0]
    arrays = {'frame_ids': raw['meta']['frame_ids'][0].numpy(),
              'initial_xyz': raw['inputs']['gs_xyz'][0, keep.cpu()].numpy(),
              'initial_attributes': raw['inputs']['gs_attr'][0, keep.cpu()].numpy()}
    for field, value in prediction.items():
        if field == 'xyz':
            value = processor.denormalize_xyz(value, batch['inputs']['gs_mask'][:, None].expand(-1, horizon, -1))
        elif field == 'log_scales':
            value = value + np.log(processor.state_dict()['scale'])
        arrays['predicted_'+field] = value[0, :, keep].cpu().numpy()
    for k, (xyz, attr, mask) in enumerate(zip(raw['targets']['target_xyz'], raw['targets']['target_attr'], raw['targets']['target_mask'])):
        arrays[f'target_xyz_{k+1}'] = xyz[0, mask[0]].numpy()
        arrays[f'target_attr_{k+1}'] = attr[0, mask[0]].numpy()
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError('Nonfinite GS output; no visualization written')
    return arrays
