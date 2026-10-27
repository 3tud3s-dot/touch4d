"""Frozen causal A/B construction from ONE real GS and robot initial state."""
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch

from scripts.dataloader.dataset import _episode_info, _episode_conditions, read_gs, GS_ATTR_FIELDS
from scripts.dataloader.geometry import tactile_world_coordinates
from scripts.dataloader.collate import collate_fn, fixed_fields
from scripts.dataloader.hand_policy import HAND_POLICY
from scripts.data_process import DataProcessor
from scripts.training.v4_runtime import prepare_batch
from scripts.experiments.fixed_window.fixed_fit import digest

H = 10
AMPLITUDE_M = .010
SIGMA_M = .040


def load_initial(root):
    """No future GS file is opened; reuse validated dataset readers/geometry."""
    info = _episode_info(Path(root)/'001-rope/episode_2', '001-rope', 2)
    c = _episode_conditions(info)
    take = slice(73, 74)
    x = {k: torch.from_numpy(c[k][take].copy()) for k in
         ('action_state', 'action_valid', 'controller_points', 'controller_mask', 'tactile', 'valid_mask')}
    xyz, spatial = tactile_world_coordinates(c['T_worlds'][take], c['openings'][take], info.slots,
                                             c['action_valid'][take], c['valid_mask'][take])
    x.update(tactile_xyz=torch.from_numpy(xyz), tactile_spatial_mask=torch.from_numpy(spatial),
             tactile_age_us=torch.from_numpy(info.ages[take].copy()))
    initial = read_gs(info, 73)
    x.update(gs_xyz=torch.from_numpy(initial['xyz'].copy()),
             gs_attr=torch.from_numpy(np.concatenate([initial[k] for k in GS_ATTR_FIELDS], -1)))
    return x, dict(root=str(Path(root).resolve()), scene='001-rope', episode=2, initial_frame=73,
                   initial_gs_sha256=digest({k:torch.as_tensor(v) for k,v in initial.items()}), initial_conditions_sha256=digest(x),
                   timestamps_us=info.timestamps[73:84].tolist(), future_real_gs_read=False)


def make_pair(initial, timestamps, source):
    if len(timestamps) != H+1 or not initial['action_valid'][0, 1] or not initial['controller_mask'][0, 1].all():
        raise ValueError('Needs valid right robot initial state and eleven timestamps')
    xyz = initial['gs_xyz']
    grip = initial['controller_points'][0, 1].mean(0)
    center_index = int((xyz-grip).square().sum(-1).argmin())
    center = xyz[center_index]
    weights = torch.exp(-(xyz-center).square().sum(-1)/(2*SIGMA_M**2))
    affected = weights >= .9  # Fixed diagnostic ROI; not a contact label or loss mask.
    amplitude = torch.linspace(0., AMPLITUDE_M, H)
    samples = []
    for sign in (1., -1.):
        x = {k: initial[k][:1].expand(H, *initial[k].shape[1:]).clone() for k in fixed_fields(H)}
        x.update(gs_xyz=xyz.clone(), gs_attr=initial['gs_attr'].clone(),
                 operation_hands=torch.tensor([False, True]))
        # Entire left hand absent. Active taxels are valid zero pressure, not missing.
        x['action_valid'][:, 0] = False; x['controller_mask'][:, 0] = False
        x['action_state'][:, 0] = 0; x['controller_points'][:, 0] = 0
        x['tactile'].zero_(); x['tactile_spatial_mask'][:, :2] = False
        x['tactile_spatial_mask'][..., 31] = False
        x['valid_mask'] = x['tactile_spatial_mask'].clone()
        x['tactile_age_us'].fill_(12000); x['tactile_age_us'][:, :2] = -1
        x['tactile_xyz'][:, :2] = 0
        x['action_state'][:, 1, 2] += sign*amplitude
        x['controller_points'][:, 1, :, 2] += sign*amplitude[:, None]
        targets = []
        for amount in amplitude:
            target = xyz.clone(); target[:, 2] += sign*amount*weights
            targets.append(target)
        samples.append(dict(inputs=x, targets=dict(target_xyz=targets,
                            target_attr=[initial['gs_attr'].clone() for _ in range(H)]),
                            meta=dict(scene='001-rope', episode_id=2, frame_ids=torch.arange(73, 84),
                                      observation_timestamp_us=torch.tensor(timestamps, dtype=torch.int64),
                                      prediction_interval=1, operation_hand_policy=HAND_POLICY)))
    raw = collate_fn(samples)
    changed = [k for k,v in raw['inputs'].items() if isinstance(v,torch.Tensor) and not torch.equal(v[0],v[1])]
    if set(changed) != {'action_state','controller_points'}:
        raise AssertionError(f'Unexpected branch identity channel: {changed}')
    assert torch.equal(raw['targets']['target_xyz'][0][0], raw['targets']['target_xyz'][0][1])
    manifest = dict(source=source, horizon=H, prediction_interval=1, initial_frame=73,
                    condition_frames=list(range(73,83)), target_frames=list(range(74,84)),
                    synthetic_targets=True, rule='target(h) = X0 + sign * (h-1)/9 * .010m * exp(-||X0-center||^2/(2*.040m^2)) * world_Z',
                    center_rule='initial GS center nearest to mean of right-hand initial controller points',
                    center_index=center_index, center_m=center.tolist(), sigma_m=SIGMA_M,
                    action_z_m=amplitude.tolist(), affected_region='weight >= 0.9, diagnostic only',
                    affected_count=int(affected.sum()), target_affected_z_last_mm=10*float(weights[affected].mean()),
                    identical_input_fields=[k for k in raw['inputs'] if k not in changed], differing_input_fields=changed,
                    tactile='identical valid zero pressure, 12000 us age; unused tactile_xyz frozen at initial state',
                    causality='h1 target identical; h>=2 follows last arrived action at t+h-1, not future action',
                    data_sha256=digest(raw), weights_sha256=hashlib.sha256(weights.numpy().tobytes()).hexdigest())
    return raw, weights, affected, manifest


def normalize(raw, model):
    p = DataProcessor('001-rope')
    return prepare_batch(raw, p, model), p


def action_variant(batch, mode):
    result = copy.deepcopy(batch)
    x = result['inputs']
    if mode == 'real': return result
    for key in ('action_state', 'controller_points'):
        value = x[key]
        if mode == 'swap': x[key] = value.flip(0)
        elif mode == 'static': x[key] = value[:, :1].expand_as(value).clone()
        elif mode == 'half': x[key] = value[:, :1] + .5*(value-value[:, :1])
        else: raise ValueError(mode)
    return result
