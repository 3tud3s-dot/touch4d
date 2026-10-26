"""Paired condition-only alternatives to P2. Coordinates for kernels are metres."""
import torch
from torch import nn
from scripts.training.model import PositionPredictor, masked
from scripts.rebuild_dataset.rebuild import GRID_ROWS, GRID_COLS

MODES = ('B0', 'A1', 'T1', 'T2', 'T3')
HAND_POLICY = '001-ep0-4-right-ep5-9-both-v1'


def hand_inputs(inputs, episodes, scenes):
    """Out-of-place gate, from declared episode protocol, never observed motion."""
    if any(s != '001-rope' for s in scenes) or ((episodes < 0) | (episodes > 9)).any():
        raise ValueError('Hand policy only defined for 001-rope episodes 0..9')
    x = dict(inputs)
    hands = torch.stack((episodes >= 5, torch.ones_like(episodes, dtype=torch.bool)), -1)
    x['operation_hands'] = hands.to(inputs['gs_xyz'].device)
    return x


def sanitize(x):
    """Finite masks precede MLP/arithmetic. Return new tensors, never edit data."""
    x = dict(x)
    hand = x['operation_hands']
    if hand.dtype != torch.bool or hand.shape != (x['gs_xyz'].shape[0], 2):
        raise ValueError('Explicit [B,2] operation_hands required')
    h = hand[:, None]
    x['action_valid'] = x['action_valid'] & h & torch.isfinite(x['action_state']).all(-1)
    x['action_state'] = masked(x['action_state'], x['action_valid'])
    x['controller_mask'] = (x['controller_mask'] & h[..., None]
                            & torch.isfinite(x['controller_points']).all(-1))
    x['controller_points'] = masked(x['controller_points'], x['controller_mask'])
    f = h.repeat_interleave(2, -1)
    age = x['tactile_age_us']
    age_valid = torch.isfinite(age) & (age >= 0) & (age <= 75000)
    x['valid_mask'] = (x['valid_mask'] & f[..., None, None] & age_valid[..., None, None]
                       & torch.isfinite(x['tactile']))
    x['tactile'] = torch.where(x['valid_mask'], x['tactile'], 0.)
    x['tactile_age_us'] = torch.where(x['valid_mask'].flatten(-2).any(-1), age, 0)
    x['tactile_spatial_mask'] = x['tactile_spatial_mask'] & x['valid_mask']
    return x


def local_geometry():
    r, c = torch.meshgrid(torch.arange(12,dtype=torch.float64), torch.arange(32,dtype=torch.float64), indexing='ij')
    pixels = torch.stack((torch.full_like(r, .007, dtype=torch.float64),
                          -.056-.002*c, .0235-.002*r), -1).double().reshape(384, 3)
    r, c = torch.meshgrid(torch.tensor(GRID_ROWS,dtype=torch.float64), torch.tensor(GRID_COLS,dtype=torch.float64), indexing='ij')
    points = torch.stack((torch.full_like(r, .007, dtype=torch.float64),
                          -.056-.002*c, (.025/12)*(11.5-r)), -1).double().reshape(15,3)
    distances = (points[:,None]-pixels[None]).square().sum(-1)
    kernel = torch.exp(-distances/(2*.01**2)).float()
    cells = distances.argmin(0)
    roi = torch.arange(384) % 32 != 31
    partition = (torch.arange(15)[:,None] == cells) & roi
    return kernel, partition, roi


class ConditionPredictor(PositionPredictor):
    """Research-only config/checkpoint namespace; original P2 stays untouched."""
    def __init__(self, condition_mode='B0', hand_policy=HAND_POLICY, **kwargs):
        if condition_mode not in MODES or hand_policy != HAND_POLICY:
            raise ValueError('Unsupported condition mode or hand policy')
        super().__init__(**kwargs)
        if self.config['controller_features'] != 7:
            raise ValueError('Condition comparison requires controller 7D')
        self.config.update(condition_mode=condition_mode, hand_policy=hand_policy)
        width = self.config['width']
        def mlp(n):
            return nn.Sequential(nn.Linear(n,width), nn.GELU(), nn.Linear(width,width))
        # All shared P2 modules are constructed first, preserving their RNG stream.
        if condition_mode == 'T1':
            replacement = mlp(769)
            # Preserve columns retained from the original 1153D MLP, including age.
            with torch.no_grad():
                replacement[0].weight.copy_(self.tactile_encoder[0].weight[:, list(range(768))+[1152]])
                replacement[0].bias.copy_(self.tactile_encoder[0].bias)
                replacement[2].load_state_dict(self.tactile_encoder[2].state_dict())
            self.tactile_encoder = replacement
        if condition_mode in ('T2','T3'):
            del self.tactile_encoder
            self.fallback_encoder = mlp(769)
            self.pressure_encoder = mlp(2)
            self.fusion = nn.Sequential(nn.Linear(2*width+4,width), nn.GELU(), nn.Linear(width,width))
            nn.init.zeros_(self.fusion[-1].weight)
            nn.init.zeros_(self.fusion[-1].bias)
            k, p, roi = local_geometry()
            self.register_buffer('kernel', k)
            self.register_buffer('partition', p)
            self.register_buffer('roi', roi)

    def controller_inputs(self, x):
        if self.config['condition_mode'] != 'A1':
            return super().controller_inputs(x)
        valid = x['controller_mask']
        xyz = masked(x['controller_points'], valid)
        previous = torch.cat((torch.zeros_like(xyz[:,:1]), xyz[:,:-1]), 1)
        endpoints = valid & torch.cat((torch.zeros_like(valid[:,:1]), valid[:,:-1]), 1)
        return torch.cat((xyz, masked(xyz-previous,endpoints), endpoints[...,None].to(xyz.dtype)), -1)

    def aggregate(self, x):
        """Return mean/max, coverage, age and separate observation masks [B,H,4,15]."""
        pressure = x['tactile'].flatten(-2)
        valid = x['valid_mask'].flatten(-2) & x['tactile_spatial_mask'].flatten(-2) & self.roi
        pressure = torch.where(valid, pressure, 0.)
        if self.config['condition_mode'] == 'T2':
            count = valid.sum(-1, keepdim=True)
            obs = count > 0
            mean = pressure.sum(-1,keepdim=True)/count.clamp_min(1)
            maximum = pressure.masked_fill(~valid, -torch.inf).amax(-1,keepdim=True)
            maximum = torch.where(obs, maximum, 0.)
            coverage = count.to(pressure.dtype)/self.roi.sum()
            mean, maximum, coverage, mean_obs, max_obs = [v.expand(*v.shape[:-1],15)
                for v in (mean,maximum,coverage,obs,obs)]
        else:
            weights = valid[...,None,:]*self.kernel
            denominator = weights.sum(-1)
            mean_obs = denominator > 0
            mean = (weights*pressure[...,None,:]).sum(-1)/denominator.clamp_min(torch.finfo(pressure.dtype).tiny)
            local = valid[...,None,:] & self.partition
            max_obs = local.any(-1)
            maximum = pressure[...,None,:].expand_as(local).masked_fill(~local,-torch.inf).amax(-1)
            maximum = torch.where(max_obs,maximum,0.)
            coverage = local.sum(-1).to(pressure.dtype)/self.partition.sum(-1).clamp_min(1)
        age = (x['tactile_age_us'].to(pressure.dtype)/75000.)[...,None].expand_as(mean)
        age = torch.where(mean_obs,age,0.)
        return mean, maximum, coverage, age, mean_obs, max_obs

    def encode_conditions(self, x):
        x = sanitize(x)
        mode = self.config['condition_mode']
        if mode in ('B0','A1'):
            return super().encode_conditions(x)
        b, h = x['action_state'].shape[:2]
        if h != self.config['horizon']:
            raise ValueError('Condition horizon mismatch')
        state = self.state_encoder(x['action_state'])
        controllers = self.controller_encoder(self.controller_inputs(x)).reshape(b,h,4,15,-1)
        cp_keep = x['controller_mask'].reshape(b,h,4,15)
        valid = x['valid_mask']
        fingers_valid = valid.flatten(-2).any(-1)
        age = x['tactile_age_us'].to(x['tactile'].dtype)/75000.
        flat = torch.cat((x['tactile'].flatten(-2), valid.flatten(-2).to(x['tactile'].dtype),age[...,None]),-1)
        if mode == 'T1':
            fingers = self.tactile_encoder(flat)
        else:
            mean, maximum, coverage, age, mean_obs, max_obs = self.aggregate(x)
            pressure = self.pressure_encoder(torch.stack((mean,maximum),-1))
            quality = torch.stack((coverage,age,mean_obs.to(mean.dtype),max_obs.to(mean.dtype)),-1)
            gate = cp_keep & mean_obs
            update = self.fusion(torch.cat((controllers,pressure,quality),-1))
            controllers = controllers + masked(update,gate)
            fingers = self.fallback_encoder(flat)
            # Finger fallback only when no valid spatial tool token can receive it.
            fingers_valid = fingers_valid & ~gate.any(-1)
        tokens = torch.cat((state,controllers.reshape(b,h,60,-1),fingers),2)
        keep = torch.cat((x['action_valid'],cp_keep.reshape(b,h,60),fingers_valid),2)
        tokens = tokens + self.token_identity.weight[None,None] + self.condition_time.weight[None,:,None]
        return masked(tokens,keep), keep
