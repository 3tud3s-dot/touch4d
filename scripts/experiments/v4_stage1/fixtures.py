"""Explicit synthetic CPU smoke data and analytic rasterizer substitute.

These exercise the real model, processor, Chamfer and rendering wrapper, but are
not evidence of real Deform360 reconstruction quality or CUDA/gsplat correctness.
"""
import torch
from scripts.dataloader.collate import collate_fn
from scripts.dataloader.hand_policy import HAND_POLICY, operation_hands


def synthetic_batch(points=144, horizon=10, prediction_interval=1):
    samples = []
    for episode, n in ((2, points), (5, max(1, points-47))):
        u = torch.linspace(-1., 1., n)
        xyz = torch.stack((.1*u, .02*u.square(), .5+.01*torch.sin(u*3)), -1)
        attr = torch.zeros(n, 11)
        attr[:, :3], attr[:, 3], attr[:, 7], attr[:, 8:] = -6., 1., .4, .1
        controller = torch.zeros(horizon, 2, 30, 3)
        for step in range(horizon):
            controller[step, ..., 0] = torch.linspace(-.1, .1, 60).reshape(2, 30)+step*.001
            controller[step, ..., 1] = .02
            controller[step, ..., 2] = .5+step*.001
        action = torch.zeros(horizon, 2, 13)
        action[..., :3] = controller.mean(-2)
        action[..., 3:12] = torch.eye(3).flatten()
        action[..., 12] = .08
        pressure = torch.zeros(horizon, 4, 12, 32)
        for step in range(horizon):
            pressure[step, :, 4:8, 15:22] = 8.+step*2
        valid = torch.ones_like(pressure, dtype=torch.bool)
        valid[..., 31] = False
        tactile_xyz = controller.reshape(horizon, 4, 15, 3).mean(-2)[:, :, None, None].expand(-1, -1, 12, 32, -1).clone()
        inputs = dict(gs_xyz=xyz, gs_attr=attr, action_state=action,
                      action_valid=torch.ones(horizon, 2, dtype=torch.bool), controller_points=controller,
                      controller_mask=torch.ones(horizon, 2, 30, dtype=torch.bool), tactile=pressure,
                      valid_mask=valid, tactile_spatial_mask=valid.clone(), tactile_xyz=tactile_xyz,
                      tactile_age_us=torch.full((horizon, 4), 12000, dtype=torch.int64),
                      operation_hands=operation_hands(["001-rope"], torch.tensor([episode]))[0])
        targets = dict(target_xyz=[], target_attr=[])
        for step in range(1, horizon+1):
            count = max(1, n-step % 3)
            moved = xyz[:count].clone()
            moved[:, 0] += .002*step
            moved[:, 1] += .001*step*(u[:count]+1).square()
            targets["target_xyz"].append(moved)
            targets["target_attr"].append(attr[:count].clone())
        depth = torch.full((horizon, 2, 3, 4), .5)
        depth[..., -1, -1] = 0
        targets.update(target_depth=depth, target_depth_valid=depth > 0,
                       camera_intrinsics=torch.tensor([[100., 0., 1.5], [0., 110., 1.], [0., 0., 1.]]).repeat(2, 1, 1),
                       camera_to_world=torch.eye(4).repeat(2, 1, 1))
        ids = 73+torch.arange(horizon+1)*prediction_interval
        meta = dict(scene="001-rope", episode_id=episode, frame_ids=ids,
                    observation_timestamp_us=1700000000000000+ids*50000,
                    prediction_interval=prediction_interval, operation_hand_policy=HAND_POLICY,
                    depth_camera_ids=["synthetic_a", "synthetic_b"])
        samples.append(dict(inputs=inputs, targets=targets, meta=meta))
    return collate_fn(samples)


def analytic_rasterizer(**args):
    """Differentiable mock used inside the ORIGINAL GaussianRenderer wrapper."""
    xyz = args["means"].mean(0)
    rgb = torch.sigmoid(xyz).expand(1, args["height"], args["width"], 3)
    alpha = torch.sigmoid(xyz.sum()).expand(1, args["height"], args["width"], 1)
    depth = xyz[2].expand_as(alpha)
    return torch.cat((rgb, depth), -1), alpha, {}
