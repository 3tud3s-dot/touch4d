"""Differentiable gsplat rendering using the 001/051 audited depth convention."""

import math
import torch


def assemble_gaussians(prediction, batch, processor, sample, horizon):
    """Remove padding; restore world metres; inherit initial attributes by index."""
    x = batch["inputs"]
    keep = x["gs_mask"][sample]
    xyz = processor.denormalize_xyz(prediction[sample, horizon], keep)[keep]
    attrs = x["gs_attr"][sample, keep].detach()
    scale = processor.state_dict()["scale"]
    return {"means": xyz.contiguous(), "scales": torch.exp(attrs[:, :3] + math.log(scale)).contiguous(),
            "quats": attrs[:, 3:7].contiguous(), "opacities": torch.sigmoid(attrs[:, 7]).contiguous(),
            "colors": (attrs[:, 8:11] * 0.28209479177387814 + .5).clamp(0, 1).contiguous()}


class DepthRenderer:
    """Lazy optional dependency: CPU model/loss tests need neither CUDA nor gsplat."""
    settings = {"render_mode": "ED", "near_plane": .01, "far_plane": 1e10,
                "eps2d": .3, "rasterize_mode": "classic", "packed": True}

    def __init__(self, rasterizer=None):
        self.rasterizer = rasterizer

    def __call__(self, prediction, batch, processor):
        if self.rasterizer is None:
            if not prediction.is_cuda:
                raise RuntimeError("Real depth rendering requires a CUDA device")
            try:
                from gsplat import rasterization
            except ImportError as exc:
                raise RuntimeError("Install the audited gsplat renderer in the server environment first") from exc
            self.rasterizer = rasterization
        y = batch["targets"]
        b, times, views, height, width = y["target_depth"].shape
        if times != 5 or prediction.shape[:2] != (b, 5):
            raise ValueError("Five prediction and target times are required")
        output = []
        for i in range(b):
            frames = []
            for k in range(5):
                gs = assemble_gaussians(prediction, batch, processor, i, k)
                images = []
                for v in range(views):
                    depth, _, _ = self.rasterizer(**gs, Ks=y["camera_intrinsics"][i, v][None],
                        viewmats=torch.linalg.inv(y["camera_to_world"][i, v])[None],
                        width=width, height=height, **self.settings)
                    images.append(depth[0, :, :, 0])
                frames.append(torch.stack(images))
            output.append(torch.stack(frames))
        return torch.stack(output)
