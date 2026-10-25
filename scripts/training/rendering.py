"""Differentiable GS RGB/alpha and audited camera-Z expected-depth rendering."""

import math
import torch


def _target_layout(batch):
    """Cross-check target sequence length against calibrated image tensors."""
    y = batch["targets"]
    depth = y["target_depth"]
    if depth.ndim != 5:
        raise ValueError("Target depth must have shape [B,H,V,height,width]")
    b, times, views, height, width = depth.shape
    if times < 1 or views < 1 or height < 1 or width < 1:
        raise ValueError("Positive target horizon, camera count and image dimensions are required")
    if len(y.get("target_xyz", [])) != times or len(y.get("target_mask", [])) != times:
        raise ValueError("Target GS/mask lists and image horizon must agree")
    for xyz, keep in zip(y["target_xyz"], y["target_mask"]):
        if (xyz.ndim != 3 or xyz.shape[0] != b or xyz.shape[-1] != 3
                or keep.shape != xyz.shape[:2] or keep.dtype != torch.bool):
            raise ValueError("Invalid independently padded target GS/mask shape")
    if (y["camera_intrinsics"].shape != (b, views, 3, 3)
            or y["camera_to_world"].shape != (b, views, 4, 4)):
        raise ValueError("Camera calibration must match the target batch and views")
    return b, times, views, height, width


def _gaussians(xyz, attrs, processor, keep):
    """Convert normalized selected GS to gsplat's world-metre representation."""
    if not keep.any():
        raise ValueError("A rendered Gaussian set must contain at least one valid point")
    if not torch.isfinite(xyz[keep]).all() or not torch.isfinite(attrs[keep]).all():
        raise FloatingPointError("Nonfinite valid Gaussian parameters; refusing to call the rasterizer")
    xyz = processor.denormalize_xyz(xyz, keep)[keep]
    attrs = attrs[keep]
    scales = torch.exp(attrs[:, :3] + math.log(processor.state_dict()["scale"]))
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise FloatingPointError("Gaussian scales must remain finite and strictly positive")
    if not (torch.linalg.vector_norm(attrs[:, 3:7], dim=-1) > 0).all():
        raise ValueError("Valid Gaussian quaternions cannot be zero")
    return {"means": xyz.contiguous(), "scales": scales.contiguous(),
            "quats": attrs[:, 3:7].contiguous(), "opacities": torch.sigmoid(attrs[:, 7]).contiguous(),
            "colors": (attrs[:, 8:11] * 0.28209479177387814 + .5).clamp(0, 1).contiguous()}


def assemble_gaussians(prediction, batch, processor, sample, horizon):
    """Remove padding and restore metres; dict predictions retain geometry grads.

    The legacy tensor path continues inheriting detached initial attributes.
    Both paths use zero-order SH appearance and the same DC-to-RGB conversion.
    """
    x = batch["inputs"]
    keep = x["gs_mask"][sample]
    if isinstance(prediction, dict):
        xyz = prediction["xyz"][sample, horizon]
        attrs = torch.cat((prediction["log_scales"][sample, horizon],
                           prediction["quaternions"][sample, horizon],
                           prediction["opacity_logits"][sample, horizon],
                           prediction["sh_dc"][sample, horizon]), dim=-1)
        return _gaussians(xyz, attrs, processor, keep)
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
        positions = prediction["xyz"] if isinstance(prediction, dict) else prediction
        b, times, views, height, width = _target_layout(batch)
        if positions.shape != (b, times, batch["inputs"]["gs_xyz"].shape[1], 3):
            raise ValueError("Prediction shape must match target H and initial Gaussian count")
        if self.rasterizer is None:
            if not positions.is_cuda:
                raise RuntimeError("Real depth rendering requires a CUDA device")
            try:
                from gsplat import rasterization
            except ImportError as exc:
                raise RuntimeError("Install the audited gsplat renderer in the server environment first") from exc
            self.rasterizer = rasterization
        y = batch["targets"]
        output = []
        for i in range(b):
            frames = []
            for k in range(times):
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


class GaussianRenderer:
    """Same calibrated camera settings for prediction, target and copy baseline.

    RGB is the black-background composited image, not divided by alpha. ED is
    expected camera-Z depth in metres. Cameras and pixel masks come from the
    target depth batch; no prediction-dependent validity threshold is applied.
    """
    settings = dict(DepthRenderer.settings, render_mode="RGB+ED")

    def __init__(self, rasterizer=None, image_size=None):
        self.rasterizer = rasterizer
        self.image_size = image_size

    def _ensure_rasterizer(self, positions):
        if self.rasterizer is None:
            if not positions.is_cuda:
                raise RuntimeError("Real Gaussian rendering requires a CUDA device")
            try:
                from gsplat import rasterization
            except ImportError as exc:
                raise RuntimeError("Install the audited gsplat renderer in the server environment first") from exc
            self.rasterizer = rasterization

    def _render(self, batch, assemble):
        y = batch["targets"]
        b, times, views, height, width = _target_layout(batch)
        if self.image_size is not None and (height, width) != tuple(self.image_size):
            raise ValueError("Target depth resolution disagrees with checkpoint render settings")
        rgb_batch, alpha_batch, depth_batch = [], [], []
        for i in range(b):
            rgb_frames, alpha_frames, depth_frames = [], [], []
            for k in range(times):
                gs = assemble(i, k)
                rgb_views, alpha_views, depth_views = [], [], []
                for v in range(views):
                    image, alpha, _ = self.rasterizer(
                        **gs, Ks=y["camera_intrinsics"][i, v][None],
                        viewmats=torch.linalg.inv(y["camera_to_world"][i, v])[None],
                        width=width, height=height,
                        backgrounds=gs["means"].new_zeros((1, 3)), **self.settings)
                    if image.shape != (1, height, width, 4) or alpha.shape != (1, height, width, 1):
                        raise ValueError("RGB+ED must return four image channels and one alpha channel")
                    rgb_views.append(image[0, ..., :3])
                    depth_views.append(image[0, ..., 3])
                    alpha_views.append(alpha[0, ..., 0])
                rgb_frames.append(torch.stack(rgb_views))
                depth_frames.append(torch.stack(depth_views))
                alpha_frames.append(torch.stack(alpha_views))
            rgb_batch.append(torch.stack(rgb_frames))
            alpha_batch.append(torch.stack(alpha_frames))
            depth_batch.append(torch.stack(depth_frames))
        return {"rgb": torch.stack(rgb_batch), "alpha": torch.stack(alpha_batch),
                "depth": torch.stack(depth_batch)}

    def __call__(self, prediction, batch, processor):
        positions = prediction["xyz"] if isinstance(prediction, dict) else prediction
        b, times, _, _, _ = _target_layout(batch)
        if positions.shape != (b, times, batch["inputs"]["gs_xyz"].shape[1], 3):
            raise ValueError("Prediction shape must match target H and initial Gaussian count")
        self._ensure_rasterizer(positions)
        return self._render(batch, lambda i, k: assemble_gaussians(prediction, batch, processor, i, k))

    @torch.no_grad()
    def render_targets(self, batch, processor):
        y = batch["targets"]
        _, times, _, _, _ = _target_layout(batch)
        if "target_attr" not in y or len(y["target_attr"]) != times:
            raise ValueError("Full target_attr for all H target GS frames is required")
        if any(attr.shape != (*xyz.shape[:2], 11) for xyz, attr in zip(y["target_xyz"], y["target_attr"])):
            raise ValueError("Target attributes must match each target Gaussian set")
        self._ensure_rasterizer(y["target_xyz"][0])
        return self._render(batch, lambda i, k: _gaussians(
            y["target_xyz"][k][i], y["target_attr"][k][i], processor, y["target_mask"][k][i]))
