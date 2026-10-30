"""Differentiable WXYZ world-increment rotations and fixed-RBF attributes."""
import math
import torch

LOG_SCALE_BOUND = math.log(1.25)
ROT_COMPONENT_BOUND = (math.pi / 6) / math.sqrt(3)


def rotvec_quaternion(omega):
    """Exp(omega) in WXYZ; torch.sinc supplies the analytic zero limit."""
    theta = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
    return torch.cat((torch.cos(theta / 2),
                      .5 * torch.sinc(theta / (2 * math.pi)) * omega), -1)


def quaternion_multiply(left, right):
    lw, lv = left[..., :1], left[..., 1:]
    rw, rv = right[..., :1], right[..., 1:]
    return torch.cat((lw * rw - (lv * rv).sum(-1, keepdim=True),
                      lw * rv + rw * lv + torch.linalg.cross(lv, rv, dim=-1)), -1)


def rotation_matrix(q):
    """Normalized rotation for diagnostics; not an extra render-path rounding."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def covariance(log_scales, quaternion):
    rotation = rotation_matrix(quaternion)
    return (rotation * torch.exp(2 * log_scales)[..., None, :]) @ rotation.transpose(-1, -2)


def adapt_attributes(raw, motion, field, gs_mask):
    """Interpolate bounded rotation vectors BEFORE Exp, relative to initial attrs.

    Original unit quaternions are not redundantly renormalized: left-multiplying
    the exact identity preserves original stored components and render rounding.
    """
    anchor_mask = field.anchor_mask[:, None, :, None]
    raw = torch.where(anchor_mask, raw, torch.zeros_like(raw))
    log_delta_a = LOG_SCALE_BOUND * raw[..., :3].tanh()
    omega_a = ROT_COMPONENT_BOUND * raw[..., 3:].tanh()
    weights = field.weights.detach()
    log_delta = torch.einsum('bna,bhac->bhnc', weights, log_delta_a)
    omega = torch.einsum('bna,bhac->bhnc', weights, omega_a)
    valid = gs_mask[:, None, :, None]
    log_delta = torch.where(valid, log_delta, torch.zeros_like(log_delta))
    omega = torch.where(valid, omega, torch.zeros_like(omega))
    # Sanitize padding before nonlinear geometry; valid inputs must already be finite.
    q0 = torch.where(valid, motion['quaternions'], torch.zeros_like(motion['quaternions']))
    scale0 = torch.where(valid, motion['log_scales'], torch.zeros_like(motion['log_scales']))
    result = dict(motion)
    result['log_scales'] = torch.where(valid, scale0 + log_delta, torch.zeros_like(scale0))
    result['quaternions'] = torch.where(valid, quaternion_multiply(rotvec_quaternion(omega), q0), torch.zeros_like(q0))
    return result, dict(log_scale_delta=log_delta, rotation_vector=omega,
                        anchor_log_scale_delta=log_delta_a, anchor_rotation_vector=omega_a)
