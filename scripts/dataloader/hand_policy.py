"""Explicit reviewed operation labels, never inferred from motion or pressure."""
import torch

HAND_POLICY = "001-ep0-4-right-ep5-9-both-v1"


def operation_hands(scenes, episodes):
    if (episodes.ndim != 1 or episodes.dtype != torch.int64
            or len(scenes) != len(episodes) or any(s != "001-rope" for s in scenes)
            or torch.any((episodes < 0) | (episodes > 9))):
        raise ValueError("Hand policy requires 001-rope episodes 0..9")
    return torch.stack((episodes >= 5, torch.ones_like(episodes, dtype=torch.bool)), -1)
