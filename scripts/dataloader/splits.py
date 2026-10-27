"""Reviewed episode splits; curriculum changes only the training subset."""

HORIZON = 5
STRIDES = {"train": 1, "val": 5, "test": 5}
SPLITS = {
    "001-rope": {"train": (1, 2, 4, 5, 7, 8), "val": (3, 6), "test": (0, 9)},
    "008-pink-cloth": {"train": (0, 2, 4, 5, 7, 9), "val": (1, 6), "test": (3, 8)},
    "051-cube": {"train": (0, 2, 3, 5, 7, 9), "val": (4, 8), "test": (1, 6)},
}
# These are reviewed curriculum groups, never a source of physical hand identity.
SINGLE_HAND_TRAIN = {
    "001-rope": (1, 2, 4),
    "008-pink-cloth": (0, 2, 4),
    "051-cube": (0, 2, 3),
}


def episode_ids(scene: str, split: str, stage: str = "all") -> tuple[int, ...]:
    """Stage 'single' starts curriculum; 'all' retains singles and adds doubles."""
    if scene not in SPLITS:
        raise ValueError(f"Unknown scene {scene!r}; choose one of {tuple(SPLITS)}")
    if split not in STRIDES:
        raise ValueError(f"Unknown split {split!r}; choose train, val or test")
    if stage not in ("single", "all"):
        raise ValueError("stage must be 'single' or 'all'")
    if split != "train" and stage != "all":
        raise ValueError("Curriculum applies only to train; val/test require stage='all'")
    return SINGLE_HAND_TRAIN[scene] if stage == "single" else SPLITS[scene][split]


def validate_horizon(horizon: int) -> int:
    """Keep the legacy default, allowing any positive prediction length."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    return horizon


def validate_prediction_interval(interval: int) -> int:
    if type(interval) is not int or interval < 1:
        raise ValueError("prediction_interval must be a positive integer")
    return interval


def window_starts(frames: int, split: str, horizon: int = HORIZON,
                  prediction_interval: int = 1) -> range:
    """H+1 sampled frames; sampling interval is independent of window stride."""
    validate_horizon(horizon)
    validate_prediction_interval(prediction_interval)
    if split not in STRIDES or frames < 0:
        raise ValueError("Expected a known split and nonnegative frame count")
    return range(0, max(0, frames - horizon * prediction_interval), STRIDES[split])
