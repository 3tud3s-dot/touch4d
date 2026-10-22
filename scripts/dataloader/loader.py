"""Build one scene's DataLoader; curriculum switching belongs to the caller."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .collate import collate_fn
from .dataset import DEFAULT_ROOT, Touch4DDataset


def _seed_worker(worker_id):
    # Top-level function so Windows spawn can pickle it. Torch seeds itself.
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_dataloader(scene: str, split: str, *, root=DEFAULT_ROOT, stage="all",
                    batch_size=1, num_workers=0, seed=0, pin_memory=False,
                    persistent_workers=False, cache_episodes=2) -> DataLoader:
    """Train: seeded shuffle, stride 1. Val/test: fixed order, stride 5.

    All splits retain the final partial batch. To advance curriculum, construct
    a new training loader with stage='all'; val/test always use stage='all'.
    Seed fixes a newly constructed loader's sampling sequence, not every epoch.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    dataset = Touch4DDataset(scene, split, root, stage=stage, cache_episodes=cache_episodes)
    if not len(dataset):
        raise ValueError(f"{scene}/{split}/{stage}: no complete six-frame windows")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                      num_workers=num_workers, collate_fn=collate_fn, drop_last=False,
                      pin_memory=pin_memory, persistent_workers=persistent_workers,
                      generator=generator, worker_init_fn=_seed_worker)
