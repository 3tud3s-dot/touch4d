"""Frozen Direct H15 training settings and initialization, without historical runners."""
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from scripts.dataloader.hand_policy import HAND_POLICY
from .train import GS_CAMERAS, gs_render_settings
from .cloth_model import ClothDynamicsPredictor


def configuration(root, depth_root, sha, variant="direct"):
    if variant != "direct":
        raise ValueError("Only the retained Direct architecture is supported")
    c = dict(source_sha=sha, scene='001-rope', architecture='v4_stage1', protocol='epochs',
                root=str(Path(root).resolve()), depth_root=str(Path(depth_root).resolve()),
                epochs=8, single_epochs=0, horizon=15, prediction_interval=1,
                batch_size=1, seed=42, lr=1e-4, weight_decay=1e-4, workers=0,
                output_mode='gs_geometry', chamfer_weight=1., rgb_weight=1., alpha_weight=.1,
                scale_weight=0., depth_weight=0., chamfer_chunk=512,
                max_train_batches=0, max_val_batches=0, overfit_windows=0,
                depth_cameras=GS_CAMERAS['001-rope'], render_settings=gs_render_settings(),
                operation_hand_policy=HAND_POLICY, best_metric='validation.model.objective')
    c['architecture'] = 'v4_g4_global_motion_v1'
    return dict(c, architecture='v4_g5_signed_local_direct_v1', variant='direct', gradient_clip_norm=1.)


def build_model(variant="direct", *, horizon=15, prediction_interval=1):
    if variant != "direct":
        raise ValueError("Only the retained Direct architecture is supported")
    return ClothDynamicsPredictor(horizon=horizon, prediction_interval=prediction_interval)


def probe_order(raw):
    """Clone real loader settings/generator; includes iterator base-seed consumption."""
    if raw.batch_size!=1 or raw.num_workers!=0 or raw.drop_last or raw.generator is None:
        raise ValueError('DataLoader settings differ from G4')
    if isinstance(raw.sampler,RandomSampler):
        if raw.sampler.replacement or raw.sampler.num_samples!=len(raw.dataset):raise ValueError('Wrong sampler')
        if raw.sampler.generator is not raw.generator:raise ValueError('Sampler uses a different generator')
        shuffle=True
    elif isinstance(raw.sampler,SequentialSampler):shuffle=False
    else:raise ValueError('Unsupported sampler')
    clone=torch.Generator().set_state(raw.generator.get_state().clone())
    proxy=DataLoader(range(len(raw.dataset)),batch_size=1,shuffle=shuffle,num_workers=0,drop_last=False,generator=clone)
    return [int(batch[0]) for batch in proxy]


def fresh_model(variant="direct"):
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    model=build_model(variant)
    if any(p.count_nonzero() for p in model.direct_motion.mapping[-1].parameters()):raise AssertionError('Not fresh')
    return model
