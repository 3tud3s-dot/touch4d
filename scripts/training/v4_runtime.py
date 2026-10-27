"""Explicit V4 selection, sparse data contract and isolated checkpoint format.

Legacy train/evaluate CLI defaults and v7 checkpoint readers are unchanged.
V4 callers use this adapter rather than feeding V4 state into PositionPredictor.
"""
from pathlib import Path
import os
import tempfile
import torch

from scripts.dataloader.collate import batch_horizon
from scripts.dataloader.hand_policy import HAND_POLICY, operation_hands
from scripts.dataloader.dataset import Touch4DDataset
from .model import PositionPredictor
from .v4_model import V4Stage1Predictor
from .losses import objective

VERSION = "touch4d-v4-stage1-smoke-v1"
BASE_SHA = "64b1320d3718e7d05d590ce854b8aeb05f14236f"
SPEC_SHA256 = "b3d2a54c46dabe1773d0c8c46964295ad3fa29f2bc6536df49af4eb3f0e5c3f6"


def build_model(architecture="legacy", **config):
    # "legacy" means unchanged PositionPredictor defaults, NOT the formal P2
    # preset. Formal P2 still requires anchors + anchor_attention explicitly.
    if architecture == "legacy":
        return PositionPredictor(**config)
    if architecture == "v4_stage1":
        return V4Stage1Predictor(**config)
    raise ValueError("architecture must be legacy or v4_stage1")


def build_dataset(scene, split, root, *, horizon=10, prediction_interval=1, **kwargs):
    return Touch4DDataset(scene, split, root, horizon=horizon, prediction_interval=prediction_interval,
                          operation_hand_policy=HAND_POLICY, **kwargs)


def prepare_batch(raw, processor, model):
    config = model.config
    if batch_horizon(raw) != config["horizon"]:
        raise ValueError("Batch horizon disagrees with V4 checkpoint")
    meta = raw["meta"]
    if meta.get("prediction_interval") != config["prediction_interval"]:
        raise ValueError("Sparse dataset prediction_interval must be explicit")
    if meta.get("operation_hand_policy") != HAND_POLICY:
        raise ValueError("V4 requires the explicit reviewed data-layer hand policy")
    expected = operation_hands(meta["scene"], meta["episode_id"])
    if not torch.equal(raw["inputs"]["operation_hands"], expected):
        raise ValueError("Hand labels disagree with episode metadata")
    timestamps = meta["observation_timestamp_us"]
    if not torch.all(timestamps[:, 1:] > timestamps[:, :-1]):
        raise ValueError("Observation timestamps must increase")
    batch = processor(raw)
    batch["inputs"]["condition_frame_ids"] = batch["meta"]["frame_ids"][:, :-1].clone()
    return batch


def v4_objective(prediction, batch, processor, renderer, chunk=512):
    return objective(prediction, batch, processor, renderer, chunk=chunk,
                     chamfer_weight=1., rgb_weight=1., alpha_weight=.1,
                     scale_weight=0., depth_weight=0.)


def save_checkpoint(path, model, optimizer, successful_steps, *, source_sha, processor_state,
                    run_state=None, replace_existing=False):
    if type(successful_steps) is not int or successful_steps < 0:
        raise ValueError("successful_steps must be a nonnegative integer")
    payload = dict(version=VERSION, config=model.config, model=model.state_dict(),
                   optimizer=optimizer.state_dict(), successful_steps=successful_steps,
                   source_sha=source_sha, base_sha=BASE_SHA, spec_sha256=SPEC_SHA256,
                   processor=processor_state, rng_state=torch.get_rng_state(), run_state=run_state,
                   cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    path = Path(path)
    if path.exists():
        if not replace_existing:
            raise FileExistsError(path)
        old = torch.load(path, map_location="cpu", weights_only=False)
        if (old.get("source_sha") != source_sha or old.get("config") != model.config
                or run_state is None or (old.get("run_state") or {}).get("run_id") != run_state.get("run_id")):
            raise ValueError("Refusing to overwrite a different run/checkpoint")
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if replace_existing:
            os.replace(temporary, path)
        else:
            # Atomic no-clobber publication: readers never see a partial file.
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, model, optimizer=None, *, processor_state=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("version") != VERSION or state.get("config") != model.config:
        raise ValueError("Incompatible V4 architecture/time/feature/checkpoint configuration")
    if processor_state is not None and state.get("processor") != processor_state:
        raise ValueError("Checkpoint normalization disagrees with processor")
    if type(state.get("successful_steps")) is not int or state["successful_steps"] < 0:
        raise ValueError("Invalid optimizer-step count")
    if state.get("spec_sha256") != SPEC_SHA256 or state.get("base_sha") != BASE_SHA:
        raise ValueError("Unexpected specification or base")
    for value in state["model"].values():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("Nonfinite checkpoint parameters")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_state"])
        if state.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return state
