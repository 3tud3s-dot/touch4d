"""Train one scene; python -B -m scripts.training.train --help.

Run only in the server's GPU allocation. No downloads or dependency installs.
For a tiny fixed-window overfit use --overfit-windows 2 --single-epochs 0.
Resume is at completed epoch boundaries; repeat the original training options,
adding --resume .../last.pt and optionally increasing --epochs.
"""

import argparse
from datetime import datetime, timezone
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import subprocess
import uuid

import torch
from torch.utils.data import DataLoader, Subset

from ..data_process import DataProcessor, SUPPORTED_SCENES
from ..dataloader.collate import collate_fn
from ..dataloader.dataset import DEFAULT_ROOT
from ..dataloader.loader import make_dataloader
from ..dataloader.splits import SPLITS
from .losses import objective
from .model import PositionPredictor
from .rendering import DepthRenderer, GaussianRenderer

VERSION = "touch4d-gs-v4"
GS_JOINT_VERSION = "touch4d-gs-v3"
GS_LOCAL_VERSION = "touch4d-gs-v2"
GS_LEGACY_VERSION = "touch4d-gs-v1"
DISPLACEMENT_VERSION = "touch4d-displacement-v3"
LEGACY_VERSION = "touch4d-displacement-v1"
INDEPENDENT_VERSION = "touch4d-displacement-v2"
MODEL_KEYS = ("width", "heads", "layers", "controller_features", "prediction_mode", "output_mode",
              "gs_local_attention", "gs_local_k", "gs_decoder", "anchor_count", "anchor_pool_k",
              "anchor_bandwidth")
DECODER_DEFAULTS = dict(gs_decoder="free", anchor_count=128, anchor_pool_k=16, anchor_bandwidth=1.)
PROJECT = Path(__file__).resolve().parents[2]
GS_CAMERAS = {
    "001-rope": ["brics-odroid-001_cam0", "brics-odroid-011_cam0", "brics-odroid-021_cam0"],
    "051-cube": ["brics-odroid-001_cam0", "brics-odroid-012_cam0", "brics-odroid-021_cam1"],
}
# Canonical defaults for existing callers/checkpoints. Fresh CLI GS runs resolve
# chamfer_weight separately to .1; missing legacy metadata must remain zero.
LOSS_DEFAULTS = dict(rgb_weight=1., alpha_weight=.1, scale_weight=.01, chamfer_weight=0.)
GS_CONVENTION = "world-left-wxyz-sh0-black-depth-valid-v1"


def gs_render_settings():
    return dict(GaussianRenderer.settings, width=1280, height=720, sh_degree=0,
                background=[0., 0., 0.], camera_model="pinhole", depth="camera_z_expected_metres",
                supervision_mask="target_depth_valid")


def loss_options(config):
    return {key: config.get(key, value) for key, value in LOSS_DEFAULTS.items()}


def make_renderer(config):
    if config.get("output_mode", "position") != "gs_geometry":
        return DepthRenderer()
    settings = config.get("render_settings")
    size = (settings["height"], settings["width"]) if settings is not None else None
    return GaussianRenderer(image_size=size)


def copy_initial(inputs, output_mode="position"):
    expand = lambda value: value[:, None].expand(-1, 5, -1, -1)
    if output_mode == "position":
        return expand(inputs["gs_xyz"])
    attr = inputs["gs_attr"]
    return dict(xyz=expand(inputs["gs_xyz"]), log_scales=expand(attr[..., :3]),
                quaternions=expand(attr[..., 3:7]), opacity_logits=expand(attr[..., 7:8]),
                sh_dc=expand(attr[..., 8:11]))


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def choose_cameras(scene, depth_root, requested=None):
    # Camera availability only; neither target geometry nor test metrics is used.
    common = None
    for episode in sorted({e for group in SPLITS[scene].values() for e in group}):
        meta = json.loads((Path(depth_root) / scene / f"episode_{episode}" / "metadata.json").read_text())
        cameras = set(meta["cameras"])
        common = cameras if common is None else common & cameras
    chosen = list(requested) if requested else sorted(common)[:2]
    if not chosen or len(chosen) != len(set(chosen)) or not set(chosen) <= common:
        raise ValueError("Camera subset must be nonempty, unique and available across the scene")
    return chosen


def build_loader(config, split, epoch=0, stage="all"):
    loader = make_dataloader(config["scene"], split, root=config["root"], stage=stage,
        batch_size=config["batch_size"], num_workers=config["workers"], seed=config["seed"]+epoch,
        with_depth=True, depth_root=config["depth_root"], depth_cameras=config["depth_cameras"])
    if split == "train" and config["overfit_windows"]:
        n = config["overfit_windows"]
        if n > len(loader.dataset):
            raise ValueError("--overfit-windows exceeds the number of training windows")
        loader = DataLoader(Subset(loader.dataset, range(n)), batch_size=config["batch_size"],
                            shuffle=False, num_workers=0, collate_fn=collate_fn)
    return loader


class EpisodeMetrics:
    """First average samples within episodes, then average episodes equally."""
    def __init__(self, scale, weight):
        self.scale, self.weight, self.episodes = scale, weight, {}

    def add(self, ids, metrics):
        for i, episode in enumerate(ids.tolist()):
            row = self.episodes.setdefault(str(episode), dict(samples=0, geometry=0., depth_m=0., depth_samples=0))
            row["samples"] += 1
            row["geometry"] += float(metrics["geometry"][i].detach())
            if bool(metrics["depth_available"][i]):
                row["depth_samples"] += 1
                row["depth_m"] += float(metrics["depth_m"][i].detach())

    def summary(self):
        if not self.episodes:
            raise ValueError("No samples evaluated")
        per = {}
        for episode, row in sorted(self.episodes.items(), key=lambda kv: int(kv[0])):
            geometry = row["geometry"] / row["samples"]
            depth = row["depth_m"] / row["depth_samples"] if row["depth_samples"] else None
            per[episode] = dict(samples=row["samples"], geometry=geometry,
                                geometry_mm=geometry*self.scale*1000,
                                depth_mae_mm=None if depth is None else depth*1000,
                                depth_samples=row["depth_samples"])
        geometry = sum(v["geometry"] for v in per.values()) / len(per)
        ds = [v["depth_mae_mm"] for v in per.values() if v["depth_mae_mm"] is not None]
        depth_mm = sum(ds)/len(ds) if ds else None
        return dict(episodes=per, geometry=geometry, geometry_mm=geometry*self.scale*1000,
                    depth_mae_mm=depth_mm, depth_available_episodes=len(ds),
                    objective=geometry + (self.weight*depth_mm/(1000*self.scale) if ds else 0))


class GaussianMetrics(EpisodeMetrics):
    """Joint best score on render-available samples; all-sample CD stays diagnostic."""
    def __init__(self, scale, config):
        super().__init__(scale, 0.)
        self.weights = loss_options(config)
        self.render = {}

    def add(self, ids, metrics):
        super().add(ids, metrics)
        for i, episode in enumerate(ids.tolist()):
            row = self.render.setdefault(str(episode), dict(samples=0, rgb=0., alpha=0.,
                scale_regularization=0., supervised_geometry=0., time_counts=[0]*5, rgb_by_horizon=[0.]*5,
                alpha_by_horizon=[0.]*5))
            if not bool(metrics["render_available"][i]):
                continue
            row["samples"] += 1
            row["supervised_geometry"] += float(metrics["geometry"][i].detach())
            for key in ("rgb", "alpha", "scale_regularization"):
                row[key] += float(metrics[key][i].detach())
            for k in range(5):
                if bool(metrics["render_time_available"][i, k]):
                    row["time_counts"][k] += 1
                    for key in ("rgb_by_horizon", "alpha_by_horizon"):
                        row[key][k] += float(metrics[key][i, k].detach())

    def summary(self):
        result = super().summary()
        missing = [ep for ep, row in self.render.items() if not row["samples"]]
        if missing:
            raise ValueError(f"No render supervision for entire episodes: {missing}")
        for ep, row in self.render.items():
            result["episodes"][ep].update(render_samples=row["samples"],
                **{key: row[key]/row["samples"] for key in ("rgb", "alpha", "scale_regularization", "supervised_geometry")})
            for key in ("rgb_by_horizon", "alpha_by_horizon"):
                result["episodes"][ep][key] = [v/n if n else None for v, n in zip(row[key], row["time_counts"])]
        for key in ("rgb", "alpha", "scale_regularization", "supervised_geometry"):
            result[key] = sum(ep[key] for ep in result["episodes"].values())/len(self.render)
        for key in ("rgb_by_horizon", "alpha_by_horizon"):
            result[key] = []
            for k in range(5):
                values = [ep[key][k] for ep in result["episodes"].values() if ep[key][k] is not None]
                result[key].append(sum(values)/len(values) if values else None)
        score = self.weights["rgb_weight"]*result["rgb"] + self.weights["alpha_weight"]*result["alpha"]
        if self.weights["chamfer_weight"] > 0:
            score += self.weights["chamfer_weight"]*result["supervised_geometry"]
        result.update(objective=score, val_score=score,
            training_objective=score+self.weights["scale_weight"]*result["scale_regularization"])
        return result


def metric_tracker(processor, config):
    scale = processor.state_dict()["scale"]
    return (GaussianMetrics(scale, config) if config.get("output_mode", "position") == "gs_geometry"
            else EpisodeMetrics(scale, config["depth_weight"]))


@torch.no_grad()
def evaluate_loader(model, loader, processor, renderer, config, device, max_batches=0):
    was_training = model.training
    model.eval()
    trackers = {name: metric_tracker(processor, config)
                for name in ("model", "copy_initial")}
    try:
        for raw in loader if not max_batches else islice(loader, max_batches):
            batch = processor(to_device(raw, device))
            outputs = {"model": model(batch["inputs"]),
                       "copy_initial": copy_initial(batch["inputs"], config.get("output_mode", "position"))}
            for name, prediction in outputs.items():
                _, metrics = objective(prediction, batch, processor, renderer,
                                       config["depth_weight"], config["chamfer_chunk"], **loss_options(config))
                trackers[name].add(batch["meta"]["episode_id"], metrics)
        return {name: tracker.summary() for name, tracker in trackers.items()}
    finally:
        model.train(was_training)


def train_epoch(model, optimizer, loader, processor, renderer, config, device):
    model.train()
    metrics = metric_tracker(processor, config)
    stream = loader if not config["max_train_batches"] else islice(loader, config["max_train_batches"])
    for raw in stream:
        batch = processor(to_device(raw, device))
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["inputs"])
        loss, values = objective(prediction, batch, processor, renderer,
                                 config["depth_weight"], config["chamfer_chunk"], **loss_options(config))
        metrics.add(batch["meta"]["episode_id"], values)
        if "render_available" in values and not bool(values["render_available"].any()):
            # Even an AdamW zero-gradient step would change weights/optimizer state.
            continue
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient norm")
        optimizer.step()
    return metrics.summary()


def checkpoint_state(model, optimizer, processor, config, epoch, best, history):
    config = dict(config)
    for key in MODEL_KEYS:
        if key in config and config[key] != model.config[key]:
            raise ValueError(f"Checkpoint config disagrees with model: {key}")
        config[key] = model.config[key]
    for key, value in LOSS_DEFAULTS.items():
        config.setdefault(key, value)
    convention = GS_CONVENTION if config["output_mode"] == "gs_geometry" else None
    if config.get("gs_convention", convention) != convention:
        raise ValueError("Checkpoint GS convention disagrees with output mode")
    config["gs_convention"] = convention
    settings = gs_render_settings() if config["output_mode"] == "gs_geometry" else None
    if config.get("render_settings", settings) != settings:
        raise ValueError("Checkpoint render settings disagree with output mode")
    config["render_settings"] = settings
    return dict(version=VERSION, config=config, processor=processor.state_dict(),
                model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                best=best, history=history, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                python_rng=random.getstate())


def checkpoint_config(value):
    """Interpret legacy metadata without changing the checkpoint on disk."""
    version = value.get("version")
    if version not in (LEGACY_VERSION, INDEPENDENT_VERSION, DISPLACEMENT_VERSION,
                       GS_LEGACY_VERSION, GS_LOCAL_VERSION, GS_JOINT_VERSION, VERSION):
        raise ValueError("Unsupported checkpoint: expected displacement-v1/v2/v3 or gs-v1/v2/v3/v4 weights; "
                         "absolute-position V0 checkpoints require the original V0 code")
    config = dict(value["config"])
    if version in (LEGACY_VERSION, INDEPENDENT_VERSION, DISPLACEMENT_VERSION):
        if config.get("output_mode", "position") != "position":
            raise ValueError("Legacy checkpoint must use position output")
        config["output_mode"] = "position"
    if config.get("output_mode") not in ("position", "gs_geometry"):
        raise ValueError("Checkpoint requires explicit output_mode")
    convention = GS_CONVENTION if config["output_mode"] == "gs_geometry" else None
    if config.get("gs_convention", None) != convention:
        raise ValueError("Checkpoint has missing or unsupported GS convention")
    config["gs_convention"] = convention
    settings = gs_render_settings() if config["output_mode"] == "gs_geometry" else None
    if config.get("render_settings") != settings:
        raise ValueError("Checkpoint has missing or unsupported render settings")
    config["render_settings"] = settings
    if version not in (GS_JOINT_VERSION, VERSION):
        if config.get("chamfer_weight", 0.) != 0:
            raise ValueError("Legacy checkpoint must use zero GS chamfer_weight")
        config["chamfer_weight"] = 0.
    for key, default in LOSS_DEFAULTS.items():
        if config["output_mode"] == "gs_geometry" and key not in config:
            raise ValueError(f"GS checkpoint requires {key}")
        config.setdefault(key, default)
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"Invalid checkpoint loss weight {key}")
    if config["output_mode"] == "position" and config["chamfer_weight"] != 0:
        raise ValueError("Position checkpoint must use zero GS chamfer_weight")
    if version == LEGACY_VERSION:
        if config.get("controller_features", 3) != 3:
            raise ValueError("Legacy checkpoint must use 3 controller features")
        config["controller_features"] = 3
    if version in (LEGACY_VERSION, INDEPENDENT_VERSION):
        if config.get("prediction_mode", "independent") != "independent":
            raise ValueError("Legacy checkpoint must use independent prediction")
        config["prediction_mode"] = "independent"
    if config.get("prediction_mode") not in ("independent", "autoregressive"):
        raise ValueError("Checkpoint requires explicit prediction_mode: independent or autoregressive")
    if config["output_mode"] == "gs_geometry" and (config["prediction_mode"] != "autoregressive"
            or config.get("depth_weight", 0.) != 0 or config["rgb_weight"] <= 0):
        raise ValueError("GS geometry requires autoregressive, positive RGB weight and diagnostic-only depth")
    features = config.get("controller_features")
    if type(features) is not int or features not in (3, 7):
        raise ValueError("Checkpoint requires explicit controller_features of 3 or 7")
    weight = value["model"].get("controller_encoder.0.weight")
    if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (config["width"], features):
        raise ValueError("Checkpoint controller weights disagree with feature configuration")
    if version not in (GS_LOCAL_VERSION, GS_JOINT_VERSION, VERSION):
        if config.get("gs_local_attention", "none") != "none" or config.get("gs_local_k", 16) != 16:
            raise ValueError("Legacy checkpoint must use disabled local GS attention")
        config.update(gs_local_attention="none", gs_local_k=16)
    if config.get("gs_local_attention") not in ("none", "knn"):
        raise ValueError("Checkpoint requires explicit gs_local_attention: none or knn")
    if type(config.get("gs_local_k")) is not int or config["gs_local_k"] < 1:
        raise ValueError("Checkpoint requires positive integer gs_local_k")
    local_weights = any(key.startswith("gs_local.") for key in value["model"])
    if local_weights != (config["gs_local_attention"] == "knn"):
        raise ValueError("Checkpoint local GS attention weights disagree with configuration")
    if version != VERSION:
        for key, default in DECODER_DEFAULTS.items():
            if config.get(key, default) != default:
                raise ValueError("Legacy checkpoint must use the free GS decoder and default anchor metadata")
            config[key] = default
    elif any(key not in config for key in DECODER_DEFAULTS):
        raise ValueError("Checkpoint requires explicit GS decoder and anchor settings")
    validate_decoder_config(config)
    if config["output_mode"] == "gs_geometry":
        free_heads = [any(key.startswith(prefix) for key in value["model"])
                      for prefix in ("scale_head.", "rotation_head.")]
        if any(present != (config["gs_decoder"] == "free") for present in free_heads):
            raise ValueError("Checkpoint GS shape heads disagree with decoder configuration")
    return config


def validate_decoder_config(config):
    if config.get("gs_decoder") not in ("free", "anchors"):
        raise ValueError("gs_decoder must be free or anchors")
    for key in ("anchor_count", "anchor_pool_k"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    bandwidth = config.get("anchor_bandwidth")
    if type(bandwidth) not in (int, float) or not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("anchor_bandwidth must be finite and positive")
    if config["gs_decoder"] == "anchors" and (config.get("output_mode") != "gs_geometry"
            or config.get("prediction_mode") != "autoregressive" or config.get("gs_local_attention") != "knn"):
        raise ValueError("Anchor decoder requires autoregressive GS geometry and local kNN features")


def resolve_chamfer_weight(requested, output_mode, state=None):
    """Fresh GS runs use .1; resume inherits its objective without changing best."""
    if requested is not None and (not math.isfinite(requested) or requested < 0):
        raise ValueError("chamfer_weight must be finite and nonnegative")
    if state is not None:
        saved = checkpoint_config(state)["chamfer_weight"]
        if requested is not None and requested != saved:
            raise ValueError("Cannot resume with different chamfer_weight; start a new run")
        return saved
    weight = (.1 if output_mode == "gs_geometry" else 0.) if requested is None else requested
    if output_mode != "gs_geometry" and weight != 0:
        raise ValueError("chamfer_weight is GS-only; position mode retains unit-weight Chamfer")
    return weight


def resolve_controller_features(requested, state=None):
    """Fresh runs default to 7; resume inherits and cannot change architecture."""
    saved = checkpoint_config(state)["controller_features"] if state is not None else None
    if requested is not None and (type(requested) is not int or requested not in (3, 7)):
        raise ValueError("controller_features must be 3 or 7")
    if saved is not None and requested is not None and requested != saved:
        raise ValueError("Cannot resume with different controller_features; start a new run")
    return saved if saved is not None else (7 if requested is None else requested)


def resolve_prediction_mode(requested, state=None):
    """Identical tensor shapes do not imply identical rollout semantics."""
    saved = checkpoint_config(state)["prediction_mode"] if state is not None else None
    if requested is not None and requested not in ("independent", "autoregressive"):
        raise ValueError("prediction_mode must be independent or autoregressive")
    if saved is not None and requested is not None and requested != saved:
        raise ValueError("Cannot resume with different prediction_mode; start a new run")
    return saved if saved is not None else ("autoregressive" if requested is None else requested)


def resolve_output_mode(requested, state=None):
    saved = checkpoint_config(state)["output_mode"] if state is not None else None
    if requested is not None and requested not in ("position", "gs_geometry"):
        raise ValueError("output_mode must be position or gs_geometry")
    if saved is not None and requested is not None and requested != saved:
        raise ValueError("Cannot resume with different output_mode; start a new run")
    return saved if saved is not None else ("gs_geometry" if requested is None else requested)


def resolve_local_attention(requested, requested_k, state=None):
    """Fresh runs use kNN/16; resume inherits both fields without changing architecture."""
    if requested is not None and requested not in ("none", "knn"):
        raise ValueError("gs_local_attention must be none or knn")
    if requested_k is not None and (type(requested_k) is not int or requested_k < 1):
        raise ValueError("gs_local_k must be a positive integer")
    if state is not None:
        config = checkpoint_config(state)
        saved, saved_k = config["gs_local_attention"], config["gs_local_k"]
        if ((requested is not None and requested != saved)
                or (requested_k is not None and requested_k != saved_k)):
            raise ValueError("Cannot resume with different local GS attention settings; start a new run")
        return saved, saved_k
    return ("knn" if requested is None else requested), (16 if requested_k is None else requested_k)


def resolve_decoder(requested, count, pool_k, bandwidth, state=None):
    """Defaults preserve free-head runs; resume cannot change even shape-free settings."""
    requested = dict(gs_decoder=requested, anchor_count=count, anchor_pool_k=pool_k,
                     anchor_bandwidth=bandwidth)
    saved = checkpoint_config(state) if state is not None else DECODER_DEFAULTS
    resolved = {}
    for key, value in requested.items():
        if state is not None and value is not None and value != saved[key]:
            raise ValueError(f"Cannot resume with different {key}; start a new run")
        resolved[key] = saved[key] if value is None else value
    # Validate scalars here; dependent mode requirements are checked after all
    # architecture flags have been resolved from a trusted checkpoint.
    validate_decoder_config(dict(resolved, output_mode="gs_geometry", prediction_mode="autoregressive",
                                 gs_local_attention="knn"))
    return resolved


def read_checkpoint(path):
    # Only load checkpoints produced by this project from a trusted location.
    value = torch.load(path, map_location="cpu", weights_only=False)
    value = dict(value, config=checkpoint_config(value))
    DataProcessor.from_state_dict(value["processor"])
    if value["config"]["scene"] != value["processor"]["scene"]:
        raise ValueError("Checkpoint scene and normalization disagree")
    return value


def recover_best_score(last_state, run):
    """Reconcile an already-published best.pt with the resumed last.pt.

    best.pt is published first, so a crash can leave it newer/better than last.
    Keep resuming model/optimizer/RNG from last, but never forget the better
    saved score. Missing, incompatible or regressed best files fail closed.
    This only reads files; checkpoint semantics are checked by read_checkpoint.
    """
    path = Path(run) / "best.pt"
    if not path.is_file():
        raise ValueError(f"Cannot safely resume: missing best checkpoint {path}")
    saved = read_checkpoint(path)
    # Canonical legacy defaults allow old best and newly saved last to coexist.
    comparable = lambda c: {k: v for k, v in {"controller_features": 3,
                            "prediction_mode": "independent", "output_mode": "position",
                            "gs_local_attention": "none", "gs_local_k": 16,
                            "gs_convention": None, "render_settings": None, **LOSS_DEFAULTS,
                            **DECODER_DEFAULTS, **c}.items()
                            if k != "epochs"}
    if (comparable(saved["config"]) != comparable(last_state["config"])
            or saved["processor"] != last_state["processor"]):
        raise ValueError("Cannot safely resume: best.pt and last.pt configurations differ")
    recorded, actual = float(last_state["best"]), float(saved["best"])
    if not all(math.isfinite(v) and v >= 0 for v in (recorded, actual)):
        raise ValueError("Cannot safely resume: invalid best objective")
    if actual > recorded:
        raise ValueError("Cannot safely resume: best.pt is worse than the best recorded in last.pt")
    if actual < recorded:
        print(f"Recovered best objective {actual} from best.pt; last.pt recorded {recorded}. "
              "Model, optimizer and RNG still resume from last.pt.", flush=True)
    return actual


def publish_checkpoints(run, state, improved):
    """Publish best first; recover_best_score handles interruption before last."""
    if improved:
        atomic_checkpoint(Path(run) / "best.pt", state)
    atomic_checkpoint(Path(run) / "last.pt", state)


def restore_training(state, model, optimizer):
    config = checkpoint_config(state)
    if any(config[k] != model.config[k] for k in MODEL_KEYS):
        raise ValueError("Cannot resume: model architecture differs from checkpoint")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same number of visible GPUs")
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", choices=("epochs", "fixed400"), default="epochs",
                   help="fixed400: Roadmap P0 exact windows, alternating updates and fit selection")
    p.add_argument("--fit-contract", help="Reviewed P0 JSON tolerances; required for a new fixed400 run")
    p.add_argument("--fit-diagnostic", action="store_true",
                   help="fixed400 baseline without acceptance thresholds; retain all evaluation checkpoints")
    p.add_argument("--scene", required=True, choices=SUPPORTED_SCENES)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--depth-root")
    p.add_argument("--depth-cameras", nargs="+")
    p.add_argument("--run-id", help="New run directory name; default UTC timestamp plus unique suffix")
    p.add_argument("--resume", help="Trusted last.pt; repeat original training options")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--single-epochs", type=int, default=1, help="Initial single-hand epochs; 0 disables this stage")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--depth-weight", type=float, default=0.,
                   help="Position mode: optional depth loss; GS mode: keep 0, depth is diagnostic only")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--controller-features", type=int, choices=(3, 7), default=None,
                   help="Fresh runs default to 7; resume inherits checkpoint (legacy: 3)")
    p.add_argument("--prediction-mode", choices=("independent", "autoregressive"), default=None,
                   help="Fresh runs default to autoregressive; resume inherits checkpoint")
    p.add_argument("--output-mode", choices=("position", "gs_geometry"), default=None,
                   help="Fresh runs default to gs_geometry; legacy resume keeps position output")
    p.add_argument("--gs-local-attention", choices=("none", "knn"), default=None,
                   help="Fresh runs default to knn; resume inherits checkpoint (legacy: none)")
    p.add_argument("--gs-local-k", type=int, default=None,
                   help="Initial GS neighbors including self; fresh default 16; supports e.g. 32")
    p.add_argument("--gs-decoder", choices=("free", "anchors"), default=None,
                   help="Fresh default free; anchors uses P1a shared displacement and inherited GS shape")
    p.add_argument("--anchor-count", type=int, default=None,
                   help="P1a initial FPS anchor cap; fresh default 128; resume inherits")
    p.add_argument("--anchor-pool-k", type=int, default=None,
                   help="P1a local features pooled per anchor; fresh default 16; resume inherits")
    p.add_argument("--anchor-bandwidth", type=float, default=None,
                   help="Positive P1a RBF bandwidth factor; fresh default 1; resume inherits")
    for name, default in LOSS_DEFAULTS.items():
        if name == "chamfer_weight":
            p.add_argument("--chamfer-weight", type=float, default=None,
                           help="GS only: fresh default .1; resume inherits checkpoint (legacy: 0)")
        else:
            p.add_argument("--"+name.replace("_", "-"), type=float, default=default)
    p.add_argument("--chamfer-chunk", type=int, default=512)
    p.add_argument("--max-train-batches", type=int, default=0, help="0 means full epoch")
    p.add_argument("--max-val-batches", type=int, default=0, help="0 means full validation")
    p.add_argument("--overfit-windows", type=int, default=0, help="Fixed first N training windows; requires --single-epochs 0")
    p.add_argument("--device", default="cuda")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.protocol == "fixed400":
        from .fixed_fit import run
        return run(args)
    if args.fit_contract or args.fit_diagnostic:
        p.error("Fit options require --protocol fixed400")
    if (min(args.epochs, args.batch_size, args.chamfer_chunk) < 1 or args.single_epochs < 0
            or args.single_epochs > args.epochs or min(args.workers, args.seed, args.max_train_batches,
                args.max_val_batches, args.overfit_windows) < 0):
        p.error("Invalid epoch, batch, worker, seed or sampling limits")
    if args.overfit_windows and args.single_epochs:
        p.error("Fixed-window overfit requires --single-epochs 0")
    if args.resume and args.run_id:
        p.error("--resume uses the existing run directory; do not also specify --run-id")
    if any(not math.isfinite(v) or v < 0 for v in (args.lr, args.weight_decay, args.depth_weight,
            args.rgb_weight, args.alpha_weight, args.scale_weight)) or args.lr == 0:
        p.error("Learning rate must be positive; weights must be finite and nonnegative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        p.error("Real training requires a server CUDA allocation; CPU unit tests are separate")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    state = None
    if args.resume:
        if Path(args.resume).name != "last.pt":
            p.error("Resume from last.pt to preserve the completed-epoch history; best.pt is for evaluation")
        state = read_checkpoint(args.resume)
        if state["config"].get("protocol") == "fixed400":
            p.error("A fixed400 checkpoint must resume with --protocol fixed400")
    args.controller_features = resolve_controller_features(args.controller_features, state)
    args.prediction_mode = resolve_prediction_mode(args.prediction_mode, state)
    args.output_mode = resolve_output_mode(args.output_mode, state)
    args.chamfer_weight = resolve_chamfer_weight(args.chamfer_weight, args.output_mode, state)
    args.gs_local_attention, args.gs_local_k = resolve_local_attention(
        args.gs_local_attention, args.gs_local_k, state)
    for key, value in resolve_decoder(args.gs_decoder, args.anchor_count, args.anchor_pool_k,
                                     args.anchor_bandwidth, state).items():
        setattr(args, key, value)
    validate_decoder_config(vars(args))
    if args.output_mode == "gs_geometry" and (args.prediction_mode != "autoregressive"
            or args.depth_weight != 0 or args.rgb_weight <= 0):
        p.error("GS geometry requires autoregressive, positive RGB weight and --depth-weight 0")
    root = Path(args.root).resolve()
    depth_root = Path(args.depth_root).resolve() if args.depth_root else root.parent / "touch4d-depth-v1"
    config = vars(args).copy()
    for key in ("run_id", "resume", "device", "protocol", "fit_contract", "fit_diagnostic"):
        config.pop(key)
    requested_cameras = args.depth_cameras
    if requested_cameras is None:
        requested_cameras = (state["config"]["depth_cameras"] if state is not None else
                             GS_CAMERAS[args.scene] if args.output_mode == "gs_geometry" else None)
    config.update(root=str(root), depth_root=str(depth_root),
                  depth_cameras=choose_cameras(args.scene, depth_root, requested_cameras),
                  gs_convention=GS_CONVENTION if args.output_mode == "gs_geometry" else None,
                  render_settings=gs_render_settings() if args.output_mode == "gs_geometry" else None)
    processor = DataProcessor(args.scene)
    model = PositionPredictor(args.width, args.heads, args.layers,
                              controller_features=args.controller_features,
                              prediction_mode=args.prediction_mode, output_mode=args.output_mode,
                              gs_local_attention=args.gs_local_attention, gs_local_k=args.gs_local_k,
                              **{key: getattr(args, key) for key in DECODER_DEFAULTS}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    renderer = make_renderer(config)
    # Fail before creating a run when the renderer dependency is absent.
    import gsplat
    start, best, history = 0, float("inf"), []
    if args.resume:
        changes = [k for k in config if k != "epochs" and config[k] != state["config"].get(k)]
        if changes:
            raise ValueError(f"Resume options differ from checkpoint: {changes}")
        start, best, history = state["epoch"]+1, state["best"], state["history"]
        if start >= args.epochs:
            raise ValueError("--epochs must exceed the number of completed epochs")
        run = Path(args.resume).resolve().parent
        best = recover_best_score(state, run)
        restore_training(state, model, optimizer)
    else:
        run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
            p.error("--run-id must contain only ASCII letters, digits, hyphens and underscores")
        run = PROJECT / "runs" / args.scene / run_id
        print(f"Creating {run}: config.json, metrics.json, best.pt, last.pt (atomic .tmp staging)", flush=True)
        run.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {run}", flush=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    atomic_json(run / "config.json", dict(training=config, processor=processor.state_dict(),
                renderer=renderer.settings, torch=torch.__version__, gsplat=gsplat.__version__, commit=commit))
    val = build_loader(config, "val")
    for epoch in range(start, args.epochs):
        stage = "single" if epoch < args.single_epochs else "all"
        train = build_loader(config, "train", epoch=epoch, stage=stage)
        training = train_epoch(model, optimizer, train, processor, renderer, config, device)
        validation = evaluate_loader(model, val, processor, renderer, config, device, args.max_val_batches)
        row = dict(epoch=epoch, stage=stage, training=training, validation=validation,
                   partial_train=bool(args.max_train_batches or args.overfit_windows),
                   partial_validation=bool(args.max_val_batches))
        history.append(row)
        score = validation["model"]["objective"]
        improved = score < best
        best = min(best, score)
        state = checkpoint_state(model, optimizer, processor, config, epoch, best, history)
        # Keep the previous completed-epoch last.pt intact until all work succeeds.
        publish_checkpoints(run, state, improved)
        atomic_json(run / "metrics.json", history)
        print(json.dumps(row, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
