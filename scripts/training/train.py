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
from .rendering import DepthRenderer

VERSION = "touch4d-position-v0"
PROJECT = Path(__file__).resolve().parents[2]


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


@torch.no_grad()
def evaluate_loader(model, loader, processor, renderer, config, device, max_batches=0):
    was_training = model.training
    model.eval()
    trackers = {name: EpisodeMetrics(processor.state_dict()["scale"], config["depth_weight"])
                for name in ("model", "copy_initial")}
    try:
        for raw in loader if not max_batches else islice(loader, max_batches):
            batch = processor(to_device(raw, device))
            outputs = {"model": model(batch["inputs"]),
                       "copy_initial": batch["inputs"]["gs_xyz"][:, None].expand(-1, 5, -1, -1)}
            for name, prediction in outputs.items():
                _, metrics = objective(prediction, batch, processor, renderer,
                                       config["depth_weight"], config["chamfer_chunk"])
                trackers[name].add(batch["meta"]["episode_id"], metrics)
        return {name: tracker.summary() for name, tracker in trackers.items()}
    finally:
        model.train(was_training)


def train_epoch(model, optimizer, loader, processor, renderer, config, device):
    model.train()
    metrics = EpisodeMetrics(processor.state_dict()["scale"], config["depth_weight"])
    stream = loader if not config["max_train_batches"] else islice(loader, config["max_train_batches"])
    for raw in stream:
        batch = processor(to_device(raw, device))
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["inputs"])
        loss, values = objective(prediction, batch, processor, renderer,
                                 config["depth_weight"], config["chamfer_chunk"])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient norm")
        optimizer.step()
        metrics.add(batch["meta"]["episode_id"], values)
    return metrics.summary()


def checkpoint_state(model, optimizer, processor, config, epoch, best, history):
    return dict(version=VERSION, config=dict(config), processor=processor.state_dict(),
                model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                best=best, history=history, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                python_rng=random.getstate())


def read_checkpoint(path):
    # Only load checkpoints produced by this project from a trusted location.
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("version") != VERSION:
        raise ValueError("Unsupported training checkpoint version")
    DataProcessor.from_state_dict(value["processor"])
    if value["config"]["scene"] != value["processor"]["scene"]:
        raise ValueError("Checkpoint scene and normalization disagree")
    return value


def recover_best_score(last_state, run):
    """Reconcile an already-published best.pt with the resumed last.pt.

    best.pt is published first, so a crash can leave it newer/better than last.
    Keep resuming model/optimizer/RNG from last, but never forget the better
    saved score. Missing, incompatible or regressed best files fail closed.
    This only reads files, and is also compatible with existing V0 checkpoints.
    """
    path = Path(run) / "best.pt"
    if not path.is_file():
        raise ValueError(f"Cannot safely resume: missing best checkpoint {path}")
    saved = read_checkpoint(path)
    comparable = lambda c: {k: v for k, v in c.items() if k != "epochs"}
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
    p.add_argument("--depth-weight", type=float, default=1.)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--chamfer-chunk", type=int, default=512)
    p.add_argument("--max-train-batches", type=int, default=0, help="0 means full epoch")
    p.add_argument("--max-val-batches", type=int, default=0, help="0 means full validation")
    p.add_argument("--overfit-windows", type=int, default=0, help="Fixed first N training windows; requires --single-epochs 0")
    p.add_argument("--device", default="cuda")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (min(args.epochs, args.batch_size, args.chamfer_chunk) < 1 or args.single_epochs < 0
            or args.single_epochs > args.epochs or min(args.workers, args.seed, args.max_train_batches,
                args.max_val_batches, args.overfit_windows) < 0):
        p.error("Invalid epoch, batch, worker, seed or sampling limits")
    if args.overfit_windows and args.single_epochs:
        p.error("Fixed-window overfit requires --single-epochs 0")
    if args.resume and args.run_id:
        p.error("--resume uses the existing run directory; do not also specify --run-id")
    if any(not math.isfinite(v) or v < 0 for v in (args.lr, args.weight_decay, args.depth_weight)) or args.lr == 0:
        p.error("Learning rate must be positive; weights must be finite and nonnegative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        p.error("Real training requires a server CUDA allocation; CPU unit tests are separate")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    root = Path(args.root).resolve()
    depth_root = Path(args.depth_root).resolve() if args.depth_root else root.parent / "touch4d-depth-v1"
    config = vars(args).copy()
    for key in ("run_id", "resume", "device"):
        config.pop(key)
    config.update(root=str(root), depth_root=str(depth_root),
                  depth_cameras=choose_cameras(args.scene, depth_root, args.depth_cameras))
    processor = DataProcessor(args.scene)
    model = PositionPredictor(args.width, args.heads, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    renderer = DepthRenderer()
    # Fail before creating a run when the renderer dependency is absent.
    import gsplat
    start, best, history = 0, float("inf"), []
    if args.resume:
        if Path(args.resume).name != "last.pt":
            p.error("Resume from last.pt to preserve the completed-epoch history; best.pt is for evaluation")
        state = read_checkpoint(args.resume)
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
