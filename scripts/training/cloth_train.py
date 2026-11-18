"""Train the existing G5-Direct graph on 008 episode_0, with a fixed E8 held-out evaluation."""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from scripts.dataloader.collate import collate_fn
from .cloth_checkpoint import atomic_save, finite, restore_rng, rng_state
from .cloth_checkpoint import same_tree
from scripts.training.train import atomic_json, evaluate_loader, make_renderer, train_epoch
from . import cloth_data as data
from .cloth_checkpoint import sha256, state_hash
from .cloth_protocol import fresh_model, probe_order

VERSION = "touch4d-g5-direct-008-episode0-v1"
SMOKE_STARTS = (113, 158, 205, 252)


class CheckedWindows:
    """Validate every visit against the frozen manifest and count actual updates."""

    def __init__(self, loader, starts):
        self.loader = loader
        self.starts = tuple(starts)
        self.visited = []
        self.updated = []

    def __iter__(self):
        for raw in self.loader:
            ids = raw["meta"]["frame_ids"]
            start = int(ids[0, 0])
            index = len(self.visited)
            if index >= len(self.starts) or start != self.starts[index]:
                raise ValueError(f"Unexpected 008 start {start} at index {index}")
            self.visited.append(start)
            yield raw

    def step(self, *_):
        if len(self.updated) + 1 != len(self.visited):
            raise ValueError("Expected exactly one optimizer update per train window")
        self.updated.append(self.visited[-1])


class Prepare:
    """Adapt raw batches to the original train/evaluate loops."""

    def __init__(self, processor, model):
        self.processor = processor
        self.model = model

    def __call__(self, raw):
        return data.prepare_batch(raw, self.processor, self.model)

    def __getattr__(self, name):
        return getattr(self.processor, name)


def train_windows(model, optimizer, processor, renderer, config, device, loader, starts):
    windows = CheckedWindows(loader, starts)
    hook = optimizer.register_step_post_hook(windows.step)
    try:
        metrics = train_epoch(model, optimizer, windows, Prepare(processor, model), renderer, config, device)
    finally:
        hook.remove()
    if tuple(windows.visited) != windows.starts or windows.updated != windows.visited:
        raise ValueError("Incomplete 008 training window coverage or skipped updates")
    json.dumps(metrics, allow_nan=False)
    finite(model.state_dict())
    finite(optimizer.state_dict())
    return {"starts": windows.visited, "successful_updates": len(windows.updated), "metrics": metrics}


def evaluate_windows(model, processor, renderer, config, device, loader, starts):
    windows = CheckedWindows(loader, starts)
    metrics = evaluate_loader(model, windows, Prepare(processor, model), renderer, config, device)
    if tuple(windows.visited) != windows.starts:
        raise ValueError("Incomplete 008 evaluation window coverage")
    for label in ("model", "copy_initial"):
        row = metrics[label]["episodes"][str(data.EPISODE)]
        if row["samples"] != len(starts) or row["render_samples"] != len(starts):
            raise ValueError(f"Incomplete {label} three-camera supervision coverage")
    json.dumps(metrics, allow_nan=False)
    return {"starts": windows.visited, "metrics": metrics}


def save_checkpoint(path, identity, model, optimizer, history):
    state = dict(version=VERSION, identity=identity, history=history,
                 model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state())
    atomic_save(path, state)


def restore_checkpoint(path, identity, model, optimizer, expected_orders):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["version"] != VERSION or state["identity"] != identity:
        raise ValueError("Checkpoint source, run, data, configuration or mode mismatch")
    history = state["history"]
    if (len(history) > len(expected_orders)
            or any(row["starts"] != starts or row["successful_updates"] != len(starts)
                   for row, starts in zip(history, expected_orders))):
        raise ValueError("Checkpoint visit/update ledger differs from the protocol")
    finite(state["model"])
    finite(state["optimizer"])
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if (not same_tree(model.state_dict(), state["model"])
            or not same_tree(optimizer.state_dict(), state["optimizer"])):
        raise ValueError("Model or AdamW state differs after strict restore")
    restore_rng(state["rng"])
    return history


def optimizer_for(model, config):
    return torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])


def subset_loader(dataset, starts):
    indices = [dataset.windows.index((data.EPISODE, start)) for start in starts]
    return DataLoader(Subset(dataset, indices), batch_size=1, num_workers=0, collate_fn=collate_fn,
                      generator=torch.Generator().manual_seed(42))


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
        raise RuntimeError("Requires one scheduled CUDA GPU")
    if args.mode == "smoke" and args.resume:
        raise ValueError("Use a clean output directory for each short test")
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_id):
        raise ValueError("Expected a SHA256 for the frozen source bundle")
    out = args.output.resolve()
    for other in (args.root.resolve(), args.depth_root.resolve(), Path.cwd().resolve()):
        if out == other or other in out.parents:
            raise ValueError("Run output must be separate from source and data")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import gsplat

    config = data.configuration(args.root, args.depth_root, data.BASE_SHA)
    train = data.make_dataset(config, "train")
    holdout = data.make_dataset(config, "val")
    data.validate_normalization(train)
    episode_path = train.episodes[data.EPISODE].path
    depth_path = train.depth_episodes[data.EPISODE].path
    data_hashes = {path.name: sha256(path) for path in
                   (episode_path / name for name in ("metadata.json", "controller.npz", "timestamps.npz", "tactile.npz"))}
    data_hashes["depth_metadata.json"] = sha256(depth_path / "metadata.json")
    device = torch.device("cuda:0")
    model = fresh_model("direct").to(device)
    processor = data.Cloth008Processor(data.SCENE)
    renderer = make_renderer(config)
    optimizer = optimizer_for(model, config)
    train_starts = [start for _, start in train.windows]
    eval_starts = [start for _, start in holdout.windows]
    orders = [[train_starts[i] for i in probe_order(data.make_loader(train, epoch))]
              for epoch in range(config["epochs"])]
    if train_starts != list(range(113, 253)) or eval_starts != list(range(268, 292)):
        raise ValueError("Expected 140 training windows and 24 held-out windows")
    smoke_orders = [list(SMOKE_STARTS[:2]), list(SMOKE_STARTS[2:])]
    identity = dict(mode=args.mode, run=str(out), source_bundle_sha256=args.source_id,
                    base_sha=data.BASE_SHA, config=config, model=model.config,
                    processor=processor.state_dict(), data_sha256=data_hashes,
                    epoch_orders=orders, held_out_starts=eval_starts,
                    torch=torch.__version__, gsplat=gsplat.__version__, cuda=torch.version.cuda,
                    initialization_sha256=state_hash(model.state_dict()))
    out.mkdir(parents=False, exist_ok=args.resume)
    import fcntl
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            if (out / "completion.json").exists():
                raise ValueError("This run is already complete")
            epochs = sorted(out.glob("epoch_*.pt"))
            if [p.name for p in epochs] != [f"epoch_{i:02d}.pt" for i in range(1, len(epochs) + 1)]:
                raise ValueError("Incomplete immutable epoch checkpoint sequence")
            checkpoint = epochs[-1] if epochs else out / "initial.pt"
            history = restore_checkpoint(checkpoint, identity, model, optimizer, orders)
            if len(history) != len(epochs):
                raise ValueError("Checkpoint does not represent its filename's epoch boundary")
        else:
            atomic_json(out / "protocol.json", identity)
            history = []
            save_checkpoint(out / "initial.pt", identity, model, optimizer, history)

        torch.cuda.reset_peak_memory_stats()
        if args.mode == "smoke":
            for index, starts in enumerate(smoke_orders, 1):
                row = train_windows(model, optimizer, processor, renderer, config, device,
                                    subset_loader(train, starts), starts)
                history.append(row)
                atomic_json(out / "updates.json", history)
                print(json.dumps(row, allow_nan=False), flush=True)
                checkpoint = out / f"chunk_{index:02d}.pt"
                save_checkpoint(checkpoint, identity, model, optimizer, history)
                if index == 1:
                    model = fresh_model("direct").to(device)
                    optimizer = optimizer_for(model, config)
                    history = restore_checkpoint(checkpoint, identity, model, optimizer, smoke_orders)
            eval_starts = [eval_starts[0]]
            eval_loader = subset_loader(holdout, eval_starts)
        else:
            for epoch in range(len(history), config["epochs"]):
                row = train_windows(model, optimizer, processor, renderer, config, device,
                                    data.make_loader(train, epoch), orders[epoch])
                history.append(row)
                checkpoint = out / f"epoch_{epoch+1:02d}.pt"
                save_checkpoint(checkpoint, identity, model, optimizer, history)
                atomic_json(out / "updates.json", history)
                print(json.dumps(dict(epoch=epoch+1, **row), allow_nan=False), flush=True)
            checkpoint = out / f"epoch_{config['epochs']:02d}.pt"
            eval_loader = data.make_loader(holdout)

        if state_hash(model.state_dict()) == identity["initialization_sha256"]:
            raise ValueError("Model weights never changed")
        evaluation = evaluate_windows(model, processor, renderer, config, device, eval_loader, eval_starts)
        atomic_json(out / "evaluation.json", evaluation)
        gpu = torch.cuda.get_device_properties(0)
        atomic_json(out / "completion.json", dict(
            status="completed", mode=args.mode, updates=sum(row["successful_updates"] for row in history),
            trained_starts=[start for row in history for start in row["starts"]], evaluated_starts=eval_starts,
            checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
            strict_mid_smoke_restore=args.mode == "smoke", source_bundle_sha256=args.source_id,
            allocated_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            reserved_peak_mib=torch.cuda.max_memory_reserved() / 2**20,
            gpu=gpu.name, gpu_uuid=str(getattr(gpu, "uuid", "")),
            job=os.environ["SLURM_JOB_ID"], step=os.environ.get("SLURM_STEP_ID", "batch")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--source-id", required=True)
    for name in ("root", "depth-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
