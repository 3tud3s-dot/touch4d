"""Read a trusted checkpoint and evaluate model and copy-initial baselines."""

import argparse
import json
from pathlib import Path
import torch

from ..data_process import DataProcessor
from .model import PositionPredictor
from .rendering import DepthRenderer
from .train import atomic_json, build_loader, evaluate_loader, read_checkpoint


def load_model(path, device):
    state = read_checkpoint(path)
    config = state["config"]
    model = PositionPredictor(**{k: config[k] for k in ("width", "heads", "layers")}).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, DataProcessor.from_state_dict(state["processor"]), config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Trusted project-produced .pt")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", help="Optional new JSON file; otherwise stdout only")
    args = p.parse_args(argv)
    if args.max_batches < 0:
        p.error("--max-batches must be nonnegative")
    output = Path(args.output).resolve() if args.output else None
    if output and (output.exists() or output.with_suffix(output.suffix+".tmp").exists()):
        raise FileExistsError(output)
    device = torch.device(args.device)
    model, processor, config = load_model(args.checkpoint, device)
    loader = build_loader(config, args.split)
    metrics = evaluate_loader(model, loader, processor, DepthRenderer(), config, device, args.max_batches)
    result = dict(scene=config["scene"], split=args.split, checkpoint=str(Path(args.checkpoint).resolve()),
                  partial=bool(args.max_batches), depth_cameras=config["depth_cameras"], metrics=metrics)
    if output:
        print(f"Creating {output} (and parent directories if needed; atomic .tmp staging)", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
