"""Read a trusted checkpoint and evaluate model and copy-initial baselines."""

import argparse
import json
from pathlib import Path
import torch

from ..data_process import DataProcessor
from .model import PositionPredictor
from .train import MODEL_KEYS, atomic_json, build_loader, evaluate_loader, read_checkpoint, make_renderer, to_device


def load_model(path, device):
    state = read_checkpoint(path)
    config = state["config"]
    model = PositionPredictor(**{k: config[k] for k in MODEL_KEYS}).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, DataProcessor.from_state_dict(state["processor"]), config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Trusted project-produced .pt")
    p.add_argument("--split", choices=("val", "test", "fit"), default="val",
                   help="fit evaluates both exact Roadmap training windows, not held-out validation")
    p.add_argument("--reference-checkpoint", help="Optional trusted free-Head checkpoint; fit comparison only")
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", help="Optional new JSON file; otherwise stdout only")
    args = p.parse_args(argv)
    if args.max_batches < 0:
        p.error("--max-batches must be nonnegative")
    if args.reference_checkpoint and args.split != "fit":
        p.error("--reference-checkpoint requires --split fit")
    if args.split == "fit" and args.max_batches:
        p.error("Fit evaluation always includes both complete windows")
    output = Path(args.output).resolve() if args.output else None
    if output and (output.exists() or output.with_suffix(output.suffix+".tmp").exists()):
        raise FileExistsError(output)
    device = torch.device(args.device)
    model, processor, config = load_model(args.checkpoint, device)
    fit_info = {}
    if args.split == "fit":
        from .fixed_fit import fixed_batches, manifest, TargetCache, evaluate_fixed, candidate
        if config["output_mode"] != "gs_geometry":
            raise ValueError("P0 fit diagnostics require Gaussian geometry output")
        raws = fixed_batches(config)
        evidence = manifest(raws, config)
        saved = read_checkpoint(args.checkpoint)
        if saved.get("fit") and saved["fit"]["manifest"] != evidence:
            raise ValueError("Fixed-window data differs from checkpoint manifest")
        models = {"model": model}
        if args.reference_checkpoint:
            reference, reference_processor, reference_config = load_model(args.reference_checkpoint, device)
            if (reference_processor.state_dict() != processor.state_dict()
                    or any(reference_config[k] != config[k] for k in
                           ("scene", "output_mode", "depth_cameras", "gs_convention", "render_settings", "horizon"))):
                raise ValueError("Reference must share scene, normalization, GS convention and rendering")
            if reference_config.get("fixed_windows") != config.get("fixed_windows"):
                raise ValueError("Reference must share frozen windows")
            models["reference"] = reference
            # Re-evaluation does not certify that historical training had paired settings.
            fit_info["reference_comparison"] = dict(checkpoint=str(Path(args.reference_checkpoint).resolve()),
                scope="same-window evaluation; training-budget/seed/mask pairing needs provenance review")
        batches = [processor(to_device(raw, device)) for raw in raws]
        metrics = evaluate_fixed(models, batches, processor, TargetCache(make_renderer(config)), config)
        fit_info.update(windows=evidence, held_out_validation=False)
        if "fit_contract" in config:
            fit_info["candidate"] = candidate(metrics, config["fit_contract"], saved["fit"]["completed_updates"])
    else:
        loader = build_loader(config, args.split)
        metrics = evaluate_loader(model, loader, processor, make_renderer(config), config, device, args.max_batches)
    result = dict(scene=config["scene"], horizon=config["horizon"],
                  prediction_protocol="online arriving action/tactile conditions", split=args.split, checkpoint=str(Path(args.checkpoint).resolve()),
                  partial=bool(args.max_batches), depth_cameras=config["depth_cameras"], metrics=metrics, **fit_info)
    if output:
        print(f"Creating {output} (and parent directories if needed; atomic .tmp staging)", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
