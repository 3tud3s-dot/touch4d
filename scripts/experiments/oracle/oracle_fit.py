"""Isolated fixed-window capacity diagnostic; never a deployable predictor.

Uses the original five-frame P1a configuration, loss and 400 alternating updates.
Target GS enter the objective only. Window-specific optimized variables contain
target-derived information and must not be used for formal prediction/evaluation.
"""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import subprocess
import time
from types import SimpleNamespace

import torch

from scripts.data_process import DataProcessor
from scripts.experiments.fixed_window.fixed_fit import (WINDOWS, WEIGHTS, TargetCache, fixed_batches, manifest,
                        prediction_metrics, candidate, better, digest)
from scripts.training.losses import objective
from scripts.training.model import PositionPredictor, initial_knn
from scripts.experiments.oracle.oracle import OracleLatent, OracleAnchor
from scripts.training.train import (PROJECT, MODEL_KEYS, atomic_json, atomic_checkpoint,
                    copy_initial, make_renderer, to_device)


@torch.no_grad()
def evaluate_oracle(model, batches, processor, renderer, config, neighbors):
    was_training = model.training
    model.eval()
    try:
        result = {}
        for name in ("model", "copy_initial"):
            rows = {}
            for i, batch in enumerate(batches):
                key = f'{int(batch["meta"]["episode_id"][0])}:{int(batch["meta"]["frame_ids"][0,0])}'
                prediction = (model(batch["inputs"], i) if name == "model" else
                              copy_initial(batch["inputs"], "gs_geometry"))
                rows[key] = prediction_metrics(prediction, batch, processor, renderer,
                                              config["chamfer_chunk"], neighbors[i])
            result[name] = dict(windows=rows)
            for metric in ("cd_mm", "rgb", "alpha"):
                values = [sum(r[metric][h] for r in rows.values())/len(rows) for h in range(5)]
                result[name][metric] = values
                result[name]["mean_"+metric] = sum(values)/len(values)
        return result
    finally:
        model.train(was_training)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=tuple(WINDOWS))
    parser.add_argument("--kind", required=True, choices=("latent", "anchor"))
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--output", required=True, help="New isolated run directory")
    parser.add_argument("--cuda-check-only", action="store_true")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("Oracle experiments require a server CUDA allocation")
    source = json.loads(Path(args.baseline_config).read_text())
    config = dict(source["training"])
    if (config["scene"] != args.scene or config["gs_decoder"] != "anchors" or
            config.get("horizon", 5) != 5 or config.get("gs_structure", "none") != "none"):
        raise ValueError("Oracle requires the frozen H=5 isotropic P1a baseline")
    for key, expected in dict(seed=42, batch_size=1, lr=1e-4, weight_decay=1e-4, **WEIGHTS).items():
        if config.get(key) != expected:
            raise ValueError(f"Paired Oracle configuration mismatch: {key}")
    raws = fixed_batches(config)
    evidence = manifest(raws, config)
    if evidence != source["manifest"]:
        raise ValueError("Actual fixed-window inputs differ from the P1a evidence")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42); random.seed(42)
    base = PositionPredictor(**{k: config[k] for k in MODEL_KEYS if k in config})
    initial_hash = digest(base.state_dict())
    if initial_hash != config["initial_model_sha256"]:
        raise ValueError("Oracle base initialization differs from the paired P1a")
    processor = DataProcessor(args.scene)
    batches = [processor(to_device(raw, "cuda")) for raw in raws]
    constructor = OracleLatent if args.kind == "latent" else OracleAnchor
    model = constructor(base, [raw["inputs"]["gs_mask"] for raw in raws]).cuda()
    # A coordinate variable has a different optimization scale from a neural
    # weight. Freeze one independent Anchor diagnostic, not a predictor ablation:
    # lr=.01 can traverse the audited motion in 200 window updates; decay=0
    # avoids an implicit penalty toward the zero-displacement initial template.
    optimization = (dict(lr=1e-2, weight_decay=0.) if args.kind == "anchor" else
                    dict(lr=config["lr"], weight_decay=config["weight_decay"]))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 **optimization)
    renderer = TargetCache(make_renderer(config))
    neighbors = [initial_knn(b["inputs"]["gs_xyz"], b["inputs"]["gs_mask"], 16) for b in batches]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    atomic_json(output/"config.json", dict(training=config, manifest=evidence,
        commit=commit, oracle_kind=args.kind, diagnostic_only=True, oracle_optimizer=optimization,
        optimized_target_information=True, initial_model_sha256=initial_hash,
        processor=processor.state_dict(), source_files={str(p.relative_to(PROJECT)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(__file__).with_name("oracle.py"))}))
    # One real forward/backward, then reload exactly fresh weights and optimizer.
    fresh = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    loss, _ = objective(model(batches[0]["inputs"], 0), batches[0], processor, renderer,
                        0., config["chamfer_chunk"], **WEIGHTS)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    stream = BytesIO()
    torch.save(model.diagnostic_checkpoint(optimizer), stream)
    stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=False)
    restored = constructor(base, [raw["inputs"]["gs_mask"] for raw in raws]).cuda()
    restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],
                                         **optimization)
    restored.load_diagnostic_checkpoint(saved, restored_optimizer)
    with torch.no_grad():
        expected = model(batches[0]["inputs"], 0)
        actual = restored(batches[0]["inputs"], 0)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    # Probe the now nonzero Head, including final-output dependence on earlier
    # latents. These diagnostic updates are discarded before the 400-step run.
    optimizer.zero_grad(set_to_none=True)
    final = model(batches[0]["inputs"], 0)["xyz"][:, -1].square().sum()
    final.backward()
    variable_grad = model.variables[0].grad
    if variable_grad is None or not torch.isfinite(variable_grad).all() or not variable_grad[-1].abs().sum() > 0:
        raise AssertionError("Oracle variable gradient missing/nonfinite")
    early_grad = float(variable_grad[0].abs().sum())
    if args.kind == "latent" and early_grad <= 0:
        raise AssertionError("Oracle AR final horizon does not reach first latent")
    for module, opt in ((model, optimizer), (restored, restored_optimizer)):
        opt.zero_grad(set_to_none=True)
        probe = module(batches[0]["inputs"], 0)["xyz"].square().mean()
        probe.backward(); opt.step()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
    torch.cuda.synchronize()
    atomic_json(output/"cuda_check.json", dict(loss=float(loss.detach()), grad_norm=float(norm),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        seconds=time.perf_counter()-t0, checkpoint_roundtrip=True,
        restored_next_update_equal=True, final_to_first_latent_gradient_l1=early_grad))
    model.load_state_dict(fresh); optimizer.state.clear(); optimizer.zero_grad(set_to_none=True)
    assert digest(model.state_dict()) == digest(fresh) and not optimizer.state
    del fresh, saved, loss, restored, restored_optimizer, stream, expected, actual, final, variable_grad, probe
    if args.cuda_check_only:
        return 0
    history, best = [], None
    t0 = time.perf_counter()
    for update in range(1, 401):
        i = (update-1) % 2
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss, values = objective(model(batches[i]["inputs"], i), batches[i], processor, renderer,
                                  0., config["chamfer_chunk"], **WEIGHTS)
        if not values["render_time_available"].all():
            raise ValueError("Unobserved horizon in frozen Oracle window")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if update % 20:
            continue
        metrics = evaluate_oracle(model, batches, processor, renderer, config, neighbors)
        row = candidate(metrics, None, update)
        improved = better(row, best, None)
        if improved:
            best = row
        history.append(dict(update=update, window_updates=[update//2]*2, metrics=metrics,
                            last_train_loss=float(loss.detach()), grad_norm_before_clip=float(norm), candidate=row))
        state = model.diagnostic_checkpoint(optimizer, config=config, history=history,
            commit=commit, completed_updates=update, best_candidate=best, manifest=evidence,
            oracle_optimizer=optimization)
        if improved:
            atomic_checkpoint(output/"lowest_cd.pt", state)
        atomic_checkpoint(output/f"step_{update:06d}.pt", state)
        atomic_checkpoint(output/"last.pt", state)
        atomic_json(output/"metrics.json", history)
        print(json.dumps(dict(update=update, cd_mm=row["rank"][0], last_cd_mm=row["rank"][1])), flush=True)
    atomic_checkpoint(output/"last_400.pt", state)
    from scripts.training.visualize import export_gaussians
    summary = dict(scene=args.scene, oracle_kind=args.kind, diagnostic_only=True,
        completed_updates=400, seconds=time.perf_counter()-t0, windows=evidence,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, checkpoints={})
    for name in ("lowest_cd", "last_400"):
        state = torch.load(output/(name+".pt"), map_location="cpu", weights_only=False)
        model.load_diagnostic_checkpoint(state); model.eval()
        metrics = evaluate_oracle(model, batches, processor, renderer, config, neighbors)
        summary["checkpoints"][name] = dict(update=state["metadata"]["completed_updates"], metrics=metrics)
        with torch.no_grad():
            for i, ((episode, start), raw, batch) in enumerate(zip(WINDOWS[args.scene], raws, batches)):
                directory = output/"visualizations"/name/f"episode-{episode}-start-{start}"
                export_gaussians(SimpleNamespace(output=str(directory), checkpoint=str(output/(name+".pt")),
                    episode=episode, start=start, split="train"), config, raw, batch,
                    model(batch["inputs"], i), processor)
    atomic_json(output/"fit_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
