"""CPU-only bounded V4 implementation smoke; never launches training or Slurm.

python -B -m scripts.experiments.v4_stage1.smoke --architecture v4_stage1 --output audit/v4-stage1-smoke/run-01
Optional --data-root reads one real sparse batch (no dataset copying).
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import torch

from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.training.losses import chamfer_samples
from scripts.training.rendering import GaussianRenderer
from scripts.training.v4_runtime import (BASE_SHA, SPEC_SHA256, build_model, build_dataset, prepare_batch,
                                        v4_objective, save_checkpoint, load_checkpoint)
from .fixtures import synthetic_batch, analytic_rasterizer


def norm(tensor):
    return float(torch.linalg.vector_norm(tensor.detach().double()))


def run(output, data_root=None, horizon=10, prediction_interval=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    begin = time.perf_counter()
    model = build_model("v4_stage1", horizon=horizon, prediction_interval=prediction_interval)
    processor = DataProcessor("001-rope")
    raw = synthetic_batch(horizon=horizon, prediction_interval=prediction_interval)
    batch = prepare_batch(raw, processor, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    renderer = GaussianRenderer(analytic_rasterizer)
    record = dict(device="cpu", data="synthetic; independent variable-size targets",
                  rendering="original GaussianRenderer with analytic mock rasterizer; NOT gsplat",
                  config=model.config, frame_ids=batch["meta"]["frame_ids"].tolist(), steps=[])
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        prediction, trace = model(batch["inputs"], return_trace=True)
        loss, parts = v4_objective(prediction, batch, processor, renderer, chunk=64)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        loss.backward()
        gradients = {name: norm(p.grad) for name, p in model.named_parameters() if p.grad is not None}
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite V4 smoke gradient/loss")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        changes = {name: norm(p-before[name]) for name, p in model.named_parameters()}
        record["steps"].append(dict(successful_step=step+1, full_mock_loss=float(loss.detach()),
                                   chamfer=float(parts["geometry"].mean()), rgb=float(parts["rgb"].detach().mean()),
                                   alpha=float(parts["alpha"].detach().mean()), raw_gradient_norms=gradients,
                                   parameter_update_norms=changes))
    # Nondegenerate learned parameters: distinguish graph presence from magnitude.
    optimizer.zero_grad(set_to_none=True)
    prediction, trace = model(batch["inputs"], return_trace=True)
    geometry = chamfer_samples(prediction["xyz"], batch, 64).mean()
    late_loss = prediction["xyz"][:, -1].square().mean()
    states = [item["state"] for item in trace["steps"]]
    gradients = torch.autograd.grad(late_loss, states, retain_graph=True, allow_unused=True)
    record["ar_state_gradients"] = [dict(is_none=g is None, norm=None if g is None else norm(g)) for g in gradients]
    if any(g is None for g in gradients):
        raise AssertionError("Disconnected AR state")
    geometry.backward()
    groups = {}
    for group in ("gs_encoder", "local", "conditions.controller", "conditions.hand", "cross", "motion"):
        values = [p.grad for name, p in model.named_parameters() if name.startswith(group) and p.grad is not None]
        groups[group] = sum(norm(v)**2 for v in values)**.5
        if not values or groups[group] == 0:
            raise AssertionError(f"No nondegenerate gradient for {group}")
    record.update(geometry_only_loss=float(geometry.detach()), geometry_gradient_groups=groups,
                  prediction_shapes={key: list(value.shape) for key, value in prediction.items()},
                  attention_shape=list(trace["steps"][-1]["alpha"].shape),
                  anchor_valid_counts=trace["field"].anchor_mask.sum(-1).tolist())
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record.update(source_sha=sha, base_sha=BASE_SHA, spec_sha256=SPEC_SHA256,
                  source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()))
    path = output / "smoke.pt"
    save_checkpoint(path, model, optimizer, 3, source_sha=sha, processor_state=processor.state_dict())
    loaded = build_model("v4_stage1", horizon=horizon, prediction_interval=prediction_interval)
    restored_optimizer = torch.optim.AdamW(loaded.parameters(), lr=1e-4, weight_decay=1e-4)
    state = load_checkpoint(path, loaded, restored_optimizer, processor_state=processor.state_dict())
    with torch.no_grad():
        actual = loaded(batch["inputs"])
        for key in actual:
            torch.testing.assert_close(actual[key], prediction[key], rtol=0, atol=0)
    record.update(checkpoint_resume_steps=state["successful_steps"], checkpoint_exact_output_match=True,
                  checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if data_root:
        dataset = build_dataset("001-rope", "train", data_root, horizon=horizon, prediction_interval=prediction_interval)
        real = prepare_batch(collate_fn([dataset[0]]), processor, model)
        with torch.no_grad():
            real_prediction = model(real["inputs"])
            real_cd = chamfer_samples(real_prediction["xyz"], real, 128)
        record["real_data"] = dict(frame_ids=real["meta"]["frame_ids"].tolist(),
                                   shape=list(real_prediction["xyz"].shape), cd=real_cd.tolist())
    else:
        record["real_data"] = "not run: no local rebuilt GS dataset supplied"
    record.update(cuda="not run", seconds=time.perf_counter()-begin, passed=True)
    (output / "smoke.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("passed", "device", "prediction_shapes", "geometry_only_loss",
                                                  "geometry_gradient_groups", "ar_state_gradients", "seconds")}, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=["v4_stage1"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--prediction-interval", type=int, default=1)
    args = parser.parse_args()
    run(args.output, args.data_root, args.horizon, args.prediction_interval)


if __name__ == "__main__":
    main()
