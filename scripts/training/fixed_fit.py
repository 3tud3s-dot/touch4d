"""Roadmap P0/P1a: paired exact-window fit protocol with unchanged loss semantics.

Entry: training.train --protocol fixed400 --scene ... --fit-contract reviewed.json.
Alternatively --fit-diagnostic retains every 20-update checkpoint without any
acceptance decision. lowest_cd.pt is a descriptive minimum, NEVER best_fit.pt.
The contract has version='p0-fit-v1', scene, supervision_review (evidence path),
tie_cd_mm, tie_h5_mm, tie_render, and windows keyed 'episode:start'. Each window
requires five-element rgb_max, alpha_max, cd_mm_max, motion_rms_mm_max,
neighbor_rms_mm_max; two-element axis_ratio_range, volume_ratio_range,
scale_mm_range; and cov_condition_max. All limits are reviewed, explicit values,
not defaults or inferred from a model's outcome. Both motion diagnostics are
reported; only cube translation RMS / rope neighbor RMS enters its fit checks.
Structural acceptance always additionally requires a human visualization review.

Only existing target_depth_valid pixels are Positive; all other pixels Ignore.
No Negative pixels are introduced without separately audited background evidence.
"""

from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace
import uuid

import torch

from ..data_process import DataProcessor
from ..dataloader.collate import collate_fn
from .losses import nearest_distances, objective, render_samples
from .model import PositionPredictor, initial_knn
from .train import (PROJECT, MODEL_KEYS, GS_CAMERAS, GS_CONVENTION, atomic_json,
                    atomic_checkpoint, build_loader, choose_cameras, checkpoint_state,
                    copy_initial, gs_render_settings, make_renderer, read_checkpoint,
                    restore_training, to_device, DECODER_DEFAULTS, resolve_decoder,
                    resolve_structure, validate_structure_config, resolve_horizon)

WINDOWS = {"001-rope": ((1, 85), (2, 73)), "051-cube": ((0, 109), (0, 132))}
UPDATES, EVAL_EVERY = 400, 20
WEIGHTS = dict(chamfer_weight=1., rgb_weight=1., alpha_weight=.1, scale_weight=.01)


def window_key(episode, start):
    return f"{episode}:{start}"


def validate_contract(contract, scene):
    """Fail closed; no production thresholds are invented by this module."""
    if (contract.get("version") != "p0-fit-v1" or contract.get("scene") != scene
            or not isinstance(contract.get("supervision_review"), str)
            or not contract["supervision_review"].strip()):
        raise ValueError("P0 contract needs version, matching scene and supervision_review evidence")
    def number(value, positive=False):
        return (type(value) in (int, float) and math.isfinite(value)
                and (value > 0 if positive else value >= 0))
    for key in ("tie_cd_mm", "tie_h5_mm", "tie_render"):
        if not number(contract.get(key)):
            raise ValueError(f"Freeze a finite nonnegative {key} before training")
    rows = contract.get("windows", {})
    if set(rows) != {window_key(*w) for w in WINDOWS[scene]}:
        raise ValueError("Contract must cover exactly the two Roadmap windows")
    for key, row in rows.items():
        for name in ("rgb_max", "alpha_max", "cd_mm_max", "motion_rms_mm_max", "neighbor_rms_mm_max"):
            values = row.get(name)
            if not isinstance(values, list) or len(values) != 5 or not all(number(v) for v in values):
                raise ValueError(f"{key}: {name} requires five finite nonnegative limits")
        for name in ("axis_ratio_range", "volume_ratio_range", "scale_mm_range"):
            values = row.get(name)
            if (not isinstance(values, list) or len(values) != 2
                    or not all(number(v, True) for v in values) or values[0] > values[1]):
                raise ValueError(f"{key}: {name} requires a positive ordered [min,max]")
        if not number(row.get("cov_condition_max"), True) or row["cov_condition_max"] < 1:
            raise ValueError(f"{key}: cov_condition_max must be finite and >= 1")
    return contract


def digest(value):
    """Hash actual tensors (including masks, calibration and conditions), not paths."""
    result = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            result.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            result.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                result.update(json.dumps(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            result.update(json.dumps(item, allow_nan=False, sort_keys=True).encode())
    visit(value)
    return result.hexdigest()


def select_windows(dataset, scene, windows=None):
    requested = WINDOWS[scene] if windows is None else tuple(tuple(w) for w in windows)
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate requested windows")
    selected = []
    for episode, start in requested:
        matches = [i for i, value in enumerate(dataset.windows) if tuple(value) == (episode, start)]
        if len(matches) != 1:
            raise ValueError(f"{scene} ({episode},{start}): expected one window, found {len(matches)}")
        raw = collate_fn([dataset[matches[0]]])
        meta = raw["meta"]
        expected = list(range(start, start+getattr(dataset, "horizon", 5)+1))
        if (meta["scene"] != [scene] or meta["episode_id"].tolist() != [episode]
                or meta["frame_ids"].tolist() != [expected]):
            raise ValueError(f"{scene} ({episode},{start}): actual scene/episode/frame_ids mismatch")
        selected.append(raw)
    return selected


def fixed_batches(config):
    # Bypass legacy 'first N' sampling, and retain the existing split/stride rules.
    loader = build_loader(dict(config, overfit_windows=0, batch_size=1), "train", stage="all")
    return select_windows(loader.dataset, config["scene"], config.get("fixed_windows"))


def frozen_windows(path, scene, horizon, state=None):
    """Freeze GT-audited frame IDs before training, without prediction-based selection."""
    saved = state["config"].get("window_evidence") if state else None
    if path:
        content = Path(path).read_bytes()
        evidence = dict(manifest=json.loads(content), sha256=hashlib.sha256(content).hexdigest())
        if saved is not None and evidence != saved:
            raise ValueError("Cannot resume with changed frozen window manifest")
    else:
        evidence = saved
    if evidence is None:
        if horizon != 5:
            raise ValueError("Extended fixed400 horizon requires --window-manifest frozen from GT audit")
        return None, None
    document = evidence["manifest"]
    if (document.get("version") != "structural-h15-windows-v1" or document.get("horizon") != horizon
            or not document.get("basis") or not document.get("evidence_sha256")):
        raise ValueError("Frozen manifest needs matching horizon and GT audit provenance")
    rows = document.get("windows", {}).get(scene, [])
    if len(rows) != 2:
        raise ValueError("Frozen manifest must select exactly two windows per scene")
    selected = []
    for row in rows:
        episode, start = row.get("episode_id"), row.get("start_frame")
        if (type(episode) is not int or type(start) is not int or min(episode, start) < 0
                or row.get("frame_ids") != list(range(start, start+horizon+1))
                or not row.get("geometry_sha256")):
            raise ValueError("Frozen manifest contains invalid full frame IDs or missing geometry provenance")
        selected.append([episode, start])
    if len({tuple(w) for w in selected}) != 2:
        raise ValueError("Frozen windows must be distinct")
    return selected, evidence


def manifest(raws, config):
    rows = []
    for raw in raws:
        meta, valid = raw["meta"], raw["targets"]["target_depth_valid"]
        if meta["depth_camera_ids"] != [config["depth_cameras"]]:
            raise ValueError("Fixed-window camera order differs from the frozen configuration")
        if meta["frame_ids"].shape[1] != config.get("horizon", 5)+1:
            raise ValueError("Fixed-window frame count disagrees with horizon")
        counts = valid.sum((-1, -2))[0]
        if not (counts > 0).all():
            raise ValueError("Every fixed-window horizon/camera needs trusted Positive pixels")
        row = dict(scene=meta["scene"][0], episode_id=int(meta["episode_id"][0]),
                   frame_ids=meta["frame_ids"][0].tolist(), cameras=meta["depth_camera_ids"][0],
                   positive_pixels=counts.tolist(), negative_pixels=0,
                   ignore_pixels=(valid.shape[-1]*valid.shape[-2]-counts).tolist(),
                   data_sha256=digest(raw))
        row["start_frame"] = row["frame_ids"][0]
        rows.append(row)
    return rows


class TargetCache:
    """Cache only these two immutable target renders; prediction stays differentiable."""
    def __init__(self, renderer):
        self.renderer, self.cache = renderer, {}

    def __call__(self, prediction, batch, processor):
        return self.renderer(prediction, batch, processor)

    def render_targets(self, batch, processor):
        key = id(batch)
        if key not in self.cache:
            with torch.no_grad():
                self.cache[key] = (batch, self.renderer.render_targets(batch, processor))
        return self.cache[key][1]


@torch.no_grad()
def geometry_diagnostics(prediction, batch, scale, neighbors=None):
    """Template-identity diagnostics only; never treat GT rows as particle IDs."""
    x = batch["inputs"]
    keep = x["gs_mask"][0]
    if x["gs_xyz"].shape[0] != 1 or not keep.any():
        raise ValueError("Fixed fit requires a nonempty batch of one")
    if any(not torch.isfinite(v[0, :, keep]).all() for v in prediction.values()):
        raise FloatingPointError("Nonfinite valid Gaussian; diagnostic/rasterization aborted")
    delta = (prediction["xyz"][0, :, keep].double()-x["gs_xyz"][0, keep].double())*scale*1000
    mean = delta.mean(1)
    rms = (delta-mean[:, None]).square().sum(-1).mean(-1).sqrt()
    if neighbors is None:
        neighbors = initial_knn(x["gs_xyz"], x["gs_mask"], 16)
    ids, edges = neighbors
    # Build deltas in padded index space but only select real, non-self edges.
    all_delta = (prediction["xyz"][0].double()-x["gs_xyz"][0].double())*scale*1000
    valid_edges = edges[0] & keep[:, None] & (ids[0] != torch.arange(len(keep), device=keep.device)[:, None])
    difference = (all_delta[:, :, None]-all_delta[:, ids[0]])[:, valid_edges]
    edge_norm = difference.norm(dim=-1)
    scales = prediction["log_scales"][0, :, keep].double().exp()*scale*1000
    ratios = (prediction["log_scales"][0, :, keep].double()-x["gs_attr"][0, keep, :3].double()).exp()
    volumes = ratios.prod(-1)
    qnorm = prediction["quaternions"][0, :, keep].double().norm(dim=-1)
    if (not torch.isfinite(scales).all() or not torch.isfinite(ratios).all()
            or not torch.isfinite(volumes).all() or not (scales > 0).all() or not (qnorm > 0).all()):
        raise FloatingPointError("Invalid Gaussian scale/quaternion/covariance")
    # For normalized rotations covariance eigenvalues are exactly squared scales.
    eigen = scales.square()
    condition = eigen.max(-1).values/eigen.min(-1).values
    if not torch.isfinite(eigen).all() or not torch.isfinite(condition).all():
        raise FloatingPointError("Nonfinite covariance spectrum")
    rows = []
    for h in range(prediction["xyz"].shape[1]):
        def extent(values):
            flat = values.flatten()
            return [float(flat.min()), float(flat.max())]
        rows.append(dict(mean_displacement_mm=mean[h].tolist(), translation_rms_mm=float(rms[h]),
            neighbor_rms_mm=float(edge_norm[h].square().mean().sqrt()) if edge_norm.shape[1] else 0.,
            neighbor_p95_mm=float(torch.quantile(edge_norm[h], .95)) if edge_norm.shape[1] else 0.,
            neighbor_edges=int(valid_edges.sum()), axis_ratio=extent(ratios[h]),
            volume_ratio=extent(volumes[h]), scale_mm=extent(scales[h]),
            covariance_eigen_mm2=extent(eigen[h]), covariance_condition_max=float(condition[h].max()),
            quaternion_norm_error=float((qnorm[h]-1).abs().max()), finite=True, covariance_positive=True))
    return rows


@torch.no_grad()
def prediction_metrics(prediction, batch, processor, renderer, chunk, neighbors=None):
    scale = processor.state_dict()["scale"]
    structure = geometry_diagnostics(prediction, batch, scale, neighbors)
    keep = batch["inputs"]["gs_mask"][0]
    cd = []
    for h in range(prediction["xyz"].shape[1]):
        p = prediction["xyz"][0, h, keep]
        t = batch["targets"]["target_xyz"][h][0, batch["targets"]["target_mask"][h][0]]
        cd.append(float((nearest_distances(p, t, chunk).mean()+nearest_distances(t, p, chunk).mean())*.5)*scale*1000)
    rendered = renderer(prediction, batch, processor)
    target = renderer.render_targets(batch, processor)
    values = {}
    for name in ("rgb", "alpha"):
        _, _, times, available = render_samples(rendered[name], target[name], batch["targets"]["target_depth_valid"])
        if not available.all():
            raise ValueError("Missing trusted render supervision in a fixed-window horizon")
        values[name] = times[0].tolist()
    return dict(cd_mm=cd, rgb=values["rgb"], alpha=values["alpha"], diagnostics=structure,
                mean_cd_mm=sum(cd)/len(cd), mean_rgb=sum(values["rgb"])/len(cd), mean_alpha=sum(values["alpha"])/len(cd))


@torch.no_grad()
def evaluate_fixed(models, batches, processor, renderer, config, neighbors=None):
    """Equal two-window / horizon means, not episode-balanced validation."""
    modes = {name: model.training for name, model in models.items()}
    try:
        for model in models.values():
            model.eval()
        results = {}
        for name in [*models, "copy_initial"]:
            rows = {}
            for i, batch in enumerate(batches):
                prediction = (copy_initial(batch["inputs"], "gs_geometry") if name == "copy_initial"
                              else models[name](batch["inputs"]))
                key = window_key(int(batch["meta"]["episode_id"][0]), int(batch["meta"]["frame_ids"][0, 0]))
                rows[key] = prediction_metrics(prediction, batch, processor, renderer,
                    config["chamfer_chunk"], None if neighbors is None else neighbors[i])
            result = dict(windows=rows)
            for metric in ("cd_mm", "rgb", "alpha"):
                result[metric] = [sum(r[metric][h] for r in rows.values())/len(rows) for h in range(config.get("horizon", 5))]
                result["mean_"+metric] = sum(result[metric])/len(result[metric])
            results[name] = result
        return results
    finally:
        for name, model in models.items():
            model.train(modes[name])


def candidate(metrics, contract, update):
    """Eligibility and fit checks are distinct; structural review remains manual."""
    if contract is None:
        row = metrics["model"]
        rank = [row["mean_cd_mm"], row["cd_mm"][-1], row["mean_rgb"]+.1*row["mean_alpha"]]
        if not all(math.isfinite(v) for v in rank):
            raise FloatingPointError("Nonfinite diagnostic score")
        return dict(update=update, eligible=None, automatic_fit_checks=None,
                    manual_structure_review="pending", diagnostic_only=True, rank=rank,
                    ineligible_reasons=["Acceptance tolerances not frozen; no pass/fail decision"], fit_reasons=[])
    reasons, fit_reasons = [], []
    for key, row in metrics["model"]["windows"].items():
        limits = contract["windows"][key]
        base = metrics["copy_initial"]["windows"][key]
        for h, diag in enumerate(row["diagnostics"]):
            label = f"{key}/h{h+1}"
            if not diag["finite"] or not diag["covariance_positive"]:
                reasons.append(label+": invalid numbers/covariance")
            for name in ("rgb", "alpha"):
                if not math.isfinite(row[name][h]) or row[name][h] > limits[name+"_max"][h]:
                    reasons.append(label+": "+name)
            for name in ("axis_ratio", "volume_ratio", "scale_mm"):
                low, high = limits[name+"_range"]
                if not low <= diag[name][0] <= diag[name][1] <= high:
                    reasons.append(label+": "+name)
            if diag["covariance_condition_max"] > limits["cov_condition_max"]:
                reasons.append(label+": covariance condition")
            if not math.isfinite(row["cd_mm"][h]):
                reasons.append(label+": nonfinite CD")
            elif row["cd_mm"][h] > limits["cd_mm_max"][h]:
                fit_reasons.append(label+": geometry tolerance")
            diagnostic, limit = (("translation_rms_mm", "motion_rms_mm_max") if contract["scene"] == "051-cube"
                                  else ("neighbor_rms_mm", "neighbor_rms_mm_max"))
            if diag[diagnostic] > limits[limit][h]:
                fit_reasons.append(label+": structure diagnostic")
        if not row["mean_cd_mm"] < base["mean_cd_mm"]:
            fit_reasons.append(key+": geometry does not improve copy_initial")
        if not row["mean_rgb"]+.1*row["mean_alpha"] < base["mean_rgb"]+.1*base["mean_alpha"]:
            fit_reasons.append(key+": rendering does not improve copy_initial")
    row = metrics["model"]
    return dict(update=update, eligible=not reasons, ineligible_reasons=reasons,
                automatic_fit_checks=not (reasons or fit_reasons), fit_reasons=fit_reasons,
                manual_structure_review="pending", rank=[row["mean_cd_mm"], row["cd_mm"][-1],
                row["mean_rgb"]+.1*row["mean_alpha"]])


def better(candidate_row, incumbent, contract):
    if contract is None:
        return incumbent is None or tuple(candidate_row["rank"]) < tuple(incumbent["rank"])
    if not candidate_row["eligible"]:
        return False
    if incumbent is None:
        return True
    for a, b, tolerance in zip(candidate_row["rank"], incumbent["rank"],
                              (contract["tie_cd_mm"], contract["tie_h5_mm"], contract["tie_render"])):
        if abs(a-b) > tolerance:
            return a < b
    return False  # Within all frozen tolerances retain the earlier checkpoint.


def recover_fit_best(state, run):
    """Best is published before last. Protect a newer best after an interruption."""
    recorded = state["fit"]["best_candidate"]
    diagnostic = state["config"].get("fit_diagnostic", False)
    path = Path(run)/("lowest_cd.pt" if diagnostic else "best_fit.pt")
    if not path.is_file():
        if recorded is not None:
            raise ValueError("Cannot resume: recorded best_fit.pt is missing")
        return None
    saved = read_checkpoint(path)
    if (saved["config"] != state["config"] or saved["processor"] != state["processor"]
            or saved["fit"]["manifest"] != state["fit"]["manifest"]):
        raise ValueError("Cannot resume: best_fit protocol, data or normalization differs")
    actual = saved["fit"]["best_candidate"]
    if (actual is None or (not diagnostic and not actual["eligible"]) or saved["fit"]["completed_updates"] != actual["update"]
            or (actual != recorded and not better(actual, recorded, state["config"]["fit_contract"]))):
        raise ValueError("Cannot resume: best_fit candidate regressed or is inconsistent")
    return actual


def publish_fit(run, state, improved):
    run = Path(run)
    diagnostic = state["config"].get("fit_diagnostic", False)
    if diagnostic:
        atomic_checkpoint(run/f'step_{state["fit"]["completed_updates"]:06d}.pt', state)
    if improved:
        atomic_checkpoint(run/("lowest_cd.pt" if diagnostic else "best_fit.pt"), state)
    atomic_checkpoint(run/"last.pt", state)
    if state["fit"]["completed_updates"] == UPDATES:
        atomic_checkpoint(run/"last_400.pt", state)


def update_loop(model, optimizer, batches, processor, renderer, config, on_evaluation,
                start=0, evaluate=None):
    """Count successful optimizer steps; empty supervision is an error, never a skip."""
    if type(start) is not int or not 0 <= start <= UPDATES or start % EVAL_EVERY:
        raise ValueError("Resume requires a completed 20-update evaluation boundary")
    if len(batches) != 2:
        raise ValueError("P0 requires exactly two windows")
    evaluate = evaluate or (lambda: evaluate_fixed({"model": model}, batches, processor, renderer, config))
    for completed in range(start, UPDATES):
        batch = batches[completed % 2]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, values = objective(model(batch["inputs"]), batch, processor, renderer,
                                  0., config["chamfer_chunk"], **WEIGHTS)
        if not bool(values["render_time_available"].all()):
            raise ValueError("P0 cannot count updates with an unobserved horizon")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (completed+1) % EVAL_EVERY == 0:
            metrics = evaluate()
            on_evaluation(completed+1, metrics, float(loss.detach()), float(norm))


def report_html(history, best, completed, decoder="free", structure="none", horizon=5):
    stage = {"free": "P0 free GS Head", "anchors": "P1a shared anchor displacement",
             "anchors_anisotropic": "P1a.5 learnable anisotropic RBF"}[decoder]
    if structure == "anchor_attention":
        stage = "P2 structural anchor attention + P1a decoder"
    best_name = "lowest_cd (diagnostic only)" if best and best.get("diagnostic_only") else "best_fit"
    rows = []
    for row in history:
        m, c = row["metrics"]["model"], row["candidate"]
        rows.append(f'<tr><td>{row["update"]}</td><td>{m["mean_cd_mm"]:.5f}</td>'
                    f'<td>{m["cd_mm"][-1]:.5f}</td><td>{m["mean_rgb"]:.6f}</td>'
                    f'<td>{m["mean_alpha"]:.6f}</td><td>{" / ".join(f"{v:.4f}" for v in m["cd_mm"])}</td>'
                    f'<td>{c["eligible"]}</td>'
                    f'<td>{c["automatic_fit_checks"]}</td></tr>')
    points = []
    maximum = max([row["metrics"]["model"]["mean_cd_mm"] for row in history] or [1.]) or 1.
    for row in history:
        points.append(f'{40+row["update"]/400*720:.2f},{200-row["metrics"]["model"]["mean_cd_mm"]/maximum*180:.2f}')
    detail = ''
    selected = []
    if best:
        selected.extend((best_name, row) for row in history if row["update"] == best["update"])
    if history:
        selected.append(("latest", history[-1]))
    for name, row in selected:
        detail += f'<h2>{name} · update {row["update"]}</h2>'
        for key, values in row["metrics"]["model"]["windows"].items():
            baseline = row["metrics"]["copy_initial"]["windows"][key]
            detail += (f'<h3>episode:start {html.escape(key)}</h3><table><tr><th>horizon</th>'
                       '<th>CD / copy mm</th><th>RGB / copy</th><th>Alpha / copy</th>'
                       '<th>mean displacement XYZ mm</th><th>translation RMS mm</th>'
                       '<th>neighbor RMS mm</th><th>axis ratio min–max</th><th>volume ratio min–max</th></tr>')
            for h, diagnostic in enumerate(values["diagnostics"]):
                detail += (f'<tr><td>{h+1}</td><td>{values["cd_mm"][h]:.4f} / {baseline["cd_mm"][h]:.4f}</td>'
                           f'<td>{values["rgb"][h]:.5f} / {baseline["rgb"][h]:.5f}</td>'
                           f'<td>{values["alpha"][h]:.5f} / {baseline["alpha"][h]:.5f}</td>'
                           f'<td>{", ".join(f"{v:.3f}" for v in diagnostic["mean_displacement_mm"])}</td>'
                           f'<td>{diagnostic["translation_rms_mm"]:.4f}</td><td>{diagnostic["neighbor_rms_mm"]:.4f}</td>'
                           f'<td>{"–".join(f"{v:.3f}" for v in diagnostic["axis_ratio"])}</td>'
                           f'<td>{"–".join(f"{v:.3f}" for v in diagnostic["volume_ratio"])}</td></tr>')
            detail += '</table>'
    return (f'<!doctype html><meta charset="utf-8"><title>{stage} fixed400</title>'
            '<style>body{font:15px system-ui;margin:30px}td,th{padding:8px;border-bottom:1px solid #ddd}</style>'
            f'<h1>{stage} · H={horizon} · {completed}/400 optimizer updates</h1><p>Online conditions t..t+H-1; targets t+1..t+H.</p><p>{best_name} update: '
            f'{None if best is None else best["update"]} · manual structural review pending.</p>'
            '<p>Fixed training windows, not held-out validation. No GT particle IDs. '
            'Positive = audited target_depth_valid; Negative = none; other pixels = Ignore.</p>'
            '<p>Curve: two-window mean CD (mm), x = optimizer updates. Full per-window/horizon '
            'metrics and covariance diagnostics: <a href="metrics.json">metrics.json</a>. '
            'Final checkpoint comparison: <a href="fit_summary.json">fit_summary.json</a>.</p>'
            f'<svg viewBox="0 0 800 220" width="800"><polyline fill="none" stroke="#246cc7" '
            f'stroke-width="2" points="{" ".join(points)}"/></svg><table><tr><th>update</th>'
            f'<th>CD mean mm</th><th>CD h{horizon} mm</th><th>RGB</th><th>Alpha</th><th>CD h1–h{horizon} mm</th><th>eligible</th>'
            '<th>automatic checks</th></tr>'+''.join(rows)+'</table>'+detail)


def run(args):
    """CUDA entry only; CPU tests exercise helpers without launching an experiment."""
    if args.resume and args.run_id:
        raise ValueError("Resume uses its existing run directory")
    if args.overfit_windows or args.max_train_batches or args.max_val_batches:
        raise ValueError("fixed400 uses explicit Roadmap windows; sampling limits are forbidden")
    if args.epochs != 20 or args.single_epochs != 1:
        raise ValueError("Do not pass epoch/curriculum overrides to fixed400; it counts 400 updates directly")
    expected = dict(seed=42, batch_size=1, lr=1e-4, weight_decay=1e-4, depth_weight=0.,
                    width=128, heads=4, layers=2, rgb_weight=1., alpha_weight=.1, scale_weight=.01)
    if any(getattr(args, key) != value for key, value in expected.items()):
        raise ValueError("P0 keeps seed42, batch1, AdamW lr/decay=1e-4 and the existing model/loss settings")
    optional = dict(controller_features=7, prediction_mode="autoregressive", output_mode="gs_geometry",
                    gs_local_attention="knn", gs_local_k=16, chamfer_weight=1.)
    if any(getattr(args, key) not in (None, value) for key, value in optional.items()):
        raise ValueError("P0/P1a requires the current 7D/k16/AR GS settings and lambda=1")
    if args.workers < 0 or args.chamfer_chunk < 1:
        raise ValueError("Invalid worker count or Chamfer chunk")
    state = read_checkpoint(args.resume) if args.resume else None
    if state is not None and (Path(args.resume).name != "last.pt" or state["config"].get("protocol") != "fixed400"):
        raise ValueError("fixed400 resumes only its own last.pt; legacy best/last semantics are unchanged")
    horizon = resolve_horizon(getattr(args, "horizon", None), state)
    selected_windows, window_evidence = frozen_windows(getattr(args, "window_manifest", None), args.scene, horizon, state)
    decoder = resolve_decoder(args.gs_decoder, args.anchor_count, args.anchor_pool_k,
                              args.anchor_bandwidth, state)
    if any(decoder[key] != value for key, value in DECODER_DEFAULTS.items() if key != "gs_decoder"):
        raise ValueError("Paired P1a fixes anchor_count=128, anchor_pool_k=16 and anchor_bandwidth=1")
    contract = (json.loads(Path(args.fit_contract).read_text(encoding="utf-8")) if args.fit_contract else
                state["config"]["fit_contract"] if state else None)
    diagnostic = args.fit_diagnostic or bool(state and state["config"].get("fit_diagnostic"))
    if diagnostic and contract is not None:
        raise ValueError("Diagnostic mode must not use acceptance thresholds")
    if contract is None and not diagnostic:
        raise ValueError("A reviewed --fit-contract is required; see fixed_fit module docstring for schema")
    if contract is not None and horizon != 5:
        raise ValueError("Legacy fit tolerances apply only to H=5; extended horizons require diagnostic mode")
    if contract is not None:
        validate_contract(contract, args.scene)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("P0 training requires a server CUDA allocation")
    import gsplat
    root = Path(args.root).resolve()
    depth_root = Path(args.depth_root).resolve() if args.depth_root else root.parent/"touch4d-depth-v1"
    cameras = choose_cameras(args.scene, depth_root, args.depth_cameras or GS_CAMERAS[args.scene])
    if cameras != GS_CAMERAS[args.scene]:
        raise ValueError("P0 freezes the audited three-camera order")
    config = dict(expected, **{k: v for k, v in optional.items() if k not in expected}, **decoder,
        gs_structure=resolve_structure(args.gs_structure, state), horizon=horizon,
        scene=args.scene, root=str(root), depth_root=str(depth_root), depth_cameras=cameras,
        workers=args.workers, chamfer_chunk=args.chamfer_chunk, overfit_windows=0, protocol="fixed400",
        fit_contract=contract, fit_diagnostic=diagnostic, gs_convention=GS_CONVENTION, render_settings=gs_render_settings())
    if selected_windows is not None:
        config.update(fixed_windows=selected_windows, window_evidence=window_evidence)
    validate_structure_config(config)
    raws = fixed_batches(config)
    evidence = manifest(raws, config)
    print(json.dumps(dict(exact_windows=evidence, updates=UPDATES, evaluation_every=EVAL_EVERY), indent=2), flush=True)
    torch.manual_seed(42); random.seed(42)
    processor = DataProcessor(args.scene)
    model = PositionPredictor(**{k: config[k] for k in MODEL_KEYS}).to(device)
    config["initial_model_sha256"] = digest(model.state_dict())
    # Include uncommitted/new Python sources as well as the Git identity. A paired
    # experiment must be attributable even before a future authorized commit.
    code = {str(path.relative_to(PROJECT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ("training", "dataloader", "data_process")
            for path in sorted((PROJECT/"scripts"/directory).glob("*.py"))
            if not path.name.startswith("test_")}
    config["source_sha256"] = digest(code)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batches = [processor(to_device(raw, device)) for raw in raws]
    renderer = TargetCache(make_renderer(config))
    neighbors = [initial_knn(b["inputs"]["gs_xyz"], b["inputs"]["gs_mask"], 16) for b in batches]
    start, best, history = 0, None, []
    if state:
        if (state["config"] != config or state["processor"] != processor.state_dict()
                or state["fit"]["manifest"] != evidence):
            raise ValueError("Cannot resume: configuration, thresholds, initial weights or actual data changed")
        run_dir = Path(args.resume).resolve().parent
        best = recover_fit_best(state, run_dir)
        start, history = state["fit"]["completed_updates"], state["history"]
        if [row["update"] for row in history] != list(range(EVAL_EVERY, start+1, EVAL_EVERY)):
            raise ValueError("Cannot resume: incomplete evaluation history")
        restore_training(state, model, optimizer)
    else:
        name = args.run_id or datetime.now(timezone.utc).strftime("p0-%Y%m%dT%H%M%SZ-")+uuid.uuid4().hex[:8]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name):
            raise ValueError("run-id must contain ASCII letters, digits, hyphens or underscores")
        run_dir = PROJECT/"runs"/args.scene/name
        print(f"Creating {run_dir}: config.json, metrics.json, report.html, fit_summary.json, "
              "best_fit.pt (reviewed contract only) OR lowest_cd.pt + step_{000020..000400}.pt (diagnostic), "
              "last.pt, last_400.pt; atomic .tmp staging; "
              "visualizations/{best_fit,lowest_cd,last_400}/episode-{id}-start-{frame}/ with prediction.npz, "
              "index.html, geometry.html and figures/*.png", flush=True)
        run_dir.mkdir(parents=True, exist_ok=False)
    try:
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
        dirty = subprocess.check_output(["git", "diff", "--", "scripts"], cwd=PROJECT)
    except (OSError, subprocess.CalledProcessError):
        source, dirty = None, b""
    if not state:
        atomic_json(run_dir/"config.json", dict(training=config, manifest=evidence, processor=processor.state_dict(),
                    commit=source, tracked_code_diff_sha256=hashlib.sha256(dirty).hexdigest(),
                    torch=torch.__version__, gsplat=gsplat.__version__))
    evaluate = lambda: evaluate_fixed({"model": model}, batches, processor, renderer, config, neighbors)
    def save(update, metrics, loss, grad_norm):
        nonlocal best
        row = candidate(metrics, contract, update)
        improved = better(row, best, contract)
        if improved:
            best = row
        history.append(dict(update=update, window_updates=[update//2]*2, last_train_loss=loss,
                            grad_norm_before_clip=grad_norm, metrics=metrics, candidate=row))
        checkpoint = checkpoint_state(model, optimizer, processor, config, update//2-1,
                                      best["rank"][0] if best else float("inf"), history)
        checkpoint["fit"] = dict(completed_updates=update, best_candidate=best, manifest=evidence)
        publish_fit(run_dir, checkpoint, improved)
        atomic_json(run_dir/"metrics.json", history)
        (run_dir/"report.html").write_text(report_html(history, best, update, config["gs_decoder"], config["gs_structure"], horizon), encoding="utf-8")
        print(json.dumps(dict(update=update, candidate=row), allow_nan=False), flush=True)
    update_loop(model, optimizer, batches, processor, renderer, config, save, start, evaluate)
    # Also repairs a crash after last.pt(400) but before last_400.pt/export publication.
    last = read_checkpoint(run_dir/"last.pt")
    if last["fit"]["completed_updates"] != UPDATES:
        raise ValueError("Missing completed 400-update state")
    atomic_checkpoint(run_dir/"last_400.pt", last)
    from .visualize import export_gaussians
    summary = dict(scene=args.scene, horizon=horizon, prediction_protocol="online conditions t..t+H-1; targets t+1..t+H",
                   completed_updates=UPDATES, windows=evidence,
                   gs_decoder=config["gs_decoder"],
                   gs_structure=config["gs_structure"],
                   inherited_shape=config["gs_decoder"] != "free",
                   acceptance="not assessed: tolerances not frozen" if diagnostic else "manual structural review pending",
                   checkpoints={})
    selected_name = "lowest_cd" if diagnostic else "best_fit"
    for name in (selected_name, "last_400"):
        path = run_dir/(name+".pt")
        if not path.is_file():
            summary["checkpoints"][name] = dict(status="no eligible checkpoint; fit not accepted")
            continue
        saved = read_checkpoint(path)
        model.load_state_dict(saved["model"]); model.eval()
        metrics = evaluate()
        selected = candidate(metrics, contract, saved["fit"]["completed_updates"])
        summary["checkpoints"][name] = dict(candidate=selected, metrics=metrics, visualizations=[])
        with torch.no_grad():
            for (episode, frame), raw, batch in zip(config.get("fixed_windows", WINDOWS[args.scene]), raws, batches):
                directory = run_dir/"visualizations"/name/f"episode-{episode}-start-{frame}"
                # A failed/partial export must not be mistaken for a complete artifact.
                if directory.exists():
                    if not all((directory/f).is_file() for f in ("index.html", "geometry.html", "prediction.npz")):
                        raise ValueError(f"Incomplete visualization export; preserve and inspect {directory}")
                else:
                    export_gaussians(SimpleNamespace(output=str(directory), checkpoint=str(path), split="train",
                        episode=episode, start=frame), config, raw, batch, model(batch["inputs"]), processor)
                summary["checkpoints"][name]["visualizations"].append(str(directory/"index.html"))
    atomic_json(run_dir/"metrics.json", history)
    final_checks = summary["checkpoints"]["last_400"]["candidate"]
    best_checks = summary["checkpoints"].get(selected_name, {}).get("candidate")
    summary["stability"] = dict(
        best_update=None if best_checks is None else best_checks["update"],
        last_400_worse_than_best=bool(best_checks and
                                    ((not diagnostic and not final_checks["eligible"]) or better(best_checks, final_checks, contract))),
        earlier_automatic_fit_but_last_failed=None if diagnostic else
            any(r["candidate"]["automatic_fit_checks"] for r in history) and not final_checks["automatic_fit_checks"],
        interpretation="Automatic checks are evidence only. If an earlier checkpoint also passes manual "
                       "structure review and last_400 fails, report: 表达能力验收通过，训练稳定性尚未通过。")
    atomic_json(run_dir/"fit_summary.json", summary)
    page = report_html(history, best, UPDATES, config["gs_decoder"], config["gs_structure"], horizon)
    for name, result in summary["checkpoints"].items():
        for path in result.get("visualizations", []):
            relative = Path(path).relative_to(run_dir).as_posix()
            page += f'<p><a href="{html.escape(relative)}">{html.escape(name+" · "+relative)}</a></p>'
    (run_dir/"report.html").write_text(page, encoding="utf-8")
    return 0
