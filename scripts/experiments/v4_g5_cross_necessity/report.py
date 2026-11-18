"""CPU-only reporting for a COMPLETED frozen G5 necessity run.

No model/renderer/optimizer is imported.  Plots use saved world XYZ and numeric
records; they are geometry diagnostics, never fresh Gaussian renders.  All new
files use exclusive creation.  Automatic numeric reporting is not an assertion
that a human reviewed the rendered evidence or approved Stage 2.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .gates import EPISODE_COUNTS, REVIEW_FLAGS, evaluate_gate


BASE = "f978e0a29b6f0f0b6097c40684b949797a2f5d7b"
CHECKPOINT_HASH = "067984c4c286306094efeff2f28ed57e3dcdd0c333972ecc6feb4b982ee3a40c"
MODES = (
    "Normal", "NoBase-All", "NoCrossValue", "NoToolMotion12",
    "NoBase+NoCrossValue", "NoBase+NoToolMotion12",
    "NoCrossValue+NoToolMotion12", "NoBase+NoAttentionPaths",
    "NoAttentionPaths", "NoBase-H2+",
)
JOINT = "NoBase+NoAttentionPaths"
CASES = (
    ("val", 3, 135), ("val", 3, 245), ("val", 6, 110), ("train", 7, 105),
    ("val", 3, 25), ("val", 6, 20), ("val", 6, 200),
)
GEOMETRY_CASES = CASES[:4]
DISPLAY = ("GT", "Copy", "Normal", "NoBase-All", "NoAttentionPaths", JOINT)
REQUIRED = (
    "val_summary", "horizons", "same_history", "branch_fields", "cancellation",
    "structure", "action_response", "same_history_regions", "gate_preliminary",
)
FIGURES = (
    "horizon_metrics.png", "same_history_effects.png", "branch_motion.png",
    "cancellation.png", "action_response.png",
    *(f"{split}_ep{ep}_s{start}_h15_geometry.png" for split, ep, start in GEOMETRY_CASES),
)
NEW_FILES = (
    "frozen_attribution_report.md", "artifact_manifest.json", "review_template.json",
    *(f"figures/{name}" for name in FIGURES),
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    def reject(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    with Path(path).open(encoding="utf8") as stream:
        result = json.load(stream, parse_constant=reject)
    _finite(result)
    return result


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite numeric record")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def _write_json(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _fmt(value, digits=5):
    if value is None:
        return "未定义/未记录"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    return ("| " + " | ".join(headers) + " |\n| "
            + " | ".join("---" for _ in headers) + " |\n"
            + "".join("| " + " | ".join(_fmt(v) for v in row) + " |\n" for row in rows)
            + "\n")


def _key(case):
    split, ep, start = case
    return f"{split}_ep{ep}_s{start}"


def _matches(row, case):
    return (row.get("split"), int(row.get("episode", -1)), int(row.get("start", -1))) == case


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return math.fsum(values)/len(values) if values else None


def _episode_mean(rows, key):
    """Windows equal within each episode, then observed Val episodes equal."""
    episodes = defaultdict(list)
    for row in rows:
        if row.get("split") == "val" and row.get(key) is not None:
            episodes[str(row["episode"])].append(float(row[key]))
    means = [_mean(items) for items in episodes.values() if items]
    return _mean(means)


def _response_available(row):
    return row.get("available", True) is True and row.get("output_change_rms_mm") is not None


def _defined_windows(rows, key):
    counts = {ep: len({r["start"] for r in rows if r.get("split") == "val"
                       and str(r.get("episode")) == ep and r.get(key) is not None})
              for ep in EPISODE_COUNTS}
    return f"ep3={counts['3']}; ep6={counts['6']}"


def _summary_metrics(values):
    joint = values.get("objective", values.get("joint"))
    if joint is None:
        normalized = values.get("supervised_geometry", values.get("geometry"))
        if normalized is None:
            raise ValueError("Missing normalized CD for Joint")
        joint = normalized + values["rgb"] + .1*values["alpha"]
    return (values["geometry_mm"], values["rgb"], values["alpha"], joint)


def _image_refs():
    result = []
    for case in CASES:
        for camera in range(3):
            for mode in DISPLAY:
                for modality in ("rgb", "alpha"):
                    result.append(f"images/{_key(case)}/h15_v{camera}_{modality}_{mode}.png")
        for h in (1, 5, 10):
            for mode in ("GT", "Normal", JOINT):
                for modality in ("rgb", "alpha"):
                    result.append(f"images/{_key(case)}/h{h:02}_v0_{modality}_{mode}.png")
    return result


def expected_images():
    """All seven fixed cases, four H, three cameras, RGB/Alpha, and 12 modes."""
    return [f"images/{_key(case)}/h{h:02}_v{camera}_{modality}_{mode}.png"
            for case in CASES for h in (1, 5, 10, 15) for camera in range(3)
            for modality in ("rgb", "alpha") for mode in ("GT", "Copy", *MODES)]


def _scope(row):
    try:
        split, episode, start = row["split"], row["episode"], row["start"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing window scope") from exc
    if split not in ("val", "train") or type(episode) is not int or type(start) is not int or start < 0:
        raise ValueError("Invalid exact window scope")
    return split, episode, start


def _row_key(row):
    scope = _scope(row)
    h = row.get("horizon")
    if type(h) is not int or not 1 <= h <= 15 or not isinstance(row.get("mode"), str):
        raise ValueError("Invalid exact mode/horizon key")
    return (*scope, h, row["mode"])


def _exact_coverage(rows, expected, key, label):
    keys = [key(row) for row in rows]
    observed = set(keys)
    if len(observed) != len(keys) or observed != expected:
        raise ValueError(f"{label}: exact coverage mismatch "
                         f"(duplicate={len(keys)-len(observed)}, missing={len(expected-observed)}, "
                         f"extra={len(observed-expected)})")


def _coverage(data, out):
    """Only completeness/invariance guards; scientific gate thresholds unchanged."""
    normal = [row for row in data["horizons"] if row.get("mode") == "Normal"]
    windows = {_scope(row) for row in normal}
    if (len(windows) != 101 or {scope for scope in windows if scope[0] == "train"} != {("train", 7, 105)}
            or sum(scope[0] == "val" for scope in windows) != 100):
        raise ValueError("Expected exactly Val100 and Train ep7/start105 Normal windows")
    _exact_coverage(normal, {(*scope, h, "Normal") for scope in windows for h in range(1, 16)},
                    _row_key, "Normal history")
    expected = {(*scope, h, mode) for scope in windows for h in (1, 5, 10, 15) for mode in MODES[1:]}
    _exact_coverage(data["same_history"], expected, _row_key, "same_history")
    _exact_coverage(data["same_history_regions"],
                    {(*key, region) for key in expected for region in ("all", "near", "middle", "far")},
                    lambda row: (*_row_key(row), row.get("region")), "same_history_regions")
    responses = {(*case, h, mode, response) for case in CASES for h in range(1, 16)
                 for mode in ("Normal", JOINT) for response in ("FreezeAction", "HalfRootTranslation")}
    _exact_coverage(data["action_response"], responses,
                    lambda row: (*_row_key(row), row.get("response")), "action_response")
    for row in data["action_response"]:
        if type(row.get("available", True)) is not bool:
            raise ValueError("Action availability must be bool")
        if row.get("available", True) and row.get("output_change_rms_mm") is None:
            raise ValueError("Available action response is missing its observed output change")
    checks_path = out/"results"/"intervention_checks.json"
    if checks_path.exists():
        checks = _json(checks_path)
        if not isinstance(checks, list):
            raise ValueError("intervention_checks must be a list")
        expected_checks = {(*case, h, mode) for case in CASES for h in (1, 5, 10, 15) for mode in MODES[1:]}
        _exact_coverage(checks, expected_checks, _row_key, "intervention_checks")
        if any(row.get("prefix_bit_exact") is not True or row.get("current_step_bit_exact") is not True
               for row in checks):
            raise ValueError("A recorded intervention prefix/current-step invariant failed")
        data["intervention_checks"] = checks
    return dict(normal_windows=len(windows), same_history=len(expected),
                same_history_regions=len(expected)*4, action_response=len(responses),
                intervention_checks=(len(data["intervention_checks"]) if "intervention_checks" in data else None),
                expected_renderings=len(expected_images()))


def _read_inputs(out):
    data = {name: _json(out/"results"/f"{name}.json") for name in REQUIRED}
    provenance, completed = _json(out/"provenance.json"), _json(out/"completed.json")
    if completed.get("state") != "INFERENCE_COMPLETED" or completed.get("val100") is not True:
        raise ValueError("Requires genuinely completed full-Val inference")
    if any(completed.get(k) is not True for k in ("checkpoint_unchanged", "model_unchanged")):
        raise ValueError("Frozen invariance checks are missing or failed")
    if completed.get("source_sha") != provenance.get("diagnostic_source_sha"):
        raise ValueError("Completion/source identity mismatch")
    if provenance.get("training_source_sha") != BASE:
        raise ValueError("Wrong formal training source")
    if (provenance.get("checkpoint_sha256") != CHECKPOINT_HASH
            or provenance.get("checkpoint_hash_after") != CHECKPOINT_HASH):
        raise ValueError("Wrong or changed frozen checkpoint")
    if not provenance.get("model_hash_before") or provenance["model_hash_before"] != provenance.get("model_hash_after"):
        raise ValueError("Model state changed or its hash evidence is missing")
    if (provenance.get("optimizer_created") is not False or provenance.get("backward_called") is not False
            or provenance.get("optimizer_updates") != 0 or provenance.get("parameters_frozen") is not True):
        raise ValueError("Not a frozen, no-training run")
    if not provenance.get("job_id") or not provenance.get("gpu"):
        raise ValueError("Missing recorded Slurm/GPU identity")
    if set((*MODES, "Copy")) != set(data["val_summary"]):
        raise ValueError("Expected the complete factorial, strict closure, H2+, and Copy modes")
    for name in REQUIRED:
        if name not in ("val_summary", "gate_preliminary"):
            if not isinstance(data[name], list) or not data[name]:
                raise ValueError("Missing nonempty raw records: " + name)
            if any(r.get("split") not in ("val", "train") for r in data[name]):
                raise ValueError("Unexpected split in " + name)
    val = [r for r in data["horizons"] if r["split"] == "val"]
    gate = evaluate_gate(data["val_summary"], val)
    recorded = data["gate_preliminary"]
    for name in ("status", "hard_gate_passed", "conservative_numeric_passed", "eligible_for_stage2"):
        if gate[name] != recorded.get(name):
            raise ValueError("Recorded/recomputed preliminary gate mismatch: " + name)
    for metric, tolerance in (("geometry_mm", 1e-4), ("rgb", 1e-6), ("alpha", 1e-6), ("objective", 1e-6)):
        expected = provenance.get("expected_frozen_validation", {}).get(metric)
        if expected is None or abs(data["val_summary"]["Normal"][metric]-expected) > tolerance:
            raise ValueError("Normal is not a faithful reproduction: " + metric)
    coverage = _coverage(data, out)
    geometry = {}
    for case in CASES:
        path = out/"arrays"/f"{_key(case)}.npz"
        with np.load(path, allow_pickle=False) as saved:
            initial = np.asarray(saved["initial_world"])
            frame_ids = np.asarray(saved["frame_ids"])
            if initial.ndim != 2 or initial.shape[-1] != 3 or not len(initial):
                raise ValueError("Bad initial world geometry: " + str(path))
            if not np.array_equal(frame_ids, np.arange(case[2], case[2]+16)):
                raise ValueError("Wrong original frame IDs: " + str(path))
            arrays = {"initial_world": initial.copy(), "frame_ids": frame_ids.copy()}
            for mode in (*MODES, "Copy"):
                value = np.asarray(saved[mode+"_world"])
                if value.shape != (15, len(initial), 3) or not np.isfinite(value).all():
                    raise ValueError("Invalid saved world rollout: " + mode)
                arrays[mode+"_world"] = value.copy()
            for h in range(1, 16):
                name = f"gt_h{h:02}_world"
                value = np.asarray(saved[name])
                if value.ndim != 2 or value.shape[-1] != 3 or not len(value) or not np.isfinite(value).all():
                    raise ValueError("Invalid independent GT point set: " + name)
                arrays[name] = value.copy()
            if not np.isfinite(initial).all():
                raise ValueError("Nonfinite initial world geometry")
            regions = []
            for region in ("near", "middle", "far"):
                name = "region_"+region
                ids = np.asarray(saved[name])
                if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= len(initial))):
                    raise ValueError("Invalid compact region indices")
                arrays[name] = ids.copy()
                regions.extend(ids.tolist())
            if sorted(regions) != list(range(len(initial))):
                raise ValueError("Near/Middle/Far must partition valid initial GS")
            geometry[_key(case)] = arrays
        for h in (1, 5, 10, 15):
            for mode in MODES:
                if not any(_matches(r, case) and r["horizon"] == h and r["mode"] == mode
                           for r in data["horizons"]):
                    raise ValueError("Missing fixed case/horizon/mode evidence")
    for relative in expected_images():
        if not (out/relative).is_file() or (out/relative).stat().st_size == 0:
            raise ValueError("Missing saved rendering: " + relative)
    return dict(data, provenance=provenance, completed=completed, gate=gate, geometry=geometry, coverage=coverage)


def _plot_line(ax, rows, key, modes, *, delta=False):
    for mode in modes:
        selected = [r for r in rows if r.get("mode") == mode and r.get("split") == "val"]
        hs = sorted({r["horizon"] for r in selected})
        values = [_episode_mean([r for r in selected if r["horizon"] == h], key) for h in hs]
        ax.plot(hs, [np.nan if v is None else v for v in values], label=mode, linewidth=1.3)
    if delta:
        ax.axhline(0, color="#777777", linewidth=.8)
    ax.set_xlabel("Horizon")
    ax.set_ylabel(key)
    ax.grid(alpha=.2)


def _save_figure(figure, path, plt):
    with Path(path).open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def _world_bounds(arrays):
    clouds = [arrays["initial_world"]]
    clouds += [arrays[mode+"_world"].reshape(-1, 3) for mode in (*MODES, "Copy")]
    clouds += [arrays[f"gt_h{h:02}_world"] for h in range(1, 16)]
    joined = np.concatenate(clouds)*1000
    lower, upper = joined.min(0), joined.max(0)
    pad = np.maximum((upper-lower)*.04, .5)
    return lower-pad, upper+pad


def _make_figures(out, evidence):
    """Plot existing arrays/statistics on CPU; never Gaussian rasterization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = evidence
    focus = ("Normal", "Copy", "NoBase-All", "NoBase-H2+", "NoCrossValue",
             "NoToolMotion12", "NoAttentionPaths", JOINT)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, metric in zip(axes.flat, ("cd_mm", "rgb", "alpha", "joint")):
        _plot_line(ax, data["horizons"], metric, focus)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Full AR Val100: windows equal within episode, episodes equal")
    fig.tight_layout()
    _save_figure(fig, out/"figures"/FIGURES[0], plt)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric in zip(axes, ("delta_cd_mm", "delta_joint")):
        _plot_line(ax, data["same_history"], metric, MODES[1:], delta=True)
    axes[0].legend(fontsize=7)
    fig.suptitle("Current-step intervention on fixed Normal history; not full AR")
    fig.tight_layout()
    _save_figure(fig, out/"figures"/FIGURES[1], plt)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    branch_names = ("base", "global_", "local", "correction", "final")
    for row, space in enumerate(("anchor_control", "gs_actual")):
        for col, metric in enumerate(("rms_mm", "mean_x_mm", "mean_y_mm", "mean_z_mm")):
            ax = axes[row, col]
            for branch in branch_names:
                selected = [r for r in data["branch_fields"] if r["space"] == space
                            and r["region"] == "all" and r["branch"] == branch and r["split"] == "val"]
                hs = sorted({r["horizon"] for r in selected})
                ax.plot(hs, [_episode_mean([r for r in selected if r["horizon"] == h], metric) for h in hs],
                        label=branch, linewidth=1.3)
            ax.set_title(space+" / "+metric)
            ax.set_xlabel("Horizon")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Normal history branch fields: Anchor controls and actual RBF GS motion")
    fig.tight_layout()
    _save_figure(fig, out/"figures"/FIGURES[2], plt)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, space in enumerate(("anchor_control", "gs_actual")):
        for col, metric in enumerate(("cancellation_index", "base_vs_rest_opposite_fraction")):
            ax = axes[row, col]
            for region in ("all", "near", "middle", "far"):
                selected = [r for r in data["cancellation"] if r["space"] == space
                            and r["region"] == region and r["split"] == "val"]
                hs = sorted({r["horizon"] for r in selected})
                values = [_episode_mean([r for r in selected if r["horizon"] == h], metric) for h in hs]
                ax.plot(hs, [np.nan if v is None else v for v in values], label=region)
            ax.set_title(space+" / "+metric)
            ax.set_xlabel("Horizon")
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle("Cancellation is descriptive, not mechanical energy or causal usefulness")
    fig.tight_layout()
    _save_figure(fig, out/"figures"/FIGURES[3], plt)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, case in zip(axes.flat, CASES):
        for branch in ("Normal", JOINT):
            for response in ("FreezeAction", "HalfRootTranslation"):
                selected = sorted([r for r in data["action_response"] if _matches(r, case)
                                   and r["mode"] == branch and r["response"] == response
                                   and _response_available(r)],
                                  key=lambda r: r["horizon"])
                ax.plot([r["horizon"] for r in selected], [r["output_change_rms_mm"] for r in selected],
                        label=branch+" / "+response)
        ax.set_title(_key(case))
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Output change from same branch / mm")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(CASES):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle("Action sensitivity only: no paired counterfactual ground truth")
    fig.tight_layout()
    _save_figure(fig, out/"figures"/FIGURES[4], plt)

    for case in GEOMETRY_CASES:
        arrays = data["geometry"][_key(case)]
        # One shared world range includes ALL saved factual modes, all H, and GT.
        lower, upper = _world_bounds(arrays)
        colors = np.full((len(arrays["initial_world"]),), "#7b8fa4", dtype="<U7")
        colors[arrays["region_near"]] = "#c75555"
        colors[arrays["region_middle"]] = "#7b8fa4"
        colors[arrays["region_far"]] = "#4f8764"
        fig = plt.figure(figsize=(16, 9))
        for index, mode in enumerate(DISPLAY, 1):
            ax = fig.add_subplot(2, 3, index, projection="3d")
            cloud = (arrays["gt_h15_world"] if mode == "GT" else arrays[mode+"_world"][14])*1000
            ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=1.5,
                       c="#686868" if mode == "GT" else colors, alpha=.75, rasterized=True)
            ax.set_xlim(lower[0], upper[0]); ax.set_ylim(lower[1], upper[1]); ax.set_zlim(lower[2], upper[2])
            ax.set_box_aspect(upper-lower)
            ax.view_init(elev=22, azim=-60)
            ax.set_title(mode, fontsize=10)
            ax.set_xlabel("World X / mm"); ax.set_ylabel("World Y / mm"); ax.set_zlabel("World Z / mm")
        fig.suptitle(_key(case)+" / H15 / fixed world bounds; GT has no material correspondence\n"
                     "Prediction colors: initial Controller-distance Near(red)/Middle(blue)/Far(green)")
        fig.tight_layout(rect=(0, 0, 1, .95))
        _save_figure(fig, out/"figures"/f"{_key(case)}_h15_geometry.png", plt)


def _render_summary_table(summary):
    rows = []
    for mode in (*MODES, "Copy"):
        values = summary[mode]
        for label, selected in (("Val100 Episode等权", values),
                                ("ep3 / 55窗", values["episodes"]["3"]),
                                ("ep6 / 45窗", values["episodes"]["6"])):
            rows.append((mode, label, *_summary_metrics(selected)))
    return _table(("模式", "范围", "CD mm", "RGB", "Alpha", "Joint"), rows)


def _image_index(case):
    key = _key(case)
    def links(h, camera, mode):
        return (f"[RGB](images/{key}/h{h:02}_v{camera}_rgb_{mode}.png) / "
                f"[Alpha](images/{key}/h{h:02}_v{camera}_alpha_{mode}.png)")
    rows = [(f"H15 / cam{camera}", *(links(15, camera, mode) for mode in DISPLAY))
            for camera in range(3)]
    intermediate = [(f"H{h} / cam0", *(links(h, 0, mode) for mode in ("GT", "Normal", JOINT)))
                    for h in (1, 5, 10)]
    return (_table(("帧/视角", *DISPLAY), rows)
            + _table(("前中期 / cam0", "GT", "Normal", JOINT), intermediate))


def _report(out, e):
    gate, summary, p = e["gate"], e["val_summary"], e["provenance"]
    synthetic = p.get("run_kind") == "SYNTHETIC_TEST_ONLY"
    pieces = ["# Touch4D G5 Motion Head / Cross-Attention 冻结必要性实验\n\n"]
    if synthetic:
        pieces.append("**SYNTHETIC_TEST_ONLY：这是临时单元测试夹具，不是真实 GPU 实验结果，不得用于研究判断。**\n\n")
    if gate["status"] == "FAIL_STAGE1":
        conclusion = "阶段一停止：联合消融未满足预注册完整 Val100 数值门槛，不进入架构重构或新训练。"
    else:
        conclusion = ("阶段一数值记录已完成，但未完成独立视觉、结构与动作响应审查；"
                      "状态 INCONCLUSIVE_NO_AUTOSTART，不自动新建简化模型或提交训练。")
    pieces += ["## 1. 结论与边界\n\n", conclusion+"\n\n",
               "本报告由 CPU 工具读取已完成的冻结推理记录生成；不是人工目视审查的替代品。"
               "没有训练、优化器更新、反向传播、权重修改或新 GS 渲染。图表只使用保存的 XYZ/JSON；"
               "原始三相机 RGB/Alpha 以链接供独立复核。负面结果与所有预设模式、七个案例均保留。\n\n",
               f"- 预筛状态：**{gate['status']}**。\n"
               "- 研究对象是同一个 G5-Direct Epoch8 冻结 Checkpoint；内部 Base 与 G4/G5 共同训练，"
               "不是历史 G0 Checkpoint。NoBase 不能称为“换回原 G0”。\n"
               "- 冻结删减改变状态分布。它只能回答当前权重的依赖，不能证明从头重训的简化模型"
               "能达到相同能力，也不能证明 Attention 在理论上无用。\n"
               "- H1 的 G4/G5 显式运动基底为零，NoBase-All 首步为 Copy；必须与 NoBase-H2+ 分开看。\n\n",
               "## 2. 固定来源与执行完整性\n\n",
               _table(("项目", "记录"), (
                   ("运行绝对路径", str(out)),
                   ("训练源码 SHA", p["training_source_sha"]),
                   ("诊断源码 SHA", p["diagnostic_source_sha"]),
                   ("Slurm Job ID", p["job_id"]), ("GPU", p["gpu"]),
                   ("PyTorch / CUDA / gsplat", f"{p.get('torch')} / {p.get('cuda')} / {p.get('gsplat')}"),
                   ("Checkpoint 绝对路径", p.get("checkpoint")),
                   ("Checkpoint SHA256 前/后", f"{p['checkpoint_sha256']} / {p['checkpoint_hash_after']}"),
                   ("模型状态 SHA256 前/后", f"{p['model_hash_before']} / {p['model_hash_after']}"),
                   ("参数量", p.get("parameter_count")),
                   ("优化更新数", p["optimizer_updates"]),
                   ("耗时秒 / 显存峰值字节", f"{p.get('elapsed_seconds')} / {p.get('peak_allocated_bytes')}"),
                   ("实际命令", p.get("command")),
               )),
               "原始完整 Val100：ep3=55、ep6=45；窗内/窗间按原协议聚合，最终 Episode 等权。"
               "Train ep7/start105 仅作额外诊断，未混入 Val 选模。无测试集。"
               "Normal 与 Checkpoint 原验证 CD/RGB/Alpha/Joint 按原容差复现；"
               "completed.json 声明推理完成、模型和 Checkpoint 不变，报告工具重新验证来源与完整 Val 门槛。\n\n",
               _table(("完整性守卫", "实际完整数量"), e["coverage"].items()),
               "原始图像检查覆盖全部七案例×4时刻×3相机×RGB/Alpha×GT/Copy/10模式，"
               "共2016张，包含本文未展示的图；存在性与文件哈希不等同于人工视觉通过。"
               "单步、区域与动作响应按精确窗口/时刻/模式键核对，缺失、重复和额外记录均拒绝。"
               "不对Anchor近端/远端空分组虚构节点或要求其必须存在。\n\n",
               "## 3. 消融语义与完整 Val100 结果\n\n",
               "三因素为：B=G0 最终3D输出、C=两层 Cross Value 更新、T=G0输入12D Tool Motion。"
               "Score/Mask/Null 继续计算，FFN 保留；逐 Anchor 几何 Beta 不与原 Cross Alpha 混同。"
               "完整8格中的 NoCrossValue+NoToolMotion12 **仍保留4D head_mass**。"
               "额外 NoAttentionPaths 同时关闭4D mass；NoBase+NoAttentionPaths 也关闭它，"
               "但因为 Base 已关闭，此角点与只关 C/T 的结果应严格等价。\n\n",
               _render_summary_table(summary),
               "完整 AR 每个模式重新执行 H15，历史状态随干预自然变化。"
               "下图展示预先选择的主要模式；完整所有模式数字见 results/horizons.json 与 CSV。\n\n",
               "![完整AR曲线](figures/horizon_metrics.png)\n\n"]
    interaction_rows = []
    combinations = (
        ("Base×Value", ("NoBase+NoCrossValue", "NoBase-All", "NoCrossValue", "Normal"), (1, -1, -1, 1)),
        ("Base×Tool12", ("NoBase+NoToolMotion12", "NoBase-All", "NoToolMotion12", "Normal"), (1, -1, -1, 1)),
        ("Value×Tool12（mass保留）", ("NoCrossValue+NoToolMotion12", "NoCrossValue", "NoToolMotion12", "Normal"), (1, -1, -1, 1)),
        ("三因素差分", (JOINT, "NoBase+NoCrossValue", "NoBase+NoToolMotion12", "NoCrossValue+NoToolMotion12",
                          "NoBase-All", "NoCrossValue", "NoToolMotion12", "Normal"), (1, -1, -1, -1, 1, 1, 1, -1)),
    )
    for name, modes, signs in combinations:
        interaction_rows.append((name, *(math.fsum(sign*_summary_metrics(summary[mode])[index]
                                                   for mode, sign in zip(modes, signs))
                                         for index in (0, 3))))
    pieces += ["联合删除可能存在交互。下表是已冻结指标的有限差分，不是显著性检验，"
               "也不将相关窗口视作独立重复。严格 mass 关闭额外格不能偷偷取代单独12D控制。\n\n",
               _table(("交互描述", "CD差分 mm", "Joint差分"), interaction_rows),
               "## 4. 固定 Normal 历史的当前步直接效应\n\n",
               "在 H1/H5/H10/H15 固定 Normal 的 GS/Anchor/previous_d/Controller 历史，"
               "重算当前步两层 Cross、G0/G4/G5 与 RBF；Cross 改变后没有复用旧 global/local。"
               "这些当前步效果不能与完整 AR 后期变化混为一谈。RGB/Alpha/Joint 仍沿原相机和有效掩码。\n\n",
               (f"已读取七案例共{len(e['intervention_checks'])}条only-step前缀/当前步逐位一致校验，"
                "精确覆盖完整且全部为True。\n\n" if "intervention_checks" in e else
                "未提供可选intervention_checks.json：不能声称本工具独立复核了七案例的GPU逐位一致记录。\n\n"),
               "![单步效果](figures/same_history_effects.png)\n\n"]
    effect_rows = []
    for h in (1, 5, 10, 15):
        for mode in ("NoBase-All", "NoBase-H2+", "NoCrossValue", "NoToolMotion12", "NoAttentionPaths", JOINT):
            selected = [r for r in e["same_history"] if r["split"] == "val" and r["horizon"] == h and r["mode"] == mode]
            improved = _episode_mean(selected, "cd_improved")
            worsened = _episode_mean(selected, "cd_worsened")
            effect_rows.append((h, mode, _episode_mean(selected, "delta_cd_mm"),
                                _episode_mean(selected, "delta_joint"), _episode_mean(selected, "change_rms_mm"),
                                improved, worsened, None if improved is None or worsened is None else 1-improved-worsened))
    pieces += [_table(("H", "干预", "ΔCD mm", "ΔJoint", "输出变化RMS mm",
                       "CD改善窗比例", "CD恶化窗比例", "未变窗比例"), effect_rows),
               "比例先在各 Episode 内按窗口计算，再对两个 Episode 等权。改善/恶化沿 Runner 的"
               "逐窗1e−6 mm数值边界；不是独立样本显著性。相对 Normal 的差异为负才代表该指标改善。\n\n",
               "## 5. Normal 三分支、RBF 与抵消\n\n",
               "Anchor control 和 RBF 后实际 GS 位移分别报告。XYZ/向量范数均为世界 mm；"
               "各分支在同一 Normal 历史上分解。P95 先逐窗计算再 Episode 等权平均，"
               "不是把所有点混成一个全局分位数。H1–H15 完整原始数据保留于 branch_fields.json。\n\n",
               "![分支运动](figures/branch_motion.png)\n\n"]
    branch_rows = []
    for space in ("anchor_control", "gs_actual"):
        for region in ("all", "near", "middle", "far"):
            for branch in ("base", "global_", "local", "correction", "final"):
                selected = [r for r in e["branch_fields"] if r["split"] == "val" and r["space"] == space
                            and r["region"] == region and r["branch"] == branch and r["horizon"] == 15]
                branch_rows.append((space, region, branch,
                    *(_episode_mean(selected, k) for k in ("mean_x_mm", "mean_y_mm", "mean_z_mm", "rms_mm",
                       "norm_p95_mm", "signed_projection_mean_mm", "final_fraction_mean")),
                    _defined_windows(selected, "signed_projection_mean_mm")))
    pieces += [_table(("H15空间", "区域", "分支", "均值X", "均值Y", "均值Z", "RMS",
                       "窗P95均值", "对最终有符号投影mm", "投影比例", "投影定义窗数"), branch_rows),
               "![抵消曲线](figures/cancellation.png)\n\n"]
    cancel_rows = []
    for space in ("anchor_control", "gs_actual"):
        for region in ("all", "near", "middle", "far"):
            selected = [r for r in e["cancellation"] if r["split"] == "val" and r["space"] == space
                        and r["region"] == region and r["horizon"] == 15]
            cancel_rows.append((space, region, *(_episode_mean(selected, k) for k in (
                "base_vs_rest_cos_mean", "base_vs_rest_opposite_fraction", "cancellation_index",
                "base_to_final_rms", "base_larger_than_final_fraction", "nearly_cancelled_fraction")),
                _defined_windows(selected, "base_vs_rest_cos_mean")))
    pieces += [_table(("H15空间", "区域", "Base与修正余弦", "反向比例", "抵消指数",
                       "Base/Final RMS", "Base大于Final比例", "近乎抵消比例", "余弦定义窗数"), cancel_rows),
               "近零向量方向/投影为未定义，绝不填成0；原始定义/未定义数量保留在 JSON。"
               "负余弦或较高抵消指数不自动证明 Base 是噪声：纠正错误基础运动也可能需要反向修正，"
               "必须结合单步误差与完整 AR 评价。投影不是机械能、材料应变或模块因果重要性。\n\n",
               "## 6. 七个固定案例：成功与失败均保留\n\n",
               "本节自动列出数值与原图，尚未据此伪造目视观察。"
               "GT 是逐帧独立重建 Gaussian，不是原始照片，没有可靠材料点对应。"
               "预测仅改变XYZ，其他11D属性固定初态。局部16NN仅为结构代理，不能叫材料应变。\n\n"]
    for case in CASES:
        pieces.append(f"### {_key(case)}\n\n")
        selected = [r for r in e["horizons"] if _matches(r, case)]
        pieces.append(_table(("模式", "平均CD mm", "H15 CD mm", "H15 RMS mm", "H15均值X/Y/Z mm"),
            [(mode, _mean([r["cd_mm"] for r in selected if r["mode"] == mode]),
              *next(((r["cd_mm"], r["rms_mm"],
                      f"({r['mean_x_mm']:.4f}, {r['mean_y_mm']:.4f}, {r['mean_z_mm']:.4f})")
                     for r in selected if r["mode"] == mode and r["horizon"] == 15),
                    (None, None, "未记录"))) for mode in (*MODES, "Copy")]))
        structure_rows = []
        for mode in ("Normal", "NoBase-All", "NoAttentionPaths", JOINT):
            for region in ("near", "middle", "far"):
                records = [r for r in e["structure"] if _matches(r, case) and r["mode"] == mode
                           and r["region"] == region and r["horizon"] == 15]
                direct = [r for r in e["same_history_regions"] if _matches(r, case) and r["mode"] == mode
                          and r["region"] == region and r["horizon"] == 15]
                structure_rows.append((mode, region, _mean([r["displacement_rms_mm"] for r in records]),
                    _mean([r.get("pred_region_to_full_gt_mm") for r in records]),
                    _mean([r["neighbor_distortion_p95"] for r in records]),
                    _mean([r["pred_region_to_full_gt_mm"] for r in direct]),
                    _mean([r["delta_region_to_full_gt_mm"] for r in direct])))
        pieces.append(_table(("模式", "区域", "完整AR RMS mm", "完整AR预测区域→完整GT mm", "16NN |比值−1| P95",
                              "固定历史预测区域→完整GT mm", "固定历史相对Normal差 mm"), structure_rows))
        if case in GEOMETRY_CASES:
            pieces.append(f"![{_key(case)} H15固定世界几何](figures/{_key(case)}_h15_geometry.png)\n\n")
        pieces.append("几何图各面板采用同一世界坐标范围，范围包含该窗所有模式、H1–15和GT；"
                      "没有逐模型缩放。GT 不按初态材料ID着色；预测颜色只是初态Controller距离分组。"
                      "区域单向距离不是GT材料区域的双向CD。以下为已有三相机原渲染，不是本报告重新渲染：\n\n")
        pieces.append(_image_index(case))
    pieces += ["## 7. 动作响应与剩余条件通路\n\n",
               "![动作响应](figures/action_response.png)\n\n",
               "FreezeAction 将完整初态Hand State与Controller保持固定；HalfRootTranslation "
               "按同一刚体平移改变Root及所属Controller，保留当前旋转、开口和指内几何。"
               "压力/有效性/空间Mask/Age/时间索引保留观测值。没有对应反事实GT，"
               "factual_gt_joint_only 只可作为对原事实目标的距离，不能当反事实预测精度。\n\n"]
    response_rows = []
    for case in CASES:
        for mode in ("Normal", JOINT):
            for response in ("FreezeAction", "HalfRootTranslation"):
                records = [r for r in e["action_response"] if _matches(r, case) and r["mode"] == mode
                           and r["response"] == response and r["horizon"] == 15]
                r = records[0] if len(records) == 1 else {}
                response_rows.append((_key(case), mode, response, _response_available(r), r.get("output_change_rms_mm"),
                    r.get("repeat_floor_mm"), r.get("measurable_above_numeric_floor"),
                    *(r.get(k) for k in ("mean_x_mm", "mean_y_mm", "mean_z_mm"))))
    unavailable = sorted({(_key((r["split"], int(r["episode"]), int(r["start"]))), r["mode"], r["response"])
                          for r in e["action_response"] if not _response_available(r)})
    pieces += [_table(("案例", "路径", "动作", "构造可用", "H15变化RMS mm", "重复数值底噪mm",
                       "高于数值底噪", "预测X", "预测Y", "预测Z"), response_rows),
               f"动作反事实不可用的案例/路径/动作组合：{len(unavailable)}。"
               "初态Root缺失而Controller仍有效等情况下，不能安全构造该动作或保留H1，"
               "Runner记available=False，未生成伪预测；图表过滤这些行，数值写未定义而不是0。"
               "有可用性缺口时，不能把缺测当作零响应，也不能据此自动通过动作合理性门槛。\n\n",
               _table(("不可用案例", "路径", "动作"), unavailable) if unavailable else "",
               "**联合 NoBase+NoAttentionPaths 已经没有触觉到输出的有效路径。**"
               "Hand Token仅由13D Hand State/身份/时间编码组成；T3只进入Controller K/V/Score。"
               "Value关闭且Base关闭后，Alpha→12D ToolMotion/4Dmass也没有最终输出影响。"
               "剩余显式动作/几何输入仍可响应，不可将其称为保留了触觉利用。"
               "对可构造且有效H1输入保持不变的FreezeAction，累计/相邻signed Basis全为零，"
               "联合消融又关闭Base，因此预期机械地输出Copy Initial；"
               "Normal相对这一输入产生差异，不能单凭该响应证明动作条件动力学已经学会或具有泛化。"
               "半幅动作不要求输出严格线性，但需检查方向、幅度及几何合法性；"
               "仅有可测变化不等于动作改善了预测，更不等于真实因果动力学。\n\n",
               "## 8. 预注册门槛、人工复核与阶段二\n\n",
               "- 用户完整Val100硬门槛：联合消融Joint相对恶化≤2%，CD相对恶化≤3%。\n"
               "- ep3/ep6须分别检查，不能互相掩盖明显退化；H10–H15不得系统性灾难退化；"
               "大运动/低变化/Near–Far不得出现新严重失败；剩余动作响应须可测且语义合理。\n"
               "- 实现另以每Episode及每H10–H15同样2%/3%作为更保守的自动启动充分条件。"
               "它不是用户额外规定的科学阈值，保守检查未通过时标为INCONCLUSIVE而非强行合格。\n\n",
               _table(("范围", "指标", "Normal", "联合消融", "比值", "上限", "通过"),
                      [(r["scope"], r["metric"], r["baseline"], r["value"],
                        r["ratio"], r["ratio_max"], r["passed"]) for r in gate["checks"]["user_hard"]]),
               _table(("Episode", "指标", "Normal", "联合消融", "比值", "通过"),
                      [(r["episode"], r["metric"], r["baseline"], r["value"], r["ratio"], r["passed"])
                       for r in gate["checks"]["conservative_episode"]]),
               f"数值硬门槛通过：{_fmt(gate['hard_gate_passed'])}；保守数值检查通过："
               f"{_fmt(gate['conservative_numeric_passed'])}；独立review通过：否（本工具未替研究人员填写）。\n\n",
               "失败/待补证据条目：\n\n"]
    pieces += [f"- {reason}\n" for reason in gate["reasons"]]
    pieces += ["\nreview_template.json 的所有审核标志为 null，不能作为通过证据；"
               "后续独立审查须依据真实图像/结构/动作记录填写新review文件与证据路径。"
               "不要修改此冻结预筛记录来追求PASS。只满足两个总体数值阈值不能自动推进。\n\n",
               "### 最终比较表\n\n",
               _table(("对象", "CD mm", "RGB", "Alpha", "Joint", "参数量/状态"), (
                   ("G5-Direct E8", *_summary_metrics(summary["Normal"]), p.get("parameter_count")),
                   ("冻结NoBase+NoAttentionPaths", *_summary_metrics(summary[JOINT]), "同一冻结模型，未删除参数"),
                   ("Unified Motion Decoder", "未执行", "未执行", "未执行", "未执行",
                    "仅在门槛与实现验收通过后才可另立研究分支"),
               )),
               "**冻结依赖与从头重训练是不同问题。** 当前报告不实现Unified、GEO、A2A新变体或G6，"
               "不申请GPU，不启动训练，不声明简化架构已经成功。\n\n",
               "## 9. 原始证据、文件数量与复核方式\n\n",
               _table(("原始JSON", "记录数量/类型"), (
                   (f"results/{name}.json", len(e[name]) if isinstance(e[name], (dict, list)) else type(e[name]).__name__)
                   for name in REQUIRED)),
               "完整同名CSV（若Runner输出）和JSON保留于results/；世界数组/帧号/输入/分支场位于arrays/；"
               "原始三相机图位于images/；运行日志位于logs/。"
               "单步与完整AR、Anchor控制与实际GS运动均未混合。"
               "null/未定义不等于零，表中缺组明确写未定义/未记录。"
               "每个文件的真实大小、SHA256和相对路径见artifact_manifest.json；"
               f"其根目录为 {out}。\n\n",
               "本报告的CPU绘图工具哈希记录于artifact_manifest.json，真实推理源码/Checkpoint/Job"
               "取自provenance.json。图像尚待独立审查，不凭自动绘图生成假造视觉结论。"
               "任务在阶段一门槛未满足或证据未完备时停止，不自动进入下一轮研究。\n"]
    return "".join(pieces)


def build(output):
    """Read an existing completed Run and exclusively create its CPU report."""
    out = Path(output).resolve()
    if not out.is_dir():
        raise ValueError("Completed Run directory does not exist")
    if (out/"figures").exists() and out not in (out/"figures").resolve().parents:
        raise ValueError("Figures path escapes the Run")
    conflicts = [str(out/name) for name in NEW_FILES if (out/name).exists()]
    if conflicts:
        raise FileExistsError("Refuse to overwrite report artifacts: " + ", ".join(conflicts))
    evidence = _read_inputs(out)
    text = _report(out, evidence)
    (out/"figures").mkdir(exist_ok=True)
    _make_figures(out, evidence)
    absent = [name for name in FIGURES if not (out/"figures"/name).is_file()]
    if absent:
        raise ValueError("CPU figure generation incomplete: " + str(absent))
    with (out/"frozen_attribution_report.md").open("x", encoding="utf8") as stream:
        stream.write(text)
    review = dict(review_status="PENDING_INDEPENDENT_REVIEW", **{flag: None for flag in REVIEW_FLAGS},
                  evidence_paths=[], note="Template only; null is not approval, no Stage2 autostart")
    _write_json(out/"review_template.json", review)
    artifacts = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(dict(path=path.relative_to(out).as_posix(), bytes=path.stat().st_size, sha256=_sha(path)))
    manifest = dict(root=str(out), kind="FROZEN_G5_NECESSITY_CPU_REPORT",
                    source_sha=evidence["provenance"]["diagnostic_source_sha"],
                    job_id=evidence["provenance"]["job_id"],
                    preliminary_gate=evidence["gate"]["status"], independent_visual_review_completed=False,
                    launches_stage2=False, tool_file=str(Path(__file__).resolve()), tool_sha256=_sha(__file__),
                    run_kind=evidence["provenance"].get("run_kind", "REAL_FROZEN_INFERENCE"),
                    files=artifacts)
    _write_json(out/"artifact_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build(args.output)
    print(json.dumps(dict(report=str(Path(args.output).resolve()/"frozen_attribution_report.md"),
                          gate=manifest["preliminary_gate"], files=len(manifest["files"]),
                          stage2_autostart=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
