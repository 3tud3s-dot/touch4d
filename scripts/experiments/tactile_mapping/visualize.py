"""CPU-only, read-only-source tactile -> controller visualization (no GS loads).

Run from the repository root with -B. All generated artifacts go below --output.
The input root contains lightweight touch4d-v1 episode files, never GS files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from scripts.dataloader.dataset import _episode_conditions, read_npz
from scripts.dataloader.geometry import hand_slots, tactile_world_coordinates
from scripts.dataloader.splits import SPLITS
from scripts.rebuild_dataset.rebuild import GRID_ROWS, GRID_COLS, POINT_IDS

EPISODES = (1, 2, 4, 5, 7, 8)
SIGMAS = (.005, .01, .02)
EPS = np.finfo(np.float64).tiny
NAMES = ("Left hand / left finger", "Left hand / right finger",
         "Right hand / left finger", "Right hand / right finger")
SOURCE_FILES = ("metadata.json", "controller.npz", "tactile.npz", "timestamps.npz")
REPO = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_geometry():
    """Protocol local surface coordinates; checked against existing world geometry."""
    r, c = np.meshgrid(np.arange(12), np.arange(32), indexing="ij")
    pixels = np.stack((np.full(r.shape, .007), -.056-.002*c, .0235-.002*r), -1).reshape(384, 3)
    r, c = np.meshgrid(GRID_ROWS, GRID_COLS, indexing="ij")
    points = np.stack((np.full(r.shape, .007), -.056-.002*c,
                       (.025/12)*(11.5-r)), -1).reshape(15, 3)
    # Opposite finger has x=-.007 for both arrays; same-finger distances are identical.
    return pixels, points


def kernel(sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive, in metres")
    pixels, points = local_geometry()
    return np.exp(-np.sum((points[:, None]-pixels[None])**2, -1)/(2*sigma*sigma))


def map_values(values, valid, controller_valid, sigma):
    """Independent [F,4,384] -> [F,4,15]; no threshold, cross-finger, or time pooling."""
    values = np.asarray(values, dtype=np.float64).reshape(-1, 4, 384)
    valid = np.asarray(valid, dtype=bool).reshape(values.shape)
    cp_valid = np.asarray(controller_valid, dtype=bool).reshape(-1, 4, 15)
    if cp_valid.shape[0] != len(values) or not np.isfinite(values[valid]).all():
        raise ValueError("Invalid mapping inputs")
    k = kernel(sigma)
    den = valid.astype(np.float64) @ k.T
    num = np.where(valid, values, 0) @ k.T
    observed = valid.any(-1, keepdims=True) & cp_valid
    mapped = np.where(observed, num/(den+EPS), 0)
    assert np.isfinite(mapped).all()
    return mapped, observed


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return None
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def masked_mean(a, m, axis=-1):
    n = m.sum(axis)
    return np.divide(np.where(m, a, 0).sum(axis), n,
                     out=np.full(n.shape, np.nan, dtype=float), where=n > 0)


def select_frames(values, mask, active):
    """Reserve midpoint, then low/high/change; all fingers must have valid observations."""
    v, m = values[:, active].reshape(len(values), -1), mask[:, active].reshape(len(values), -1)
    score = masked_mean(v, m)
    eligible = mask[:, active].any(-1).all(-1)
    both = m[1:] & m[:-1]
    change = np.r_[np.nan, masked_mean(np.abs(v[1:]-v[:-1]), both)]
    change_ok = eligible & np.r_[False, eligible[:-1]] & np.isfinite(change)
    chosen = []
    used = set()
    def take(kind, order, allowed):
        frame = next((int(i) for i in order if allowed[i] and int(i) not in used), None)
        if frame is None:
            raise ValueError(f"Not enough distinct valid frames for {kind}")
        used.add(frame)
        chosen.append(dict(frame=frame, reason=kind, raw_mean=float(score[frame]),
                           change=None if not np.isfinite(change[frame]) else float(change[frame])))
    ids = np.arange(len(values))
    take("midpoint (nearest fully observed)", np.lexsort((ids, abs(ids-len(values)//2))), eligible)
    take("lowest valid mean", np.lexsort((ids, score)), eligible)
    take("highest valid mean", np.lexsort((ids, -score)), eligible)
    take("largest adjacent-frame pixel change", np.lexsort((ids, -np.nan_to_num(change, nan=-np.inf))), change_ok)
    return sorted(chosen, key=lambda x: x["frame"]), score, change


def load_episode(root, ep):
    path = root/"001-rope"/f"episode_{ep}"
    meta = json.loads((path/"metadata.json").read_text(encoding="utf-8"))
    slots = hand_slots(meta)
    assert meta["scene"] == "001-rope" and meta["episode"] == ep
    assert meta["controller_point_ids"] == POINT_IDS
    expected = (None, 0) if ep < 5 else (1, 0)
    assert slots == expected, f"Unexpected hand binding: episode {ep}"
    tm = read_npz(path/"timestamps.npz")
    ts, ages = tm["observation_timestamp_us"], tm["tactile_age_us"]
    assert np.all(np.diff(ts) > 0) and len(ts) == meta["frames"]
    for f in range(4):
        raw = tm[f"raw_tactile_timestamp_us_{f}"]
        assert np.all(np.diff(raw) >= 0)
        idx = np.searchsorted(raw, ts, side="right")-1
        actual_age = np.where(idx >= 0, ts-raw[np.maximum(idx, 0)], -1)
        np.testing.assert_array_equal(idx, tm["tactile_raw_index"][:, f])
        np.testing.assert_array_equal(actual_age, ages[:, f])
    # Existing strict array validation, without Dataset construction or any GS loading.
    data = _episode_conditions(SimpleNamespace(path=path, frames=len(ts), ages=ages, slots=slots))
    hand_keep = np.array([ep >= 5, True])
    active = np.flatnonzero(np.repeat(hand_keep, 2)).tolist()
    xyz, spatial = tactile_world_coordinates(data["T_worlds"], data["openings"], slots,
                                            data["action_valid"], data["valid_mask"])
    mask = spatial & np.repeat(hand_keep, 2)[None, :, None, None]
    cp_mask = data["controller_mask"] & hand_keep[None, :, None]
    values = np.where(mask, data["tactile"], 0).reshape(len(ts), 4, 384)
    mask = mask.reshape(values.shape)
    cp_mask = cp_mask.reshape(len(ts), 4, 15)
    pixels, points = local_geometry()
    distance = np.linalg.norm(points[:, None]-pixels[None], axis=-1)
    error = 0.
    for i in (0, len(ts)//2, len(ts)-1):
        for f in active:
            m = mask[i, f]
            cp = data["controller_points"][i, f//2, (f%2)*15:(f%2+1)*15]
            d = np.linalg.norm(cp[:, None]-xyz[i, f].reshape(1, 384, 3), axis=-1)
            if m.any():
                error = max(error, float(np.max(abs(d[:, m]-distance[:, m]))))
    assert error < 2e-6, "Local geometry is inconsistent with the existing transforms"
    selected, score, change = select_frames(values, mask, active)
    return dict(episode=ep, frames=len(ts), active=active, values=values, mask=mask,
                cp_mask=cp_mask, timestamp_us=ts, seconds=(ts-ts[0])/1e6,
                selected=selected, score=score, change=change, local_distance_error_m=error,
                hand_keep=hand_keep.tolist(), metadata=meta)


def self_tests():
    checks = []
    def check(name, ok):
        if not bool(ok):
            raise AssertionError(name)
        checks.append(name)
    p, q = local_geometry()
    v = np.full((2, 4, 384), 7.)
    m = np.ones(v.shape, bool); m[..., 31::32] = False
    cp = np.ones((2, 4, 15), bool)
    for sig in SIGMAS:
        k = kernel(sig); weights = m[..., None, :]*k
        weights /= weights.sum(-1, keepdims=True)+EPS
        y, obs = map_values(v, m, cp, sig)
        check(f"sigma={sig}: weights sum to one", np.allclose(weights.sum(-1), 1, atol=1e-12))
        check(f"sigma={sig}: constant preserved", np.allclose(y, 7, atol=1e-12))
        check(f"sigma={sig}: zero observed", map_values(v*0, m, cp, sig)[1].all())
        missing = m.copy(); missing[:, :2] = False
        bad = v.copy(); bad[:, :2] = np.nan
        y, obs = map_values(bad, missing, cp, sig)
        check(f"sigma={sig}: hand isolated", not obs[:, :2].any() and np.all(y[:, :2] == 0) and np.allclose(y[:, 2:], 7))
        zero, absent = map_values(v, m*False, cp, sig)
        check(f"sigma={sig}: no observation", np.all(zero == 0) and not absent.any())
    impulse = np.zeros_like(v); impulse[:, 2, 6*32+16] = 100
    y, _ = map_values(impulse, m, cp, .01)
    check("single-finger impulse at CP7", np.all(y[:, 2].argmax(-1) == 7) and not y[:, [0, 1, 3]].any())
    shifted = np.zeros_like(v); shifted[:, 2, 6*32+30] = 100
    check("spatial shift reaches tip CP9", np.all(map_values(shifted, m, cp, .01)[0][:, 2].argmax(-1) == 9))
    bad = v.copy(); bad[..., 31::32] = 1e12
    check("invalid pixels cannot leak", np.array_equal(map_values(bad, m, cp, .01)[0], map_values(v, m, cp, .01)[0]))
    cp[:, :, 14] = False
    y, observed = map_values(v, m, cp, .01)
    check("invalid CP isolated", np.all(y[..., 14] == 0) and not observed[..., 14].any())
    m[:] = False; m[..., 0] = True
    check("sparse constant remains constant", np.allclose(map_values(v, m, np.ones_like(cp), .005)[0], 7, atol=1e-12))
    m[:] = True; m[..., 31::32] = False
    a = select_frames(np.arange(8)[:, None, None]*np.ones((8, 4, 384)), np.repeat(m[:1], 8, 0), [2, 3])[0]
    b = select_frames(np.arange(8)[:, None, None]*np.ones((8, 4, 384)), np.repeat(m[:1], 8, 0), [2, 3])[0]
    check("selection reproducible and distinct", a == b and len({x['frame'] for x in a}) == 4)
    check("CP31 remains geometrically present", np.allclose(q.reshape(3, 5, 3)[:, -1, 1], -.118))
    return checks


def summarize(data, mapped, observed):
    rows = []
    pixels, points = local_geometry()
    for f in data["active"]:
        v, m = data["values"][:, f], data["mask"][:, f]
        y, o = mapped[:, f], observed[:, f]
        raw_mean, out_mean = masked_mean(v, m), masked_mean(y, o)
        peak = np.max(np.where(m, v, 0), -1)
        out_peak = np.max(np.where(o, y, 0), -1)
        valid_frame = m.any(-1) & o.any(-1)
        ratios = out_peak[valid_frame & (peak > 0)]/peak[valid_frame & (peak > 0)]
        pairs = valid_frame[1:] & valid_frame[:-1]
        keyframes = []
        for sel in data["selected"]:
            i = sel["frame"]; raw_id = int(np.argmax(v[i])); mapped_id = int(np.argmax(y[i]))
            nearest = int(np.argmin(np.linalg.norm(points-pixels[raw_id], axis=-1)))
            keyframes.append(dict(frame=i, raw_peak=float(peak[i]), mapped_peak=float(out_peak[i]),
                                  raw_peak_rc=list(divmod(raw_id, 32)), mapped_peak_id=mapped_id,
                                  closest_cp_to_raw_peak=nearest,
                                  peak_location_distance_mm=float(np.linalg.norm(points[mapped_id]-pixels[raw_id])*1000)))
        rows.append(dict(stream=f, name=NAMES[f], valid_frames=int(valid_frame.sum()),
                         mean_correlation=corr(raw_mean, out_mean),
                         adjacent_change_correlation=corr(np.diff(raw_mean)[pairs], np.diff(out_mean)[pairs]),
                         peak_retention_p50=None if len(ratios)==0 else float(np.median(ratios)),
                         keyframes=keyframes))
    return rows


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "figure.facecolor":"#f4f6fa", "axes.facecolor":"#e3e7ec",
                         "savefig.facecolor":"#f4f6fa"})
    cmap = plt.colormaps["magma"].copy(); cmap.set_bad("#bbc3cc")
    return cmap


def raw_axis(ax, v, m, cmap, norm, overlay=False):
    ax.imshow(np.ma.array(v.reshape(12, 32), mask=~m.reshape(12, 32)),
              cmap=cmap, norm=norm, origin="upper", interpolation="nearest", aspect="auto")
    ax.set(xlim=(-.5, 31.5), ylim=(11.5, -.5), xlabel="Taxel column (toward fingertip)", ylabel="Taxel row")
    ax.set_xticks([0, 8, 16, 23, 31]); ax.set_yticks([0, 6, 11])
    if overlay:
        _, points = local_geometry()
        c = (-.056-points[:, 1])/.002
        r = (.0235-points[:, 2])/.002
        ax.scatter(c, r, s=260, facecolors="#e9f3ff", edgecolors="#1e293b", linewidths=.6, zorder=3)
        for i, (cx, ry) in enumerate(zip(c, r)):
            ax.text(cx, ry, str(i), ha="center", va="center", fontsize=8, color="#0f172a", zorder=4)
        # Leave room for edge labels without moving their geometric centers.
        ax.set(xlim=(-1.1, 32.1), ylim=(12.05, -1.05))
    else:
        rr, cc = np.where(~m.reshape(12, 32))
        ax.scatter(cc, rr, marker="x", s=8, color="#576677", linewidths=.5)


def cp_axis(ax, y, obs, cmap, norm):
    ax.imshow(np.ma.array(y.reshape(3, 5), mask=~obs.reshape(3, 5)), cmap=cmap, norm=norm,
              interpolation="nearest", aspect="auto")
    ax.set_xticks(range(5), GRID_COLS); ax.set_yticks(range(3), GRID_ROWS)
    ax.set(xlabel="Controller column", ylabel="Controller row")
    for i in range(15):
        label = f"{i}\n{y[i]:.1f}" if obs[i] else f"{i}\nN/A"
        ax.text(i%5, i//5, label, color="white" if y[i] < norm.vmax*.65 else "#14141e",
                ha="center", va="center", fontsize=9,
                bbox=dict(facecolor="#151c30", alpha=.34, edgecolor="none", pad=1))


def keyframe_figure(data, selection, mapped, observed, out, cmap):
    frame = selection["frame"]; active = data["active"]
    fig, axes = plt.subplots(len(active), 3, figsize=(16, 3.15*len(active)+1), squeeze=False, layout="constrained")
    fig.suptitle(f"001-rope | Episode {data['episode']} | Frame {frame} | {data['seconds'][frame]:.3f} s | sigma=0.010 m\n"
                 f"{selection['reason']} | baseline-subtracted positive response (sensor units; no /100)", fontsize=14)
    for row, f in enumerate(active):
        vmax = max(float(data['values'][:, f].max()), 1.)
        norm = Normalize(0, vmax)
        raw_axis(axes[row, 0], data['values'][frame, f], data['mask'][frame, f], cmap, norm)
        raw_axis(axes[row, 1], data['values'][frame, f], data['mask'][frame, f], cmap, norm, True)
        cp_axis(axes[row, 2], mapped[frame, f], observed[frame, f], cmap, norm)
        for col, title in enumerate(("A  12 x 32 response", "B  Local-geometry CP overlay", "C  3 x 5 weighted response")):
            axes[row, col].set_title(f"{NAMES[f]}\n{title}", fontsize=10)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[row].tolist(), shrink=.8,
                     label="Response (same linear scale A/B/C)")
    fig.savefig(out, dpi=125); plt.close(fig)


def time_figure(data, mapped, observed, metrics, out, cmap):
    fig, axes = plt.subplots(len(data['active']), 2, figsize=(17, 3.2*len(data['active'])+1),
                             gridspec_kw={"width_ratios":[1.5, 1]}, squeeze=False, layout="constrained")
    fig.suptitle(f"001-rope | Episode {data['episode']} | Frames 0-{data['frames']-1} | sigma=0.010 m\n"
                 "15-controller time maps | gray=N/A | zero is valid | per-finger mapped-value color range", fontsize=14)
    for row, (f, metric) in enumerate(zip(data['active'], metrics)):
        y, o = mapped[:, f], observed[:, f]
        norm = Normalize(0, max(float(y.max()), 1.))
        ax = axes[row, 0]
        ax.imshow(np.ma.array(y.T, mask=~o.T), origin="upper", aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set(title=NAMES[f], xlabel="Frame (no time resampling)", ylabel="Local CP ID 0-14")
        ax.set_yticks(range(15)); ticks=np.linspace(0, data['frames']-1, 6).astype(int)
        ax.set_xticks(ticks, [f"{i}\n{data['seconds'][i]:.2f}s" for i in ticks])
        for sel in data['selected']:
            ax.axvline(sel['frame'], color="#36dfd3", alpha=.65, linewidth=.8)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Response")
        ax=axes[row, 1]
        raw_mean=masked_mean(data['values'][:, f], data['mask'][:, f]); mapped_mean=masked_mean(y, o)
        ax.plot(data['seconds'], raw_mean, color="#465dd5", label="Valid raw-pixel mean", linewidth=1.4)
        ax.plot(data['seconds'], mapped_mean, color="#dc673d", label="15-CP mean", linewidth=1.2)
        ax.set(xlabel="Elapsed time (s)", ylabel="Response (sensor units)",
               title=f"{NAMES[f]} | mean trend r={metric['mean_correlation']:.3f}")
        ax.legend(fontsize=8); ax.grid(alpha=.18)
    fig.savefig(out, dpi=125); plt.close(fig)


def bandwidth_figure(data, f, frame, all_maps, out, cmap):
    fig, axes=plt.subplots(1, 4, figsize=(18, 4.6), layout="constrained")
    vmax=max(float(data['values'][frame, f].max()), 1.); norm=Normalize(0, vmax)
    fig.suptitle(f"001-rope | Episode {data['episode']} | Frame {frame} | {NAMES[f]}\n"
                 "Bandwidth diagnostic (candidate only) | same frame-specific linear color scale", fontsize=14)
    raw_axis(axes[0], data['values'][frame, f], data['mask'][frame, f], cmap, norm, True)
    axes[0].set_title(f"Input + CPs | peak={vmax:.1f}")
    for ax, sigma in zip(axes[1:], SIGMAS):
        y, obs=all_maps[sigma]; cp_axis(ax, y[frame, f], obs[frame, f], cmap, norm)
        ax.set_title(f"sigma={sigma:.3f} m\npeak={y[frame, f].max():.2f} ({100*y[frame, f].max()/vmax:.1f}% of raw)")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist(), shrink=.7, label="Response")
    fig.savefig(out, dpi=140); plt.close(fig)


def write_reports(out, results):
    rows=results['episodes']; selected=[]; metric_rows=[]; cards=[]
    for data in rows:
        ep=data['episode']
        for s in data['selected']:
            selected.append(f"| {ep} | {s['frame']} | {s['time_s']:.3f} | {s['reason']} | {s['raw_mean']:.4f} |")
        for r in data['metrics']['0.01']:
            metric_rows.append(f"| {ep} | {r['name']} | {r['mean_correlation']:.4f} | {r['adjacent_change_correlation']:.4f} | {100*r['peak_retention_p50']:.1f}% |")
        imgs=''.join(f'<figure><figcaption>Frame {s["frame"]} · {html.escape(s["reason"])}</figcaption><a href="images/ep{ep}_frame{s["frame"]:04d}.png" target="_blank"><img loading="lazy" src="images/ep{ep}_frame{s["frame"]:04d}.png" alt="Episode {ep} frame {s["frame"]} all participating fingers"></a></figure>' for s in data['selected'])
        cards.append(f'<section id="ep{ep}" class="episode" data-ep="{ep}"><h2>Episode {ep} · {"仅右手" if ep<5 else "双手"}</h2><p>{data["frames"]} 帧 · 关键帧色域在同一指的整个 episode 固定；A/B/C 共用色标。点击图片可放大查看。</p>{imgs}<h3>全序列 · 15 控制点时间图</h3><a href="images/ep{ep}_time.png" target="_blank"><img loading="lazy" src="images/ep{ep}_time.png" alt="Episode {ep} controller temporal maps"></a></section>')
    flat=[m for d in rows for m in d['metrics']['0.01']]
    corrs=[m['mean_correlation'] for m in flat]; changes=[m['adjacent_change_correlation'] for m in flat]; peaks=[m['peak_retention_p50'] for m in flat]
    bandwidth=[]
    for sigma in SIGMAS:
        rr=[r['peak_retention_p50'] for d in rows for r in d['metrics'][str(sigma)]]
        bandwidth.append(f"| {sigma:.3f} | {100*min(rr):.1f}%–{100*max(rr):.1f}% |")
    conclusion=(f"σ=0.01 m 时，18 条参与手指序列的均值趋势 Pearson r 为 {min(corrs):.3f}–{max(corrs):.3f}，"
                f"相邻均值变化相关为 {min(changes):.3f}–{max(changes):.3f}；各序列逐帧峰值保留率的中位数为 {100*min(peaks):.1f}%–{100*max(peaks):.1f}%。"
                "15 点能给出粗粒度空间分布和时间趋势，但局部尖峰会被明显平均；不能当作 384 像素的无损替代，也不能据此证明动力学预测改善。全序列峰值统计包含低响应背景，不等于接触峰值衰减率。")
    examples=[]
    for ep in (2,8):
        d=next(d for d in rows if d['episode']==ep)
        frame=next(s['frame'] for s in d['selected'] if s['reason']=='highest valid mean')
        f=int(next(p for p in results['bandwidth_images'] if f'bandwidth_ep{ep}_' in p).split('_stream')[1].split('.')[0])
        case=[]
        for sig in SIGMAS:
            finger=next(m for m in d['metrics'][str(sig)] if m['stream']==f)
            case.append(next(k for k in finger['keyframes'] if k['frame']==frame))
        examples.append(f"| ep{ep} / {frame} / {NAMES[f]} | {case[0]['raw_peak']:.1f} | " + ' | '.join(f"{k['mapped_peak']:.2f} / CP{k['mapped_peak_id']}" for k in case) + ' |')
    report=f'''# 001-rope：触觉 → 15 Controller Points 可视化验收

日期：{results['created_local']}。CPU-only；未加载 GS／checkpoint、未训练、未修改正式数据与模型。

## 结论

{conclusion}

这是空间加权平均，不是接触分类或力标定。宽核会混合较远位置，小核更局部但仍受 3×5 采样间距限制；细小接触斑、多峰、尖端极值及局部瞬态幅值可能丢失。近邻响应出现在合理的控制点并不意味着真实接触点或物理力已经恢复。σ=0.01 m 仅供可视化，未确定为生产参数。

## 数据与方法

- 源：`/data1/userdata/tcweng/projects/touch4d/datasets/touch4d-v1/001-rope/episode_{{1,2,4,5,7,8}}/`。24 个轻量文件下载前有 SHA256，副本及运行后哈希核对一致；清单见 `source_sha256.txt`。不含 GS／RGB／depth。
- 六个 episode 均来自既定 Train split；ep1/2/4 整组屏蔽左手，ep5/7/8 保留双手，不按静止状态过滤。数据副本保留原始文件，输出数值与 Mask 实施屏蔽。
- A 列“原始”表示本次映射的输入 `max(raw−median,0)`，不是未扣基线的 ADC。单位为传感器响应数值；未做 `/100`、力标定、逐帧幅值归一化或接触阈值。
- 复用 `dataset._episode_conditions`、`geometry.hand_slots/tactile_world_coordinates`，核对 Controller 网格常量／ID。局部公式由协议定义，世界与局部距离抽检最大差 `{max(d['local_distance_error_m'] for d in rows):.3g} m`。
- 同指距离核 `exp(−d²/(2σ²))`，仅按数据／时间／空间有效性和 Hand Mask 归一化；还应用 Controller Mask。ε={EPS:.4g}（float64 tiny）。无有效采样输出 0、observed=False；有效零值 observed=True。
- 触觉 12×32 存储槽，col31 无效；Controller rows=(0,6,11)、cols=(0,8,16,23,31)，行优先 ID0–14。Controller col31 仍有效，从邻近有效像素聚合，不复制无效 taxel。
- B 列按真实局部 Y/Z 投影叠加，Controller 的浮点触觉行坐标约 −0.229、6.021、11.229，不强行对齐整数行。row 增大表示夹指局部 Z 减小，图像朝向不代表世界上下。
- A/B/C 同指同 episode 使用相同线性色域 `[0, episode raw max]`；灰色／叉号表示无效，深色低值包含有效零值。时间图为了可读使用本指映射值色域，旁列原始／映射均值趋势仍用原单位；带宽图单帧内共享 raw peak 色域。
- C 中小数显示到一位，显示 0.0 不代表严格为零，完整精度见 NPZ；有效性必须读取 observed mask。
- 所有 15 点都有宽核支持时不表示 15 处都有真实接触。输出不是总压力守恒映射，原始像素均值与 Controller 均值也不必相等。

## 24 个确定性代表帧

先保留最近全指有效的中间帧，再在未选帧中依次取参与指有效像素均值最低、最高、相邻帧共同有效像素绝对变化均值最大。并列取较小帧号，四帧去重；没有使用预测结果。最大变化使用过去帧差，仅用于离线展示选帧。若全 episode 某指失效则不伪装低响应样本，直接报错。

| Episode | Frame | 相对时间 s | 依据 | 参与指有效像素均值 |
|---|---:|---:|---|---:|
{chr(10).join(selected)}

## σ=0.01 m 的全序列指标

| Episode | 手／指 | 均值趋势 r | 相邻均值变化 r | 峰值保留率中位数 |
|---|---|---:|---:|---:|
{chr(10).join(metric_rows)}

峰值保留率为每个有效帧 `max(mapped15)/max(valid raw372)`，仅分母大于零时定义，零响应帧并未从映射中删除。趋势相关分别取有效帧，以及两端均有效的相邻帧；不是跨手合并或接触识别准确率。

## 带宽敏感性

| σ（m） | 18 条序列各自峰值保留率中位数的范围 |
|---:|---:|
{chr(10).join(bandwidth)}

少量图示采用 ep2、ep8 的“最高均值”代表帧，各选该帧均值最高的参与手指，同一输入对比三个带宽；选择不依赖映射后效果。完整数值包含所有六个 episode 和三个 σ。单帧 raw 峰值与 mapped 峰值不一定在同一邻域：平均聚合可能偏向较宽响应区，不能只以峰值位置相同作为通过条件。

| 带宽图示 | 输入峰值 | 5 mm 峰值／位置 | 10 mm 峰值／位置 | 20 mm 峰值／位置 |
|---|---:|---|---|---|
{chr(10).join(examples)}

具体空间观察：ep2 第72帧右手左指的高响应主要在末端列，映射集中到 CP4/9/14，指尖位置关系合理；但 10 mm 核使这三行接近，丢失了原始竖向起伏。该帧最强单像素在 row2/col28，10 mm 映射最大值在 CP14，而最近的控制点是 CP4；这是局部平均与点采样导致的峰值重排，不能把它解读为精确接触位置恢复。5 mm 保留更多行间差异，20 mm 更接近广域平均。均值趋势高相关只证明整体起伏接近，不证明每个局部模式都被保留。

## 验收与复现

- CPU 自检：**{len(results['checks'])} passed / 0 failed**，含三带宽权重归一化、恒定输入、有效零值、全缺失、NaN／无效像素隔离、整手屏蔽、同指脉冲／空间移位、Controller Mask、稀疏有效输入、选帧复现。详见 `results.json`。
- 数据检查：时间戳最近过去配对、0–75 ms、无效列、左右手映射及已确认 episode 模式均核对；输入下载校验 {len(results['source_hashes'])} 文件，运行前后不变。
- 交付：24 张关键帧三列图、6 张完整时间图、2 张带宽图；共72个关键帧指观测。`mapped/episode_*.npz` 保留全部四槽位、三个 σ、observed mask、帧号、时间和选择统计；单手左侧永远为零且无观测。
- 代码基准 `{results['git_head']}`，分支 `{results['git_branch']}`；新诊断脚本未提交，SHA256 `{results['script_sha256']}`。源码文件指纹记录在 `results.json`。
- 命令：`python -B -m scripts.experiments.tactile_mapping.visualize --input-root audit/tactile-controller-viz-v1/input --output audit/tactile-controller-viz-v1`。仅用 NumPy／Matplotlib 和现有读取／几何函数；不调用模型或 CUDA。
- 定位：映射的几何／数值正确性通过；粗空间分布与趋势的可视化可供审阅。局部峰值保真明显有限，未来动力学收益未验证；不据此确定生产带宽或修改模型。
- 可视化验收边界：本轮已直接抽查单手、双手、带宽及时间 PNG。浏览器自动化打开 `file://` 被安全策略拦截，未完成真实浏览器截图／点击验收；本地 HTML 资源链接、图像解码和选项完整性通过 CPU 静态检查。
'''
    (out/'report.md').write_text(report, encoding='utf-8')
    bw=''.join(f'<a href="{p}" target="_blank"><img loading="lazy" src="{p}" alt="bandwidth comparison"></a>' for p in results['bandwidth_images'])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>触觉 → Controller · 001-rope</title>
<style>body{margin:0;background:#edf1f7;color:#1e293b;font:16px/1.65 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:28px}h1{font-size:32px;margin:0}h2{margin-top:12px}header,section,.notice{background:white;border:1px solid #dce3ec;border-radius:16px;padding:24px;margin-bottom:22px}nav{position:sticky;top:0;background:#edf1f7ee;padding:12px 0;display:flex;gap:8px;flex-wrap:wrap;z-index:4}button,a.pill{border:1px solid #becbdd;border-radius:8px;padding:8px 15px;background:white;color:#233b59;cursor:pointer;font-size:15px;text-decoration:none}button.active{background:#254aaa;color:white}img{display:block;width:100%;height:auto;border-radius:8px}figure{margin:22px 0}figcaption{font-weight:600;margin:10px 0}small{color:#586b82}.hidden{display:none}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.grid div{padding:12px;background:#f0f4fa;border-radius:10px}p{margin:10px 0}a{color:#254aaa}@media(max-width:700px){main{padding:12px}.grid{grid-template-columns:1fr}}</style>
<main><header><small>TOUCH4D / CPU AUDIT / TRAIN EPISODES ONLY</small><h1>从 384 像素到 15 个控制点</h1><p>001-rope · 同指 Gaussian 空间加权 · σ=0.01 m 候选 · 原始响应、几何依据与时间变化</p><div class="grid"><div>24 个代表帧<br>6 个训练 Episode</div><div>72 个参与手指观测<br>单手整组屏蔽左手</div><div>3 个候选带宽<br>不引入接触阈值</div></div><p><a href="report.md">Markdown 验收报告</a> · <a href="results.json">全部数值与样本</a> · <a href="source_sha256.txt">源文件 SHA256</a></p></header>
<div class="notice"><strong>如何阅读</strong><p>A：扣基线、截零后的输入（不是未处理 ADC）；B：真实局部坐标投影的控制点；C：15 点加权值。灰色是无观测，深色零值仍可能有效。A/B/C 共用色标；时间图使用单独的映射值色域。图中编号均为每指 0–14，行列方向沿传感器索引。</p><p>CONCLUSION</p></div>
<nav><button class="active" data-select="all">全部</button>NAV<button data-select="bandwidth">带宽对照</button><a class="pill" href="#top" onclick="window.scrollTo(0,0)">顶部</a></nav>CARDS<section id="bandwidth" class="episode" data-ep="bandwidth"><h2>0.005 / 0.01 / 0.02 m</h2><p>同一输入，统一线性色域。核越宽不一定每个峰值都单调降低，但整体局部性会减弱；15 点采样本身也限制细节表达。候选带宽未作为生产参数验收。</p>BW</section><footer>全部图像、数据位于本目录，无 CDN、无在线请求。点击图像打开原尺寸 PNG。</footer></main>
<script>for(const b of document.querySelectorAll('button[data-select]'))b.onclick=()=>{document.querySelectorAll('button[data-select]').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('section.episode').forEach(s=>s.classList.toggle('hidden',b.dataset.select!=='all'&&s.dataset.ep!==b.dataset.select));};</script></html>'''
    page=page.replace('CONCLUSION',html.escape(conclusion)).replace('NAV',''.join(f'<button data-select="{e}">EP {e}</button>' for e in EPISODES)).replace('CARDS',''.join(cards)).replace('BW',bw)
    (out/'index.html').write_text(page, encoding='utf-8')


def verify_artifacts(out, results):
    """Static resource checks only; not a browser screenshot or DOM interaction test."""
    from PIL import Image
    class Page(HTMLParser):
        def __init__(self):
            super().__init__(); self.images=[]; self.links=[]; self.buttons=[]; self.sections=[]
        def handle_starttag(self, tag, attrs):
            a=dict(attrs)
            if tag=='img': self.images.append(a['src'])
            if tag=='a' and not a.get('href','').startswith('#'): self.links.append(a['href'])
            if tag=='button': self.buttons.append(a['data-select'])
            if tag=='section': self.sections.append(a['data-ep'])
    page=Page(); page.feed((out/'index.html').read_text(encoding='utf-8'))
    assert len(page.images)==len(set(page.images))==32
    assert set(page.buttons)=={'all','bandwidth',*(str(e) for e in EPISODES)}
    assert set(page.sections)=={'bandwidth',*(str(e) for e in EPISODES)}
    for rel in page.images+page.links:
        assert '://' not in rel and (out/rel).is_file(), rel
    for rel in page.images:
        with Image.open(out/rel) as im:
            assert im.width>=1500 and im.height>=500
            im.verify()
    for d in results['episodes']:
        with np.load(out/'mapped'/f"episode_{d['episode']}.npz",allow_pickle=False) as z:
            assert len(np.unique(z['selected_frames']))==4
            for sigma in SIGMAS:
                y=z[f'mapped_sigma_{sigma:g}'];obs=z[f'observed_sigma_{sigma:g}']
                assert y.shape==obs.shape==(d['frames'],4,15)
                assert np.isfinite(y).all() and np.all(y[~obs]==0)
                if d['episode']<5: assert np.all(y[:,:2]==0) and not obs[:,:2].any()
    return dict(png_decoded=32,html_image_references=32,episode_sections=6,bandwidth_sections=1,
                mapped_episodes=6,hand_mask_verified=True,browser_screenshot_verified=False,
                browser_limitation='file:// navigation rejected by browser security policy; no workaround attempted')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--refresh',action='store_true',help='Regenerate only this script\'s own existing outputs after visual QA.')
    args=parser.parse_args()
    checks=self_tests()
    if args.self_test:
        print(json.dumps({'passed':len(checks),'checks':checks},indent=2));return
    if args.input_root is None or args.output is None:parser.error('--input-root and --output are required')
    out=args.output.resolve(); root=args.input_root.resolve()
    if (out/'index.html').exists():
        if not args.refresh:raise FileExistsError('Output exists; preserve historical results or explicitly refresh own QA outputs')
        assert json.loads((out/'results.json').read_text())['generator']=='tactile_mapping.visualize.v1'
    assert tuple(SPLITS['001-rope']['train']) == EPISODES
    before={str(p.relative_to(root)):digest(p) for ep in EPISODES for name in SOURCE_FILES for p in [root/'001-rope'/f'episode_{ep}'/name]}
    expected={line.split()[1]:line.split()[0] for line in (out/'source_sha256.txt').read_text().splitlines() if line.strip()}
    assert expected == {k.replace('\\','/'):v for k,v in before.items()}, 'Downloaded input checksum mismatch'
    out.mkdir(parents=True,exist_ok=True);(out/'images').mkdir(exist_ok=True);(out/'mapped').mkdir(exist_ok=True)
    cmap=style()
    code_paths=['scripts/dataloader/geometry.py','scripts/dataloader/dataset.py','scripts/rebuild_dataset/rebuild.py',
                'docs/DATASET_PROTOCOL.md','docs/proposals/P2_CONDITION_ENCODER_HAND_MODE_AUDIT.md']
    results=dict(generator='tactile_mapping.visualize.v1',created_local=datetime.now(timezone.utc).astimezone().isoformat(),
                 git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
                 script_sha256=digest(__file__),source_hashes=before,code_hashes={p:digest(REPO/p) for p in code_paths},
                 checks=checks,scene='001-rope',sigmas_m=SIGMAS,epsilon=EPS,episodes=[],bandwidth_images=[])
    for ep in EPISODES:
        d=load_episode(root,ep); all_maps={s:map_values(d['values'],d['mask'],d['cp_mask'],s) for s in SIGMAS}
        y,obs=all_maps[.01]; metrics={str(s):summarize(d,*all_maps[s]) for s in SIGMAS}
        for selection in d['selected']:
            selection['time_s']=float(d['seconds'][selection['frame']])
            keyframe_figure(d,selection,y,obs,out/'images'/f"ep{ep}_frame{selection['frame']:04d}.png",cmap)
        time_figure(d,y,obs,metrics['0.01'],out/'images'/f'ep{ep}_time.png',cmap)
        if ep in (2,8):
            frame=next(s['frame'] for s in d['selected'] if s['reason']=='highest valid mean')
            means=masked_mean(d['values'][frame],d['mask'][frame]);f=max(d['active'],key=lambda x:means[x])
            path=f'images/bandwidth_ep{ep}_frame{frame:04d}_stream{f}.png'
            bandwidth_figure(d,f,frame,all_maps,out/path,cmap);results['bandwidth_images'].append(path)
        arrays=dict(frame_ids=np.arange(d['frames']),timestamp_us=d['timestamp_us'],hand_keep=d['hand_keep'],
                    selected_frames=[s['frame'] for s in d['selected']],input_valid_mask=d['mask'],
                    input_values=d['values'],controller_valid_mask=d['cp_mask'],raw_mean_score=d['score'],change_score=d['change'])
        for s,(yy,oo) in all_maps.items():
            arrays[f'mapped_sigma_{s:g}']=yy;arrays[f'observed_sigma_{s:g}']=oo
        np.savez_compressed(out/'mapped'/f'episode_{ep}.npz',**arrays)
        results['episodes'].append({k:d[k] for k in ('episode','frames','active','hand_keep','selected','local_distance_error_m')}|{'metrics':metrics})
        print(f'episode {ep}: selected {[s["frame"] for s in d["selected"]]}',flush=True)
    assert before=={k:digest(root/k) for k in before}, 'Input changed during visualization'
    results['input_unchanged']=True
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write_reports(out,results)
    results['artifact_checks']=verify_artifacts(out,results)
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'output':str(out),'checks_passed':len(checks),'png_count':len(list((out/'images').glob('*.png')))},ensure_ascii=False))


if __name__ == '__main__':
    main()
