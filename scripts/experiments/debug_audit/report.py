"""Assemble existing debug outputs into an offline report. No model/GPU calls.

python -B -m scripts.experiments.debug_audit.report --root /path/to/debug-results
Requires diagnosis/, E1/, E2/ at terminal completion. All three final output
files must be absent. Optional conclusion.json is written by the investigator,
not inferred or fabricated by this reporting utility.
"""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np


OUTPUTS = ("report.html", "results.json", "training_comparison.csv")
QUESTIONS = {
    "Q1": "是否发现真正的代码 Bug？具体在哪里？",
    "Q2": "D0 不受 Condition Encoder 限制时，能否拟合运动？",
    "Q3": "CD-only 与 Full Loss 谁更容易产生有效运动？证据是什么？",
    "Q4": "Condition Encoder 梯度与实际参数更新是否有效？",
    "Q5": "近静态预测最可能关联哪个环节？哪些原因仍未排除？",
    "Q6": "下一轮最值得检查或修改的最小组件是什么？",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else None


def norm(value):
    return math.sqrt(sum(float(v)**2 for v in value)) if value is not None else None


def geometry_data(path, horizon=5):
    """Deterministic display sampling; never used for numeric CD evaluation."""
    with np.load(path, allow_pickle=False) as arrays:
        initial = arrays["initial_xyz_m"]
        predictions = arrays["prediction_xyz_m"]
        frames = arrays["frame_ids"].tolist()
        if predictions.shape != (horizon, len(initial), 3) or len(frames) != horizon+1:
            raise ValueError(f"Incomplete geometry frames: {path}")
        if not np.isfinite(initial).all() or not np.isfinite(predictions).all():
            raise ValueError(f"Nonfinite exported geometry: {path}")
        ids = np.linspace(0, len(initial)-1, min(900, len(initial)), dtype=int)
        targets = []
        for h in range(1, horizon+1):
            target = arrays[f"gt_h{h}_xyz_m"]
            if not np.isfinite(target).all():
                raise ValueError(f"Nonfinite GT geometry: {path}")
            take = np.linspace(0, len(target)-1, min(900, len(target)), dtype=int)
            targets.append((target[take]*1000).tolist())
        unique = len({hashlib.sha256(frame.tobytes()).hexdigest() for frame in predictions})
        return {"initial": (initial[ids]*1000).tolist(),
                "predicted": (predictions[:, ids]*1000).tolist(), "gt": targets,
                "frame_ids": frames, "unique_prediction_states": unique,
                "units": "world millimetres", "display_sampling_only": True}


def source_paths(root, prefix, evidence):
    folder = (root / prefix / evidence["folder"]).resolve()
    if not folder.is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes the report root")
    if evidence["horizon"] != 5 or evidence["cameras"] != 3 or len(evidence["frame_ids"]) != 6:
        raise ValueError("Expected independent five-step, three-camera evidence")
    relative = folder.relative_to(root).as_posix()
    checked = []
    for h in range(1, 6):
        for camera in range(3):
            for mode in ("rgb", "alpha"):
                for label in ("gt", "pred", "copy"):
                    path = folder / f"h{h:02d}-c{camera}-{mode}-{label}.png"
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    with path.open("rb") as stream:
                        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                            raise ValueError(f"Not a PNG: {path}")
                    checked.append(path.relative_to(root).as_posix())
    geometry = geometry_data(folder / "geometry.npz")
    if geometry["frame_ids"] != evidence["frame_ids"]:
        raise ValueError("Geometry and render frame metadata disagree")
    return relative, geometry, checked


def assemble(root, diagnosis_name="diagnosis"):
    root = Path(root).resolve()
    if Path(diagnosis_name).name!=diagnosis_name:raise ValueError('One diagnosis directory name required')
    diagnosis = root / diagnosis_name
    stages = {name: read_json(root / (diagnosis_name if name=='diagnosis' else name) / "status.json") for name in ("diagnosis", "E1", "E2")}
    if any(value.get("status") != "complete" for value in stages.values()):
        raise ValueError("Report requires all audit and paired-fit stages to complete")
    history = read_json(diagnosis / "checkpoint_results.json")
    gradient = read_json(diagnosis / "gradient_audit.json")
    oracle = read_json(diagnosis / "oracle_results.json")
    trials = {mode: {"config": read_json(root / mode / "config.json"),
                    "evaluations": read_json(root / mode / "evaluations.json"),
                    "updates": read_json(root / mode / "updates.json"),
                    "parameter_updates": read_json(root / mode / "parameter_updates.json")}
              for mode in ("E1", "E2")}
    for mode, trial in trials.items():
        if set(trial["evaluations"]) != {"0", "100", "200", "400"}:
            raise ValueError(f"Missing frozen evaluation milestones in {mode}")
        if [row["update"] for row in trial["updates"]] != list(range(1, 401)):
            raise ValueError(f"Not 400 successful optimizer updates in {mode}")
    paired = trials["E1"]["config"]["initial_state_sha256"] == trials["E2"]["config"]["initial_state_sha256"]
    if not paired:
        raise ValueError("E1/E2 initial parameter hashes differ")
    before = read_json(diagnosis / "historical_hashes_before.json")
    after = read_json(diagnosis / "historical_hashes_after.json")
    if before != after:
        raise ValueError("Historical checkpoint hashes changed")
    rows, checked = [], set()

    def add(name, category, metrics, copy_metrics, evidence, prefix, provenance, perturbations=None):
        for key in ("cd_mm", "rgb", "alpha"):
            if len(metrics[key]) != 5 or len(copy_metrics[key]) != 5:
                raise ValueError("Metric horizon mismatch")
        folder, geometry, files = source_paths(root, prefix, evidence)
        checked.update(files)
        rows.append(dict(id=len(rows), name=name, category=category, metrics=metrics,
                         copy=copy_metrics, evidence=dict(evidence, folder=folder),
                         provenance=provenance, geometry=geometry, perturbations=perturbations or {}))

    for record in history:
        metrics = record["metrics"]
        add(f'{record["label"]} · {record["split"]} ep{record["episode"]}/start{record["start"]}',
            "historical", metrics["model"], metrics["copy_initial"], record["evidence"], diagnosis_name,
            {k: v for k, v in record.items() if k not in ("metrics", "evidence")},
            {key: metrics[key] for key in ("freeze_action", "zero_tactile")})
    for mode, trial in trials.items():
        for update, evaluation in sorted(trial["evaluations"].items(), key=lambda item: int(item[0])):
            metrics = evaluation["metrics"]
            add(f'{mode} {"CD-only" if mode == "E1" else "Full Loss"} · update {update}',
                "paired_fit", metrics["model"], metrics["copy_initial"], evaluation["evidence"], mode,
                dict(scene="001-rope", split="train", episode=2, start=73, update=int(update),
                     checkpoint=f"{mode}/checkpoint_{update}.pt", config=trial["config"]),
                {key: metrics[key] for key in ("freeze_action", "zero_tactile")})
    for mode, result in oracle.items():
        add(f'Oracle D0 {mode} · update 400', "offline_oracle", result["final"],
            result["baselines"]["copy_initial"], result["visuals"]["final"], diagnosis_name,
            dict(scene="001-rope", split="train", episode=2, start=73, update=400,
                 checkpoint=f"{diagnosis_name}/oracle-{mode}/prediction_400.pt", config=result["config"]))
    result = oracle["cd"]
    add("GT 质心平移 Oracle · 非真实材料运动", "offline_oracle",
        result["baselines"]["centroid_translation"], result["baselines"]["copy_initial"],
        result["visuals"]["centroid"], diagnosis_name,
        dict(scene="001-rope", split="train", episode=2, start=73, update=0,
             checkpoint=f"{diagnosis_name}/oracle-cd/centroid_translation.pt", offline_target_oracle=True))
    conclusion = read_json(root / "conclusion.json") if (root / "conclusion.json").exists() else None
    extra = {name: read_json(root / name) for name in ("data_audit.json", "execution_manifest.json", "acceptance.json", "sampling_audit.json", "checkpoint_seen.json")
             if (root / name).exists()}
    return dict(schema="touch4d-debug-audit-report-v1", rows=rows, stages=stages, gradients=gradient,
                oracle=oracle, trials=trials, historical_checkpoint_sha256=before,
                paired_initial_parameters_exact=paired, image_files_checked=len(checked),
                historical_checkpoint_hashes_unchanged=True, conclusion=conclusion, extra=extra,
                questions=QUESTIONS, limitations=[
                    "H5 单窗口过拟合和离线 Oracle 是表达/优化诊断，不能证明泛化或动作条件动力学已被学会。",
                    "GT 各帧为独立重建点集，质心变化不等于真实材料粒子运动，不按行监督跨帧 GT。",
                    "历史 H15 为 H5 权重外推；本报告数值与训练对照均为 H5。",
                    "条件扰动对输出有影响，不等于条件改善预测；ΔCD=扰动−真实，正值才表示该对照中真实条件更好。",
                    "同一已训练权重上的单批次梯度夹角只能提供局部竞争证据，不能独立证明全训练近静态退化的根因。"])


def csv_rows(results):
    for row in results["rows"]:
        m, cp = row["metrics"], row["copy"]
        for h in range(5):
            motion = m.get("motion", [{}]*5)[h]
            anchor = m.get("anchors", [{}]*5)[h]
            nr = m.get("nonrigid_residual_rms_mm", [None]*5)[h]
            nr = motion.get("nonrigid_residual_rms_mm") if nr is None else nr
            perturb = row["perturbations"]
            yield dict(experiment=row["name"], category=row["category"],
                       episode=row["provenance"].get("episode"), start=row["provenance"].get("start"),
                       update=row["provenance"].get("update", row["provenance"].get("completed_updates")),
                       horizon=h+1, frame=row["evidence"]["frame_ids"][h+1], cd_mm=m["cd_mm"][h],
                       copy_cd_mm=cp["cd_mm"][h], delta_cd_vs_copy_mm=m["cd_mm"][h]-cp["cd_mm"][h],
                       rgb=m["rgb"][h], alpha=m["alpha"][h], centered_cd_mm=motion.get("centered_cd_mm"),
                       displacement_rms_mm=motion.get("displacement_rms_mm"), residual_rms_mm=nr,
                       predicted_centroid_motion_mm=norm(motion.get("predicted_centroid_motion_mm", motion.get("displacement_mean_mm"))),
                       gt_set_centroid_motion_mm=norm(motion.get("gt_centroid_motion_mm")),
                       centroid_error_mm=motion.get("centroid_error_mm"),
                       anchor_rms_mm=anchor.get("rms_mm", motion.get("anchor_rms_mm")),
                       anchor_max_mm=anchor.get("max_mm", motion.get("anchor_max_mm")),
                       freeze_action_delta_cd_mm=perturb.get("freeze_action", {}).get("delta_cd_mm", [None]*5)[h],
                       zero_tactile_delta_cd_mm=perturb.get("zero_tactile", {}).get("delta_cd_mm", [None]*5)[h])


def render(results):
    # JSON is data, never executable markup. Escape HTML script terminators.
    payload = json.dumps(results, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    return PAGE.replace("__PAYLOAD__", payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--diagnosis-dir", default="diagnosis")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    for name in OUTPUTS:
        if (root / name).exists():
            raise FileExistsError(root / name)
    result = assemble(root,args.diagnosis_dir)
    page = render(result)
    # Keep geometry only in the self-contained HTML; exact arrays remain NPZ.
    numerical = dict(result, rows=[{k: v for k, v in row.items() if k != "geometry"} for row in result["rows"]])
    with (root / "results.json").open("x", encoding="utf-8") as stream:
        json.dump(numerical, stream, ensure_ascii=False, indent=2, allow_nan=False)
    records = list(csv_rows(result))
    with (root / "training_comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    with (root / "report.html").open("x", encoding="utf-8") as stream:
        stream.write(page)
    print(json.dumps({"report": str(root / "report.html"), "records": len(result["rows"]),
                      "png_files_checked": result["image_files_checked"], "conclusion_present": result["conclusion"] is not None}))


PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Touch4D · P3–P4 近静态退化 Debug Audit</title>
<style>:root{color-scheme:light;--blue:#17669e;--red:#b64235;--green:#298361}*{box-sizing:border-box}body{margin:0;background:#f4f6f9;color:#203040;font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1460px;margin:auto;padding:24px}h1{font-size:27px;margin-bottom:6px}h2{font-size:21px}h3{font-size:16px;margin:8px 0}.card{background:white;border:1px solid #dce3eb;border-radius:12px;padding:18px;margin:18px 0}.note{padding:12px 16px;background:#fff3d8;border-left:4px solid #cf9620}.muted{color:#536578}.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center}.controls label{display:flex;gap:7px;align-items:center}select,button{padding:7px;border:1px solid #b9c8d5;border-radius:6px;background:white;max-width:100%}button{cursor:pointer}input[type=range]{min-width:240px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #e3e9ef;white-space:nowrap}th{background:#edf3f8;position:sticky;top:0}td.num{font-variant-numeric:tabular-nums}.scroll{overflow:auto}tr.selected{background:#fff3d8}.tri{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}figure{margin:0;min-width:0}figure img{display:block;width:100%;background:#f2f2f2;border:1px solid #dce3eb;min-height:100px}figcaption{font-weight:600;margin-bottom:7px}canvas{width:100%;height:240px;border:1px solid #dce3eb;border-radius:6px}pre{white-space:pre-wrap;word-break:break-all;max-height:380px;overflow:auto;background:#f7f9fc;padding:12px;font-size:12px}a{color:var(--blue)}.pill{padding:4px 8px;border-radius:5px;background:#eef4fa}.bad{color:var(--red)}.good{color:var(--green)}details{margin:10px 0}summary{cursor:pointer;font-weight:600}@media(max-width:800px){main{padding:12px}.tri,.two{grid-template-columns:1fr}canvas{height:220px}}
</style><main><h1>Touch4D · P3–P4 近静态预测根因调查</h1><p class="muted">独立 Debug 分支 · 原模型与旧 checkpoint 保持只读 · H5 · 单训练窗口 ep2 / start73 · 有界 400 updates</p>
<div class="note">本报告区分工程正确性、局部表达能力与科学假设。离线 Oracle 可读取目标以拟合上限；正常预测只读取初态和因果条件。验证 ep3 只读，不参与本轮训练或调参。</div>
<section class="card"><h2>根因结论与证据级别</h2><div id="conclusion"></div><details><summary>范围与解释边界</summary><ul id="limits"></ul></details></section>
<section class="card"><h2>执行验收</h2><div class="scroll" id="status"></div><p id="integrity"></p><details><summary>源码、Job ID 与冻结 checkpoint 索引</summary><pre id="manifest"></pre></details></section>
<section class="card"><h2>历史 checkpoint、配对训练与 Oracle</h2><p class="muted">选择一条记录查看五步结果。历史 best/last 按原有规则保留；E1/E2 完整展示 0 / 100 / 200 / 400，不按本轮结果挑选有利 checkpoint。</p><div class="controls"><label>记录 <select id="record"></select></label><label>h <input id="h" type="range" min="1" max="5" value="5"></label><button id="play">播放</button><span class="pill" id="frame"></span></div><div class="controls" style="margin-top:12px"><label>相机 <select id="camera"><option value="0">Camera 0</option><option value="1">Camera 1</option><option value="2">Camera 2</option></select></label><label>模态 <select id="modality"><option value="rgb">RGB</option><option value="alpha">Alpha</option></select></label></div><p id="case"></p><div class="tri"><figure><figcaption>GT：目标 GS 渲染</figcaption><img id="gt" alt="GT"></figure><figure><figcaption>所选预测 / Oracle</figcaption><img id="pred" alt="Prediction"></figure><figure><figcaption>复制初态</figcaption><img id="copy" alt="Copy Initial"></figure></div><p class="muted" id="images"></p><div class="scroll" id="horizons"></div><details><summary>该结果的完整配置与来源</summary><pre id="provenance"></pre></details></section>
<section class="card"><h2>Gaussian 中心与预测轨迹</h2><p class="muted">确定性抽样仅用于显示（每帧最多 900 点），指标仍使用完整点集。GT 每帧独立抽样，不画跨帧 GT 轨迹；预测保持初态模板身份。此处为中心投影，不是椭球渲染。</p><div class="controls"><label>投影 <select id="projection"><option value="0,1">世界 XY</option><option value="0,2">世界 XZ</option><option value="1,2">世界 YZ</option></select></label><label><input id="tracks" type="checkbox" checked>预测轨迹（初态→当前 h）</label><a id="npz" href="#">完整几何 NPZ</a></div><div class="tri"><figure><figcaption>GT centers</figcaption><canvas id="g_gt"></canvas></figure><figure><figcaption>Pred centers / trajectories</figcaption><canvas id="g_pred"></canvas></figure><figure><figcaption>Copy centers</figcaption><canvas id="g_copy"></canvas></figure></div></section>
<section class="card"><h2>五步几何误差与运动幅度</h2><div class="two"><figure><figcaption>CD / Copy CD / 去质心 CD（mm）</figcaption><canvas id="cd_chart"></canvas></figure><figure><figcaption>预测 RMS / 非刚性 RMS / GT 集合质心变化（mm）</figcaption><canvas id="motion_chart"></canvas></figure></div><p class="muted">GT 集合质心变化受独立重建、遮挡与采样影响，不代表对应材料点的真实运动。非刚性 RMS 仅描述预测中去掉共同平移后的剩余位移。</p></section>
<section class="card"><h2>全部结果的固定口径汇总</h2><p>ΔCD = Pred − Copy：负值为几何改善。Action/Tactile ΔCD = 扰动 − 真实：正值表示该对照中真实条件有益。</p><div class="scroll" id="summary"></div><p><a href="training_comparison.csv">逐 horizon CSV</a> · <a href="results.json">完整数值 JSON</a></p></section>
<section class="card"><h2>配对 400-step 曲线</h2><div class="two"><figure><figcaption>训练 Loss（单位与权重保持原定义）</figcaption><canvas id="fit_loss"></canvas></figure><figure><figcaption>共同检查点 CD（mm）</figcaption><canvas id="fit_cd"></canvas></figure></div><div class="two"><figure><figcaption>Oracle CD-only / Full：CD（mm）</figcaption><canvas id="oracle_cd"></canvas></figure><figure><figcaption>训练记录的未加权 CD 项（归一化）</figcaption><canvas id="train_geometry"></canvas></figure></div><p class="muted">E1/E2 Loss 组成不同，不以总 Loss 数值大小直接比较拟合质量。Oracle 独立位移参数 lr=1e-3、无 weight decay；网络 lr=1e-4，属于不同容量诊断，不是相同优化器控制变量。</p></section>
<section class="card"><h2>单批次 Loss 与梯度</h2><p id="gradient_note"></p><div class="scroll" id="losses"></div><div class="controls"><label>Loss <select id="grad_loss"><option>full</option><option>cd</option><option>rgb</option><option>alpha</option></select></label></div><div class="scroll" id="grad_modules"></div><h3>方向关系：相同参数、相同 Batch</h3><div class="scroll" id="cosines"></div><details><summary>原始梯度证据 JSON</summary><pre id="grad_json"></pre></details></section>
<section class="card"><h2>配对训练中的实际参数更新</h2><div class="controls"><label>实验 <select id="update_mode"><option>E1</option><option>E2</option></select></label><label>成功更新 <select id="update_step"></select></label></div><div class="scroll" id="updates"></div><p class="muted">参数范数、原始梯度、裁剪梯度、AdamW 实际更新量分别显示。零梯度不一定等同零更新，AdamW 历史动量与 weight decay 会影响更新。</p></section>
</main><script>"use strict";const D=__PAYLOAD__;const $=id=>document.getElementById(id);const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const num=x=>x==null?'—':Number(x).toPrecision(6);const avg=v=>{const a=(v||[]).filter(x=>x!=null&&Number.isFinite(Number(x)));return a.length?a.reduce((s,x)=>s+Number(x),0)/a.length:null};const len=v=>v?Math.sqrt(v.reduce((s,x)=>s+x*x,0)):null;const table=(headers,rows)=>'<table><thead><tr>'+headers.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';const option=(el,v,label)=>{const o=document.createElement('option');o.value=v;o.textContent=label;el.appendChild(o)};
function drawLine(id,series,xlabels=null){const el=$(id),r=el.getBoundingClientRect(),ratio=devicePixelRatio||1;el.width=r.width*ratio;el.height=r.height*ratio;const c=el.getContext('2d');c.scale(ratio,ratio);const w=r.width,h=r.height,L=58,R=16,T=38,B=30;c.clearRect(0,0,w,h);const finite=series.flatMap(s=>s.values.filter(Number.isFinite));let lo=Math.min(0,...finite),hi=Math.max(...finite,lo+1e-9);if(!finite.length){c.fillText('无可用数值',20,30);return}const span=hi-lo;hi+=span*.06;lo-=span*.03;c.font='11px system-ui';for(let j=0;j<=4;j++){const y=T+(h-T-B)*j/4;c.strokeStyle='#e4e9ef';c.beginPath();c.moveTo(L,y);c.lineTo(w-R,y);c.stroke();c.fillStyle='#536578';c.fillText((hi-(hi-lo)*j/4).toPrecision(3),4,y+4)}let ox=L;series.forEach(s=>{c.fillStyle=s.color;c.fillRect(ox,12,10,4);c.fillText(s.name,ox+14,18);ox+=c.measureText(s.name).width+30;c.strokeStyle=s.color;c.lineWidth=1.7;c.beginPath();let live=false;s.values.forEach((v,i)=>{if(v==null||!Number.isFinite(v)){live=false;return}const x=L+(w-L-R)*i/Math.max(1,s.values.length-1),y=T+(h-T-B)*(hi-v)/(hi-lo);if(live)c.lineTo(x,y);else c.moveTo(x,y);live=true});c.stroke()});const n=Math.max(...series.map(s=>s.values.length));[0,Math.floor((n-1)/2),n-1].forEach(i=>{c.fillStyle='#536578';c.fillText(xlabels?xlabels[i]:String(i+1),L+(w-L-R)*i/Math.max(1,n-1)-4,h-8)})}
function geometry(row,h){const g=row.geometry,[a,b]=$('projection').value.split(',').map(Number),all=[g.initial,...g.gt,...g.predicted].flat();let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;all.forEach(p=>{xmin=Math.min(xmin,p[a]);xmax=Math.max(xmax,p[a]);ymin=Math.min(ymin,p[b]);ymax=Math.max(ymax,p[b])});let span=Math.max(xmax-xmin,ymax-ymin,1)*1.12,cx=(xmin+xmax)/2,cy=(ymin+ymax)/2;for(const [id,points,color] of [['g_gt',g.gt[h-1],'#298361'],['g_pred',g.predicted[h-1],'#17669e'],['g_copy',g.initial,'#8b9299']]){const el=$(id),r=el.getBoundingClientRect(),d=devicePixelRatio||1;el.width=r.width*d;el.height=r.height*d;const c=el.getContext('2d');c.scale(d,d);const unit=Math.min(r.width-28,r.height-28)/span;const xy=p=>[r.width/2+(p[a]-cx)*unit,r.height/2-(p[b]-cy)*unit];if(id==='g_pred'&&$('tracks').checked){c.strokeStyle='#d99c55';c.globalAlpha=.55;const stride=Math.max(1,Math.floor(g.initial.length/70));for(let i=0;i<g.initial.length;i+=stride){c.beginPath();let p=xy(g.initial[i]);c.moveTo(...p);for(let k=0;k<h;k++){p=xy(g.predicted[k][i]);c.lineTo(...p)}c.stroke()}}c.globalAlpha=.8;c.fillStyle=color;points.forEach(p=>{const q=xy(p);c.fillRect(q[0],q[1],2,2)});c.globalAlpha=1;c.fillStyle='#536578';c.font='11px system-ui';c.fillText('同一世界坐标范围 / mm',8,r.height-8)}$('npz').href=row.evidence.folder+'/geometry.npz'}
function detail(){const row=D.rows[+$('record').value],h=+$('h').value,c=$('camera').value,mode=$('modality').value,m=row.metrics,motion=m.motion||[];$('frame').textContent=`h=${h} / frame ${row.evidence.frame_ids[h]}`;$('case').textContent=`${row.name}；初态 frame ${row.evidence.frame_ids[0]}；目标 ${row.evidence.frame_ids.slice(1).join(', ')}。`;
for(const label of ['gt','pred','copy'])$(label).src=`${row.evidence.folder}/h${String(h).padStart(2,'0')}-c${c}-${mode}-${label}.png`;$('images').textContent=`几何导出中 ${row.geometry.unique_prediction_states}/5 个不同预测状态（零初始化的 update 0 与 Copy 可能全部相同；不以文件唯一性代替运动幅度）。`;
$('horizons').innerHTML=table(['h','Frame','CD mm','Copy CD','RGB MAE','Alpha MAE','Pred RMS mm','非刚性 RMS','GT质心变化 mm','Anchor RMS'],m.cd_mm.map((v,k)=>[k+1,row.evidence.frame_ids[k+1],num(v),num(row.copy.cd_mm[k]),num(m.rgb[k]),num(m.alpha[k]),num(motion[k]?.displacement_rms_mm),num(m.nonrigid_residual_rms_mm?.[k]??motion[k]?.nonrigid_residual_rms_mm),num(len(motion[k]?.gt_centroid_motion_mm)),num(m.anchors?.[k]?.rms_mm??motion[k]?.anchor_rms_mm)]));$('provenance').textContent=JSON.stringify(row.provenance,null,2);drawLine('cd_chart',[{name:'Pred',color:'#17669e',values:m.cd_mm},{name:'Copy',color:'#8b9299',values:row.copy.cd_mm},{name:'去质心',color:'#298361',values:motion.map(v=>v.centered_cd_mm)}]);drawLine('motion_chart',[{name:'Pred RMS',color:'#17669e',values:motion.map(v=>v.displacement_rms_mm)},{name:'非刚性',color:'#b64235',values:m.nonrigid_residual_rms_mm??motion.map(v=>v.nonrigid_residual_rms_mm)},{name:'GT质心',color:'#298361',values:motion.map(v=>len(v.gt_centroid_motion_mm))}]);geometry(row,h)}
function grad(){const g=D.gradients,loss=$('grad_loss').value;$('grad_modules').innerHTML=table(['模块','参数数','参数Norm','原始梯度Norm','裁剪梯度Norm','AdamW更新Norm','相对更新','有梯度张量','非零梯度张量','更新张量'],Object.entries(g[loss].modules).map(([k,v])=>[k,v.parameter_count,num(v.parameter_norm),num(v.raw_gradient_norm),num(v.clipped_gradient_norm),num(v.actual_update_norm),num(v.relative_update_norm),v.tensors_with_gradient,v.tensors_with_nonzero_gradient,v.tensors_updated]));$('grad_json').textContent=JSON.stringify(g[loss],null,2)}
function updateSteps(){const x=D.trials[$('update_mode').value].parameter_updates;$('update_step').innerHTML='';x.forEach((v,i)=>option($('update_step'),i,v.update));updates()}
function updates(){const x=D.trials[$('update_mode').value].parameter_updates[+$('update_step').value];$('updates').innerHTML=table(['模块','参数Norm','原始梯度Norm','裁剪梯度Norm','实际更新Norm','相对更新'],Object.entries(x.actual).map(([k,v])=>[k,num(v.parameter_norm),num(x.raw[k].gradient_norm),num(x.clipped[k].gradient_norm),num(v.update_norm),num(v.relative_update)]))}
function curves(){const colors={E1:'#17669e',E2:'#b64235'};drawLine('fit_loss',Object.entries(D.trials).map(([k,t])=>({name:k,color:colors[k],values:t.updates.map(v=>v.loss)})));drawLine('train_geometry',Object.entries(D.trials).map(([k,t])=>({name:k,color:colors[k],values:t.updates.map(v=>v.geometry)})));drawLine('fit_cd',Object.entries(D.trials).map(([k,t])=>({name:k,color:colors[k],values:['0','100','200','400'].map(s=>t.evaluations[s].metrics.model.mean_cd_mm)})),['0','100','200','400']);drawLine('oracle_cd',Object.entries(D.oracle).map(([k,o])=>({name:k,color:k==='cd'?'#17669e':'#b64235',values:['0','100','200','400'].map(s=>o.milestones[s].mean_cd_mm)})),['0','100','200','400'])}
$('limits').innerHTML=D.limitations.map(x=>'<li>'+esc(x)+'</li>').join('');$('status').innerHTML=table(['阶段','状态','SHA','Job','耗时 s','峰值 allocated GiB'],Object.entries(D.stages).map(([k,v])=>[k,v.status,v.source_sha,v.job_id,num(v.elapsed_seconds),num(v.peak_allocated_bytes/2**30)]));$('integrity').textContent=`E1/E2 初始参数哈希一致：${D.paired_initial_parameters_exact}；历史 checkpoint 哈希未改变：${D.historical_checkpoint_hashes_unchanged}；已检查 ${D.image_files_checked} 个 PNG 文件。`;$('manifest').textContent=JSON.stringify({stages:D.stages,historical_checkpoint_sha256:D.historical_checkpoint_sha256,extra:D.extra},null,2);
if(D.conclusion){const q=D.conclusion.questions??D.conclusion.answers??D.conclusion;$('conclusion').innerHTML=Object.entries(D.questions).map(([k,title])=>'<h3>'+esc(k+' · '+title)+'</h3><p>'+esc(typeof q[k]==='object'?JSON.stringify(q[k]):q[k]??'待研究者填写')+'</p>').join('');if(D.conclusion.evidence)$('conclusion').innerHTML+='<h3>根因证据表</h3><pre>'+esc(JSON.stringify(D.conclusion.evidence,null,2))+'</pre>';if(D.conclusion.summary)$('conclusion').innerHTML='<p><strong>'+esc(D.conclusion.summary)+'</strong></p>'+$('conclusion').innerHTML}else{$('conclusion').innerHTML='<p class="note">尚未提供 conclusion.json。以下为真实指标展示，报告程序不会自动把相关性写成因果结论。</p>'+Object.entries(D.questions).map(([k,t])=>'<p>'+esc(k+' · '+t)+' — 待研究者集中验收</p>').join('')}
D.rows.forEach((r,i)=>option($('record'),i,r.name));const finalE2=D.rows.findIndex(r=>r.name==='E2 Full Loss · update 400');if(finalE2>=0)$('record').value=finalE2;$('summary').innerHTML=table(['实验 / 窗口','Mean CD mm','Copy CD','Pred−Copy','RGB','Alpha','Action ΔCD','Tactile ΔCD','h5 Pred RMS','h5 非刚性 RMS'],D.rows.map(r=>[r.name,num(avg(r.metrics.cd_mm)),num(avg(r.copy.cd_mm)),num(avg(r.metrics.cd_mm)-avg(r.copy.cd_mm)),num(avg(r.metrics.rgb)),num(avg(r.metrics.alpha)),num(avg(r.perturbations.freeze_action?.delta_cd_mm)),num(avg(r.perturbations.zero_tactile?.delta_cd_mm)),num(r.metrics.motion?.[4]?.displacement_rms_mm),num(r.metrics.nonrigid_residual_rms_mm?.[4]??r.metrics.motion?.[4]?.nonrigid_residual_rms_mm)]));const g=D.gradients;$('gradient_note').textContent=`冻结 S0 checkpoint；优化器起点：${g.optimizer_start}。${g.note}`;$('losses').innerHTML=table(['反向项','Loss','CD归一化','RGB','Alpha','裁剪前Norm','裁剪后Norm','裁剪倍率','更新前CD mm','一步更新后CD mm'],['cd','rgb','alpha','full'].map(k=>{const v=g[k];return[k,num(v.loss),num(v.unweighted_components.geometry),num(v.unweighted_components.rgb),num(v.unweighted_components.alpha),num(v.global_gradient_norm_before_clip),num(v.global_gradient_norm_after_clip),num(v.effective_clip_multiplier),num(v.cd_before_mm),num(v.cd_after_one_update_mm)]}));let cos=[];Object.entries(g.comparisons).forEach(([pair,groups])=>['anchor_displacement_head','condition_encoders_combined','all_parameters'].forEach(name=>{const v=groups[name];if(v)cos.push([pair,name,num(v.raw_gradient_cosine),num(v.clipped_gradient_cosine),num(v.actual_adamw_update_cosine)])}));$('cosines').innerHTML=table(['对照','参数组','原始梯度Cos','裁剪梯度Cos','实际AdamW更新Cos'],cos);
['record','camera','modality','projection','tracks'].forEach(id=>$(id).onchange=detail);$('h').oninput=detail;$('grad_loss').onchange=grad;$('update_mode').onchange=updateSteps;$('update_step').onchange=updates;let timer=null;$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放'}else{timer=setInterval(()=>{$('h').value=(+$('h').value%5)+1;detail()},700);$('play').textContent='暂停'}};window.onresize=()=>{detail();curves()};detail();grad();updateSteps();curves();</script></html>'''


if __name__ == "__main__":
    main()
