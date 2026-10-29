"""CPU-only plots and Chinese report from completed frozen inference records."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .interventions import MODES
from .run import write_json

COLORS = dict(zip((*MODES,"Copy"),("black","#db4437","#e98e13","#1a73e8","#34a853","#9c27b0","#999999")))
BRANCH_COLORS = dict(base="#db4437", global_="#1a73e8", local="#34a853", correction="#e98e13", final="black")


def table(headers,rows):
    return ("| "+" | ".join(headers)+" |\n| "+" | ".join("---" for _ in headers)+" |\n"+
            "\n".join("| "+" | ".join(map(str,r))+" |" for r in rows)+"\n")


def f(value,digits=6):
    return "未定义" if value is None else f"{value:.{digits}f}"


def save(fig,path):
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path,dpi=150,bbox_inches="tight")
    plt.close(fig)


def overview(out,data):
    curves=data["horizon_val_summary"]
    fig,axs=plt.subplots(2,2,figsize=(14,9))
    for mode in (*MODES,"Copy"):
        rows=sorted((r for r in curves if r["mode"]==mode),key=lambda r:r["horizon"])
        for ax,key in zip(axs.flat,("cd_mm","rgb","alpha","joint")):
            ax.plot([r["horizon"] for r in rows],[r[key] for r in rows],label=mode,color=COLORS[mode])
            ax.set_title("Full Val100, episode-equal "+key);ax.set_xlabel("Horizon");ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8)
    save(fig,out/"figures"/"val100_horizon_metrics.png")
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    for mode in MODES[1:]:
        rows=sorted((r for r in data["same_history_val_summary"] if r["mode"]==mode),key=lambda r:r["horizon"])
        for ax,key in zip(axs,("delta_cd_mm","delta_joint")):
            ax.plot([r["horizon"] for r in rows],[r[key] for r in rows],"o-",label=mode,color=COLORS[mode])
            ax.axhline(0,color="gray",lw=.8);ax.set_title("Same Normal AR history: "+key);ax.set_xlabel("Horizon")
    axs[0].legend(fontsize=8)
    save(fig,out/"figures"/"same_history_changes.png")
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    for region in ("all","near","middle","far"):
        rows=sorted((r for r in data["cancellation_val_summary"] if r["space"]=="gs_actual" and r["region"]==region),key=lambda r:r["horizon"])
        for ax,key in zip(axs,("base_vs_rest_cos_mean","cancellation_index")):
            ax.plot([r["horizon"] for r in rows],[np.nan if r[key] is None else r[key] for r in rows],label=region)
            ax.set_title("Actual GS, episode-equal "+key);ax.set_xlabel("Horizon")
    axs[0].axhline(0,color="gray",lw=.8);axs[0].legend()
    save(fig,out/"figures"/"val100_cancellation.png")


def case_plots(out,data):
    generated=[]
    fields=data["branch_fields"];cancellation=data["cancellation"]
    for path in sorted((out/"arrays").glob("*.npz")):
        key=path.stem
        # stem split_epN_sSTART is also the source record identity.
        split,ep,start=key.split("_");ep=int(ep[2:]);start=int(start[1:])
        rows=[r for r in fields if r["split"]==split and r["episode"]==ep and r["start"]==start and
              r["space"]=="gs_actual" and r["region"]=="all"]
        fig,axs=plt.subplots(2,3,figsize=(16,8))
        for branch in BRANCH_COLORS:
            current=sorted((r for r in rows if r["branch"]==branch),key=lambda r:r["horizon"])
            for ax,name in zip(axs[0],("mean_x_mm","mean_y_mm","mean_z_mm")):
                ax.plot([r["horizon"] for r in current],[r[name] for r in current],label=branch,color=BRANCH_COLORS[branch])
                ax.set_title(name);ax.axhline(0,color="gray",lw=.5)
            axs[1,0].plot([r["horizon"] for r in current],[r["rms_mm"] for r in current],label=branch,color=BRANCH_COLORS[branch])
        c=sorted((r for r in cancellation if r["split"]==split and r["episode"]==ep and r["start"]==start
                  and r["space"]=="gs_actual" and r["region"]=="all"),key=lambda r:r["horizon"])
        for ax,name in zip(axs[1,1:],("base_vs_rest_cos_mean","cancellation_index")):
            ax.plot([r["horizon"] for r in c],[np.nan if r[name] is None else r[name] for r in c],"o-")
            ax.set_title(name)
        axs[1,0].set_title("Actual GS displacement RMS (mm)")
        for ax in axs.flat:ax.set_xlabel("Horizon");ax.grid(alpha=.15)
        axs[0,0].legend(fontsize=8)
        fig.suptitle(key+" / Normal-history branch decomposition; not material strain")
        dest=out/"figures"/(key+"_motion_decomposition.png")
        save(fig,dest);generated.append(dest.relative_to(out).as_posix())
        arrays=np.load(path)
        if (ep,start) not in ((3,135),(3,245),(6,110),(7,105)):
            continue
        all_points=[arrays["initial_world"],*[arrays[mode+"_world"].reshape(-1,3) for mode in (*MODES,"Copy")],
                    *[arrays[f"gt_h{h:02}_world"] for h in range(1,16)]]
        minimum=np.min(np.stack([p.min(0) for p in all_points]),0)
        maximum=np.max(np.stack([p.max(0) for p in all_points]),0)
        pad=np.maximum((maximum-minimum)*.05,.002)
        lo,hi=(minimum-pad)*1000,(maximum+pad)*1000
        # One identical world-space box across modes AND H1-H15 for this window.
        for h in (1,5,10,15):
            fig=plt.figure(figsize=(17,9))
            for index,name in enumerate(("GT",*MODES,"Copy")):
                points=arrays[f"gt_h{h:02}_world"] if name=="GT" else arrays[name+"_world"][h-1]
                ax=fig.add_subplot(2,4,index+1,projection="3d")
                p=points*1000
                ax.scatter(p[:,0],p[:,1],p[:,2],s=.3,color="#336699",alpha=.6,rasterized=True)
                ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),title=name,
                       xlabel="X mm",ylabel="Y mm",zlabel="Z mm")
                ax.set_box_aspect(hi-lo);ax.view_init(elev=20,azim=-60)
            fig.suptitle(f"{key} H{h}; fixed WORLD box, all modes/frames included; independent GT sets")
            dest=out/"figures"/f"{key}_h{h:02}_geometry.png"
            save(fig,dest);generated.append(dest.relative_to(out).as_posix())
    return generated


def build(out):
    if not (out/"completed.json").is_file():
        raise RuntimeError("Inference is incomplete; no scientific report will be fabricated")
    if (out/"final_report.md").exists():
        raise FileExistsError("Do not overwrite an existing report")
    names=("val_summary","horizon_val_summary","same_history_val_summary","branch_fields","cancellation",
           "cancellation_val_summary","windows","horizons","structure","equivalence","intervention_checks")
    data={name:json.loads((out/"results"/(name+".json")).read_text()) for name in names}
    provenance=json.loads((out/"provenance.json").read_text())
    overview(out,data);figures=case_plots(out,data)
    s=data["val_summary"];normal=s["Normal"]
    report=[
        "# Touch4D G5 Frozen Motion Attribution V1\n\n",
        "本报告仅针对 G5-Direct Epoch8 Best 内部联合训练后的三个位移分支；base_d 不是历史独立 G0 Checkpoint。\n\n",
        "## 1. 固定对象与实际执行\n\n",
        f"- 训练源码：{provenance['training_source_sha']}\n",
        f"- 诊断源码：{provenance['diagnostic_source_sha']}\n",
        f"- Checkpoint：{provenance['checkpoint']}\n",
        f"- SHA256：{provenance['checkpoint_sha256']}\n",
        f"- Slurm Job：{provenance['job_id']}；GPU：{provenance['gpu']}；"
        f"PyTorch/CUDA/gsplat：{provenance['torch']}/{provenance['cuda']}/{provenance['gsplat']}\n",
        f"- 耗时 {provenance['elapsed_seconds']/60:.2f} min；峰值已分配显存 "
        f"{provenance['peak_allocated_bytes']/2**30:.3f} GiB。\n",
        "- 完整 Val100（ep3=55、ep6=45），另只读 Train ep7/start105。无训练、优化器、反向传播或参数搜索。\n\n",
        "## 2. 消融语义与数值等价性\n\n",
        "Normal=base+global+local；NoBase-All 从 H1 关闭 base；NoBase-H2+ 仅 H1 保留 base；"
        "NoGlobal 只移除最终全局平移，Local 读取的原始 G4 per-hand context 保持不变；"
        "NoLocal 只移除局部输出；BaseOnly 只保留本 Checkpoint 内部 base。\n\n",
        "A：取 Normal 的相同历史，H1/5/10/15 单步重组；只改变当前累计 Anchor 输出。"
        "B：六条件各自完整 H15 AR，previous_d/GS 历史自然随输出改变。A 与 B 不可混为当前步独立贡献。\n\n",
        f"Anchor 加和逐位等价检查 {len(data['equivalence'])} 条；RBF 分支场之和与最终场最大误差 "
        f"{max(r['projected_sum_max_error_mm'] for r in data['equivalence']):.9f} mm；"
        f"独立单步 Hook/重组前缀与当前步逐位检查 {len(data['intervention_checks'])} 条。\n\n",
        "H1 的 G4/G5 有符号基底为零；Normal 的 H1 因而仅依赖 base，NoBase-All 的 H1 是 Copy。"
        "这不预设 H1 的 base 必然优于 Copy，需要下表实测。\n\n",
        "## 3. 完整 Val100 结果（Episode 等权）\n\n",
        table(["条件","CD mm","RGB","Alpha","Joint","ΔCD vs Normal","ΔJoint"],
              [[mode,f(v["geometry_mm"]),f(v["rgb"]),f(v["alpha"]),f(v["objective"]),
                f(v["geometry_mm"]-normal["geometry_mm"]),f(v["objective"]-normal["objective"])]
               for mode,v in s.items()]),
        "\nΔ<0 为该冻结干预下改善。Normal 已与 Checkpoint 保存的完整验证摘要核对；"
        "Joint 使用归一化 CD，不将毫米与 RGB 直接相加。\n\n",
    ]
    for ep in ("3","6"):
        report.extend([f"### ep{ep} 单独汇总\n\n",
            table(["条件","窗口数","CD mm","RGB","Alpha","Joint"],
                [[mode,v["episodes"][ep]["samples"],f(v["episodes"][ep]["geometry_mm"]),
                  f(v["episodes"][ep]["rgb"]),f(v["episodes"][ep]["alpha"]),
                  f(v["episodes"][ep]["supervised_geometry"]+v["episodes"][ep]["rgb"]+.1*v["episodes"][ep]["alpha"])]
                 for mode,v in s.items()]),"\n"])
    report.extend(["## 4. 同 Normal AR 历史的单步直接效应\n\n",
        table(["条件","H","CD mm","ΔCD mm","ΔRGB","ΔAlpha","ΔJoint","XYZ变化 RMS mm"],
            [[r["mode"],r["horizon"],f(r["cd_mm"]),f(r["delta_cd_mm"]),f(r["delta_rgb"]),
              f(r["delta_alpha"]),f(r["delta_joint"]),f(r["change_rms_mm"])]
             for r in data["same_history_val_summary"]]),
        "\n以上是当前步干预，不把后续历史变化混入。全部逐窗记录在 results/same_history.csv/json。\n\n",
        "## 5. 分支反向运动与抵消\n\n",
        "各分支先在 Anchor 控制空间统计，再用同一初态固定 RBF 映射到 GS 后独立统计。"
        "Anchor d 不是 FPS Anchor 实际位置变化，也不是刚体/材料轨迹。\n\n",
        "相反比例仅在双方向量范数均>1e−6 mm 时定义，未定义另计；不把零向量当方向错误。"
        "RMS 抵消指数=1−RMS(final)/(RMS(base)+RMS(global+local))，不是机械能。"
        "强抵消描述门槛固定为 |final|<0.1(|base|+|correction|)，只用于描述，不作为接触/模型门控。\n\n",
        table(["H","base RMS mm","G4+G5 RMS mm","final RMS mm","base/最终 RMS","base vs rest cos",
               "相反方向比例","抵消指数","base范数大于final比例"],
            [[r["horizon"],f(r["base_rms_mm"]),f(r["correction_rms_mm"]),f(r["final_rms_mm"]),
              f(r["base_to_final_rms"]),f(r["base_vs_rest_cos_mean"]),f(r["base_vs_rest_opposite_fraction"]),
              f(r["cancellation_index"]),f(r["base_larger_than_final_fraction"])]
             for r in data["cancellation_val_summary"] if r["space"]=="gs_actual" and r["region"]=="all"]),
        "\n这里先每窗统计，再 Episode 等权；方向比例不是把全 Val 的 Gaussian 混成独立样本。"
        "完整分区/Anchor/GS、XYZ 均值、RMS、分支对最终余弦和未定义数均保留原始表。\n\n",
        "## 6. 指定案例与局部结构\n\n"])
    for ep,start,split in ((3,135,"val"),(3,245,"val"),(6,110,"val"),(7,105,"train"),
                            (3,25,"val"),(6,20,"val"),(6,200,"val")):
        case=[r for r in data["windows"] if r["episode"]==ep and r["start"]==start and r["split"]==split]
        report.extend([f"### {split} ep{ep}/start{start}\n\n",
            table(["条件","H1–15 CD mm","RGB","Alpha","Joint"],
                  [[r["mode"],f(r["cd_mm"]),f(r["rgb"]),f(r["alpha"]),f(r["joint"])] for r in case]),"\n"])
        for mode in ("Normal","NoBase-All","NoBase-H2+"):
            h=next(r for r in data["horizons"] if r["split"]==split and r["episode"]==ep and
                   r["start"]==start and r["mode"]==mode and r["horizon"]==15)
            report.append(f"- {mode} H15 平均 XYZ=({f(h['mean_x_mm'],3)}, {f(h['mean_y_mm'],3)}, "
                          f"{f(h['mean_z_mm'],3)}) mm；RMS={f(h['rms_mm'],3)} mm。\n")
        report.append("\n"+table(["条件","H15区域","运动RMS mm","16NN |长度比−1| P95"],
            [[r["mode"],r["region"],f(r["displacement_rms_mm"],3),f(r["neighbor_distortion_p95"],4)]
             for r in data["structure"] if r["split"]==split and r["episode"]==ep and r["start"]==start
             and r["horizon"]==15 and r["region"] in ("near","middle","far") and r["mode"] in
             ("Normal","NoBase-All","NoBase-H2+")])+"\n")
    report.extend(["## 7. 必须回答的归因问题\n\n"])
    cancel15=[r for r in data["cancellation"] if r["split"]=="val" and r["space"]=="gs_actual"
              and r["region"]=="all" and r["horizon"]==15]
    defined=[r for r in cancel15 if r["base_vs_rest_cos_mean"] is not None]
    report.append(f"**Q1：G0 是否经常与 G4+G5 反向？** H15 在 {len(defined)} 个方向已定义窗口中，"
                  f"{sum(r['base_vs_rest_cos_mean']<0 for r in defined)} 窗的逐点余弦均值为负。"
                  "还须结合第5节各H/各区的相反比例，不能用全局平均掩盖区域相反运动。\n\n")
    ratios=[r["base_to_final_rms"] for r in cancel15 if r["base_to_final_rms"] is not None]
    report.append("**Q2：base 是否产生大于最终的多余运动？** H15 逐窗 base/final RMS 比值"
                  f"中位数={f(float(np.median(ratios))) if ratios else '未定义'}，"
                  f"P95={f(float(np.quantile(ratios,.95))) if ratios else '未定义'}。"
                  "大幅抵消是描述性冗余证据，不能单凭反向就断定不合理；它也可能是联合分工。\n\n")
    report.append("**Q3：关闭 base 的当前步几何误差？** 第4节给出相同历史 H1/5/10/15 的 ΔCD、"
                  "ΔRGB/Alpha/Joint，负值改善、正值恶化；它与完整 AR 的结果分开判断。\n\n")
    for mode in ("NoBase-All","NoBase-H2+"):
        v=s[mode]
        report.append(f"**Q4 / {mode} 全Val100：** ΔCD={f(v['geometry_mm']-normal['geometry_mm'])} mm，"
                      f"ΔJoint={f(v['objective']-normal['objective'])}。"
                      "仅表示当前冻结模型的该干预结果。\n\n")
    report.append("**Q5：H1 依赖而 H5–H15 成为负担？** H1 的有符号分支数值严格为零，所以函数上依赖 base；"
                  "是否有益由 H1 Normal/Copy 指标判断。后续是否成为负担应看单步及完整AR两套指标；"
                  "不能只用 H1 结构事实或 H15 抵消比例得出结论。\n\n")
    report.append("**Q6：低变化漂移是否与 base 有关？** 第6节保留 ep3/25、ep6/20、ep6/200 的"
                  "三条件 H15 有符号位移和全窗指标。关闭 base 后变化可作为其在冻结模型中的作用证据，"
                  "但非 base 单独造成全部漂移的证明，后续还混有 AR 分布变化。\n\n")
    nb=s["NoBase-H2+"]
    if nb["objective"]<normal["objective"] and nb["geometry_mm"]<normal["geometry_mm"]:
        message="本次 NoBase-H2+ 的 CD 与 Joint 均改善，支持将统一有符号 Decoder 作为下一阶段可检验假设。"
    else:
        message="本次 NoBase-H2+ 未同时改善 CD 与 Joint，不支持直接以冻结删除代替原模型。"
    report.append("**Q7：是否支持移除 G0、改为统一有符号解码器？** "+message+
                  "无论正负结果，都不能判断重新训练后的无base架构价值。未进行或授权任何新架构实现、训练或调参。\n\n")
    # Preserve the strongest beneficial AND detrimental windows, without hiding the rest.
    base={(r["episode"],r["start"]):r for r in data["windows"] if r["split"]=="val" and r["mode"]=="Normal"}
    paired=sorted((dict(episode=r["episode"],start=r["start"],
                       delta_joint=r["joint"]-base[(r["episode"],r["start"])]["joint"],
                       delta_cd_mm=r["cd_mm"]-base[(r["episode"],r["start"])]["cd_mm"])
                   for r in data["windows"] if r["split"]=="val" and r["mode"]=="NoBase-H2+"),
                  key=lambda r:r["delta_joint"])
    report.extend(["### NoBase-H2+ 最有利与最不利窗口（数值描述，不重新选训练样本）\n\n",
        table(["方向","Episode","Start","ΔCD mm","ΔJoint"],
            [[label,r["episode"],r["start"],f(r["delta_cd_mm"]),f(r["delta_joint"])]
             for label,rows in (("Joint差最小",paired[:3]),("Joint差最大",paired[-3:])) for r in rows]),
        "\n全部100窗均在原始表保留，不能把这些相关窗当作独立重复实验。\n\n",
        "## 8. 图像、原始记录与限制\n\n",
        "- figures/val100_horizon_metrics.png：完整验证逐H曲线。\n",
        "- figures/same_history_changes.png：相同历史的单步变化。\n",
        "- figures/val100_cancellation.png：实际GS场抵消。\n",
        "- figures/*_motion_decomposition.png：七案例H1–15分支XYZ/RMS/方向。\n",
        "- figures/*_geometry.png：四重点窗H1/5/10/15固定世界范围几何对照，所有模式共用尺度。\n",
        "- images/{split}_ep{episode}_s{start}/：四重点窗，GT/Copy/六条件、三相机RGB/Alpha关键帧。\n",
        "- arrays/：七指定窗的初态、独立GT、所有条件世界XYZ及Normal分支Anchor/GS位移（米）。\n",
        "- results/：全量CSV/JSON；windows/：每窗独立证据分片，防止异常退出丢掉已完成数据。\n",
        "- provenance*.json、manifest.json、completed.json：实际命令、来源、版本、哈希与完整性。\n\n",
        "GT 是未来独立重建 Gaussian 渲染，不是真实照片；不能由行号建立GT材料对应。"
        "Near/Middle/Far 是初态到有效操作Controller距离排名25/50/25，不是接触标签。"
        "16NN长度变化只是结构代理。冻结消融会改变后续状态分布；指标改善不证明无base架构重训一定更好，"
        "指标退化也不证明它必然失败。没有用Oracle监督，没有修改Loss、Renderer、模型或数据。\n\n",
        f"模型状态哈希前/后：{provenance['model_hash_before']} / {provenance['model_hash_after']}。\n\n",
        "本报告的数据推断需结合代表性PNG人工复核；自动绘图完成不等于已完成逐图视觉验收。"
        "本轮在真实推理、报告、原始证据交付后停止。\n",
    ])
    with (out/"final_report.md").open("x",encoding="utf8") as stream:
        stream.write("".join(report))
    paths=sorted(p for p in out.rglob("*") if p.is_file() and p.suffix in (".json",".csv",".png",".md",".npz"))
    write_json(out/"artifact_manifest.json",{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    write_json(out/"report_complete.json",dict(state="REPORT_BUILT",figures=figures,parameters_updated=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    build(Path(p.parse_args().output).resolve())


if __name__=="__main__":
    main()
