"""CPU-only graph preview, fixed-scale figures and truthful Chinese report.

--graphs-only can run before formal training.  It reads only the two frozen
calibration NPZs, never a model/checkpoint, and emits no fictitious final result.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DISPLAY={'copy':'Copy Initial','historical_g5':'Historical G5','c_best':'C G5-ARAP Best',
         'c_last':'C Last','d_best':'D G5-ARAP-Iso Best','d_last':'D Last'}
COLORS={'copy':'#888888','historical_g5':'#3857a6','c_best':'#df831c','d_best':'#228b60'}
SELECTED=(1,5,10,15)
MAIN=(('val',3,135),('val',3,245),('val',6,110),('train',7,105))


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _savefig(fig,path,plt,**kwargs):
    if Path(path).exists():
        plt.close(fig)
        raise FileExistsError('Existing figure preserved: '+str(path))
    fig.savefig(path,**kwargs);plt.close(fig)


def _sample_indices(count,maximum):
    return np.linspace(0,count-1,min(count,maximum),dtype=np.int64) if count else np.empty(0,dtype=np.int64)


def graph_figures(root,graph_root):
    """All GS are drawn; deterministic sparse edges plus longest-P99 markers."""
    root=Path(root);graph_root=Path(graph_root)
    if not root.is_absolute() or not graph_root.is_absolute():raise ValueError('Absolute graph input/output directories required')
    figures=root/'figures';figures.mkdir(parents=True,exist_ok=True)
    manifest_path=figures/'initial_graph_visualization.json'
    sources=[graph_root/f'graph_ep{ep}_start{start}.npz' for ep,start in ((2,141),(7,105))]
    source_hashes={str(p):sha256(p) for p in sources}
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text(encoding='utf-8'))
        if old['input_sha256']!=source_hashes or any(sha256(root/row['figure'])!=row['figure_sha256'] for row in old['figures']):
            raise FileExistsError('Existing graph preview belongs to different inputs or changed outputs')
        return old
    plt=_plt();records=[]
    for path,(ep,start) in zip(sources,((2,141),(7,105))):
        with np.load(path,allow_pickle=False) as arrays:
            xyz=arrays['initial_world'].astype(np.float64);keep=arrays['mask'].astype(bool);edges=arrays['edges'].astype(np.int64)
        if not np.isfinite(xyz[keep]).all() or edges.ndim!=2 or edges.shape[-1]!=2:
            raise ValueError('Invalid initial graph arrays')
        if len(edges) and (edges.min()<0 or edges.max()>=len(xyz) or not keep[edges].all()):
            raise ValueError('Invalid/padded graph endpoint')
        xyz=xyz*1000
        lengths=np.linalg.norm(xyz[edges[:,0]]-xyz[edges[:,1]],axis=-1) if len(edges) else np.empty(0)
        threshold=float(np.quantile(lengths,.99)) if len(edges) else None
        long_ids=np.flatnonzero(lengths>=threshold) if threshold is not None else np.empty(0,dtype=np.int64)
        gray_ids=_sample_indices(len(edges),2500);red_ids=long_ids[_sample_indices(len(long_ids),500)]
        fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
        for axis,(a,b),title in zip(axes,((0,1),(0,2),(1,2)),('World XY','World XZ','World YZ')):
            for ids,color,width,alpha in ((gray_ids,'#858b92',.25,.23),(red_ids,'#c92336',.75,.7)):
                if len(ids):
                    segments=xyz[edges[ids]][:,:,[a,b]]
                    from matplotlib.collections import LineCollection
                    axis.add_collection(LineCollection(segments,colors=color,linewidths=width,alpha=alpha))
            axis.scatter(xyz[keep,a],xyz[keep,b],s=1.8,c='#202a37',alpha=.65,rasterized=True)
            axis.set_aspect('equal',adjustable='box');axis.autoscale();axis.grid(alpha=.15)
            axis.set(title=title,xlabel=f'World {"XYZ"[a]} (mm)',ylabel=f'World {"XYZ"[b]} (mm)')
        fig.suptitle(f'Initial spatial graph ep{ep}/start{start} | {int(keep.sum())} GS, {len(edges)} edges\n'
                     'Gray: deterministic subset; red: longest 1% (display capped). Not material topology.')
        output=figures/f'initial_graph_ep{ep}_start{start}.png'
        if output.exists():raise FileExistsError(output)
        _savefig(fig,output,plt,dpi=160)
        records.append(dict(episode=ep,start=start,valid_gs=int(keep.sum()),full_edges=len(edges),
            gray_display_edges=len(gray_ids),longest_p99_threshold_mm=threshold,
            longest_p99_full_count=len(long_ids),red_display_edges=len(red_ids),
            sampling='canonical edge order, evenly spaced deterministic indices; max gray2500/red500',
            figure=str(output.relative_to(root)),figure_sha256=sha256(output)))
    result=dict(input_sha256=source_hashes,figures=records,graph_source='initial GS only; unchanged NPZ full graph retained',
        reviewed_by_human=False,note='Preview generation does not automatically approve cross-segment topology.')
    with manifest_path.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    return result


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        result=[]
        for row in csv.DictReader(stream):
            converted={}
            for key,value in row.items():
                if value=='':converted[key]=None
                elif value in ('True','False'):converted[key]=value=='True'
                else:
                    try:converted[key]=float(value)
                    except (ValueError,TypeError):converted[key]=value
            result.append(converted)
        return result


def epoch_plot(root,rows):
    plt=_plt();fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,(key,title,baseline) in zip(axes.flat,(('val_cd_mm','Val CD (mm)',2.402626),
            ('val_rgb','Val RGB MAE',.077624),('val_alpha','Val Alpha MAE',.077699),('val_joint','Val original Joint',.100694))):
        for label,color in (('c','#df831c'),('d','#228b60')):
            selected=sorted((r for r in rows if r['model']==label),key=lambda r:r['epoch'])
            ax.plot([r['epoch'] for r in selected],[r[key] for r in selected],marker='o',label=label.upper(),color=color)
        ax.axhline(baseline,color='#3857a6',linestyle='--',label='Historical G5 E8')
        ax.set(title=title,xlabel='Completed epoch');ax.grid(alpha=.2);ax.legend()
    _savefig(fig,root/'figures'/'epoch_validation_metrics.png',plt,dpi=160)


def horizon_plot(root,rows):
    plt=_plt();fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for label,color in COLORS.items():
        values=sorted((r for r in rows if r['model']==label and r['episode']=='episode_equal' and r['region']=='all'),key=lambda r:r['horizon'])
        for ax,key,title in zip(axes.flat,('cd_mm','displacement_rms_mm','global_translation_removed_rms_mm','neighbor_distortion_p95'),
                ('Val CD (mm)','Predicted GS motion RMS (mm)','Translation-removed RMS (mm)','Fixed 16-other-NN |ratio-1| P95')):
            ax.plot([r['horizon'] for r in values],[r[key] for r in values],label=DISPLAY[label],color=color)
            ax.set(title=title,xlabel='Horizon');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    _savefig(fig,root/'figures'/'horizon_geometry_motion.png',plt,dpi=160)


def geometry_figures(root):
    plt=_plt();outputs=[]
    for split,ep,start in MAIN:
        path=root/'arrays'/f'{split}_ep{ep}_start{start}.npz'
        if not path.is_file():continue
        with np.load(path,allow_pickle=False) as arrays:
            predictions={label:arrays[label+'_world'].astype(np.float64)*1000 for label in COLORS}
            targets=[arrays[f'target_world_h{h:02d}'].astype(np.float64)*1000 for h in range(1,16)]
            initial=arrays['initial_world'].astype(np.float64)*1000
        # One common physical cube for every model AND all H1--H15 in a window.
        merged=np.concatenate([initial,*targets,*[value.reshape(-1,3) for value in predictions.values()]],0)
        if not np.isfinite(merged).all():raise ValueError('Nonfinite geometry figure input')
        lo,hi=merged.min(0),merged.max(0);center=(lo+hi)/2;radius=max(float((hi-lo).max())*.55,1.)
        for h in SELECTED:
            fig=plt.figure(figsize=(14,9),constrained_layout=True)
            models=[('gt',targets[h-1])]+[(name,value[h-1]) for name,value in predictions.items()]
            for index,(label,xyz) in enumerate(models,1):
                axis=fig.add_subplot(2,3,index,projection='3d')
                axis.scatter(*xyz.T,s=.8,c='#384653',alpha=.65,rasterized=True)
                axis.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),
                    zlim=(center[2]-radius,center[2]+radius),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',
                    title='Independent GT GS' if label=='gt' else DISPLAY[label])
                axis.set_box_aspect((1,1,1));axis.view_init(elev=22,azim=-60)
            fig.suptitle(f'{split} ep{ep}/start{start} H{h} | common world bounds/view across models and horizons')
            output=root/'figures'/f'ep{ep}_start{start}_h{h:02d}_geometry.png'
            _savefig(fig,output,plt,dpi=150);outputs.append(str(output.relative_to(root)))
    return outputs


def rgb_contact_figures(root):
    """Source PNG pixels only, one common crop per camera across all comparisons."""
    from PIL import Image,ImageDraw,ImageFont
    labels=('gt',*COLORS)
    try:font=ImageFont.truetype('arial.ttf',18)
    except OSError:font=ImageFont.load_default()
    outputs=[]
    for split,ep,start in MAIN:
        folder=root/'images'/f'{split}_ep{ep}_start{start}'
        if not folder.is_dir():continue
        bounds={}
        for camera in range(3):
            points=[]
            for h in SELECTED:
                for label in labels:
                    file=folder/f'h{h:02d}_cam{camera}_alpha_{label}.png'
                    if file.is_file():
                        with Image.open(file) as source:
                            mask=np.asarray(source.convert('L'))>8
                        yy,xx=np.where(mask)
                        if len(xx):points.append((xx.min(),yy.min(),xx.max()+1,yy.max()+1))
            bounds[camera]=(max(0,min(p[0] for p in points)-24),max(0,min(p[1] for p in points)-24),
                min(1280,max(p[2] for p in points)+24),min(720,max(p[3] for p in points)+24)) if points else (0,0,1280,720)
        for channel in ('rgb','alpha'):
            canvas=Image.new('RGB',(1500,50+len(labels)*290),'#f0f3f7');draw=ImageDraw.Draw(canvas)
            draw.text((12,10),f'ep{ep}/start{start} H15 {channel.upper()} | fixed source-pixel crop per camera',font=font,fill='black')
            for row,label in enumerate(labels):
                for camera in range(3):
                    source=folder/f'h15_cam{camera}_{channel}_{label}.png'
                    with Image.open(source) as image:tile=image.convert('RGB').crop(bounds[camera])
                    tile.thumbnail((490,250),Image.Resampling.LANCZOS)
                    x=camera*500+(500-tile.width)//2;y=50+row*290
                    draw.text((camera*500+8,y),f'{"GT reconstruction" if label=="gt" else DISPLAY[label]} | cam{camera}',font=font,fill='black')
                    canvas.paste(tile,(x,y+30+(250-tile.height)//2))
            output=root/'figures'/f'ep{ep}_start{start}_h15_{channel}_comparison.png'
            if output.exists():raise FileExistsError(output)
            canvas.save(output);outputs.append(str(output.relative_to(root)))
    return outputs


def _number(value,digits=6):
    return '缺失' if value is None else f'{value:.{digits}f}'


def _table(rows,fields):
    lines=['| '+' | '.join(title for title,_ in fields)+' |','|'+'|'.join('---' for _ in fields)+'|']
    for row in rows:
        lines.append('| '+' | '.join(str(row.get(key,'缺失')) if not isinstance(row.get(key),(float,int)) else _number(row[key]) for _,key in fields)+' |')
    return '\n'.join(lines)


def _episode_equal(rows,key):
    by_episode={}
    for row in rows:
        if row.get(key) is not None:by_episode.setdefault(row['episode'],[]).append(row[key])
    return float(np.mean([np.mean(values) for values in by_episode.values()])) if by_episode else None


def verify_sources(root):
    path=root/'artifact_hashes.json'
    if not path.is_file():raise FileNotFoundError('GPU evaluation artifact hash manifest required')
    manifest=json.loads(path.read_text(encoding='utf-8'))
    for name,expected in manifest.items():
        source=(root/name).resolve()
        if not source.is_relative_to(root.resolve()) or not source.is_file() or sha256(source)!=expected:
            raise ValueError('Missing or changed synchronized evaluation artifact: '+name)
    completion=json.loads((root/'completion.json').read_text(encoding='utf-8'))
    count=len(list((root/'images').rglob('*.png')))
    if not completion.get('passed') or completion.get('images')!=count:
        raise ValueError('Evaluation incomplete or synchronized image count differs')
    return count


def regularizer_plot(root,rows):
    plt=_plt();fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for label,color in COLORS.items():
        for ax,key,title in zip(axes,('arap','iso','kabsch_fallback_fraction'),
                ('Consecutive local ARAP proxy','Initial-edge ISO proxy','Kabsch identity fallback fraction')):
            x=[];y=[]
            for h in range(1,16):
                selected=[r for r in rows if r['model']==label and r['split']=='val' and r['horizon']==h]
                x.append(h);y.append(_episode_equal(selected,key))
            ax.plot(x,y,label=DISPLAY[label],color=color);ax.set(title=title,xlabel='Horizon');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    _savefig(fig,root/'figures'/'regularizer_horizons.png',plt,dpi=160)


def final_report(root):
    """Aggregate actual artifacts; pixel-level adoption verdict remains explicit."""
    root=Path(root)
    if (root/'final_report.md').exists() or (root/'report_provenance.json').exists():
        raise FileExistsError('Existing report preserved; use a new comparison directory')
    image_count=verify_sources(root)
    provenance=json.loads((root/'provenance.json').read_text());checkpoints=json.loads((root/'checkpoint_provenance.json').read_text())
    summary=read_csv(root/'results'/'val_summary.csv');epochs=read_csv(root/'results'/'epoch_metrics.csv')
    horizons=read_csv(root/'results'/'horizon_summary.csv');paired=read_csv(root/'results'/'paired_window_effects.csv')
    structure=read_csv(root/'results'/'structure_summary.csv');independent=read_csv(root/'results'/'independent_16nn.csv')
    raw_horizons=read_csv(root/'results'/'horizon_metrics.csv')
    regularizers=read_csv(root/'results'/'regularizer_diagnostics.csv');windows=read_csv(root/'results'/'window_metrics.csv')
    epoch_plot(root,epochs);horizon_plot(root,horizons);regularizer_plot(root,regularizers)
    geometry=geometry_figures(root);contacts=rgb_contact_figures(root)
    overall=[row for row in summary if row['episode']=='episode_equal'];lookup={r['model']:r for r in overall}
    structural=[]
    for label in DISPLAY:
        selected=[r for r in independent if r['model']==label and r['split']=='val' and r['horizon']==15 and r['region']=='all']
        motion=next(r for r in horizons if r['model']==label and r['episode']=='episode_equal' and r['horizon']==15 and r['region']=='all')
        solver=[r for r in regularizers if r['model']==label and r['split']=='val' and r['horizon']==15]
        timing=[r for r in windows if r['model']==label and r['split']=='val']
        rank0=_episode_equal(solver,'kabsch_rank0_fraction');rank1=_episode_equal(solver,'kabsch_rank1_fraction')
        structural.append(dict(model=label,stretch=_episode_equal(selected,'stretch_fraction'),compression=_episode_equal(selected,'compression_fraction'),
            abs_ratio_p95=_episode_equal(selected,'abs_ratio_minus_one_p95'),motion_rms_mm=motion['displacement_rms_mm'],
            residual_rms_mm=motion['global_translation_removed_rms_mm'],arap=_episode_equal(solver,'arap'),iso=_episode_equal(solver,'iso'),
            fallback_fraction=_episode_equal(solver,'kabsch_fallback_fraction'),
            rank01_fraction=rank0+rank1 if rank0 is not None and rank1 is not None else None,
            regularizer_ms=_episode_equal(timing,'regularizer_seconds')*1000))
    cases=[]
    for split,episode,start in MAIN:
        for label in ('historical_g5','c_best','d_best'):
            selected={r['region']:r for r in raw_horizons if r['model']==label and r['split']==split and r['episode']==episode and r['start']==start and r['horizon']==15}
            if selected:
                cases.append(dict(window=f'ep{episode}/start{start}',model=label,mean_y_mm=selected['all']['mean_y_mm'],
                    mean_z_mm=selected['all']['mean_z_mm'],near_rms_mm=selected.get('near',{}).get('displacement_rms_mm'),
                    far_rms_mm=selected.get('far',{}).get('displacement_rms_mm')))
    changes=[];counterexamples=[]
    for label,reference in (('c_best','historical_g5'),('d_best','c_best'),('c_last','c_best'),('d_last','d_best')):
        selected=[r for r in paired if r['model']==label and r['reference']==reference]
        changes.append(f'- {DISPLAY[label]} 对 {DISPLAY[reference]}：Joint 改善 {sum(bool(r["improved_joint"]) for r in selected)} 窗、恶化 {sum(bool(r["worsened_joint"]) for r in selected)} 窗；完整验证等权 Joint 差 {_number(lookup[label]["joint"]-lookup[reference]["joint"])}。')
        ordered=sorted(selected,key=lambda r:r['delta_joint'])
        for suffix,records in (('较好',ordered[:2]),('较差',ordered[-2:])):
            for r in records:counterexamples.append(dict(comparison=f'{label} vs {reference}',case=suffix,episode=r['episode'],start=r['start'],joint_delta=r['delta_joint'],cd_delta_mm=r['delta_cd_mm']))
    epoch_text=[];evidence=[]
    for short in ('c','d'):
        last=checkpoints[short+'_last'];best=checkpoints[short+'_best']
        epoch_text.append(f'- {short.upper()}：完成 {last["epoch"]} Epoch，累计成功更新 {last["successful_updates"]}，Best 为 Epoch {best["epoch"]}。')
    for label,row in checkpoints.items():
        evidence.append(f'- {label}：`{row["path"]}`；SHA256 `{row["sha256"]}`；训练源码 `{row["training_source_sha"]}`。')
    text=f'''# G5-ARAP / G5-ARAP-Iso 配对实验报告

## 1. 实现与冻结协议

C 与 D 均使用原 G5-Direct 网络，从相同 Seed42 共同初态训练，没有新状态、GEO、A2A、接触门控或额外位移分支。只预测 XYZ，11D 属性继承初态；原 Base/Global/Local、三相机、Renderer 和原 CD+RGB+0.1Alpha 保留。

C 增加相邻预测时刻的局部 SO(3) 对齐残差：Kabsch 旋转在 no_grad 下求解，但残差中的前后预测 XYZ 均可求导。按初态邻边平方和归一化，按有效中心、时间、窗口等权。D 再增加相对初态边长的 log-ratio SmoothL1，beta=0.1。固定初态互邻图只是空间近邻先验，不是真实材料拓扑；局部 SO(3) 不是形变梯度，不据此推断物理应变。

实际冻结权重：λARAP={provenance['calibration']['lambda_arap']:.12g}，λIso={provenance['calibration']['lambda_iso']:.12g}（C 的 Iso 训练权重为0）。仅历史 G5 的训练窗 ep2/start141、ep7/start105 一次向下校准，分别保护输出 XYZ 梯度比不超过5%/1%；这不是所有未来训练 Batch 的梯度保证或最优权重搜索。Best 按完整 Val100 的原始 Joint 选择，不包含新增正则。

参数量与共同初始化记录：`{provenance['runs']['c']['protocol']['identity']['initialization'].get('parameter_counts',{})}`。C/D 没有新增网络参数；正式训练不加载历史 G5 或 Smoke 更新权重。

## 2. 执行与来源

评估源码：`{provenance['evaluation_source_sha']}`；实际训练源码：`{provenance['training_source_sha']}`，分别记录，不以报告提交冒充训练提交。

评估 Job/Step `{provenance['resource']['job_id']}.{provenance['resource']['step_id']}`，GPU `{provenance['resource']['gpu_uuid']}`。评估无优化器、无 backward、零参数更新，模型及 Checkpoint 哈希前后保持。完整数据为 ep3=55、ep6=45，先窗口平均，再 Episode 等权；训练 ep7/start105 仅作机制案例，不进入验证汇总。

{chr(10).join(epoch_text)}

CPU、独立 Review、校准、四次 Smoke、调度/恢复及失败记录保留在两个 Run 和共同准备目录 `{provenance['common']['path']}`。静态汇总不凭缺失日志宣称零失败或没有恢复，实际终态须由 Launcher/实时进程核验。

## 3. 完整 Val100：原始指标与正反结果

{_table(overall,[('模型','model'),('CD mm','cd_mm'),('RGB','rgb'),('Alpha','alpha'),('原 Joint','joint')])}

{chr(10).join(changes)}

ep3/ep6 分项见 `results/val_summary.csv`，逐 Horizon 见 `results/horizon_metrics.csv`，全部配对变化见 `results/paired_window_effects.csv`。同 Epoch 对照及成功更新数见 `results/epoch_metrics.csv`，训练原 Loss、未加权/加权 ARAP、未加权/加权 Iso、总 Loss 分列；训练总 Loss 不直接与历史 Joint 比较。保留 Best/Last 退化，不延长预算或按结果调参。

下表按全部配对差选取每组改善/恶化两端，供同时检查正例与反例；数值选例不代替视觉结论：

{_table(counterexamples,[('对照','comparison'),('端点','case'),('Episode','episode'),('Start','start'),('Joint差','joint_delta'),('CD差/mm','cd_delta_mm')])}

## 4. 结构与有效运动：防止少动换规整

{_table(structural,[('模型','model'),('过伸比例','stretch'),('过压比例','compression'),('窗口|ratio−1|P95均值','abs_ratio_p95'),('GS RMS/mm','motion_rms_mm'),('去平移RMS/mm','residual_rms_mm'),('ARAP','arap'),('Iso','iso'),('Kabsch回退比例','fallback_fraction'),('rank0/1比例','rank01_fraction'),('正则诊断耗时/ms','regularizer_ms')])}

上表伸压尾部与长度失真来自**独立的原始初态16-other-NN**：initial_knn(...,17) 去Self，保留有效初态长度>1e-8的有向近邻，不做训练图的互邻和长度筛选；并非在被训练的同一图上自证成功。`results/independent_16nn.csv` 逐窗口/Horizon/区域保留 ratio P05/P50/P95/P99、|ratio−1|P95、过伸/过压比例及有效边数。训练互邻图统计另在 `results/structure_summary.csv`，ARAP/Iso 在 `results/regularizer_diagnostics.csv`，三者口径明确分开。

0.8/1.2 尾部只是与历史结果一致的诊断阈值，不是本轮 Iso 的零惩罚容忍区。P95 列是逐窗口 P95 的 Episode 等权均值，非全部边混池 P95。求解低秩与 SVD 失败回退分别记录；低秩不自动等于错误或材料断裂。正则耗时是冻结评估中同步测量的独立计算时间，不是端到端训练加速比；逐Epoch耗时、实际尝试和显存峰值另保留在 `results/epoch_metrics.csv`。

主要案例 H15 实际 GS 运动：

{_table(cases,[('窗口','window'),('模型','model'),('均值Y/mm','mean_y_mm'),('均值Z/mm','mean_z_mm'),('Near RMS/mm','near_rms_mm'),('Far RMS/mm','far_rms_mm')])}

`results/horizon_metrics.csv` 保存全H1–H15 XYZ、Near/Middle/Far RMS、去平移残差及单向 GT 距离。分区为初态 Controller 距离四分位，不是接触标签；初态无有效 Controller 时不删除窗口，分区标记不可用，完整原始指标和全GS诊断仍保留。须结合 ep7 Near 大运动、ep3 负Y、ep6 下移及低变化 ep3/start25、ep6/start20、ep6/start200 判断是否因欠运动换来规整。

七个代表 NPZ 和 `results/controller_motion.csv` 保留每手 Controller/Root 世界坐标及双端有效累计/相邻运动，不把相反手平均。Hk 条件在 t+k−1，目标 GS 在 t+k；时间关联不是物理因果证明。GT 质心是独立重建点集统计，不是材料轨迹。

## 5. 固定可视化与研究验收

主四窗口保存 H1/H5/H10/H15、cam0/1/2、RGB/Alpha 的 GT / Copy / 历史G5 / CBest / DBest，共 {image_count} 张原始 PNG。GT 是独立重建 GS 渲染，不是原相机照片。几何图在同一窗口全部模型和Horizon使用相同世界范围、视角与比例，联系图每行最多三个相机；初态缺口和画面裁切不解释为模型新生成的三维断裂。

以下图片需实际打开，补充散开/剪切、局部伸压、弯曲、远端跟随、欠运动、漂移及更差案例。**本自动稿不声称完成像素级研究审查，也不凭 Joint 自动决定采用。**

{chr(10).join('- `'+name+'`' for name in [*geometry,*contacts])}

应分别判断：C 是否减少局部失真且保留有效动作；D 是否在长期漂移与弯曲自由度间更合理；是否存在数值接近而视觉更自然的结果；哪些窗口更差。没有GEO，新增Scale/Rotation不可能是本轮收益来源。空间图在折叠处可能跨段，两个训练窗图预审结果应同时保留。

## 6. 建议与证据边界

当前自动汇总支持上列真实数值和可复核证据索引。是否保留 ARAP、Iso 或两者暂不采用，须在实际查看图片后补完明确建议。正则降低、低CD、低秩统计或潜在的局部规整，均不能证明物理材料真实性；冻结GT无可靠逐点材料对应。性能不改善也是有效结果，不追加第三组、不调权重、不延长训练。

## 7. Checkpoint 与产物

{chr(10).join(evidence)}

比较目录：`{root.resolve()}`。核心 CSV：`results/epoch_metrics.csv`、`results/val_summary.csv`、`results/horizon_metrics.csv`、`results/horizon_summary.csv`、`results/paired_window_effects.csv`、`results/structure_summary.csv`、`results/independent_16nn.csv`、`results/regularizer_diagnostics.csv`、`results/controller_motion.csv`。数组 `arrays/`；源渲染 `images/`；完整来源 `provenance.json`、`checkpoint_provenance.json`。Best/Last 路径、SHA、成功更新与训练源码分别保留。
'''
    output=root/'final_report.md';output.write_text(text,encoding='utf-8')
    generated=[output,*sorted((root/'figures').glob('*.png'))]
    record=dict(report_module_sha256=sha256(__file__),generated={str(p.relative_to(root)):sha256(p) for p in generated},
        note='CPU plots/text only; no model/GPU execution; actual pixel research review and final adoption decision pending separate inspection.')
    with (root/'report_provenance.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
    return str(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--graph-root');parser.add_argument('--graphs-only',action='store_true')
    args=parser.parse_args();root=Path(args.root)
    if args.graphs_only and not args.graph_root:parser.error('--graphs-only requires --graph-root')
    if args.graph_root:print(json.dumps(graph_figures(root,args.graph_root),indent=2))
    if not args.graphs_only:print(final_report(root))


if __name__=='__main__':main()
