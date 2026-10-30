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


DISPLAY={'copy':'Copy Initial','historical_g5':'Historical G5','a_best':'A G5-Struct Best',
         'a_last':'A Last','b_best':'B State-Struct Best','b_last':'B Last'}
COLORS={'copy':'#888888','historical_g5':'#3857a6','a_best':'#df831c','b_best':'#228b60'}
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
        for label,color in (('a','#df831c'),('b','#228b60')):
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


def final_report(root):
    root=Path(root)
    if (root/'final_report.md').exists() or (root/'report_provenance.json').exists():
        raise FileExistsError('Existing final report preserved; use a new comparison directory for a rerun')
    image_count=verify_sources(root)
    provenance=json.loads((root/'provenance.json').read_text());checkpoints=json.loads((root/'checkpoint_provenance.json').read_text())
    summary=read_csv(root/'results'/'val_summary.csv');epochs=read_csv(root/'results'/'epoch_metrics.csv')
    horizons=read_csv(root/'results'/'horizon_summary.csv');paired=read_csv(root/'results'/'paired_window_effects.csv')
    usage=read_csv(root/'results'/'state_usage.csv')
    structure=read_csv(root/'results'/'structure_summary.csv');raw_horizons=read_csv(root/'results'/'horizon_metrics.csv')
    epoch_plot(root,epochs);horizon_plot(root,horizons)
    geometry=geometry_figures(root);contacts=rgb_contact_figures(root)
    overall=[row for row in summary if row['episode']=='episode_equal']
    lookup={row['model']:row for row in overall}
    structure_rows=[]
    for label in ('copy','historical_g5','a_best','a_last','b_best','b_last'):
        selected=[r for r in structure if r['model']==label and r['split']=='val' and r['horizon']==15]
        motion=next(r for r in horizons if r['model']==label and r['episode']=='episode_equal' and r['horizon']==15 and r['region']=='all')
        structure_rows.append(dict(model=label,stretch_fraction=_episode_equal(selected,'stretch_fraction'),
            compression_fraction=_episode_equal(selected,'compression_fraction'),
            mean_window_abs_ratio_p95=_episode_equal(selected,'abs_ratio_minus_one_p95'),
            gs_motion_rms_mm=motion['displacement_rms_mm'],residual_rms_mm=motion['global_translation_removed_rms_mm']))
    cases=[]
    for split,episode,start in MAIN:
        for label in ('historical_g5','a_best','b_best'):
            selected={r['region']:r for r in raw_horizons if r['model']==label and r['split']==split and r['episode']==episode and r['start']==start and r['horizon']==15}
            if not selected:continue
            cases.append(dict(window=f'ep{episode}/start{start}',model=label,
                mean_y_mm=selected['all']['mean_y_mm'],mean_z_mm=selected['all']['mean_z_mm'],
                near_rms_mm=selected.get('near',{}).get('displacement_rms_mm'),
                far_rms_mm=selected.get('far',{}).get('displacement_rms_mm')))
    differences=[]
    for label,reference in (('a_best','historical_g5'),('b_best','a_best'),('a_last','a_best'),('b_last','b_best')):
        records=[r for r in paired if r['model']==label and r['reference']==reference]
        differences.append(f'- {DISPLAY[label]} 相对 {DISPLAY[reference]}：Joint 改善 {sum(bool(r["improved_joint"]) for r in records)} 窗、恶化 {sum(bool(r["worsened_joint"]) for r in records)} 窗；全验证 Episode 等权 Joint 差 {_number(lookup[label]["joint"]-lookup[reference]["joint"])}。')
    epoch_text=[]
    for short in ('a','b'):
        last=checkpoints[short+'_last'];best=checkpoints[short+'_best']
        epoch_text.append(f'- {short.upper()}：完成 {last["epoch"]} Epoch，累计成功更新 {last["successful_updates"]}；Best 为 Epoch {best["epoch"]}。')
    evidence=[]
    for label,row in checkpoints.items():
        evidence.append(f'- {label}：`{row["path"]}`；SHA256 `{row["sha256"]}`；训练源码 `{row["training_source_sha"]}`。')
    state_cases={}
    for row in usage:
        if row['horizon']==15 and row['region']=='all':state_cases[f'ep{int(row["episode"])}/start{int(row["start"])}']=row
    state_text='\n'.join(f'- {name}：H15 关闭读出后 GS 输出变化 RMS {_number(row["gs_output_change_rms_mm"],3)} mm；窗口 Joint 差 {_number(row["window_delta_joint_off_minus_normal"])}。' for name,row in state_cases.items())
    text=f'''# G5-Struct / G6-State-Struct 自主实验报告

## 1. 实现与预设协议

A 保留原 G5-Direct 网络，只在最终 GS XYZ 上加入初态互邻图的弱长度约束。B 在相同约束下增加逐 Anchor 301D→128D→GRUCell 状态及零初始化特征读出，没有第四个位移头。两组只预测 XYZ，11D 属性继承初态，没有 GEO、A2A 或接触门控。

参数量：`{provenance['runs']['a']['protocol']['identity']['initialization'].get('parameter_counts',{})}`。共享初始化逐项核验，新增 B 模块与数据生成器随机数隔离；正式训练不加载历史 G5/Smoke 权重。

冻结结构权重为 **{provenance['calibration']['lambda_struct']:.12g}**；校准仅使用 ep2/start141、ep7/start105，向下保护输出空间梯度尺度，不是权重搜索。5% 只约束这两个冻结参考输出，不是所有训练批次或参数梯度的全程保证。训练参数和窗口顺序见保存的 protocol；Best 按原始 Val Joint 选择，不包含结构项。训练总 Loss 不能直接与历史 Joint 比较。

## 2. 实际执行与工程证据

评估源码：`{provenance['evaluation_source_sha']}`。实际训练源码：`{provenance['training_source_sha']}`；两者分别记录，报告版本不冒充训练版本。
GPU/Step：`{provenance['resource']['job_id']}.{provenance['resource']['step_id']}`，GPU `{provenance['resource']['gpu_uuid']}`。本评估无优化器、无 backward、零参数更新；Checkpoint 与模型哈希前后保持。

{chr(10).join(epoch_text)}

CPU、独立 Review、Smoke、调度方式和恢复原因请以比较目录及两个 Run 的原始日志/持久化 Launcher 记录为准。本自动稿不凭缺失日志填造“零失败”或“没有恢复”。完整验证为 ep3=55、ep6=45，先各 Episode 平均，再等权。

## 3. 完整 Val100 原始指标

{_table(overall,[('模型','model'),('CD mm','cd_mm'),('RGB','rgb'),('Alpha','alpha'),('原 Joint','joint')])}

{chr(10).join(differences)}

ep3/ep6 分项在 `results/val_summary.csv`；逐窗口配对差值在 `results/paired_window_effects.csv`。Best 与 Last 均保留，不只展示最好结果。同 Epoch 对照在 `results/epoch_metrics.csv`，其中原 Joint、未加权结构项、加权结构项、总 Loss 分列。

## 4. 有效运动与结构代理

`results/horizon_metrics.csv` 保留每窗口、H1–H15、Near/Middle/Far 的实际 GS 位移 RMS、有符号 XYZ、去整体平移残差和单向几何误差。分区来自初态 Controller 距离四分位，不是接触标签。

若初态没有有效 Controller，完整 Val CD/RGB/Alpha/Joint 仍保留；Near/Middle/Far 行标记 region_available=False、计数0及未定义值，不伪造分区。汇总同时提供 available_windows。七个代表窗口的 NPZ 另存逐手 Controller 世界 XYZ/有效掩码、Root XYZ/有效性及 operation_hands；`results/controller_motion.csv` 分别记录左右手双端有效的累计/相邻运动，不把相反双手平均。Hk 使用 t+k−1 条件预测 t+k GS；这些时间关联不证明物理因果性。

`results/structure_summary.csv` 保存过伸/过压比例、长度比及 |ratio−1| 分位数。结构训练图与独立 16-other-NN 代理分别记录：前者源于原始含 Self 的 16NN 互邻筛选，后者沿用历史 G4 诊断，不假称材料拓扑或真实应变。结构指标下降可能由运动减少取得，应结合 ep7 Near RMS、ep3 负 Y、ep6 下移和低变化窗口漂移审阅。

H15 全验证结构与运动（每窗口统计后 Episode 等权；P95 列是各窗口 P95 的均值，并非全部边混池的 P95）：

{_table(structure_rows,[('模型','model'),('过伸比例','stretch_fraction'),('过压比例','compression_fraction'),('窗口|ratio−1|P95均值','mean_window_abs_ratio_p95'),('GS运动RMS/mm','gs_motion_rms_mm'),('去平移RMS/mm','residual_rms_mm')])}

主要案例 H15 的真实预测 GS 运动：

{_table(cases,[('窗口','window'),('模型','model'),('均值Y/mm','mean_y_mm'),('均值Z/mm','mean_z_mm'),('Near RMS/mm','near_rms_mm'),('Far RMS/mm','far_rms_mm')])}

这些位移基于各预测共同的初态 GS 身份。另存的 GT 质心变化只是独立重建点集统计，不等同真实材料运动。

## 5. B 状态的最小必要性检查

{state_text}

这是同一 B Best 权重的完整 AR 读出关闭，包含后续历史变化；不是重新训练的 A，也不是当前步独立因果贡献。B 还增加了参数量与特征变换能力，没有容量匹配的第三组控制，因此性能差异不能全部归因于时间记忆。潜变量有非零输出或关闭后改变预测，不等于学得唯一物理状态。

## 6. 图像与正反案例核验

已输出主四窗口 H1/H5/H10/H15、三相机 RGB/Alpha 的 GT / Copy / 历史 G5 / A Best / B Best；本目录实际保留 {image_count} 张源 PNG，数量与 GPU 完成记录一致。未来 GT 是独立重建 GS 渲染，不是原相机照片，不存在可靠材料点对应。

本脚本只生成证据和数值汇总，**没有自动进行像素级研究判读**。以下图片需要实际打开后补充成功、欠运动、异常拉伸/压缩、漂移及更差案例；不能仅按 Joint 差确定采用建议。

{chr(10).join('- `'+name+'`' for name in [*geometry,*contacts])}

初态已存在的投影缺口不能当作新模型生成断裂；空间图可能有折叠绳段跨段近邻，两个训练窗的图检查记录必须保留。二维投影和 16NN 距离变化不证明真实材料应变。

## 7. 采用建议与边界

当前自动稿仅支持上表的数值结论及状态读出依赖量。是否保留结构项、递推状态或两者暂不采用，须在实际像素审读后补全；不以“少动换规整”作为成功，也不因结果差自动追加训练、调权重或实施第三组模型。

## 8. Checkpoint 与产物索引

{chr(10).join(evidence)}

比较目录：`{root.resolve()}`。
核心数据：`results/epoch_metrics.csv`、`results/val_summary.csv`、`results/horizon_metrics.csv`、`results/horizon_summary.csv`、`results/paired_window_effects.csv`、`results/structure_summary.csv`、`results/state_usage.csv`、`results/controller_motion.csv`。
原始数组：`arrays/`；源渲染：`images/`；来源记录：`provenance.json`、`checkpoint_provenance.json`。GPU 是否仍有本轮运行进程须由 Launcher/实时进程核查给出，静态报告不猜测终态。
'''
    output=root/'final_report.md'
    if output.exists():raise FileExistsError('Existing final report preserved; use a new comparison directory for a rerun')
    output.write_text(text,encoding='utf-8')
    generated=[output,*sorted((root/'figures').glob('*.png'))]
    record=dict(report_module_sha256=sha256(__file__),generated={str(p.relative_to(root)):sha256(p) for p in generated},
        note='CPU plotting and text aggregation only; no model execution or GPU inference; visual review remains a separate human/agent pixel inspection.')
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
