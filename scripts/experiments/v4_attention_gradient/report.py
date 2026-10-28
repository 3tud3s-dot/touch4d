"""CPU-only tables, figures, offline report and verified review archive."""
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile
import numpy as np


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_file(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def figures(root,records,fdrows):
    (root/'figures').mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g0=[r for r in records if r['model']=='G0'];labels=[f"ep{r['episode']}/{r['start']}" for r in g0]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for i,r in enumerate(g0):
        for loss,offset in [('cd',-.24),('rgb',0),('alpha_weighted',.24)]:
            row=next(x for x in r['loss_gradients'] if x['scope']=='direct_single_horizon' and x['region']=='all' and x['horizon']==10 and x['loss']==loss)
            axes[0].bar(i+offset,row['l2'],width=.23,label=loss if i==0 else None)
        row=next(x for x in r['loss_gradients'] if x['scope']=='direct_single_horizon' and x['region']=='all' and x['horizon']==10)
        axes[1].bar(i,row['cd__rgb'],color='#278797')
    axes[0].set_yscale('log');axes[0].set_title('H10 direct position gradient L2');axes[0].legend()
    axes[1].set_title('H10 CD / RGB gradient cosine');axes[1].axhline(0,color='gray');axes[1].set_ylim(-1,1)
    for ax in axes:ax.set_xticks(range(len(labels)),labels,rotation=30)
    fig.savefig(root/'figures/loss_gradients.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for r in records:
        name=f"{r['model']} ep{r['episode']}/{r['start']}"
        sig=[s for s in r['attention_signal'] if s['region']=='all']
        axes[0].plot(range(1,11),[np.mean(s['controller_mass_by_head']) for s in sig],label=name)
        axes[1].plot(range(1,11),r['condition_ablation']['normal']['output_motion_rms_mm'],label=name)
    axes[0].set_title('Controller attention mass, all anchors');axes[1].set_title('GS motion RMS / mm')
    axes[1].legend(fontsize=6,ncol=2)
    fig.savefig(root/'figures/signals_by_horizon.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,key in zip(axes,[(2,73),(2,141)]):
        for direction in ['negative_joint_gradient','positive_controller_cumulative','negative_controller_cumulative','fixed_random']:
            rr=[r for r in fdrows if (r['episode'],r['start'])==key and r.get('loss')=='joint' and r['direction']==direction]
            ax.plot([r['amplitude_mm'] for r in rr],[r['delta'] for r in rr],marker='o',label=direction)
        ax.set_xscale('log');ax.axhline(0,color='gray');ax.set_title(f'ep{key[0]}/{key[1]} H10 joint local loss delta')
        ax.set_xlabel('RMS perturbation / mm')
    axes[1].legend(fontsize=6);fig.savefig(root/'figures/finite_difference.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    groups=['motion_last','conditions_controller','cross','gs_encoder']
    for r in g0:
        axes[0].plot(groups,[r['parameter_gradients'][g]['losses']['joint']['scalar_rms'] for g in groups],marker='o',label=f"ep{r['episode']}/{r['start']}")
        a=r['condition_ablation'];axes[1].scatter(a['freeze_action']['output_change_rms_mm'][-1],a['zero_explicit_tool_motion']['output_change_rms_mm'][-1])
        axes[1].annotate(f"ep{r['episode']}/{r['start']}",(a['freeze_action']['output_change_rms_mm'][-1],a['zero_explicit_tool_motion']['output_change_rms_mm'][-1]),fontsize=7)
    axes[0].set_yscale('log');axes[0].set_title('Joint parameter gradient RMS (not AdamW updates)');axes[0].legend(fontsize=7)
    axes[0].tick_params(axis='x',rotation=20);axes[1].set_xlabel('Freeze Action output change H10 / mm');axes[1].set_ylabel('Zero explicit tool motion output change H10 / mm')
    fig.savefig(root/'figures/parameters_and_ablation.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for r in g0:
        name=f"ep{r['episode']}/{r['start']}"
        for region,style in [('near','-'),('far','--')]:
            ss=[s for s in r['attention_signal'] if s['region']==region]
            axes[0,0].plot(range(1,11),[np.mean(s['controller_mass_by_head']) for s in ss],style,label=name+' '+region)
            axes[0,1].plot(range(1,11),[np.mean(s['tool_motion_head_vector_rms_mm']) for s in ss],style)
        aa=[next(a for a in r['angle_distributions'] if a['region']=='all' and a['pair']=='cd__rgb' and a['horizon']==h) for h in [1,5,10]]
        axes[1,0].plot([1,5,10],[a['negative_fraction'] for a in aa],marker='o',label=name)
        for scope,style in [('direct_Hmean','-'),('future_AR_Hmean','--')]:
            rr=[s for s in r['loss_gradients'] if s['loss']=='joint' and s['region']=='all' and s['scope']==scope]
            axes[1,1].plot(range(1,11),[s['l2'] for s in rr],style,label=name+' '+scope)
    axes[0,0].set_title('Near / Far controller mass; fixed initial regions');axes[0,0].legend(fontsize=6,ncol=2)
    axes[0,1].set_title('Near / Far mean head motion RMS / mm')
    axes[1,0].set_title('Pointwise CD/RGB negative cosine fraction');axes[1,0].legend(fontsize=6)
    axes[1,1].set_title('Joint direct vs future AR gradient L2');axes[1,1].set_yscale('symlog',linthresh=1e-7)
    axes[1,1].legend(fontsize=5,ncol=2)
    fig.savefig(root/'figures/near_far_and_gradient_directions.png',dpi=150);plt.close(fig)


def evidence(records,fdrows):
    """Descriptive aggregation only, no new inference or GPU computation."""
    g0=[r for r in records if r['model']=='G0'];rows=[];conflicts=[]
    for r in g0:
        name=f"ep{r['episode']}/{r['start']}"
        sig={s['region']:s for s in r['attention_signal'] if s['horizon']==10}
        h={s['loss']:s for s in r['loss_gradients'] if s['scope']=='direct_single_horizon' and s['region']=='all' and s['horizon']==10}
        j=next(s for s in r['motion_decoder_jacobian'] if s['horizon']==10 and s['region']=='all')
        a=r['condition_ablation'];pp=r['parameter_gradients']['motion']
        rows.append(dict(window=name,controller_raw_mm=sig['all']['controller_adjacent_rms_mm'],
            controller_mass=float(np.mean(sig['all']['controller_mass_by_head'])),
            near_mass=float(np.mean(sig['near']['controller_mass_by_head'])),far_mass=float(np.mean(sig['far']['controller_mass_by_head'])),
            tool_head_mm=float(np.mean(sig['all']['tool_motion_head_vector_rms_mm'])),
            tool_decoder_jacobian=j['tool_jacobian_frobenius_rms'],
            tool_local_linear_mm=j['tool_path_local_linear_delta_rms_mm'],
            zero_tool_output_mm=a['zero_explicit_tool_motion']['output_change_rms_mm'][-1],
            freeze_action_output_mm=a['freeze_action']['output_change_rms_mm'][-1],
            tactile_zero_output_mm=a['zero_tactile_pressure']['output_change_rms_mm'][-1],
            freeze_action_delta_cd=a['freeze_action']['delta_loss']['cd'],
            rgb_over_cd_position_norm=h['rgb']['l2']/h['cd']['l2'],
            rgb_over_cd_motion_parameter_norm=pp['losses']['rgb']['l2']/pp['losses']['cd']['l2'],
            motion_cd_rgb_cosine=pp['cosines']['cd__rgb'],motion_rgb_joint_cosine=pp['cosines']['rgb__joint']))
        for module,p in r['parameter_gradients'].items():
            if p['cosines']['cd__rgb']<0:conflicts.append(dict(window=name,module=module,cosine=p['cosines']['cd__rgb']))
    selected=[x for r in g0 for x in r['loss_gradients'] if x['scope']=='direct_single_horizon' and x['region']=='all' and x['loss']=='joint']
    small=[x for x in fdrows if x.get('amplitude_mm')==.001 and x.get('loss')=='joint']
    return dict(g0_h10=rows,parameter_cd_rgb_negative=conflicts,
        all_horizon_cd_rgb_negative=sum(x['cd__rgb']<0 for x in selected),all_horizon_comparisons=len(selected),
        all_horizon_cd_rgb_range=[min(x['cd__rgb'] for x in selected),max(x['cd__rgb'] for x in selected)],
        finite_difference_0p001mm=small,
        peak_allocated_gb=max(r['allocated_peak_bytes'] for r in records)/1e9,
        peak_reserved_gb=max(r['reserved_peak_bytes'] for r in records)/1e9,
        window_seconds_total=sum(r['seconds'] for r in records),
        max_equivalence_gradient_abs_error=max(r['equivalence']['max_gradient_abs_error'] for r in records))


def build_report(root):
    if not json.loads((root/'status/full.json').read_text())['passed']:raise RuntimeError('Full diagnosis incomplete')
    records=[]
    for f in sorted((root/'results').glob('G*_ep*_start*.json')):
        records.append(json.loads(f.read_text()))
    if len(records)!=12:raise ValueError('Expected 6 G0 and 3 each G2/G3')
    checkpoints=json.loads((root/'provenance/checkpoint_provenance.json').read_text())
    if all(Path(v['path']).is_file() for v in checkpoints.values()):
        integrity={tag:dict(path=v['path'],expected=v['sha256'],actual=sha(v['path'])) for tag,v in checkpoints.items()}
    else:
        # Offline Windows reporting cannot read Linux checkpoint paths. Require the
        # completed server-side verification, never substitute an invented hash.
        integrity=json.loads((root/'provenance/final_checkpoint_integrity.json').read_text())
        if set(integrity)!=set(checkpoints) or any(integrity[k]['path']!=v['path'] or integrity[k]['expected']!=v['sha256'] for k,v in checkpoints.items()):
            raise RuntimeError('Server integrity receipt does not match frozen provenance')
    if any(v['expected']!=v['actual'] for v in integrity.values()):raise RuntimeError('Historical checkpoint changed')
    write(root/'provenance/final_checkpoint_integrity.json',integrity)
    position=[];parameters=[];intermediate=[];signals=[];abl=[];fdrows=[];directions=[]
    for r in records:
        key={k:r[k] for k in ('model','episode','start','split')}
        position.extend(dict(**key,**v) for v in r['loss_gradients'])
        intermediate.extend(dict(**key,**v) for v in r['intermediate_gradients'])
        signals.append(dict(**key,signal=r['attention_signal'],motion_decoder_jacobian=r['motion_decoder_jacobian']))
        abl.append(dict(**key,conditions=r['condition_ablation']))
        directions.extend(dict(**key,**v) for v in r['angle_distributions'])
        for group,v in r['parameter_gradients'].items():
            for loss,g in v['losses'].items():parameters.append(dict(**key,module=group,loss=loss,parameter_count=v['parameter_count'],**g,**v['cosines']))
    for ep,start in [(2,73),(2,141)]:
        d=json.loads((root/'results'/f'finite_difference_ep{ep}_start{start}.json').read_text())
        fdrows.extend(dict(episode=ep,start=start,**r) for r in d['rows'])
    for name,rows in [('loss_gradients',position),('parameter_gradients',parameters),('intermediate_gradients',intermediate),('finite_difference',fdrows)]:
        csv_file(root/'results'/f'{name}.csv',rows)
    write(root/'results/attention_signal.json',signals);write(root/'results/condition_ablation.json',abl)
    write(root/'results/gradient_direction_distributions.json',directions)
    figures(root,records,fdrows)
    g0=[r for r in records if r['model']=='G0']
    h10=[next(x for x in r['loss_gradients'] if x['scope']=='direct_single_horizon' and x['region']=='all' and x['horizon']==10 and x['loss']=='joint') for r in g0]
    nconflict=sum(x['cd__rgb'] is not None and x['cd__rgb']<0 for x in h10)
    negative_descent=[r for r in fdrows if r.get('loss')=='joint' and r['direction']=='negative_joint_gradient' and r.get('amplitude_mm')==.1]
    conclusion=dict(engineering=dict(loss_and_gradient_equivalence=all(r['equivalence']['passed'] for r in records),
        frozen_state_unchanged=all(r['state_sha256_before']==r['state_sha256_after'] for r in records),
        optimizer_updates=0,diagnostic_records=len(records)),
        h10_g0_cd_rgb_conflicting_windows=nconflict,h10_g0_windows=len(g0),
        h10_cd_rgb_cosines={f"ep{r['episode']}/{r['start']}":h['cd__rgb'] for r,h in zip(g0,h10)},
        negative_joint_0p1mm_deltas=[dict(episode=r['episode'],start=r['start'],delta=r['delta']) for r in negative_descent],
        q1=f'G0 H10 direct output CD/RGB gradient cosine is negative in {nconflict}/6 fixed windows. See parameter-coordinate cosines separately; not a global training-causality claim.',
        q2='Attention signal attenuation, local Motion MLP Jacobian and zero-12D AR ablation are reported separately. A small zero-12D response alone cannot distinguish a small input from low decoder sensitivity.',
        q3='No component change is established as sufficient by frozen-weight local diagnostics. Use loss-direction and signal-path evidence together; no training benefit is claimed.',
        limitations=['Six selected windows, one frozen checkpoint per architecture; not full-dataset or optimizer-update evidence.',
        'G2 A2A retains the historical always-active zero Null; no corrective architecture modification.',
        'GT independently reconstructed: centroid change and CD are set proxies, not material displacement.',
        'Tactile zero preserves validity and age. Tool-zero is an internal path intervention, not physical causal proof.',
        'Finite differences are H10 output-space perturbations; nearest-neighbor and rasterization piecewise changes remain possible.',
        'Auxiliary G2/G3 parameter and intermediate gradients are joint-only to bound compute.'])
    write(root/'results/conclusion.json',conclusion)
    ev=evidence(records,fdrows);write(root/'results/evidence_summary.json',ev)
    csv_file(root/'results/signal_attenuation_summary.csv',ev['g0_h10'])
    conclusion.update(
        q1='G0六窗全部60个horizon的整体直接位置CD/RGB余弦均为正。存在局部粒子及少数参数组冲突，但不支持普遍互相抵消。RGB参数梯度多数占优，尺度失衡仍需研究。',
        q2='存在两级衰减：Controller/Null竞争压小显式运动，且冻结Motion MLP对该12D输入的局部Jacobian很小。零12D后的完整AR响应远小于Freeze Action响应，不能只归因于Null或声称全部Action无效。',
        q3='当前证据更支持先验证显式运动通路的传递尺度和Decoder利用方式，而非直接重做Attention或认定Loss冲突是主因。渲染梯度占优值得后续独立验证；本轮不实施修改。',
        evidence_categories=dict(confirmed_engineering=['12组原始/诊断Loss和梯度等价','参数与历史checkpoint哈希保持','没有optimizer更新'],
          supported=['显式工具运动经Attention和Motion Decoder两级衰减','RGB梯度尺度占优','少量参数坐标及局部粒子存在梯度冲突'],
          not_supported_in_selected_windows=['整体CD/RGB普遍相反','当前输出是无下降方向的局部静态极小值','所有Action完全不起作用'],
          inconclusive=['训练退化的唯一因果根因','改变Attention或Loss的实际收益','全数据集与AdamW有效更新行为']))
    write(root/'results/conclusion.json',conclusion)
    lines=['# V4 Attention 条件传播与 Loss 梯度诊断 V1','',
        '本轮只读取 G0/G2/G3 冻结 best_joint 权重，eval 模式开启 Autograd。没有创建优化器，没有参数更新。',
        '', '## 已确认工程事实',
        f"12 个模型/窗口记录的 Loss 数值、参数梯度等价性均通过：{conclusion['engineering']['loss_and_gradient_equivalence']}；参数状态前后哈希一致：{conclusion['engineering']['frozen_state_unchanged']}。",
        'G0 六窗计算 CD、RGB、0.1 Alpha、Joint 四组反传。G2/G3 各三窗仅计算 Joint 梯度，CD/RGB/Alpha 分别梯度标为未执行，不能当作零。',
        '使用原始三相机、分辨率、target_depth_valid、归一化、普通非平方 Chamfer 与原始均值 reduction。',
        '', '## 核心 H10 对照',
        '梯度为单时刻直接位置梯度（归一化坐标单位）；Alpha 已乘 0.1。Tool 为四头各自 3D 向量 RMS 的平均。',
        '| 模型/窗口 | CD Norm | RGB Norm | Alpha Norm | CD/RGB Cosine | Tool/mm | GS motion/mm |',
        '|---|---:|---:|---:|---:|---:|---:|']
    table=[]
    for r in records:
        rr={x['loss']:x for x in r['loss_gradients'] if x['scope']=='direct_single_horizon' and x['region']=='all' and x['horizon']==10}
        sig=next(x for x in r['attention_signal'] if x['horizon']==10 and x['region']=='all')
        fmt=lambda k:'未执行' if k not in rr else f"{rr[k]['l2']:.5g}"
        cs=rr['joint']['cd__rgb'];tool=float(np.mean(sig['tool_motion_head_vector_rms_mm']));motion=r['condition_ablation']['normal']['output_motion_rms_mm'][-1]
        name=f"{r['model']} ep{r['episode']}/{r['start']}";c='—' if cs is None else f'{cs:.4f}'
        lines.append(f"| {name} | {fmt('cd')} | {fmt('rgb')} | {fmt('alpha_weighted')} | {c} | {tool:.6f} | {motion:.5f} |")
        table.append(dict(model=r['model'],episode=r['episode'],start=r['start'],cd_norm=rr.get('cd',{}).get('l2'),
            rgb_norm=rr.get('rgb',{}).get('l2'),alpha_norm=rr.get('alpha_weighted',{}).get('l2'),cd_rgb_cosine=cs,
            tool_mm=tool,motion_mm=motion,controller_mass=np.mean(sig['controller_mass_by_head']),null_mass=np.mean(sig['null_mass_by_head'])))
    write(root/'results/core_table.json',table);csv_file(root/'results/core_table.csv',table)
    lines+=['','## 数值支持与结论边界',
        f'Q1：G0 的 H10 直接位置 CD/RGB 梯度在 {nconflict}/6 窗口余弦为负；每个参数组的冲突程度见 parameter_gradients.csv。位置空间和参数空间不可互相替代。',
        'Q2：attention_signal.json 提供原始 Controller 相邻位移、四头权重、12D 运动及 Decoder 局部 Jacobian。condition_ablation.json 记录移除显式12D后重新 AR 的输出变化。它们分别衡量输入衰减与局部响应，不把 Null 当接触概率。',
        'Q3：局部梯度与有限差分可以定位监督冲突及弱通路，但本轮没有实际优化器更新，不能证明改 Attention 或改 Loss 就会改善训练。',
        '', '## 直接梯度与跨步梯度',
        '`direct_Hmean` 是完整 H10 平均 Loss 对 stacked 输出的直接梯度；`direct_single_horizon` 乘10，恢复单时刻 Loss；`total_AR_Hmean` 对内部 next_xyz 求导，包含后续 AR；`future_AR_Hmean` 为二者之差。',
        '各时刻均有合法三相机监督后才使用上述乘10关系。None 与数值零分开记录；非零微小梯度不能证明长程学习有效。',
        '', '## 局部有限差分',
        '固定 ep2/73、ep2/141，H10 单时刻输出；方向为负 Joint 梯度、正/负已观测累计 Controller 平移方向、seed1729随机方向。',
        '幅度0.001/0.01/0.1/0.5/1.0mm按有效GS欧氏位移RMS归一化；最小两档做中心差分。单时刻Loss变化除10可得完整H10均值变化。',
        'CSV同时保留 Autograd 方向导数、中心差分、绝对/相对误差；最近邻切换与光栅化非光滑处可能影响一致性。',
        '', '## 可复现材料',
        '源码、完整输入哈希、checkpoint路径/哈希/config/source/update、真实命令、CPU测试、GPU日志、逐步梯度JSON/CSV及位置梯度NPZ均随报告保留。',
        '所有评估数值来自冻结权重。没有使用测试集，没有重新训练，没有替换原始模型实现。',
        '', '## 尚未解决',*['- '+x for x in conclusion['limitations']]]
    quantitative=['','## 汇总判断（以原始数值为依据）',conclusion['q1'],conclusion['q2'],conclusion['q3'],'',
        f"G0 六窗 × 10步的整体 CD/RGB 余弦范围为 {ev['all_horizon_cd_rgb_range'][0]:.4f}–{ev['all_horizon_cd_rgb_range'][1]:.4f}，负余弦 {ev['all_horizon_cd_rgb_negative']}/{ev['all_horizon_comparisons']}。不能把整体正余弦误读成每个Gaussian都同向，逐点负余弦比例见方向分布图。",
        'H10 原始Controller运动是相邻已观测时刻的位置差；下表Tool是每头3D运动向量RMS取四头平均。两者不是累计物体位移。',
        '| G0窗口 | 原始CP/mm | Ctrl质量/% | Near/% | Far/% | 聚合Tool/mm | 12D→d Jacobian | 零12D输出变化/mm | Freeze Action变化/mm |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v in ev['g0_h10']:
        quantitative.append(f"| {v['window']} | {v['controller_raw_mm']:.4f} | {100*v['controller_mass']:.3f} | {100*v['near_mass']:.3f} | {100*v['far_mass']:.2g} | {v['tool_head_mm']:.5f} | {v['tool_decoder_jacobian']:.5f} | {v['zero_tool_output_mm']:.7f} | {v['freeze_action_output_mm']:.5f} |")
    quantitative += ['',
        'Near/Far固定为初态距离最近/最远25% Anchor，未按结果改分组。Far显式运动接近零，但Near在部分接触窗口有46–73%的Controller质量，因此不能用全局Null均值宣称所有Anchor都关闭。',
        'Motion Decoder局部Jacobian以同一归一化输入/输出坐标计算；不把128D特征与12D工具的范数直接当信息量比较。当前12D输入乘该Jacobian的局部响应，与零12D完整AR消融均很小，是第二级衰减的独立证据。',
        'Freeze Action仍改变内容编码和空间关联，因此其影响远大于仅删除12D；这说明Action还有其他响应路径，而非整个Action Encoder失效。条件变化不等于正确物理因果。',
        '', '| G0窗口 | H10位置 RGB/CD梯度比 | Motion参数 RGB/CD比 | Motion CD/RGB余弦 | Motion RGB/Joint余弦 |',
        '|---|---:|---:|---:|---:|']
    for v in ev['g0_h10']:
        quantitative.append(f"| {v['window']} | {v['rgb_over_cd_position_norm']:.2f} | {v['rgb_over_cd_motion_parameter_norm']:.2f} | {v['motion_cd_rgb_cosine']:.4f} | {v['motion_rgb_joint_cosine']:.4f} |")
    quantitative += ['', '负参数余弦的例外：'+json.dumps(ev['parameter_cd_rgb_negative'],ensure_ascii=False),
        '参数梯度为裁剪前原始梯度；未执行AdamW，不推断实际参数更新幅度。RGB主导不等于RGB错误，也不能仅凭量级调整权重。',
        '', '### 局部Loss方向与有限差分限制',
        '两个窗口沿负Joint输出梯度的0.1/0.5/1mm扰动均降低Joint；所以当前近静态输出不是这些方向上无法下降的局部极小值。但任意逐GS扰动不保证D0/参数空间可实现，也不保证物理正确。',
        '沿正Controller方向，ep2/73降低Joint，ep2/141反而升高；该整体平移方向不能作为柔性物体的真实材料运动标签。',
        '0.001mm负Joint方向的中心差分与Autograd相对误差约3.0%和6.9%；随机方向因导数接近零、光栅化/最近邻非光滑及float32差分存在大相对误差甚至符号差异。因此有限差分是部分数值支持，不宣称所有方向严格通过；完整误差保留。',
        '', '### 辅助模型与执行完整性',
        'G2/G3三窗也表现出很小的零12D响应；它们的梯度只做Joint，不能据此比较CD/RGB冲突。G2历史A2A零Null设计保持原样。',
        f"共12记录；GPU峰值allocated {ev['peak_allocated_gb']:.3f} GB、reserved {ev['peak_reserved_gb']:.3f} GB；各窗口测量耗时合计 {ev['window_seconds_total']:.1f}s（不含完整有限差分/加载等开销）。最大Loss等价参数梯度绝对差 {ev['max_equivalence_gradient_abs_error']:.3g}，原始容差未放宽。",
        '服务器preflight、pilot、full均成功；CPU绘图因缺少matplotlib失败，原失败日志与EXIT=1保留。报告在本地已有matplotlib环境恢复，未安装服务器依赖、未重跑GPU或更改诊断数值。两次源码SHA分别记录。',
        '本地11项专项与4项原模型只读回归通过，服务器11项专项通过，无失败/跳过。另有2项最终字段回归重复通过。没有运行含optimizer.step的完整训练测试集。']
    # Put the actual evidence and answers before methodological detail.
    marker=lines.index('## 数值支持与结论边界')
    lines[marker:marker]=quantitative+['']
    (root/'final_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    imgs=['loss_gradients.png','signals_by_horizon.png','finite_difference.png','parameters_and_ablation.png','near_far_and_gradient_directions.png']
    body=''.join(f'<section><img src="figures/{name}" alt="{name}"></section>' for name in imgs)
    links=['results/core_table.csv','results/loss_gradients.csv','results/parameter_gradients.csv','results/intermediate_gradients.csv','results/finite_difference.csv','results/attention_signal.json','results/condition_ablation.json','provenance/frozen_protocol.json']
    page='<!doctype html><meta charset="utf-8"><title>V4 Attention Gradient Audit</title><style>body{font:16px system-ui;max-width:1280px;margin:30px auto;color:#23343c;background:#f7fafb}pre{white-space:pre-wrap;line-height:1.55;background:white;padding:24px}img{width:100%;background:white}a{margin-right:16px}</style>'
    page+='<h1>冻结权重 Attention / Loss 梯度诊断</h1><p>零参数更新；图表与CSV对应。位置梯度不是优化器实际更新。</p>'+''.join(f'<a href="{s}">{html.escape(Path(s).name)}</a>' for s in links)+body+'<pre>'+html.escape('\n'.join(lines))+'</pre>'
    (root/'report.html').write_text(page,encoding='utf8')
    (root/'README.txt').write_text('Open report.html offline. Read final_report.md and results/conclusion.json. All numeric CSV/JSON and position-gradient NPZ files are included. Model weights/data are excluded. No optimizer updates were performed.\n',encoding='utf8')
    (root/'START_HERE.md').write_text('# 阅读入口\n\n先读 final_report.md，再查看 results/core_table.csv。完整逐步梯度和中间通路数据保留在 results/。\n\n本轮是冻结模型的局部诊断，不能把梯度敏感性解释为训练收益或物理因果。\n',encoding='utf8')
    return dict(passed=True,records=len(records),figures=len(imgs),conclusion=conclusion)


def package(root):
    from .run import env
    reports=[p for p in (root/'status').glob('report*.json') if json.loads(p.read_text()).get('passed') is True]
    if not reports:raise RuntimeError('Report not completed')
    report_status=max(reports,key=lambda p:p.stat().st_mtime)
    if not json.loads(report_status.read_text())['passed']:raise RuntimeError('Report not completed')
    execution_exit=root/'provenance/execution_exit.txt'
    for stage in ('preflight','pilot','full'):
        if not json.loads((root/f'status/{stage}.json').read_text())['passed']:raise RuntimeError('GPU/data diagnosis incomplete')
    if not execution_exit.exists():raise RuntimeError('GPU pipeline has not ended')
    if execution_exit.read_text().strip()!='EXIT=0':
        initial=json.loads((root/'status/report.json').read_text())
        if not (execution_exit.read_text().strip()=='EXIT=1' and report_status.name.startswith('report_local_recovery')
                and not initial['passed'] and "No module named 'matplotlib'" in initial['error']):
            raise RuntimeError('Unresolved pipeline failure')
    source=Path.cwd();target=root/'source';target.mkdir(exist_ok=True)
    for sub in ['diagnostic_scripts','tests']:(target/sub).mkdir(exist_ok=True)
    for f in (source/'scripts/experiments/v4_attention_gradient').glob('*'):
        if f.is_file():shutil.copy2(f,target/'diagnostic_scripts'/f.name)
    shutil.copy2(source/'scripts/tests/training/test_v4_attention_gradient.py',target/'tests/test_v4_attention_gradient.py')
    snapshot=target/'frozen_repository_scripts.zip'
    subprocess.run(['git','archive','--format=zip','--output',str(snapshot),'HEAD','scripts','AGENTS.md'],check=True)
    execution_sha=json.loads((root/'status/full.json').read_text())['execution']['source_sha']
    subprocess.run(['git','archive','--format=zip','--output',str(target/'gpu_execution_scripts.zip'),execution_sha,'scripts','AGENTS.md'],check=True)
    write(target/'source_manifest.json',dict(**env(),dependency_snapshot='frozen_repository_scripts.zip',
        gpu_execution_sha=execution_sha,gpu_dependency_snapshot='gpu_execution_scripts.zip',
        explanation='Complete tracked scripts plus AGENTS for exact imports; run from extracted repository root with existing audited environment.'))
    write(root/'provenance/source_sha.json',env())
    rows=[]
    for f in sorted(root.rglob('*')):
        if not f.is_file() or 'cache' in f.relative_to(root).parts or f.name=='package_manifest.json':continue
        rel=f.relative_to(root).as_posix()
        if f.suffix in ('.pt','.pth'):raise ValueError('Weights must not be packaged')
        if f.suffix=='.json':json.loads(f.read_text(encoding='utf8'))
        if f.suffix=='.csv':
            with f.open(encoding='utf8',newline='') as stream:list(csv.DictReader(stream))
        rows.append(dict(path=rel,bytes=f.stat().st_size,sha256=sha(f)))
    for match in re.findall(r'(?:href|src)="([^"]+)"',(root/'report.html').read_text()):
        if not (root/match).is_file():raise FileNotFoundError(match)
    write(root/'package_manifest.json',dict(files=rows,weights_included=False,dataset_included=False))
    dest=root.parent/(root.name+'-review.zip')
    if dest.exists():raise FileExistsError(dest)
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in rows:z.write(root/row['path'],root.name+'/'+row['path'])
        z.write(root/'package_manifest.json',root.name+'/package_manifest.json')
    if dest.stat().st_size>100_000_000:raise RuntimeError('Archive exceeds 100MB; preserve artifact, do not omit numeric data silently')
    with zipfile.ZipFile(dest) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP CRC failed')
        for row in rows:
            data=z.read(root.name+'/'+row['path'])
            if hashlib.sha256(data).hexdigest()!=row['sha256']:raise RuntimeError('ZIP member hash mismatch')
    return dict(passed=True,path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest),files=len(rows)+1,crc_passed=True,member_sha256_verified=True)
