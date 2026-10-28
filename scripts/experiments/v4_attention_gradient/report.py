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


def build_report(root):
    if not json.loads((root/'status/full.json').read_text())['passed']:raise RuntimeError('Full diagnosis incomplete')
    records=[]
    for f in sorted((root/'results').glob('G*_ep*_start*.json')):
        records.append(json.loads(f.read_text()))
    if len(records)!=12:raise ValueError('Expected 6 G0 and 3 each G2/G3')
    checkpoints=json.loads((root/'provenance/checkpoint_provenance.json').read_text())
    integrity={tag:dict(path=v['path'],expected=v['sha256'],actual=sha(v['path'])) for tag,v in checkpoints.items()}
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
    (root/'final_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    imgs=['loss_gradients.png','signals_by_horizon.png','finite_difference.png','parameters_and_ablation.png']
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
    if not json.loads((root/'status/report.json').read_text())['passed']:raise RuntimeError('Report not completed')
    execution_exit=root/'provenance/execution_exit.txt'
    if not execution_exit.exists() or execution_exit.read_text().strip()!='EXIT=0':raise RuntimeError('GPU pipeline has not ended successfully')
    source=Path.cwd();target=root/'source';target.mkdir(exist_ok=True)
    for sub in ['diagnostic_scripts','tests']:(target/sub).mkdir(exist_ok=True)
    for f in (source/'scripts/experiments/v4_attention_gradient').glob('*'):
        if f.is_file():shutil.copy2(f,target/'diagnostic_scripts'/f.name)
    shutil.copy2(source/'scripts/tests/training/test_v4_attention_gradient.py',target/'tests/test_v4_attention_gradient.py')
    snapshot=target/'frozen_repository_scripts.zip'
    subprocess.run(['git','archive','--format=zip','--output',str(snapshot),'HEAD','scripts','AGENTS.md'],check=True)
    write(target/'source_manifest.json',dict(**env(),dependency_snapshot='frozen_repository_scripts.zip',
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
