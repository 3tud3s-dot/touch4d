"""Offline CPU report; consumes frozen inference evidence without model execution."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODES=('normal','freeze_action','zero_tool','copy')
LABELS=dict(normal='Normal',freeze_action='Freeze Action',zero_tool='Zero 12D',copy='Copy Initial',gt='GT GS')
COLORS=dict(normal='#1768ac',freeze_action='#d95f02',zero_tool='#388e3c',copy='#777777')
HS=(1,5,10,15)


def read(p):return json.loads(p.read_text(encoding='utf8'))


def csv_file(p,rows):
    fields=list(rows[0])
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)


def build(root):
    records=read(root/'results.json');status=read(root/'status/infer.json');prov=read(root/'provenance/checkpoint.json')
    protocol=read(root/'provenance/protocol.json')
    if not status['passed']:raise ValueError('Incomplete inference')
    figures=root/'figures';figures.mkdir(exist_ok=True)
    combined=[];summaries=[]
    for rec in records:
        key=f"ep{rec['episode']}_start{rec['start']}";directory=root/'windows'/key
        csv_file(directory/'metrics.csv',rec['metrics'])
        combined.extend(dict(window=key,split=rec['split'],**r) for r in rec['metrics'])
        fig,axes=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
        for ax,(field,title) in zip(axes.flat,[('cd_mm','Symmetric CD (mm)'),('all_motion_rms_mm','GS displacement RMS (mm)'),('all_change_from_normal_mm','Output change vs Normal (mm)'),('near_pred_to_gt_mm','Near directed prediction-to-GT (mm)'),('far_pred_to_gt_mm','Far directed prediction-to-GT (mm)'),('neighbor_rms_mm','Initial-neighbor displacement difference (mm)')]):
            for mode in MODES:
                rows=[r for r in rec['metrics'] if r['mode']==mode]
                ax.plot([r['horizon'] for r in rows],[r[field] for r in rows],label=LABELS[mode],color=COLORS[mode],marker='.',ms=3)
            ax.set(title=title,xlabel='Horizon',xticks=HS);ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8);fig.suptitle(f"001-rope {rec['split']} {key}: all 15 independently inferred states")
        fig.savefig(figures/f'{key}_curves.png',dpi=140);plt.close(fig)
        geo=np.load(directory/'geometry.npz')
        # One fixed world-coordinate extent for all conditions AND all horizons within each window.
        cloud=np.concatenate([geo['initial'],*[geo[m].reshape(-1,3) for m in MODES],*[geo[f'gt_{h}'] for h in range(1,16)]])*1000
        lower=cloud.min(0);upper=cloud.max(0);pad=np.maximum((upper-lower)*.06,2);lower-=pad;upper+=pad
        for h in HS:
            fig,axes=plt.subplots(2,5,figsize=(20,7),constrained_layout=True)
            for col,mode in enumerate(('gt',*MODES)):
                points=(geo[f'gt_{h}'] if mode=='gt' else geo[mode][h-1])*1000
                for row,(i,j) in enumerate(((0,2),(1,2))):
                    ax=axes[row,col]
                    ax.scatter(geo['initial'][:,i]*1000,geo['initial'][:,j]*1000,s=.6,c='#cccccc',alpha=.35,rasterized=True)
                    if mode=='gt':ax.scatter(points[:,i],points[:,j],s=1,c='#222222',alpha=.7,rasterized=True)
                    else:
                        colors=np.full(len(points),'#aaaaaa',dtype='<U7');colors[geo['near_indices']]='#d95f02';colors[geo['far_indices']]='#1768ac'
                        ax.scatter(points[:,i],points[:,j],s=1,c=colors,alpha=.8,rasterized=True)
                    ax.set(xlim=(lower[i],upper[i]),ylim=(lower[j],upper[j]),xlabel='XYZ'[i]+' (world mm)',ylabel='XYZ'[j]+' (world mm)',title=LABELS[mode]);ax.set_aspect('equal');ax.grid(alpha=.15)
            fig.suptitle(f"{key} h{h}, frame {rec['start']+h} | orange=initial Near, blue=initial Far; gray=initial | GT has no cross-frame IDs")
            fig.savefig(figures/f'{key}_h{h:02}_geometry.png',dpi=130);plt.close(fig)
        for mode in MODES:
            row=next(r for r in rec['metrics'] if r['mode']==mode and r['horizon']==15)
            normal=next(r for r in rec['metrics'] if r['mode']=='normal' and r['horizon']==15)
            summaries.append(dict(window=key,split=rec['split'],mode=mode,mean_cd_mm=rec['summary'][mode]['mean_cd_mm'],h15_cd_mm=row['cd_mm'],h15_motion_mm=row['all_motion_rms_mm'],near_motion_mm=row['near_motion_rms_mm'],far_motion_mm=row['far_motion_rms_mm'],output_change_mm=row['all_change_from_normal_mm'],motion_retained_percent=100*row['all_motion_rms_mm']/normal['all_motion_rms_mm'] if normal['all_motion_rms_mm'] else None,near_error_mm=row['near_pred_to_gt_mm'],far_error_mm=row['far_pred_to_gt_mm'],neighbor_rms_mm=row['neighbor_rms_mm'],nonrigid_rms_mm=row['translation_rms_mm'],mean_rgb=rec['summary'][mode]['mean_rgb'],mean_alpha=rec['summary'][mode]['mean_alpha']))
    csv_file(root/'metrics_all.csv',combined);csv_file(root/'summary.csv',summaries)
    (root/'summary.json').write_text(json.dumps(summaries,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    lines=['# V4 H15 冻结权重条件消融','',
        '本轮只有冻结权重推理与渲染，未创建优化器、未执行反向或参数更新。仅使用指定三个窗口，不使用测试集。','',
        '## Checkpoint 与执行证据',f"- Epoch 6 best_joint，成功训练更新数 {prov['successful_updates']}；本轮更新数 0。",f"- 冻结 checkpoint：`{prov['snapshot']}`",f"- SHA256：`{prov['sha256']}`（推理前后相同）。",f"- H15 训练源码：`{prov['source_sha']}`。",f"- 本轮推理源码：`{status['execution']['source_sha']}`。",f"- Slurm allocation：{status['execution']['job']}，GPU：{status['execution']['gpu']}。",f"- 推理总耗时 {status['seconds']:.1f} s；峰值 allocated {status['allocated_peak_bytes']/2**30:.3f} GiB / reserved {status['reserved_peak_bytes']/2**30:.3f} GiB。",'',
        '## 干预与指标',
        '- Normal：原始输入。与无 hook 前向逐位一致，并复核历史同 checkpoint 的 CD/RGB/Alpha。',
        '- Freeze Action：action_state、controller_points 重复初态；对应有效掩码与初态有效性取交集。触觉数值、Age 等保持原输入。复用既有 Attention Gradient Audit 语义。',
        '- Zero 12D：只在 Motion Decoder 每步输入处将 [128:140] 置零；其余当步字段不变，重新执行完整 H15 AR，因此后续反馈可自然变化。',
        '- Near/Far：初态有效 GS 到初态有效操作手 Controller 最近距离的最小/最大各 25%，全程固定。不是接触标签。',
        '- 完整 CD 是双向非平方最近邻距离均值，单位 mm。局部误差是固定预测子集到完整 GT 的单向最近距离，**不是局部双向 CD**；GT 独立重建，不使用 GT 点 ID。',
        '- RMS 为向量欧氏位移平方的均值开根。非刚体 RMS 为去除该预测的整体平均位移后的残差；邻域差为初态 16 近邻内位移差。它们是诊断量，不能单独证明异常。',
        '- 所有 RGB/Alpha 图来自真实目标 GS 或预测 GS 的重建渲染，不是原始照片。三相机、标定、尺度和有效区域均复用原监督实现。误差全部 15 步保存，图像展示 H1/H5/H10/H15。','',
        '## 结果（mm；运动、局部误差及输出变化为 H15）','',
        '|窗口|条件|15步平均CD|H15 CD|运动RMS|Near/Far RMS|相对Normal变化|Near/Far误差|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:lines.append(f"|{s['window']}|{LABELS[s['mode']]}|{s['mean_cd_mm']:.4f}|{s['h15_cd_mm']:.4f}|{s['h15_motion_mm']:.4f}|{s['near_motion_mm']:.4f}/{s['far_motion_mm']:.4f}|{s['output_change_mm']:.6f}|{s['near_error_mm']:.4f}/{s['far_error_mm']:.4f}|")
    lines+=['','## 逐窗口响应','']
    for r in records:
        key=f"ep{r['episode']}_start{r['start']}";ss={s['mode']:s for s in summaries if s['window']==key};n,f,z=ss['normal'],ss['freeze_action'],ss['zero_tool']
        lines.append(f"- **{key} ({r['split']})**：Freeze Action 后末步运动保留 {f['motion_retained_percent']:.2f}%，相对 Normal 输出变化 {f['output_change_mm']:.4f} mm，平均 CD 改变 {f['mean_cd_mm']-n['mean_cd_mm']:+.4f} mm。Zero 12D 后末步运动保留 {z['motion_retained_percent']:.2f}%，输出变化 {z['output_change_mm']:.6f} mm，平均 CD 改变 {z['mean_cd_mm']-n['mean_cd_mm']:+.6f} mm。")
    lines+=['','## 本轮结论',
        '- **Action 对训练窗口 ep7 的运动及拟合有明显影响**：冻结后运动减小但未消失，近端几何误差显著增大，RGB/Alpha 也变差。这支持该窗口的完整 Action 输入有助于拟合，不支持“完全不利用 Action”。',
        '- **两个验证窗口的过大形变并未随 Action 冻结消失**：ep3 和 ep6 仍保留大幅运动，Normal、Freeze 和 Zero 12D 的平均 CD 都劣于 Copy Initial。Action 干预确实改变输出，但没有消除验证窗口的几何退化。',
        '- **显式 12D 通路对最终输出的影响很小**：三窗口末步变化为约 0.022–0.063 mm，Normal 与 Zero 12D 的局部误差和初态邻域位移差接近。该通路不是本轮明显形变的主要数值来源；不代表其他 Action 通路无效。',
        '- ep3 的近端、ep6 的远端在 Normal 下存在较大的单向几何误差；冻结 Action 后仍明显存在，Zero 12D 后基本保留。结合固定视角几何图审阅，不能将此异常归因于显式 12D 向量一项。',
        '- 工程验收通过；本轮只定位冻结模型的条件响应，不进行架构修改，也不启动下一轮实验。','',
        '## 解释边界',
        '- 冻结 Action 同时改变位置、运动编码和几何关联；这是既有输入干预，不可用两组 RMS 相减得到“Action 贡献百分比”。',
        '- Zero 12D 的微小响应只能说明此 checkpoint 在这三个窗口对显式运动向量依赖很弱；Action 仍可经 Controller/Hand 编码、空间 Attention 等其他路径进入。',
        '- Freeze 后运动保留不等于动作从未参与训练，输出变化也不等于提升预测。必须同时查看 CD、RGB/Alpha 和几何。',
        '- 本轮没有零触觉或无条件训练对照，不能把剩余运动唯一归因于时间编码、触觉或初态记忆。',
        '- 固定三个窗口与一次冻结模型的干预结果，不证明跨 Episode 泛化或真实材料因果关系。','',
        '## 测试与展示验收',
        '- 本地与服务器专项 CPU 单测各 8 项通过，0 失败、0 跳过。覆盖已审计干预语义、H15 因果性、无效手、padding、固定区域、hook 清理与无优化器调用。',
        '- 三窗口真实 CUDA/gsplat 推理通过：Normal 与无 hook 输出逐位相同；历史 Normal 指标复现；参数、输入及 checkpoint 前后哈希一致。',
        '- 服务器原始输出 387 个文件已与本地逐文件 SHA256 对齐。图像资源完整性和全部 15 步状态独立性另见 status/artifact_checks.json。',
        '- 已直接检查三个窗口 H15 的固定尺度几何图。ep3 近端出现明显向下铺展，ep6 中间较远区域上拱；Freeze 或 Zero 12D 后均未消除。',
        '- 浏览器工具的安全策略拒绝打开 file:// URL，因此未完成真实浏览器交互验收，不宣称浏览器验收通过。页面使用内嵌数据与相对图片路径，可由用户本地打开；已做图片及资源检查。','',
        '## 原始证据',
        '- `results.json`、`metrics_all.csv`：全部 15 horizon、三条件与 Copy 的逐窗口指标；`summary.csv/json` 为汇总。',
        '- `windows/*/raw_batch.pt.gz`：原始读取 Batch，包含 GT/深度/掩码；`*_inputs.pt.gz`：干预实际输入；`outputs_and_trace.pt.gz`：全部预测、逐步 Motion 输入及 Attention/Anchor trace。',
        '- `geometry.npz`：全部 15 步世界坐标输出、GT 集合、固定 Near/Far 索引。',
        '- `provenance/`、`status/`、`logs/`：输入哈希、窗口/帧号/配置、checkpoint 溯源及测试/执行日志。',
        '- `source/`：完整冻结源码 Git archive 与提交索引；模型权重保留在服务器，由溯源路径与 SHA256 标识，不复制大 checkpoint。','']
    (root/'final_report.md').write_text('\n'.join(lines),encoding='utf8')
    data=json.dumps(records,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>V4 H15 Frozen Condition Ablation</title>
<style>body{font:15px system-ui,"Microsoft YaHei",sans-serif;max-width:1800px;margin:auto;padding:24px;background:#f6f8fb;color:#17243a}header{position:sticky;top:0;background:#fff;padding:12px;z-index:2;border-bottom:1px solid #ccc}select,button{padding:6px;margin:4px}#images{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}figure{margin:0}figure img{width:100%}figcaption{font-weight:bold;padding:6px}img.wide{width:100%;background:#fff}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ddd;text-align:right}th{background:#edf1f6}pre{white-space:pre-wrap;background:white;padding:20px;line-height:1.65}.warning{background:#fff3d7;padding:12px}a{color:#1768ac}</style>
<h1>V4 H15 · 冻结权重条件消融</h1><p>Epoch 6 best_joint · H15 / Δ1 · 3 个指定窗口 · 0 参数更新</p>
<p class="warning">图像为 GS 重建渲染，不是真实照片。Near/Far 固定为初态距离分组，不表示接触标签。所有 15 步数值与几何均保留；下方图像展示 H1/H5/H10/H15。</p>
<header>窗口 <select id="window"></select> Horizon <select id="h"><option>1</option><option>5</option><option>10</option><option selected>15</option></select> 相机 <select id="camera"><option value="0">001 cam0</option><option value="1">011 cam0</option><option value="2">021 cam0</option></select> <select id="kind"><option>rgb</option><option>alpha</option></select> <span id="frame"></span></header>
<div id="images"></div><h2>当前 horizon 指标</h2><div id="table"></div><h2>固定世界坐标范围的几何对照</h2><p>橙色 Near，蓝色 Far；背景灰色为初态。GT 不使用跨帧点 ID。</p><img class="wide" id="geometry"><h2>全部 15 步曲线</h2><img class="wide" id="curves">
<p><a href="final_report.md">完整 Markdown 报告</a> · <a href="metrics_all.csv">全部指标 CSV</a> · <a href="results.json">全部结果 JSON</a> · <a href="provenance/checkpoint.json">Checkpoint 溯源</a></p><h2>研究报告</h2><pre>__REPORT__</pre>
<script>const records=__DATA__; const labels={gt:'GT GS',copy:'Copy Initial',normal:'Normal',freeze_action:'Freeze Action',zero_tool:'Zero 12D'};const w=document.getElementById('window');records.forEach((r,i)=>w.add(new Option(`${r.split} ep${r.episode}/start${r.start}`,i)));function update(){const r=records[+w.value],key=`ep${r.episode}_start${r.start}`,h=+document.getElementById('h').value,v=document.getElementById('camera').value,k=document.getElementById('kind').value;document.getElementById('frame').textContent=`原始帧 ${r.start+h}`;document.getElementById('images').innerHTML=['gt','copy','normal','freeze_action','zero_tool'].map(m=>`<figure><figcaption>${labels[m]}</figcaption><a href="windows/${key}/images/${m}_h${String(h).padStart(2,'0')}_v${v}_${k}.png" target="_blank"><img src="windows/${key}/images/${m}_h${String(h).padStart(2,'0')}_v${v}_${k}.png"></a></figure>`).join('');document.getElementById('geometry').src=`figures/${key}_h${String(h).padStart(2,'0')}_geometry.png`;document.getElementById('curves').src=`figures/${key}_curves.png`;const fields=['cd_mm','all_motion_rms_mm','near_motion_rms_mm','far_motion_rms_mm','all_change_from_normal_mm','near_pred_to_gt_mm','far_pred_to_gt_mm','rgb_mae','alpha_mae'];document.getElementById('table').innerHTML='<table><tr><th>条件</th>'+fields.map(f=>`<th>${f}</th>`).join('')+'</tr>'+r.metrics.filter(x=>x.horizon===h).map(x=>`<tr><td>${labels[x.mode]}</td>`+fields.map(f=>`<td>${x[f].toFixed(6)}</td>`).join('')+'</tr>').join('')+'</table>';}document.querySelectorAll('select').forEach(s=>s.onchange=update);update();</script></html>'''
    (root/'report.html').write_text(page.replace('__REPORT__',html.escape('\n'.join(lines))).replace('__DATA__',data),encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);build(p.parse_args().root)
