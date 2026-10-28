"""CPU-only offline figure/report assembly; no checkpoint or model execution."""
import argparse,csv,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw

HS=(5,10,15)
MODES=('copy','normal')
LABELS={'gt':'GT GS','copy':'Copy Initial','normal':'H15 Normal'}


def load(p):return json.loads(p.read_text(encoding='utf8'))


def choose(rows,loss,snapshot='best',init=None):
    candidates=[r for r in rows if r['method']=='D0-'+loss and r['snapshot']==snapshot and (init is None or r['initialization']==init)]
    return min(candidates,key=lambda r:r[loss])


def mode_of(row):return row['method'][3:]+'_'+row['initialization']+'_'+row['snapshot']


def build(root):
    complete=load(root/'provenance/completion.json')
    assert complete['passed'] and complete['fits']==24 and complete['network_updates']==0
    correction=dict(raw_record='completion.json',affected_fields=['execution.no_optimizer','execution.no_backward'],
                    reason='Metadata helper inherited flags from frozen forward-only audit; Oracle did perform D-only backward and Adam steps. Raw execution record preserved unchanged.',
                    optimizer_scope='oracle_anchor_displacements_only',oracle_updates=7200,network_updates=0,
                    evidence=['config.json','pilot.json','../results/optimization_curves.csv','../arrays/*.pt'])
    (root/'provenance/metadata_erratum.json').write_text(json.dumps(correction,indent=2),encoding='utf8')
    rows=load(root/'results/results.json');curves=list(csv.DictReader((root/'results/optimization_curves.csv').open(encoding='utf8')))
    assert len(curves)==24*301
    summary=[];choices={}
    for ep,start in ((7,105),(3,135)):
        prefix=f'ep{ep}_start{start}'
        loaded={h:np.load(root/'arrays'/f'{prefix}_h{h:02}.npz') for h in HS}
        points=np.concatenate([a[k].reshape(-1,3) for a in loaded.values() for k in a.files if not k.endswith('_heat') and k not in ('near','far')])*1000
        low=points.min(0);high=points.max(0);pad=np.maximum((high-low)*.06,2);low-=pad;high+=pad
        heatmax=max(float(a[k].max()) for a in loaded.values() for k in a.files if k.endswith('_heat'))
        for h,a in loaded.items():
            key=f'{prefix}_h{h:02}';rr=[r for r in rows if r['window']==key]
            cc=choose(rr,'cd');jj=choose(rr,'joint');choices[key]={'cd':mode_of(cc),'joint':mode_of(jj)}
            normal=next(r for r in rr if r['method']=='normal');copy=next(r for r in rr if r['method']=='copy')
            for r in rr:
                summary.append(dict(r,cd_improvement_vs_copy_percent=100*(1-r['cd_mm']/copy['cd_mm']),cd_improvement_vs_normal_percent=100*(1-r['cd_mm']/normal['cd_mm'])))
            modes=['gt','copy','normal',mode_of(cc),mode_of(jj)]
            labels=[LABELS.get(m,m) for m in modes]
            fig,axes=plt.subplots(3,5,figsize=(20,10),constrained_layout=True)
            for c,m in enumerate(modes):
                p=a[m]*1000
                for r,(i,j) in enumerate(((0,2),(1,2))):
                    ax=axes[r,c];ax.scatter(p[:,i],p[:,j],s=.9,c='#333333',alpha=.6,rasterized=True)
                    ax.set(xlim=(low[i],high[i]),ylim=(low[j],high[j]),xlabel='XYZ'[i]+' world mm',ylabel='XYZ'[j]+' world mm',title=labels[c]);ax.set_aspect('equal');ax.grid(alpha=.15)
                ax=axes[2,c]
                if m=='gt':ax.text(.5,.5,'Independent GT reconstruction\nNo material IDs / strain truth',ha='center',va='center',transform=ax.transAxes);ax.axis('off')
                else:
                    sc=ax.scatter(p[:,0],p[:,2],s=2,c=a[m+'_heat'],vmin=0,vmax=max(heatmax,1e-8),cmap='magma',rasterized=True)
                    ax.set(xlim=(low[0],high[0]),ylim=(low[2],high[2]),xlabel='X world mm',ylabel='Z world mm');ax.set_aspect('equal')
            fig.colorbar(sc,ax=list(axes[2,1:]),shrink=.55,label='Mean absolute initial 16-NN distance ratio change (proxy)')
            fig.suptitle(f'{prefix} H{h}, target frame {start+h}; fixed axes + heat scale across all H and initializations\nOracle columns: lowest own objective across both starts; all starts/last saved separately')
            fig.savefig(root/'figures'/f'{prefix}_h{h}_geometry.png',dpi=140);plt.close(fig)
            # Fixed-template trajectories only: never draw GT row-wise arrows.
            fig,axes=plt.subplots(1,4,figsize=(18,5),constrained_layout=True)
            for ax,m in zip(axes,modes[1:]):
                p=a[m]*1000;x=a['initial']*1000;ax.scatter(x[:,0],x[:,2],s=.7,c='#cccccc')
                for group,color in [('near','#d95f02'),('far','#1768ac')]:
                    ids=a[group];ids=ids[::max(1,len(ids)//70)];delta=p[ids]-x[ids]
                    ax.quiver(x[ids,0],x[ids,2],delta[:,0],delta[:,2],angles='xy',scale_units='xy',scale=1,color=color,width=.004,label=group)
                ax.set(xlim=(low[0],high[0]),ylim=(low[2],high[2]),title=LABELS.get(m,m),xlabel='X world mm',ylabel='Z world mm');ax.set_aspect('equal')
            axes[0].legend();fig.suptitle(f'{key}: initial Near/Far displacement vectors; equal world scale, no arrow amplification')
            fig.savefig(root/'figures'/f'{key}_displacements.png',dpi=140);plt.close(fig)
            fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
            for loss in ('cd','joint'):
                for init in ('zero','normal'):
                    curve=[r for r in curves if r['window']==key and r['loss']==loss and r['initialization']==init]
                    for ax,field in zip(axes,['cd_mm','joint','previous_gradient_norm']):
                        ax.plot([int(r['step']) for r in curve],[float(r[field]) for r in curve],label=loss+'/'+init,ls='--' if init=='zero' else '-')
                        ax.set(title=field,xlabel='Successful Oracle updates');ax.grid(alpha=.2)
            axes[0].axhline(normal['cd_mm'],c='gray',ls=':',label='Normal');axes[0].legend(fontsize=8);fig.suptitle(key)
            fig.savefig(root/'figures'/f'{key}_curves.png',dpi=140);plt.close(fig)
            # Five-column 3-camera contact sheets for RGB and alpha, selected best only.
            for kind in ('rgb','alpha'):
                canvas=Image.new('RGB',(5*512,3*310+45),'white');draw=ImageDraw.Draw(canvas)
                for c,label in enumerate(labels):draw.text((c*512+8,10),label,fill='black')
                for v in range(3):
                    for c,m in enumerate(modes):
                        im=Image.open(root/'images'/f'{key}_{m}_v{v}_{kind}.png').convert('RGB');im.thumbnail((512,288));canvas.paste(im,(c*512,v*310+45));draw.text((c*512+8,v*310+335),f'camera {v}',fill='black')
                canvas.save(root/'images'/f'{key}_{kind}_three_camera.png')
    with (root/'results/summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    cp=load(root/'provenance/checkpoint.json')
    lines=['# V4 D0 Oracle 表达能力诊断 V1','',
        '## 协议与工程验收',
        '- 仅 Train ep7/start105、Val ep3/start135；分别独立拟合目标 H5、H10、H15。Val GT 仅用于离线 Oracle，不训练网络或选择超参数。',
        '- 固定初态 GS、128 FPS Anchor、初态 RBF 和其余 11D 属性；唯一变量为该时刻 128×3 累计 Anchor 位移。无 GT 点 ID、无单点 XYZ 优化。',
        '- 两种初始化（零、对应 horizon 的原始 H15 预测累计 Anchor 位移）× CD/Joint × 两窗口 × 三时刻，共 24 次，均 300 次更新。',
        '- Adam lr=0.001、betas=(0.9,0.999)、eps=1e-8，无衰减、无裁剪、无正则、无调参。Pilot 是第一拟合的首步，计入 300 次。',
        '- CD 使用生产非平方双向最近邻均值；Joint=CD+RGB+0.1 Alpha。每个 horizon 单独拟合，沿用原像素/相机/sample reduction，不将三个目标混成一个优化问题。',
        '- 零位移与 Copy 逐位一致；各 H 的预测 Anchor 复现 Normal 逐位一致；重建 FPS/RBF 逐位一致。真实 CUDA Pilot 验证联合 Loss 与生产 objective、梯度及参数冻结。',
        '- GPU 前本地与服务器各 8 项专项测试通过，收尾元数据回归后共 9 项。曾修正新诊断邻域统计索引错误；首个 GPU 尝试在优化前因坐标导出缺少 mask 停止，已修复并在 run-v2 重跑，失败记录保留在父目录；无生产模型改动。',
        '- 元数据勘误：原 completion.json 的 execution.no_optimizer/no_backward 沿用了前一纯推理审计的标志，属于日志字段错误。本轮实际执行了 7200 次 D-only Adam 更新和反向，网络更新为0。原记录未改写，正确语义与证据见 provenance/metadata_erratum.json；工具与回归测试已修正。',
        f"- 推理/优化源码 `{complete['execution']['source_sha']}`，训练源码 `{cp['source_sha']}`；Slurm {complete['execution']['job']}，{complete['execution']['gpu']}。",
        f"- Checkpoint `{cp['snapshot']}`；SHA256 `{cp['sha256']}`。网络参数与文件前后哈希不变，网络更新数 0。",
        f"- GPU 总耗时 {complete['seconds']:.1f} s，峰值 allocated {complete['peak_allocated']/2**30:.3f} / reserved {complete['peak_reserved']/2**30:.3f} GiB。",'',
        '## 结果：两个初始化中最低自身目标的 best（完整双初始化/best/last 均在 CSV）','',
        '|窗口/H|Copy CD mm|Normal CD mm|D0-CD mm（初始化/步数）|D0-Joint CD mm（初始化/步数）|CD Oracle 结构失真 P95|',
        '|---|---:|---:|---:|---:|---:|']
    for ep,start in ((7,105),(3,135)):
        for h in HS:
            key=f'ep{ep}_start{start}_h{h:02}';rr=[r for r in rows if r['window']==key];c=next(r for r in rr if r['method']=='copy');n=next(r for r in rr if r['method']=='normal');o=choose(rr,'cd');j=choose(rr,'joint')
            lines.append(f"|{key}|{c['cd_mm']:.4f}|{n['cd_mm']:.4f}|{o['cd_mm']:.4f} ({o['initialization']}/{o['selected_step']})|{j['cd_mm']:.4f} ({j['initialization']}/{j['selected_step']})|{o['distortion_p95']:.3f}|")
    lines+=['','## 收敛与初始化差异','', '|窗口/H|Loss|初始化|初始目标|best目标/step|last目标|250→300目标变化|','|---|---|---|---:|---:|---:|---:|']
    for key in choices:
        rr=[r for r in rows if r['window']==key]
        for loss in ('cd','joint'):
            for init in ('zero','normal'):
                cc=[r for r in curves if r['window']==key and r['loss']==loss and r['initialization']==init];b=choose(rr,loss,init=init);last=choose(rr,loss,'last',init)
                lines.append(f"|{key}|{loss}|{init}|{float(cc[0]['objective']):.6f}|{b[loss]:.6f}/{b['selected_step']}|{last[loss]:.6f}|{float(cc[-1]['objective'])-float(cc[250]['objective']):+.6f}|")
    lines+=['','## Q1：D0-CD 是否能明显降低几何误差？',
        '能。在这两个固定窗口和三个目标时刻，直接优化 Anchor 位移可以明显低于 Copy 与 Normal。H15：ep7 的 CD 从 Normal 2.942 mm 降到 1.684 mm；ep3 从 11.236 mm 降到 1.721 mm。两种初始化均有改善，说明并非只能依赖原网络给出正确起点。这里的“明显”指观测到的数值幅度，不是统计显著性检验。',
        '', '## Q2：低 CD 时是否仍有局部失真？',
        '仍有非零局部拉伸/压缩，不能视为真实材料应变。但关键 H15 图中，Normal 的严重铺展并非 D0 不可避免的结果：ep3 的近端扇形铺展在两种 Oracle 主图中明显缓解。选定 CD Oracle 的 P95 |ratio−1|：ep7 为0.327（Normal 0.514），ep3 为0.281（Normal 1.090）。选定 Joint Oracle 则分别为0.286和0.112。',
        '初始化仍影响结果：ep3/H15 Joint 从退化 Normal 初始化的 P95 为0.559，明显高于零初始化的0.112。不能只看主图而忽略这一局部解差异。ep7 的局部末端形态仍不能与独立重建 GT 完全一致；固定初态属性与点数也是限制。三相机图显示 Joint 通常更贴近渲染目标，但不证明真实接触物理。',
        '', '## Q3：优先条件 Attention / 动力学，还是 Decoder？',
        '本轮更支持先定位网络为何未预测出合适的 Anchor 累计位移，以及条件信息利用、递推与优化问题，而不是把当前退化首先归因于固定 D0 的几何表达不足。特别是 ep3，在同一 D0、同一联合监督下，零初始化 Joint Oracle 达到 CD 1.652 mm、RGB 0.02435、Alpha 0.01459；Normal 为11.236 mm、0.28106、0.39279。',
        '这并未确定 Attention 是唯一根因，也没有证明 D0 在所有柔性形变上足够。Oracle 移除了条件预测难度并使用未来 GT；全部预算为300步，若曲线末端仍改善、初始化间有差异，不能称为理论表达上限。下一步优先分析“条件/递推到 Anchor 位移”的差距；本轮到此停止，不实施新的模块。',
        '', '## 图像与文件验收',
        '- 直接检查两张 H15 几何主图及 ep3/H15 三相机 RGB 拼图；426张 PNG 的解码完整性通过。60条总结、7224条逐步曲线完整，所有原始 Normal 指标复现前一冻结审计。',
        '- 未执行真实浏览器交互验收：此前浏览器工具拒绝 file:// 路径。离线 HTML 使用内嵌 JSON 和相对图片，无 CDN。',
        '', '## 解释边界',
        '- 300 次是有界优化，不是收敛证明或 Decoder 的理论上限。best 与 last、两种初始化的差异均须结合曲线看。',
        '- Near/Far 固定为初态到有效操作手 Controller 距离排序的最小/最大各25%；局部误差为该预测子集到完整 GT 的单向距离，不是局部双向 CD。',
        '- 16-NN 是初态固定有向边，去除自边及初始距离≤1e-8的退化边。ratio=d_pred/d_initial；P95失真=quantile95(|ratio−1|)。这些不是材料应变或物理真实性证明。',
        '- 图中世界坐标范围及热图颜色范围在同一窗口的所有 H、初始化、best/last 之间固定；位移箭头不放大。',
        '- 图像是真实目标/预测 GS 的三相机渲染，并非原始 RGB 照片。各条件继承同一初态属性；GT 各帧独立重建。',
        '- 本轮只诊断表达与可优化性，Oracle 的未来 GT 信息不会进入正常 V4 推理。','',
        '## 证据路径',
        '- `results/summary.csv`：全量 baseline/Oracle best/last、两初始化、改善幅度及所有几何/渲染指标。',
        '- `results/optimization_curves.csv`：24×301条（含step0），记录每步 CD、RGB、Alpha、Joint、梯度及更新量。',
        '- `results/geometry_metrics.csv`：全部原始结构指标；`arrays/`：全部预测集合、热图、best/last Anchor 和优化器状态。',
        '- `figures/ep3_start135_h15_geometry.png`、`figures/ep7_start105_h15_geometry.png`：五列 H15 主图。',
        '- `images/*_three_camera.png`：三相机五列对照；全部单条件原始分辨率图片同目录保存。',
        '- `provenance/`：配置、数据来源、等价检查、Pilot、checkpoint和完成记录；`logs/`：测试及执行日志。','']
    (root/'final_report.md').write_text('\n'.join(lines),encoding='utf8')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>D0 Oracle Capacity</title><style>body{font:15px system-ui,"Microsoft YaHei";margin:24px;background:#f6f8fb;color:#17243a}header{position:sticky;top:0;background:white;padding:12px;z-index:3}select{padding:7px;margin:5px}.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}img{max-width:100%}.wide{width:100%}pre{white-space:pre-wrap;background:#fff;padding:18px;line-height:1.7}figure{margin:0}figcaption{padding:6px;font-weight:bold}</style><h1>D0 Oracle · 冻结网络表达能力诊断</h1><p>仅优化 Anchor 位移；三相机 GS 渲染，不是真实照片。GT 无材料点 ID；低 CD 不等于物理正确。</p><header><select id="w"><option>ep7_start105</option><option>ep3_start135</option></select><select id="h"><option>5</option><option>10</option><option selected>15</option></select><select id="init"><option value="selected">两初始化的最低自身目标 best</option><option>zero</option><option>normal</option></select><select id="snapshot"><option>best</option><option>last</option></select><select id="v"><option value="0">Camera 001</option><option value="1">Camera 011</option><option value="2">Camera 021</option></select><select id="kind"><option>rgb</option><option>alpha</option></select></header><div class="grid" id="images"></div><h2>固定尺度几何与结构失真（默认 best 选择）</h2><img class="wide" id="geo"><img class="wide" id="disp"><h2>完整收敛曲线</h2><img class="wide" id="curve"><p><a href="results/summary.csv">全部结果 CSV</a> · <a href="results/optimization_curves.csv">全部优化曲线</a> · <a href="results/geometry_metrics.csv">结构指标</a> · <a href="final_report.md">Markdown 报告</a></p><pre>__REPORT__</pre><script>const choices=__CHOICES__;function update(){const val=id=>document.getElementById(id).value,w=val('w'),h=+val('h'),key=w+'_h'+String(h).padStart(2,'0'),init=val('init'),snap=val('snapshot'),v=val('v'),kind=val('kind');let modes=['gt','copy','normal',...['cd','joint'].map(l=>init==='selected'?choices[key][l]:l+'_'+init+'_'+snap)];document.getElementById('images').innerHTML=modes.map(m=>`<figure><figcaption>${m}</figcaption><a target="_blank" href="images/${key}_${m}_v${v}_${kind}.png"><img src="images/${key}_${m}_v${v}_${kind}.png"></a></figure>`).join('');document.getElementById('geo').src=`figures/${w}_h${h}_geometry.png`;document.getElementById('disp').src=`figures/${key}_displacements.png`;document.getElementById('curve').src=`figures/${key}_curves.png`;document.getElementById('snapshot').disabled=init==='selected';}document.querySelectorAll('select').forEach(x=>x.onchange=update);update();</script></html>'''
    (root/'report.html').write_text(page.replace('__REPORT__',html.escape('\n'.join(lines))).replace('__CHOICES__',json.dumps(choices)),encoding='utf8')
    (root/'results/display_choices.json').write_text(json.dumps(choices,indent=2),encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);build(p.parse_args().root)
