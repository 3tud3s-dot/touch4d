"""Offline geometry plots and trajectories; no renderer, GPU or network required."""
import html
import json
from pathlib import Path
import numpy as np
from scripts.training.visualize import geometry_data, geometry_html


def curve(series,title,target=None):
    vals=[v for _,rows,color in series for _,v in rows]
    if target is not None: vals.append(target)
    lo=min(0.,min(vals)); hi=max(vals); span=max(hi-lo,1e-9)
    out=f'<figure><figcaption>{html.escape(title)} · {lo:.3g}–{hi:.3g}</figcaption><svg viewBox="0 0 640 230"><path d="M35 20V200H620" stroke="#aaa" fill="none"/>'
    for name,rows,color in series:
        pts=' '.join(f'{35+n/400*585:.2f},{200-(v-lo)/span*175:.2f}' for n,v in rows)
        out+=f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>'
    if target is not None:
        y=200-(target-lo)/span*175; out+=f'<path d="M35 {y}H620" stroke="#777" stroke-dasharray="5 5"/>'
    return out+'</svg><small>蓝：有条件；橙：无条件；横轴0–400 updates。虚线：目标。</small></figure>'


def scatter(initial,gt,pred,title,axes):
    ids=np.linspace(0,len(initial)-1,min(1000,len(initial)),dtype=int)
    arrays=[initial[ids][:,axes],gt[ids][:,axes],pred[ids][:,axes]]
    both=np.concatenate(arrays); lo=both.min(0); hi=both.max(0); span=max((hi-lo).max(),.001)
    center=(lo+hi)/2
    factor=min(400/max(hi[0]-lo[0],.001),190/max(hi[1]-lo[1],.001))
    out=f'<figure><figcaption>{html.escape(title)}</figcaption><svg viewBox="0 0 440 230">'
    for pts,color in zip(arrays,['#aab2bc','#198556','#ce5b27']):
        pos=(pts-center)*factor
        out+=''.join(f'<circle cx="{220+x:.2f}" cy="{115-y:.2f}" r="1" fill="{color}" fill-opacity=".55"/>' for x,y in pos)
    return out+'</svg></figure>'


def write_report(directory):
    directory=Path(directory); results=json.loads((directory/'results.json').read_text())
    manifest=json.loads((directory/'data_manifest.json').read_text()); figures=directory/'figures'; figures.mkdir(exist_ok=True)
    table=''; projections=''; links=''; selected={}
    for mode,d in results.items():
        best=next(r for r in d['evaluations'] if r['update']==d['best_update']); last=d['evaluations'][-1]
        selected[mode]=dict(best=best,last=last,best_update=d['best_update'],perturbations=d['perturbations'])
        for label,r in [('best',best),('last',last)]:
            for b,name in enumerate(('A (+Z)','B (−Z)')):
                table+=f'<tr><td>{mode}</td><td>{label}/{r["update"]}</td><td>{name}</td><td>{r["branch_mean_cd_mm"][b]:.5f}</td>'
                table+=''.join(f'<td>{r["cd_mm"][b][h]:.5f}</td>' for h in (0,4,9))
                table+=f'<td>{r["affected_signed_z_mm"][b][-1]:.3f}</td><td>{r["predicted_separation_mm"][-1]:.3f}</td></tr>'
        for label in ('best','last'):
            with np.load(directory/mode/f'{label}_predictions.npz') as z:
                for b in range(2):
                    xyz=z['initial_xyz'][b]; attr=z['initial_attr'][b]
                    arrays=dict(initial_xyz=xyz,initial_attributes=attr,frame_ids=z['frame_ids'][b],
                                predicted_xyz=z['predicted_xyz'][b],predicted_log_scales=np.broadcast_to(attr[None,:,:3],(10,len(xyz),3)),
                                predicted_quaternions=np.broadcast_to(attr[None,:,3:7],(10,len(xyz),4)))
                    for h in range(10): arrays[f'target_xyz_{h+1}']=z['target_xyz'][b,h]; arrays[f'target_attr_{h+1}']=attr
                    name=f'{mode}_{label}_{"AB"[b]}.html'
                    page=geometry_html(geometry_data(arrays),f'{mode} / {label} / branch {"AB"[b]}')
                    page=page.replace('<a href="index.html">RGB / Alpha 多相机渲染</a>','<a href="../report.html">返回配对实验报告</a>')
                    page=page.replace('未来 GT 是独立点集，没有跨帧粒子 ID；仅预测模板可连成轨迹。','本实验目标由同一模板人工形变生成，点序号可对应；训练仍使用点集 Chamfer。')
                    (figures/name).write_text(page,encoding='utf-8')
                    links+=f'<option value="figures/{name}">{mode} / {label} / {"AB"[b]}</option>'
                    if label=='best':
                        for h in (0,4,9):
                            for axes,plane in [((0,2),'XZ'),((1,2),'YZ')]:
                                projections+=scatter(xyz,z['target_xyz'][b,h],z['predicted_xyz'][b,h],f'{mode} · {"AB"[b]} · h{h+1} · {plane}',axes)
    colors=['#2875bd','#d47724']
    charts=''
    for title,read in [('平均 CD (mm)',lambda r:r['mean_cd_mm']),('末步 A/B 全点分离 RMS (mm)',lambda r:r['predicted_separation_mm'][-1])]:
        series=[(m,[(r['update'],read(r)) for r in d['evaluations']],c) for (m,d),c in zip(results.items(),colors)]
        target=results['conditioned']['evaluations'][0]['target_separation_mm'][-1] if '分离' in title else None
        charts+=curve(series,title,target)
    perturb=''
    for mode,d in results.items():
        for key,r in d['perturbations'].items():
            perturb+=f'<tr><td>{mode}</td><td>{key}</td><td>{r["affected_signed_z_mm"][0][-1]:.3f}</td><td>{r["affected_signed_z_mm"][1][-1]:.3f}</td><td>{r["predicted_separation_mm"][-1]:.3f}</td></tr>'
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>V4 Paired Action Identifiability</title>
<style>body{{font:15px system-ui;margin:24px;background:#f6f8fb;color:#18314c}}table{{border-collapse:collapse;background:white}}td,th{{padding:8px;border:1px solid #ced9e4}}.grid{{display:grid;grid-template-columns:repeat(2,1fr)}}figure{{margin:8px;padding:12px;background:white}}svg{{width:100%}}iframe{{width:100%;height:800px;border:1px solid #bbb}}select{{padding:10px}}pre{{white-space:pre-wrap}}</style>
<h1>V4 · Paired Action Identifiability V1</h1>
<p>真实 001-rope ep2/frame73 初态 + 合成 Action/GT；H10/Δ1；A/B同批，每模型400次，seed42。纯几何诊断 CD；正式 V4 联合 Loss 未修改。首步条件与目标相同。</p>
<p>近端目标有符号末步位移：±{manifest['target_affected_z_last_mm']:.3f} mm；最大±10 mm。局部权重是人为生成规则，不是绳索物理模型。灰=初态、绿=GT、橙=预测。</p>
<table><tr><th>模型</th><th>Checkpoint</th><th>分支</th><th>平均CD/mm</th><th>h1</th><th>h5</th><th>h10</th><th>近端Z/mm</th><th>A/B分离RMS/mm</th></tr>{table}</table>
<div class="grid">{charts}</div><h2>完整十帧几何与轨迹</h2><select id="pick">{links}</select><iframe id="view" title="十帧几何"></iframe>
<script>const p=document.getElementById('pick'),v=document.getElementById('view');p.onchange=()=>v.src=p.value;p.onchange();</script>
<h2>H1/H5/H10 两组目标与预测</h2><div class="grid">{projections}</div>
<h2>冻结权重的动作扰动（best checkpoint）</h2><table><tr><th>模型</th><th>输入</th><th>A近端Z/mm</th><th>B近端Z/mm</th><th>分离RMS/mm</th></tr>{perturb}</table>
<p>static/half 为只读响应探针，没有作为训练样本；不要求其符合严格线性插值。swap等于交换两个训练输入，不构成独立泛化证据。完整逐horizon指标、每个Head的Attention/Null、梯度与更新量见JSON。</p>
<p><a href="results.json">完整指标</a> · <a href="data_manifest.json">生成规则及数据指纹</a> · <a href="config.json">配置/源码SHA</a> · <a href="smoke/acceptance.json">CUDA验收</a> · <a href="conditioned/training.csv">有条件训练CSV</a> · <a href="unconditioned/training.csv">无条件训练CSV</a></p>
<p>本实验只检验受控条件通路的可学习性，不验证真实接触物理、跨Episode泛化或真实触觉收益。</p></html>'''
    (directory/'report.html').write_text(body,encoding='utf-8')
    (directory/'summary.json').write_text(json.dumps(selected,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('directory');write_report(p.parse_args().directory)
