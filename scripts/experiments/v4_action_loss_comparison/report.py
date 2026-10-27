"""Offline CD/MSE comparison; all geometry panels use shared world coordinates."""
import html
import json
from pathlib import Path
import numpy as np
from scripts.training.visualize import geometry_data,geometry_html


def chart(title,series,reference=None):
    values=[y for name,rows,color in series for x,y in rows]
    if reference is not None:values.append(reference)
    lo=min(0,min(values));hi=max(values);span=max(hi-lo,1e-12)
    text=f'<figure><h3>{html.escape(title)}</h3><svg viewBox="0 0 650 250"><path d="M55 20V210H630" stroke="#999" fill="none"/>'
    for name,rows,color in series:
        points=' '.join(f'{55+x/400*575:.2f},{210-(y-lo)/span*180:.2f}' for x,y in rows)
        text+=f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
    if reference is not None:
        y=210-(reference-lo)/span*180;text+=f'<path d="M55 {y}H630" stroke="#555" stroke-dasharray="5 4"/>'
    text+=f'<text x="0" y="25">{hi:.3g}</text><text x="0" y="210">{lo:.3g}</text><text x="55" y="235">0</text><text x="600" y="235">400</text></svg>'
    text+=' · '.join(f'<span style="color:{color}">{html.escape(name)}</span>' for name,rows,color in series)
    return text+'</figure>'


def projection(x,t,p,title,axes,bounds):
    lo,hi=bounds;lo=lo[list(axes)];hi=hi[list(axes)];center=(lo+hi)/2
    factor=min(410/max(hi[0]-lo[0],.001),200/max(hi[1]-lo[1],.001))
    ids=np.linspace(0,len(x)-1,min(1000,len(x)),dtype=int)
    s=f'<figure><h3>{title}</h3><svg viewBox="0 0 450 240">'
    for points,color in [(x,'#9ca3af'),(t,'#168552'),(p,'#d65a2c')]:
        uv=(points[ids][:,axes]-center)*factor
        s+=''.join(f'<circle cx="{225+a:.3f}" cy="{120-b:.3f}" r="1.1" fill="{color}" fill-opacity=".6"/>' for a,b in uv)
    return s+'</svg></figure>'


def write_report(root):
    root=Path(root);d=json.loads((root/'results.json').read_text());config=json.loads((root/'config.json').read_text())
    figs=root/'figures';figs.mkdir(exist_ok=True)
    arrays={(m,l):dict(np.load(root/m/f'{l}_predictions.npz')) for m in ('cd','mse') for l in ('best','last')}
    all_points=np.concatenate([v[k].reshape(-1,3) for v in arrays.values() for k in ('initial_xyz','predicted_xyz','target_xyz')])
    bounds=(all_points.min(0),all_points.max(0))
    table='';panels='';selected={};gradient=''
    for mode,m in d.items():
        best=next(x for x in m['evaluations'] if x['update']==m['best_update']);last=m['evaluations'][-1]
        selected[mode]=dict(best=best,last=last,best_update=m['best_update'],probes=m['probes'])
        for label,r in [('best',best),('last',last)]:
            for b,name in enumerate('AB'):
                table+=f'<tr><td>{mode}</td><td>{label}/{r["update"]}</td><td>{name}</td><td>{r["branch_mean_cd_mm"][b]:.4f}</td><td>{r["branch_rmse_mm"][b]:.4f}</td><td>{r["affected_signed_z_mm"][b][-1]:+.3f}</td><td>{r["predicted_separation_mm"][-1]:.3f}</td></tr>'
        first=m['updates'][0]
        gradient+=f'<tr><td>{mode}</td><td>{first["raw_loss"]:.6g}</td><td>{first["gradient_norm"]:.6g}</td><td>{first["actual_updates"]["motion"]["update_norm"]:.6g}</td><td>{sum(x["clipped"] for x in m["updates"])}</td></tr>'
        for label in ('best','last'):
            z=arrays[mode,label]
            for b,name in enumerate('AB'):
                attr=z['initial_attr'][b];n=len(attr)
                a=dict(initial_xyz=z['initial_xyz'][b],initial_attributes=attr,frame_ids=z['frame_ids'][b],predicted_xyz=z['predicted_xyz'][b],
                    predicted_log_scales=np.broadcast_to(attr[None,:,:3],(10,n,3)),predicted_quaternions=np.broadcast_to(attr[None,:,3:7],(10,n,4)))
                for h in range(10):a[f'target_xyz_{h+1}']=z['target_xyz'][b,h];a[f'target_attr_{h+1}']=attr
                page=geometry_html(geometry_data(a),f'{mode}/{label}/{name}')
                page=page.replace('<a href="index.html">RGB / Alpha 多相机渲染</a>','<a href="../report.html">返回Loss对照</a>')
                page=page.replace('未来 GT 是独立点集，没有跨帧粒子 ID；仅预测模板可连成轨迹。','本实验为同一模板生成的合成GT，有已知逐点对应；正式真实GS没有此对应。')
                (figs/f'{mode}_{label}_{name}.html').write_text(page,encoding='utf-8')
    for label in ('best','last'):
        for b,name in enumerate('AB'):
            for h in (0,4,9):
                for axes,plane in [((0,2),'XZ'),((1,2),'YZ')]:
                    for mode in ('cd','mse'):
                        z=arrays[mode,label]
                        panels+=projection(z['initial_xyz'][b],z['target_xyz'][b,h],z['predicted_xyz'][b,h],f'{mode}/{label}/{name}/h{h+1}/{plane}',axes,bounds)
    charts='';colors=['#1261ad','#71a1d3','#c64919','#e6a13a']
    specs=[('分支平均CD / mm',lambda r,b:r['branch_mean_cd_mm'][b]),
           ('分支对应点RMSE / mm',lambda r,b:r['branch_rmse_mm'][b]),
           ('H10近端有符号Z / mm',lambda r,b:r['affected_signed_z_mm'][b][-1])]
    for title,fn in specs:
        series=[]
        for j,mode in enumerate(('cd','mse')):
            for b,name in enumerate('AB'):series.append((mode+'/'+name,[(r['update'],fn(r,b)) for r in d[mode]['evaluations']],colors[2*j+b]))
        charts+=chart(title,series)
    charts+=chart('全点A/B分离RMS / mm',[(mode,[(r['update'],r['predicted_separation_mm'][-1]) for r in d[mode]['evaluations']],color) for mode,color in zip(('cd','mse'),(colors[0],colors[2]))],d['cd']['evaluations'][0]['target_separation_mm'][-1])
    for mode,color in zip(('cd','mse'),(colors[0],colors[2])):
        charts+=chart(f'{mode}训练Loss（原始量纲；系数1）',[(mode,[(r['update'],r['training_loss']) for r in d[mode]['evaluations']],color)])
    static=''
    for mode,m in d.items():
        for label,q in m['probes'].items():
            r=q['conditions']['static']
            static+=f'<tr><td>{mode}/{label}/{q["update"]}</td>'+''.join(f'<td>{r["affected_signed_z_mm"][0][h]:+.3f}</td>' for h in (0,4,9))+f'<td>{r["motion_rms_mm"][0][-1]:.3f}</td></tr>'
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>V4 Action Identifiability V2 · CD vs MSE</title>
<style>body{{font:15px/1.6 system-ui;margin:24px;background:#f4f7fa;color:#193249}}table{{border-collapse:collapse;background:white;width:100%}}th,td{{padding:8px;border:1px solid #ccd7e2}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}figure{{margin:0;background:white;padding:12px}}svg{{width:100%}}iframe{{width:100%;height:750px;border:1px solid #bbb}}code{{overflow-wrap:anywhere}}</style>
<h1>V4 · Paired Action Identifiability V2：CD vs Correspondence MSE</h1>
<p>相同V1冻结数据与seed42初始化；每步A/B同批；各400更新；H10/Δ1。仅训练Loss变化，MSE系数固定1，不校准或搜索。目标近端Z为±9.514mm，全点分离RMS为6.704mm。h1共享条件/目标。</p>
<p>CD为非平方欧氏集合距离；MSE为归一化XYZ欧氏误差平方和的有效点/时间/分支平均。RMSE是对应欧氏误差的RMS，不是三坐标分量平均的RMSE。两种训练Loss不可直接数值比较。</p>
<table><tr><th>模型</th><th>Checkpoint/update</th><th>分支</th><th>CD/mm</th><th>对应RMSE/mm</th><th>H10近端Z/mm</th><th>全点分离/mm</th></tr>{table}</table>
<div class="grid">{charts}</div><h2>静止Action偏置</h2><table><tr><th>模型/Checkpoint</th><th>h1近端Z/mm</th><th>h5</th><th>h10</th><th>h10全点运动RMS/mm</th></tr>{static}</table>
<h2>梯度尺度与实际更新</h2><table><tr><th>组</th><th>初始原始Loss</th><th>初始全参数梯度范数</th><th>首步Motion实际更新范数</th><th>触发clip次数/400</th></tr>{gradient}</table>
<p>保留原clip=1及AdamW配置。零初始化末层导致初始上游梯度为零；完整JSON包含随后各模块原始/裁剪后梯度与实际更新。Null权重是网络关联权重，不是真实接触概率。</p>
<h2>全部10帧几何：左CD，右MSE</h2><label>Checkpoint <select id="checkpoint"><option>best</option><option>last</option></select></label> <label>分支<select id="branch"><option>A</option><option>B</option></select></label><div class="grid"><iframe id="cd"></iframe><iframe id="mse"></iframe></div>
<script>function refresh(){{for(let m of ['cd','mse'])document.getElementById(m).src=`figures/${{m}}_${{document.getElementById('checkpoint').value}}_${{document.getElementById('branch').value}}.html`;}}document.getElementById('checkpoint').onchange=refresh;document.getElementById('branch').onchange=refresh;refresh();</script>
<h2>统一世界坐标投影：H1/H5/H10</h2><p>所有静态面板使用相同投影、全局坐标范围与尺度；左CD，右MSE。灰=初态，绿=GT，橙=预测。交互三维页可独立旋转，静态投影用于严格同视角对照。</p><div class="grid">{panels}</div>
<h2>证据</h2><p><a href="results.json">全部数值、每头Attention/运动与梯度</a> · <a href="data_identity.json">V1数据一致性及哈希</a> · <a href="cd_reproduction.json">CD复现检查</a> · <a href="config.json">冻结配置</a> · <a href="smoke/acceptance.json">CUDA/恢复验收</a> · <a href="cd/training.csv">CD逐次更新</a> · <a href="mse/training.csv">MSE逐次更新</a></p>
<p>源码：<code>{config['source_sha']}</code>；Job {config['job_id']}。本实验仅对合成目标使用对应MSE，不适用于独立重建的真实GS。优化尺度不同，单次对照不能证明CD是唯一根因。</p></html>'''
    (root/'report.html').write_text(body,encoding='utf-8')
    (root/'summary.json').write_text(json.dumps(selected,indent=2,allow_nan=False),encoding='utf-8')
