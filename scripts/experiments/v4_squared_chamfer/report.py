"""Three-loss offline report with untouched historical controls."""
import json
from pathlib import Path
import shutil
import numpy as np
from scripts.experiments.v4_action_loss_comparison.report import chart,projection
from scripts.training.visualize import geometry_data,geometry_html


def write_report(root,previous):
    root,previous=Path(root),Path(previous)
    d=json.loads((root/'results.json').read_text());cfg=json.loads((root/'config.json').read_text())
    modes=('cd','squared_cd','mse');colors=('#2774b5','#ce5b25','#3d9261')
    for mode in ('cd','mse'):
        dest=root/'historical'/mode;dest.mkdir(parents=True,exist_ok=False)
        for name in ('metrics.json','training.csv','best_predictions.npz','last_predictions.npz'):
            shutil.copyfile(previous/mode/name,dest/name)
    figs=root/'figures';figs.mkdir()
    arrays={(m,l):dict(np.load((root/m if m=='squared_cd' else root/'historical'/m)/f'{l}_predictions.npz')) for m in modes for l in ('best','last')}
    all_xyz=np.concatenate([v[k].reshape(-1,3) for v in arrays.values() for k in ('initial_xyz','predicted_xyz','target_xyz')])
    bounds=(all_xyz.min(0),all_xyz.max(0));table='';grad='';static='';panels='';summaries={}
    for mode in modes:
        m=d[mode];best=next(r for r in m['evaluations'] if r['update']==m['best_update'])
        summaries[mode]=dict(best=best,last=m['evaluations'][-1],best_update=m['best_update'],probes=m['probes'],checkpoint_hashes=m['checkpoint_sha256'])
        first=m['updates'][0]
        grad+=f'<tr><td>{mode}</td><td>{first["raw_loss"]:.7g}</td><td>{first["gradient_norm"]:.7g}</td><td>{first["actual_updates"]["motion"]["update_norm"]:.7g}</td><td>{sum(r["clipped"] for r in m["updates"])}</td></tr>'
        for label,r in [('best',best),('last',m['evaluations'][-1])]:
            for b,name in enumerate('AB'):
                table+=f'<tr><td>{mode}/{label}/{r["update"]}</td><td>{name}</td><td>{r["branch_mean_cd_mm"][b]:.4f}</td><td>{r["branch_rmse_mm"][b]:.4f}</td><td>{r["affected_signed_z_mm"][b][-1]:+.4f}</td><td>{r["predicted_separation_mm"][-1]:.4f}</td></tr>'
            q=m['probes'][label]['conditions']['static']
            static+=f'<tr><td>{mode}/{label}</td>'+''.join(f'<td>{q["affected_signed_z_mm"][0][h]:+.4f}</td>' for h in (0,4,9))+f'<td>{q["motion_rms_mm"][0][-1]:.4f}</td></tr>'
            z=arrays[mode,label]
            for b,name in enumerate('AB'):
                attr=z['initial_attr'][b];n=len(attr)
                a=dict(initial_xyz=z['initial_xyz'][b],initial_attributes=attr,frame_ids=z['frame_ids'][b],predicted_xyz=z['predicted_xyz'][b],predicted_log_scales=np.broadcast_to(attr[None,:,:3],(10,n,3)),predicted_quaternions=np.broadcast_to(attr[None,:,3:7],(10,n,4)))
                for h in range(10):a[f'target_xyz_{h+1}']=z['target_xyz'][b,h];a[f'target_attr_{h+1}']=attr
                page=geometry_html(geometry_data(a),f'{mode}/{label}/{name}')
                page=page.replace('<a href="index.html">RGB / Alpha 多相机渲染</a>','<a href="../report.html">返回三组Loss对照</a>')
                page=page.replace('未来 GT 是独立点集，没有跨帧粒子 ID；仅预测模板可连成轨迹。','此合成GT具有已知对应；真实独立重建GS没有此对应。')
                (figs/f'{mode}_{label}_{name}.html').write_text(page,encoding='utf-8')
    for label in ('best','last'):
        for b,name in enumerate('AB'):
            for h in (0,4,9):
                for mode in modes:
                    z=arrays[mode,label]
                    panels+=projection(z['initial_xyz'][b],z['target_xyz'][b,h],z['predicted_xyz'][b,h],f'{mode}/{label}/{name}/h{h+1} XZ',(0,2),bounds)
    charts=''
    for title,fn in [('普通CD / mm',lambda r,b:r['branch_mean_cd_mm'][b]),('对应RMSE / mm',lambda r,b:r['branch_rmse_mm'][b]),('H10近端有符号Z / mm',lambda r,b:r['affected_signed_z_mm'][b][-1])]:
        for b,name in enumerate('AB'):
            charts+=chart(title+' / '+name,[(mode,[(r['update'],fn(r,b)) for r in d[mode]['evaluations']],color) for mode,color in zip(modes,colors)])
    charts+=chart('分支分离 RMS / mm',[(m,[(r['update'],r['predicted_separation_mm'][-1]) for r in d[m]['evaluations']],c) for m,c in zip(modes,colors)],d['cd']['evaluations'][0]['target_separation_mm'][-1])
    for mode,color in zip(modes,colors):charts+=chart(mode+' 原始训练Loss（不可跨量纲直接比较）',[(mode,[(r['update'],r['training_loss']) for r in d[mode]['evaluations']],color)])
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>V4 Squared Chamfer Control</title><style>body{font:15px/1.65 system-ui;margin:24px;background:#f4f7fa;color:#193249}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ccd7e2;padding:8px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}figure{background:white;margin:0;padding:8px}svg{width:100%}iframe{width:100%;height:720px}code{overflow-wrap:anywhere}</style><h1>V4：普通 CD / Squared CD / Correspondence MSE</h1>
<p>普通CD：最近邻、非平方；Squared CD：同一最近邻、逐匹配距离平方、双向0.5平均；MSE：已知对应、平方。仅新增 squared_cd 400次更新；CD/MSE为冻结V2结果，不重训。H10/Δ1、seed42、batch2、AdamW 1e-4/1e-4、clip1、Loss系数1。相同数据和初始模型哈希；真实GS不能按数组行号使用对应MSE。</p>
<p>目标近端 Z ±9.514mm，分离RMS 6.704mm。h1共享条件和目标。best按各自训练Loss选择，必须同时看last400；不能比较不同量纲的训练Loss大小来判断性能。</p>
<table><tr><th>模型/checkpoint/update</th><th>分支</th><th>普通CD/mm</th><th>对应RMSE/mm</th><th>H10近端Z/mm</th><th>分离RMS/mm</th></tr>'''+table+'''</table><div class="grid">'''+charts+'''</div>
<h2>静止动作</h2><table><tr><th>模型</th><th>h1 Z/mm</th><th>h5 Z/mm</th><th>h10 Z/mm</th><th>h10运动RMS/mm</th></tr>'''+static+'''</table>
<h2>初始梯度与实际更新</h2><table><tr><th>Loss</th><th>初始Loss</th><th>总梯度范数</th><th>Motion首步更新范数</th><th>裁剪次数/400</th></tr>'''+grad+'''</table><p>完整JSON记录每个模块原始梯度、裁剪后梯度、实际参数更新，以及每个Head的Null和带符号工具运动。不同梯度尺度、AdamW与裁剪仍是混杂因素，不将某一种机制视为唯一根因。Null不是接触概率。</p>
<h2>十帧几何并排</h2><select id="cp"><option>best</option><option>last</option></select><select id="branch"><option>A</option><option>B</option></select><div class="grid"><iframe id="cd"></iframe><iframe id="squared_cd"></iframe><iframe id="mse"></iframe></div><script>function refresh(){for(let m of ['cd','squared_cd','mse'])document.getElementById(m).src='figures/'+m+'_'+document.getElementById('cp').value+'_'+document.getElementById('branch').value+'.html';}document.getElementById('cp').onchange=refresh;document.getElementById('branch').onchange=refresh;refresh();</script>
<h2>H1/H5/H10 同视角同尺度投影</h2><p>左普通CD，中Squared CD，右MSE。灰=初态，绿=GT，橙=预测。所有面板采用相同世界坐标范围。</p><div class="grid">'''+panels+'''</div><h2>证据</h2><p><a href="results.json">三组完整数值</a> · <a href="summary.json">best/last及条件扰动</a> · <a href="data_identity.json">冻结数据身份</a> · <a href="historical_hashes.json">历史结果前后哈希</a> · <a href="config.json">配置</a> · <a href="smoke/acceptance.json">CUDA恢复验收</a> · <a href="squared_cd/training.csv">新实验400次记录</a></p><p>新实验源码 <code>'''+cfg['source_sha']+'''</code>；Slurm Job '''+cfg['job_id']+'''。历史源码 e32933f。checkpoint路径与哈希保存在metrics/summary；权重留服务器。本实验不证明真实物理或跨episode泛化。</p></html>'''
    (root/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False),encoding='utf-8')
    (root/'report.html').write_text(body,encoding='utf-8')
