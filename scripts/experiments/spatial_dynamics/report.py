"""CPU collection and offline report, keeping every window and horizon in JSON."""
import argparse,csv,html,json
from pathlib import Path
import numpy as np


def collect(mapping):
    results={};table=[]
    for name,path in mapping.items():
        root=Path(path);status=json.loads((root/'status.json').read_text());config=json.loads((root/'config.json').read_text())
        if status['status']!='complete':raise RuntimeError(f'Unfinished run: {name}')
        data=json.loads((root/'metrics.json').read_text());history=json.loads((root/'updates.json').read_text())
        n=status['resources']['successful_updates']
        if len(history)!=n or [r['update'] for r in history]!=list(range(1,n+1)):raise ValueError('Optimizer count mismatch')
        results[name]=dict(status=status,config=config,metrics=data,updates=history,path=str(root))
        s=data['model']['summary'];base=data['copy_initial']['summary']
        table.append(dict(experiment=name,code_sha=status['source_sha'],seed=config['seed'],updates=n,
            spatial=config['model']['spatial'],decoder=config['model']['decoder'],feedback=config['model']['feedback'],
            r1=config['model']['regularization'],horizon=config['model']['horizon'],job=config['slurm_job_id'],
            cd_mm=s['mean_cd_mm'],copy_cd_mm=base['mean_cd_mm'],cd_minus_copy_mm=s['mean_cd_mm']-base['mean_cd_mm'],
            relative_copy_percent=100*(s['mean_cd_mm']/base['mean_cd_mm']-1),rgb=float(np.mean(s['rgb'])),alpha=float(np.mean(s['alpha'])),
            joint=s['joint'],action_delta_cd_mm=float(np.mean(data['freeze_action']['delta_cd_mm'])),
            tactile_delta_cd_mm=float(np.mean(data['zero_tactile']['delta_cd_mm'])),
            gpu_peak_gib=status['resources']['allocated_peak_bytes']/2**30,seconds=status['resources']['elapsed_seconds'],
            checkpoint=str(root/'last.pt'),checkpoint_sha256=status['checkpoint_sha256']))
    return results,table


def render(results,table,conclusion,visuals):
    columns=['experiment','updates','cd_mm','copy_cd_mm','relative_copy_percent','rgb','alpha','action_delta_cd_mm','tactile_delta_cd_mm','gpu_peak_gib']
    headings=['实验','更新','CD mm','复制初态 mm','相对复制 %','RGB','Alpha','冻结动作 ΔCD','零触觉 ΔCD','峰值 GiB']
    rows=''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:.6f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in columns)+'</tr>' for r in table)
    encoded=json.dumps(results,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    vdata=json.dumps(visuals,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    return '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P3–P4 Spatial Dynamics</title>
<style>body{font:14px system-ui;color:#182635;background:#f4f7fa;margin:24px}h1{font-size:24px}section{background:white;padding:18px;border-radius:10px;margin:18px 0}table{border-collapse:collapse;font-size:12px;width:100%}td,th{padding:7px;border:1px solid #dce2e9;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}select,button,input{margin:6px;padding:5px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.grid img{width:100%}.note{background:#fff1cf;padding:10px}canvas{width:100%;height:220px}pre{white-space:pre-wrap}a{color:#1263a3}iframe{width:100%;height:760px;border:0}</style>
<h1>P3–P4：空间交互与形变动力学</h1><p>001-rope · A0/T3 · 固定24训练 / 8验证窗口 · 验证episode等权 · 独立点集，无GT粒子ID</p>
<section><h2>集中结论</h2><pre>'''+html.escape(conclusion)+'''</pre><p>工程通过、可优化性与科学收益分别判断。单seed筛选与两seed复验均不构成统计显著性证明；条件敏感也不等于条件有益。</p></section>
<section><h2>实验总表（同预算 last，不隐藏退化）</h2><div class="scroll"><table><thead><tr>'''+''.join('<th>'+t+'</th>' for t in headings)+'''</tr></thead><tbody>'''+rows+'''</tbody></table></div><p>ΔCD=扰动条件−真实条件；正值说明真实条件在该对照中更有帮助。<a href="results.json">完整JSON</a> · <a href="experiments.csv">CSV</a> · <a href="decision-log.json">阶段分流依据</a></p></section>
<section><h2>每窗口 / 每Horizon</h2><select id="run"></select><select id="window"></select><canvas id="chart"></canvas><div class="scroll" id="detail"></div><h3>训练 loss</h3><canvas id="loss"></canvas><pre id="provenance"></pre></section>
<section><h2>H15 目的性案例</h2><p class="note" id="warning"></p><select id="case"></select><select id="variant"></select><select id="camera"><option value="0">相机1</option><option value="1">相机2</option><option value="2">相机3</option></select><select id="view"><option value="rgb">RGB</option><option value="alpha">Alpha</option></select><button id="play">播放</button><input id="frame" type="range" min="0" max="15" value="1"><b id="frameLabel"></b><div class="grid"><div>GT<img id="gt"></div><div>S0<img id="baseline"></div><div>候选<img id="candidate"></div><div>复制初态<img id="copy"></div></div><p><a id="geometry" target="_blank">独立 centers / ellipsoids / 预测轨迹</a> · <a id="render" target="_blank">完整渲染与误差</a></p><pre id="visualEvidence"></pre></section>
<script>const D='''+encoded+''', V='''+vdata+''';const $=id=>document.getElementById(id);
function option(sel,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;sel.append(o)}
Object.keys(D).forEach(k=>option($('run'),k,k));
function line(id,series){const c=$(id);c.width=c.clientWidth*devicePixelRatio;c.height=220*devicePixelRatio;const ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);const all=series.flatMap(s=>s.v),max=Math.max(...all,1e-9),pad=35*devicePixelRatio;ctx.font=`${11*devicePixelRatio}px system-ui`;series.forEach((s,j)=>{ctx.strokeStyle=s.color;ctx.fillStyle=s.color;ctx.fillText(s.name,pad+j*160*devicePixelRatio,15*devicePixelRatio);ctx.beginPath();s.v.forEach((v,i)=>{const x=pad+i*(c.width-2*pad)/Math.max(1,s.v.length-1),y=c.height-pad-v/max*(c.height-2*pad);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()});ctx.fillStyle='#555';ctx.fillText('0',5,c.height-pad);ctx.fillText(max.toPrecision(4),5,pad)}
function choose(){const r=D[$('run').value];$('window').innerHTML='';r.metrics.model.windows.forEach((w,i)=>option($('window'),i,`ep${w.episode} start${w.start}`));detail()}
function detail(){const r=D[$('run').value],i=+$('window').value,w=r.metrics.model.windows[i],copy=r.metrics.copy_initial.windows[i];line('chart',[{name:'预测 CD(mm)',color:'#1d70b8',v:w.cd_mm},{name:'复制初态',color:'#888',v:copy.cd_mm}]);line('loss',[{name:'训练联合 loss',color:'#1d70b8',v:r.updates.map(u=>u.loss)}]);let t='<table><tr><th>h</th><th>CD mm</th><th>Copy mm</th><th>RGB</th><th>Alpha</th><th>去质心 CD</th><th>质心误差</th><th>邻距变化 mm</th></tr>';w.cd_mm.forEach((v,h)=>{let m=w.motion[h];t+='<tr>'+[h+1,v,copy.cd_mm[h],w.rgb[h],w.alpha[h],m.centered_cd_mm,m.centroid_error_mm,m.neighbor_distance_change_mm].map(x=>'<td>'+Number(x).toFixed(5)+'</td>').join('')+'</tr>'});$('detail').innerHTML=t+'</table>';$('provenance').textContent=JSON.stringify({source:r.status.source_sha,config:r.config.model,path:r.path,checkpoint_sha256:r.status.checkpoint_sha256,resources:r.status.resources},null,2)}
$('run').onchange=choose;$('window').onchange=detail;if(Object.keys(D).length)choose();
const cases=[...new Set(V.map(v=>`${v.episode}:${v.start}`))];cases.forEach(c=>option($('case'),c,c));const labels=[...new Set(V.map(v=>v.label))];labels.forEach(c=>option($('variant'),c,c));let timer=null;
function visual(){const [e,s]=$('case').value.split(':').map(Number),item=V.find(v=>v.episode===e&&v.start===s&&v.label===$('variant').value);if(!item)return;const base=V.find(v=>v.episode===e&&v.start===s&&v.label==='S0-last')||item,h=+$('frame').value,c=$('camera').value,m=$('view').value;function src(v,kind){return `visuals/${v.folder}/figures/frame_${h}_view_${c}_${m}_${kind}.png`}$('gt').src=src(item,'gt');$('candidate').src=src(item,'pred');$('baseline').src=src(base,'pred');$('copy').src=src(item,'copy');$('warning').textContent=item.warning||'正式H15训练权重：在线条件逐步到达，非初态预知未来触觉。';$('frameLabel').textContent=`h=${h} / frame ${item.frames[h]}`;$('geometry').href=`visuals/${item.folder}/geometry.html`;$('render').href=`visuals/${item.folder}/index.html`;$('visualEvidence').textContent=JSON.stringify({checkpoint:item.checkpoint,sha256:item.checkpoint_sha256,updates:item.updates,unique_states:item.unique_prediction_states,model:item.config},null,2)}
['case','variant','camera','view'].forEach(k=>$(k).onchange=visual);$('frame').oninput=visual;$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放'}else{timer=setInterval(()=>{$('frame').value=(+$('frame').value+1)%16;visual()},500);$('play').textContent='暂停'}};if(V.length)visual();window.onresize=detail;</script></html>'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--output',required=True);p.add_argument('--conclusion',required=True);p.add_argument('--visuals');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError(out/'results.json')
    results,table=collect(json.loads(Path(a.runs).read_text()))
    (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    with (out/'experiments.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    visuals=json.loads(Path(a.visuals).read_text()) if a.visuals else []
    (out/'report.html').write_text(render(results,table,Path(a.conclusion).read_text(encoding='utf-8'),visuals),encoding='utf-8')
if __name__=='__main__':main()
