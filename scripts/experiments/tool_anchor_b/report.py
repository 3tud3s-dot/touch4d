"""Offline H15 paired report. No GPU/model inference and no CDN."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

PAGE='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tool-to-Anchor B · H15</title>
<style>body{font:15px/1.6 system-ui;background:#f3f5f8;color:#24364a;margin:24px}section{background:white;padding:20px;margin:18px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}img{width:100%}canvas{width:100%;height:250px;background:#fafbfc;border:1px solid #ccd5df;touch-action:none}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;white-space:nowrap}.scroll{overflow:auto}pre{white-space:pre-wrap;max-height:480px;overflow:auto}select,input,button{margin:6px;padding:6px}.note{background:#fff2d5;padding:12px}</style>
<h1>Tool-to-Anchor Spatial Cross-Attention B · H15</h1><p class="note">两窗、seed42、各400成功更新；Base=A0/T3+Hand Mask+原P2内容交互。只验收运动拟合非退化，不证明条件辨识或泛化。</p>
<section><h2>集中验收</h2><pre id="verdict"></pre><div id="summary" class="scroll"></div></section>
<section><h2>GT / Copy / Base / Candidate</h2><select id="case"><option value="0">ep1 初态85 → 86–100</option><option value="1">ep2 初态73 → 74–88</option></select><select id="checkpoint"><option value="best">各自最低CD best</option><option value="last_400">last_400</option></select><input id="h" type="range" min="1" max="15" value="15"><button id="play">播放</button><select id="camera"><option value="0">Camera 0</option><option value="1">Camera 1</option><option value="2">Camera 2</option></select><select id="type"><option value="rgb">RGB</option><option value="alpha">Alpha</option></select><p id="label"></p><div id="shortcuts"></div><div class="grid"><div>GT<img id="gt"></div><div>Copy Initial<img id="copy"></div><div>Base / S0<img id="base"></div><div>Candidate / B<img id="candidate"></div></div><div class="grid"><div>GT centers<canvas id="cgt"></canvas></div><div>Copy centers<canvas id="ccopy"></canvas></div><div>Base centers + tracks<canvas id="cbase"></canvas></div><div>B centers + tracks<canvas id="ccandidate"></canvas></div></div><p>拖动任一三维中心视图同步旋转，滚轮缩放。预测轨迹使用自身初态模板身份；GT逐帧独立重建，不画跨帧粒子对应。抽样仅用于显示，指标使用全部有效点。</p><div id="horizons" class="scroll"></div></section>
<section><h2>400更新过程</h2><canvas id="curve"></canvas><p>蓝色Base，橙色B；实线每20次两窗H15平均CD，0–400。</p><div id="cost" class="scroll"></div></section>
<section><h2>逐步原始数据、数据来源与版本</h2><p><a href="report.md">Markdown报告</a> · <a href="results.json">全部JSON</a> · <a href="metrics.csv">全部逐步CSV</a> · <a href="manifest.json">冻结窗口manifest</a></p><pre id="manifest"></pre></section>
<script>const D=__DATA__,el=id=>document.getElementById(id),f=v=>v==null?'—':Number(v).toPrecision(5),avg=a=>a.reduce((s,v)=>s+v,0)/a.length;
function table(head,rows){return '<table><tr>'+head.map(v=>'<th>'+v+'</th>').join('')+'</tr>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</table>'}
el('verdict').textContent=JSON.stringify(D.conclusion,null,2);let summary=[];for(let mode of ['base','candidate'])for(let label of ['best','last_400']){let ck=D.runs[mode].final[label];ck.cases.forEach((c,i)=>summary.push([mode,label,ck.update,'ep'+D.manifest.windows[i].episode,f(c.mean_cd_mm),f(c.cd_mm[0]),f(c.cd_mm[4]),f(c.cd_mm[9]),f(c.cd_mm[14]),f(c.mean_rgb),f(c.mean_alpha),f(100*(1-c.mean_cd_mm/D.copy[i].mean_cd_mm))+'%']))}el('summary').innerHTML=table(['模型','checkpoint','update','窗口','CD均值mm','h1','h5','h10','h15','RGB','Alpha','vs Copy'],summary);
let yaw=.25,pitch=.2,zoom=1,drag=null;function rotate(p){let [x,y,z]=p;let a=Math.cos(yaw)*x+Math.sin(yaw)*z,b=-Math.sin(yaw)*x+Math.cos(yaw)*z;return [a,Math.cos(pitch)*y-Math.sin(pitch)*b,Math.sin(pitch)*y+Math.cos(pitch)*b]}
function view(geo,h){let all=[...geo.initial,...geo.gt.flat(),...geo.base.flat(),...geo.candidate.flat()];let lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];all.forEach(p=>p.forEach((v,j)=>{lo[j]=Math.min(lo[j],v);hi[j]=Math.max(hi[j],v)}));let center=lo.map((v,j)=>(v+hi[j])/2),span=Math.max(...hi.map((v,j)=>v-lo[j]),1);for(let name of ['gt','copy','base','candidate']){let canvas=el('c'+name),ctx=canvas.getContext('2d');canvas.width=canvas.clientWidth;canvas.height=250;let project=p=>{let v=rotate(p.map((x,j)=>x-center[j]));return [canvas.width/2+v[0]/span*canvas.width*.9*zoom,125-v[1]/span*canvas.width*.9*zoom]};let pts=name==='copy'?geo.initial:geo[name][h];ctx.fillStyle=name==='candidate'?'#d07132':name==='base'?'#2673aa':name==='gt'?'#398464':'#999';pts.forEach(p=>{let [x,y]=project(p);ctx.fillRect(x,y,2,2)});if(name==='base'||name==='candidate'){ctx.strokeStyle='#9faebc';for(let i=0;i<pts.length;i+=Math.max(1,Math.floor(pts.length/12))){ctx.beginPath();[geo.initial,...geo[name].slice(0,h+1)].forEach((frame,k)=>{let [x,y]=project(frame[i]);if(k)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke()}}}}
function show(){let i=+el('case').value,h=+el('h').value,label=el('checkpoint').value,cam=el('camera').value,typ=el('type').value;let prefix='h'+String(h).padStart(2,'0')+'-c'+cam+'-'+typ+'-';el('gt').src='paired/base/best-case'+i+'/'+prefix+'gt.png';el('copy').src='paired/base/best-case'+i+'/'+prefix+'copy.png';for(let mode of ['base','candidate'])el(mode).src='paired/'+mode+'/'+label+'-case'+i+'/'+prefix+'pred.png';el('label').textContent='GT frame '+D.manifest.windows[i].frames[h]+' / h'+h+'，Base update '+D.runs.base.final[label].update+'，B update '+D.runs.candidate.final[label].update+'；相机 '+D.manifest.windows[i].camera_ids[0][+cam];let a=D.runs.base.final[label].cases[i],b=D.runs.candidate.final[label].cases[i];el('horizons').innerHTML=table(['h','Copy CD','Base CD','B CD','Base RMS','B RMS','Base去平移RMS','B去平移RMS','Base邻域P95','B邻域P95'],a.cd_mm.map((v,k)=>[k+1,f(D.copy[i].cd_mm[k]),f(v),f(b.cd_mm[k]),f(a.motion[k].rms_mm),f(b.motion[k].rms_mm),f(a.motion[k].residual_rms_mm),f(b.motion[k].residual_rms_mm),f(a.diagnostics[k].neighbor_p95_mm),f(b.diagnostics[k].neighbor_p95_mm)]));view(D.geometry[label][i],h-1)}
['case','checkpoint','camera','type'].forEach(k=>el(k).onchange=show);el('h').oninput=show;[1,5,10,15].forEach(h=>{let b=document.createElement('button');b.textContent='h'+h;b.onclick=()=>{el('h').value=h;show()};el('shortcuts').appendChild(b)});let timer;el('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;el('play').textContent='播放'}else{timer=setInterval(()=>{el('h').value=(+el('h').value%15)+1;show()},600);el('play').textContent='暂停'}};
for(let id of ['cgt','ccopy','cbase','ccandidate']){let c=el(id);c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=null;c.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.01;pitch+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];show()}};c.onwheel=e=>{e.preventDefault();zoom=Math.max(.2,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));show()}}
let canvas=el('curve'),ctx=canvas.getContext('2d');canvas.width=canvas.clientWidth;canvas.height=250;let maximum=Math.max(...Object.values(D.runs).flatMap(r=>Object.values(r.evaluations).map(v=>v.mean_cd_mm)));for(let mode of ['base','candidate']){ctx.strokeStyle=mode==='base'?'#2673aa':'#d07132';ctx.beginPath();Object.entries(D.runs[mode].evaluations).forEach(([step,r],i)=>{let x=40+(canvas.width-60)*(+step)/400,y=220-r.mean_cd_mm/maximum*190;if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke()}ctx.fillText('0 updates',5,245);ctx.fillText('400 updates',canvas.width-100,245);ctx.fillText(maximum.toPrecision(4)+' mm',5,20);
el('cost').innerHTML=table(['模型','参数量','峰值 allocated GiB','reserved GiB','训练+每20评估秒'],['base','candidate'].map(mode=>[mode,D.pairing[mode+'_parameters'],f(D.runs[mode].status.peak_allocated_bytes/2**30),f(D.runs[mode].status.peak_reserved_bytes/2**30),f(D.runs[mode].status.fit_and_evaluation_seconds)]));el('manifest').textContent=JSON.stringify({source_sha:D.manifest.source_sha,windows:D.manifest.windows.map(w=>({episode:w.episode,start:w.start,frames:w.frames,sigma_tool_mm:w.sigma_tool_mm})),pairing:D.pairing,status:D.status},null,2);show();</script></html>'''

def main(root):
    root=Path(root);read=lambda p:json.loads((root/p).read_text(encoding='utf-8'))
    runs={mode:{key:read(f'paired/{mode}/{key}.json') for key in ('config','status','evaluations','updates','gradients','final')} for mode in ('base','candidate')}
    for mode,r in runs.items():
        if r['status']['status']!='complete' or len(r['updates'])!=400:raise ValueError('Incomplete run')
        if list(map(int,r['evaluations']))!=list(range(0,401,20)):raise ValueError('Missing evaluation intervals')
    geometry={};rows=[]
    for label in ('best','last_400'):
        geometry[label]=[]
        for i in range(2):
            arrays={mode:np.load(root/f'paired/{mode}/{label}-case{i}/geometry.npz') for mode in runs}
            a=arrays['base'];ids=np.linspace(0,len(a['initial_xyz_m'])-1,min(700,len(a['initial_xyz_m'])),dtype=int)
            geo=dict(initial=(a['initial_xyz_m'][ids]*1000).tolist(),gt=[])
            for h in range(15):
                points=a[f'gt_h{h+1}_xyz_m'];idx=np.linspace(0,len(points)-1,min(700,len(points)),dtype=int)
                geo['gt'].append((points[idx]*1000).tolist())
            for mode,data in arrays.items():
                geo[mode]=(data['prediction_xyz_m'][:,ids]*1000).tolist();data.close()
            geometry[label].append(geo)
    for mode,r in runs.items():
        for step,e in r['evaluations'].items():
            for case,c in enumerate(e['cases']):
                for h in range(15):rows.append(dict(model=mode,update=int(step),case=case,horizon=h+1,
                    cd_mm=c['cd_mm'][h],rgb=c['rgb'][h],alpha=c['alpha'][h],
                    displacement_rms_mm=c['motion'][h]['rms_mm'],residual_rms_mm=c['motion'][h]['residual_rms_mm'],
                    neighbor_rms_mm=c['diagnostics'][h]['neighbor_rms_mm'],neighbor_p95_mm=c['diagnostics'][h]['neighbor_p95_mm']))
    data=dict(manifest=read('manifest.json'),runs=runs,copy=read('paired/copy.json'),pairing=read('paired/pairing.json'),
        status=read('paired/status.json'),geometry=geometry,conclusion=read('conclusion.json') if (root/'conclusion.json').exists() else {'status':'Awaiting centralized acceptance'})
    for name in ('report.html','results.json','metrics.csv'):
        if (root/name).exists():raise FileExistsError(root/name)
    (root/'report.html').write_text(PAGE.replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/')),encoding='utf-8')
    (root/'results.json').write_text(json.dumps({k:v for k,v in data.items() if k!='geometry'},ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (root/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(json.dumps(dict(rows=len(rows),windows=2,horizons=15,checkpoints=['best','last_400'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();main(args.root)
