"""Assemble completed experiments into an offline report, without inference."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

PAGE='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Condition Identifiability V1</title>
<style>body{font:15px/1.6 system-ui;margin:24px;background:#f3f5f8;color:#234}section{background:white;padding:20px;margin:18px 0;border-radius:10px}table{border-collapse:collapse;font-size:13px}th,td{padding:7px;border-bottom:1px solid #ddd;white-space:nowrap}.scroll{overflow:auto}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}img{width:100%}select,button,input{margin:8px;padding:6px}canvas{width:100%;height:240px;border:1px solid #ddd}pre{white-space:pre-wrap;max-height:400px;overflow:auto}.note{background:#fff4d9;padding:14px}</style>
<h1>Touch4D · Condition Identifiability V1</h1><p class="note">相同GS与时间的合成实验检验Action可学习性；真实双窗口初态并非完全相同，不能把记忆窗口等同条件动力学。两实验只报告固定400更新，无长期训练。</p>
<section><h2>结论与边界</h2><pre id="conclusion"></pre></section>
<section><h2>400更新：正常条件、复制初态与干预</h2><div class="scroll" id="summary"></div><p>干预ΔCD=干预−真实条件，正值才表示真实条件在该对照中更有帮助。输出变化不等同误差改善。</p></section>
<section><h2>五帧、三相机</h2><select id="record"></select><input id="h" type="range" min="1" max="5" value="5"><button id="play">播放</button><select id="cam"><option>0</option><option>1</option><option>2</option></select><select id="mod"><option>rgb</option><option>alpha</option></select><p id="label"></p><div class="grid"><div>GT<img id="gt"></div><div>Pred<img id="pred"></div><div>Copy<img id="copy"></div></div><div class="scroll" id="steps"></div><div class="grid"><div>GT centers<canvas id="cg"></canvas></div><div>Pred centers<canvas id="cp"></canvas></div><div>Copy centers<canvas id="cc"></canvas></div></div><p>中心为世界XZ投影、最多600点显示；GT每帧独立点集，不画对应粒子轨迹。完整几何见各图像目录geometry.npz。</p></section>
<section><h2>训练曲线</h2><div class="grid"><div>Synthetic Full Loss<canvas id="ls"></canvas></div><div>Real Full Loss<canvas id="lr"></canvas></div><div>正常条件检查点CD<canvas id="cd"></canvas></div></div></section>
<section><h2>CPU窗口审计、数据来源与运行状态</h2><pre id="audit"></pre><pre id="status"></pre><p><a href="results.json">完整结果 JSON</a> · <a href="metrics.csv">完整逐步CSV</a></p></section>
<script>const D=__DATA__,el=id=>document.getElementById(id),avg=a=>a.reduce((s,v)=>s+v,0)/a.length,fmt=x=>x==null?'—':Number(x).toPrecision(5),escape=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));function table(head,rows){return '<table><tr>'+head.map(s=>'<th>'+escape(s)+'</th>').join('')+'</tr>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+escape(v)+'</td>').join('')+'</tr>').join('')+'</table>'}
el('conclusion').textContent=D.conclusion?JSON.stringify(D.conclusion,null,2):'实验完成；结论待集中验收。';let rows=[];for(const [mode,r] of Object.entries(D.runs)){r.evaluations['400'].cases.forEach((c,i)=>{for(const [key,m] of Object.entries(c)){if(!m.cd_mm)continue;rows.push([mode,i,key,fmt(m.mean_cd_mm),fmt(m.mean_rgb),fmt(m.mean_alpha),fmt(m.delta_cd_mm?avg(m.delta_cd_mm):null),fmt(m.output_difference_rms_mm?.[4]),fmt(m.motion?.[4]?.displacement_rms_mm)])}})}el('summary').innerHTML=table(['实验','案例','条件','CD mm','RGB','Alpha','ΔCD mm','h5干预输出差RMS','h5运动RMS'],rows);
D.visuals.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.name;el('record').appendChild(o)});el('record').value=D.visuals.length-1;
function centers(r,k){let all=[r.geometry.initial,...r.geometry.gt,...r.geometry.pred].flat(),xmin=Infinity,xmax=-Infinity,zmin=Infinity,zmax=-Infinity;all.forEach(p=>{xmin=Math.min(xmin,p[0]);xmax=Math.max(xmax,p[0]);zmin=Math.min(zmin,p[2]);zmax=Math.max(zmax,p[2])});let span=Math.max(xmax-xmin,zmax-zmin,1)*1.1;for(let [id,pts,color] of [['cg',r.geometry.gt[k],'#398765'],['cp',r.geometry.pred[k],'#286d9e'],['cc',r.geometry.initial,'#888']]){let c=el(id),ctx=c.getContext('2d');c.width=c.clientWidth;c.height=240;ctx.fillStyle=color;pts.forEach(p=>ctx.fillRect(c.width/2+(p[0]-(xmin+xmax)/2)/span*c.width,120-(p[2]-(zmin+zmax)/2)/span*c.width,2,2))}}
function show(){let r=D.visuals[+el('record').value],h=+el('h').value;el('label').textContent=r.name+' · h'+h+' · frames '+r.frames.join(',');for(let a of ['gt','pred','copy'])el(a).src=r.folder+'/h'+String(h).padStart(2,'0')+'-c'+el('cam').value+'-'+el('mod').value+'-'+a+'.png';let m=r.metrics;el('steps').innerHTML=table(['h','CD mm','RGB','Alpha','位移RMS mm','方向Cos'],m.cd_mm.map((v,k)=>[k+1,fmt(v),fmt(m.rgb[k]),fmt(m.alpha[k]),fmt(m.motion[k].displacement_rms_mm),fmt(m.motion[k].direction_cosine)]));centers(r,h-1)}
function line(id,series){let c=el(id),ctx=c.getContext('2d');c.width=c.clientWidth;c.height=240;let all=series.flatMap(s=>s.values),max=Math.max(...all,1e-9);series.forEach(s=>{ctx.strokeStyle=s.color;ctx.beginPath();s.values.forEach((v,i)=>{let x=35+(c.width-45)*(s.x?s.x[i]/400:i/(s.values.length-1)),y=210-v/max*180;if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke()});ctx.fillStyle='#345';ctx.fillText('0',25,230);ctx.fillText('400 updates',c.width-90,230);ctx.fillText(max.toPrecision(3),2,20)}
line('ls',[{color:'#286d9e',values:D.runs.synthetic.updates.map(v=>v.loss)}]);line('lr',[{color:'#c46634',values:D.runs.real.updates.map(v=>v.loss)}]);line('cd',Object.entries(D.runs).map(([name,r])=>({color:name==='real'?'#c46634':'#286d9e',x:[0,100,200,400],values:['0','100','200','400'].map(k=>avg(r.evaluations[k].cases.map(c=>c.normal.mean_cd_mm)))})));
el('audit').textContent=JSON.stringify({windows:D.audit.windows,chosen:D.audit.chosen,target_pair_cd_mm:D.audit.target_pair_cd_mm,limitation:D.audit.limitation},null,2);el('status').textContent=JSON.stringify(Object.fromEntries(Object.entries(D.runs).map(([k,v])=>[k,v.status])),null,2);['record','cam','mod'].forEach(k=>el(k).onchange=show);el('h').oninput=show;let timer;el('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;el('play').textContent='播放'}else{timer=setInterval(()=>{el('h').value=(+el('h').value%5)+1;show()},700);el('play').textContent='暂停'}};show();</script></html>'''

def main(root):
    root=Path(root);read=lambda p:json.loads((root/p).read_text(encoding='utf-8'))
    runs={mode:{k:read(f'{mode}/{k}.json') for k in ('config','status','evaluations','updates','gradients')} for mode in ('synthetic','real')}
    if any(r['status']['status']!='complete' or len(r['updates'])!=400 for r in runs.values()):raise ValueError('Incomplete experiment')
    if runs['synthetic']['config']['initial_state_sha256']!=runs['real']['config']['initial_state_sha256']:raise ValueError('Initial parameters differ')
    visuals=[];rows=[]
    for mode,r in runs.items():
        for step,e in r['evaluations'].items():
            for case,c in enumerate(e['cases']):
                for condition,m in c.items():
                    if not isinstance(m,dict) or 'cd_mm' not in m:continue
                    for h in range(5):rows.append(dict(experiment=mode,step=int(step),case=case,condition=condition,horizon=h+1,cd_mm=m['cd_mm'][h],rgb=m['rgb'][h],alpha=m['alpha'][h],delta_cd_mm=m.get('delta_cd_mm',[None]*5)[h],output_difference_rms_mm=m.get('output_difference_rms_mm',[None]*5)[h]))
                if e['evidence']:
                    info=e['evidence'][case];folder=root/mode/info['folder']
                    for h in range(1,6):
                        for cam in range(3):
                            for typ in ('rgb','alpha'):
                                for label in ('gt','pred','copy'):
                                    if not (folder/f'h{h:02d}-c{cam}-{typ}-{label}.png').is_file():raise FileNotFoundError(folder)
                    with np.load(folder/'geometry.npz') as a:
                        sample=lambda p:(p[np.linspace(0,len(p)-1,min(600,len(p)),dtype=int)]*1000).tolist()
                        geo=dict(initial=sample(a['initial_xyz_m']),pred=[sample(p) for p in a['prediction_xyz_m']],gt=[sample(a[f'gt_h{h}_xyz_m']) for h in range(1,6)])
                    visuals.append(dict(name=f'{mode} case{case} update{step}',folder=f'{mode}/{info["folder"]}',frames=info['frames'],metrics=c['normal'],geometry=geo))
    result=dict(audit=read('audit.json'),runs=runs,visuals=visuals,conclusion=read('conclusion.json') if (root/'conclusion.json').exists() else None)
    for name in ('report.html','results.json','metrics.csv'):
        if (root/name).exists():raise FileExistsError(root/name)
    (root/'report.html').write_text(PAGE.replace('__DATA__',json.dumps(result,ensure_ascii=False,allow_nan=False).replace('</','<\\/')),encoding='utf-8')
    (root/'results.json').write_text(json.dumps(dict(result,visuals=[{k:v for k,v in r.items() if k!='geometry'} for r in visuals]),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (root/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(json.dumps(dict(records=len(visuals),csv_rows=len(rows))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
