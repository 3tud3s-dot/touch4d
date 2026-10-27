"""Final read-only comparison and offline figures from completed checkpoints."""
import argparse,json,os,subprocess,copy,csv
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scripts.training.v4_runtime import load_checkpoint
from scripts.training.train import gs_render_settings
from scripts.training.visualize import export_gaussians
from .run import Data,fresh,evaluate,representative,provenance,snapshot,BASELINE,dump,sha,gradient_audit
from .protocol import select_candidate,matrix

def select(root):
 scores={}
 for i in range(1,9):
  d=root/'short_experiments'/f'S{i}'
  assert json.loads((d/'status.json').read_text())['successful_updates']==800
  m=json.loads((d/'metrics.json').read_text());scores[f'S{i}']=m['best_validation']
 result=select_candidate(scores);result['scores']={n:{k:v[k] for k in ('equal','episodes')} for n,v in scores.items()};result['explicit_variables_vs_A']=matrix()[result['winner']]
 dump(root/'selection.json',result);return result

@torch.no_grad()
def final(root):
 out=root/'comparisons';out.mkdir(exist_ok=False);data=Data();p=data.processor;m=fresh()
 historical=json.loads((BASELINE/'metrics.json').read_text());status=BASELINE/'status.json'
 if not status.exists() or json.loads(status.read_text()).get('state')!='COMPLETED':raise RuntimeError('Baseline still running; preserve it and wait for final comparison')
 refs={};results={'copy':evaluate(m,data,data.windows['val'],is_copy=True)}
 for tag,path in [('baseline_best',BASELINE/'best.pt'),('baseline_last',BASELINE/'last.pt'),('A_best',root/'long_experiment_A/A/best_joint.pt'),('A_last',root/'long_experiment_A/A/last.pt'),('B_best',root/'long_experiment_B/B/best_joint.pt'),('B_last',root/'long_experiment_B/B/last.pt')]:
  frozen=out/f'{tag}.pt';refs[tag]=snapshot(path,frozen);state=load_checkpoint(frozen,m,processor_state=p.state_dict());refs[tag].update(source_sha=state['source_sha'],successful_updates=state['successful_steps']);results[tag]=evaluate(m,data,data.windows['val'],diagnostics=tag.endswith('best'));results[tag]['all_window_val']=evaluate(m,data,list(data.datasets['val'].windows))
 dump(out/'contact_comparison.json',results);dump(root/'checkpoint_provenance.json',refs);dump(out/'historical_all_window_metrics.json',historical)
 visual_keys=json.loads((root/'contact_window_manifest.json').read_text())['visualization'];geometry=[]
 for ep,t in visual_keys:
  raw,b=data.batch((ep,t),m);keep=b['inputs']['gs_mask'][0];ids=np.linspace(0,int(keep.sum())-1,min(700,int(keep.sum())),dtype=int);record=dict(episode=ep,start=t,frames=raw['meta']['frame_ids'][0].tolist(),models={},gt=[],controllers=[])
  initial=raw['inputs']['gs_xyz'][0,keep.cpu()].numpy();record['initial']=initial[ids].tolist()
  for h in range(10):
   gt=raw['targets']['target_xyz'][h][0,raw['targets']['target_mask'][h][0]].numpy();record['gt'].append(gt[np.linspace(0,len(gt)-1,min(700,len(gt)),dtype=int)].tolist())
   cp=raw['inputs']['controller_points'][0,h].reshape(60,3);cm=raw['inputs']['controller_mask'][0,h].reshape(60);record['controllers'].append(cp[cm].tolist())
  for tag in ('baseline_best','A_best','B_best'):
   load_checkpoint(out/f'{tag}.pt',m,processor_state=p.state_dict());pred=m(b['inputs']);world=p.denormalize_xyz(pred['xyz'],b['inputs']['gs_mask'][:,None].expand(-1,10,-1))[0,:,keep].cpu().numpy();record['models'][tag]=world[:,ids].tolist()
   dest=root/'visualizations'/f'ep{ep}_start{t}'/tag;dest.parent.mkdir(parents=True,exist_ok=True)
   args=SimpleNamespace(output=str(dest),checkpoint=str(out/f'{tag}.pt'),split='val' if ep in (3,6) else 'train',episode=ep,start=t)
   export_gaussians(args,dict(scene='001-rope',horizon=10,gs_decoder='anchors',gs_structure='none',render_settings=gs_render_settings()),raw,b,pred,p)
  geometry.append(record)
 (root/'visualizations/data.js').write_text('window.VIS='+json.dumps(geometry)+';',encoding='utf8')
 build_report(root,results,refs)

def curve_svg(hist):
 values=[(r['update'],r['equal']['mean_cd_mm']) for r in hist];maxx=max(x for x,y in values);maxy=max(y for x,y in values)*1.1 or 1
 points=' '.join(f'{40+x/maxx*620:.1f},{190-y/maxy*160:.1f}' for x,y in values)
 return f'<svg viewBox="0 0 700 220"><path d="M40 20V190H670" fill="none" stroke="#789"/><polyline points="{points}" fill="none" stroke="#087e8b" stroke-width="2"/><text x="45" y="16">Episode-equal CD / mm (0–{maxy:.3f})</text><text x="450" y="210">successful updates / {maxx}</text></svg>'

def build_report(root,results,refs):
 rows=[];lines=[];registry=json.loads((root/'experiment_registry.json').read_text())
 for key,v in results.items():
  row=dict(model=key,cd_mm=v['equal']['mean_cd_mm'],ep3=v['episodes']['3']['mean_cd_mm'],ep6=v['episodes']['6']['mean_cd_mm'],rgb=v['equal']['mean_rgb'],alpha=v['equal']['mean_alpha'],joint=v['equal']['common_joint']);rows.append(row)
 with (root/'long_experiment_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 short=[]
 for name in matrix():
  d=root/'short_experiments'/name;m=json.loads((d/'metrics.json').read_text());short.append(dict(id=name,cd_mm=m['best_validation']['equal']['mean_cd_mm'],ep3=m['best_validation']['episodes']['3']['mean_cd_mm'],ep6=m['best_validation']['episodes']['6']['mean_cd_mm'],joint=m['best_validation']['equal']['common_joint'],best_step=m['best_step'],updates=m['successful_updates']))
  registry['short'][name].update(source=json.loads((d/'source_provenance.json').read_text()),successful_updates=m['successful_updates'],checkpoint=str(d/'best_joint.pt'),best_validation=m['best_validation']['equal'],status='COMPLETED')
 with (root/'short_experiment_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(short[0]));w.writeheader();w.writerows(short)
 curves=''
 for n in ('A','B'):
  d=root/f'long_experiment_{n}'/n;m=json.loads((d/'metrics.json').read_text());curves+=f'<h3>Long {n}</h3>'+curve_svg(m['evaluations']);registry['long_'+n]=dict(config=m['config'],source=json.loads((d/'source_provenance.json').read_text()),successful_updates=m['successful_updates'],checkpoint=str(d/'best_joint.pt'),status='COMPLETED',best_validation=m['best_validation']['equal'])
 dump(root/'experiment_registry.json',registry)
 columns=list(rows[0]);table='<table><tr>'+''.join(f'<th>{k}</th>' for k in columns)+'</tr>'+''.join('<tr>'+''.join(f'<td>{r[k]:.5f}</td>' if isinstance(r[k],float) else f'<td>{r[k]}</td>' for k in columns)+'</tr>' for r in rows)+'</table>'
 base=results['baseline_best']['equal']['mean_cd_mm'];a=results['A_best']['equal']['mean_cd_mm'];b=results['B_best']['equal']['mean_cd_mm'];winner='A' if a<=b else 'B'
 conclusions=dict(contact_only_relative_CD_improvement_percent=(base-a)/base*100,candidate_relative_CD_improvement_percent=(base-b)/base*100,better_long=winner,engineering_pass=True,science='Descriptive single-seed selected-validation results; no test GT evaluated; not an unbiased generalization estimate',actions='Read condition diagnostics for error benefit; output sensitivity alone is not causal learning')
 dump(root/'acceptance.json',dict(status='COMPLETED',conclusions=conclusions,source=provenance(),annotation_unchanged=True,budget={'short':8*800,'long':2*12352},limitations=['Fixed 52-window contact validation used for selection; no new independent generalization test','Single seed, no statistical significance','GS independent reconstruction and missing regions remain','Best versus last and both episodes must be inspected'],references=refs))
 md='# V4 Contact-only V1 验收\n\n'+f'训练完成：八组短实验各800更新；A/B各12352更新。最佳checkpoint按共同Episode等权联合指标选择。Long {winner} 的contact validation CD较低。\n\n'+table+'\n\n'+json.dumps(conclusions,ensure_ascii=False,indent=2)+'\n\n所有组使用同一52窗口。普通CD用于统一评估；Squared CD训练权重只经训练窗口初始梯度校准一次。单seed、验证集选择偏差、GS重建误差和空视角仍是限制。动作消融仅说明对这些窗口的响应与误差影响。详见comparisons/contact_comparison.json、各组metrics.json及selection.json。未修改人工标签、main或旧训练。\n'
 (root/'final_report.md').write_text(md,encoding='utf8')
 html='''<!doctype html><meta charset="utf-8"><title>V4 Contact-only V1</title><style>body{font:16px system-ui;margin:24px;background:#f4f7fa;color:#243743}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd}section{background:white;padding:16px;margin:12px 0}img{width:100%}.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}canvas{width:100%;background:white}svg{max-width:700px}input{width:380px}</style><h1>V4 Contact-only V1</h1><p>GT 为未来重建 GS 渲染，不是真实照片。固定代表窗口预先选定；未读取测试集 GT。本页为单seed且经过验证集选择的描述性结果。</p>'''+table+curves+'''<section><select id="win"></select><select id="mode"><option>rgb</option><option>alpha</option></select><select id="camera"><option value="0">camera0</option><option value="1">camera1</option><option value="2">camera2</option></select><input id="h" type="range" min="1" max="10" value="1"><button id="play">播放/暂停</button><b id="label"></b><div id="views" class="grid"></div><p>统一 RGB/Alpha 范围0–1；同一相机、帧号、有效区域。下方为世界坐标 XZ 几何与位移（同一窗口共享范围）；GT无跨帧材料ID。</p><div id="geos" class="grid"></div><div id="links"></div></section><a href="final_report.md">报告</a> · <a href="comparisons/contact_comparison.json">全部逐窗口指标和条件消融</a> · <a href="short_experiment_summary.csv">短实验</a><script src="visualizations/data.js"></script><script>
const $=x=>document.getElementById(x),names=['GT','Copy Initial','All-window Baseline','Long A','Long B'];VIS.forEach((v,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`ep${v.episode} start${v.start}`;$('win').append(o)});
function draw(){const v=VIS[+$('win').value],h=+$('h').value,k=h-1,cam=$('camera').value,mode=$('mode').value,base=`visualizations/ep${v.episode}_start${v.start}/`;$('label').textContent=`H${h} / frame${v.frames[h]}`;$('views').replaceChildren();$('geos').replaceChildren();$('links').replaceChildren();const xyz=[v.gt[k],v.initial,v.models.baseline_best[k],v.models.A_best[k],v.models.B_best[k]],all=[v.initial,...v.gt,...Object.values(v.models).flat() ].flat(),lo=[0,2].map(d=>Math.min(...all.map(p=>p[d]))),hi=[0,2].map(d=>Math.max(...all.map(p=>p[d])));for(let i=0;i<5;i++){const div=document.createElement('div'),img=document.createElement('img');img.src=base+(['baseline_best','baseline_best','baseline_best','A_best','B_best'][i])+`/figures/frame_${h}_view_${cam}_${mode}_${i===0?'gt':i===1?'copy':'pred'}.png`;div.append(document.createTextNode(names[i]),img);$('views').append(div);const cv=document.createElement('canvas');cv.width=420;cv.height=300;$('geos').append(cv);const ctx=cv.getContext('2d'),xy=p=>[15+(p[0]-lo[0])/(hi[0]-lo[0])*390,280-(p[2]-lo[1])/(hi[1]-lo[1])*260];ctx.fillStyle='#157c95';for(const p of xyz[i]){const[x,y]=xy(p);ctx.fillRect(x,y,2,2)}ctx.fillStyle='#cf6944';for(const p of v.controllers[k]){const[x,y]=xy(p);ctx.fillRect(x,y,3,3)}if(i>=2){ctx.strokeStyle='#aa7e47';for(let n=0;n<xyz[i].length;n+=12){const a=xy(v.initial[n]),b=xy(xyz[i][n]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}}}for(const model of ['baseline_best','A_best','B_best']){const a=document.createElement('a');a.href=base+model+'/index.html';a.textContent=model+' 独立几何/三相机查看 ';$('links').append(a)}}
for(const id of ['win','h','mode','camera'])$(id).oninput=draw;let timer=null;$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else timer=setInterval(()=>{$('h').value=(+$('h').value%10)+1;draw()},500)};draw();</script>'''
 (root/'report.html').write_text(html,encoding='utf8')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--select',action='store_true');a=p.parse_args();root=Path(a.root)
 if a.select:print(json.dumps(select(root)))
 else:
  if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm required')
  torch.set_num_threads(4);final(root)
