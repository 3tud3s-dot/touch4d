"""Final read-only comparison and offline figures from completed checkpoints."""
import argparse,json,os,subprocess,copy,csv,html as html_lib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scripts.training.v4_runtime import load_checkpoint
from scripts.training.train import gs_render_settings
from scripts.training.visualize import export_gaussians
from .run import Data,fresh,evaluate,representative,provenance,snapshot,BASELINE,FROZEN,dump,sha,gradient_audit
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
 manifest=json.loads((root/'contact_window_manifest.json').read_text())
 if sha(FROZEN/'contact_boundaries.json')!=manifest['annotation_sha256']:raise RuntimeError('Frozen annotation changed')
 for name in matrix():
  state=json.loads((root/'short_experiments'/name/'status.json').read_text());assert state['status']=='COMPLETED' and state['successful_updates']==800
 for name in ('A','B'):
  state=json.loads((root/f'long_experiment_{name}'/name/'status.json').read_text());assert state['status']=='COMPLETED' and state['successful_updates']==12352
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
   cp=raw['inputs']['controller_points'][0,h].reshape(60,3);cm=raw['inputs']['controller_mask'][0,h].reshape(60)&raw['inputs']['operation_hands'][0].repeat_interleave(30);record['controllers'].append(cp[cm].tolist())
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

def multi_curve(series,title,xlabel):
 """Offline SVG with explicit units; only finite, nonempty original data."""
 if not series or any(not xy for xy in series.values()):raise ValueError('Empty curve')
 points=[p for seq in series.values() for p in seq]
 if not np.isfinite(np.asarray(points)).all():raise ValueError('Nonfinite curve')
 xlo,xhi=min(x for x,y in points),max(x for x,y in points);ylo=min(0.,min(y for x,y in points));yhi=max(y for x,y in points);dx=max(xhi-xlo,1.);dy=max(yhi-ylo,1e-12)
 out=['<svg viewBox="0 0 700 280" role="img">',f'<title>{html_lib.escape(title)}</title><text x="40" y="18">{html_lib.escape(title)}</text><path d="M40 38V208H670" fill="none" stroke="#789"/>']
 for i,(name,xy) in enumerate(series.items()):
  color=['#167b91','#cb6847','#8052a0','#4e8c4a','#546675'][i%5];coords=' '.join(f'{40+(x-xlo)/dx*625:.2f},{208-(y-ylo)/dy*155:.2f}' for x,y in xy)
  out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.7"/><text x="{40+i*128}" y="240" fill="{color}">{html_lib.escape(name)}</text>')
 out.append(f'<text x="40" y="32">Y: {ylo:.5g}–{yhi:.5g}</text><text x="250" y="272">{html_lib.escape(xlabel)}: {xlo:g}–{xhi:g}</text></svg>');return ''.join(out)

def loss_curves(metrics,title):
 rows=metrics['updates'];groups=np.array_split(np.arange(len(rows)),min(120,len(rows)));series={}
 for key in ('loss','geometry','rgb','alpha'):
  series[key]=[(float(np.mean([rows[i]['update'] for i in g])),float(np.mean([rows[i][key] for i in g]))) for g in groups]
 return multi_curve(series,title+' (120 contiguous-bin means; raw geometry)', 'successful updates')

def summary_sections(root,results,analysis,short,refs):
 """Compute all prose from completed records; never infer physical causality."""
 def cd(tag,ep=None):return results[tag]['equal' if ep is None else 'episodes']['mean_cd_mm'] if ep is None else results[tag]['episodes'][ep]['mean_cd_mm']
 def change(a,b):return (a-b)/a*100
 base,a,b=cd('baseline_best'),cd('A_best'),cd('B_best');selection=json.loads((root/'selection.json').read_text());winner='A' if a<=b else 'B'
 verdict='本轮未达到接触验证CD提升目标，两组长实验均不优于原Baseline。' if min(a,b)>=base else '至少一组长实验在该接触验证口径下低于原Baseline；仍须检查逐Episode与渲染代价。'
 lines=['# V4 Contact-only V1 验收','',f'**工程流程完成。{verdict}**','',f'训练源码 `{refs["A_best"]["source_sha"]}`；历史基线源码 `{refs["baseline_best"]["source_sha"]}`。八组短实验各800次成功更新，A/B各12352次。H10/Δ1、seed42、同一初始化；未读取测试集GT进行评估。','',
 '## 统一口径','', '主要指标使用同一52个接触验证窗口：ep3内36窗口平均、ep6内16窗口平均，再对Episode等权。best按共同的普通CD（归一化）+RGB+0.1Alpha选取，lowest_cd仅为诊断。所有模型还报告原完整验证集，窗口集合不同的均值不混比。','',
 '| 模型 | 接触CD/mm | ep3 | ep6 | RGB | Alpha | 共同联合指标 | 相对Copy改善 | 相对Baseline改善 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for tag,v in results.items():
  q=v['equal'];lines.append(f'| {tag} | {q["mean_cd_mm"]:.5f} | {cd(tag,"3"):.5f} | {cd(tag,"6"):.5f} | {q["mean_rgb"]:.5f} | {q["mean_alpha"]:.5f} | {q["common_joint"]:.5f} | {change(cd("copy"),cd(tag)):+.2f}% | {change(base,cd(tag)):+.2f}% |')
 lines+=['','## 八组短实验与选择','', '| ID | 最佳更新 | CD/mm | ep3 | ep6 | 联合指标 |','|---|---:|---:|---:|---:|---:|']
 for r in short:lines.append(f'| {r["id"]} | {r["best_step"]} | {r["cd_mm"]:.5f} | {r["ep3"]:.5f} | {r["ep6"]:.5f} | {r["joint"]:.5f} |')
 lines+=['',f'B选择 `{selection["winner"]}`，fallback={selection["fallback"]}。冻结规则要求相对S5，ep3/ep6 CD均不变差且渲染组合不变差，才按总体CD排名；否则使用S6探索性Loss对照。严格零退化规则会拒绝极小波动，不能把“未入选”等同于没有任何收益。','',
 '## 七个研究问题','',
 f'1. **仅接触窗口是否提高验证表现？** Long A相对原Baseline的CD变化为{change(base,a):+.2f}%（正数为改善）；ep3/ep6分别为{change(cd("baseline_best","3"),cd("A_best","3")):+.2f}% / {change(cd("baseline_best","6"),cd("A_best","6")):+.2f}%。这是同一验证集的描述性结果；不同训练曝光与采样序列也随接触筛选改变。',
 f'2. **Squared CD是否更好？** Long B相对A的CD变化为{change(a,b):+.2f}%；ep3/ep6分别为{change(cd("A_best","3"),cd("B_best","3")):+.2f}% / {change(cd("A_best","6"),cd("B_best","6")):+.2f}%。本轮只支持这个固定λ和seed下的对照，不能推广为所有数据均优于普通CD。',
 '3. **均衡采样是否缓解分化？** 下表保留S5→S7、S6→S8的逐Episode差异；如果一项改善而另一项退化，只能说存在取舍，不能称为统一改善。本轮没有均衡采样的长期验证。',
 '4. **是否存在单双手实现Bug？** 已执行的四个真实窗口Mask扰动、四头有符号运动公式、实际FPS Anchor历史位置、跨步图连接和既有回归测试未发现确定性Bug。30/60工具会改变softmax分母，属于结构性质；Null不是接触概率。该检查不能排除未覆盖的所有错误。',
 f'5. **哪组长实验更好？** 按best_joint checkpoint的Episode等权CD，Long {winner}较低；下方同时保留last，不能据best单点声称稳定。',
 f'6. **相对原Baseline是否可信涨点？** A/B总体CD改善分别为{change(base,a):+.2f}% / {change(base,b):+.2f}%。共同52窗口、预算和初始化受控，但只有一个seed且使用该验证集选模/筛选，尚不是独立、无偏的泛化证明。RGB/Alpha与ep3/ep6必须一并查看。',
 '7. **是否伴随更正确的运动和条件利用？** 逐帧几何/RGB/Alpha与下方条件消融共同判断。ΔCD=扰动−正常，正值支持正常条件在这些窗口上有帮助；输出变化量只表示敏感性。不能由运动更大、Attention更集中或单次消融推出接触因果机制。','',
 '| 短实验变化 | ep3 ΔCD/mm | ep6 ΔCD/mm |','|---|---:|---:|']
 for pair,v in analysis['short_factorial_delta_cd_mm'].items():lines.append(f'| {pair} | {v["3"]:+.5f} | {v["6"]:+.5f} |')
 lines+=['','## 条件利用与训练稳定性','', '| 模型 | Episode | 扰动 | ΔCD/mm | 输出变化RMS/mm（h均值） |','|---|---|---|---:|---:|']
 for tag in ('baseline_best','A_best','B_best'):
  for ep in ('3','6'):
   for mode,c in analysis['models'][tag][ep]['conditions'].items():lines.append(f'| {tag} | {ep} | {mode} | {c["delta_cd_mm"]:+.5f} | {np.mean(c["output_change_rms_mm"]):.5f} |')
 lines+=['','| Long | best更新 | best CD | last CD | 裁剪比例 | 峰值allocated GiB |','|---|---:|---:|---:|---:|---:|']
 for n in ('A','B'):
  o=analysis['optimization'][n];lines.append(f'| {n} | {o["best_step"]} | {cd(n+"_best"):.5f} | {cd(n+"_last"):.5f} | {o["clipped_fraction"]:.3%} | {o["allocated_peak"]/2**30:.3f} |')
 lines+=['','几何/RGB/Alpha梯度方向与模块范数见 `comparisons/research_analysis.json`。几何项使用对应训练组的实际权重；RGB/Alpha分项梯度按原始分量记录。全部更新的原始梯度与实际参数更新量保存在各组metrics.json。','',
 '| Long | 几何–RGB梯度cos | 几何–Alpha梯度cos | RGB–Alpha梯度cos |','|---|---:|---:|---:|']
 for n in ('A','B'):
  audit=analysis['optimization'][n].get('gradient_audit')
  if audit:
   co=audit['cosine'];lines.append(f'| {n} | {co["geometry_rgb"]:+.5f} | {co["geometry_alpha"]:+.5f} | {co["rgb_alpha"]:+.5f} |')
 lines+=['','以上梯度方向只对应预先固定的训练ep1代表Batch与各自best权重，不代表全部数据的梯度关系。负cos只表示当次方向竞争，不足以证明根因。','',
 '## 原完整验证集（同口径另列）','', '| 模型 | 全验证CD/mm（Episode等权） | RGB | Alpha |','|---|---:|---:|---:|']
 for tag,v in results.items():
  if 'all_window_val' in v:
   q=v['all_window_val']['equal'];lines.append(f'| {tag} | {q["mean_cd_mm"]:.5f} | {q["mean_rgb"]:.5f} | {q["mean_alpha"]:.5f} |')
 lines+=['','## 边界、证据与复现','',
 '人工标注是整体接触操作区间，不是逐帧物理接触GT。训练709、验证52、测试51窗口；测试仅核查标注契约。接触样本在12352更新下重复访问约17.42次，相比原1544窗口的8轮曝光不同。没有新增seed43、Loss搜索或架构修改。历史Baseline的best由原完整验证集选择，本轮best由接触验证集选择；选模口径也不同，因此需结合相同更新预算的last结果，不能把best差异完全归因于接触筛选。',
 'λ_sq仅在六个训练Episode代表窗口、相同初始化下，按XYZ梯度范数比值的中位数校准一次；见provenance/calibration.json。共享初始化与配对样本序列见provenance/short_pairing_verification.json。',
 '冻结审计曾发现010相机的空深度；本轮实际沿用正式V4的001/011/021三路相机，不能将010异常直接归因于本轮训练。CPU逐帧审计确认709训练及52验证窗口的每个目标都有三路非空有效区域，见provenance/selected_camera_audit.json；这只确认像素有效性，不证明重建完整。始终沿用target_depth_valid。图像展示完整重建GS渲染，指标仅统计有效像素；GT不是原始照片。GS逐帧独立重建、无材料点ID，缺失区域和初态属性继承限制仍存在。',
 '工程验收、训练拟合与研究收益分开判断。不可仅靠低CD判定物理正确；不可将小幅单seed差异视为统计显著。固定可视化窗口在训练前按每Episode中位起点冻结，无结果选窗。',
 '原始记录：contact_window_manifest.json、bug_audit.json、selection.json、各组metrics.json/training_metrics.csv、comparisons/contact_comparison.json、checkpoint_provenance.json、pipeline_submission.json。测试日志和CUDA Smoke位于根目录与provenance/。','']
 charts='<h2>逐 horizon：统一接触验证</h2>'
 for ep in ('3','6'):
  for key,unit in [('cd_mm','CD / mm'),('rgb','RGB MAE'),('alpha','Alpha MAE'),('motion_rms_mm','Pred motion RMS / mm')]:
   series={tag:[(i+1,y) for i,y in enumerate(results[tag]['episodes'][ep][key])] for tag in ('copy','baseline_best','A_best','B_best')}
   charts+=multi_curve(series,f'ep{ep} {unit}','horizon')
  for key,label in [('translation_rms_mm','Residual RMS after mean translation / mm'),('neighbor_rms_mm','Initial-neighbor displacement RMS / mm')]:
   if all('structure' in analysis['models'][tag][ep] for tag in ('copy','baseline_best','A_best','B_best')):
    series={tag:[(i+1,y) for i,y in enumerate(analysis['models'][tag][ep]['structure'][key])] for tag in ('copy','baseline_best','A_best','B_best')}
    charts+=multi_curve(series,f'ep{ep} {label}','horizon')
 return '\n'.join(lines),charts

def scientific_tables(root,results):
 """Descriptive comparisons only; every row retains both validation episodes."""
 modes={}
 for tag,v in results.items():
  modes[tag]={}
  for ep,e in v['episodes'].items():
   copy_cd=np.array(results['copy']['episodes'][ep]['cd_mm']);base_cd=np.array(results['baseline_best']['episodes'][ep]['cd_mm']);cd=np.array(e['cd_mm'])
   r=dict(**e,copy_improvement_percent=((copy_cd-cd)/copy_cd*100).tolist(),baseline_improvement_percent=((base_cd-cd)/base_cd*100).tolist())
   windows=[w for w in v['windows'] if str(w['episode'])==ep]
   r['structure']={key:np.mean([[d[key] for d in w['diagnostics']] for w in windows],axis=0).tolist() for key in ('translation_rms_mm','neighbor_rms_mm','neighbor_p95_mm')}
   r['conditions']={}
   for mode in ('freeze_action','zero_tactile'):
    ww=[w for w in windows if mode in w.get('conditions',{})]
    if ww:
     r['conditions'][mode]=dict(delta_cd_mm=float(np.mean([w['conditions'][mode]['delta_cd_mm'] for w in ww])),output_change_rms_mm=np.mean([w['conditions'][mode]['output_change_rms_mm'] for w in ww],axis=0).tolist(),signed_motion_change_mm=np.mean([np.array(w['conditions'][mode]['metrics']['signed_mean_motion_mm'])-np.array(w['signed_mean_motion_mm']) for w in ww],axis=0).tolist(),normal_motion_rms_mm=np.mean([w['motion_rms_mm'] for w in ww],axis=0).tolist(),perturbed_motion_rms_mm=np.mean([w['conditions'][mode]['metrics']['motion_rms_mm'] for w in ww],axis=0).tolist())
   modes[tag][ep]=r
 short={n:json.loads((root/'short_experiments'/n/'metrics.json').read_text()) for n in matrix()};pairs={}
 for a,b in [('S5','S6'),('S7','S8'),('S5','S7'),('S6','S8')]:
  pairs[a+'_to_'+b]={ep:short[b]['best_validation']['episodes'][ep]['mean_cd_mm']-short[a]['best_validation']['episodes'][ep]['mean_cd_mm'] for ep in ('3','6')}
 optimization={}
 for n,m in short.items():
  optimization[n]=dict(clipped_fraction=float(np.mean([r['clipped'] for r in m['updates']])),best_step=m['best_step'],last_validation=m['evaluations'][-1]['equal'],train=m['train_diagnostic']['equal'],train_copy=m['train_copy']['equal'],allocated_peak=m['allocated_peak'],reserved_peak=m['reserved_peak'],seconds=m['seconds'])
 for n in ('A','B'):
  m=json.loads((root/f'long_experiment_{n}'/n/'metrics.json').read_text());optimization[n]=dict(clipped_fraction=float(np.mean([r['clipped'] for r in m['updates']])),best_step=m['best_step'],last_validation=m['evaluations'][-1]['equal'],allocated_peak=m['allocated_peak'],reserved_peak=m['reserved_peak'],seconds=m['seconds'],gradient_audit=m['gradient_audit'])
 out=dict(models=modes,short_factorial_delta_cd_mm=pairs,optimization=optimization,interpretation='Positive ablation delta CD means normal conditions helped on these windows; sensitivity is not contact causality. Best checkpoints selected on this same validation set; single seed.')
 dump(root/'comparisons/research_analysis.json',out);return out

def build_report(root,results,refs):
 for name in matrix():
  s=json.loads((root/'short_experiments'/name/'status.json').read_text());assert s['status']=='COMPLETED' and s['successful_updates']==800
 for name in ('A','B'):
  s=json.loads((root/f'long_experiment_{name}'/name/'status.json').read_text());assert s['status']=='COMPLETED' and s['successful_updates']==12352
 expected={tuple(k) for k in json.loads((root/'contact_window_manifest.json').read_text())['windows']['val']}
 assert len(expected)==52
 assert all({(w['episode'],w['start']) for w in v['windows']}==expected for v in results.values())
 analysis=scientific_tables(root,results)
 rows=[];lines=[];registry=json.loads((root/'experiment_registry.json').read_text())
 for key,v in results.items():
  row=dict(model=key,cd_mm=v['equal']['mean_cd_mm'],ep3=v['episodes']['3']['mean_cd_mm'],ep6=v['episodes']['6']['mean_cd_mm'],rgb=v['equal']['mean_rgb'],alpha=v['equal']['mean_alpha'],joint=v['equal']['common_joint']);rows.append(row)
 with (root/'long_experiment_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 short=[]
 for name in matrix():
  d=root/'short_experiments'/name;m=json.loads((d/'metrics.json').read_text());short.append(dict(id=name,cd_mm=m['best_validation']['equal']['mean_cd_mm'],ep3=m['best_validation']['episodes']['3']['mean_cd_mm'],ep6=m['best_validation']['episodes']['6']['mean_cd_mm'],joint=m['best_validation']['equal']['common_joint'],best_step=m['best_step'],updates=m['successful_updates']))
  registry['short'][name].update(source=json.loads((d/'source_provenance.json').read_text()),configuration=json.loads((d/'config.json').read_text()),successful_updates=m['successful_updates'],checkpoint=str(d/'best_joint.pt'),best_validation=m['best_validation']['equal'],status='COMPLETED')
 with (root/'short_experiment_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(short[0]));w.writeheader();w.writerows(short)
 curves='<h2>训练与固定验证曲线</h2><p>训练曲线为连续区间均值，仅用于显示；原始每次更新保存在CSV/JSON。普通CD与平方CD的训练Loss量纲不同，不直接比较它们的绝对值。</p>'
 for n in ('A','B'):
  d=root/f'long_experiment_{n}'/n;m=json.loads((d/'metrics.json').read_text());curves+=f'<h3>Long {n}</h3>'+curve_svg(m['evaluations'])+loss_curves(m,f'Long {n} / {m["config"]["loss"]}');registry['long_'+n]=dict(config=m['config'],source=json.loads((d/'source_provenance.json').read_text()),successful_updates=m['successful_updates'],checkpoint=str(d/'best_joint.pt'),status='COMPLETED',best_validation=m['best_validation']['equal'])
 curves+='<details><summary>八组短实验曲线</summary>'
 for name in matrix():
  m=json.loads((root/'short_experiments'/name/'metrics.json').read_text());curves+=f'<h3>{name}</h3>'+curve_svg(m['evaluations'])+loss_curves(m,f'{name} / {m["config"]["loss"]}')
 curves+='</details>'
 dump(root/'experiment_registry.json',registry)
 columns=list(rows[0]);table='<table><tr>'+''.join(f'<th>{k}</th>' for k in columns)+'</tr>'+''.join('<tr>'+''.join(f'<td>{r[k]:.5f}</td>' if isinstance(r[k],float) else f'<td>{r[k]}</td>' for k in columns)+'</tr>' for r in rows)+'</table>'
 base=results['baseline_best']['equal']['mean_cd_mm'];a=results['A_best']['equal']['mean_cd_mm'];b=results['B_best']['equal']['mean_cd_mm'];winner='A' if a<=b else 'B'
 conclusions=dict(contact_only_relative_CD_improvement_percent=(base-a)/base*100,candidate_relative_CD_improvement_percent=(base-b)/base*100,better_long=winner,engineering_pass=True,science='Descriptive single-seed selected-validation results; no test GT evaluated; not an unbiased generalization estimate',actions='Read condition diagnostics for error benefit; output sensitivity alone is not causal learning')
 previous=json.loads((root/'acceptance.json').read_text()) if (root/'acceptance.json').exists() else {'source':provenance()}
 dump(root/'acceptance.json',dict(status='COMPLETED',conclusions=conclusions,source=previous['source'],summary_source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),annotation_unchanged=True,budget={'short':8*800,'long':2*12352},limitations=['Fixed 52-window contact validation used for selection; no new independent generalization test','Single seed, no statistical significance','GS independent reconstruction and missing regions remain','Best versus last and both episodes must be inspected'],references=refs))
 md,horizon_charts=summary_sections(root,results,analysis,short,refs);curves+=horizon_charts
 lookup={n:{(w['episode'],w['start']):w for w in v['windows']} for n,v in results.items()};failures={}
 for n in ('A_best','B_best'):
  failures[n]=[{'episode':ep,'start':t,'delta_cd_vs_copy_mm':w['mean_cd_mm']-lookup['copy'][ep,t]['mean_cd_mm'],'delta_cd_vs_baseline_mm':w['mean_cd_mm']-lookup['baseline_best'][ep,t]['mean_cd_mm']} for (ep,t),w in lookup[n].items() if w['mean_cd_mm']>min(lookup['copy'][ep,t]['mean_cd_mm'],lookup['baseline_best'][ep,t]['mean_cd_mm'])]
 dump(root/'comparisons/failure_windows.json',failures)
 curves+='<details><summary>研究结论与原始证据索引</summary><pre style="white-space:pre-wrap">'+html_lib.escape(md)+'</pre></details><p><a href="comparisons/failure_windows.json">全部相对Copy或Baseline退化窗口（数值清单）</a></p>'
 (root/'final_report.md').write_text(md,encoding='utf8')
 html='''<!doctype html><meta charset="utf-8"><title>V4 Contact-only V1</title><style>body{font:16px system-ui;margin:24px;background:#f4f7fa;color:#243743}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd}section{background:white;padding:16px;margin:12px 0}img{width:100%}.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}canvas{width:100%;background:white}svg{width:min(100%,700px);display:inline-block}input{width:380px}</style><h1>V4 Contact-only V1</h1><nav><a href="#viewer">同步图像与几何</a> · <a href="final_report.md">完整中文验收</a> · <a href="comparisons/research_analysis.json">数值分析</a></nav><p>GT 为未来重建 GS 渲染，不是真实照片。固定代表窗口预先选定；未读取测试集 GT。本页为单seed且经过验证集选择的描述性结果。</p>'''+table+curves+'''<section id="viewer"><h2>固定窗口同步对照</h2><select id="win"></select><select id="mode"><option>rgb</option><option>alpha</option></select><select id="camera"><option value="0">001_cam0</option><option value="1">011_cam0</option><option value="2">021_cam0</option></select><input id="h" type="range" min="1" max="10" value="1"><button id="play">播放/暂停</button><b id="label"></b><div id="views" class="grid"></div><p>完整GS渲染，RGB/Alpha范围0–1；数值指标只使用target_depth_valid像素，空视角不贡献监督。下方为世界坐标 XZ 几何与位移（同一窗口共享范围）；GT无跨帧材料ID。</p><div id="geos" class="grid"></div><div id="links"></div></section><a href="final_report.md">报告</a> · <a href="comparisons/contact_comparison.json">全部逐窗口指标和条件消融</a> · <a href="short_experiment_summary.csv">短实验</a><script src="visualizations/data.js"></script><script>
const $=x=>document.getElementById(x),names=['GT','Copy Initial','All-window Baseline','Long A','Long B'];VIS.forEach((v,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`ep${v.episode} start${v.start}`;$('win').append(o)});
function draw(){const v=VIS[+$('win').value],h=+$('h').value,k=h-1,cam=$('camera').value,mode=$('mode').value,base=`visualizations/ep${v.episode}_start${v.start}/`;$('label').textContent=`H${h} / frame${v.frames[h]}`;$('views').replaceChildren();$('geos').replaceChildren();$('links').replaceChildren();const xyz=[v.gt[k],v.initial,v.models.baseline_best[k],v.models.A_best[k],v.models.B_best[k]],all=[v.initial,...v.gt,...Object.values(v.models).flat() ].flat(),lo=[0,2].map(d=>Math.min(...all.map(p=>p[d]))),hi=[0,2].map(d=>Math.max(...all.map(p=>p[d])));for(let i=0;i<5;i++){const div=document.createElement('div'),img=document.createElement('img');img.src=base+(['baseline_best','baseline_best','baseline_best','A_best','B_best'][i])+`/figures/frame_${h}_view_${cam}_${mode}_${i===0?'gt':i===1?'copy':'pred'}.png`;div.append(document.createTextNode(names[i]),img);$('views').append(div);const cv=document.createElement('canvas');cv.width=420;cv.height=300;$('geos').append(cv);const ctx=cv.getContext('2d'),xy=p=>[15+(p[0]-lo[0])/(hi[0]-lo[0])*390,280-(p[2]-lo[1])/(hi[1]-lo[1])*260];ctx.fillStyle='#157c95';for(const p of xyz[i]){const[x,y]=xy(p);ctx.fillRect(x,y,2,2)}ctx.fillStyle='#cf6944';for(const p of v.controllers[k]){const[x,y]=xy(p);ctx.fillRect(x,y,3,3)}if(i>=2){ctx.strokeStyle='#aa7e47';for(let n=0;n<xyz[i].length;n+=12){const a=xy(v.initial[n]),b=xy(xyz[i][n]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}}}for(const model of ['baseline_best','A_best','B_best']){const a=document.createElement('a');a.href=base+model+'/index.html';a.textContent=model+' 独立几何/三相机查看 ';$('links').append(a)}}
for(const id of ['win','h','mode','camera'])$(id).oninput=draw;let timer=null;$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else timer=setInterval(()=>{$('h').value=(+$('h').value%10)+1;draw()},500)};draw();</script>'''
 (root/'report.html').write_text(html,encoding='utf8')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);mode=p.add_mutually_exclusive_group();mode.add_argument('--select',action='store_true');mode.add_argument('--summarize',action='store_true');a=p.parse_args();root=Path(a.root)
 if a.select:print(json.dumps(select(root)))
 elif a.summarize:
  if not (root/'acceptance.json').exists():raise RuntimeError('Summarize requires completed real-GPU comparison')
  build_report(root,json.loads((root/'comparisons/contact_comparison.json').read_text()),json.loads((root/'checkpoint_provenance.json').read_text()))
 else:
  if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm required')
  torch.set_num_threads(4);final(root)
