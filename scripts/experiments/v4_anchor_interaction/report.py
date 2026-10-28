"""Read-only final checkpoint evaluation, fixed-window exports, offline HTML."""
import csv
import json
import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scripts.training.v4_runtime import load_checkpoint
from scripts.training.train import gs_render_settings
from scripts.training.visualize import export_gaussians
from .model import MODES, paired_models
from .run import Data, fresh, dump, sha, provenance, evaluate, spatial, attention, KEYS, reduce_rows


def curve(series,title):
    xy=[p for v in series.values() for p in v];xmin=min(x for x,y in xy);xmax=max(x for x,y in xy)
    ymax=max(y for x,y in xy)*1.05 or 1.;colors=['#087f8c','#d27839','#7852a1','#3d8b52']
    s=[f'<svg viewBox="0 0 700 240"><text x="35" y="18">{title}</text><path d="M35 30V190H680" fill="none" stroke="#889"/>']
    for j,(tag,points) in enumerate(series.items()):
        coords=' '.join(f'{35+(x-xmin)/max(xmax-xmin,1)*640:.2f},{190-y/ymax*150:.2f}' for x,y in points)
        s.append(f'<polyline points="{coords}" fill="none" stroke="{colors[j%4]}" stroke-width="2"/><text x="{40+j*160}" y="217" fill="{colors[j%4]}">{tag}</text>')
    return ''.join(s)+f'<text x="40" y="35">0–{ymax:.4g}</text></svg>'


@torch.no_grad()
def final(root):
    root=Path(root);data=Data();p=data.processor;protocol=json.loads((root/'provenance/frozen_protocol.json').read_text())
    results={};refs={};histories={};geometry=[]
    for tag in MODES:
        path=root/'runs'/tag;status=json.loads((path/'status.json').read_text())
        if status['successful_updates']!=1600 or status['status']!='COMPLETED':raise ValueError('Incomplete run')
        histories[tag]=json.loads((path/'metrics.json').read_text())
        m=fresh(tag);results[tag]={};refs[tag]={}
        for ck,name in [('best_joint.pt','best_joint'),('lowest_cd.pt','lowest_cd'),('last.pt','last')]:
            file=path/ck;before=sha(file);state=load_checkpoint(file,m,processor_state=p.state_dict())
            r=evaluate(m,data,data.windows['val'],diagnostics=name=='best_joint')
            r['contact']=reduce_rows([row for row in r['windows'] if (row['episode'],row['start']) in data.contact])
            results[tag][name]=r
            if sha(file)!=before:raise RuntimeError('Checkpoint modified during read')
            refs[tag][name]=dict(path=str(file),sha256=before,updates=state['successful_steps'],source_sha=state['source_sha'])
            dump(root/'comparison'/f'{tag}_{name}.json',r)
        del m;torch.cuda.empty_cache()
    m=fresh('G0');copy=evaluate(m,data,data.windows['val'],is_copy=True)
    copy['contact']=reduce_rows([r for r in copy['windows'] if (r['episode'],r['start']) in data.contact])
    dump(root/'comparison/copy_initial.json',copy)
    del m;torch.cuda.empty_cache()
    template=paired_models()['G0']
    for ep,t in KEYS:
        raw,b=data.batch((ep,t),template);keep=b['inputs']['gs_mask'][0];scale=p.state_dict()['scale']*1000
        initial=raw['inputs']['gs_xyz'][0,keep.cpu()].numpy();ids=np.linspace(0,len(initial)-1,min(900,len(initial)),dtype=int)
        record=dict(episode=ep,start=t,frame_ids=raw['meta']['frame_ids'][0].tolist(),initial=initial[ids].tolist(),models={},regions={},gt=[],controllers=[],errors={},spatial={},attention={})
        for h in range(10):
            gt=raw['targets']['target_xyz'][h][0,raw['targets']['target_mask'][h][0]].numpy()
            record['gt'].append(gt[np.linspace(0,len(gt)-1,min(900,len(gt)),dtype=int)].tolist())
            cp=raw['inputs']['controller_points'][0,h].reshape(60,3);cm=raw['inputs']['controller_mask'][0,h].reshape(60)&raw['inputs']['operation_hands'][0].repeat_interleave(30)
            record['controllers'].append(cp[cm].tolist())
        for tag in MODES:
            m=fresh(tag);load_checkpoint(root/'runs'/tag/'best_joint.pt',m,processor_state=p.state_dict())
            pred,tr=m(b['inputs'],return_trace=True);sp,regions=spatial(pred,b,scale)
            world=p.denormalize_xyz(pred['xyz'],b['inputs']['gs_mask'][:,None].expand(-1,10,-1))[0,:,keep].cpu().numpy()
            record['models'][tag]=world[:,ids].tolist();record['spatial'][tag]=sp;record['attention'][tag]=attention(tr,scale)
            label=np.zeros(len(initial),dtype=int);label[regions['near'].cpu().numpy()]=1;label[regions['far'].cpu().numpy()]=2
            record['regions']=label[ids].tolist()
            target=b['targets']['target_xyz'][-1][0,b['targets']['target_mask'][-1][0]]
            from scripts.training.losses import nearest_distances
            record['errors'][tag]=(nearest_distances(pred['xyz'][0,-1,keep],target,512)*scale)[ids].cpu().tolist()
            arrays={}
            for layer,steps in tr.get('interactions',{}).items():
                for h in (0,4,9):
                    z=steps[h]
                    if 'hybrid_weights' in z:arrays[f'layer{layer}_h{h+1}_hybrid']=z['hybrid_weights'].cpu().numpy()
                    for j,a in enumerate(z.get('anchor_weights',[])):arrays[f'layer{layer}_h{h+1}_a2a{j}']=a.cpu().numpy()
            arrays.update(anchor_xyz=np.stack([tr['steps'][h]['anchor_xyz'].cpu().numpy() for h in (0,4,9)]),
                controller_attention=np.stack([tr['steps'][h]['alpha'][...,:60].cpu().numpy() for h in (0,4,9)]))
            dest=root/'visualizations'/f'ep{ep}_start{t}'/tag;dest.parent.mkdir(exist_ok=True)
            args=SimpleNamespace(output=str(dest),checkpoint=str(root/'runs'/tag/'best_joint.pt'),episode=ep,start=t,split='train' if ep in (2,7) else 'val')
            export_gaussians(args,dict(scene='001-rope',horizon=10,gs_decoder='anchors',gs_structure='none',render_settings=gs_render_settings()),raw,b,pred,p)
            np.savez_compressed(dest/'attention.npz',**arrays)
            del m,pred,tr;torch.cuda.empty_cache()
        geometry.append(record)
    (root/'visualizations/data.js').write_text('window.VIS='+json.dumps(geometry,allow_nan=False)+';',encoding='utf8')
    dump(root/'comparison/fixed_windows.json',geometry)
    dump(root/'comparison/checkpoint_provenance.json',refs)
    build_report(root,results,copy,histories,refs)


def build_report(root,results,copy,histories,refs):
    previous=json.loads((root/'acceptance.json').read_text()) if (root/'acceptance.json').exists() else {}
    protocol=json.loads((root/'provenance/frozen_protocol.json').read_text())
    base=results['G0']['best_joint']['equal']['mean_cd_mm']
    winner=min(MODES,key=lambda tag:results[tag]['best_joint']['equal']['mean_cd_mm'])
    lines=['# V4 Anchor–Anchor Interaction V1 验收','',
        '四组均为All-Window、H10/Δ1、seed42、1600次成功更新；原始CD+RGB+0.1Alpha。G0为同预算重新训练的原V4，未用历史长期权重替代。测试集GT未读取。',
        '', '完整验证集102窗口及辅助Contact52均先Episode内平均，再ep3/ep6等权。best_joint仅依据完整验证集联合Loss、每200更新选择；lowest_cd单列；Contact不参与选择。历史长期Baseline不可作为1600更新的严格配对对照。',
        '', '| 模型/权重 | 全Val CD/mm | ep3 | ep6 | RGB | Alpha | Joint | Contact CD/mm | 相对G0 best改善 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    rows=[]
    for tag in MODES:
        for name in ('best_joint','lowest_cd','last'):
            r=results[tag][name];q=r['equal'];contact=r['contact']['equal']['mean_cd_mm']
            row=dict(model=tag,checkpoint=name,step=refs[tag][name]['updates'],cd_mm=q['mean_cd_mm'],ep3=r['episodes']['3']['mean_cd_mm'],ep6=r['episodes']['6']['mean_cd_mm'],rgb=q['mean_rgb'],alpha=q['mean_alpha'],joint=q['common_joint'],contact_cd_mm=contact)
            rows.append(row);lines.append(f'| {tag}/{name} | {row["cd_mm"]:.5f} | {row["ep3"]:.5f} | {row["ep6"]:.5f} | {row["rgb"]:.5f} | {row["alpha"]:.5f} | {row["joint"]:.5f} | {contact:.5f} | {(base-row["cd_mm"])/base*100:+.2f}% |')
    q=copy['equal'];lines+=['',f'Copy Initial：完整验证CD={q["mean_cd_mm"]:.5f}mm；RGB={q["mean_rgb"]:.5f}，Alpha={q["mean_alpha"]:.5f}；Contact52 CD={copy["contact"]["equal"]["mean_cd_mm"]:.5f}mm。','',
        f'**本轮完整验证best_joint最低CD为{winner}。** '+('新增A2A/Hybrid均未在主几何指标超过G0。' if winner=='G0' else '该排名只是单seed、1600更新的描述性对照，须同时检查逐Episode、末步、渲染、Near/Far和条件消融，不能直接认定物理建模或泛化改善。'),
        '', '## 资源与结构','', '| 模型 | 参数量 | 相对G0新增 | 总训练/验证秒 | 峰值allocated/GiB | 裁剪比例 |','|---|---:|---:|---:|---:|---:|']
    for tag in MODES:
        h=histories[tag];n=protocol['initializations'][tag]['parameters'];b=protocol['initializations']['G0']['parameters']
        lines.append(f'| {tag} | {n} | {n-b} | {h["seconds"]:.1f} | {h["allocated_peak"]/2**30:.3f} | {np.mean([r["clipped"] for r in h["updates"]])*100:.2f}% |')
    lines+=['','G1/G2是相同两层模块分别置于两层Cross之前/之后，均无自身Key，但保留残差及退让Null。G3仅128个Anchor Query，Keys为60Controller+128Anchor+Null，共同Softmax；Controller子集不重新归一化，四头工具运动仅聚合前60个Key。几何系数可学习，未固定距离排名。所有组使用原始V4的151D Motion输入和累计位移/固定RBF。G1/G2参数量更大，不能把全部差异只归因于放置位置。',
        '', '## H10 Near/Far（固定初态25%分区）','', '| 模型 | Near位移/mm | Far位移/mm | Near/Far | Near→GT误差/mm | Far→GT误差/mm | 刚体去除残差/mm |','|---|---:|---:|---:|---:|---:|---:|']
    for tag in MODES:
        eps=[]
        for ep in (3,6):
            rr=[r['spatial'] for r in results[tag]['best_joint']['windows'] if r['episode']==ep]
            eps.append([np.mean([z['near']['motion_rms_mm'] for z in rr]),np.mean([z['far']['motion_rms_mm'] for z in rr]),np.mean([z['near_far_ratio'] for z in rr]),np.mean([z['near']['pred_to_gt_mean_mm'] for z in rr]),np.mean([z['far']['pred_to_gt_mean_mm'] for z in rr]),np.mean([z['proper_rigid_removed_rms_mm'] for z in rr])])
        v=np.mean(eps,axis=0);lines.append('| '+tag+' | '+' | '.join(f'{x:.5f}' for x in v)+' |')
    lines+=['','Near/Far根据初态GS到操作手有效Controller的距离排序固定；区域误差为单向区域→独立GT点集最近距离，不能当作材料点追踪或双向区域CD。Near/Far比接近1不必然更好。刚体去除采用初态与预测身份的proper Kabsch，未按GT行号匹配。完整逐窗、局部相对Copy改善、邻域变化和H1–H10误差均在comparison原始JSON。','',
        '## 条件扰动（best_joint，完整验证Episode等权）','', '| 模型 | 冻结Action ΔCD/mm | 输出变化/mm | 触觉置零 ΔCD/mm | 输出变化/mm |','|---|---:|---:|---:|---:|']
    for tag in MODES:
        vals=[]
        for ep in (3,6):
            rr=[r for r in results[tag]['best_joint']['windows'] if r['episode']==ep]
            vals.append([np.mean([r['conditions']['freeze_action']['delta_cd_mm'] for r in rr]),np.mean([r['conditions']['freeze_action']['output_change_rms_mm'] for r in rr]),np.mean([r['conditions']['zero_tactile']['delta_cd_mm'] for r in rr]),np.mean([r['conditions']['zero_tactile']['output_change_rms_mm'] for r in rr])])
        lines.append('| '+tag+' | '+' | '.join(f'{v:+.5f}' for v in np.mean(vals,axis=0))+' |')
    lines+=['','ΔCD=扰动−正常；正值表示该对照下正常条件有帮助。输出变化只是敏感性，不能证明条件因果或泛化。Attention质量必须结合几何误差判断；Null/Anchor/Controller质量不等于接触概率或物理作用力。G3的共享Softmax可能竞争外部工具质量，不能忽略Controller Mass。',
        '', '## 工程与复现','',f'训练/评估源码 `{refs["G0"]["best_joint"]["source_sha"]}`；基准 `{protocol["base_sha"]}`。四组共享初始同名权重哈希和统一1600样本顺序已冻结。G0与原V4非零头逐位等价、Mask/因果/四头运动/梯度/跨架构checkpoint单测及真实gsplat Smoke证据见tests/和provenance/。',
        '六个窗口在训练前固定：Train ep2/141、ep7/107；Val ep3/140、ep6/115、ep3/230、ep6/160。未按结果选窗。所有图像是GS重建渲染，GT不是原始照片。原11D属性继承初态；GT独立重建，不代表有固定材料粒子ID。',
        '本轮停止于4×1600；未追加架构、seed、Loss、长训练，未修改main或历史结果。浏览器真实交互验收另见provenance/browser_acceptance.json（缺少该文件时表示尚未验证）；静态/图像检查不能代替浏览器验收。']
    analysis={}
    for tag in MODES:
        model={}
        for ep in (3,6):
            rr=[r for r in results[tag]['best_joint']['windows'] if r['episode']==ep]
            model[str(ep)]={
                'h10_attention':{key:np.mean([r['attention'][9][key] for r in rr],axis=0).tolist() for key in ('controller_mass_by_head','anchor_mass_by_head','null_mass_by_head')},
                'conditions':{mode:dict(delta_cd_mm=float(np.mean([r['conditions'][mode]['delta_cd_mm'] for r in rr])),output_rms_mm=float(np.mean([r['conditions'][mode]['output_change_rms_mm'] for r in rr]))) for mode in ('freeze_action','zero_tactile')},
                'near_far':{region:{key:float(np.mean([r['spatial'][region][key] for r in rr])) for key in ('motion_rms_mm','pred_to_gt_mean_mm','copy_to_gt_mean_mm','local_improvement_mm')} for region in ('near','far')}}
        analysis[tag]=model
    dump(root/'comparison/research_analysis.json',analysis)
    lines+=['','## 研究判断与Attention证据','',
        '本轮三个新增交互均未超过同预算G0，不能选择Pre、Post或Hybrid作为已验证更优的结构。差距小于1%，仅支持该seed/预算的负面结果，不证明长期训练必然无效。G1/G2的Near位移更小而Far位移没有明显增加；Near→GT误差没有同步改善，因此比值变小不能作为协调改善证据。',
        '', '| 模型 | H10 Controller Mass | H10 Anchor Mass | H10 Null Mass | H10位移RMS/mm |','|---|---:|---:|---:|---:|']
    for tag in MODES:
        values=[np.mean([analysis[tag][ep]['h10_attention'][key] for ep in ('3','6')]) for key in ('controller_mass_by_head','anchor_mass_by_head','null_mass_by_head')]
        lines.append(f'| {tag} | {values[0]:.5f} | {values[1]:.5f} | {values[2]:.5f} | {results[tag]["best_joint"]["equal"]["motion_rms_mm"][9]:.5f} |')
    lines+=['','上表先Episode等权、再四头平均，仅用于概览；每个Head的原始质量见research_analysis.json与固定窗口attention.npz。G3的内部Anchor分支获得大量质量，Controller质量低于G0；这与竞争假设相符，但不是降低条件收益的独立因果证明。所有模型H10运动仍小，不能把微小CD差异当作成功学习了真实柔性传播。',
        '实现细节：G1/G2的A2A除排除自身的Anchor keys外，复用了固定零logit/零Value的数值Null，正常有效Anchor情况下也参与Softmax，并非只在全无效时启用；两组完全一致。这是相对“仅128个Anchor候选”字面定义的实现差异，不能将本轮结果等同于无Null的A2A消融。未追加无Null实验。G3按照指定共同Softmax保留Null。',
        '下一步结论仅为保留G0作为本轮对照；不自动采用新结构，不追加训练、层数或Loss搜索。更强的长期结论仍未验证。']
    tests=[]
    for label,name in [('local','local-cpu.txt'),('server','server-cpu.txt')]:
        path=root/'tests'/name
        if path.exists():
            import re
            text=path.read_text(errors='replace');n=re.search(r'Ran (\d+) tests',text);skip=re.search(r'OK \(skipped=(\d+)\)',text)
            tests.append(dict(location=label,total=int(n[1]) if n else None,skipped=int(skip[1]) if skip else 0,passed='\nOK' in text))
    lines+=['','## 实际验收记录','',f'CPU测试：{tests}。实际CUDA/gsplat每组两次更新、完整H10梯度及checkpoint恢复均通过；见provenance/cuda_gate.json。所有训练与评估使用Slurm30994中的用户授权tmux单GPU，四组顺序运行；未创建四个额外作业，未中断30995。']
    (root/'final_report.md').write_text('\n'.join(lines),encoding='utf8')
    with (root/'comparison/summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tables=['<h2>数值结果</h2><table><tr>'+''.join('<th>'+c+'</th>' for c in rows[0])+'</tr>']
    for row in rows:tables.append('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row.values())+'</tr>')
    tables.append('</table>')
    charts=curve({tag:[(r['update'],r['equal']['mean_cd_mm']) for r in histories[tag]['evaluations']] for tag in MODES},'Full val CD/mm vs successful updates')
    charts+=curve({tag:list(enumerate(results[tag]['best_joint']['equal']['cd_mm'],1)) for tag in MODES},'H1-H10 best_joint CD/mm')
    charts+=curve({tag:[(float(np.mean([r['update'] for r in histories[tag]['updates'][i:i+20]])),float(np.mean([r['loss'] for r in histories[tag]['updates'][i:i+20]]))) for i in range(0,1600,20)] for tag in MODES},'Training joint loss: contiguous 20-update means')
    page=HTML.replace('__TABLE__',''.join(tables)+charts).replace('__WINNER__',winner)
    (root/'report.html').write_text(page,encoding='utf8')
    label_path=__import__(__package__+'.run',fromlist=['FROZEN']).FROZEN/'contact_boundaries.json'
    annotation_ok=protocol['annotation_sha256']==sha(label_path) if label_path.exists() else previous.get('annotation_unchanged')
    dump(root/'acceptance.json',dict(status='COMPLETED',engineering=True,source=previous.get('source') or provenance(),summary_source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),successful_updates={tag:histories[tag]['successful_updates'] for tag in MODES},
        winner_by_full_validation_CD=winner,scientific_claim='single-seed short-budget descriptive ablation only; inspect per-episode/local/render/conditional tradeoffs',
        checkpoint_provenance=refs,annotation_unchanged=annotation_ok,tests=tests,
        implementation_warning='G1/G2 retain an always-available fixed-zero numerical null in A2A, beyond the literal anchor-only key set; disclosed in report, no extra experiment',
        browser_interaction='consult provenance/browser_acceptance.json'))


HTML='''<!doctype html><meta charset="utf-8"><title>V4 Anchor Interaction</title>
<style>body{font:15px Arial,sans-serif;background:#f4f6fa;color:#213042;margin:22px}section{background:white;padding:15px;margin:15px 0}button,select,input{font-size:15px;margin:5px;padding:5px}.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px}img,canvas{width:100%;background:#eef1f7}h3{font-size:16px}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccd;padding:5px}svg{max-width:700px;width:48%}.small{font-size:12px}</style>
<h1>V4 Anchor–Anchor Attention 消融</h1><p>All-Window · H10/Δ1 · Seed42 · 每组1600次更新。最低完整验证best_joint CD：__WINNER__。单seed短预算结果，不能直接证明泛化或物理传播。</p>
<p><a href="final_report.md">完整中文报告</a> · <a href="comparison/summary.csv">best/lowest/last数值</a> · <a href="provenance/frozen_protocol.json">冻结数据与配置</a>。GT为逐帧重建GS渲染。</p>
<section>窗口<select id="window"></select><button id="back">←</button><button id="play">播放</button><button id="next">→</button><input id="frame" type="range" min="0" max="10" value="1"><b id="label"></b><br>
相机<select id="camera"><option value="0">001_cam0</option><option value="1">011_cam0</option><option value="2">021_cam0</option></select>模式<select id="mode"><option>rgb</option><option>alpha</option></select><p id="links"></p>
<div class="grid" id="images"></div><h3>同步3D中心与运动轨迹：拖动旋转、滚轮缩放</h3><p class="small">Near红、Far蓝、中间灰；GT青色；Controller绿色。箭头由对应初态中心指向当前预测，GT没有材料点轨迹。全部模型共用坐标范围。</p><div class="grid" id="canvases"></div><details><summary>当前窗口H10 Near/Far误差与当前步四头Attention</summary><pre id="localmetrics"></pre></details></section>
<section>__TABLE__</section><section><h2>打开方式</h2>离线打开report.html即可；若浏览器限制本地资源，在本目录执行 <code>python -m http.server 8765 --bind 127.0.0.1</code> 并访问 http://127.0.0.1:8765/report.html。独立ellipsoid/轨迹查看器在每列链接中。完整四头Attention、Near/Far和条件扰动原始数据保留在comparison/与visualizations/。</section>
<script src="visualizations/data.js"></script><script>
const tags=['GT','Copy','G0','G1','G2','G3'],w=document.querySelector('#window'),f=document.querySelector('#frame'),cam=document.querySelector('#camera'),mode=document.querySelector('#mode');let timer=null,yaw=.25,pitch=.2,zoom=1;
VIS.forEach((d,i)=>{let o=new Option('ep'+d.episode+' / start'+d.start,i);w.add(o)});
function points(d,tag,h){return tag==='GT'?(h?d.gt[h-1]:d.initial):tag==='Copy'||!h?d.initial:d.models[tag][h-1]}
function draw(){const d=VIS[+w.value],h=+f.value,all=d.initial.concat(...Object.values(d.models).map(x=>x[9]),...d.gt);let center=[0,1,2].map(k=>all.reduce((s,p)=>s+p[k],0)/all.length),range=Math.max(...[0,1,2].map(k=>Math.max(...all.map(p=>p[k]))-Math.min(...all.map(p=>p[k]))))||1;
document.querySelectorAll('canvas').forEach((c,i)=>{let ctx=c.getContext('2d');c.width=420;c.height=340;ctx.fillStyle='#eef1f7';ctx.fillRect(0,0,420,340);const proj=p=>{let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],a=x*Math.cos(yaw)-y*Math.sin(yaw),b=x*Math.sin(yaw)+y*Math.cos(yaw),v=z*Math.cos(pitch)-b*Math.sin(pitch);return [210+a/range*310*zoom,170-v/range*310*zoom]};let ps=points(d,tags[i],h);ps.forEach((p,j)=>{let q=proj(p);ctx.fillStyle=i===0?'#28a5af':d.regions[j]===1?'#df6650':d.regions[j]===2?'#447dcc':'#85919a';ctx.fillRect(q[0],q[1],2.3,2.3);if(i>1&&h&&j%25===0){let a=proj(d.initial[j]);ctx.strokeStyle='#876f9a';ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...q);ctx.stroke()}});d.controllers[Math.max(0,h-1)].forEach(p=>{let q=proj(p);ctx.fillStyle='#2f9854';ctx.fillRect(q[0]-2,q[1]-2,4,4)})})}
function update(){const d=VIS[+w.value],h=+f.value;document.querySelector('#label').textContent='H'+h+' · 原始frame '+d.frame_ids[h];document.querySelector('#images').innerHTML=tags.map(tag=>{let m=tag==='GT'||tag==='Copy'?'G0':tag,role=tag==='GT'?'gt':tag==='Copy'?'copy':'pred',base='visualizations/ep'+d.episode+'_start'+d.start+'/'+m;return '<div><b>'+tag+'</b><a href="'+base+'/geometry.html"> 3D ellipsoids/paths</a><img src="'+base+'/figures/frame_'+h+'_view_'+cam.value+'_'+mode.value+'_'+role+'.png"></div>'}).join('');document.querySelector('#links').textContent='h0为同一真实初态；h1–h10为独立保存的预测。图像为GS渲染，误差仅使用原有效像素。';document.querySelector('#localmetrics').textContent=JSON.stringify(Object.fromEntries(['G0','G1','G2','G3'].map(tag=>[tag,{H10_spatial:d.spatial[tag],attention:d.attention[tag][Math.max(0,h-1)]}])),null,2);draw()}
document.querySelector('#canvases').innerHTML=tags.map(t=>'<div><b>'+t+'</b><canvas></canvas></div>').join('');document.querySelectorAll('canvas').forEach(c=>{let down=false,old;c.onpointerdown=e=>{down=true;old=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>down=false;c.onpointermove=e=>{if(down){yaw+=(e.clientX-old[0])*.01;pitch+=(e.clientY-old[1])*.01;old=[e.clientX,e.clientY];draw()}};c.onwheel=e=>{e.preventDefault();zoom*=e.deltaY>0?.9:1.1;draw()}});
[w,f,cam,mode].forEach(x=>x.oninput=update);document.querySelector('#next').onclick=()=>{f.value=Math.min(10,+f.value+1);update()};document.querySelector('#back').onclick=()=>{f.value=Math.max(0,+f.value-1);update()};document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.querySelector('#play').textContent='播放'}else{timer=setInterval(()=>{f.value=(+f.value+1)%11;update()},600);document.querySelector('#play').textContent='暂停'}};document.onkeydown=e=>{if(e.key==='ArrowRight')document.querySelector('#next').click();if(e.key==='ArrowLeft')document.querySelector('#back').click()};update();
</script>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='CPU-only rebuild from completed raw results, no inference or training')
    parser.add_argument('--root',required=True);parser.add_argument('--summarize',action='store_true',required=True)
    args=parser.parse_args();root=Path(args.root)
    if not (root/'acceptance.json').is_file():raise ValueError('Requires a completed GPU report receipt')
    histories={tag:json.loads((root/'runs'/tag/'metrics.json').read_text()) for tag in MODES}
    if any(v['successful_updates']!=1600 for v in histories.values()):raise ValueError('Incomplete training')
    results={tag:{name:json.loads((root/'comparison'/f'{tag}_{name}.json').read_text()) for name in ('best_joint','lowest_cd','last')} for tag in MODES}
    build_report(root,results,json.loads((root/'comparison/copy_initial.json').read_text()),histories,json.loads((root/'comparison/checkpoint_provenance.json').read_text()))
