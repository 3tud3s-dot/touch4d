"""Offline P3 evidence/visualization; reuses the production renderer and exports.

--prepare-assets runs in CUDA after the eight fits; --build runs on CPU using
mirrored exports under audit/M{0..3}/{scene}. No training or threshold tuning.
"""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from .p3_fit import AUDIT, PROJECT, VARIANTS, WINDOWS, sha, training_config, load_raws
from .train import to_device, make_renderer, copy_initial, atomic_json
from .model import initial_knn
from .losses import nearest_distances
from .visualize import png_bytes, geometry_data, ellipsoid_axes
from ..data_process import DataProcessor
from ..dataloader.dataset import read_gs, GS_ATTR_FIELDS


def cd(a,b):
    a=torch.as_tensor(a,dtype=torch.float32);b=torch.as_tensor(b,dtype=torch.float32)
    return float((nearest_distances(a,b).mean()+nearest_distances(b,a).mean())*.5)*1000


def write_image(path,value):
    value=np.rint(np.clip(value,0,1)*255).astype(np.uint8)
    if value.ndim==2:value=np.repeat(value[...,None],3,-1)
    path.write_bytes(png_bytes(value))


def structure(arrays):
    initial=arrays['initial_xyz'].astype(np.float64);pred=arrays['predicted_xyz'].astype(np.float64)
    targets=[arrays[f'target_xyz_{h}'].astype(np.float64) for h in range(1,11)]
    groups=lambda x:np.array_split(np.argsort(x[:,0],kind='stable'),4)
    regions=groups(initial);ic=np.array([initial[g].mean(0) for g in regions])
    tc=np.array([[t[g].mean(0) for g in groups(t)] for t in targets])
    motion=((tc-ic)**2).sum(-1).mean(0)
    moving=0 if motion[0]>=motion[3] else 3;less=3-moving
    ids,mask=initial_knn(torch.from_numpy(initial.astype(np.float32))[None],torch.ones(1,len(initial),dtype=torch.bool),16)
    ids=ids[0].numpy();mask=mask[0].numpy();edges=mask&(ids!=np.arange(len(initial))[:,None])
    radius0=np.sqrt((((initial[:,None]-initial[ids])**2).sum(-1)*mask).sum(-1)/mask.sum(-1))
    rows=[]
    for h,(p,t) in enumerate(zip(pred,targets)):
        d=p-initial;dm=d.mean(0)
        r=np.sqrt((((p[:,None]-p[ids])**2).sum(-1)*mask).sum(-1)/mask.sum(-1))
        edge=np.linalg.norm(d[:,None]-d[ids],axis=-1)[edges]*1000
        scale=arrays['predicted_log_scales'][h]-arrays['initial_attributes'][:,:3]
        opacity=arrays['predicted_opacity_logits'][h]-arrays['initial_attributes'][:,7:8]
        bounds=lambda v:np.quantile(v,[0,.05,.5,.95,1]).tolist()
        rows.append(dict(horizon=h+1,frame=int(arrays['frame_ids'][h+1]),
            cd_mm=cd(p,t),centered_cd_mm=cd(p-p.mean(0),t-t.mean(0)),
            centered_x_quartile_cd_mm=[cd((p-p.mean(0))[regions[j]],(t-t.mean(0))[groups(t)[j]]) for j in range(4)],
            translation_rms_mm=float(np.sqrt(((d-dm)**2).sum(-1).mean())*1000),
            centroid_error_mm=float(np.linalg.norm(p.mean(0)-t.mean(0))*1000),
            moving_end_error_mm=float(np.linalg.norm(p[regions[moving]].mean(0)-tc[h,moving])*1000),
            lesser_end_error_mm=float(np.linalg.norm(p[regions[less]].mean(0)-tc[h,less])*1000),
            neighbor_rms_mm=float(np.sqrt((edge**2).mean())) if len(edge) else 0.,
            expansion_p95_mm=float(np.quantile(np.maximum(r-radius0,0)*1000,.95)),
            scale_ratio_quantiles=bounds(np.exp(scale)),opacity_delta_quantiles=bounds(opacity),
            scale_regularization=float((scale**2).mean())))
    return dict(rows=rows,moving_x_quartile=moving,lesser_x_quartile=less,
        convention='Initial prediction X-quartile identities; each GT independently X-ranked. Set diagnostics, not material trajectories.')


def source_template_images(scene,episode):
    """Keep native RGB and refined masks distinct from GS-rendered pseudo-GT."""
    import h5py,io,os,tarfile
    folder=AUDIT/'template';folder.mkdir(exist_ok=True)
    path=PROJECT/'datasets/deform360/processed'/scene/f'episode_{episode}'/'cameras.tar'
    from .train import GS_CAMERAS
    metadata=[]
    with tarfile.open(path) as archive:
        for v,camera in enumerate(GS_CAMERAS[scene]):
            with h5py.File(io.BytesIO(archive.extractfile(camera+'/mask_refined.h5').read()),'r') as f:
                masks=f['data'][132:143]>0
            fd=os.memfd_create('p3-template-video')
            try:
                with os.fdopen(os.dup(fd),'wb') as stream:stream.write(archive.extractfile(camera+'/undistorted.mp4').read())
                command=['ffmpeg','-v','error','-threads','1','-i',f'/proc/self/fd/{fd}','-vf',
                    'select=between(n\\,132\\,142)','-vsync','0','-frames:v','11','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
                data=subprocess.run(command,pass_fds=(fd,),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
            finally:os.close(fd)
            frames=np.frombuffer(data,np.uint8).reshape(11,720,1280,3)
            for i,frame in enumerate(range(132,143)):
                (folder/f'frame{frame}_view{v}_source.png').write_bytes(png_bytes(frames[i]))
                write_image(folder/f'frame{frame}_view{v}_mask.png',masks[i].astype(float))
                overlay=frames[i].astype(float)/255
                overlay[masks[i]]=.7*overlay[masks[i]]+.3*np.array([0,1,0])
                write_image(folder/f'frame{frame}_view{v}_overlay.png',overlay)
                metadata.append(dict(frame=frame,camera=camera,mask_pixels=int(masks[i].sum())))
    return metadata


@torch.no_grad()
def prepare_assets():
    if not torch.cuda.is_available():raise RuntimeError('Asset rendering requires CUDA')
    for scene in WINDOWS:
        config=training_config(scene,'M2');raws,ds,_=load_raws(config);p=DataProcessor(scene)
        for (ep,start),raw in zip(WINDOWS[scene],raws):
            folder=AUDIT/'history'/f'{scene}-ep{ep}-start{start}';folder.mkdir(parents=True,exist_ok=True)
            batch=p(to_device(raw,'cuda'));prev=read_gs(ds.episodes[ep],start-1)
            np.savez_compressed(folder/'history.npz',xyz=prev['xyz'],attr=np.concatenate([prev[k] for k in GS_ATTR_FIELDS],-1),frame_id=start-1)
            # Replace only the template; static calibration/target layout stays shared.
            other=dict(batch,inputs=dict(batch['inputs']))
            other['inputs'].update(gs_xyz=batch['inputs']['gs_prev_xyz'],gs_attr=batch['inputs']['gs_prev_attr'],
                                   gs_mask=batch['inputs']['gs_prev_mask'],gs_lengths=batch['inputs']['gs_prev_lengths'])
            render=make_renderer(config)(copy_initial(other['inputs'],'gs_geometry'),other,p)
            for v in range(3):
                for mode in ('rgb','alpha'):write_image(folder/f'view{v}_{mode}.png',render[mode][0,0,v].cpu().numpy())
            del render,other,batch
    cfg=training_config('051-cube','M0');raws,ds,_=load_raws(cfg);p=DataProcessor('051-cube');raw=raws[1]
    batch=p(to_device(raw,'cuda'));folder=AUDIT/'template';folder.mkdir(exist_ok=True)
    native={frame:read_gs(ds.episodes[0],frame) for frame in range(132,143)}
    records=[]
    for ref in (132,133):
        points=native[ref]['xyz'];attr=np.concatenate([native[ref][k] for k in GS_ATTR_FIELDS],-1)
        keep=torch.ones((1,len(points)),dtype=torch.bool,device='cuda')
        x=torch.from_numpy(points)[None].cuda();a=torch.from_numpy(attr)[None].cuda()
        x=p._xyz(x,keep);a=a.clone();a[...,:3]-=np.log(p.state_dict()['scale'])
        other=dict(batch,inputs=dict(batch['inputs'],gs_xyz=x,gs_attr=a,gs_mask=keep))
        predicted=copy_initial(other['inputs'],'gs_geometry');predicted['xyz']=predicted['xyz'].clone()
        for h in range(10):
            target=batch['targets']['target_xyz'][h][0,batch['targets']['target_mask'][h][0]]
            predicted['xyz'][0,h]+=target.mean(0)-x[0].mean(0)
        # This is a retrospective centroid-alignment diagnostic, never inference.
        rendered=make_renderer(cfg)(predicted,other,p)
        for h,frame in enumerate(range(133,143)):
            if frame<134:continue
            target=native[frame]['xyz']
            records.append(dict(reference=ref,target=frame,raw_cd_mm=cd(points,target),
                centered_cd_mm=cd(points-points.mean(0),target-target.mean(0))))
            for v in range(3):
                for mode in ('rgb','alpha'):write_image(folder/f'ref{ref}_to{frame}_view{v}_{mode}.png',rendered[mode][0,h,v].cpu().numpy())
        del rendered,predicted,other
    source=source_template_images('051-cube',0)
    np.savez_compressed(folder/'reference_geometry.npz',**{f'xyz_{f}':z['xyz'] for f,z in native.items()},
        **{f'attr_{f}':np.concatenate([z[k] for k in GS_ATTR_FIELDS],-1) for f,z in native.items()})
    atomic_json(AUDIT/'template_audit.json',dict(pair_raw_cd_mm=cd(native[132]['xyz'],native[133]['xyz']),
        pair_centered_cd_mm=cd(native[132]['xyz']-native[132]['xyz'].mean(0),native[133]['xyz']-native[133]['xyz'].mean(0)),
        references=records,source_masks=source,shared_targets=list(range(134,143)),
        interpretation='GT-only retrospective shape/centroid comparison; no frame133 enters the prediction of frame133; independent reconstructions have no Gaussian IDs.',
        aligned_renders='Each reference translated to GT centroid for diagnosis only; not a model prediction or extra training loss.'))


def build_report():
    (AUDIT/'data').mkdir(exist_ok=True)
    results=dict(schema='p3-results-v0',runs=[],comparisons={},limitations=[
        'Online conditions through t+k-1; future tactile is not known at the initial time.',
        'X-ranked regions are GT set diagnostics, not material IDs. Padding/opacity are not visibility truth.',
        'Single seed and overlapping fitted windows; 5% is an attention heuristic, not statistical significance.',
        'Fixed template count cannot guarantee restoration of absent geometry. Scale/opacity can conceal errors.',
        'Raw camera RGB/mask and GS pseudo-GT are distinct; existing occlusion and camera clipping remain.'])
    bundles=[];run_manifest=[]
    for scene in WINDOWS:
        for variant in VARIANTS:
            folder=AUDIT/variant/scene
            summary=json.loads((folder/'fit_summary.json').read_text());config=json.loads((folder/'config.json').read_text())
            history=json.loads((folder/'metrics.json').read_text())
            if summary['completed_updates']!=400 or [r['update'] for r in history]!=list(range(20,401,20)):
                raise ValueError('Incomplete formal run')
            result=dict(scene=scene,variant=variant,config=config['training'],commit=config['commit'],
                        runtime=summary['runtime'],checkpoints=summary['checkpoints'])
            results['runs'].append(result)
            run_manifest.append(dict(scene=scene,variant=variant,config=config,completed_updates=400,
                                     summary_sha256=sha(folder/'fit_summary.json'),metrics_sha256=sha(folder/'metrics.json')))
            for tag in ('lowest_cd','last_400'):
                chosen=summary['checkpoints'][tag];rows=chosen['metrics']['model']['windows']
                for ep,start in WINDOWS[scene]:
                    directory=folder/'visualizations'/tag/f'episode-{ep}-start-{start}'
                    with np.load(directory/'prediction.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
                    if arrays['predicted_xyz'].shape[0]!=10 or arrays['frame_ids'].tolist()!=list(range(start,start+11)):
                        raise ValueError('Missing actual ten-step export')
                    unique=len({arrays['predicted_xyz'][h].tobytes() for h in range(10)})
                    if unique!=10:raise ValueError('Repeated predicted horizons')
                    geometry=geometry_data(arrays,limit=300)
                    histfolder=AUDIT/'history'/f'{scene}-ep{ep}-start{start}'
                    with np.load(histfolder/'history.npz') as f:
                        xyz,attr=f['xyz'],f['attr'];ids=np.linspace(0,len(xyz)-1,min(300,len(xyz)),dtype=int)
                        prev=dict(centers=xyz.round(6).tolist(),ids=ids.tolist(),axes=ellipsoid_axes(attr[ids,:3],attr[ids,3:7]).round(9).tolist())
                    diagnostics=structure(arrays);rows[f'{ep}:{start}']['structure']=diagnostics
                    ident=f'{scene}-ep{ep}-t{start}-{variant}-{tag}'
                    bundle=dict(id=ident,scene=scene,episode=ep,start=start,variant=variant,checkpoint=tag,
                        updates=chosen['update'],config=config['training'],commit=config['commit'],geometry=geometry,
                        previous=prev,history_frames=[start-1,start],frames=list(range(start-1,start+11)),
                        metrics=rows[f'{ep}:{start}'],copy_metrics=chosen['metrics']['copy_initial']['windows'][f'{ep}:{start}'],
                        curve=[dict(update=r['update'],cd=r['metrics']['model']['mean_cd_mm'],rgb=r['metrics']['model']['mean_rgb'],
                                    alpha=r['metrics']['model']['mean_alpha']) for r in history],
                        images=directory.relative_to(AUDIT).as_posix()+'/figures',history_images=histfolder.relative_to(AUDIT).as_posix(),
                        prediction_sha256=sha(directory/'prediction.npz'),unique_horizons=unique)
                    path=AUDIT/'data'/f'{ident}.js'
                    path.write_text('window.P3_DATA=window.P3_DATA||{};window.P3_DATA['+json.dumps(ident)+']='+json.dumps(bundle,separators=(',',':'),allow_nan=False)+';',encoding='utf8')
                    bundles.append({k:bundle[k] for k in ('id','scene','episode','start','variant','checkpoint','updates')}|dict(path='data/'+path.name))
    # Paired per-window facts at each independently selected descriptive checkpoint.
    for scene in WINDOWS:
        comparisons={}
        for first,second,label in (('M0','M1','architecture'),('M1','M2','H1_history'),('M2','M3','H2_refinement')):
            a=next(r for r in results['runs'] if r['scene']==scene and r['variant']==first)
            b=next(r for r in results['runs'] if r['scene']==scene and r['variant']==second)
            comparisons[label]={}
            for tag in ('lowest_cd','last_400'):
                am=a['checkpoints'][tag]['metrics']['model'];bm=b['checkpoints'][tag]['metrics']['model']
                wins={k:100*(am['windows'][k]['mean_cd_mm']-v['mean_cd_mm'])/am['windows'][k]['mean_cd_mm'] for k,v in bm['windows'].items()}
                comparisons[label][tag]=dict(mean_cd_improvement_percent=100*(am['mean_cd_mm']-bm['mean_cd_mm'])/am['mean_cd_mm'],
                    per_window_cd_improvement_percent=wins,both_windows_at_least_5_percent=all(v>=5 for v in wins.values()),
                    interpretation='Requires joint geometry/render/structure review; not automatic hypothesis acceptance.')
        results['comparisons'][scene]=comparisons
    template=json.loads((AUDIT/'template_audit.json').read_text())
    template['model_common_targets']=[]
    for run in results['runs']:
        if run['scene']!='051-cube' or run['variant'] not in ('M2','M3'):continue
        for tag,checkpoint in run['checkpoints'].items():
            window=checkpoint['metrics']['model']['windows']['0:132']
            for row in window['structure']['rows'][1:]:
                h=row['horizon']-1
                template['model_common_targets'].append(dict(model=run['variant'],checkpoint=tag,
                    update=checkpoint['update'],target=row['frame'],raw_cd_mm=row['cd_mm'],
                    centered_cd_mm=row['centered_cd_mm'],rgb=window['rgb'][h],alpha=window['alpha'][h],
                    local_shape_cd_mm=row['centered_x_quartile_cd_mm'],scale_ratio_quantiles=row['scale_ratio_quantiles'],
                    opacity_delta_quantiles=row['opacity_delta_quantiles']))
    template['prediction_scope']='M2/M3 trained at start132; inputs131/132, predictions133..142. Comparison134..142 uses no true133 as input.'
    atomic_json(AUDIT/'template_audit.json',template)
    atomic_json(AUDIT/'results.json',results);atomic_json(AUDIT/'run_manifest.json',run_manifest)
    index=dict(bundles=bundles,results=results,template=template,
               acceptance=json.loads((AUDIT/'acceptance.json').read_text()))
    (AUDIT/'data/index.js').write_text('window.P3_INDEX='+json.dumps(index,separators=(',',':'),allow_nan=False)+';',encoding='utf8')
    (AUDIT/'report.html').write_text(HTML,encoding='utf8')
    print(json.dumps(dict(report=str(AUDIT/'report.html'),bundles=len(bundles),horizon=10)),flush=True)


HTML = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>touch4d · P3 双帧与模板</title>
<style>body{font:14px system-ui;margin:20px;background:#f4f6fa;color:#203249}h1{font-size:24px}.bar,.box{background:white;border-radius:10px;padding:12px;margin:12px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}canvas{width:100%;height:280px;touch-action:none}img{width:100%;background:#101820}button,select,input{margin:5px;padding:5px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:6px;border-bottom:1px solid #dde4eb;text-align:right}th:first-child,td:first-child{text-align:left}.warn{color:#9b5221}svg{width:100%;height:160px}#summary{overflow:auto}.crop img{object-fit:cover}small{color:#617487}</style>
<h1>touch4d · P3-v0：两帧历史与有界模板更新</h1><p>固定窗口过拟合 · H10 · 400次更新 · seed42 · 原 P2 与三组 P3 配对。历史 t−1/t 为观测，未来10帧为模型预测；动作/触觉按时间逐步到达。</p>
<div class="box"><a href="results.json">结果 JSON</a> · <a href="acceptance.json">CUDA 验收</a> · <a href="run_manifest.json">运行清单</a> · <a href="template_audit.json">模板审计</a><div id="summary"></div><p class="warn">工程通过不等于假设得到支持。5%只是值得关注的经验幅度；单窗口收益不能称稳定收益。GT无跨帧粒子ID，较少运动端不是固定端。</p></div>
<div class="bar"><select id="window"></select><select id="model"><option>M0</option><option>M1</option><option>M2</option><option>M3</option></select><select id="checkpoint"><option value="lowest_cd">最低CD</option><option value="last_400">第400次</option></select><button id="prev">上一帧</button><button id="play">播放</button><button id="next">下一帧</button><input id="frame" type="range" min="0" max="11" value="1"><b id="label"></b><br><select id="mode"><option value="centers">Gaussian centers</option><option value="ellipsoids">1σ ellipsoids</option><option value="trajectories">预测轨迹</option></select><select id="view"><option value="orbit">拖拽视角</option><option value="xy">XY</option><option value="xz">XZ</option><option value="yz">YZ</option></select><select id="render"><option value="rgb">RGB</option><option value="alpha">Alpha</option></select><span id="status"></span></div>
<div id="metadata" class="box"></div><div class="grid box"><div>GT / 真实观测<canvas id="g0"></canvas></div><div>模型预测 / 历史观测<canvas id="g1"></canvas></div><div>复制 GS_t<canvas id="g2"></canvas></div></div>
<div id="images" class="box"></div><div class="box"><h3>逐 horizon 指标（长度 mm）</h3><div id="metrics"></div><h3>结构与形态曲线</h3><div id="structplot"></div><h3>400次优化曲线：平均 CD</h3><div id="curve"></div></div>
<div class="box"><h2>051 frame132/133 模板审计</h2><div id="templateinfo"></div><p>以下将参考模板平移到目标集合质心，仅作回顾性形态诊断，绝不是模型预测。真实 frame133 没有用于预测 frame133。</p><select id="templateframe"></select><select id="templatecamera"><option value="0">cam001</option><option value="1">cam012（有底边裁切）</option><option value="2">cam021</option></select><select id="templatemode"><option value="rgb">RGB</option><option value="alpha">Alpha</option></select><div class="grid"><div>GT<img id="tg"></div><div>frame132 对齐<img id="t132"></div><div>frame133 对齐<img id="t133"></div></div><h3>原始 RGB 与物体 mask（点击图像可放大）</h3><select id="sourcemode"><option value="overlay">mask叠加</option><option value="source">原始RGB</option><option value="mask">物体mask</option></select><div class="grid"><div>frame132<img id="s132"></div><div>frame133<img id="s133"></div><div>当前目标<img id="st"></div></div><small>观察右下方局部厚度和边缘时须同时核对原图、掩码与多视角；GS形态差异不能全部归因于噪声。完全缺失的模板区域不保证可由有限尺度/透明度修复。</small></div>
<script src="data/index.js"></script><script>
const $=id=>document.getElementById(id),I=window.P3_INDEX;let D=null,request=0,yaw=.4,pitch=.5,zoom=1,drag=null,timer=null;
const windows=[...new Set(I.bundles.map(b=>`${b.scene}|${b.episode}|${b.start}`))];for(const w of windows){let o=document.createElement('option');o.value=w;o.textContent=w.replaceAll('|',' · ');$('window').appendChild(o)}
function table(headers,rows){return '<table><tr>'+headers.map(s=>`<th>${s}</th>`).join('')+'</tr>'+rows.map(r=>'<tr>'+r.map(s=>`<td>${s}</td>`).join('')+'</tr>').join('')+'</table>'}
const fmt=v=>typeof v==='number'?v.toFixed(4):v;
$('summary').innerHTML=table(['Scene','模型','checkpoint/更新','平均CD','h1','h3','h5','h10','RGB','Alpha'],I.results.runs.flatMap(r=>['lowest_cd','last_400'].map(tag=>{let c=r.checkpoints[tag],m=c.metrics.model;return [r.scene,r.variant,tag+'/'+c.update,m.mean_cd_mm,...[0,2,4,9].map(h=>m.cd_mm[h]),m.mean_rgb,m.mean_alpha].map(fmt)})));
function plot(series,labels){let all=series.flat(),max=Math.max(...all,1e-8),min=Math.min(0,...all),range=max-min||1,colors=['#2979b8','#df7b2a','#248655'];return '<svg viewBox="0 0 800 160">'+series.map((v,k)=>`<polyline fill="none" stroke="${colors[k%3]}" stroke-width="2" points="${v.map((y,i)=>`${30+i/Math.max(1,v.length-1)*730},${130-(y-min)/range*110}`).join(' ')}"/><text x="${30+k*245}" y="155" fill="${colors[k%3]}">${labels[k]} (${Math.min(...v).toFixed(3)}–${Math.max(...v).toFixed(3)})</text>`).join('')+'</svg>'}
function load(){let seq=++request;const [scene,ep,start]=$('window').value.split('|');const entry=I.bundles.find(b=>b.scene===scene&&b.episode===+ep&&b.start===+start&&b.variant===$('model').value&&b.checkpoint===$('checkpoint').value);$('status').textContent='加载…';const apply=()=>{if(seq!==request)return;D=window.P3_DATA[entry.id];$('status').textContent='';zoom=1;update()};if(window.P3_DATA&&window.P3_DATA[entry.id])return apply();let s=document.createElement('script');s.src=entry.path;s.onload=apply;s.onerror=()=>{$('status').textContent='资源加载失败';};document.head.appendChild(s)}
function bounds(){let lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];for(const f of [D.previous,D.geometry.initial,...D.geometry.prediction,...D.geometry.target])for(const p of f.centers)p.forEach((v,j)=>{lo[j]=Math.min(lo[j],v);hi[j]=Math.max(hi[j],v)});return [lo.map((v,j)=>(v+hi[j])/2),Math.max(...hi.map((v,j)=>v-lo[j]),.001)]}
function draw(id,f,color,isPrediction){const c=$(id),rect=c.getBoundingClientRect();c.width=Math.round(rect.width*devicePixelRatio);c.height=Math.round(rect.height*devicePixelRatio);const ctx=c.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,c.width,c.height);const [center,radius]=bounds(),scale=Math.min(c.width,c.height)*.85/radius*zoom;
function project(p){let [x,y,z]=p.map((v,j)=>v-center[j]),a,b;switch($('view').value){case'xy':a=x;b=y;break;case'xz':a=x;b=z;break;case'yz':a=y;b=z;break;default:a=Math.cos(yaw)*x-Math.sin(yaw)*y;b=Math.cos(pitch)*z-Math.sin(pitch)*(Math.sin(yaw)*x+Math.cos(yaw)*y)}return[c.width/2+a*scale,c.height/2-b*scale]}
function line(points){ctx.beginPath();points.forEach((p,i)=>{let [x,y]=project(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}
ctx.fillStyle=color;ctx.strokeStyle=color;ctx.lineWidth=.7*devicePixelRatio;
if($('mode').value==='ellipsoids'){ctx.globalAlpha=.3;f.ids.forEach((idx,j)=>{for(const [a,b] of [[0,1],[0,2],[1,2]]){let pts=[];for(let s=0;s<=16;s++){let angle=s*Math.PI/8;pts.push(f.centers[idx].map((v,k)=>v+Math.cos(angle)*f.axes[j][a][k]+Math.sin(angle)*f.axes[j][b][k]))}line(pts)}});ctx.globalAlpha=1}else{for(const p of f.centers){let [x,y]=project(p);ctx.fillRect(x,y,1.5*devicePixelRatio,1.5*devicePixelRatio)}if(isPrediction&&$('mode').value==='trajectories'&&+$('frame').value>1){ctx.strokeStyle='#db8428';for(const i of D.geometry.initial.ids)line([D.geometry.initial.centers[i],...D.geometry.prediction.slice(0,+$('frame').value-1).map(p=>p.centers[i])])}}}
function update(){if(!D)return;let s=+$('frame').value,h=s-1;const observed=s<2,initial=D.geometry.initial;const gt=s===0?D.previous:s===1?initial:D.geometry.target[h-1],pred=s===0?D.previous:s===1?initial:D.geometry.prediction[h-1];$('label').textContent=`frame ${D.frames[s]} · ${s===0?'t-1 观测':s===1?'t 观测':'h='+h}`;$('metadata').textContent=`${D.scene} ep${D.episode} start${D.start} · ${D.variant} · ${D.checkpoint} update ${D.updates} · ${D.commit.slice(0,8)} · history=${D.config.history} · refinement=${D.config.template_refinement} · 两帧独立点数 ${D.previous.centers.length}/${initial.centers.length} · 椭球/轨迹显示固定最多300点，其余centers全量。${s===0&&D.config.history===1?'t-1仅作诊断展示，本模型不读取它。':''}`;
draw('g0',gt,'#248655',false);draw('g1',pred,'#2979b8',!observed);draw('g2',initial,'#98a3ad',false);
$('images').innerHTML=[0,1,2].map(v=>'<h4>'+D.config.depth_cameras[v]+'</h4><div class="grid">'+['gt','pred','copy'].map(name=>{let src=(s===0&&name!=='copy')?`${D.history_images}/view${v}_${$('render').value}.png`:`${D.images}/frame_${Math.max(h,0)}_view_${v}_${$('render').value}_${name}.png`;return `<div>${name}<img src="${src}"></div>`}).join('')+'</div>').join('');
$('metrics').innerHTML=table(['h/frame','CD','去质心CD','RGB','Alpha','运动端','较少运动端','邻域差','铺展p95','平移RMS','质心误差','scale reg'],D.metrics.structure.rows.map((r,i)=>[(i+1)+'/'+r.frame+( [1,3,5,10].includes(i+1)?' ★':''),r.cd_mm,r.centered_cd_mm,D.metrics.rgb[i],D.metrics.alpha[i],r.moving_end_error_mm,r.lesser_end_error_mm,r.neighbor_rms_mm,r.expansion_p95_mm,r.translation_rms_mm,r.centroid_error_mm,r.scale_regularization].map(fmt)));
$('structplot').innerHTML=plot([D.metrics.structure.rows.map(r=>r.expansion_p95_mm),D.metrics.structure.rows.map(r=>r.translation_rms_mm),D.metrics.structure.rows.map(r=>r.centered_cd_mm)],['局部半径增量p95','平移一致性RMS','去质心CD'])+table(['h','尺度比 min/p5/median/p95/max','opacity偏移 min/p5/median/p95/max'],D.metrics.structure.rows.map(r=>[r.horizon,r.scale_ratio_quantiles.map(fmt).join(' / '),r.opacity_delta_quantiles.map(fmt).join(' / ')]));$('curve').innerHTML=plot([D.curve.map(r=>r.cd)],['20…400次 meanCD']);}
function step(delta){$('frame').value=(+$('frame').value+delta+12)%12;update()}
for(const id of ['window','model','checkpoint'])$(id).onchange=load;
$('frame').oninput=update;for(const id of ['mode','view','render'])$(id).onchange=update;window.onresize=update;$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放'}else{timer=setInterval(()=>step(1),500);$('play').textContent='暂停'}};
document.querySelectorAll('canvas').forEach(c=>{c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=c.onpointercancel=()=>drag=null;c.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.01;pitch+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];$('view').value='orbit';update()}};c.onwheel=e=>{e.preventDefault();zoom=Math.min(20,Math.max(.1,zoom*Math.exp(-e.deltaY*.001)));update()}});
for(let f=134;f<=142;f++){let o=document.createElement('option');o.value=f;o.textContent='共同目标 frame '+f;$('templateframe').appendChild(o)}
$('templateinfo').innerHTML=`132→133 原始 CD ${I.template.pair_raw_cd_mm.toFixed(4)} mm；去质心 CD ${I.template.pair_centered_cd_mm.toFixed(4)} mm。`+table(['reference','target','原始CD','去质心CD'],I.template.references.map(r=>[r.reference,r.target,fmt(r.raw_cd_mm),fmt(r.centered_cd_mm)]));
$('templateinfo').innerHTML+='<h3>M2/M3：共同目标134–142，输入保持131/132</h3>'+table(['模型/checkpoint','target','原始CD','去质心CD','RGB','Alpha','去质心X四分区CD'],I.template.model_common_targets.map(r=>[r.model+'/'+r.checkpoint,r.target,fmt(r.raw_cd_mm),fmt(r.centered_cd_mm),fmt(r.rgb),fmt(r.alpha),r.local_shape_cd_mm.map(fmt).join(' / ')]));
function templateUpdate(){const f=+$('templateframe').value,v=$('templatecamera').value,mode=$('templatemode').value,source=$('sourcemode').value;$('tg').src=`M2/051-cube/visualizations/lowest_cd/episode-0-start-132/figures/frame_${f-132}_view_${v}_${mode}_gt.png`;for(const r of [132,133]){$('t'+r).src=`template/ref${r}_to${f}_view${v}_${mode}.png`;$('s'+r).src=`template/frame${r}_view${v}_${source}.png`}$('st').src=`template/frame${f}_view${v}_${source}.png`;}
for(const id of ['templateframe','templatecamera','templatemode','sourcemode'])$(id).onchange=templateUpdate;document.addEventListener('click',e=>{if(e.target.tagName==='IMG')window.open(e.target.src,'_blank')});templateUpdate();load();
</script></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-assets',action='store_true');parser.add_argument('--build',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.prepare_assets:prepare_assets()
    if args.build:build_report()
    if not args.prepare_assets and not args.build:parser.error('Choose --prepare-assets and/or --build')

if __name__=='__main__':main()
