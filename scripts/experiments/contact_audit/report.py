"""Local CPU plots/report from frozen evidence. All contact labels are candidates."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
from .audit import dump, runs, windows, project


def infer(rows, obs, quantile=1., near_mm=20., min_run=3):
    """Exploratory rule; neither calibrated contact detector nor annotation GT.

    Background = farthest distance quartile with both tactile streams available.
    Pressure thresholds = per-finger background quantiles, never pressure > 0.
    Spatial gate and run length are explicit sensitivity parameters, not physical
    contact tolerances. No hole filling; missing observations remain uncertain.
    """
    valid = obs['valid_mask'].reshape(len(rows),2,-1)
    pressure = obs['tactile'].reshape(len(rows),2,-1)
    complete = valid.any(-1).all(-1)
    distance = np.array([r['controller_min_mm'] if r['controller_min_mm'] is not None else np.nan for r in rows])
    far_cutoff = np.nanquantile(distance, .75)
    background = complete & (distance >= far_cutoff)
    if background.sum() < 5:
        raise ValueError('Insufficient distant background for candidate noise thresholds')
    thresholds, strong, exceed = [], [], []
    for j in range(2):
        mean = np.array([r[f'finger{j}_mean'] if r[f'finger{j}_mean'] is not None else np.nan for r in rows])
        peak = np.array([r[f'finger{j}_maximum'] if r[f'finger{j}_maximum'] is not None else np.nan for r in rows])
        tm, tp = float(np.quantile(mean[background],quantile)), float(np.quantile(peak[background],quantile))
        cnt = ((pressure[:,j] > tp) & valid[:,j]).sum(-1)
        strong.append((mean > tm) & (peak > tp) & (cnt >= 2))
        exceed.append(cnt)
        thresholds.append(dict(mean=tm, maximum=tp, comparison='strict >',minimum_above_peak_taxels=2))
    tactile = np.any(strong,axis=0)
    sustained = np.zeros(len(rows),bool)
    for a,b,value in runs(tactile.tolist()):
        if value and b-a+1 >= min_run:
            sustained[a:b+1] = True
    labels=[]
    for f,r in enumerate(rows):
        if not complete[f] or not r['right_pose_valid'] or r['controller_valid_count'] != 30:
            label='UNCERTAIN'
        elif sustained[f] and distance[f] <= near_mm:
            label='CONTACT'
        elif not tactile[f] and distance[f] > near_mm:
            label='NO_CONTACT'
        else:
            label='UNCERTAIN'
        labels.append(label)
    return dict(labels=labels, thresholds=thresholds, background_frames=np.flatnonzero(background).tolist(),
        far_background_cutoff_mm=float(far_cutoff),quantile=quantile,near_mm=near_mm,min_run_frames=min_run,
        tactile_strong=np.asarray(strong).T.tolist(),exceed_counts=np.asarray(exceed).T.tolist(),
        tactile_sustained=sustained.tolist())


def intervals(labels, rows):
    return [dict(label=value,start_frame=a,end_frame=b,start_timestamp=rows[a]['timestamp_us'],
                 end_timestamp=rows[b]['timestamp_us'],duration_seconds=(rows[b]['timestamp_us']-rows[a]['timestamp_us'])/1e6,
                 frames=b-a+1,review_status='automatic candidate; pending researcher review',
                 supporting_evidence='See per-frame tactile, exact controller-to-GS NN, opening, velocity and RGB.',
                 uncertainties='GS centers are not surfaces; sparse controllers; sensor noise/age; no manual GT.')
            for a,b,value in runs(labels)]


def build(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    directory=Path(directory)
    if (directory/'report.html').exists():
        raise FileExistsError('Report already exists; preserve previous results')
    rows=json.loads((directory/'frame_evidence.json').read_text())
    geom=json.loads((directory/'geometry.json').read_text())
    cal=json.loads((directory/'calibration.json').read_text())
    manifest=json.loads((directory/'run_manifest.json').read_text())
    obs=np.load(directory/'observations.npz',allow_pickle=False)
    result=infer(rows,obs)
    labels=result['labels']; spans=intervals(labels,rows)
    candidates=[v for v in spans if v['label']=='CONTACT']
    wa=windows(labels)
    envelope = [(a,b) for a,b,v in runs(result['tactile_sustained']) if v]
    for e in wa['windows']:
        t=e['start_frame']
        e['response_onset_overlap']=any(t<a<=t+10 for a,b in envelope)
        e['response_release_overlap']=any(t<b+1<=t+10 for a,b in envelope)
    for key in ['response_onset_overlap','response_release_overlap']:
        wa['counts'][key]=sum(e[key] for e in wa['windows'])
    wa['boundary_note']='onset_overlap/release_overlap count conservative segment edges including missing-data breaks; response_* count tactile-envelope edges, not proven physical contact transitions.'
    visual_path=directory/'visual_review.json'
    visual=json.loads(visual_path.read_text(encoding='utf-8')) if visual_path.exists() else dict(notes=[],inspected_frames_by_camera={})
    inspected=set(f for fs in visual['inspected_frames_by_camera'].values() for f in fs)
    sensitivity=[]
    for quantile in [.95,.99,1.]:
        for distance in [10.,20.,30.]:
            r=infer(rows,obs,quantile,distance)
            sensitivity.append(dict(quantile=quantile,near_mm=distance,thresholds=r['thresholds'],
                intervals=[x for x in intervals(r['labels'],rows) if x['label']=='CONTACT'],
                window_counts=windows(r['labels'])['counts']))
    # Positive runs are explicitly descriptive and must not become contact labels.
    positive_runs={}
    for j in range(2):
        positive=[r[f'finger{j}_positive_count']>0 and r[f'finger{j}_valid_count']>0 for r in rows]
        positive_runs[f'finger{j}']=[dict(start_frame=a,end_frame=b,duration_seconds=rows[b]['time_s']-rows[a]['time_s']) for a,b,v in runs(positive) if v]
    for f,r in enumerate(rows):
        r['candidate_label']=labels[f]
        r['review_status']='pending researcher review'
        r['rgb_support']=('assistant inspected; see visual_review.json; human confirmation pending' if f in inspected else 'available in 3 synchronized real-RGB views; image review pending')
        r['geometry_support']='near center proxy' if r['controller_min_mm'] is not None and r['controller_min_mm']<=20 else 'far center proxy'
        r['tactile_support']='sustained above distant background' if result['tactile_sustained'][f] else 'weak/missing/short response'
        for j in range(2):
            r[f'finger{j}_above_background_peak_count']=result['exceed_counts'][f][j]
            r[f'finger{j}_strong']=result['tactile_strong'][f][j]
        r['boundary_uncertain']=labels[f]=='UNCERTAIN' or any(abs(f-s['start_frame'])<=1 or abs(f-s['end_frame'])<=1 for s in candidates)
    with (directory/'frame_metrics.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(directory/'contact_timeline.json',dict(frames=rows,intervals=spans,interval_convention='closed',
        status='automatic candidate; not contact ground truth',positive_response_runs=positive_runs))
    dump(directory/'candidate_intervals.json',dict(interval_convention='closed',automatic_candidates=candidates,
        reviewed_contact_intervals=[],tactile_response_envelopes=[dict(start_frame=a,end_frame=b,duration_seconds=rows[b]['time_s']-rows[a]['time_s']) for a,b in envelope],rule=result,sensitivity=sensitivity,
        limitations=['Thresholds are exploratory, not validated production values.',
                    'Geometry gate 10/20/30 mm and three-frame persistence are sensitivity conventions, not measured physical contact tolerances.',
                    'Missing tactile frames are UNCERTAIN, never filled between contact intervals.',
                    'Candidate splits caused by missing data do not establish physical release/recontact.']))
    dump(directory/'h10_window_audit.json',wa)
    plots=directory/'plots';plots.mkdir(exist_ok=True)
    keys=directory/'keyframes';keys.mkdir(exist_ok=True)
    t=np.array([r['time_s'] for r in rows]); frames=np.arange(len(rows))
    colors={'CONTACT':'#81c995','NO_CONTACT':'#9ac1e8','UNCERTAIN':'#f8cb75'}
    fig,ax=plt.subplots(6,1,figsize=(15,15),sharex=True)
    for j in range(2):
        ax[0].plot(frames,[r[f'finger{j}_maximum'] for r in rows],label=f'right finger {j} max')
        ax[0].axhline(result['thresholds'][j]['maximum'],ls='--',alpha=.5)
        ax[1].plot(frames,[r[f'finger{j}_mean'] for r in rows],label=f'finger {j} mean')
        ax[4].plot(frames,[r[f'finger{j}_age_ms'] for r in rows],label=f'finger {j} age')
    ax[2].plot(frames,[r['controller_min_mm'] for r in rows],label='minimum NN')
    ax[2].plot(frames,[r['controller_median_mm'] for r in rows],label='median NN')
    ax[2].axhline(20,ls='--',color='grey',label='exploratory 20mm proximity')
    ax[3].plot(frames,[r['opening_m']*1000 for r in rows],label='opening mm')
    ax[3].plot(frames,[r['controller_step_rms_mm'] for r in rows],label='controller step RMS mm')
    ax[5].plot(frames,[r['adjacent_sampled_set_cd_mm'] for r in rows],label='approx independent-set CD mm')
    ax[5].plot(frames,np.linalg.norm(np.array([r['gs_centroid_m'] for r in rows])-rows[0]['gs_centroid_m'],axis=-1)*1000,label='GS centroid change from f0 mm')
    for a in ax:
        for s in spans:a.axvspan(s['start_frame']-.5,s['end_frame']+.5,color=colors[s['label']],alpha=.15)
        a.legend(loc='upper right');a.grid(alpha=.2)
    for a,y in zip(ax,['Raw response (peak)','Raw response (mean)','Controller / GS center mm','mm','Age ms','Set changes mm']):a.set_ylabel(y)
    ax[-1].set_xlabel('Original frame ID; colored labels are automatic candidates')
    fig.suptitle('001-rope ep1 | all 191 frames | right hand only | no physical contact GT')
    fig.tight_layout();fig.savefig(plots/'timeline.png',dpi=130);plt.close(fig)
    vmax=float(np.max(obs['tactile']))
    fig,ax=plt.subplots(2,1,figsize=(15,7),sharex=True)
    for j in range(2):
        a=np.ma.array(obs['tactile'][:,j].reshape(len(rows),384).T,mask=~obs['valid_mask'][:,j].reshape(len(rows),384).T)
        cmap=plt.get_cmap('magma').copy();cmap.set_bad('#b0b0b0')
        im=ax[j].imshow(a,aspect='auto',origin='lower',vmin=0,vmax=vmax,cmap=cmap)
        ax[j].set_ylabel(f'Right finger {j} taxel ID');fig.colorbar(im,ax=ax[j],label='raw response')
    ax[-1].set_xlabel('Original frame ID; grey = invalid, dark = valid zero')
    fig.tight_layout();fig.savefig(plots/'taxel_timeline.png',dpi=130);plt.close(fig)
    # All frames: synchronized original RGB and separate sensor heatmaps.
    cmap=plt.get_cmap('magma')
    heat_paths=[]
    for f in frames:
        canvas=Image.new('RGB',(660,170),'white');draw=ImageDraw.Draw(canvas)
        for j in range(2):
            a=obs['tactile'][f,j];m=obs['valid_mask'][f,j]
            rgb=(cmap(np.clip(a/vmax,0,1))[...,:3]*255).astype(np.uint8);rgb[~m]=[176,176,176]
            tile=Image.fromarray(rgb).resize((320,120),Image.Resampling.NEAREST)
            canvas.paste(tile,(j*330,30));draw.text((j*330+4,4),f'R finger {j}: frame {f}',fill='black')
        draw.text((4,153),f'raw response 0..{vmax:.0f}; grey invalid, black valid zero',fill='black')
        name=f'keyframes/tactile_{f:06d}.png';canvas.save(directory/name);heat_paths.append(name)
    # Filmstrips for independent review across the whole episode, not just selected phases.
    keyids=sorted(set([0,20,35,45,55,65,70,74,76,80,90,110,130,136,140,145,150,160,175,190]+
                       [max(0,min(len(rows)-1,s[k]+d)) for s in candidates for k in ['start_frame','end_frame'] for d in [-2,0,2]]))
    for camera in cal:
        sheet=Image.new('RGB',(960,200*((len(keyids)+2)//3)),'white');draw=ImageDraw.Draw(sheet)
        for idx,f in enumerate(keyids):
            img=Image.open(directory/'rgb'/camera/f'{f:06d}.jpg').resize((320,180))
            x=(idx%3)*320;y=(idx//3)*200;sheet.paste(img,(x,y))
            draw.text((x+4,y+181),f'f{f} {rows[f]["time_s"]:.3f}s {labels[f]}',fill='black')
        sheet.save(keys/f'filmstrip_{camera}.jpg',quality=90)
    # Projection diagnostic on one camera, kept separate from unmodified RGB.
    for f in keyids:
        camera=next(iter(cal));c=cal[camera]
        img=Image.open(directory/'rgb'/camera/f'{f:06d}.jpg').copy();draw=ImageDraw.Draw(img)
        q=np.asarray(geom[f]['controller']);uv,ok=project(q,np.array(c['intrinsics']),np.array(c['camera_to_world']))
        for i,((u,v),valid) in enumerate(zip(uv,ok)):
            if valid and geom[f]['controller_mask'][i] and 0<=u<c['width'] and 0<=v<c['height']:
                u*=.5;v*=.5;draw.ellipse((u-2,v-2,u+2,v+2),fill='cyan' if i<15 else 'orange')
        img.save(keys/f'projection_{f:06d}.jpg',quality=90)
    contacts=', '.join(f"{s['start_frame']}–{s['end_frame']} ({s['duration_seconds']:.3f}s)" for s in candidates) or '无'
    summary=f'''# 001-rope / episode 1 接触证据审计

**自动候选，待研究人员人工确认；不是接触 GT，也未写入 Dataset。**

- No explicit contact labels found in inspected sources. 检查范围见 `label_source_inventory.json`：原始场景 metadata、processed episode metadata/split/robot、重构 metadata/触觉/控制点/时间戳和深度元信息。原始标签是 move center、bimanual=no、nonprehensile=no；processed 74–138 帧属于选段，不能解释成接触标注。
- 全部 {len(rows)} 帧，0–{len(rows)-1}，实际跨度 {t[-1]:.6f}s；右手 slot0，两指，30控制点。触觉为扣 median 后截零的原始响应，未经力标定。有效零压力与缺测分开。
- 当前保守规则产生 **{len(candidates)} 个连续 CONTACT 候选段：{contacts}**。缺测导致的分段不能解释为真实分离／再次接触。
- 整体呈一个主要触觉响应包络 {envelope}；81、99帧分别因一指年龄80.513、76.698ms超限而被保守拆段，不能理解成三次接触。边界71与144/145是候选证据边界，不是精确物理接触/松开时刻。
- 背景取 controller–GS 最近距离最大的四分位、且双指有效的帧，共 {len(result['background_frames'])} 帧，距离下界 {result['far_background_cutoff_mm']:.2f}mm。每指阈值取该背景逐帧均值/峰值最大值，另要求至少2个 taxel 超背景峰值、连续至少3帧。具体阈值和95/99/100%分位、10/20/30mm几何敏感性全部保存。
- CONTACT 需触觉强响应及距 GS center ≤20mm、双指和位姿有效；低响应且距 GS >20mm 为 NO_CONTACT；其余 UNCERTAIN，不填洞。20mm与3帧是待审计的诊断约定，不是已验证物理阈值；GS center不是物体表面。
- 开口、速度、GS独立集合变化和3路真实 undistorted RGB 用于交叉核验，不用于制造材料点轨迹。逐帧 RGB 尚未经研究人员确认，边界存在遮挡、传感器年龄与几何采样误差。
- H10/Δ1 要求闭区间 t…t+10 共11帧：自动候选持续接触窗口 **{wa['counts']['all_candidate_contact']}**，全非接触 {wa['counts']['all_no_contact']}，含 UNCERTAIN {wa['counts']['includes_uncertain']}，跨候选开始 {wa['counts']['onset_overlap']}，跨候选结束 {wa['counts']['release_overlap']}。分类可重叠，不能相加。
- 若只统计主要响应包络的建立/消退边界（不把缺测切口当作新接触），跨建立 **{wa['counts']['response_onset_overlap']}**、跨消退 **{wa['counts']['response_release_overlap']}**。这些也不是已确认的物理事件。
- **人工确认的接触区间=0，当前可正式确认的训练窗口=0。** 上述候选窗口仅供审核，不直接筛选正式训练集。

## 复现和证据

- 原始只读收集 SHA：`{manifest['source_sha']}`；命令见 `run_manifest.json`。
- 本地构建：`python -B -m scripts.experiments.contact_audit.report --directory <审计目录>`。
- `source_integrity.json` 保存读取前后 SHA256；`frame_metrics.csv` 和 `observations.npz` 保存完整时间、位姿/开口/控制点及双指触觉/有效性/年龄。
- `geometry.json` 每帧确定性抽样最多512个GS，仅用于轻量几何查看；原始全部GS用于最近距离、点数/质心/包围盒和尺度统计。相邻集合CD查询抽样512点、参考完整集合，不是粒子位移。
- `report.html` 离线逐帧浏览；`plots/timeline.png`、`plots/taxel_timeline.png`；`keyframes/filmstrip_*.jpg` 全过程关键帧；`rgb/` 3路191帧真实图像。投影使用相机 calibration 的 camera_to_world 逆变换，原图独立保留。
- 不使用GPU、模型、checkpoint、optimizer或训练；未改变任何数据文件。HTML静态资源可核验，浏览器交互验收另行记录，不假称已截图验收。

## 真实图像交叉检查（助手检查，仍待研究人员确认）

{chr(10).join('- '+note for note in visual['notes'])}
'''
    (directory/'audit_summary.md').write_text(summary,encoding='utf-8')
    data=dict(rows=rows,geom=geom,cameras=list(cal),labels=labels,intervals=spans,heat=heat_paths)
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Episode 1 接触审计</title>
<style>body{font:16px system-ui;background:#f5f7fa;color:#172334;max-width:1450px;margin:auto;padding:24px}h1{font-size:26px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}img{max-width:100%}figure{margin:0;background:white;padding:8px}button{padding:8px;margin:3px}canvas{background:white;max-width:100%;border:1px solid #ccc}pre{white-space:pre-wrap}#bar{position:sticky;top:0;background:#fff;padding:10px;z-index:2}a{margin-right:12px}.note{background:#fff1cb;padding:12px}</style>
<h1>001-rope / ep1 · 接触证据审计</h1><p class="note">CONTACT / NO_CONTACT / UNCERTAIN 全为自动候选，未经研究人员确认。有效零压力不是缺测；GS center 不是表面。当前正式确认 H10 窗口：0。</p>
<p><a href="audit_summary.md">中文摘要</a><a href="frame_metrics.csv">完整逐帧指标</a><a href="candidate_intervals.json">规则 / 区间 / 敏感性</a><a href="h10_window_audit.json">H10窗口</a><a href="label_source_inventory.json">标签来源检查</a><a href="source_integrity.json">数据完整性</a></p>
<div id="bar"><button id="prev">上一帧</button><button id="play">播放/暂停</button><button id="next">下一帧</button><input id="slider" type="range" min="0" max="190" value="0" style="width:55%"><strong id="title"></strong><div id="intervals"></div></div>
<div class="grid" id="rgb"></div><p>三路真实 RGB（undistorted），不是 GS 渲染。视频按帧解码，全部相机时间戳与191个观测时间精确一致。</p>
<div class="grid"><figure><canvas id="geometry" width="640" height="400"></canvas><figcaption>GS中心抽样 + 右手控制点；世界XYZ正交投影，拖动旋转。不按GS索引跨帧连线。</figcaption></figure><figure><img id="tactile"><figcaption>双指同色标，灰色=无效，黑色=有效零响应</figcaption></figure><figure><pre id="metrics"></pre></figure></div>
<h2>全时段证据</h2><img src="plots/timeline.png"><img src="plots/taxel_timeline.png"><h2>候选规则与限制</h2><pre id="summary"></pre>
<script>const D=__DATA__;const summary=__SUMMARY__;document.getElementById('summary').textContent=summary;
let f=0,timer=null,yaw=.5,pitch=.3,drag=null;const $=id=>document.getElementById(id),slider=$('slider');slider.max=D.rows.length-1;
$('rgb').innerHTML=D.cameras.map((c,i)=>`<figure><img id="rgb${i}"><figcaption>${c}</figcaption></figure>`).join('');
$('intervals').innerHTML=D.intervals.map(s=>`<button onclick="show(${s.start_frame})">${s.label}: ${s.start_frame}–${s.end_frame}</button>`).join('');
const all=D.geom.flatMap(g=>g.xyz.concat(g.controller.filter((p,i)=>g.controller_mask[i])));const lo=[0,1,2].map(i=>all.reduce((v,p)=>Math.min(v,p[i]),Infinity)),hi=[0,1,2].map(i=>all.reduce((v,p)=>Math.max(v,p[i]),-Infinity));const center=lo.map((v,i)=>(v+hi[i])/2),scale=300/Math.max(...hi.map((v,i)=>v-lo[i]));
function draw(){const cv=$('geometry'),ctx=cv.getContext('2d');ctx.clearRect(0,0,640,400);function xy(p){let [x,y,z]=p.map((v,i)=>v-center[i]);let u=x*Math.cos(yaw)-y*Math.sin(yaw),v=x*Math.sin(yaw)+y*Math.cos(yaw);return[320+u*scale,200-(z*Math.cos(pitch)-v*Math.sin(pitch))*scale]};
for(const p of D.geom[f].xyz){let [x,y]=xy(p);ctx.fillStyle='#627b9b';ctx.fillRect(x,y,2,2)}D.geom[f].controller.forEach((p,i)=>{if(!D.geom[f].controller_mask[i])return;let [x,y]=xy(p);ctx.fillStyle=i<15?'#009db4':'#ee8800';ctx.beginPath();ctx.arc(x,y,3,0,7);ctx.fill()});ctx.fillStyle='#222';ctx.fillText('World metres | fixed bounds across all frames',10,20)}
function show(n){f=Math.max(0,Math.min(D.rows.length-1,n));slider.value=f;let r=D.rows[f];$('title').textContent=`f${f} · ${r.time_s.toFixed(3)} s · ${D.labels[f]}`;D.cameras.forEach((c,i)=>$('rgb'+i).src=`rgb/${c}/${String(f).padStart(6,'0')}.jpg`);$('tactile').src=D.heat[f];$('metrics').textContent=JSON.stringify({timestamp_us:r.timestamp_us,opening_mm:r.opening_m*1000,min_GS_center_mm:r.controller_min_mm,finger0_max:r.finger0_maximum,finger1_max:r.finger1_maximum,finger0_valid:r.finger0_valid_count,finger1_valid:r.finger1_valid_count,finger0_age_ms:r.finger0_age_ms,finger1_age_ms:r.finger1_age_ms,tactile:r.tactile_support,rgb:r.rgb_support,review:r.review_status},null,2);draw()}
slider.oninput=()=>show(+slider.value);$('prev').onclick=()=>show(f-1);$('next').onclick=()=>show(f+1);$('play').onclick=()=>{if(timer){clearTimeout(timer);timer=null}else tick()};function tick(){let next=(f+1)%D.rows.length;let delay=next?Math.max(20,(D.rows[next].time_s-D.rows[f].time_s)*1000):100;show(next);timer=setTimeout(tick,delay)};
$('geometry').onpointerdown=e=>{drag=[e.clientX,e.clientY];e.target.setPointerCapture(e.pointerId)};$('geometry').onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.01;pitch+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];draw()}};$('geometry').onpointerup=()=>drag=null;show(0);
</script></html>'''
    html=html.replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False)).replace('__SUMMARY__',json.dumps(summary,ensure_ascii=False))
    (directory/'report.html').write_text(html,encoding='utf-8')
    dump(directory/'report_manifest.json',dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         keyframes=keyids,thresholds=result['thresholds'],report_device='CPU',human_reviewed=False,html_external_dependencies=False))
    print(json.dumps(dict(candidate_intervals=[(s['start_frame'],s['end_frame']) for s in candidates],windows=wa['counts'],thresholds=result['thresholds'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True)
    build(p.parse_args().directory)
