"""CPU-only final tables/plots of completed condition attribution; no inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from scripts.experiments.v4_g5_cross_attention_audit.report import table,save
from scripts.experiments.v4_g5_motion_attribution.run import write_json,csv_write
from .interventions import MODES,KV_MODES,CHANNELS


def f(value):
    return '未定义' if value is None else f'{value:.9g}'


def analyze(windows):
    normal={(r['split'],r['episode'],r['start']):r for r in windows if r['mode']=='Normal'}
    counts=[];counterexamples=[]
    for mode in MODES[1:]:
        val=[r for r in windows if r['split']=='val' and r['mode']==mode]
        for ep in ('all',3,6):
            rows=val if ep=='all' else [r for r in val if r['episode']==ep]
            record=dict(mode=mode,episode=ep,windows=len(rows))
            for key,tol in [('cd_mm',1e-4),('joint',1e-6)]:
                delta=[r[key]-normal[r['split'],r['episode'],r['start']][key] for r in rows]
                record[key+'_improved']=sum(v < -tol for v in delta)
                record[key+'_degraded']=sum(v > tol for v in delta)
                record[key+'_numerical_tie']=sum(abs(v)<=tol for v in delta)
            counts.append(record)
        paired=[dict(r,delta_joint=r['joint']-normal[r['split'],r['episode'],r['start']]['joint'],
                     delta_cd_mm=r['cd_mm']-normal[r['split'],r['episode'],r['start']]['cd_mm']) for r in val]
        ordered=sorted(paired,key=lambda r:r['delta_joint'])
        for label,rows in [('most_improved',ordered[:3]),('most_degraded',ordered[-3:])]:
            counterexamples.extend(dict(r,selection=label) for r in rows)
    return counts,counterexamples


def plots(out,data):
    paths=[];colors=dict(zip((*MODES,'Copy'),plt.get_cmap('tab10').colors))
    def emit(fig,name):
        path=out/'figures'/name
        if path.exists():raise FileExistsError(path)
        save(fig,path);paths.append(path.relative_to(out).as_posix())
    fig,axes=plt.subplots(2,2,figsize=(16,10),layout='constrained')
    for mode in (*MODES,'Copy'):
        rows=sorted((r for r in data['horizons_val_summary'] if r['mode']==mode),key=lambda r:r['horizon'])
        for ax,key in zip(axes.flat,('cd_mm','rgb','alpha','joint')):
            ax.plot([r['horizon'] for r in rows],[r[key] for r in rows],label=mode,color=colors[mode]);ax.set(title='Val100 episode-equal '+key,xlabel='Horizon');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);emit(fig,'val100_horizon_metrics.png')
    for metric in ('delta_joint','change_rms_mm'):
        fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
        for mode,ax in zip(KV_MODES,axes.flat):
            for channel in CHANNELS:
                rows=sorted((r for r in data['kv_attribution_val_summary'] if r['mode']==mode and r['channel']==channel),key=lambda r:r['horizon'])
                ax.plot([r['horizon'] for r in rows],[r[metric] for r in rows],'o-',label=channel)
            ax.axhline(0,color='gray',lw=.5);ax.set(title=mode,xlabel='Horizon',ylabel=metric);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7);emit(fig,'kv_'+metric+'.png')
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    for mode in MODES[1:]:
        for ax,branch in zip(axes,('base','global_','local')):
            rows=sorted((r for r in data['branch_changes_val_summary'] if r['mode']==mode and r['space']=='gs_actual' and r['region']=='all' and r['branch']==branch),key=lambda r:r['horizon'])
            ax.plot([r['horizon'] for r in rows],[r['change_rms_mm'] for r in rows],label=mode,color=colors[mode]);ax.set(title='Full AR branch '+branch,xlabel='Horizon',ylabel='change RMS mm');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);emit(fig,'branch_changes.png')
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for ep in (3,6):
        rows=[r for r in data['condition_activity'] if r['episode']==ep and r['operation_hand']]
        for ax,key in zip(axes.flat,('positive_fraction_valid','local_mean_p95','local_max_p95','coverage_mean')):
            vals=[r[key] for r in rows if r[key] is not None]
            if vals:ax.hist(vals,bins=35,alpha=.5,label=f'ep{ep}',density=True)
            ax.set(title=key,xlabel='input normalized units / ratio',ylabel='window-frame/finger density')
    axes[0,0].legend();emit(fig,'t3_activity_distributions.png')
    for key in ('val_ep3_s135','val_ep3_s245','val_ep6_s110','train_ep7_s105','val_ep3_s25','val_ep6_s20','val_ep6_s200'):
        path=out/'arrays'/f'{key}.npz'
        if not path.exists():continue
        with np.load(path) as source:arrays={k:source[k] for k in source.files}
        modes=['GT',*MODES,'Copy']
        allxyz=np.concatenate([arrays['initial_world'],*[arrays[m+'_world'].reshape(-1,3) for m in (*MODES,'Copy')],*[arrays[f'gt_h{h:02}_world'] for h in (1,5,10,15)]])*1000
        lower=allxyz.min(0);upper=allxyz.max(0);middle=(lower+upper)/2;extent=max(upper-lower)*.55
        for h,axis in [(1,(0,2)),(5,(0,2)),(10,(0,2)),(15,(0,2)),(15,(0,1))]:
            fig,axes=plt.subplots(4,3,figsize=(15,15),layout='constrained')
            for ax,mode in zip(axes.flat,modes):
                points=(arrays[f'gt_h{h:02}_world'] if mode=='GT' else arrays[mode+'_world'][h-1])*1000
                ids=np.linspace(0,len(points)-1,min(len(points),2000),dtype=int)
                ax.scatter(points[ids,axis[0]],points[ids,axis[1]],s=1,color='black' if mode=='GT' else colors[mode])
                ax.set(title=mode,xlim=(middle[axis[0]]-extent,middle[axis[0]]+extent),ylim=(middle[axis[1]]-extent,middle[axis[1]]+extent),
                       xlabel='XYZ'[axis[0]]+' mm',ylabel='XYZ'[axis[1]]+' mm');ax.set_aspect('equal');ax.grid(alpha=.15)
            for ax in list(axes.flat)[len(modes):]:ax.axis('off')
            plane='XYZ'[axis[0]]+'XYZ'[axis[1]];fig.suptitle(f'{key} H{h} world{plane}; SAME scale, independent GT GS, not material strain')
            emit(fig,f'{key}_h{h:02}_geometry_{plane}.png')
    font=ImageFont.truetype(matplotlib.font_manager.findfont('DejaVu Sans'),18)
    for directory in sorted((out/'images').iterdir()):
        if not directory.is_dir():continue
        for camera in range(3):
            for channel in ('rgb','alpha'):
                modes=['GT',*MODES,'Copy'];sheet=Image.new('RGB',(1920,1602),'white');draw=ImageDraw.Draw(sheet)
                draw.text((8,4),f'{directory.name} H15 camera{camera} {channel}; GT = reconstructed GS, NOT photograph',fill='black',font=font)
                for i,mode in enumerate(modes):
                    x=(i%3)*640;y=40+(i//3)*390;draw.text((x+8,y),mode,fill='black',font=font)
                    with Image.open(directory/f'h15_v{camera}_{channel}_{mode}.png') as image:sheet.paste(image.convert('RGB').resize((640,360),Image.Resampling.LANCZOS),(x,y+26))
                path=out/'figures'/f'{directory.name}_h15_v{camera}_{channel}_comparison.png'
                if path.exists():raise FileExistsError(path)
                sheet.save(path);paths.append(path.relative_to(out).as_posix())
    return paths


def build(out):
    out=Path(out)
    if not (out/'completed.json').is_file():raise RuntimeError('Inference not completed')
    if (out/'final_report.md').exists():raise FileExistsError(out/'final_report.md')
    provenance=json.loads((out/'provenance.json').read_text(encoding='utf8'))
    names=('val_summary','windows','horizons_val_summary','kv_attribution_val_summary','branch_changes_val_summary','condition_activity','condition_activity_groups','activity_group_summary')
    data={n:json.loads((out/'results'/f'{n}.json').read_text(encoding='utf8')) for n in names}
    for mode in (*MODES,'Copy'):
        episodes=data['val_summary'][mode]['episodes']
        if {ep:r['samples'] for ep,r in episodes.items()}!={'3':55,'6':45}:raise ValueError('Incomplete validation summary')
        rows=[r for r in data['windows'] if r['split']=='val' and r['mode']==mode]
        if len(rows)!=100 or len({(r['episode'],r['start']) for r in rows})!=100:raise ValueError('Incomplete validation windows')
    counts,examples=analyze(data['windows']);csv_write(out/'results'/'paired_counts.csv',counts);csv_write(out/'results'/'counterexamples.csv',examples)
    write_json(out/'results'/'paired_counts.json',counts);write_json(out/'results'/'counterexamples.json',examples)
    images=plots(out,data);summ=data['val_summary'];normal=summ['Normal'];lines=[]
    lines.extend(['# Touch4D G5 — Frozen Condition Information Attribution V1','',
        '本报告为冻结模型的信息路径干预，不是重训消融或真实物理因果实验。完整 Val100 按 Episode 等权；Train7 只作机制案例。','',
        '## 固定模型、信息流与工程验收','',f"诊断源码 `{provenance['diagnostic_source_sha']}`；训练源码 `{provenance['training_source_sha']}`。",
        f"Slurm Job `{provenance['job_id']}` / Step `{provenance.get('step_id')}`；Checkpoint `{provenance['checkpoint']}`。",
        f"Checkpoint SHA256 `{provenance['checkpoint_sha256']}`；模型状态前后 `{provenance['model_hash_before']}` / `{provenance['model_hash_after']}`。",
        'A0/T3 经同一个13D Controller MLP进入 K/V；Hand 经独立 Encoder 和 Hand Projection 融合。真实坐标还进入几何 Score。',
        'Hand Token、G4每手12D、G5每Anchor每手18D与几何具有绕过 Cross 的直接路径；T3没有独立绕过 Cross 的原始特征路径。',
        'Hand Fusion Off及Direct Hand Token Off关闭的是包含手身份/时间编码的Hand上下文，不是仅去掉原始13D物理状态；Controller身份/时间编码保留。',
        '## 完整 Val100',''])
    rows=[]
    for mode in (*MODES,'Copy'):
        row=summ[mode];rows.append([mode,f(row['geometry_mm']),f(row['rgb']),f(row['alpha']),f(row['objective']),f(row['objective']-normal['objective'])])
    lines.extend([table(['Mode','CD mm','RGB','Alpha','Joint','ΔJoint vs Normal'],rows),'',
        '## 十个研究问题',''])
    questions=[
        ('Q1 当前是否利用 T3 压力数值？','T3PressureZero'),
        ('Q2 Coverage/Age/Validity 是否有额外价值？','T3All6Off'),
        ('Q4 Cross 内 A0 Geometry 是否有独立贡献？','CrossA0GeometryOff'),
        ('Q5 Cross Hand Fusion 是否重要？','CrossHandFusionOff'),
        ('Q6 直接 Hand Token 有多大作用？','DirectHandTokenOff')]
    for question,mode in questions:
        r=summ[mode];delta=r['objective']-normal['objective'];count=next(v for v in counts if v['mode']==mode and v['episode']=='all')
        lines.extend([f'### {question}','',f"{mode}：CD {f(r['geometry_mm'])} mm、Joint {f(r['objective'])}，ΔJoint {f(delta)}；改善/恶化窗口 {count['joint_improved']}/{count['joint_degraded']}。",
            '这只说明冻结路径替换的净效应；变差支持该路径在当前分布下有用，变好表明当前利用方式可能有害，近零不证明信息天生无用。',''])
    p=summ['T3PressureZero']['objective'];a=summ['T3All6Off']['objective']
    lines.extend([f'六维全关与只关压力的 Joint 差为 {f(a-p)}。这是联合质量通道与非线性交互的差异，不能独立拆解 Coverage、Age、Validity 的物理价值。','',
        '### Q3 触觉/几何主要经 Routing 还是 Value？','',
        '以下为相同 Normal 历史的 H15 单步；FixedRouting 逐层固定 Normal Alpha，同时用于原tool12/mass4读出。ChangedKey 仅替换独立Key投影，Normal Value及几何位置保留；第二层Query自然变化。',
        table(['Mode','Channel','ΔCD mm','ΔJoint','GS change RMS mm','Query feature RMS'],
              [[r['mode'],r['channel'],f(r['delta_cd_mm']),f(r['delta_joint']),f(r['change_rms_mm']),f(r['query_change_feature_rms'])]
               for r in data['kv_attribution_val_summary'] if r['mode'] in KV_MODES and r['horizon']==15]),'',
        '不能将三个非线性干预效应相加或换算成独立因果百分比。ChangedKey 同时保留原 QK 与 Pairwise 中的 Key 作用，不是纯 QK 项隔离。','',
        '### Q6 补充：Hand12D 与 Anchor-specific6D','',
        table(['Mode','CD mm','Joint','ΔJoint'],[[m,f(summ[m]['geometry_mm']),f(summ[m]['objective']),f(summ[m]['objective']-normal['objective'])] for m in ('SignedHand12Off','LocalController6Off')]),
        'Hand12Off 重算 G4 Contribution 并向 G5传递一致上下文；Local6Off 在Mapping上下文和3x18矩阵乘法同时置零。它们均不是完全No-Action。','',
        '### Q7 ep3/ep6、高低活动窗口依赖是否不同？','',
        table(['Episode','Mode','CD mm','RGB','Alpha','Joint'],[[ep,m,f(summ[m]['episodes'][ep]['geometry_mm']),f(summ[m]['episodes'][ep]['rgb']),f(summ[m]['episodes'][ep]['alpha']),
            f(summ[m]['episodes'][ep]['supervised_geometry']+summ[m]['episodes'][ep]['rgb']+.1*summ[m]['episodes'][ep]['alpha'])] for ep in ('3','6') for m in MODES]),'',
        '活动分组规则在任何预测之前冻结：有效Taxel中正响应比例，Episode内Q25/Q75；相同分位数为INDETERMINATE，无读数为NO_OBSERVATION。不是接触阈值，重叠窗口/连续帧不独立。',
        '```json',json.dumps(data['condition_activity_groups']['thresholds'],ensure_ascii=False,indent=2),'```','',
        table(['Mode','Activity','CD mm','Joint','Episode counts'],[[r['mode'],r['activity_group'],f(r['cd_mm']),f(r['joint']),str(r['episode_window_counts'])] for r in data['activity_group_summary']]),'',
        '### Q8 哪些消融改善预测？','',
        table(['Mode','CD improve/worse','Joint improve/worse'],[[r['mode'],str(r['cd_mm_improved'])+'/'+str(r['cd_mm_degraded']),str(r['joint_improved'])+'/'+str(r['joint_degraded'])] for r in counts if r['episode']=='all']),
        '完整反例见 results/counterexamples.csv；保留最改善和最恶化各3个窗口，不据此追加实验。','',
        '### Q9 是否足以支持真实接触动力学？','',
        '不足。冻结模型对信息路径的依赖、该路径当前净预测收益、真实物理因果性是不同命题。有效零触觉不等于缺测，压力不是标定接触力；GT为独立重建点集。','',
        '### Q10 对 Object State Z / Action Dynamics / Tactile Fusion 的启示','',
        '应分别保留可观测物体状态、带合法性标志的有符号动作、压力及质量通道，避免将运动基底存在误解为已利用触觉。',
        '本轮K/V反事实可约束后续融合位置假设，但不能直接决定删除某条路径；需要未来严格配对从头训练与留出数据才能检验简化/重构收益。本轮不实施架构。','',
        '## 图像与限制','',
        '世界坐标图统一尺度；RGB/Alpha是原三相机GS渲染，不是真实照片。GS其余11D继承初态；二维厚度或裂隙不能直接当作材料应变。',
        'Near/Middle/Far是初态距离分区，不是接触标签。16NN是结构代理，无GT材料对应。特征RMS无量纲，GS运动RMS用固定Scale换算mm；各分支RMS不可相加。',''])
    lines.extend(f'![{Path(path).stem}]({path})' for path in images)
    lines.extend(['','## 产物与停止','',f'实际结果目录：`{out.resolve()}`。完整原始CSV/JSON、输入分组、逐窗口记录、输入/目标数组、日志和溯源均保留。',
                  '零训练更新；未改模型/Loss、未用Oracle、未使用Test、未推送GitHub。本轮完成后停止。',''])
    with (out/'final_report.md').open('x',encoding='utf8') as stream:stream.write('\n'.join(lines))
    write_json(out/'report_complete.json',dict(state='CPU_REPORT_GENERATED',images=images,questions=10))
    return images


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    build(parser.parse_args().output)
