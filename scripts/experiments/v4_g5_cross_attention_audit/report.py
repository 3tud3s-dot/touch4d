"""CPU report only; consumes completed raw records, never instantiates a model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .interventions import MODES

COLORS=dict(zip((*MODES,'Copy'),('black','#db4437','#e98e13','#1a73e8','#34a853','#9c27b0','#999999')))


def f(v):return '未定义' if v is None else f'{v:.6f}'


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'


def save(fig,path):
    if path.exists():raise FileExistsError(path)
    fig.savefig(path,dpi=140,bbox_inches='tight');plt.close(fig)


def plots(out,data):
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for mode in (*MODES,'Copy'):
        rows=sorted((r for r in data['horizons_val_summary'] if r['mode']==mode),key=lambda r:r['horizon'])
        for ax,key in zip(axes.flat,('cd_mm','rgb','alpha','joint')):
            ax.plot([r['horizon'] for r in rows],[r[key] for r in rows],label=mode,color=COLORS[mode])
            ax.set(title='Val100 episode-equal '+key,xlabel='Horizon');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);save(fig,out/'figures'/'val100_horizon_metrics.png')
    fig,axes=plt.subplots(1,3,figsize=(17,4),layout='constrained')
    for mode in MODES[1:]:
        rows=sorted((r for r in data['same_history_val_summary'] if r['mode']==mode),key=lambda r:r['horizon'])
        for ax,key in zip(axes,('delta_cd_mm','delta_joint','query_change_feature_rms')):
            ax.plot([r['horizon'] for r in rows],[r[key] for r in rows],'o-',label=mode,color=COLORS[mode])
            ax.set(title='Same Normal history: '+key,xlabel='Horizon');ax.axhline(0,color='gray',lw=.5)
    axes[0].legend(fontsize=7);save(fig,out/'figures'/'same_history_changes.png')
    fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    for layer in (1,2):
        for head in range(4):
            rows=sorted((r for r in data['attention_trace_val_summary'] if r['layer']==layer and r['head']==head and r['region']=='all'),key=lambda r:r['horizon'])
            for ax,key in zip(axes.flat,('update_to_query_ratio','null_mass','attention_entropy_nats','final_layer_tool_vector_rms_mm')):
                if key.startswith('final_layer') and layer!=2:continue
                ax.plot([r['horizon'] for r in rows],[r[key] for r in rows],label=f'L{layer}/H{head}')
                ax.set(title=key,xlabel='Horizon');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);save(fig,out/'figures'/'attention_mechanisms.png')
    generated=[]
    for path in sorted((out/'arrays').glob('*.npz')):
        key=path.stem;split,ep,start=key.split('_');ep=int(ep[2:]);start=int(start[1:])
        selected=[r for r in data['horizons'] if r['split']==split and r['episode']==ep and r['start']==start]
        fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
        for mode in (*MODES,'Copy'):
            rows=sorted((r for r in selected if r['mode']==mode),key=lambda r:r['horizon'])
            for ax,name in zip(axes.flat,('mean_x_mm','mean_y_mm','mean_z_mm','rms_mm','cd_mm','joint')):
                ax.plot([r['horizon'] for r in rows],[r[name] for r in rows],color=COLORS[mode],label=mode)
                ax.set(title=name,xlabel='Horizon');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7);fig.suptitle(key+' / independent complete AR modes')
        dest=out/'figures'/(key+'_curves.png');save(fig,dest);generated.append(dest.relative_to(out).as_posix())
        if (ep,start) not in ((3,135),(3,245),(6,110),(7,105)):continue
        with np.load(path) as arrays:
            clouds=[arrays[m+'_world'].reshape(-1,3) for m in (*MODES,'Copy')]
            clouds += [arrays[f'gt_h{h:02}_world'] for h in range(1,16)]
            lo=np.stack([p.min(0) for p in clouds]).min(0)*1000
            hi=np.stack([p.max(0) for p in clouds]).max(0)*1000
            pad=np.maximum((hi-lo)*.05,2);lo-=pad;hi+=pad
            for h in (1,5,10,15):
                for dims in ((0,2),(0,1)) if h==15 else ((0,2),):
                    plane='XZ' if dims==(0,2) else 'XY'
                    fig,axes=plt.subplots(4,2,figsize=(14,14),layout='constrained')
                    for ax,name in zip(axes.flat,('GT',*MODES,'Copy')):
                        p=(arrays[f'gt_h{h:02}_world'] if name=='GT' else arrays[name+'_world'][h-1])*1000
                        ax.scatter(p[:,dims[0]],p[:,dims[1]],s=.35,color='#336699',alpha=.6,rasterized=True)
                        ax.set(xlim=(lo[dims[0]],hi[dims[0]]),ylim=(lo[dims[1]],hi[dims[1]]),title=name,
                               xlabel='XYZ'[dims[0]]+' mm',ylabel='XYZ'[dims[1]]+' mm')
                        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
                    fig.suptitle(f'{key} H{h} {plane}; same WORLD box across ALL modes/H1-H15; GT independent sets')
                    dest=out/'figures'/f'{key}_h{h:02}_{plane}_geometry.png';save(fig,dest);generated.append(dest.relative_to(out).as_posix())
    return generated


def build(out):
    if not (out/'completed.json').exists():raise RuntimeError('Inference incomplete')
    if (out/'final_report.md').exists():raise FileExistsError('Existing report is protected')
    names=('val_summary','horizons','horizons_val_summary','same_history_val_summary','same_history_regions','attention_trace_val_summary','windows','structure','intervention_checks')
    data={name:json.loads((out/'results'/(name+'.json')).read_text(encoding='utf8')) for name in names}
    p=json.loads((out/'provenance.json').read_text(encoding='utf8'));summary=data['val_summary'];normal=summary['Normal']
    figs=plots(out,data)
    report=['# Touch4D G5-Direct Frozen Cross-Attention Necessity Audit V1\n\n',
        '## 1. 固定对象、协议与运行\n\n',
        f"训练源码 `{p['training_source_sha']}`；诊断源码 `{p['diagnostic_source_sha']}`。\n\n",
        f"Checkpoint：{p['checkpoint']}\n\nSHA256：`{p['checkpoint_sha256']}`\n\n",
        f"Slurm {p['job_id']}，{p['gpu']}，PyTorch/CUDA/gsplat {p['torch']}/{p['cuda']}/{p['gsplat']}；"
        f"耗时{p['elapsed_seconds']/60:.2f}分钟，峰值已分配{p['peak_allocated_bytes']/2**30:.3f}GiB。\n\n",
        '完整Val100（ep3=55，ep6=45），按窗口在Episode内平均、再Episode等权；另只读Train ep7/start105。H15/Δ1、原三相机、原CD+RGB+0.1Alpha。没有训练、优化器、反向传播、改权重或读取测试集。\n\n',
        '## 2. 路径语义与验证\n\n',
        'NoCrossValue只关闭两层projection的加性输出，保留Query残差、LayerNorm、FFN、评分、Alpha；两层随改变后的Query自然重新计算。Tool12仅置零Motion输入[128:140]；Mass4仅置零[146:150]。其余直接条件、18D基底均保留。NoCrossInfo不是No-Action，也不是删除全部Attention。\n\n',
        f"Normal、原模型逐位等价；NoCrossInfo的K/V及Score隔离检查见real_preflight.json。单步独立重算与only-step原模型前缀/当前步核对{len(data['intervention_checks'])}条。模型状态及Checkpoint前后哈希相同。\n\n",
        'A在Normal相同历史上重新计算Cross、Base、Global、Local；B每模式完整H15 AR，两者不混作当前步独立效应。\n\n',
        '## 3. 完整验证结果\n\n',
        table(['条件','CD mm','RGB','Alpha','Joint','ΔCD mm','ΔJoint','Joint变化%'],
              [[mode,f(s['geometry_mm']),f(s['rgb']),f(s['alpha']),f(s['objective']),f(s['geometry_mm']-normal['geometry_mm']),
                f(s['objective']-normal['objective']),f(100*(s['objective']/normal['objective']-1))] for mode,s in summary.items()]),
        'Joint采用归一化CD，不把mm直接加到RGB。变化是固定Checkpoint的效应量，不宣称多Seed统计显著性。\n\n']
    for ep in ('3','6'):
        report+=[f'### ep{ep}\n\n',table(['条件','窗口数','CD mm','RGB','Alpha','Joint'],
            [[mode,s['episodes'][ep]['samples'],f(s['episodes'][ep]['geometry_mm']),f(s['episodes'][ep]['rgb']),f(s['episodes'][ep]['alpha']),
              f(s['episodes'][ep]['supervised_geometry']+s['episodes'][ep]['rgb']+.1*s['episodes'][ep]['alpha'])] for mode,s in summary.items()])]
    report+=['## 4. 相同Normal历史下的单步干预\n\n',
        table(['条件','H','ΔCD mm','ΔRGB','ΔAlpha','ΔJoint','XYZ变化RMS mm','Query变化特征RMS'],
              [[r['mode'],r['horizon'],f(r['delta_cd_mm']),f(r['delta_rgb']),f(r['delta_alpha']),f(r['delta_joint']),f(r['change_rms_mm']),f(r['query_change_feature_rms'])] for r in data['same_history_val_summary']]),
        '区域效应和Base/Global/Local变化见same_history_regions.csv。Query数值是无量纲特征RMS，不能称作mm。区域Pred→GT是一向最近距离，不能作为有GT区域材料对应的误差。\n\n',
        '## 5. 代表案例、正反证据\n\n']
    for split,ep,start in (('val',3,135),('val',3,245),('val',6,110),('train',7,105),('val',3,25),('val',6,20),('val',6,200)):
        selected=[r for r in data['windows'] if (r['split'],r['episode'],r['start'])==(split,ep,start)]
        report += [f'### {split} ep{ep}/start{start}\n\n',table(['条件','CD mm','RGB','Alpha','Joint'],
            [[r['mode'],f(r['cd_mm']),f(r['rgb']),f(r['alpha']),f(r['joint'])] for r in selected])]
    report+=['## 6. Attention机制证据与边界\n\n',
        'attention_trace.csv保留两层、四头、初始Near/Middle/Far：Value投影更新RMS及相对原Query比例、Controller/Null质量、完整softmax熵；另有条件于Controller质量的熵、最终层历史有效质量、每头三维工具运动。质量为零的条件熵标为未定义，不填造零。权重/熵/更新大小均不自动代表有用，更不是物理接触概率。\n\n',
        '当前NoCrossInfo保留FFN和GS邻域、G4/G5直接Action通路，但T3触觉特征通向输出的现有Cross路径也随之关闭；仍计算触觉编码不等于输出能读取它。未来统一Decoder若要保留触觉，须明确提供输入，不能假设当前直接分支已独立读取T3。未来无Attention架构必须另做从头、同预算训练才能判断。冻结删除可能造成分布偏移。11D非XYZ属性继承初态；未来GT图是独立重建GS渲染，非原相机照片。16NN是结构代理，不是真实材料应变。\n\n',
        '## 7. 图像、原始数据和待集中研究判读\n\n',
        'figures/含全部Horizon曲线、机制趋势和固定世界尺度几何；images/含四重点窗H1/H5/H10/H15三相机RGB/Alpha。arrays/保留七窗输入和世界坐标；logs/保留实际命令、测试和Slurm日志。数值自动汇总不是已完成人工图像审查的声明，集中判读需附实际查看范围及九个问题的明确回答。\n\n']
    report += [f'- [{name}]({name})\n' for name in figs]
    (out/'final_report.md').write_text(''.join(report),encoding='utf8')
    (out/'report_complete.json').write_text(json.dumps(dict(state='REPORT_BUILT',figures=figs,parameters_updated=False),indent=2),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    build(Path(parser.parse_args().output).resolve())
