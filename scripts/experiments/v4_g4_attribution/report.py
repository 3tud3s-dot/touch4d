"""Local CPU report and scientific figures from frozen attribution outputs."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from .run import csv_write,ALL_MODES

COLORS={'Normal':'#1468a0','NoGlobal':'#dc7430','NoLeft':'#49a070','NoRight':'#b1448d','G0':'#777777','Copy':'#bba433'}


def retained(base,normal,disabled):
    gain=base-normal
    return None if abs(gain)<1e-12 else 100*(base-disabled)/gain


def group_score(summary,mode,group):
    if group=='all':return summary[mode]
    r=dict(summary[mode]['episodes'][str(group)])
    r['objective']=r['supervised_geometry']+r['rgb']+.1*r['alpha']
    return r


def aggregate_curve(rows,mode,metric):
    result=[]
    for h in range(1,16):
        per=[np.mean([r[metric] for r in rows if r['mode']==mode and r['episode']==ep and r['horizon']==h]) for ep in (3,6)]
        result.append(float(np.mean(per)))
    return result


def figures(root,horizons,direct,local):
    folder=root/'figures';folder.mkdir(exist_ok=True)
    val=[r for r in horizons if r['episode'] in (3,6)]
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    for ax,metric,title in zip(axs.flat,('cd_mm','rgb','alpha','change_from_normal_rms_mm'),
                             ('Val100 CD (mm)','Val100 RGB MAE','Val100 Alpha MAE','Val100 change vs Normal (mm)')):
        for m in ALL_MODES:ax.plot(range(1,16),aggregate_curve(val,m,metric),label=m,color=COLORS[m])
        ax.set(title=title,xlabel='Horizon');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Full validation, episode-equal averaging; frozen checkpoints')
    fig.tight_layout();fig.savefig(folder/'full_val_horizon_curves.png',dpi=150);plt.close(fig)
    for ep,start in ((3,135),(7,105)):
        a=np.load(root/'arrays'/f'ep{ep}_start{start}.npz')
        points=np.concatenate([a[k].reshape(-1,3) for k in list(ALL_MODES)+[f'gt_h{h:02d}' for h in (1,5,10,15)]])*1000
        center=(points.min(0)+points.max(0))/2;radius=float((points.max(0)-points.min(0)).max())*.55
        fig=plt.figure(figsize=(25,14));labels=('GT',)+ALL_MODES
        for row,h in enumerate((1,5,10,15)):
            for col,m in enumerate(labels):
                pos=(a[f'gt_h{h:02d}'] if m=='GT' else a[m][h-1])*1000
                pos=pos[::max(1,len(pos)//5000)]
                ax=fig.add_subplot(4,7,row*7+col+1,projection='3d')
                ax.scatter(*pos.T,s=.45,c=COLORS.get(m,'#234960'),alpha=.7)
                ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),
                       title=f'{m} H{h}',xlabel='X mm',ylabel='Y mm',zlabel='Z mm')
                ax.set_box_aspect((1,1,1));ax.view_init(25,-65);ax.tick_params(labelsize=5)
        fig.suptitle(f'ep{ep}/start{start} | fixed world axes/scale | GT has no material IDs')
        fig.tight_layout();fig.savefig(folder/f'ep{ep}_start{start}_geometry.png',dpi=125);plt.close(fig)
        cases=[r for r in horizons if r['episode']==ep and r['start']==start]
        regions=[r for r in local if r['episode']==ep and r['start']==start]
        fig,axs=plt.subplots(2,3,figsize=(16,8))
        for m in ALL_MODES:
            rows=sorted([r for r in cases if r['mode']==m],key=lambda r:r['horizon'])
            for ax,key in zip(axs[0],('mean_x_mm','mean_y_mm','mean_z_mm')):
                ax.plot(range(1,16),[r[key] for r in rows],label=m,color=COLORS[m])
            for ax,region in zip(axs[1],('near','middle','far')):
                rr=sorted([r for r in regions if r['mode']==m and r['region']==region],key=lambda r:r['horizon'])
                ax.plot(range(1,16),[r['displacement_rms_mm'] for r in rr],label=m,color=COLORS[m])
        for ax,title in zip(axs.flat,('Mean X (mm)','Mean Y (mm)','Mean Z (mm)','Near RMS (mm)','Middle RMS (mm)','Far RMS (mm)')):
            ax.set(title=title,xlabel='Horizon');ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.suptitle(f'ep{ep}/start{start} full AR signed motion and fixed regions')
        fig.tight_layout();fig.savefig(folder/f'ep{ep}_start{start}_signed_regions.png',dpi=150);plt.close(fig)
        fig,axs=plt.subplots(1,3,figsize=(15,4))
        for ax,axis in zip(axs,'xyz'):
            normal={r['horizon']:r[f'mean_{axis}_mm'] for r in cases if r['mode']=='Normal'}
            off={r['horizon']:r[f'mean_{axis}_mm'] for r in cases if r['mode']=='NoGlobal'}
            ax.plot(range(1,16),[normal[h]-off[h] for h in range(1,16)],label='Normal - NoGlobal full AR')
            d=[r for r in direct if r['episode']==ep and r['start']==start and r['mode']=='NoGlobal']
            ax.scatter([r['horizon'] for r in d],[r[f'on_minus_off_{axis}_mm'] for r in d],c='red',marker='x',label='same-history direct only')
            ax.set(title=axis.upper()+' difference (mm)',xlabel='Horizon');ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.suptitle(f'ep{ep}/start{start}: current-step translation is not total AR attribution')
        fig.tight_layout();fig.savefig(folder/f'ep{ep}_start{start}_direct_vs_ar.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    status=json.loads((root/'status.json').read_text());assert status['state']=='COMPLETED'
    summary=json.loads((root/'summary.json').read_text());provenance=json.loads((root/'provenance.json').read_text())
    records={name:json.loads((root/'results'/f'{name}.json').read_text()) for name in ('windows','horizons','direct_effects','local_motion')}
    horizons,direct,local=records['horizons'],records['direct_effects'],records['local_motion']
    figures(root,horizons,direct,local)
    full=[];gains=[]
    for group in ('all',3,6):
        for m in ALL_MODES:
            row=group_score(summary,m,group)
            full.append(dict(group=group,mode=m,cd_mm=row['geometry_mm'],rgb=row['rgb'],alpha=row['alpha'],joint=row['objective']))
        for label,key in [('CD','geometry_mm'),('RGB','rgb'),('Alpha','alpha'),('Joint','objective')]:
            b,n,z=[group_score(summary,m,group)[key] for m in ('G0','Normal','NoGlobal')]
            gains.append(dict(group=group,metric=label,g0=b,normal=n,no_global=z,total_improvement=b-n,
                              remaining_improvement=b-z,conditional_full_ar_branch_improvement=z-n,retained_percent=retained(b,n,z)))
    csv_write(root/'results/full_validation.csv',full);csv_write(root/'results/improvement_retention.csv',gains)
    decompositions=[]
    for ep,start in ((3,135),(7,105)):
        rows=[r for r in horizons if r['episode']==ep and r['start']==start]
        for h in (1,5,10,15):
            for axis in 'xyz':
                means={m:next(r[f'mean_{axis}_mm'] for r in rows if r['mode']==m and r['horizon']==h) for m in ('Normal','NoGlobal','G0')}
                d=next(r[f'on_minus_off_{axis}_mm'] for r in direct if r['episode']==ep and r['start']==start and r['horizon']==h and r['mode']=='NoGlobal')
                total=means['Normal']-means['G0'];branch=means['Normal']-means['NoGlobal'];shared=means['NoGlobal']-means['G0']
                decompositions.append(dict(episode=ep,start=start,horizon=h,axis=axis,total_normal_minus_g0_mm=total,
                    direct_current_mm=d,history_interaction_mm=branch-d,retrained_shared_no_global_vs_g0_mm=shared,
                    full_ar_branch_mm=branch,sum_error_mm=total-(d+(branch-d)+shared)))
    csv_write(root/'results/signed_decomposition.csv',decompositions)
    table='| 条件 | CD mm | RGB | Alpha | Joint |\n|---|---:|---:|---:|---:|\n'
    for r in full:
        if r['group']=='all':table+=f"| {r['mode']} | {r['cd_mm']:.5f} | {r['rgb']:.6f} | {r['alpha']:.6f} | {r['joint']:.6f} |\n"
    gain=next(r for r in gains if r['group']=='all' and r['metric']=='Joint')
    d15=[r for r in decompositions if r['episode']==3 and r['horizon']==15 and r['axis']=='y'][0]
    text=f'''# G4-v1 冻结归因实验

## 范围与原始模型保护

模型基准源码 `{provenance['model_base_sha']}`；本次诊断源码 `{provenance['diagnostic_sha']}`；Slurm Job {provenance['job_id']}。
G4 Best 为 Epoch {provenance['g4_selected_epoch']}，G0 Best 为 Epoch {provenance['g0_selected_epoch']}。
固定两份 checkpoint，完整 Val100（ep3=55、ep6=45），额外仅 Train ep7/start105。
使用原始 CD + RGB + 0.1 Alpha、三相机、有效区域和归一化。完整汇总先按 episode 平均再等权平均。
没有优化器、反向传播或参数更新；参数与 checkpoint 前后哈希一致，见 provenance.json。

## 完整验证集

{table}

相对 G0 的 Joint 改善中，关闭全局分支后保留 **{gain['retained_percent']:.2f}%**。
总改善={gain['total_improvement']:+.8f}；NoGlobal 剩余改善={gain['remaining_improvement']:+.8f}；完整 AR 下分支关闭的条件差异={gain['conditional_full_ar_branch_improvement']:+.8f}。
保留比例不截断至0–100%；若符号相反或有抵消，必须按原值解释。CD/RGB/Alpha 和 ep3/ep6 单独结果见 CSV。

**NoGlobal 不是原始 G0。** 它保留联合训练后的 G4 共享权重，只关闭新增输出。两个指定窗口已验证：将 G4 的共享参数装入原 G0 类可逐位复现 NoGlobal。
因此剩余改善可与重新训练后的共享网络联系，但这不是独立训练的机制因果实验；关闭训练中使用的分支也可能造成分布偏移。

## ep3 H15 有符号 Y 分解

Normal−G0 总变化：{d15['total_normal_minus_g0_mm']:+.5f} mm。
同一 Normal 历史下当前步直接平移：{d15['direct_current_mm']:+.5f} mm。
完整 AR 分支差异减去当前直接平移：{d15['history_interaction_mm']:+.5f} mm。
NoGlobal 与原 G0 的差异：{d15['retrained_shared_no_global_vs_g0_mm']:+.5f} mm。
三项代数相加等于总变化，但历史项包含非线性与上下文交互，不能称为唯一物理或训练因果分配。
单步 H1/H5/H10/H15 在相同历史上重算，并通过仅在该步改写输出的真实 forward 逐位复核。H1 新分支贡献严格为零。

## 逐步与局部证据

- `results/horizons.csv`：全部101窗口、6种条件、H1–H15 的 CD/RGB/Alpha、运动与相对 Normal 变化。
- `results/local_motion.csv`：两个指定窗口的 Near/Middle/Far、整体位移、去全局平移残差和固定16NN失真。
- `results/direct_effects.csv`：相同 Normal AR 历史下，H1/H5/H10/H15 分别去掉全部/左/右新增贡献的直接效应。
- `results/hand_contributions.csv`：所有 G4 条件的 raw/effective 两手贡献、输入有效性、basis 与 global_delta。
- `results/signed_decomposition.csv`：有符号直接、历史交互和重新训练共享网络差异。
- `arrays/`：两个窗口全部预测状态的世界坐标（米）与独立 GT；没有假设 GT 点对应。

Near/Far 为初态到有效 Controller 距离的底/顶四分位，Middle 为余下部分；16NN 变化只是结构代理，不是真实材料应变。
NoLeft/NoRight 只关闭新增分支的对应贡献，不屏蔽原始 Hand/Controller/Tactile、Cross-Attention 或151D Motion Head。
输入 hand_valid 与输出干预不是同一概念，raw 贡献不会被伪装成无效观测。

## 研究边界

本实验衡量冻结模型中的干预响应，不把 Oracle 当标签，也不重新训练消融模型。
不能只凭全局平移改善就宣称动作辨识通过；也不能把完整 AR 变化都归于当前步。
是否支持 G5，应结合 ep7 Far 移动、局部失真以及单双手结果判断；本轮不实现 G5/G6。
'''
    (root/'final_report.md').write_text(text,encoding='utf8')
    figures_list=sorted((root/'figures').glob('*.png'))
    page='<meta charset="utf-8"><title>G4 Frozen Attribution</title><style>body{font:16px sans-serif;max-width:1600px;margin:30px auto;background:#f7f8fa}pre{white-space:pre-wrap}img{width:100%}</style><h1>G4-v1 冻结归因</h1><p><a href="final_report.md">中文报告</a> | <a href="results/full_validation.csv">完整验证集 CSV</a></p><pre>'+html.escape(text)+'</pre>'
    for img in figures_list:
        with Image.open(img) as im:im.verify()
        page+=f'<h3>{img.stem}</h3><img loading="lazy" src="figures/{img.name}">'
    (root/'report.html').write_text(page,encoding='utf8')
    manifest=[dict(path=p.relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('artifact_manifest.json','report_complete.json')]
    (root/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    (root/'report_complete.json').write_text(json.dumps(dict(passed=True,png_count=len(figures_list),source_sha=provenance['diagnostic_sha'])),encoding='utf8')
    print('REPORT COMPLETE',root)


if __name__=='__main__':main()
