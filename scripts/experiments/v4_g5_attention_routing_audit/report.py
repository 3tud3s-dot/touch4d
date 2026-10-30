"""CPU-only plots/report on completed frozen routing results; no new inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from scripts.experiments.v4_g5_cross_attention_audit.report import table,f,save
from .routing import MODES

COLORS=dict(zip((*MODES,'Copy'),('black','#db4437','#e98e13','#1a73e8','#34a853','#9c27b0','#a45555','#999999')))


def contact_sheets(out):
    """H15, three cameras RGB/Alpha: original pixels only, no re-rendering.

    Nine fixed conditions, three columns; a 640x360 preview per original1280x720
    image. Full original files at all four selected horizons remain in images/.
    """
    font=ImageFont.truetype(matplotlib.font_manager.findfont('DejaVu Sans'),18)
    generated=[]
    for key in ('val_ep3_s135','val_ep3_s245','val_ep6_s110','train_ep7_s105'):
        for camera in range(3):
            for channel in ('rgb','alpha'):
                sheet=Image.new('RGB',(1920,1212),'white');draw=ImageDraw.Draw(sheet)
                draw.text((8,4),f'{key} H15 camera{camera} {channel}: GT reconstructed GS, frozen inference; original images retained',fill='black',font=font)
                for i,mode in enumerate(('GT',*MODES,'Copy')):
                    x=(i%3)*640;y=40+(i//3)*390
                    draw.text((x+8,y),mode,fill='black',font=font)
                    with Image.open(out/'images'/key/f'h15_v{camera}_{channel}_{mode}.png') as picture:
                        sheet.paste(picture.convert('RGB').resize((640,360),Image.Resampling.LANCZOS),(x,y+26))
                path=out/'figures'/f'{key}_h15_v{camera}_{channel}_comparison.png'
                sheet.save(path);generated.append(path.relative_to(out).as_posix())
    return generated


def plots(out,data):
    generated=[]
    def emit(fig,name):
        path=out/'figures'/name;save(fig,path);generated.append(path.relative_to(out).as_posix())
    fig,axes=plt.subplots(2,2,figsize=(15,9),layout='constrained')
    for mode in (*MODES,'Copy'):
        rows=sorted((r for r in data['horizons_val_summary'] if r['mode']==mode),key=lambda r:r['horizon'])
        for ax,key in zip(axes.flat,('cd_mm','rgb','alpha','joint')):
            ax.plot([r['horizon'] for r in rows],[r[key] for r in rows],label=mode,color=COLORS[mode])
            ax.set(title='Val100 episode-equal '+key,xlabel='Horizon');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);emit(fig,'val100_horizon_metrics.png')
    fig,axes=plt.subplots(1,3,figsize=(17,4),layout='constrained')
    for mode in MODES[1:]:
        rows=sorted((r for r in data['same_history_val_summary'] if r['mode']==mode),key=lambda r:r['horizon'])
        for ax,key in zip(axes,('delta_cd_mm','delta_joint','query_change_feature_rms')):
            ax.plot([r['horizon'] for r in rows],[r[key] for r in rows],'o-',label=mode,color=COLORS[mode]);ax.axhline(0,color='gray',lw=.5)
            ax.set(title='Same Normal history: '+key,xlabel='Horizon')
    axes[0].legend(fontsize=7);emit(fig,'same_history_changes.png')
    fig,axes=plt.subplots(1,3,figsize=(17,4),layout='constrained')
    for mode in MODES:
        for ax,branch in zip(axes,('base','global_','local')):
            rows=sorted((r for r in data['branch_changes_val_summary'] if r['mode']==mode and r['space']=='gs_actual' and r['region']=='all' and r['branch']==branch),key=lambda r:r['horizon'])
            ax.plot([r['horizon'] for r in rows],[r['change_rms_mm'] for r in rows],label=mode,color=COLORS[mode])
            ax.set(title='Full AR vs Normal: '+branch,xlabel='Horizon',ylabel='GS field change RMS mm');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);emit(fig,'val100_branch_changes.png')
    fig,axes=plt.subplots(1,3,figsize=(17,4),layout='constrained')
    for mode in (*MODES,'Copy'):
        for ax,region in zip(axes,('near','middle','far')):
            rows=sorted((r for r in data['structure_val_summary'] if r['mode']==mode and r['region']==region),key=lambda r:r['horizon'])
            ax.plot([r['horizon'] for r in rows],[r['pred_to_gt_mm'] for r in rows],label=mode,color=COLORS[mode])
            ax.set(title='Full AR '+region+' Pred->GT',xlabel='Horizon',ylabel='one-way distance mm');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);emit(fig,'val100_regional_geometry.png')
    for head in range(4):
        fig,axes=plt.subplots(1,3,figsize=(17,4),layout='constrained')
        for mode in MODES:
            rows=sorted((r for r in data['routing_metrics_val_summary'] if r['history']=='full_AR' and r['mode']==mode and r['head']==head and r['layer']==2 and r['region']=='all'),key=lambda r:r['horizon'])
            for ax,key in zip(axes,('total_variation','entropy_route_nats','conditional_distance_route_mm')):
                ax.plot([r['horizon'] for r in rows],[r[key] for r in rows],label=mode,color=COLORS[mode]);ax.set(title=f'L2 Head{head}: '+key,xlabel='Horizon');ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);emit(fig,f'routing_mechanisms_head{head}.png')
    for path in sorted((out/'arrays').glob('*.npz')):
        if path.stem.endswith('_routing'):continue
        key=path.stem;split,ep,start=key.split('_');ep=int(ep[2:]);start=int(start[1:])
        records=[r for r in data['horizons'] if (r['split'],r['episode'],r['start'])==(split,ep,start)]
        fig,axes=plt.subplots(2,3,figsize=(17,8),layout='constrained')
        for mode in (*MODES,'Copy'):
            rows=sorted((r for r in records if r['mode']==mode),key=lambda r:r['horizon'])
            for ax,metric in zip(axes.flat,('mean_x_mm','mean_y_mm','mean_z_mm','rms_mm','cd_mm','joint')):
                ax.plot([r['horizon'] for r in rows],[r[metric] for r in rows],label=mode,color=COLORS[mode]);ax.set(title=metric,xlabel='Horizon');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7);fig.suptitle(key+' independent full AR');emit(fig,key+'_curves.png')
        if (ep,start) not in ((3,135),(3,245),(6,110),(7,105)):continue
        with np.load(path) as arrays:
            clouds=[arrays[m+'_world'].reshape(-1,3) for m in (*MODES,'Copy')]+[arrays[f'gt_h{h:02}_world'] for h in range(1,16)]
            lo=np.stack([p.min(0) for p in clouds]).min(0)*1000;hi=np.stack([p.max(0) for p in clouds]).max(0)*1000
            pad=np.maximum((hi-lo)*.05,2);lo-=pad;hi+=pad
            for h in (1,5,10,15):
                for dims in ((0,2),(0,1)) if h==15 else ((0,2),):
                    plane='XZ' if dims==(0,2) else 'XY';fig,axes=plt.subplots(3,3,figsize=(17,13),layout='constrained')
                    for ax,name in zip(axes.flat,('GT',*MODES,'Copy')):
                        p=(arrays[f'gt_h{h:02}_world'] if name=='GT' else arrays[name+'_world'][h-1])*1000
                        ax.scatter(p[:,dims[0]],p[:,dims[1]],s=.3,color='#336699',alpha=.6,rasterized=True)
                        ax.set(xlim=(lo[dims[0]],hi[dims[0]]),ylim=(lo[dims[1]],hi[dims[1]]),title=name,xlabel='XYZ'[dims[0]]+' mm',ylabel='XYZ'[dims[1]]+' mm')
                        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
                    fig.suptitle(f'{key} H{h} {plane}; same WORLD box all modes/H1-H15; GT independent sets')
                    emit(fig,f'{key}_h{h:02}_{plane}_geometry.png')
        if (ep,start) not in ((3,135),(7,105)):continue
        with np.load(out/'arrays'/(key+'_routing.npz')) as routing:
            with np.load(path) as arrays:keep=arrays['anchor_valid']
            for layer in (1,2):
                for head in range(4):
                    fig,axes=plt.subplots(4,2,figsize=(14,13),layout='constrained')
                    for ax,mode in zip(axes.flat,MODES):
                        matrix=routing[f'{mode}_h15_l{layer}_route'][keep,head]
                        im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
                        ax.axvline(29.5,color='white',lw=.7);ax.axvline(59.5,color='white',lw=.7)
                        ax.set(title=mode,xlabel='Left0..29 / Right30..59 / Null60',ylabel='Initial FPS Anchor valid slot')
                    axes.flat[-1].axis('off');fig.colorbar(im,ax=list(axes.flat),label='alpha_route, fixed0..1',shrink=.7)
                    fig.suptitle(f'{key} H15 layer{layer} Head{head}; current-mode Query/masses, not fixed Normal mass')
                    emit(fig,f'{key}_h15_l{layer}_head{head}_routing.png')
    generated.extend(contact_sheets(out))
    return generated


def build(out):
    if not (out/'completed.json').is_file():raise RuntimeError('Inference incomplete')
    if (out/'final_report.md').exists():raise FileExistsError('Existing report protected')
    names=('val_summary','windows','horizons','horizons_val_summary','same_history_val_summary',
        'same_history_regions_val_summary','branch_changes_val_summary','structure_val_summary',
        'routing_metrics_val_summary','routing_checks','intervention_checks')
    data={name:json.loads((out/'results'/(name+'.json')).read_text(encoding='utf8')) for name in names}
    p=json.loads((out/'provenance.json').read_text(encoding='utf8'));s=data['val_summary'];normal=s['Normal']
    figures=plots(out,data)
    report=['# Touch4D G5 Frozen Attention Routing Audit V1\n\n',
        '## 1. 固定对象、工程执行及严格路径\n\n',
        f"训练源码 `{p['training_source_sha']}`；诊断源码 `{p['diagnostic_source_sha']}`；Slurm `{p['job_id']}`。\n\n",
        f"Checkpoint：{p['checkpoint']}\n\nSHA256：`{p['checkpoint_sha256']}`\n\n",
        f"GPU {p['gpu']}，PyTorch/CUDA/gsplat {p['torch']}/{p['cuda']}/{p['gsplat']}；计时{p['elapsed_seconds']/60:.2f}分钟，峰值已分配{p['peak_allocated_bytes']/2**30:.3f}GiB。\n\n",
        '仅替换两层W_O前的Value聚合。Q/K/V及评分、FFN仍执行，原alpha_learned返回值逐位核对；Motion Head的12D/4D仍读取alpha_learned。所有Value与同一步Normal逐位一致，模型及Checkpoint前后哈希相同。\n\n',
        'MassMatched保留的是当前Query下的Learned Null及手别质量；路由改变第二层Query和后续AR，不能称其他变量完全不变，也不是去掉全部Learned Score。UniformRaw同时改变Null门控和空间分配；UniformGlobalMass还可能改变左右手质量。\n\n',
        '固定完整Val100，ep3=55/ep6=45，Episode等权，H15/Δ1、原三相机CD+RGB+0.1Alpha；另仅Train ep7/start105只读诊断。无测试集、Optimizer、Backward或更新。\n\n',
        '## 2. 完整Val100与分组\n\n',table(['模式','CD mm','RGB','Alpha','Joint','ΔCD mm','ΔJoint'],
            [[mode,f(v['geometry_mm']),f(v['rgb']),f(v['alpha']),f(v['objective']),f(v['geometry_mm']-normal['geometry_mm']),f(v['objective']-normal['objective'])] for mode,v in s.items()])]
    for ep in ('3','6'):
        report += [f'### ep{ep}\n\n',table(['模式','窗口数','CD mm','RGB','Alpha','Joint'],
            [[mode,v['episodes'][ep]['samples'],f(v['episodes'][ep]['geometry_mm']),f(v['episodes'][ep]['rgb']),f(v['episodes'][ep]['alpha']),f(v['episodes'][ep]['supervised_geometry']+v['episodes'][ep]['rgb']+.1*v['episodes'][ep]['alpha'])] for mode,v in s.items()])]
    report += ['## 3. 相同Normal历史的单步干预\n\n',table(['模式','H','ΔCD mm','ΔRGB','ΔAlpha','ΔJoint','GS变化RMS mm','Query变化RMS'],
        [[r['mode'],r['horizon'],f(r['delta_cd_mm']),f(r['delta_rgb']),f(r['delta_alpha']),f(r['delta_joint']),f(r['change_rms_mm']),f(r['query_change_feature_rms'])] for r in data['same_history_val_summary']]),
        f"\n独立only-step原模型前缀/当步逐位检查{len(data['intervention_checks'])}条。所有路由调用质量检查{len(data['routing_checks'])}条，FP32容差2e-6；Null质量要求按模式逐位保留。A重新计算所有依赖Head，不从Normal旧分支直接相减代表路由效果。B完整AR包含历史变化。\n\n",
        '## 4. 指定案例与正反证据\n\n']
    for split,ep,start in (('val',3,135),('val',3,245),('val',6,110),('train',7,105),('val',3,25),('val',6,20),('val',6,200)):
        records=[r for r in data['windows'] if (r['split'],r['episode'],r['start'])==(split,ep,start)]
        report += [f'### {split} ep{ep}/start{start}\n\n',table(['模式','CD mm','RGB','Alpha','Joint'],[[r['mode'],f(r['cd_mm']),f(r['rgb']),f(r['alpha']),f(r['joint'])] for r in records])]
    report += ['## 5. 机制与解释边界\n\n',
        '每层每Head保留Null/Left/Right Mass、熵、Top5一致率、加权距离、TV/JS分布差异、Value变化及Local变化。Top-k仅有效Controller，按节点ID稳定处理并列；Uniform并列排名不表示真实偏好。Controller质量为零时Top-k和条件距离未定义，绝不填造零。\n\n',
        '距离有条件于Controller质量和未归一化两种，不能把Null更多造成的较小未归一化距离解释为更接近物体。W_O和Local指标为整体Head联合量，虽与每Head的路由统计同行，不能当独立Head因果贡献。\n\n',
        'RBF使用原sigma；LearnedGeometry使用实际训练后signedXYZ和平方距离系数，不假设单调递减。Permute只改变有效手内权重与原Value的对应，保留分布、Null和手质量；n≤1退化明确保留，未重排Value。\n\n',
        'GT逐帧独立重建，非材料对应；区域Pred→GT为一向误差，16NN为结构代理。GT图为GS渲染非照片，其他11D继承初态。冻结替换的分布偏移不等于新架构重训结论。本轮未训练或实现新Attention。\n\n',
        '## 6. 图片及集中研究判读\n\n',
        '以下图来自真实完成记录。自动生成图不等于人工视觉复核；最终集中判读须补全九个问题、正反例和实际查看范围。完整JSON/CSV及输入、源码、Checkpoint和作业信息位于results/、arrays/、logs/、provenance.json。\n\n']
    report += [f'- [{name}]({name})\n' for name in figures]
    (out/'final_report.md').write_text(''.join(report),encoding='utf8')
    (out/'report_complete.json').write_text(json.dumps(dict(state='REPORT_BUILT',figures=figures,new_model_inference=False),indent=2),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    build(Path(parser.parse_args().output).resolve())
