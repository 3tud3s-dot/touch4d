"""Local CPU plots and Chinese report from immutable Smoke/formal CSV evidence."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def build(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    root=Path(root);result=root/'results';fig=root/'figures';fig.mkdir(exist_ok=True)
    summary=rows(result/'val_summary.csv');formal=any(r['mode']=='geometry_best' for r in summary)
    target='geometry_best' if formal else 'geometry_4updates'
    modes=['original',target]+(['geometry_last'] if formal else [])
    equal=[r for r in summary if r['episode']=='episode_equal' and r['mode'] in modes]
    provenance=json.loads((root/'provenance.json').read_text());completion=json.loads((root/'completion.json').read_text())
    fig0,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,key in zip(axes,('rgb','alpha','joint')):
        ax.bar([r['mode'] for r in equal],[float(r[key]) for r in equal]);ax.set_title('Val100 episode-equal '+key);ax.tick_params(axis='x',rotation=15)
    fig0.savefig(fig/'val_render_metrics.png',dpi=160);plt.close(fig0)
    horizons=rows(result/'horizons.csv')
    fig0,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,key in zip(axes,('rgb','alpha','joint')):
        for mode in modes:
            ys=[]
            for h in range(1,16):
                epvalues=[np.mean([float(r[key]) for r in horizons if r['split']=='val' and r['mode']==mode and int(r['episode'])==ep and int(r['horizon'])==h]) for ep in (3,6)]
                ys.append(np.mean(epvalues))
            ax.plot(range(1,16),ys,label=mode)
        ax.set_xlabel('Horizon');ax.set_title(key);ax.legend(fontsize=7)
    fig0.savefig(fig/'horizon_render_metrics.png',dpi=160);plt.close(fig0)
    geometry=rows(result/'geometry_statistics.csv')
    fig0,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for mode in modes[1:]:
        selected=[r for r in geometry if r['split']=='val' and r['mode']==mode]
        for ax,key in zip(axes,('scale_ratio_mean','rotation_deg_mean')):
            ys=[np.mean([float(r[key]) for r in selected if int(r['horizon'])==h]) for h in range(1,16)]
            ax.plot(range(1,16),ys,label=mode);ax.set_title(key+' (window mean)');ax.set_xlabel('Horizon');ax.legend()
    fig0.savefig(fig/'geometry_horizon.png',dpi=160);plt.close(fig0)
    for path in fig.glob('*_attributes.npz'):
        if target not in path.name:continue
        a=np.load(path);keep=a['mask'][0]
        # NumPy advanced indexing can place N before H; use explicit indexing.
        xyz=a['xyz'][0][:,keep];delta=a['log_delta'][0][:,keep];omega=a['rotation_vector'][0][:,keep]
        low=xyz.reshape(-1,3).min(0);high=xyz.reshape(-1,3).max(0);center=(low+high)/2;radius=max(high-low)/2*1.05
        f,ax=plt.subplots(2,2,figsize=(10,9),subplot_kw={'projection':'3d'},constrained_layout=True)
        for column,h in enumerate((5,15)):
            for row,(value,lo,hi,title) in enumerate(((np.exp(delta[h-1]).mean(-1),.8,1.25,'local-axis mean scale ratio'),
                                                     (np.linalg.norm(omega[h-1],axis=-1)*180/np.pi,0,30,'world rotation vector norm (deg)'))):
                chart=ax[row,column];p=xyz[h-1];sc=chart.scatter(*p.T,c=value,vmin=lo,vmax=hi,s=3,cmap='viridis')
                chart.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),
                    title=f'H{h}: {title}',xlabel='world X (m)',ylabel='world Y (m)',zlabel='world Z (m)')
                chart.set_box_aspect((1,1,1));f.colorbar(sc,ax=chart,shrink=.55)
        f.savefig(path.with_suffix('.png'),dpi=150);plt.close(f)
    # Comparison panels contain exactly three full-frame images per row.
    panels=[]
    for ep,start in ((3,135),(3,245),(6,110),(7,105)):
        for h in (5,15):
            for camera in range(3):
                prefix=f'ep{ep}_start{start}'
                names=[f'{prefix}_original_gt_h{h}_camera{camera}_rgb.png',f'{prefix}_original_prediction_h{h}_camera{camera}_rgb.png',f'{prefix}_{target}_prediction_h{h}_camera{camera}_rgb.png']
                if not all((root/'images'/n).exists() for n in names):continue
                first_image=Image.open(root/'images'/names[0]);w,ht=first_image.size;first_image.close()
                canvas=Image.new('RGB',(w*3,(ht+35)*2),'white');draw=ImageDraw.Draw(canvas)
                for row,channel in enumerate(('rgb','alpha')):
                    for col,(name,label) in enumerate(zip(names,('GT reconstructed GS','Frozen original G5',target))):
                        source=root/'images'/name.replace('_rgb.png','_'+channel+'.png')
                        with Image.open(source) as im:canvas.paste(im,(col*w,row*(ht+35)+35))
                        draw.text((col*w+8,row*(ht+35)+8),f'{label} | H{h} camera{camera} {channel}',fill='black')
                dest=fig/f'{prefix}_h{h}_camera{camera}_comparison.png';canvas.save(dest);panels.append(dest.name)
    paired=rows(result/'paired_window_effects.csv')
    if formal:paired=[r for r in paired if r['mode']==target]
    improvements={k:dict(improved=sum(float(r['delta_'+k])<0 for r in paired),worsened=sum(float(r['delta_'+k])>0 for r in paired),
        maximum_regression=max(float(r['delta_'+k]) for r in paired)) for k in ('rgb','alpha','joint')}
    g=[r for r in geometry if r['split']=='val' and r['mode']==target]
    effects=dict(scale_min=min(float(r['scale_ratio_min']) for r in g),scale_max=max(float(r['scale_ratio_max']) for r in g),
        rotation_max_deg=max(float(r['rotation_deg_max']) for r in g),max_scale_saturation=max(float(r['log_scale_saturation_fraction']) for r in g),
        covariance_min_eigenvalue=min(float(r['covariance_min_eigenvalue']) for r in g))
    text=['# Touch4D G5-GEO-v1 '+('八轮 Head-only 验收' if formal else '四次诊断更新验收'),'',
        '本报告只讨论输出端 Gaussian 属性适配。原 G5 中心运动完全冻结，Scale/Rotation 的变化不代表真实材料应变，也不能修复 XYZ 方向或 Anchor 结构失真。','',
        '## 结果与验收','', '| 模式 | Val CD mm | RGB | Alpha | Joint |','|---|---:|---:|---:|---:|']
    for r in equal:text.append('| '+r['mode']+' | '+' | '.join(f'{float(r[k]):.9f}' for k in ('cd_mm','rgb','alpha','joint'))+' |')
    text.extend(['','全部验证统计使用 ep3 55、ep6 45 个窗口并按 Episode 等权。逐相机、逐 Horizon 和全部窗口原始表均保留。','',
        '1. 新头仅改变 log-scale 与 WXYZ 世界左乘旋转。XYZ、Opacity、SH DC 和原 G5 AR 输入保持不变。',
        '2. 更新前 off 与可微 zero 路径经过完整 Val100 等价检查；更新后 off 可回退。',
        '3. XYZ/CD 隔离是硬性要求；真实结果见 baseline_equivalence.csv。',
        '4. Scale/Rotation 的真实 gsplat 梯度和参数更新记录在 Smoke smoke_updates.csv；隐藏层初次零梯度不视为故障。',
        '5. 骨干与原 Checkpoint 哈希前后核验，未修改生产模型、Loss 或 Renderer。','',
        '### 逐窗口正反变化（目标模式相对原 G5）','',json.dumps(improvements,ensure_ascii=False,indent=2),'',
        '### 几何安全边界与观测','',json.dumps(effects,ensure_ascii=False,indent=2),'',
        '四次更新不是收敛性或泛化证据。若 RGB/Alpha/Joint 退化，应明确区分属性隔离工程通过与渲染非退化未通过，不隐藏恶化窗口。','',
        '## 图像与副作用检查','',
        '下列图片使用原相机、原分辨率与统一画幅，每行三张。GT 是未来 Gaussian 独立重建渲染，非原始相机照片。数值与图像不支持把覆盖改善直接称为物理形变改善。','',
        '![Val metrics](figures/val_render_metrics.png)','','![Horizon metrics](figures/horizon_render_metrics.png)','','![Geometry](figures/geometry_horizon.png)',''])
    for name in panels:text.extend([f'### {name}',f'![三相机对照](figures/{name})',''])
    text.extend(['## 溯源与文件','',f'结果目录：`{root.resolve()}`','',
        '- `results/val_summary.csv`：完整 Val100 与分 Episode 指标。',
        '- `results/horizons.csv`、`camera_metrics.csv`：时刻与相机分项。',
        '- `results/paired_window_effects.csv`：全部配对差值。',
        '- `results/geometry_statistics.csv`：尺度、角度、限幅、协方差。',
        '- `provenance.json`、`completion.json`：运行、版本、真实完成状态与哈希。','',
        '```json',json.dumps(completion,ensure_ascii=False,indent=2),'```','',
        '## 尚未成立的结论','',
        '不能宣称本模型修复运动方向、学到真实接触物理、符合中心 Jacobian、保证未知数据不退化。有限限幅也不能代替视觉检查。后续训练收益须依据完整八轮结果；禁止以 Smoke 的微小改善代替正式结果。',''])
    (root/'final_report.md').write_text('\n'.join(text),encoding='utf-8')
    (root/'results'/'report_statistics.json').write_text(json.dumps(dict(paired=improvements,geometry=effects,panels=panels),indent=2),encoding='utf-8')
    print(str(root/'final_report.md'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);build(parser.parse_args().root)
