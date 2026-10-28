import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.experiments.v4_motion_field.visualize import vectors,save

ORACLES=['cd_zero_best','cd_normal_best','joint_zero_best','joint_normal_best']
COLORS=['#8457a5','#c74685','#168b72','#6a9f25']
HAND_COLORS={'left':'#d77a20','right':'#2474ad'}


def read(root,name):return list(csv.DictReader((root/'results'/f'{name}.csv').open(encoding='utf8')))
def series(rows,key):return [float(r[key]) if r[key] else np.nan for r in rows]


def render(root):
    cs=read(root,'controller_motion');os=read(root,'object_motion');ts=read(root,'tool_motion_alignment');dirs=read(root,'signed_direction_comparison')
    figs=root/'figures';figs.mkdir(exist_ok=True)
    for window in ('ep3_start135','ep7_start105'):
        z=dict(np.load(root/'fields'/f'{window}.npz'));hs=np.arange(1,16)
        c=[r for r in cs if r['window']==window and r['finger']=='all'];o=[r for r in os if r['window']==window];t=[r for r in ts if r['window']==window]
        normal=[r for r in o if r['method']=='normal' and r['region']=='all']
        fig,axes=plt.subplots(4,3,figsize=(18,15))
        for axis,k in enumerate('xyz'):
            key=f'mean_{k}_mm'
            for hand,color in HAND_COLORS.items():
                rr=[r for r in c if r['hand']==hand and r['kind']=='cumulative']
                if any(r['count']!='0' for r in rr):axes[0,axis].plot(hs,series(rr,key),color=color,lw=2,label='Controller '+hand)
                rr=[r for r in c if r['hand']==hand and r['kind']=='adjacent']
                if any(r['count']!='0' for r in rr):axes[1,axis].plot(hs,series(rr,key),color=color,lw=2,label='Controller '+hand)
            means=np.array(series(normal,key));axes[0,axis].plot(hs,means,'k-',label='Normal object mean')
            axes[1,axis].plot(hs,np.diff(np.r_[0,means]),'k-',label='Normal object step delta')
            for method,color in zip(ORACLES,COLORS):
                rr=[r for r in o if r['method']==method and r['region']=='all']
                axes[0,axis].scatter([int(r['horizon']) for r in rr],series(rr,key),color=color,s=38,label=method,zorder=5)
            for region,color in [('near','#c97324'),('middle','#507854'),('far','#2c78a8')]:
                rr=[r for r in o if r['method']=='normal' and r['region']==region]
                axes[2,axis].plot(hs,series(rr,key),color=color,label='Normal '+region)
                for method,marker in zip(ORACLES,['o','s','^','x']):
                    rr=[r for r in o if r['method']==method and r['region']==region]
                    axes[2,axis].scatter([int(r['horizon']) for r in rr],series(rr,key),color=color,marker=marker,s=30,label=region+'/'+method)
            for head,color in enumerate(['#2474ad','#d77a20','#168b72','#8457a5']):
                rr=[r for r in t if r['head']==str(head) and r['hand_contribution']=='combined_model_input']
                axes[3,axis].plot(hs,series(rr,key),color=color,label=f'Tool motion head {head}')
            for row,title in enumerate(['Cumulative controller / object','Adjacent observed / object step','Normal regions + Oracle markers','Actual 12D input, anchor mean']):
                axes[row,axis].set_title(f'{title} | {k.upper()}');axes[row,axis].set_ylabel('Signed displacement (mm)');axes[row,axis].axhline(0,color='grey',lw=.6)
                axes[row,axis].set_xticks([1,5,10,15]);axes[row,axis].grid(alpha=.2);axes[row,axis].set_xlabel('Prediction H; condition = start+H-1')
        for row in [0,1,3]:axes[row,0].legend(fontsize=7)
        axes[2,0].legend(fontsize=5,ncol=2)
        fig.suptitle(window+' | Oracle markers only at H5/H10/H15 (independent fits); no Oracle interpolation')
        fig.tight_layout();save(fig,figs/f'{window}_signed_motion_summary.png')
        fig,axes=plt.subplots(4,4,figsize=(18,13))
        for head in range(4):
            for axis,k in enumerate('xyz'):
                for hand,color in [('left','#d77a20'),('right','#2474ad'),('combined_model_input','#222222')]:
                    rr=[r for r in t if r['head']==str(head) and r['hand_contribution']==hand]
                    axes[axis,head].plot(hs,series(rr,f'mean_{k}_mm'),color=color,label=hand)
                axes[axis,head].set_title(f'Head {head}, {k.upper()} motion input');axes[axis,head].set_ylabel('mm');axes[axis,head].axhline(0,color='grey',lw=.5)
            rr=[r for r in t if r['head']==str(head) and r['hand_contribution']=='combined_model_input']
            axes[3,head].plot(hs,series(rr,'null_weight_mean'),label='Null mean')
            axes[3,head].plot(hs,series(rr,'valid_history_mass_mean'),label='Valid history mass')
            axes[3,head].set_ylim(-.03,1.03);axes[3,head].set_title('Attention mass (not contact probability)')
        for ax in axes.flat:ax.grid(alpha=.2);ax.set_xticks([1,5,10,15]);ax.set_xlabel('H / observed condition t+H-1');ax.legend(fontsize=6)
        fig.suptitle(window+' | four heads retained; per-hand contributions, null not renormalized away');fig.tight_layout();save(fig,figs/f'{window}_tool12_alignment.png')
        fig,axes=plt.subplots(3,2,figsize=(14,12))
        for ri,region in enumerate(['near','middle','far']):
            rr=[r for r in o if r['method']=='normal' and r['region']==region]
            axes[ri,0].plot(hs,series(rr,'rms_mm'),'k-',label='Normal RMS')
            axes[ri,0].plot(hs,series(rr,'residual_rms_mm'),'k--',label='Residual around regional mean')
            for method,color in zip(ORACLES,COLORS):
                rr=[r for r in o if r['method']==method and r['region']==region]
                axes[ri,0].scatter([int(r['horizon']) for r in rr],series(rr,'rms_mm'),color=color,label=method)
            for hand,color in HAND_COLORS.items():
                rr=[r for r in dirs if r['window']==window and r['method']=='normal' and r['region']==region and r['hand']==hand and r['finger']=='all' and r['controller_kind']=='cumulative']
                if rr:axes[ri,1].plot([int(r['horizon']) for r in rr],series(rr,'mean_direction_cosine'),color=color,label='Normal vs '+hand)
                for method,marker in zip(ORACLES,['o','s','^','x']):
                    rr=[r for r in dirs if r['window']==window and r['method']==method and r['region']==region and r['hand']==hand and r['finger']=='all' and r['controller_kind']=='cumulative']
                    if rr:axes[ri,1].scatter([int(r['horizon']) for r in rr],series(rr,'mean_direction_cosine'),color=color,marker=marker,label=method+'/'+hand)
            axes[ri,0].set_title(region+' displacement RMS');axes[ri,1].set_title(region+' direction vs each hand (undefined omitted)');axes[ri,1].set_ylim(-1.05,1.05)
        for ax in axes.flat:ax.legend(fontsize=6,ncol=2);ax.grid(alpha=.2);ax.set_xticks([1,5,10,15]);ax.set_xlabel('H')
        fig.tight_layout();save(fig,figs/f'{window}_near_middle_far.png')
        x=z['initial']*1000;positions=[x]
        for h in (5,10,15):
            positions.append(x+z['normal_displacement'][h-1]*1000)
            for m in ORACLES:positions.append(x+z[f'h{h}_{m}']*1000)
        cp=z['controller_world']*1000;valid=z['controller_valid'];positions += [cp[j,valid[j]] for j in range(15)]
        allpos=np.concatenate(positions);lo=allpos.min(0);hi=allpos.max(0);span=hi-lo;limits=list(zip(lo-.06*span-1,hi+.06*span+1))
        fig=plt.figure(figsize=(22,12))
        for ri,h in enumerate((5,10,15)):
            for col,m in enumerate(['normal']+ORACLES):
                ax=fig.add_subplot(3,5,ri*5+col+1,projection='3d');u=z['normal_displacement'][h-1] if m=='normal' else z[f'h{h}_{m}']
                vectors(ax,x,u*1000,limits,True,color='#515861',count=110)
                for hand,name in enumerate(['left','right']):
                    ix=np.arange(hand*30,hand*30+30);ix=ix[valid[0,ix]&valid[h-1,ix]][::3]
                    if len(ix):ax.quiver(*cp[0,ix].T,*(z['controller_cumulative'][h-1,ix]*1000).T,length=1,normalize=False,color=HAND_COLORS[name],linewidth=1.2)
                ax.set_title(f'H{h} {m}',fontsize=9)
        fig.suptitle(window+' | grey = object; orange = left, blue = right controller; world mm, arrows x1')
        fig.tight_layout();save(fig,figs/f'{window}_world_directions.png')
        fig,ax=plt.subplots(figsize=(17,7));ax.axis('off')
        blocks=[('Observed t+H-1\nworld XYZ / pose / tactile',.02,.65),('Processor\n(x-center)/scale',.26,.65),('13D Controller\nXYZ + cumulative A0 + T3',.50,.65),('Condition tokens\nHand + Controller',.75,.65),
                ('Geometry attention\n128 anchors x 60 tools x 4 heads',.75,.25),('12D = alpha x adjacent delta\n+ query128 + anchor history3\n+ previous D3 + flags5',.42,.25),('Motion MLP -> cumulative D\nX = X0 + fixed W D',.08,.25)]
        for text,a,b in blocks:ax.text(a,b,text,transform=ax.transAxes,bbox=dict(boxstyle='round,pad=.6',facecolor='#edf3f8',edgecolor='#849caf'),fontsize=11,va='center')
        for x1,y1,x2,y2 in [(.21,.65,.25,.65),(.45,.65,.49,.65),(.70,.65,.74,.65),(.85,.57,.85,.34),(.74,.25,.70,.25),(.40,.25,.36,.25)]:ax.annotate('',xy=(x2,y2),xytext=(x1,y1),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
        ax.text(.04,.02,'H1: adjacent motion/history zero. H2..15: observed backward difference, not future command.\nNo identity between Controller and object displacement is assumed; mean tool motion may cancel across anchors.',transform=ax.transAxes,fontsize=11)
        ax.set_title(window+' | actual production data path (numeric norms: results/pathway_values.csv)',fontsize=15)
        save(fig,figs/f'{window}_condition_pathway.png')
    print('Generated',len(list(figs.glob('*.png'))),'figures')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);render(p.parse_args().root)
