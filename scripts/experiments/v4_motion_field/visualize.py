"""Fixed world axes and arrow scale, both initializations and snapshots."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

METHODS=['normal','cd_zero','cd_normal','joint_zero','joint_normal']
LABELS=['Normal','CD / Zero init','CD / Normal init','Joint / Zero init','Joint / Normal init']
COLORS=['#27699c','#d47921','#6f62a2','#219a7b','#c25779']


def field(z,method,snapshot):return z['normal'] if method=='normal' else z[f'{method}_{snapshot}']


def setup(ax,limits,is3d=False):
    ax.set_xlim(limits[0]);ax.set_xlabel('World X (mm)')
    if is3d:
        ax.set_ylim(limits[1]);ax.set_zlim(limits[2]);ax.set_ylabel('Y (mm)');ax.set_zlabel('Z (mm)')
        ax.set_box_aspect([b-a for a,b in limits]);ax.view_init(24,-64)
    else:
        ax.set_ylim(limits[2]);ax.set_ylabel('World Z (mm)');ax.set_aspect('equal')
    ax.grid(alpha=.16)


def vectors(ax,x,u,limits,is3d=False,color='#27699c',count=160):
    ix=np.linspace(0,len(x)-1,min(count,len(x)),dtype=int)
    setup(ax,limits,is3d)
    if is3d:
        ax.scatter(*x.T,s=1,c='#aeb8c2',alpha=.16)
        ax.quiver(*x[ix].T,*u[ix].T,length=1,normalize=False,arrow_length_ratio=.18,color=color,linewidth=.6)
    else:
        ax.scatter(x[:,0],x[:,2],s=1,c='#aeb8c2',alpha=.22)
        ax.quiver(x[ix,0],x[ix,2],u[ix,0],u[ix,2],angles='xy',scale_units='xy',scale=1,width=.003,color=color)


def heat(ax,x,value,limits,maximum,label,cmap='magma'):
    setup(ax,limits)
    valid=np.isfinite(value)
    ax.scatter(x[~valid,0],x[~valid,2],s=3,c='lightgrey')
    im=ax.scatter(x[valid,0],x[valid,2],c=value[valid],s=4,cmap=cmap,vmin=0,vmax=maximum)
    plt.colorbar(im,ax=ax,fraction=.035,pad=.02,label=label)


def save(fig,p):
    fig.savefig(p,dpi=145,bbox_inches='tight');plt.close(fig)


def render(root,summaries_only=False):
    figs=root/'figures';figs.mkdir(exist_ok=True)
    specs={}
    for window in ('ep7_start105','ep3_start135'):
        zs={h:dict(np.load(root/'fields'/f'{window}_h{h:02}.npz')) for h in (5,10,15)}
        allpos=[];maxdiff=0
        for z in zs.values():
            x=z['initial']*1000;allpos.append(x)
            for snap in ('best','last'):
                for m in METHODS:
                    u=field(z,m,snap)*1000;allpos.append(x+u)
                    if m!='normal':maxdiff=max(maxdiff,float(z[f'{m}_{snap}_difference_mm'].max()))
            # Include displacement control arrows as well: no cropping large controls.
            for k in [k for k in z if k.endswith('_anchor_control')]:allpos.append(z['anchor_initial']*1000+z[k]*1000)
        pts=np.concatenate(allpos);lo=pts.min(0);hi=pts.max(0);span=hi-lo
        limits=list(zip(lo-span*.06-1,hi+span*.06+1));specs[window]=dict(world_axis_limits_mm=[list(v) for v in limits],difference_color_max_mm=maxdiff,angle_color_max_deg=180,arrow_scale=1)
        fig=plt.figure(figsize=(22,12))
        for row,(h,z) in enumerate(zs.items()):
            for col,m in enumerate(METHODS):
                ax=fig.add_subplot(3,5,row*5+col+1,projection='3d');vectors(ax,z['initial']*1000,field(z,m,'best')*1000,limits,True,color=COLORS[col])
                ax.set_title(f'{LABELS[col]} | H{h} | best')
        fig.suptitle(window+' | 3D world motion, actual displacement arrows x1; all four Oracle solutions retained',fontsize=15)
        fig.tight_layout();save(fig,figs/f'{window}_motion_field_summary.png')
        if summaries_only:continue
        for h,z in zs.items():
            x=z['initial']*1000
            for snap in ('best','last'):
                stem=f'{window}_h{h:02}_{snap}'
                fig=plt.figure(figsize=(22,13))
                for col,m in enumerate(METHODS):
                    ax=fig.add_subplot(3,5,col+1,projection='3d');vectors(ax,x,field(z,m,snap)*1000,limits,True,COLORS[col]);ax.set_title(LABELS[col])
                    ax=fig.add_subplot(3,5,col+6)
                    val=np.zeros(len(x)) if m=='normal' else z[f'{m}_{snap}_difference_mm']
                    heat(ax,x,val,limits,maxdiff,'Difference (mm)')
                    ax=fig.add_subplot(3,5,col+11)
                    val=np.full(len(x),np.nan) if m=='normal' else z[f'{m}_{snap}_angle_deg']
                    heat(ax,x,val,limits,180,'Angle (deg)','viridis')
                fig.suptitle(stem+' | 3D vectors / Normal-Oracle difference / angle (grey = undefined)',fontsize=15)
                fig.tight_layout();save(fig,figs/f'{stem}_fields.png')
                fig,axes=plt.subplots(2,5,figsize=(20,9))
                for col,m in enumerate(METHODS):
                    d=z['normal_anchor_control'] if m=='normal' else z[f'{m}_{snap}_anchor_control']
                    vectors(axes[0,col],z['anchor_initial']*1000,d*1000,limits,color=COLORS[col],count=128)
                    axes[0,col].set_title(LABELS[col]+' | D control (not physical FPS motion)')
                    vectors(axes[1,col],x,field(z,m,snap)*1000,limits,color=COLORS[col])
                    axes[1,col].set_title('Fixed W @ D -> GS displacement')
                fig.suptitle(stem+' | controls vs mapped displacement; arrows x1, same world axes')
                fig.tight_layout();save(fig,figs/f'{stem}_anchors.png')
                fig,axes=plt.subplots(3,4,figsize=(18,12))
                for col,m in enumerate(METHODS[1:]):
                    o=field(z,m,snap)*1000;n=z['normal']*1000
                    for reg,color in [('near','#d47921'),('far','#27699c')]:
                        ix=z[reg]
                        for arr,ls,label in [(n,'--','Normal'),(o,'-',m)]:
                            v=np.sort(np.linalg.norm(arr[ix],axis=1));axes[0,col].plot(v,np.arange(1,len(v)+1)/len(v),ls,color=color,label=f'{reg}/{label}')
                        a=z[f'{m}_{snap}_angle_deg'][ix];a=np.sort(a[np.isfinite(a)])
                        if len(a):axes[1,col].plot(a,np.arange(1,len(a)+1)/len(a),color=color,label=reg)
                        for arr,ls,label in [(n,'--','Normal'),(o,'-',m)]:
                            axes[2,col].plot(['X','Y','Z'],arr[ix].mean(0),ls,marker='o',color=color,label=f'{reg}/{label}')
                    axes[0,col].set_title(LABELS[col+1]);axes[0,col].set_xlabel('Displacement magnitude (mm)');axes[0,col].set_ylabel('ECDF')
                    axes[1,col].set_xlabel('Normal vs Oracle angle (deg), defined only');axes[1,col].set_xlim(0,180);axes[1,col].set_ylabel('ECDF')
                    axes[2,col].set_ylabel('Mean signed displacement (mm)')
                    for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=6)
                # Shared scales across columns, explicitly distributions rather than spatial axes.
                for row in (0,2):
                    low=min(a.get_xlim()[0] for a in axes[row]) if row==0 else min(a.get_ylim()[0] for a in axes[row])
                    high=max(a.get_xlim()[1] for a in axes[row]) if row==0 else max(a.get_ylim()[1] for a in axes[row])
                    for ax in axes[row]:(ax.set_xlim if row==0 else ax.set_ylim)(low,high)
                fig.suptitle(stem+' | fixed Near/Far groups; no pressure/contact inference')
                fig.tight_layout();save(fig,figs/f'{stem}_near_far.png')
        rows=list(csv.DictReader((root/'results/motion_field_comparison.csv').open()))
        fig,axes=plt.subplots(2,3,figsize=(16,8))
        metrics=['difference_rms_mm','direction_angle_median_deg','normal_over_oracle_rms','mean_translation_difference_mm','residual_difference_rms_mm','direction_error_energy_fraction']
        for ax,metric in zip(axes.flat,metrics):
            for col,m in enumerate(METHODS[1:]):
                loss,init=m.split('_')
                for snap,ls in [('best','-'),('last','--')]:
                    rr=[r for r in rows if r['window']==window and r['level']=='gs' and r['oracle']==loss and r['initialization']==init and r['snapshot']==snap]
                    ax.plot([int(r['horizon']) for r in rr],[float(r[metric]) for r in rr],ls,marker='o',color=COLORS[col+1],label=f'{m}/{snap}')
            ax.set_title(metric);ax.set_xticks([5,10,15]);ax.set_xlabel('Saved horizon (no H1-H4 Oracle)');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=6);fig.suptitle(window+' | temporal comparison of independent Oracle fits');fig.tight_layout()
        save(fig,figs/f'{window}_horizon_trends.png')
    (root/'provenance/plot_scales.json').write_text(json.dumps(specs,indent=2),encoding='utf8')
    print('Figures generated:',len(list(figs.glob('*.png'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--summaries-only',action='store_true');a=p.parse_args();render(a.root,a.summaries_only)
