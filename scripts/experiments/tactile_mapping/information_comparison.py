"""Step 2b: deterministic CPU descriptive comparison; no model, GPU, or GS loads.

python -B -m scripts.experiments.tactile_mapping.information_comparison
Outputs are new and must not overwrite an existing completed audit.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from . import visualize as base

SIGMA = .01
LEVELS = (.25, .5, .75)  # Descriptive contours, NEVER an input/contact mask.
REPS = ('A', 'B', 'C', 'D')
ROI = np.arange(384) % 32 != 31
PIXELS, POINTS = base.local_geometry()
rr, cc = np.meshgrid(np.linspace(0, 11, 6), np.linspace(0, 31, 10), indexing='ij')
DENSE = np.stack((np.full(rr.shape, .007), -.056-.002*cc,
                  (.025/12)*(11.5-rr)), -1).reshape(60, 3)
K60 = np.exp(-cdist(DENSE, PIXELS, 'sqeuclidean')/(2*SIGMA**2))
PARTITION = cdist(PIXELS, POINTS).argmin(1)  # tie -> lowest local CP id
CELL_SIZE = np.array([np.sum(ROI & (PARTITION == i)) for i in range(15)])


def clean(obj):
    if isinstance(obj, dict): return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)): return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray): return clean(obj.tolist())
    if isinstance(obj, (np.floating, float)): return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    return obj


def representations(values, mask, cp_mask, age_us):
    """Mean support is Gaussian; Max/Coverage support is fixed Voronoi cell."""
    v=np.asarray(values,dtype=np.float64).reshape(-1,4,384)
    m=np.asarray(mask,dtype=bool).reshape(v.shape)
    c=np.asarray(cp_mask,dtype=bool).reshape(-1,4,15)
    age=np.asarray(age_us,dtype=np.int64)
    assert age.shape==v.shape[:2] and np.isfinite(v[m]).all() and np.all(v[m]>=0)
    a, a_mask=base.map_values(v,m,c,SIGMA)
    max_values=np.zeros_like(a); max_mask=np.zeros_like(a_mask); coverage=np.zeros_like(a)
    for i in range(15):
        ix=ROI & (PARTITION==i); local=m[...,ix]
        seen=local.any(-1)&c[...,i]
        max_mask[...,i]=seen
        max_values[...,i]=np.where(seen,np.max(np.where(local,v[...,ix],-np.inf),-1),0)
        coverage[...,i]=np.where(c[...,i],local.sum(-1)/CELL_SIZE[i],0)
    age_mask=a_mask & (age[...,None]>=0)&(age[...,None]<=75000)
    age_values=np.where(age_mask,age[...,None],0)
    b=np.stack((a,max_values),-1); bm=np.stack((a_mask,max_mask),-1)
    cv=np.stack((a,max_values,coverage,age_values),-1)
    cm=np.stack((a_mask,max_mask,c,age_mask),-1)
    den=m.astype(float)@K60.T
    d_mask=np.broadcast_to(m.any(-1,keepdims=True)&c.any(-1,keepdims=True),den.shape).copy()
    d=np.where(d_mask,(np.where(m,v,0)@K60.T)/(den+base.EPS),0)
    return {'A':a[...,None],'B':b,'C':cv,'D':d[...,None]}, {'A':a_mask[...,None],'B':bm,'C':cm,'D':d_mask[...,None]}


def contour(values, valid, shape, xyz, fraction):
    """Four-connected superlevel components and pressure-weighted centers."""
    safe=np.where(valid,values,0); peak=safe.max()
    if not valid.any() or peak<=0: return 0,np.empty((0,3))
    ids,n=label((valid&(safe>=fraction*peak)).reshape(shape))
    centers=[]
    for i in range(1,n+1):
        inside=ids.ravel()==i; w=safe[inside]
        centers.append(np.sum(xyz[inside]*w[:,None],0)/w.sum())
    return n,np.asarray(centers).reshape(-1,3)


def peak_location(v,m,q,raw,raw_mask):
    """Lowest-id predicted maximizer; distance to nearest tied RAW maximizer."""
    raw_peak=np.max(np.where(raw_mask,raw,0))
    if raw_peak<=0 or not m.any(): return np.nan
    y=np.where(m,v,-np.inf); y_max=y.max()
    i=np.flatnonzero(m & np.isclose(y,y_max,rtol=1e-12,atol=0))[0]
    gt=PIXELS[raw_mask & np.isclose(raw,raw_peak,rtol=1e-12,atol=0)]
    return float(np.linalg.norm(gt-q[i],axis=-1).min()*1000)


def entropy(v,m,bin_width,overflow):
    """Mean marginal plug-in entropy across coordinates; missing excluded."""
    h=[]
    for j in range(v.shape[1]):
        x=v[:,j][m[:,j]]
        if len(x)==0: continue
        codes=np.floor(np.minimum(np.maximum(x,0),overflow)/bin_width).astype(int)
        _,counts=np.unique(codes,return_counts=True); p=counts/counts.sum()
        h.append(float(-(p*np.log2(p)).sum()))
    return float(np.mean(h)) if h else None


def redundancy(v,m):
    # Compare pairs only on complete frames, excluding constant dimensions.
    x=v[m.all(-1)]
    if len(x)<3:return None
    x=x[:,np.std(x,axis=0)>1e-12]
    if x.shape[1]<2:return None
    matrix=np.corrcoef(x.T)
    return float(np.mean(np.abs(matrix[np.triu_indices(len(matrix),1)])))


def event_comparison(raw,y,valid):
    """Offline per-trace range normalized crossings; no representation thresholding."""
    result={}
    if valid.sum()<3:return result
    def norm(x):
        lo,hi=x[valid].min(),x[valid].max()
        return (x-lo)/(hi-lo) if hi>lo else None
    a,b=norm(raw),norm(y)
    if a is None or b is None:return result
    pair=valid[1:]&valid[:-1]
    for level in (.1,.5,.9):
        offsets=[]; unmatched=0; counts=[]
        for sign in (1,-1):
            i=np.flatnonzero((np.diff((a>=level).astype(int))==sign)&pair)+1
            j=np.flatnonzero((np.diff((b>=level).astype(int))==sign)&pair)+1
            counts.append([len(i),len(j)])
            if len(i) and len(j):
                p,q=linear_sum_assignment(abs(i[:,None]-j[None,:]));offsets.extend((j[q]-i[p]).tolist())
            unmatched+=abs(len(i)-len(j))
        result[str(level)]={'raw_prediction_up_down_counts':counts,'unmatched':unmatched,
                            'matched_offsets_frames':offsets,
                            'median_absolute_offset_frames':np.median(np.abs(offsets)) if offsets else None}
    return result


def finger_metrics(d,f,feat,masks,frame_rows):
    v=d['values'][:,f];m=d['mask'][:,f]
    raw_mean=base.masked_mean(v,m); raw_peak=np.max(np.where(m,v,0),-1)
    good=m.any(-1); ids=np.flatnonzero(good)
    high=good & (raw_mean>=np.quantile(raw_mean[good],.75))
    raw_components={level:[contour(v[i],m[i],(12,32),PIXELS,level) for i in range(len(v))] for level in LEVELS}
    summaries={}; pressure_cache={}
    for rep in REPS:
        pressure_channel=1 if rep in ('B','C') else 0
        y=feat[rep][:,f,:,pressure_channel];o=masks[rep][:,f,:,pressure_channel]
        ymean=base.masked_mean(feat[rep][:,f,:,0],masks[rep][:,f,:,0])
        peak=np.max(np.where(o,y,0),-1);valid=good&o.any(-1)
        points,shape=(DENSE,(6,10)) if rep=='D' else (POINTS,(3,5))
        rows=[]
        for i in ids:
            if not valid[i]:continue
            row=dict(episode=d['episode'],stream=f,frame=int(i),time_s=float(d['seconds'][i]),representation=rep,
                     raw_mean=raw_mean[i],representation_mean=ymean[i],raw_peak=raw_peak[i],representation_peak=peak[i],
                     peak_retention=peak[i]/raw_peak[i] if raw_peak[i]>0 else np.nan,
                     peak_location_error_mm=peak_location(y[i],o[i],points,v[i],m[i]),
                     high_mean_quartile=bool(high[i]))
            for level in LEVELS:
                n0,p0=raw_components[level][i]
                n1,p1=contour(y[i],o[i],shape,points,level)
                loc=np.nan
                if n0 and n1:
                    distances=cdist(p0,p1)*1000;a,b=linear_sum_assignment(distances);loc=distances[a,b].mean()
                suffix=str(level)
                row[f'raw_components_{suffix}']=n0;row[f'components_{suffix}']=n1
                row[f'matched_center_distance_mm_{suffix}']=loc
            rows.append(row);frame_rows.append(row)
        def med(key,subset=rows):
            return float(np.nanmedian([x[key] for x in subset])) if subset else None
        pair=valid[1:]&valid[:-1]
        summary={'valid_frames':int(valid.sum()),'mean_trend_r':base.corr(raw_mean[valid],ymean[valid]),
                 'peak_trend_r':base.corr(raw_peak[valid],peak[valid]),
                 'mean_change_r':base.corr(np.diff(raw_mean)[pair],np.diff(ymean)[pair]),
                 'peak_change_r':base.corr(np.diff(raw_peak)[pair],np.diff(peak)[pair]),
                 'peak_retention_median':med('peak_retention'),
                 'peak_location_error_mm_median':med('peak_location_error_mm'),
                 'high_quartile_peak_retention_median':med('peak_retention',[x for x in rows if x['high_mean_quartile']]),
                 'high_quartile_peak_location_error_mm_median':med('peak_location_error_mm',[x for x in rows if x['high_mean_quartile']]),
                 'contours':{},'mean_events':event_comparison(raw_mean,ymean,valid),
                 'peak_events':event_comparison(raw_peak,peak,valid),
                 'marginal_entropy_bits':{str(b):entropy(y,o,b,256) for b in (1.,4.)},
                 'mean_absolute_intercoordinate_r':redundancy(y,o)}
        for level in LEVELS:
            s=str(level); n0=np.array([r[f'raw_components_{s}'] for r in rows]);n1=np.array([r[f'components_{s}'] for r in rows]);multiple=n0>=2
            h=np.array([r['high_mean_quartile'] for r in rows]);mh=multiple&h
            summary['contours'][s]={'raw_count_mean':float(n0.mean()),'output_count_mean':float(n1.mean()),
                'count_equal_fraction':float((n0==n1).mean()),
                'fewer_components_when_raw_multiple_fraction':float((n1[multiple]<n0[multiple]).mean()) if multiple.any() else None,
                'high_quartile_fewer_components_fraction':float((n1[mh]<n0[mh]).mean()) if mh.any() else None,
                'matched_center_distance_mm_median':med(f'matched_center_distance_mm_{s}')}
        summaries[rep]=summary
        pressure_cache[rep]={'mean':ymean,'peak':peak,'valid':valid}
    cv=feat['C'][:,f,:,2];cm=masks['C'][:,f,:,2]
    av=feat['C'][:,f,:,3];am=masks['C'][:,f,:,3]
    return dict(stream=f,name=base.NAMES[f],raw_valid_frames=int(good.sum()),summaries=summaries,
                quality={'coverage_full_fraction':float((cv[cm]==1).mean()),
                         'coverage_zero_fraction':float((cv[cm]==0).mean()),
                         'coverage_entropy_bits_0_1_bins':entropy(cv,cm,.1,1),
                         'age_entropy_bits_5ms_bins':entropy(av,am,5000,75000),
                         'age_min_max_us':[float(av[am].min()),float(av[am].max())],
                         'age_is_identical_across_15':bool(np.all(av==av[:,:1]))}),pressure_cache


def single_finger(patterns,age=32000):
    p=np.asarray(patterns).reshape(-1,384);v=np.zeros((len(p),4,384));v[:,0]=p
    m=np.zeros(v.shape,bool);m[:,0]=ROI
    cp=np.zeros((len(p),4,15),bool);cp[:,0]=True
    ages=np.full((len(p),4),age,dtype=np.int64)
    return v,m,cp,ages


def synthetic_suite():
    ids=np.flatnonzero(ROI);patterns=np.zeros((len(ids),384));patterns[np.arange(len(ids)),ids]=100
    v,m,cp,age=single_finger(patterns);features,masks=representations(v,m,cp,age)
    singles={}; codes_all={}
    for rep in REPS:
        channel=1 if rep in ('B','C') else 0
        y=features[rep][:,0,:,channel];o=masks[rep][:,0,:,channel];q=DENSE if rep=='D' else POINTS
        errs=[peak_location(y[i],o[i],q,patterns[i],ROI) for i in range(len(ids))]
        pressure=(features[rep][:,0,:,:2] if rep in ('B','C') else features[rep][:,0]).reshape(len(ids),-1)
        quant={}
        for width in (.25,1.,4.):
            codes=np.floor(pressure/width).astype(np.int64)
            _,inv,count=np.unique(codes,axis=0,return_inverse=True,return_counts=True)
            groups=[np.flatnonzero(inv==j) for j in np.flatnonzero(count>=2)]
            example=None
            if groups:
                g=max(groups,key=len);p0,p1=int(g[0]),int(g[-1])
                example={'raw_pixel_ids':[int(ids[p0]),int(ids[p1])],
                         'raw_separation_mm':float(np.linalg.norm(PIXELS[ids[p0]]-PIXELS[ids[p1]])*1000),
                         'feature_max_absolute_difference':float(np.max(abs(pressure[p0]-pressure[p1])))}
            quant[str(width)]={'unique_codes':len(count),'patterns':len(ids),
                               'collision_pairs':int(np.sum(count*(count-1)//2)),
                               'largest_group':int(count.max()),'example':example}
        singles[rep]={'mean_location_error_mm':float(np.mean(errs)),'p95_location_error_mm':float(np.quantile(errs,.95)),
                      'mean_peak_retention':float(np.mean(y.max(-1)/100)), 'quantized_codes':quant}
        codes_all[rep]=pressure
    pair_patterns=[];pair_meta=[]
    for separation in (4,8,12,16,20,24,32):
        dx=separation//2
        for row in (2,6,9):
            for col in range(31-dx):
                p=np.zeros(384);p[row*32+col]=p[row*32+col+dx]=100
                pair_patterns.append(p);pair_meta.append((separation,row,col,col+dx))
    values,mask,cp,age=single_finger(pair_patterns);ff,mm=representations(values,mask,cp,age)
    pairs={}
    for separation in (4,8,12,16,20,24,32):
        ix=[i for i,x in enumerate(pair_meta) if x[0]==separation];pair={}
        for rep in REPS:
            c=1 if rep in ('B','C') else 0; shape=(6,10) if rep=='D' else (3,5);q=DENSE if rep=='D' else POINTS
            pair[rep]={}
            for level in LEVELS:
                n=[contour(ff[rep][i,0,:,c],mm[rep][i,0,:,c],shape,q,level)[0] for i in ix]
                pair[rep][str(level)]={'exactly_two_components_fraction':float(np.mean(np.array(n)==2)),
                                      'mean_count':float(np.mean(n)),'cases':len(n)}
        pairs[str(separation)]=pair
    linear={}
    for name,k in [('A',base.kernel(SIGMA)),('D',K60)]:
        w=k[:,ROI]/k[:,ROI].sum(-1,keepdims=True);s=np.linalg.svd(w,compute_uv=False)
        linear[name]={'matrix_shape':list(w.shape),'rank':int(np.linalg.matrix_rank(w)),
                      'stable_rank':float(np.sum(s*s)/(s[0]*s[0])),
                      'condition_number':float(s[0]/s[-1]),'singular_values':s.tolist()}
    return {'single_impulse_count':len(ids),'single_impulse_amplitude':100,'single_impulses':singles,
            'pair_count':len(pair_meta),'pairs':pairs,'linear_operators':linear}


def numeric_tests():
    checks=base.self_tests()
    def check(name,value):
        assert bool(value),name
        checks.append('2b: '+name)
    patterns=np.zeros((2,384));patterns[0,6*32+14]=100;patterns[1,6*32+14]=patterns[1,6*32+18]=100
    v,m,cp,age=single_finger(patterns);f,o=representations(v,m,cp,age)
    check('A exactly matches prior mapping',np.array_equal(f['A'][...,0],base.map_values(v,m,cp,SIGMA)[0]))
    check('B C mean and max shared',np.array_equal(f['A'][...,0],f['B'][...,0]) and np.array_equal(f['B'],f['C'][...,:2]))
    check('local max not broadcast',np.count_nonzero(f['B'][0,0,:,1])==1 and f['B'][0,0,:,1].max()==100)
    check('two same-cell peaks can merge in max',np.array_equal(f['B'][0,0,:,1],f['B'][1,0,:,1]))
    check('coverage excludes permanent dead column',np.all(f['C'][:,0,:,2]==1))
    check('age replicated not pressure',np.all(f['C'][:,0,:,3]==32000))
    check('no cross finger or hand',all(np.all(x[:,1:]==0) for x in f.values()))
    cell=PARTITION[6*32+14];cut=m.copy();cut[:,0,PARTITION==cell]=False
    cf,co=representations(v,cut,cp,age)
    check('empty max cell invalid but distant mean valid',not co['B'][:,0,cell,1].any() and co['A'][:,0,cell,0].all())
    check('empty coverage zero remains quality valid',np.all(cf['C'][:,0,cell,2]==0) and co['C'][:,0,cell,2].all())
    zz,zo=representations(v*0,m,cp,age)
    check('zero valid for all pressure channels',zo['B'][:,0].all() and np.all(zz['B']==0))
    nn,no=representations(v,np.zeros_like(m),cp,age)
    check('no readings invalid mean max age',not no['C'][...,[0,1,3]].any() and np.all(nn['C']==0))
    constant=np.where(m,30,0);ff,mo=representations(constant,m,cp,age)
    check('60-point constant preserved',np.allclose(ff['D'][:,0],30,atol=1e-12))
    check('partition complete disjoint',CELL_SIZE.sum()==372 and np.all(CELL_SIZE>0))
    bad=v.copy();bad[~m]=np.nan
    check('invalid NaNs isolated',all(np.array_equal(x,representations(bad,m,cp,age)[0][r]) for r,x in f.items()))
    later=v.copy();later[1]+=1000
    lf,lo=representations(later,m,cp,age)
    check('no temporal lookahead',all(np.array_equal(f[r][0],lf[r][0]) for r in REPS))
    return checks


def plot_case(case,out):
    v,m,cp,age=single_finger(case['values'],case['age_us']);m[:,0]=case['mask']
    f,o=representations(v,m,cp,age)
    fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    fig.suptitle(case['title']+'\nA:15x1 | B:15x2 | C:15x4 | D:60x1 per finger | sigma=10 mm',fontsize=14)
    cmap=base.style();pressure=Normalize(0,max(float(v[0,0].max()),1))
    panels=[('Raw 12x32',v[0,0],m[0,0],(12,32),pressure,cmap),
            ('A: Gaussian Mean',f['A'][0,0,:,0],o['A'][0,0,:,0],(3,5),pressure,cmap),
            ('D: Dense Mean (virtual 6x10)',f['D'][0,0,:,0],o['D'][0,0,:,0],(6,10),pressure,cmap),
            ('B: Mean (=A)',f['B'][0,0,:,0],o['B'][0,0,:,0],(3,5),pressure,cmap),
            ('B: LOCAL Max',f['B'][0,0,:,1],o['B'][0,0,:,1],(3,5),pressure,cmap),
            ('C: Mean (=A)',f['C'][0,0,:,0],o['C'][0,0,:,0],(3,5),pressure,cmap),
            ('C: LOCAL Max (=B)',f['C'][0,0,:,1],o['C'][0,0,:,1],(3,5),pressure,cmap),
            ('C: Coverage (quality, not pressure)',f['C'][0,0,:,2],o['C'][0,0,:,2],(3,5),Normalize(0,1),'viridis'),
            ('C: Age ms (same across 15)',f['C'][0,0,:,3]/1000,o['C'][0,0,:,3],(3,5),Normalize(0,75),'cividis')]
    for ax,(title,x,valid,shape,norm,colors) in zip(axes.ravel(),panels):
        cm=plt.colormaps[colors].copy() if isinstance(colors,str) else colors
        cm.set_bad('#bbc3cc')
        im=ax.imshow(np.ma.array(x.reshape(shape),mask=~valid.reshape(shape)),origin='upper',aspect='auto',interpolation='nearest',norm=norm,cmap=cm)
        ax.set_title(title,fontsize=10); ax.set_xlabel('Column');ax.set_ylabel('Row')
        if shape==(3,5):
            ax.set_xticks(range(5),base.GRID_COLS);ax.set_yticks(range(3),base.GRID_ROWS)
            for i in range(15):
                ax.text(i%5,i//5,f'{x[i]:.1f}' if valid[i] else 'N/A',ha='center',va='center',fontsize=9,
                        color='white',bbox=dict(facecolor='black',alpha=.35,edgecolor='none',pad=1))
        elif shape==(6,10):
            ax.set_xticks([0,3,6,9],['0','10.3','20.7','31']);ax.set_yticks(range(6),['0','2.2','4.4','6.6','8.8','11'])
        fig.colorbar(im,ax=ax,shrink=.75,label='fraction' if 'Coverage' in title else 'ms' if 'Age' in title else 'sensor units')
    fig.savefig(out,dpi=120);plt.close(fig)


def plot_temporal(series,out):
    fig,axes=plt.subplots(6,2,figsize=(17,18),layout='constrained')
    fig.suptitle('001-rope | full timelines | representative finger = largest raw temporal range per episode\nNo time filtering; masks create gaps; B/C Mean=A, B/C global Max=raw Max',fontsize=14)
    for row,s in enumerate(series):
        t=s['seconds']; ax=axes[row,0]
        ax.plot(t,s['raw_mean'],label='Raw mean',color='#25395c')
        ax.plot(t,s['traces']['A']['mean'],label='A/B/C mean',color='#d36635')
        ax.plot(t,s['traces']['D']['mean'],label='D mean',color='#237f97')
        ax.set(title=f"EP{s['episode']} | {base.NAMES[s['stream']]}",xlabel='Time s',ylabel='Mean response');ax.legend(fontsize=8)
        ax=axes[row,1]; ax.plot(t,s['raw_peak'],label='Raw peak',color='#25395c',lw=2)
        for r,color in [('A','#d36635'),('B','#7a55b4'),('D','#237f97')]:
            x=np.where(s['traces'][r]['valid'],s['traces'][r]['peak'],np.nan)
            ax.plot(t,x,label=r+('/C max' if r=='B' else ' peak'),color=color,linestyle='--' if r=='B' else '-',lw=1)
        ax.set(title=f"EP{s['episode']} | peak channels",xlabel='Time s',ylabel='Peak response');ax.legend(fontsize=8)
    fig.savefig(out,dpi=110);plt.close(fig)


def plot_resolution(synth,out):
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    fig.suptitle('SYNTHETIC two-impulse resolution | 477 cases | sigma=10 mm fixed\nSeparated = exactly two 4-connected superlevel components; not a contact classifier')
    for ax,level in zip(axes,LEVELS):
        for rep,color in [('A','#d36635'),('B','#7a55b4'),('D','#237f97')]:
            xs=list(map(int,synth['pairs']));ys=[synth['pairs'][str(x)][rep][str(level)]['exactly_two_components_fraction'] for x in xs]
            ax.plot(xs,ys,'o-',label=rep+('/C Max' if rep=='B' else ' Mean'),color=color)
        ax.set(title=f'Contour = {int(level*100)}% of own peak',xlabel='True separation mm',ylabel='Fraction resolved',ylim=(-.03,1.03));ax.legend();ax.grid(alpha=.2)
    fig.savefig(out,dpi=130);plt.close(fig)


def avg(values):
    x=[v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else None


def aggregate(episodes):
    keys=['peak_retention_median','peak_location_error_mm_median','high_quartile_peak_location_error_mm_median',
          'mean_trend_r','peak_trend_r','mean_change_r','peak_change_r']
    per_episode=[]
    for ep in episodes:
        row={'episode':ep['episode'],'representations':{}}
        for rep in REPS:
            row['representations'][rep]={k:avg([f['summaries'][rep][k] for f in ep['fingers']]) for k in keys}
        per_episode.append(row)
    macro={r:{k:avg([e['representations'][r][k] for e in per_episode]) for k in keys} for r in REPS}
    return per_episode,macro


def write_reports(out,metrics):
    macro=metrics['episode_equal_macro']; synth=metrics['synthetic']; rows=[]
    for e in metrics['episode_summary']:
        r=e['representations'];rows.append(f"| {e['episode']} | "+' | '.join(f"{r[k]['peak_location_error_mm_median']:.2f} / {100*r[k]['peak_retention_median']:.1f}%" for k in REPS)+' |')
    synthetic_rows=[]
    for rep in REPS:
        s=synth['single_impulses'][rep]
        synthetic_rows.append(f"| {rep} | {s['mean_location_error_mm']:.2f} | {s['p95_location_error_mm']:.2f} | {100*s['mean_peak_retention']:.1f}% | "+' / '.join(str(s['quantized_codes'][str(b)]['unique_codes']) for b in (.25,1.,4.))+' |')
    resolution=[]
    for dist,p in synth['pairs'].items():
        resolution.append(f"| {dist} | "+' | '.join(f"{100*p[r]['0.5']['exactly_two_components_fraction']:.1f}%" for r in REPS)+' |')
    quality=[f['quality'] for e in metrics['episodes'] for f in e['fingers']]
    cases='\n'.join(f"- `{c['image']}`：{c['description']}" for c in metrics['cases'])
    temporal='\n'.join(f"| {r} | {macro[r]['mean_trend_r']:.4f} | {macro[r]['peak_trend_r']:.4f} | {macro[r]['peak_change_r']:.4f} |" for r in REPS)
    ent=[]
    for r in REPS:
        entries=[f['summaries'][r] for e in metrics['episodes'] for f in e['fingers']]
        ent.append(f"| {r} | {avg([x['marginal_entropy_bits']['1.0'] for x in entries]):.3f} | {avg([x['marginal_entropy_bits']['4.0'] for x in entries]):.3f} | {avg([x['mean_absolute_intercoordinate_r'] for x in entries]):.3f} |")
    report=f'''# Step 2b：触觉空间压缩的信息保留

日期：{metrics['created_local']}。基准 `{metrics['git_head']}`，分支 `{metrics['branch']}`。CPU-only，未加载 GS／checkpoint、未训练或调用 GPU；正式 P2、原始 NPZ 和上一轮审计均未修改。

## 1. 口径与表示

六个 Train episode：1、2、4（右手），5、7、8（双手）。共 {sum(e['frames'] for e in metrics['episodes'])} 个观测帧、{sum(f['raw_valid_frames'] for e in metrics['episodes'] for f in e['fingers'])} 个有效指-帧，18 条指序列。全部有效帧参与统计，不把相邻帧视为独立实验重复，不计算虚假的显著性／置信区间。主要汇总先对每指取时间统计，再在 episode 内指等权，最后六个 episode 等权。

| 表示 | 每指 Shape | 定义与有效性 |
|---|---|---|
| A | 15×1 | 完全复用上一轮 Gaussian Mean，σ=0.01 m |
| B | 15×2 | A + 固定最近控制点分区的 Max；不是整指 Max 广播 |
| C | 15×4 | B + 分区有效像素数／分区结构可用像素数 + 原始 age_us |
| D | 60×1 | 6×10 虚拟网格，覆盖同一夹指局部坐标范围，仍为 σ=0.01 m |

每指384槽位、永久屏蔽col31，共372结构可用像素。15点沿用rows=(0,6,11)、cols=(0,8,16,23,31)。D使用rows=linspace(0,11,6)、cols=linspace(0,31,10)及相同Controller局部公式；**每指60点、四指240点**，不是既有总共60控制点。未改变任何真实Controller坐标。

Max/Coverage的Voronoi分区仅由同指局部几何确定，等距归最低ID。Mean仍是全指Gaussian支持，故局部分区空时可能Mean有效、Max无观测；逐特征Mask独立保存。Coverage=0可作为有效质量信息，不能解释为有效零压力。Age来自最近过去raw样本的微秒差，未插值；无有效压力观测时Age值置0且其Mask=False，原始age另保留。Coverage/Age没有增加压力采样位置。

单手整组屏蔽左手；双手不按运动过滤。有效零响应保留。所有计算是当前帧空间映射，没有时间平滑或未来输入。触觉单位为扣基线截零后的传感器数值，非牛顿，未除以100。

## 2. 真实数据：峰值与位置

下表每格为“峰值位置误差 mm / 峰值保留率”；先每指取帧中位数，再指等权。B/C峰值统计使用Max通道，Mean通道与A完全相同，不能用Max的100%掩盖空间压缩。

| Episode | A | B（Max） | C（Max） | D |
|---|---|---|---|---|
{chr(10).join(rows)}

位置误差以输出最大节点（并列最低ID）到原始最大像素集合最近距离定义；零幅值无定义。它是峰值位置描述，不是接触定位真值。原始热点可包含噪声；完整结果另列每条序列原始均值上四分位的描述性子集，避免只用大量低响应帧解释强响应，但不把该分位当接触阈值或修改输入Mask。

高响应区域计数使用各场自身峰值的25%／50%／75%等值线、4连通域，无插值。B/C分别对Max通道计数；原始12×32与粗网格的分辨率不同，不能把连通域直接称为真实接触数量。完整逐帧表给出原始／输出数量；Hungarian最小距离匹配连通域的幅值加权中心，并单独记录数量丢失，避免只报告已匹配部分。阈值只用于离线描述，不用于模型输入或参与手判断。

## 3. 合成测试：与真实数据分开

### 单尖峰扫面

对372个有效像素逐个放置幅值100的单尖峰，其他值零，所有读数有效。已知真位置，不借助任何预测模型。

| 表示 | 平均位置误差 mm | P95 mm | 平均峰值保留率 | 量化步长0.25 / 1 / 4时不同编码数（共372） |
|---|---:|---:|---:|---|
{chr(10).join(synthetic_rows)}

“不同编码数”将每个压力特征按固定步长向下取整，整向量完全相同算碰撞；C在此质量通道恒定，故压力编码与B相同。未执行生产量化。完整指标包含碰撞对数、最大组和示例，不把有限精度碰撞当作浮点输出严格相等或任务信息丢失的证明。

### 双尖峰扫面

477个等幅双尖峰，在row2/6/9、所有容许横向起点及下列间距扫面。表中为50%相对等值线下恰好两个连通域的比例；25%与75%的敏感性见图与JSON。相邻高响应格连成一片也不保证两个热点可被单独辨认。

| 真间距 mm | A | B | C | D |
|---:|---:|---:|---:|---:|
{chr(10).join(resolution)}

本测试检验固定10mm核下的分辨能力，不把加密与缩窄核混合。未训练反演器；可能从微弱模式差异学习更多信息，但本轮不能据此宣称能恢复热点。

## 4. 全序列时间信息

| 表示 | 平均压力趋势r | 峰值趋势r | 相邻峰值变化r |
|---|---:|---:|---:|
{temporal}

B/C的全局Max等于原始有效最大值是分区完备时的代数性质，不是学到了触觉动力学；局部Max也会保留噪声尖峰。同一分区内热点移动／合并仍可能只保留一个最大值。A/B/C平均压力趋势相同。

出现／消失另用每条真实／压缩序列各自全序列min–max归一化后的10%／50%／90%跨越事件描述；只对连续两帧均有效的位置检测，按上升／下降分别最小距离匹配，保留未匹配数与有符号帧偏移。这是离线形态比较，非接触阈值；范围使用完整序列仅用于评估，不进入表示。空间聚合没有增加时间延迟，跨越时刻差异可能只是幅值和空间分布变化，不等价于物理滞后。数据本身原有0–75ms触觉年龄仍存在。

## 5. 信息论和质量通道

描述性插件熵：逐坐标统计有效帧，压力固定1／4传感器单位分箱，截到256并保留最高溢出箱；先求每坐标边际熵，再指序列平均。缺失不编码成零压力。表中B/C使用Max通道，A/D使用Mean。冗余性是完整帧中非恒定坐标两两Pearson绝对值平均；不是联合熵、有效比特率或与GS的互信息。

| 压力通道 | 边际熵 bits（1单位） | 边际熵 bits（4单位） | 平均坐标间绝对r |
|---|---:|---:|---:|
{chr(10).join(ent)}

Coverage完整覆盖比例（18指序列范围）为 {100*min(q['coverage_full_fraction'] for q in quality):.2f}%–{100*max(q['coverage_full_fraction'] for q in quality):.2f}%；Age在每指15点上完全重复。Coverage以0.1、Age以5000μs分箱的边际熵另存JSON。它们可能帮助识别不可靠观测，但不是新增的压力空间细节。

相同全有效Mask的线性Mean算子：A矩阵15×372，rank={synth['linear_operators']['A']['rank']}，stable rank={synth['linear_operators']['A']['stable_rank']:.2f}；D矩阵60×372，rank={synth['linear_operators']['D']['rank']}，stable rank={synth['linear_operators']['D']['stable_rank']:.2f}。stable rank定义为Σs²/s_max²；完整奇异值和条件数可复核。秩或维度更多不等于未来形变信息更多，宽核使许多方向的响应高度相关。

结构推断：B严格包含A；C严格包含B并附质量信息，所以不能在理想无损表达意义上比其子集少信息。但D与A/B不是相互包含关系，不能仅用60>15断言支配。上述表示均是原始触觉／Mask／时间的确定性函数；对未来GS的互信息或预测收益，本轮完全未测。

## 6. Q1／Q2／Q3 与建议

**Q1：15×2增加什么？** 保留15个局部分区的极值及其时间变化，补回Mean削弱的尖峰幅值，且没有把整指最大值广播到所有点。不能恢复区内峰的精确位置、数量或分布；Max可能强化噪声。Mean+Max描述互补，不等于完整空间信息。

**Q2：60×1是否明显改善空间定位和区分？** 本次不支持它全面优于15×2/15×4。真实数据的峰位置误差宏平均为A 10.50、B/C 4.00、D 9.19mm；D相对A改善约12.5%，但未优于B/C的局部Max定位。合成单尖峰中D为4.72mm、B/C为5.18mm，说明密网格确有定位收益。双热点在4–12mm间距时，四种表示在三档等值线下均未分成两个区域；32mm、50%等值线时A/B/D分别有40%/100%/73.3%分出两区。该结果依赖离散网格和等值线定义，不代表完整表示不可被后续网络解码。固定10mm宽核仍会混合邻近热点，不能把更多节点当作更窄的空间感受范围。结论只针对本轮网格、带宽和数据。

**Q3：首个Condition Encoder候选。** 优先把15×2作为压力表示的最小候选，Coverage/Age和Mask继续作为明确的质量信息保留并单独消融，而不把15×4当成4倍压力信息。是否把这两个质量标量拼入MLP，仍需后续训练对照；不能因为本数据覆盖率高就删除有效性／年龄协议。固定带宽下单纯增加到60×1尚不能保证热点可分；不要先扩大工具Token数量并同时改变融合机制。最终采用由后续同预算动力学训练的验证结果决定，本轮只给CPU依据。

## 7. 图像、复现与边界

五个案例：一个真实尖峰（按原始peak/mean最大选取）、已知相邻双热点、已知均匀面，以及真实最大相邻变化的前／后帧。真实案例只按原始值选取，不按表示的好坏挑图。图中Mean/Max共用压力色标；Coverage和Age用独立色标，分别标单位。

{cases}

`images/temporal.png`展示每个episode原始均值变化范围最大的参与手指全序列（仅展示选择；18指均参与统计）。`images/synthetic_resolution.png`展示多等值线分辨率。

CPU自检 {len(metrics['checks'])} 项通过；逐帧指标见 `frame_metrics.csv`，逐表示数组和逐通道Mask见 `representations/episode_*.npz`，完整方法／事件／合成／源哈希见 `metrics.json`。不存在训练、GS渲染、Slurm或GPU步骤。脚本检查全部PNG可解码及HTML本地资源存在；不声称浏览器截图或动力学收益验收。

命令：`python -B -m scripts.experiments.tactile_mapping.information_comparison --input-root audit/tactile-controller-viz-v1/input --output audit/tactile-information-comparison-v1`。设置 `CUDA_VISIBLE_DEVICES=`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_THREADING_LAYER=SEQUENTIAL`，使用现有CPU Python。源脚本SHA256：`{metrics['script_sha256']}`，旧映射脚本与全部旧结果的前后哈希一致性记录在JSON。
'''
    (out/'report.md').write_text(report,encoding='utf-8')
    pics=''.join(f'<section><h2>{html.escape(c["description"])}</h2><a href="{c["image"]}"><img src="{c["image"]}" loading="lazy" alt="{html.escape(c["title"])}"></a></section>' for c in metrics['cases'])
    summary='<table><tr><th>Episode</th><th>A 位置/峰值</th><th>B</th><th>C</th><th>D</th></tr>'+''.join('<tr>'+''.join(f'<td>{html.escape(x.strip())}</td>' for x in row.strip('|').split('|'))+'</tr>' for row in rows)+'</table>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Step 2b · 触觉信息保留</title><style>body{margin:0;background:#eef2f7;color:#203047;font:16px/1.65 system-ui}main{max-width:1500px;margin:auto;padding:28px}header,section{background:white;padding:24px;border-radius:16px;margin:20px 0;border:1px solid #d7dfeb}h1{font-size:32px}h2{font-size:22px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #dfe5ec;text-align:left}a{color:#254da8}p.note{background:#fff5da;padding:16px;border-radius:10px}</style><main><header><small>TOUCH4D / CPU ONLY / STEP 2b</small><h1>局部统计，还是更多空间采样？</h1><p>A：15 Mean · B：15 Mean+Max · C：15 Mean+Max+Coverage+Age · D：每指60 Mean</p><p><a href="report.md">完整报告与 Q1–Q3</a> · <a href="metrics.json">完整统计</a> · <a href="frame_metrics.csv">逐帧 CSV</a></p><p class="note">Max 保留峰值，不代表区内空间信息保留。60 点也不能自动消除 10 mm 核的平滑。Coverage／Age 分别展示，不混作压力。所有阈值仅用于离线描述。</p></header><section><h2>真实数据：位置误差 mm / 峰值保留率</h2>TABLE<p>每指先取帧中位数，再在episode内等权。B/C取Max通道；均值通道相同。包括低响应帧，不是接触精度。</p></section>CASES<section><h2>全序列趋势（18 指均纳入统计）</h2><img src="images/temporal.png" loading="lazy" alt="six episode temporal comparisons"></section><section><h2>合成双热点：固定带宽的分辨能力</h2><img src="images/synthetic_resolution.png" loading="lazy" alt="synthetic separation sweep"></section><footer>离线自包含，无CDN。图片可点击查看原尺寸。没有GPU模型训练或预测结果。</footer></main></html>'''
    (out/'index.html').write_text(page.replace('TABLE',summary).replace('CASES',pics),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=Path('audit/tactile-controller-viz-v1/input'))
    parser.add_argument('--output',type=Path,default=Path('audit/tactile-information-comparison-v1'))
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--refresh',action='store_true',help='Regenerate this audit only after QA; never overwrites prior Step2 output')
    args=parser.parse_args(); checks=numeric_tests()
    if args.self_test:print(json.dumps({'passed':len(checks),'checks':checks},indent=2));return
    root=args.input_root.resolve();out=args.output.resolve()
    old=root.parent
    if out==old or out in root.parents:raise ValueError('Output must be independent of source audit')
    if (out/'metrics.json').exists():
        if not args.refresh:raise FileExistsError('Use a new audit directory or explicitly refresh own QA outputs')
        assert json.loads((out/'metrics.json').read_text(encoding='utf-8'))['generator']=='tactile_information_comparison.v1'
    protected=[p for p in old.rglob('*') if p.is_file()]+list((base.REPO/'docs').rglob('*.md'))+[base.REPO/'AGENTS.md',Path(base.__file__)]
    before={str(p):base.digest(p) for p in protected}
    expected={line.split()[1]:line.split()[0] for line in (old/'source_sha256.txt').read_text().splitlines() if line.strip()}
    assert all(base.digest(root/p)==h for p,h in expected.items())
    out.mkdir(parents=True,exist_ok=True);(out/'images').mkdir(exist_ok=True);(out/'representations').mkdir(exist_ok=True)
    base.style(); frame_rows=[];eps=[];timelines=[];cases=[];spike=None;fast=None
    metrics={'generator':'tactile_information_comparison.v1','created_local':datetime.now().astimezone().isoformat(),
             'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
             'script_sha256':base.digest(__file__),'checks':checks,'sigma_m':SIGMA,'contour_fractions':LEVELS,
             'pressure_entropy_bins':[1,4],'pressure_entropy_overflow':256,'coverage_denominator':CELL_SIZE.tolist(),
             'pixel_to_local_max_partition':PARTITION.tolist(),'virtual60_coordinates_m':DENSE.tolist(),
             'source_hashes':expected,'protected_hashes':before}
    for ep in base.EPISODES:
        d=base.load_episode(root,ep); age=base.read_npz(root/'001-rope'/f'episode_{ep}'/'timestamps.npz')['tactile_age_us']
        features,masks=representations(d['values'],d['mask'],d['cp_mask'],age)
        arrays={f'{r}_values':features[r] for r in REPS}|{f'{r}_mask':masks[r] for r in REPS}
        arrays.update(frame_ids=np.arange(d['frames']),timestamp_us=d['timestamp_us'],age_us_original=age,
                      hand_keep=d['hand_keep'],input_valid_mask=d['mask'])
        np.savez_compressed(out/'representations'/f'episode_{ep}.npz',**arrays)
        fingers=[];representative=None
        for f in d['active']:
            fm,traces=finger_metrics(d,f,features,masks,frame_rows);fingers.append(fm)
            v=d['values'][:,f];m=d['mask'][:,f];mean=base.masked_mean(v,m);peak=np.max(v,axis=-1);valid=m.any(-1)
            score=np.where(valid&(mean>0),peak/np.maximum(mean,1e-300),-np.inf)
            idx=int(np.argmax(score))
            def case(i,description):
                return {'title':f"REAL 001-rope | EP{ep} frame{i} | {base.NAMES[f]}",
                        'description':description,'episode':ep,'frame':int(i),'stream':f,
                        'values':v[i].copy(),'mask':m[i].copy(),'age_us':int(age[i,f])}
            if spike is None or score[idx]>spike[0]:spike=(float(score[idx]),case(idx,'真实：原始 peak/mean 最大的局部尖峰（不等于接触）'))
            common=m[1:]&m[:-1];change=base.masked_mean(abs(v[1:]-v[:-1]),common)
            change=np.where(valid[1:]&valid[:-1],change,-np.inf);k=int(np.nanargmax(change))+1
            if fast is None or change[k-1]>fast[0]:fast=(float(change[k-1]),case(k-1,'真实：最大相邻触觉变化之前'),case(k,'真实：最大相邻触觉变化之后'))
            spread=np.nanmax(mean)-np.nanmin(mean)
            if representative is None or spread>representative[0]:
                representative=(spread,{'episode':ep,'stream':f,'seconds':d['seconds'],'raw_mean':mean,
                                       'raw_peak':np.where(valid,peak,np.nan),'traces':traces})
        eps.append({'episode':ep,'frames':d['frames'],'active':d['active'],'fingers':fingers});timelines.append(representative[1])
        print(f'EP{ep}: {d["frames"]} frames / {len(fingers)} active fingers',flush=True)
    pair=np.zeros(384);pair[6*32+14]=pair[6*32+18]=100
    cases=[spike[1],{'title':'SYNTHETIC | two equal hotspots | row6 col14/18 | 8 mm separation','description':'合成：8mm相邻双热点，同一15点分区',
                    'values':pair,'mask':ROI,'age_us':32000},
           {'title':'SYNTHETIC | uniform response 30 | all 372 available taxels','description':'合成：大面积均匀响应',
            'values':np.where(ROI,30,0),'mask':ROI,'age_us':32000},fast[1],fast[2]]
    for i,c in enumerate(cases):
        c['image']=f'images/case_{i+1}.png';plot_case(c,out/c['image'])
    synth=synthetic_suite();plot_resolution(synth,out/'images/synthetic_resolution.png');plot_temporal(timelines,out/'images/temporal.png')
    metrics.update(episodes=eps,synthetic=synth,cases=[clean(c) for c in cases],
                   selection={'spike_peak_mean_score':spike[0],'fast_mean_absolute_pixel_change':fast[0]},
                   frame_metric_rows=len(frame_rows))
    metrics['episode_summary'],metrics['episode_equal_macro']=aggregate(eps)
    with (out/'frame_metrics.csv').open('w',newline='',encoding='utf-8-sig') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(frame_rows[0]));writer.writeheader();writer.writerows(clean(frame_rows))
    assert before=={p:base.digest(p) for p in before},'Protected source or prior audit changed'
    metrics['protected_unchanged']=True
    metrics=clean(metrics)
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write_reports(out,metrics)
    from PIL import Image
    for p in (out/'images').glob('*.png'):
        with Image.open(p) as im:im.verify()
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ('href','src'):
                    assert '://' not in v and (out/v).is_file(),v
    Links().feed((out/'index.html').read_text(encoding='utf-8'))
    print(json.dumps({'output':str(out),'checks_passed':len(checks),'frames':sum(e['frames'] for e in eps),
                      'images':len(list((out/'images').glob('*.png'))),'macro':metrics['episode_equal_macro']},ensure_ascii=False))


if __name__=='__main__':main()
