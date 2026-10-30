"""Input-only T3 statistics and preregistered within-Episode activity strata.

No prediction, GT motion, contact threshold or output error is used for grouping.
Pressure units are DataProcessor normalized (raw stored value /100).
"""
import numpy as np
import torch


def distribution(values,prefix):
    a=values.detach().float().cpu().numpy().reshape(-1)
    if not len(a):return {prefix+'_'+k:None for k in ('mean','p50','p95','max')}
    return {prefix+'_mean':float(a.mean()),prefix+'_p50':float(np.quantile(a,.5)),
            prefix+'_p95':float(np.quantile(a,.95)),prefix+'_max':float(a.max())}


@torch.inference_mode()
def activity_rows(encoder,x,scope):
    if len(x['gs_xyz'])!=1:raise ValueError('Activity audit requires Batch1')
    rows=[]
    for k in range(encoder.horizon):
        _,_,features,cp_valid,_=encoder.features(x,k)
        tactile=features[0,:,7:].reshape(4,15,6)
        for finger in range(4):
            hand=finger//2;allowed=bool(x['operation_hands'][0,hand])
            pressure=x['tactile'][0,k,finger].flatten();age=x['tactile_age_us'][0,k,finger]
            age_ok=bool(torch.isfinite(age)&(age>=0)&(age<=75000))
            spatial=x['tactile_spatial_mask'][0,k,finger].flatten()&encoder.roi
            read_valid=x['valid_mask'][0,k,finger].flatten()
            effective=spatial&read_valid&torch.isfinite(pressure)&age_ok&allowed
            count=int(effective.sum());nonzero=int((effective&(pressure!=0)).sum())
            positive=int((effective&(pressure>0)).sum())
            cp=cp_valid[0,finger*15:(finger+1)*15]
            f=tactile[finger,cp]
            row=dict(**scope,horizon=k+1,condition_frame=scope['start']+k,hand='left' if hand==0 else 'right',
                finger=finger%2,operation_hand=allowed,taxel_slots=384,spatial_slots=int(spatial.sum()),
                effective_valid_taxels=count,valid_fraction_slots=count/384,
                nonzero_valid_taxels=nonzero,positive_valid_taxels=positive,
                nonzero_fraction_valid=nonzero/count if count else None,
                positive_fraction_valid=positive/count if count else None,
                valid_zero_taxels=count-nonzero,age_us=float(age) if torch.isfinite(age) else None,
                age_valid=age_ok,valid_controllers=int(cp.sum()),
                mean_valid_points=int((f[:,4]>0).sum()),max_valid_points=int((f[:,5]>0).sum()))
            row.update(distribution(pressure[effective],'taxel_pressure'))
            for col,name in enumerate(('local_mean','local_max','coverage','age_normalized','mean_valid','max_valid')):
                row.update(distribution(f[:,col],name))
            rows.append(row)
    return rows


def freeze_groups(rows):
    """Per-Episode quartiles of positive/valid taxels across all sampled steps.

    Ties at Q25==Q75 are INDETERMINATE, not artificial high/low classes.
    No observations are NO_OBSERVATION. Middle samples remain MIDDLE.
    """
    keys=sorted({(r['split'],r['episode'],r['start']) for r in rows});windows=[]
    for split,ep,start in keys:
        subset=[r for r in rows if (r['split'],r['episode'],r['start'])==(split,ep,start) and r['operation_hand']]
        valid=sum(r['effective_valid_taxels'] for r in subset)
        positive=sum(r['positive_valid_taxels'] for r in subset)
        windows.append(dict(split=split,episode=ep,start=start,activity=positive/valid if valid else None,
                            valid_taxels=valid,positive_taxels=positive))
    thresholds={}
    for ep in sorted({r['episode'] for r in windows if r['split']=='val'}):
        subset=[r for r in windows if r['split']=='val' and r['episode']==ep]
        values=[r['activity'] for r in subset if r['activity'] is not None]
        q25,q75=np.quantile(values,[.25,.75]).tolist() if values else (None,None)
        thresholds[str(ep)]=dict(q25=q25,q75=q75,observed_windows=len(values),total_windows=len(subset),
                                 distinguishable=q25 is not None and q25<q75)
        for row in subset:
            a=row['activity']
            row['activity_group']=('NO_OBSERVATION' if a is None else 'INDETERMINATE' if q25==q75
                else 'LOW' if a<=q25 else 'HIGH' if a>=q75 else 'MIDDLE')
    for row in windows:
        if row['split']!='val':row['activity_group']='MECHANISM_CASE_ONLY'
    return dict(rule='within-Episode input-only Q25/Q75; positive valid taxels / all effective valid taxels over15 steps; exact pressure>0 is descriptive, NOT contact',
                frozen_before_inference=True,quantile_method='numpy linear',thresholds=thresholds,windows=windows)
