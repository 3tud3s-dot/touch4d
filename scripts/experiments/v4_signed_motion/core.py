import numpy as np
import torch

EPS_M=1e-6


def deltas(cp,valid,step):
    current=cp[step];m=valid[step]
    cumulative_valid=m&valid[0]
    adjacent_valid=torch.zeros_like(m) if step==0 else m&valid[step-1]
    cumulative=torch.where(cumulative_valid[...,None],current-cp[0],0.)
    adjacent=torch.zeros_like(current) if step==0 else torch.where(adjacent_valid[...,None],current-cp[step-1],0.)
    return cumulative,cumulative_valid,adjacent,adjacent_valid


def tool_aggregation(alpha,adjacent):
    return torch.einsum('ahj,jd->ahd',alpha[...,:60],adjacent)


def cosine(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite direction')
    if np.linalg.norm(a)<=EPS_M or np.linalg.norm(b)<=EPS_M:return None
    return float(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))


def stats(v):
    v=np.asarray(v,dtype=np.float64)
    if not len(v):return dict(count=0,mean_x_mm=None,mean_y_mm=None,mean_z_mm=None,rms_mm=None,residual_rms_mm=None)
    mean=v.mean(0)
    return dict(count=len(v),**{f'mean_{axis}_mm':float(x)*1000 for axis,x in zip('xyz',mean)},
                rms_mm=float(np.sqrt(np.mean(np.sum(v*v,axis=-1))))*1000,
                residual_rms_mm=float(np.sqrt(np.mean(np.sum((v-mean)**2,axis=-1))))*1000)


def frame_contract(frame_ids,start,horizon,delta):
    ids=np.asarray(frame_ids)
    expected=np.arange(start,start+(horizon+1)*delta,delta)
    if not np.array_equal(ids,expected):raise ValueError('Unexpected original frame IDs')
    return ids[:-1],ids[1:]
