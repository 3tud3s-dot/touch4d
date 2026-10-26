"""Deterministic application of the pre-result Roadmap; no parameter search."""
import argparse,json
from pathlib import Path
import numpy as np


def mean_motion(result,key,ep=None):
    rows=[r for r in result['model']['windows'] if ep is None or r['episode']==ep]
    values=[m[key] for r in rows for m in r['motion'] if m.get(key) is not None]
    return float(np.mean(values)) if values else None


def score(result):return result['model']['summary']['mean_cd_mm']


def quality(new,base):
    reasons=[]
    for key in ('rgb','alpha'):
        if np.mean(new['model']['summary'][key])>1.05*np.mean(base['model']['summary'][key]):reasons.append(key)
    if mean_motion(new,'neighbor_distance_change_mm')>1.1*max(mean_motion(base,'neighbor_distance_change_mm'),1e-8):reasons.append('neighbor_change')
    def span(r):return float(np.mean([max(m['bbox_span_ratio']) for w in r['model']['windows'] for m in w['motion']]))
    if span(new)>1.1*span(base):reasons.append('bbox')
    return reasons


def eligible(new,base):
    reasons=quality(new,base)
    if score(new)>.99*score(base):reasons.append('below_1pct')
    for ep in ('3','6'):
        if new['model']['summary']['episodes'][ep]['mean_cd_mm']>base['model']['summary']['episodes'][ep]['mean_cd_mm']:reasons.append('episode_'+ep)
    return reasons


def stage1(results):
    order=['S1','S2-C0','S2-C1','S3-Pre','S3-Post','S4-C0','S4-C1']
    base=results['S0'];reasons={s:eligible(results[s],base) for s in order}
    allowed=[s for s in order if not reasons[s]]
    if allowed:
        best=min(score(results[s]) for s in allowed)
        selected=next(s for s in order if s in allowed and score(results[s])<=1.01*best)
    else:selected='S2-C1'
    local=None
    for s,ref in [('S4-C1','S4-C0'),('S4-C0','S2-C1')]:
        if not quality(results[s],results[ref]) and all(mean_motion(results[s],'near_cd_mm',ep) is not None
            and mean_motion(results[ref],'near_cd_mm',ep) is not None
            and mean_motion(results[s],'near_cd_mm',ep)<=.99*mean_motion(results[ref],'near_cd_mm',ep) for ep in (3,6)):
            local=s;break
    return dict(selected=selected,diagnostic_only=not allowed,rejection_reasons=reasons,
        d1_backbone=local,d1_status='TRIGGERED' if local else 'NOT_TRIGGERED',
        interpretation='Single seed screen; no adoption or significance claim.')


def d2_r1_trigger(d2,d0):
    return all(mean_motion(d2,'centroid_error_mm',ep)<=.99*mean_motion(d0,'centroid_error_mm',ep) for ep in (3,6)) and mean_motion(d2,'neighbor_distance_change_mm')>1.1*mean_motion(d0,'neighbor_distance_change_mm')


def h15_trigger(pairs):
    reasons={}
    for seed,values in pairs.items():
        new,base=values
        reasons[str(seed)]=eligible(new,base)
        if score(new)>new['copy_initial']['summary']['mean_cd_mm']:reasons[str(seed)].append('not_better_than_copy')
    return dict(triggered=all(not r for r in reasons.values()),reasons=reasons)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    mapping=json.loads(Path(a.runs).read_text());results={k:json.loads((Path(v)/'metrics.json').read_text()) for k,v in mapping.items()}
    decision=stage1(results);path=Path(a.output)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(decision,indent=2),encoding='utf-8');print(json.dumps(decision))
if __name__=='__main__':main()
