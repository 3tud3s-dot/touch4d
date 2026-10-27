"""CPU-only selection, before any predictions: nearest initial GS among high action-difference pairs."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from scripts.data_process import DataProcessor
from scripts.training.losses import nearest_distances
from scripts.training.train import atomic_json
from .common import dataset,batch_at,sha

def cd(a,b):
    return float((nearest_distances(a,b).mean()+nearest_distances(b,a).mean())*.5)

def sample(x,n=256):
    return x[torch.linspace(0,len(x)-1,min(n,len(x))).long()]

def run(root,output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    ds=dataset(root,False);scale=DataProcessor('001-rope').state_dict()['scale']*1000
    candidates=[];clouds=[];motions=[];files={}
    for ep in ds.episode_ids:
        starts=sorted(set(np.linspace(0,ds.episodes[ep].frames-6,12,dtype=int).tolist()))
        for start in starts:
            b=batch_at(ds,ep,start);x=b['inputs'];active=x['operation_hands'][0]
            cp=x['controller_points'][0,:,active];valid=x['controller_mask'][0,:,active]
            if not valid.all():continue
            initial=x['gs_xyz'][0,x['gs_mask'][0]];counts=[len(initial)]+[int(t.sum()) for t in b['targets']['target_mask']]
            # Avoid obvious cardinality discontinuity only; this is not a claim of complete GT quality.
            ratios=np.array(counts)/counts[0]
            if ratios.min()<.8 or ratios.max()>1.25:continue
            motion=(cp-cp[:1])*scale
            cloud=initial*scale
            candidates.append(dict(episode=ep,start=start,frames=list(range(start,start+6)),hands=active.tolist(),
                counts=counts,point_count_ratios=ratios.tolist(),controller_h4_rms_mm=float(motion[-1].square().sum(-1).mean().sqrt()),
                tactile_valid_per_step=x['valid_mask'].flatten(2).sum(-1).tolist()))
            clouds.append(cloud);motions.append(motion)
    pairs=[]
    for i,j in itertools.combinations(range(len(candidates)),2):
        a,b=candidates[i],candidates[j]
        if a['episode']==b['episode'] or a['hands']!=b['hands']:continue
        difference=float((motions[i][1:]-motions[j][1:]).square().sum(-1).mean().sqrt())
        pairs.append(dict(i=i,j=j,action_trajectory_difference_rms_mm=difference,
                          initial_sampled_world_cd_mm=cd(sample(clouds[i]),sample(clouds[j]))))
    if not pairs:raise ValueError('No cross-episode same-hand pair; do not invent one')
    threshold=float(np.quantile([p['action_trajectory_difference_rms_mm'] for p in pairs],.75))
    pool=[p for p in pairs if p['action_trajectory_difference_rms_mm']>=threshold]
    pool=sorted(pool,key=lambda p:(p['initial_sampled_world_cd_mm'],-p['action_trajectory_difference_rms_mm'],p['i'],p['j']))[:8]
    for p in pool:p['initial_full_world_cd_mm']=cd(clouds[p['i']],clouds[p['j']])
    chosen=min(pool,key=lambda p:(p['initial_full_world_cd_mm'],-p['action_trajectory_difference_rms_mm'],p['i'],p['j']))
    windows=[candidates[chosen[k]] for k in ('i','j')]
    batches=[batch_at(ds,w['episode'],w['start']) for w in windows]
    target_difference=[]
    for h in range(5):
        a,b=[v['targets']['target_xyz'][h][0,v['targets']['target_mask'][h][0]]*scale for v in batches]
        target_difference.append(cd(a,b))
    for w in windows:
        folder=ds.episodes[w['episode']].path
        for f in folder.rglob('*.json'):files[str(f)]=sha(f)
        for f in folder.glob('*.npz'):files[str(f)]=sha(f)
        for frame in w['frames']:
            f=folder/'gs'/f'frame_{frame:06d}.npz';files[str(f)]=sha(f)
    result=dict(status='frozen_before_training',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scene='001-rope',split='train',horizon=5,candidates=candidates,pairs=pairs,refined_pairs=pool,chosen=chosen,
        windows=windows,action_difference_p75_mm=threshold,target_pair_cd_mm=target_difference,
        target_h5_to_initial_pair_cd_ratio=target_difference[-1]/max(chosen['initial_full_world_cd_mm'],1e-9),
        selection='12 evenly spaced starts per train episode; same declared hand mode, different episode; count ratio 0.8..1.25; top action-difference quartile; lowest initial world CD in sampled-shortlist refined with full clouds',
        limitation='Initial states are close, not identical. Initial GS can identify a trajectory; only the synthetic test removes this confound. Cardinality filter does not certify mask/reconstruction quality. No model/checkpoint or validation/test used.',
        synthetic_base=dict(episode=2,start=73,frames=list(range(73,79))),source_file_sha256=files)
    out.parent.mkdir(parents=True,exist_ok=True);atomic_json(out,result)
    print(json.dumps(dict(windows=windows,chosen=chosen,target_pair_cd_mm=target_difference),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4)
    run(a.root,a.output)
