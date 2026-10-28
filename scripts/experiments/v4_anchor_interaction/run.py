"""Frozen 4x1600 all-window protocol; no test data, automatic retries or tuning."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
import numpy as np
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.training.v4_runtime import build_dataset, prepare_batch, v4_objective, save_checkpoint, load_checkpoint
from scripts.training.train import GS_CAMERAS, gs_render_settings, copy_initial, to_device, atomic_json
from scripts.training.rendering import GaussianRenderer
from scripts.training.losses import nearest_distances
from scripts.experiments.fixed_window.fixed_fit import TargetCache, digest, prediction_metrics
from .model import paired_models, MODES

SHARED = Path('/data1/userdata/tcweng/projects/touch4d')
FROZEN = Path('/data1/userdata/tcweng/experiments/touch4d/contact-boundaries-001-frozen-v1')
KEYS = [(2,141),(7,107),(3,140),(6,115),(3,230),(6,160)]
BASE = '2e8afae7a74a1a4138f45e49dc8add83df6b579e'
UPDATES = 1600


def dump(p, x): atomic_json(Path(p), x)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def provenance():
    return dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                job=os.environ.get('SLURM_JOB_ID'), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpu=torch.cuda.get_device_name(), torch=torch.__version__)
def seed():
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
def fresh(tag):
    seed(); models=paired_models(); model=models[tag]; del models
    return model.cuda()


class Data:
    def __init__(self):
        self.processor=DataProcessor('001-rope')
        self.datasets={s:build_dataset('001-rope',s,SHARED/'datasets/touch4d-v1',horizon=10,prediction_interval=1,
            with_depth=True,depth_root=SHARED/'datasets/touch4d-depth-v1',depth_cameras=GS_CAMERAS['001-rope']) for s in ('train','val')}
        self.windows={s:list(d.windows) for s,d in self.datasets.items()}
        if len(self.windows['train'])!=1544 or len(self.windows['val'])!=102:
            raise ValueError('Expected original all-window 1544 train / 102 val')
        self.indices={s:{tuple(k):i for i,k in enumerate(d.windows)} for s,d in self.datasets.items()}
        self.labels=json.loads((FROZEN/'contact_boundaries.json').read_text())
        def contact(k):
            e,t=k; a=self.labels['episodes'][str(e)]
            if a['review_status']!='MANUAL_CONFIRMED':raise ValueError('Unconfirmed annotation')
            return a['contact_start']<=t and t+10<a['contact_end_exclusive']
        self.contact=[k for k in self.windows['val'] if contact(k)]
        if len(self.contact)!=52:raise ValueError('Expected 52 unchanged contact diagnostics')

    def batch(self,key,model,device='cuda'):
        ep,t=key
        if ep in (0,9):raise ValueError('Test GS forbidden')
        split='train' if ep in (1,2,4,5,7,8) else 'val'
        if tuple(key) not in self.indices[split]:
            raise ValueError('Extra window must be audited separately, never reindexed')
        raw=collate_fn([self.datasets[split][self.indices[split][tuple(key)]]])
        if raw['meta']['frame_ids'].tolist()!=[list(range(t,t+11))]:raise ValueError('Bad timeline')
        batch=to_device(prepare_batch(raw,self.processor,model),device)
        if not batch['targets']['target_depth_valid'].flatten(2).any(-1).all():raise ValueError('No valid supervision')
        return raw,batch


def frozen_order(windows, steps=1600):
    rng=random.Random(42); order=[]
    while len(order)<steps:
        current=list(windows);rng.shuffle(current);order.extend(current)
    return order[:steps]


def freeze(root):
    root.mkdir(parents=True,exist_ok=False)
    for name in ('provenance','architecture_configs','tests','runs','comparison','visualizations'):(root/name).mkdir()
    data=Data(); models=paired_models(); base=models['G0'].state_dict(); configs={}
    for tag,m in models.items():
        shared={n:m.state_dict()[n] for n in base}
        if digest(shared)!=digest(base):raise ValueError('Shared initialization mismatch')
        configs[tag]=dict(model=m.config,mode=MODES[tag],parameters=sum(p.numel() for p in m.parameters()),
            shared_initial_sha256=digest(shared),initial_sha256=digest(m.state_dict()),
            new_parameters=[n for n in m.state_dict() if n not in base],
            added_initialization='GeometryCrossBlock q and final pairwise zero; geometry_score=(0,0,0,-0.5); other Linear defaults at seed4201; type vectors zero')
        dump(root/'architecture_configs'/f'{tag}.json',configs[tag])
    for key in KEYS:
        split='train' if key[0] in (2,7) else 'val'
        if key not in data.windows[split]:raise ValueError('Requested diagnostic window not in original stride: '+str(key))
    dump(root/'provenance/frozen_protocol.json',dict(base_sha=BASE,freeze_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        train=data.windows['train'],val=data.windows['val'],contact_val=data.contact,diagnostics=KEYS,
        order=frozen_order(data.windows['train']),seed=42,updates=1600,eval_every=200,
        optimizer=dict(type='AdamW',lr=1e-4,weight_decay=1e-4,clip=1.),
        loss=dict(cd=1.,rgb=1.,alpha=.1),cameras=GS_CAMERAS['001-rope'],render_settings=gs_render_settings(),
        annotation_sha256=sha(FROZEN/'contact_boundaries.json'),annotation_csv_sha256=sha(FROZEN/'contact_boundaries.csv'),
        normalization=data.processor.state_dict(),checkpoint_selection='full val episode-equal ordinary normalized CD + RGB + 0.1Alpha, every200; contact not used',
        initializations=configs,near_far='initial current GS to active valid Controller center distance; stable rank lowest/highest25%; fixed over rollout',
        region_error='one-way mean nearest distance to whole future independent GT set; not material point correspondence',
        rigid_residual='proper Kabsch initial-to-predicted identity transform; no GT per-index matching',
        no_test_gt=True))


def reduce_rows(rows):
    def avg(rr):
        keys=('cd_mm','rgb','alpha','motion_rms_mm','signed_mean_motion_mm','common_joint')
        out={k:np.mean([r[k] for r in rr],axis=0).tolist() for k in keys}
        out.update(mean_cd_mm=float(np.mean(out['cd_mm'])),mean_rgb=float(np.mean(out['rgb'])),mean_alpha=float(np.mean(out['alpha'])))
        return out
    eps={str(ep):avg([r for r in rows if r['episode']==ep]) for ep in sorted({r['episode'] for r in rows})}
    equal={k:np.mean([r[k] for r in eps.values()],axis=0).tolist() for k in next(iter(eps.values()))}
    return dict(equal=equal,episodes=eps,windows=rows)


@torch.no_grad()
def metrics(pred,b,p,renderer):
    r=prediction_metrics(pred,b,p,renderer,512)
    keep=b['inputs']['gs_mask'][0];scale=p.state_dict()['scale']*1000
    delta=(pred['xyz'][0,:,keep]-b['inputs']['gs_xyz'][0,keep])*scale
    r.update(motion_rms_mm=delta.square().sum(-1).mean(-1).sqrt().tolist(),signed_mean_motion_mm=delta.mean(1).tolist())
    r['common_joint']=r['mean_cd_mm']/scale+r['mean_rgb']+.1*r['mean_alpha']
    return r


def alter(b,mode):
    c=copy.deepcopy(b);x=c['inputs']
    if mode=='freeze_action':
        for n in ('action_state','controller_points'):x[n]=x[n][:,:1].expand_as(x[n]).clone()
        for n in ('action_valid','controller_mask'):x[n]=x[n]&x[n][:,:1]
    elif mode=='zero_tactile':x['tactile'].zero_()
    else:raise ValueError(mode)
    return c


@torch.no_grad()
def attention(trace,scale):
    am=trace['field'].anchor_mask; rows=[]
    for h,t in enumerate(trace['steps']):
        a=t['alpha'][am]
        row=dict(controller_mass_by_head=a[...,:60].sum(-1).mean(0).tolist(),null_mass_by_head=a[...,-1].mean(0).tolist(),
            anchor_mass_by_head=a[...,60:-1].sum(-1).mean(0).tolist(),
            tool_motion_mean_mm=(t['tool_motion'][am].reshape(-1,4,3).mean(0)*scale).tolist(),
            valid_controllers=int(t['token_mask62'][0,2:].sum()),layers={})
        for layer,records in trace.get('interactions',{}).items():
            z=records[h];stats={}
            if 'hybrid_weights' in z:
                v=z['hybrid_weights'][am]
                stats=dict(anchor_mass=v[...,60:-1].sum(-1).mean(0).tolist(),controller_mass=v[...,:60].sum(-1).mean(0).tolist(),null_mass=v[...,-1].mean(0).tolist())
            if 'anchor_weights' in z:
                xyz=t['anchor_xyz'];dist=(xyz[:,:,None]-xyz[:,None]).norm(dim=-1)*scale
                stats['a2a']=[]
                for weights in z['anchor_weights']:
                    v=weights[am];bins=[]
                    for lo,hi in ((0,10),(10,30),(30,float('inf'))):
                        g=(dist>=lo)&(dist<hi)&am[:,:,None]&am[:,None]
                        bins.append((weights[...,:128]*g[:,:,None]).sum(-1)[am].mean(0).tolist())
                    stats['a2a'].append(dict(anchor_mass=v[...,:128].sum(-1).mean(0).tolist(),null_mass=v[...,-1].mean(0).tolist(),distance_bins_mm=[[0,10],[10,30],[30,None]],distance_mass=bins))
            row['layers'][layer]=stats
        rows.append(row)
    return rows


@torch.no_grad()
def spatial(pred,b,scale):
    x=b['inputs']; keep=x['gs_mask'][0];initial=x['gs_xyz'][0,keep]
    cp=x['controller_points'][0,0].reshape(60,3);cm=x['controller_mask'][0,0].reshape(60)&x['operation_hands'][0].repeat_interleave(30)
    cm=cm & torch.isfinite(cp).all(-1)
    if not cm.any():raise ValueError('No valid initial controller for fixed Near/Far')
    distance=torch.cdist(initial[None],cp[cm][None])[0].min(-1).values
    order=distance.argsort(stable=True);n=max(1,len(order)//4);regions={'near':order[:n],'far':order[-n:]}
    current=pred['xyz'][0,-1,keep];delta=(current-initial)*scale
    gt=b['targets']['target_xyz'][-1][0,b['targets']['target_mask'][-1][0]]
    out={}
    for name,ix in regions.items():
        now=float(nearest_distances(current[ix],gt,512).mean()*scale)
        before=float(nearest_distances(initial[ix],gt,512).mean()*scale)
        out[name]=dict(count=len(ix),motion_rms_mm=float(delta[ix].square().sum(-1).mean().sqrt()),
            pred_to_gt_mean_mm=now,copy_to_gt_mean_mm=before,local_improvement_mm=before-now,
            signed_motion_mm=delta[ix].mean(0).tolist(),distance_range_mm=[float(distance[ix].min()*scale),float(distance[ix].max()*scale)])
    out['near_far_ratio']=out['near']['motion_rms_mm']/max(out['far']['motion_rms_mm'],1e-12)
    xc=initial-initial.mean(0);yc=current-current.mean(0)
    u,_,vh=torch.linalg.svd(xc.T@yc);d=torch.eye(3,device=xc.device);d[-1,-1]=torch.det(u@vh)
    residual=(yc-xc@(u@d@vh))*scale
    out['proper_rigid_removed_rms_mm']=float(residual.square().sum(-1).mean().sqrt())
    out['translation_removed_rms_mm']=float((delta-delta.mean(0)).square().sum(-1).mean().sqrt())
    for name,ix in regions.items():out[name]['proper_rigid_removed_rms_mm']=float(residual[ix].square().sum(-1).mean().sqrt())
    out['region_definition']='initial-space rank, fixed; future GT error is one-way nearest distance, no GT ID'
    return out,regions


@torch.no_grad()
def evaluate(model,data,windows,diagnostics=False,is_copy=False):
    model.eval();rows=[];scale=data.processor.state_dict()['scale']*1000
    for ep,t in windows:
        _,b=data.batch((ep,t),model);renderer=TargetCache(GaussianRenderer())
        if is_copy:pred=copy_initial(b['inputs'],'gs_geometry');trace=None
        elif diagnostics:pred,trace=model(b['inputs'],return_trace=True)
        else:pred=model(b['inputs']);trace=None
        row=metrics(pred,b,data.processor,renderer);row.update(episode=ep,start=t)
        row['spatial'],_=spatial(pred,b,scale)
        if diagnostics:
            row['attention']=attention(trace,scale);row['conditions']={}
            for mode in ('freeze_action','zero_tactile'):
                c=alter(b,mode);other,tr=model(c['inputs'],return_trace=True);z=metrics(other,b,data.processor,renderer)
                keep=b['inputs']['gs_mask'][0];delta=(other['xyz'][0,:,keep]-pred['xyz'][0,:,keep])*scale
                row['conditions'][mode]=dict(delta_cd_mm=z['mean_cd_mm']-row['mean_cd_mm'],metrics=z,
                    output_change_rms_mm=delta.square().sum(-1).mean(-1).sqrt().tolist(),attention=attention(tr,scale))
        rows.append(row)
    return reduce_rows(rows)


def param_stats(model,before=None):
    groups={}
    prefixes=['gs_encoder','local','conditions.controller','conditions.hand','cross','motion']
    prefixes += [n for n,_ in model.named_modules() if n.endswith('.a2a') or n.endswith('.pairwise')]
    prefixes += [n for n,_ in model.named_parameters() if n.endswith('.geometry_score') or n.endswith('.node_types')]
    for group in prefixes:
        ps=[(n,p) for n,p in model.named_parameters() if n.startswith(group)]
        r=dict(gradient_norm=sum(float(p.grad.detach().double().square().sum()) for n,p in ps if p.grad is not None)**.5,
               missing_gradients=sum(p.grad is None for n,p in ps))
        if before is not None:r['update_norm']=sum(float((p.detach()-before[n]).double().square().sum()) for n,p in ps)**.5
        groups[group]=r
    return groups


def update(model,opt,b,p):
    model.train();opt.zero_grad(set_to_none=True)
    before={n:q.detach().clone() for n,q in model.named_parameters()}
    pred=model(b['inputs']);loss,parts=v4_objective(pred,b,p,TargetCache(GaussianRenderer()),512)
    loss.backward()
    if not torch.isfinite(loss) or any(q.grad is not None and not torch.isfinite(q.grad).all() for q in model.parameters()):raise FloatingPointError('Nonfinite loss/gradient')
    groups=param_stats(model);gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
    if any(not torch.isfinite(q).all() for q in model.parameters()):raise FloatingPointError('Nonfinite parameters')
    return dict(loss=float(loss.detach()),geometry=float(parts['geometry'].mean()),rgb=float(parts['rgb'].mean()),alpha=float(parts['alpha'].mean()),
        gradient_norm=gn,clipped=gn>1.,raw_gradients=groups,actual_updates=param_stats(model,before))


def smoke(root):
    data=Data();rows=[]
    for tag in MODES:
        m=fresh(tag);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4)
        _,b=data.batch((2,141),m);torch.cuda.reset_peak_memory_stats();begin=time.time()
        updates=[update(m,opt,b,data.processor) for _ in range(2)]
        for name in ('gs_encoder','conditions.controller','cross','motion'):
            if not any(row['actual_updates'][name]['update_norm']>0 for row in updates):
                raise AssertionError('No actual optimizer update: '+tag+'/'+name)
        for name,stats in updates[-1]['actual_updates'].items():
            if name.endswith('.a2a') and stats['update_norm']<=0:
                raise AssertionError('New A2A branch did not update')
        pred,tr=m(b['inputs'],return_trace=True)
        gradients=torch.autograd.grad(pred['xyz'][:,-1].square().mean(),[t['state'] for t in tr['steps']],allow_unused=True)
        if any(g is None or not torch.isfinite(g).all() for g in gradients):raise AssertionError('Missing/nonfinite AR graph')
        norms=[float(g.double().norm()) for g in gradients]
        ck=root/'provenance'/f'smoke_{tag}.pt'
        save_checkpoint(ck,m,opt,2,source_sha=provenance()['source_sha'],processor_state=data.processor.state_dict())
        clone=fresh(tag);co=torch.optim.AdamW(clone.parameters(),lr=1e-4,weight_decay=1e-4)
        load_checkpoint(ck,clone,co,processor_state=data.processor.state_dict())
        with torch.no_grad():torch.testing.assert_close(m(b['inputs'])['xyz'],clone(b['inputs'])['xyz'],rtol=0,atol=0)
        row=dict(tag=tag,updates=updates,shape=list(pred['xyz'].shape),graph_grad_norms=norms,checkpoint_exact=True,
            allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved(),seconds=time.time()-begin,source=provenance())
        rows.append(row);dump(root/'provenance'/f'smoke_{tag}.json',row)
        del m,opt,b,pred,tr,clone,co,gradients;torch.cuda.empty_cache()
    dump(root/'provenance/cuda_gate.json',dict(passed=True,cases=rows,source=provenance()))


def train(root,tag):
    gate=json.loads((root/'provenance/cuda_gate.json').read_text())
    if not gate['passed'] or gate['source']['source_sha']!=provenance()['source_sha']:raise ValueError('CUDA gate/source mismatch')
    protocol=json.loads((root/'provenance/frozen_protocol.json').read_text())
    data=Data();p=data.processor;m=fresh(tag);out=root/'runs'/tag;out.mkdir(exist_ok=False)
    if digest(m.state_dict())!=protocol['initializations'][tag]['initial_sha256']:raise ValueError('Initialization drift')
    if protocol['annotation_sha256']!=sha(FROZEN/'contact_boundaries.json'):raise ValueError('Annotation drift')
    if [list(k) for k in data.windows['val']]!=protocol['val']:raise ValueError('Validation drift')
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4)
    ledger=dict(tag=tag,run_id=str(out),source=provenance(),model=m.config,updates=[],evaluations=[],successful_updates=0,
        best_joint=None,lowest_cd=None,initial_model_hash=digest(m.state_dict()),sample_order_sha256=digest(protocol['order']))
    dump(out/'config.json',dict(protocol=protocol,model=m.config,source=provenance()))
    def save(name):save_checkpoint(out/name,m,opt,ledger['successful_updates'],source_sha=provenance()['source_sha'],processor_state=p.state_dict(),run_state=ledger,replace_existing=(out/name).exists())
    save('initial.pt');start=time.time();torch.cuda.reset_peak_memory_stats()
    for step,key in enumerate(protocol['order'],1):
        before=time.time();_,b=data.batch(key,m);row=update(m,opt,b,p);row.update(update=step,key=key,seconds=time.time()-before)
        ledger['updates'].append(row);ledger['successful_updates']=step
        if step%200==0:
            val=evaluate(m,data,data.windows['val']);val['update']=step;ledger['evaluations'].append(val)
            score=val['equal']['common_joint'];cd=val['equal']['mean_cd_mm']
            if ledger['best_joint'] is None or score<ledger['best_joint']:
                ledger['best_joint']=score;ledger['best_step']=step;save('best_joint.pt')
            if ledger['lowest_cd'] is None or cd<ledger['lowest_cd']:
                ledger['lowest_cd']=cd;ledger['lowest_cd_step']=step;save('lowest_cd.pt')
            save('last.pt');dump(out/'metrics.json',ledger)
            print(json.dumps(dict(tag=tag,update=step,validation=val['equal'])),flush=True)
    if ledger['successful_updates']!=1600:raise ValueError('Wrong successful update count')
    ledger.update(seconds=time.time()-start,allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved(),
        checkpoint_hashes={n:sha(out/n) for n in ('initial.pt','best_joint.pt','lowest_cd.pt','last.pt')})
    dump(out/'metrics.json',ledger)
    with (out/'training_metrics.csv').open('w',newline='') as f:
        fields=['update','key','seconds','loss','geometry','rgb','alpha','gradient_norm','clipped'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in ledger['updates'])
    dump(out/'status.json',dict(status='COMPLETED',successful_updates=1600,source=provenance()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--expected-sha',required=True)
    parser.add_argument('--stage',choices=['freeze','smoke','train','all'],required=True)
    parser.add_argument('--group',choices=list(MODES));args=parser.parse_args()
    actual=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if actual!=args.expected_sha or subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Dirty or drifting execution source')
    torch.set_num_threads(4);seed();root=Path(args.root)
    if args.stage=='freeze':freeze(root);return
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Requires authorized Slurm GPU')
    try:
        if args.stage in ('smoke','all'):smoke(root)
        if args.stage=='train':train(root,args.group)
        if args.stage=='all':
            for tag in MODES:train(root,tag)
            from .report import final
            final(root)
    except BaseException:
        dump(root/'provenance'/f'failure_{args.stage}_{args.group or "all"}.json',dict(source=provenance(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
