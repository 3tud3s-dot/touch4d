"""Stage-gated frozen checkpoint audit. No optimizer is constructed or called."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

from scripts.experiments.v4_anchor_interaction.model import paired_models
from scripts.experiments.v4_anchor_interaction.run import Data
from scripts.experiments.fixed_window.fixed_fit import digest, TargetCache
from scripts.training.rendering import GaussianRenderer
from scripts.training.v4_runtime import load_checkpoint
from .core import components, verify_equivalence, same_state
from .diagnose import audit, ablations, finite_differences, values

BASE_SHA='d81d04544ccfd68f548ed3406342c6bb11e95bf9'
TRAIN_SHA='0baaedf5fb62ef8c9b2fb4cc9c3cc7ed7fbe2a26'
HISTORY=Path('/data1/userdata/tcweng/experiments/touch4d/v4-anchor-interaction-ablation-v1')
WINDOWS=[(2,73),(2,141),(7,107),(2,20),(3,140),(6,115)]
AUX=[(2,73),(2,141),(3,140)]


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for piece in iter(lambda:f.read(1024*1024),b''):h.update(piece)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(path)


def env():
    return dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        command=[sys.executable,*sys.argv],python=sys.version,platform=platform.platform(),
        torch=torch.__version__,cuda_build=torch.version.cuda,job=os.environ.get('SLURM_JOB_ID'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cwd=str(Path.cwd()),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_initialized() else None,
        numpy=np.__version__)


def model_for(tag):
    models=paired_models(); m=models[tag];del models
    return m.eval()


def checked_model(tag,device='cpu'):
    refs=json.loads((HISTORY/'comparison/checkpoint_provenance.json').read_text())
    ref=refs[tag]['best_joint'];path=Path(ref['path'])
    if not path.is_relative_to(HISTORY) or sha(path)!=ref['sha256']:
        raise ValueError('Checkpoint path/hash mismatch')
    m=model_for(tag)
    from scripts.data_process import DataProcessor
    processor=DataProcessor('001-rope')
    state=load_checkpoint(path,m,processor_state=processor.state_dict())
    arch=json.loads((HISTORY/'architecture_configs'/f'{tag}.json').read_text())['model']
    if (state['source_sha']!=TRAIN_SHA or state['source_sha']!=ref['source_sha'] or
        state['successful_steps']!=ref['updates'] or state['config']!=arch or
        state['successful_steps']!={'G0':1400,'G2':1400,'G3':1600}[tag]):
        raise ValueError('Frozen checkpoint metadata mismatch')
    m=m.to(device).eval()
    return m,dict(**ref,config=state['config'],processor=state['processor'],
                  state_sha256=digest(m.state_dict()),file_sha256_before=sha(path))


def preflight(root):
    if root.exists():raise FileExistsError('New independent output must not exist')
    for d in ('provenance','results','figures','logs','source','status'):(root/d).mkdir(parents=True,exist_ok=True)
    data=Data();cp={}
    for tag in ('G0','G2','G3'):
        model,ref=checked_model(tag);cp[tag]=ref
    windows=[]
    for key in WINDOWS:
        raw,b=data.batch(key,model,device='cpu');x,y=b['inputs'],b['targets']
        ids=raw['meta']['frame_ids'][0].tolist()
        if ids!=list(range(key[1],key[1]+11)):raise ValueError('Misaligned window')
        if not y['target_depth_valid'].flatten(3).any(-1).all():raise ValueError('Empty target camera/horizon')
        keep=x['gs_mask'][0];finite=bool(torch.isfinite(x['gs_xyz'][0,keep]).all() and torch.isfinite(x['gs_attr'][0,keep]).all())
        gt=[]
        for h in range(10):
            q=y['target_xyz'][h][0,y['target_mask'][h][0]]
            if not len(q) or not torch.isfinite(q).all():raise ValueError('Missing/nonfinite target GS')
            gt.append(len(q))
        if not finite:raise ValueError('Invalid initial GS')
        windows.append(dict(episode=key[0],start=key[1],split='train' if key[0] in (2,7) else 'val',
            frame_ids=ids,condition_frame_ids=x['condition_frame_ids'][0].tolist(),
            timestamps=raw['meta']['observation_timestamp_us'][0].tolist(),initial_points=int(keep.sum()),
            target_points=gt,operation_hands=x['operation_hands'][0].tolist(),
            valid_controllers=x['controller_mask'][0].sum((-1,-2)).tolist(),
            active_valid_controllers=(x['controller_mask'][0]&x['operation_hands'][0][None,:,None]).sum((-1,-2)).tolist(),
            target_valid_pixels=y['target_depth_valid'].sum((-1,-2))[0].tolist(),
            raw_data_sha256=digest(raw),normalized_data_sha256=digest(b)))
        del raw,b;gc.collect()
    protocol=dict(base_sha=BASE_SHA,training_sha=TRAIN_SHA,windows=windows,auxiliary_windows=AUX,
        models=cp,loss_weights={'cd':1,'rgb':1,'alpha':.1},horizon=10,prediction_interval=1,
        no_optimizer=True,no_parameter_updates=True,eval_mode=True,
        finite_difference=dict(windows=[[2,73],[2,141]],amplitudes_mm=[.001,.01,.1,.5,1.],
            directions=['negative_joint_gradient','positive_controller_cumulative','negative_controller_cumulative','fixed_random'],
            random_seed=1729,small_scales_central_difference=True),
        regions='bottom/top25% initial distance to active valid controllers; independent GS and Anchor rankings',
        full_val_not_recomputed=True,auxiliary_gradient_scope='joint only; G0 all four loss components',
        legacy_g1_g2_null_deviation='G2 A2A has an always-active zero Null, retained exactly from frozen training')
    write(root/'provenance/frozen_protocol.json',protocol)
    write(root/'provenance/checkpoint_provenance.json',cp)
    write(root/'provenance/environment.json',env())
    return dict(passed=True,windows=len(windows),checkpoint_hashes_verified=True,real_render_gate='pending CUDA pilot')


def require_preflight(root):
    if not json.loads((root/'status/preflight.json').read_text())['passed']:raise RuntimeError('Preflight failed')
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU stages require a Slurm allocation')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(42);np.random.seed(42)


def run_window(root,data,tag,key,full_components):
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();model,ref=checked_model(tag,'cuda')
    before=digest(model.state_dict());raw,b=data.batch(key,model)
    fp=json.loads((root/'provenance/frozen_protocol.json').read_text())
    frozen=next(w for w in fp['windows'] if (w['episode'],w['start'])==key)
    if digest(raw)!=frozen['raw_data_sha256'] or digest(b)!=frozen['normalized_data_sha256']:
        raise ValueError('Window data drifted from CPU preflight')
    renderer=TargetCache(GaussianRenderer())
    # Full official/decomposed parameter-gradient equivalence for every model/window.
    equality=verify_equivalence(model,b,data.processor,renderer)
    result,pred,arrays=audit(model,b,data.processor,renderer,full_components=full_components)
    result['condition_ablation']=ablations(model,b,data.processor,renderer,pred)
    with torch.no_grad():
        copy=dict(pred);copy['xyz']=b['inputs']['gs_xyz'][:,None].expand_as(pred['xyz'])
        copy_loss,copy_times=components(copy,b,data.processor,renderer,with_time=True)
        result['copy_losses']=values(copy_loss);result['copy_per_horizon_losses']=copy_times
        for row in result['loss_gradients']:
            row['standalone_copy_loss']=copy_times[row['loss']][row['horizon']-1]
        from scripts.training.losses import nearest_distances
        init=b['inputs']['gs_xyz'][0,b['inputs']['gs_mask'][0]];scale=data.processor.state_dict()['scale']*1000
        gt_proxy=[]
        for h in range(10):
            gt=b['targets']['target_xyz'][h][0,b['targets']['target_mask'][h][0]]
            gt_proxy.append(dict(horizon=h+1,copy_cd_mm=float((nearest_distances(init,gt).mean()+nearest_distances(gt,init).mean())*.5)*scale,
                centroid_shift_mm=float((gt.mean(0)-init.mean(0)).norm())*scale))
        result['gt_set_change_proxy']=gt_proxy
    result.update(model=tag,episode=key[0],start=key[1],split=frozen['split'],source=env(),checkpoint=ref,
        equivalence=equality,state_sha256_after=same_state(model,before),seconds=time.monotonic()-start,
        allocated_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved())
    if sha(ref['path'])!=ref['sha256']:raise RuntimeError('Checkpoint file changed')
    name=f'{tag}_ep{key[0]}_start{key[1]}'
    write(root/'results'/f'{name}.json',result)
    np.savez_compressed(root/'results'/f'{name}_position_gradients.npz',**arrays)
    print(json.dumps(dict(event='window_complete',name=name,seconds=result['seconds'],loss=result['losses'])),flush=True)
    # Return this graph-free batch only for bounded finite differences; caller releases it.
    del renderer,raw;torch.cuda.empty_cache()
    return model,b,pred


def pilot(root):
    require_preflight(root);data=Data();model,b,pred=run_window(root,data,'G0',(2,73),True)
    before=digest(model.state_dict())
    fd=finite_differences(pred,b,data.processor,GaussianRenderer(),pilot=True)
    write(root/'results/pilot_finite_difference.json',fd);same_state(model,before)
    return dict(passed=True,model='G0',window=[2,73],finite_difference_executed=True,
                no_parameter_updates=True,environment=env())


def full(root):
    require_preflight(root)
    if not json.loads((root/'status/pilot.json').read_text())['passed']:raise RuntimeError('Pilot gate not passed')
    data=Data();completed=[]
    for tag,keys in [('G0',WINDOWS),('G2',AUX),('G3',AUX)]:
        for key in keys:
            name=f'{tag}_ep{key[0]}_start{key[1]}'
            if tag=='G0' and key==(2,73):
                old=json.loads((root/'results'/f'{name}.json').read_text())
                if not old['equivalence']['passed']:raise RuntimeError('Pilot artifact invalid')
                model,_=checked_model(tag,'cuda');_,b=data.batch(key,model)
                with torch.no_grad():pred=model(b['inputs'])
            else:
                if (root/'results'/f'{name}.json').exists():raise FileExistsError('No implicit diagnostic restart')
                model,b,pred=run_window(root,data,tag,key,tag=='G0')
            if tag=='G0' and key in ((2,73),(2,141)):
                before=digest(model.state_dict())
                fd=finite_differences(pred,b,data.processor,GaussianRenderer())
                write(root/'results'/f'finite_difference_ep{key[0]}_start{key[1]}.json',fd)
                same_state(model,before)
            completed.append(name);del model,b,pred;gc.collect();torch.cuda.empty_cache()
    return dict(passed=True,completed=completed,g0_windows=6,auxiliary_windows_per_model=3,
                no_parameter_updates=True,environment=env())


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['preflight','pilot','full','report','package'],required=True)
    parser.add_argument('--expected-sha',required=True)
    parser.add_argument('--postprocess-attempt',default='initial',
                        help='CPU report/package recovery only; preserves the original failed status')
    args=parser.parse_args();actual=env()['source_sha']
    if not re.fullmatch(r'[a-z0-9_]{1,40}',args.postprocess_attempt):raise ValueError('Invalid attempt label')
    if args.postprocess_attempt!='initial' and args.stage not in ('report','package'):
        raise ValueError('GPU/data stages cannot be retried through postprocessing')
    status_name=args.stage if args.postprocess_attempt=='initial' else args.stage+'_'+args.postprocess_attempt
    if actual!=args.expected_sha:raise ValueError('Execution source SHA changed')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Execution worktree dirty')
    if args.stage=='preflight':
        try:result=preflight(args.root)
        except Exception:
            if args.root.exists():write(args.root/'status/preflight.json',dict(passed=False,error=traceback.format_exc(),source=env()))
            raise
    else:
        if (args.root/'status'/f'{status_name}.json').exists():raise FileExistsError('Stage already attempted; preserve evidence')
        try:
            write(args.root/'status'/f'{status_name}_running.json',env())
            if args.stage=='pilot':result=pilot(args.root)
            elif args.stage=='full':result=full(args.root)
            else:
                from .report import build_report, package
                result=build_report(args.root) if args.stage=='report' else package(args.root)
        except Exception:
            write(args.root/'status'/f'{status_name}.json',dict(passed=False,error=traceback.format_exc(),source=env()))
            raise
    write(args.root/'status'/f'{status_name}.json',dict(**result,execution=env()))
    print(json.dumps(dict(stage=args.stage,**result),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
