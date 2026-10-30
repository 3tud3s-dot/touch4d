"""Frozen paired H15 protocol, one calibration, and authorized GPU identity."""
import json
import math
import os
from pathlib import Path
import subprocess
import importlib.metadata
import copy
import torch
from scripts.experiments.v4_g5_signed_local import protocol as original
from scripts.experiments.v4_g5_signed_local.diagnostics import sha256
from .model import ARCHITECTURES

BASE='f978e0a29b6f0f0b6097c40684b949797a2f5d7b'
REFERENCE_HASH='067984c4c286306094efeff2f28ed57e3dcdd0c333972ecc6feb4b982ee3a40c'
REFERENCE='/data1/userdata/tcweng/experiments/touch4d/v4-g5-direct-v1-training/run-001/best.pt'
GPU_UUID='GPU-4048d136-505b-fca2-631f-fda0ab7ab9af'
JOB_ID='32087'
BUCKETS={'g5_struct':'v4-g5-struct-v1-training','g6_state_struct':'v4-g6-state-struct-v1-training'}
TRAIN_WINDOWS=((2,141),(7,105))
SMOKE_ORDER=TRAIN_WINDOWS+TRAIN_WINDOWS
WINDOWS=(('val',3,135),('val',3,245),('val',6,110),('train',7,105),
         ('val',3,25),('val',6,20),('val',6,200))
STRUCTURE=dict(graph='initial_original_16nn_mutual_undirected',epsilon=1e-6,
    spacing_median='midpoint_even',length_lower_multiplier=.1,length_upper_multiplier=2.,
    ratio_min=.8,ratio_max=1.2,smooth_l1_beta=.1,reduction='edges_and_horizon_per_window_then_batch')


def configuration(root,depth_root,source,variant,lambda_struct=.01):
    if variant not in ARCHITECTURES or not isinstance(lambda_struct,(float,int)) or not math.isfinite(lambda_struct) or not 0<=lambda_struct<=.01:
        raise ValueError('Invalid frozen variant/structure weight')
    return dict(original.configuration(root,depth_root,source,'direct'),variant=variant,
        architecture=ARCHITECTURES[variant],lambda_struct=float(lambda_struct),structure=copy.deepcopy(STRUCTURE),
        model_selection='complete_val_episode_equal_original_joint')


def common_configuration(c):
    if c!=configuration(c['root'],c['depth_root'],c['source_sha'],c['variant'],c['lambda_struct']):
        raise ValueError('Frozen structural/state protocol changed')
    return original.configuration(c['root'],c['depth_root'],c['source_sha'],'direct')


def preflight(c,reference):
    raw,data=original.preflight(common_configuration(c),reference)
    keys=data['manifest']['train']['window_keys']
    if any(list(k) not in keys for k in TRAIN_WINDOWS):raise ValueError('Prescribed train window absent')
    return raw,data


def source_guard(expected):
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source!=expected or subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Clean committed pinned source required')
    files=subprocess.check_output(['git','diff','--name-only',BASE,'HEAD'],text=True).splitlines()
    if any(not (p.startswith('scripts/experiments/v4_g5_struct_state/') or
                p.startswith('scripts/tests/training/test_v4_struct')) for p in files):
        raise ValueError('Changes outside isolated experiment: '+str(files))
    return source,{p:sha256(p) for p in files}


def component_hashes():
    root=Path(__file__).parent
    return {p.name:sha256(p) for p in sorted(root.glob('*.py'))
            if p.stem in ('model','state','basis','structure','calibrate','runner')}


def calibration_read(path):
    d=json.loads(Path(path).read_text())
    if not d.get('passed') or d.get('reference_sha256')!=REFERENCE_HASH or d.get('windows')!=[list(w) for w in TRAIN_WINDOWS] or d.get('components')!=component_hashes():
        raise ValueError('Calibration provenance/math/window mismatch')
    if d.get('rule')!='candidate0.01_downward_once_max_two_training_xyz_gradient_ratios_5percent':
        raise ValueError('Unknown calibration rule')
    if not math.isfinite(d['lambda_struct']) or not 0<=d['lambda_struct']<=.01:
        raise ValueError('Invalid frozen calibration weight')
    records=d.get('records',[])
    if len(records)!=2 or [(r.get('episode'),r.get('start')) for r in records]!=list(TRAIN_WINDOWS):
        raise ValueError('Calibration must contain exactly the two prescribed train records')
    ratios=[]
    for row in records:
        gj,gs=row['joint_gradient_norm'],row['struct_gradient_norm']
        if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in (gj,gs)):
            raise ValueError('Bad calibration gradient norm')
        ratios.append(gs/gj if gj else (float('inf') if gs else 0.))
    maximum=max(ratios)
    expected=min(.01,.05/maximum) if maximum else .01
    if not math.isclose(d['lambda_struct'],expected,rel_tol=1e-12,abs_tol=0.):
        raise ValueError('Calibration weight does not follow the one-shot downward rule')
    return d


def graph_review_read(common,calibration_file):
    root=Path(common);review=json.loads((root/'graph_review.json').read_text())
    if review.get('passed') is not True or review.get('calibration_sha256')!=sha256(calibration_file):
        raise ValueError('Initial graph review does not match frozen calibration')
    calibration=calibration_read(calibration_file)
    for (ep,start),record in zip(TRAIN_WINDOWS,calibration['records']):
        array=f'graph_ep{ep}_start{start}.npz';figure=f'initial_graph_ep{ep}_start{start}.png'
        if review.get('graph_array_sha256',{}).get(array)!=record['graph_array_sha256'] or sha256(root/'calibration'/array)!=record['graph_array_sha256']:
            raise ValueError('Reviewed initial graph array changed')
        if review.get('figure_sha256',{}).get(figure)!=sha256(root/'figures'/figure):
            raise ValueError('Reviewed graph visualization changed or missing')
    return review


def configure_runtime(threads=2):
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def resource_guard(*,require_idle=False,allowed_pids=()):
    if os.environ.get('SLURM_JOB_ID')!=JOB_ID or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Only authorized Allocation32087 Step permitted')
    job=subprocess.check_output(['scontrol','show','job',JOB_ID,'-o'],text=True)
    if 'UserId=tcweng(1025)' not in job or 'JobState=RUNNING' not in job:
        raise RuntimeError('Authorized allocation invalid or wrong owner')
    line=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(line)!=1:raise RuntimeError('Ambiguous GPU namespace')
    uuid,name,total,used,free=[s.strip() for s in line[0].split(',')]
    if uuid!=GPU_UUID or '5090' not in name:raise RuntimeError('Wrong authorized GPU UUID')
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    pids=[int(v.strip()) for v in raw.splitlines() if v.strip()]
    allowed={os.getpid(),*allowed_pids}
    foreign=[p for p in pids if p not in allowed]
    if require_idle and foreign:raise RuntimeError('Preserve other GPU processes: '+str(foreign))
    return dict(job_id=JOB_ID,step_id=os.environ['SLURM_STEP_ID'],gpu_uuid=uuid,gpu=name,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),slurm_step_gpus=os.environ.get('SLURM_STEP_GPUS'),
        total_mib=int(total),used_mib=int(used),free_mib=int(free),pids=pids,foreign_pids=foreign,
        reserve_mib=max(4096,math.ceil(int(total)*.15)),allocation_record=job.strip())


def environment():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    d=dict(torch=torch.__version__,cuda=torch.version.cuda,gsplat=importlib.metadata.version('gsplat'),gpu=torch.cuda.get_device_name(),tf32=False)
    if d['torch']!='2.7.1+cu128' or d['cuda']!='12.8' or d['gsplat']!='1.4.0' or '5090' not in d['gpu']:
        raise RuntimeError('Frozen CUDA/gsplat runtime differs')
    return d


def output_check(path,variant,resume=False):
    raw=Path(path);out=raw.resolve()
    if not raw.is_absolute() or raw.is_symlink() or out.parent.name!=BUCKETS[variant]:raise ValueError('Wrong independent variant output namespace')
    if resume:
        if not (out/'initial.pt').is_file():raise ValueError('No durable initial boundary')
    elif out.exists():raise FileExistsError(out)
    return out
