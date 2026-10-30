"""One frozen paired protocol. Neither variant changes the G5 computation graph."""
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import torch
from scripts.data_process import DataProcessor
from scripts.experiments.v4_g5_signed_local import protocol as original
from scripts.experiments.v4_g5_signed_local.diagnostics import sha256,state_hash

BASE='f978e0a29b6f0f0b6097c40684b949797a2f5d7b'
REFERENCE='/data1/userdata/tcweng/experiments/touch4d/v4-g5-direct-v1-training/run-001/best.pt'
REFERENCE_HASH='067984c4c286306094efeff2f28ed57e3dcdd0c333972ecc6feb4b982ee3a40c'
VARIANTS=('g5_arap','g5_arap_iso')
BUCKETS={v:'v4-'+v.replace('_','-')+'-v1-training' for v in VARIANTS}
ARCHITECTURES={v:'v4_'+v+'_supervision_v1' for v in VARIANTS}
TRAIN_WINDOWS=((2,141),(7,105))
SMOKE_ORDER=TRAIN_WINDOWS+TRAIN_WINDOWS
RULE='downward_once_two_training_full_h15_xyz_gradient_arap5percent_iso1percent'
STRUCTURE=dict(graph='initial_original_16nn_mutual_undirected',epsilon=1e-6,
    spacing_median='midpoint_even',length_lower_multiplier=.1,length_upper_multiplier=2.,
    arap='consecutive_local_SO3_no_grad_kabsch_normalized_center_residual',arap_denominator_epsilon=1e-12,
    iso='initial_edge_log_length_ratio_smooth_l1',iso_epsilon=1e-6,iso_beta=.1,
    reduction='valid_centers_or_edges_and_horizon_per_window_then_batch')


class CalibrationZeroGradientError(ValueError):
    """Inspect graph/solver/rigid-motion diagnostics; never silently pass a gate."""


def fresh_model(variant):
    if variant not in VARIANTS:raise ValueError('Unknown ARAP/ISO variant')
    model=original.fresh_model('direct')
    model.experiment_variant=variant  # Metadata only: no tensor, parameter, state or Forward change.
    return model


def paired_initialization(verify_outputs=True):
    models={v:fresh_model(v) for v in VARIANTS}
    a,b=(models[v] for v in VARIANTS)
    if a.config!=b.config or a.state_dict().keys()!=b.state_dict().keys():raise AssertionError('Paired G5 architectures differ')
    for name,value in a.state_dict().items():
        if not torch.equal(value,b.state_dict()[name]):raise AssertionError('Fresh shared parameter differs: '+name)
    if any(p.count_nonzero() for p in a.direct_motion.mapping[-1].parameters()):raise AssertionError('Fresh Local output must be zero')
    exact=None
    if verify_outputs:
        from scripts.experiments.v4_stage1.fixtures import synthetic_batch
        from scripts.training.v4_runtime import prepare_batch
        x=prepare_batch(synthetic_batch(40,15,1),DataProcessor('001-rope'),a)['inputs']
        with torch.no_grad():
            expected=a(x);other=b(x)
            exact={k:torch.equal(value,other[k]) for k,value in expected.items()}
        if not all(exact.values()):raise AssertionError('Paired fresh H15 differs')
    return dict(seed=42,shared_exact=True,network='unchanged original G5-Direct',
        variant_model_hashes={v:state_hash(m.state_dict()) for v,m in models.items()},
        parameter_counts={v:sum(p.numel() for p in m.parameters()) for v,m in models.items()},
        h15_all_attributes_exact=exact,initialization='Fresh seed42; no historical or Smoke weight load')


def configuration(root,depth_root,source,variant,lambda_arap=.01,lambda_iso=.001):
    if variant not in VARIANTS:raise ValueError('Unknown ARAP/ISO variant')
    if not all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in (lambda_arap,lambda_iso)):
        raise ValueError('Finite scalar regularizer weights required')
    if not 0<=lambda_arap<=.01 or not 0<=lambda_iso<=.001:raise ValueError('Weights may only decrease')
    if variant=='g5_arap':lambda_iso=0.
    return dict(original.configuration(root,depth_root,source,'direct'),variant=variant,
        architecture=ARCHITECTURES[variant],lambda_arap=float(lambda_arap),lambda_iso=float(lambda_iso),
        structure=copy.deepcopy(STRUCTURE),model_selection='complete_val_episode_equal_original_joint')


def common_configuration(c):
    expected=configuration(c['root'],c['depth_root'],c['source_sha'],c['variant'],c['lambda_arap'],c['lambda_iso'])
    if c!=expected:raise ValueError('Frozen ARAP/ISO training protocol changed')
    return original.configuration(c['root'],c['depth_root'],c['source_sha'],'direct')


def preflight(c,reference):
    raw,data=original.preflight(common_configuration(c),reference)
    keys=data['manifest']['train']['window_keys']
    if any(list(k) not in keys for k in TRAIN_WINDOWS):raise ValueError('Fixed Smoke/calibration train window absent')
    return raw,data


def source_guard(expected):
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source!=expected or subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Clean committed pinned source required')
    files=subprocess.check_output(['git','diff','--name-only',BASE,'HEAD'],text=True).splitlines()
    if any(not(p.startswith('scripts/experiments/v4_g5_arap_iso/') or p.startswith('scripts/tests/training/test_v4_g5_arap_iso')) for p in files):
        raise ValueError('Changes outside isolated ARAP/ISO experiment: '+str(files))
    return source,{p:sha256(p) for p in files}


def component_hashes():
    root=Path(__file__).parent
    return {p.name:sha256(p) for p in sorted(root.glob('*.py')) if p.stem in ('structure','arap','runner','calibrate','protocol')}


def calibrated_weights(records):
    if len(records)!=2 or [(r.get('episode'),r.get('start')) for r in records]!=list(TRAIN_WINDOWS):
        raise ValueError('Exactly the prescribed two train calibration records required')
    weights={}
    for term,default,cap in (('arap',.01,.05),('iso',.001,.01)):
        ratios=[]
        for row in records:
            gj,gr=row['joint_gradient_norm'],row[term+'_gradient_norm']
            if not all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) and v>=0 for v in (gj,gr)) or gj<=0:
                raise ValueError('Nonfinite/zero original Joint calibration gradient')
            if gr<=0:raise CalibrationZeroGradientError(f'{term} has zero gradient in ep{row["episode"]}/start{row["start"]}; inspect retained calibration diagnostics')
            ratio=gr/gj
            if not math.isfinite(ratio):raise ValueError('Nonfinite calibration gradient ratio')
            ratios.append(ratio)
        maximum=max(ratios)
        weights['lambda_'+term]=min(default,cap/maximum) if maximum else default
    return weights


def calibration_read(path):
    d=json.loads(Path(path).read_text())
    if d.get('passed') is not True or d.get('reference_sha256')!=REFERENCE_HASH or d.get('windows')!=[list(w) for w in TRAIN_WINDOWS]:
        raise ValueError('Calibration provenance/window mismatch')
    if d.get('rule')!=RULE or d.get('components')!=component_hashes():raise ValueError('Calibration math changed')
    expected=calibrated_weights(d.get('records',[]))
    for name,value in expected.items():
        if not isinstance(d.get(name),(int,float)) or not math.isclose(d[name],value,rel_tol=1e-12,abs_tol=0.):
            raise ValueError('Calibration does not follow downward-only rule')
    return d


def gate_read(common,calibration_file,source,data,initial):
    d=json.loads((Path(common)/'gate.json').read_text())
    calibration=calibration_read(calibration_file)
    if calibration.get('source_sha')!=source or calibration.get('manifest_hash')!=data['manifest_hash']:
        raise ValueError('Calibration execution source/Manifest differs from formal run')
    if d.get('passed') is not True or d.get('source_sha')!=source or d.get('calibration_sha256')!=sha256(calibration_file):
        raise ValueError('Same-source completed preparation gate required')
    if d.get('manifest_hash')!=data['manifest_hash'] or d.get('epoch_order_hashes')!=data['epoch_order_hashes'] or d.get('initialization')!=initial:
        raise ValueError('Gate Manifest/window order/paired initialization mismatch')
    _reviewed_graph_gate(common,calibration,d)
    for v in VARIANTS:
        proof=d.get('smoke',{}).get(v,{})
        path=Path(proof.get('completion_path',''))
        if not path.is_file() or proof.get('completion_sha256')!=sha256(path):raise ValueError('Smoke proof changed/missing')
        row=json.loads(path.read_text())
        if row.get('passed') is not True or type(row.get('successful_updates')) is not int or row['successful_updates']!=4 or row.get('variant')!=v or row.get('source_sha')!=source or row.get('calibration_sha256')!=sha256(calibration_file):
            raise ValueError('Both variants require exactly four same-source Smoke updates')
        if row.get('manifest_hash')!=data['manifest_hash'] or row.get('epoch_order_hashes')!=data['epoch_order_hashes']:
            raise ValueError('Smoke and formal Manifest/window order differ')
        if row.get('initialization',{}).get('variant_model_hashes')!=initial['variant_model_hashes']:
            raise ValueError('Smoke and formal fresh paired initialization differ')
    return d


def _reviewed_graph_gate(common,calibration,gate):
    root=Path(common).resolve();proof_file=Path(gate.get('graph_review_path','')).resolve()
    if proof_file!=root/'reviewed_graph_equivalence.json' or not proof_file.is_file() or sha256(proof_file)!=gate.get('graph_review_sha256'):
        raise ValueError('Reviewed graph payload proof missing or changed')
    proof=json.loads(proof_file.read_text())
    if proof.get('passed') is not True or proof.get('exact_initial_graph_payload') is not True:
        raise ValueError('Initial spatial graph must match the previously reviewed graph payload')
    review_file=Path(proof.get('review_path',''))
    if not review_file.is_file() or sha256(review_file)!=proof.get('review_sha256'):raise ValueError('Original graph review changed')
    review=json.loads(review_file.read_text())
    if review.get('passed') is not True:raise ValueError('Original graph review did not pass')
    expected={f'graph_ep{ep}_start{start}.npz' for ep,start in TRAIN_WINDOWS}
    if set(proof.get('arrays',{}))!=expected:raise ValueError('Both reviewed train-window graph payloads required')
    for (ep,start),record in zip(TRAIN_WINDOWS,calibration['records']):
        name=f'graph_ep{ep}_start{start}.npz';array=proof['arrays'][name]
        current=Path(array.get('current_path','')).resolve();old=Path(array.get('reviewed_path',''))
        if current!=root/'calibration'/name or array.get('same_payload') is not True:
            raise ValueError('Current reviewed graph path/payload differs')
        if not current.is_file() or sha256(current)!=array.get('current_sha256') or array.get('current_sha256')!=record['graph_array_sha256']:
            raise ValueError('Calibrated graph array changed after review')
        if not old.is_file() or sha256(old)!=array.get('reviewed_sha256') or review.get('graph_array_sha256',{}).get(name)!=array['reviewed_sha256']:
            raise ValueError('Original reviewed graph array changed')
        figure=f'initial_graph_ep{ep}_start{start}.png'
        if sha256(review_file.parent/'figures'/figure)!=review.get('figure_sha256',{}).get(figure):
            raise ValueError('Original reviewed graph figure changed')


def configure_runtime(threads=2):
    torch.set_num_threads(threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def _uuid(value):
    if isinstance(value,bytes) and len(value)==16:return value.hex()
    found=re.search(r'(?i)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{32})',str(value))
    return found[1].replace('-','').lower() if found else None


def resource_guard(*,require_idle=False,allowed_pids=(),inspect_cuda=True):
    job_id=os.environ.get('SLURM_JOB_ID','');step=os.environ.get('SLURM_STEP_ID','')
    if not job_id.isdigit() or not step or job_id=='32087':raise RuntimeError('Independent real Slurm job/step required; never Allocation32087')
    uid=os.getuid()
    job=subprocess.check_output(['scontrol','show','job',job_id,'-o'],text=True)
    if 'JobState=RUNNING' not in job or re.search(r'UserId=[^ ]+\('+str(uid)+r'\)',job) is None:
        raise RuntimeError('Slurm allocation not running or wrong owner')
    step_record=subprocess.check_output(['scontrol','show','step',job_id+'.'+step,'-o'],text=True)
    if re.search(r'\bStepId='+re.escape(job_id+'.'+step)+r'(?:\s|$)',step_record) is None or 'State=RUNNING' not in step_record:
        raise RuntimeError('Real running Slurm step required; login shell execution rejected')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    rows=[[v.strip() for v in line.split(',')] for line in raw.strip().splitlines()]
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(visible)!=1 or not visible[0]:raise RuntimeError('Exactly one explicitly allocated visible GPU required')
    authoritative=None
    if inspect_cuda:
        if torch.cuda.device_count()!=1:raise RuntimeError('Single CUDA device required')
        authoritative=_uuid(getattr(torch.cuda.get_device_properties(0),'uuid',None))
    step_gpu=os.environ.get('SLURM_STEP_GPUS','')
    if authoritative:
        selected=[r for r in rows if _uuid(r[1])==authoritative]
        mapping='authoritative_cuda_device_uuid'
    else:
        # A controlling preparation process must not initialize a CUDA context
        # while its GPU children enforce idle ownership. Never interpret CUDA
        # logical ordinal 0 as physical NVML index 0 after Slurm remapping.
        physical=step_gpu.split(',')
        visible_uuid=_uuid(visible[0])
        if len(physical)==1 and physical[0]:
            selected=[r for r in rows if r[0]==physical[0] or (_uuid(physical[0]) is not None and _uuid(r[1])==_uuid(physical[0]))]
            mapping='slurm_step_gpus_physical_or_uuid'
            one_gpu=(re.search(r'(?:^|[, ])gres/gpu=1(?:[, ]|$)',job) is not None or
                re.search(r'TresPerNode=gres/gpu(?::[^: ,]+)?:1(?:\s|$)',job) is not None)
            if not selected and len(rows)==1 and one_gpu:
                selected=rows;mapping='single_nvml_device_namespace_job_gpu_count_one'
        elif visible_uuid:
            selected=[r for r in rows if _uuid(r[1])==visible_uuid];mapping='visible_gpu_uuid'
        elif len(rows)==1:
            selected=rows;mapping='single_nvml_device_namespace'
        else:raise RuntimeError('Cannot map logical CUDA ordinal to allocated physical GPU')
    if len(selected)!=1 or '5090' not in selected[0][2]:raise RuntimeError('Unambiguous allocated RTX5090 required')
    index,uuid,name,total,used,free=selected[0]
    apps=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    pids=[int(v.strip()) for v in apps.splitlines() if v.strip()];foreign=[p for p in pids if p not in {os.getpid(),*allowed_pids}]
    if require_idle and foreign:raise RuntimeError('Preserve other GPU processes: '+str(foreign))
    reserve=max(4096,math.ceil(float(total)*.15))
    if require_idle and float(free)<reserve:raise RuntimeError('GPU safety reserve unavailable')
    return dict(job_id=job_id,step_id=step,gpu_uuid=uuid,gpu=name,physical_index=index,cuda_visible_devices=visible[0],
        slurm_step_gpus=step_gpu,mapping=mapping,authoritative_cuda_uuid=authoritative,
        total_mib=int(total),used_mib=int(used),free_mib=int(free),reserve_mib=reserve,foreign_pids=foreign,allocation_record=job.strip(),step_record=step_record.strip())


def environment():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    d=dict(torch=torch.__version__,cuda=torch.version.cuda,gsplat=importlib.metadata.version('gsplat'),gpu=torch.cuda.get_device_name(),tf32=False)
    if d['torch']!='2.7.1+cu128' or d['cuda']!='12.8' or d['gsplat']!='1.4.0' or '5090' not in d['gpu']:
        raise RuntimeError('Frozen CUDA/gsplat runtime differs')
    return d


def output_check(path,variant,resume=False):
    if variant not in VARIANTS:raise ValueError('Unknown variant')
    raw=Path(path);out=raw.resolve()
    if not raw.is_absolute() or raw.is_symlink() or out.parent.name!=BUCKETS[variant]:raise ValueError('Wrong independent variant namespace')
    if resume:
        if not(out/'initial.pt').is_file():raise ValueError('No durable initial boundary')
    elif out.exists():raise FileExistsError(out)
    return out
