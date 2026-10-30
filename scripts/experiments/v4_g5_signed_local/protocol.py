"""Frozen G5 paired protocol; actual index-only DataLoader probes read no GS."""
import copy
import json
import math
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from scripts.data_process import DataProcessor
from scripts.experiments.v4_g4_global_motion import protocol as g4
from scripts.experiments.full_training.v4_epochs import data_manifest
from scripts.dataloader.loader import make_dataloader
from scripts.experiments.v4_g4_global_motion.model import G4GlobalMotionPredictor
from .checkpoint import build_model
from .diagnostics import state_hash

BASE='51ab5c455929128c66941d3d3f75c1d4421c9507'
BUCKETS={v:f'v4-g5-{v}-h15-delta4-v1-training' for v in ('direct','propagate')}
GRAPH_FILES=tuple('scripts/experiments/v4_g5_signed_local/'+v for v in ('model.py','motion.py','propagation.py'))


def configuration(root,depth_root,sha,variant):
    if variant not in BUCKETS:raise ValueError('Expected direct or propagate')
    c=g4.configuration(root,depth_root,sha)
    return dict(c,architecture=f'v4_g5_signed_local_{variant}_v1',variant=variant,gradient_clip_norm=1.,prediction_interval=4)


def validate_configuration(c):
    if c!=configuration(c['root'],c['depth_root'],c['source_sha'],c['variant']):
        raise ValueError('Frozen G5 training protocol changed')


def probe_order(raw):
    """Clone real loader settings/generator; includes iterator base-seed consumption."""
    if raw.batch_size!=1 or raw.num_workers!=0 or raw.drop_last or raw.generator is None:
        raise ValueError('DataLoader settings differ from G4')
    if isinstance(raw.sampler,RandomSampler):
        if raw.sampler.replacement or raw.sampler.num_samples!=len(raw.dataset):raise ValueError('Wrong sampler')
        if raw.sampler.generator is not raw.generator:raise ValueError('Sampler uses a different generator')
        shuffle=True
    elif isinstance(raw.sampler,SequentialSampler):shuffle=False
    else:raise ValueError('Unsupported sampler')
    clone=torch.Generator().set_state(raw.generator.get_state().clone())
    proxy=DataLoader(range(len(raw.dataset)),batch_size=1,shuffle=shuffle,num_workers=0,drop_last=False,generator=clone)
    return [int(batch[0]) for batch in proxy]


def loader(c,split,epoch=0):
    if split not in ('train','val'):raise ValueError('No test split')
    return make_dataloader(c['scene'],split,root=c['root'],stage='all',batch_size=1,
        num_workers=0,seed=c['seed']+epoch,with_depth=True,depth_root=c['depth_root'],
        depth_cameras=c['depth_cameras'],horizon=15,prediction_interval=4,
        operation_hand_policy=c['operation_hand_policy'])


def preflight(c,reference):
    validate_configuration(c)
    # Verify unchanged historical protocol/data first, then build new sparse windows.
    _,old=g4.preflight(g4.configuration(c['root'],c['depth_root'],c['source_sha']),reference)
    raw={s:loader(c,s) for s in ('train','val')}
    manifest=json.loads(json.dumps(data_manifest(raw)))
    if manifest['train']['episodes']!={'1':131,'2':232,'4':304,'5':247,'7':165,'8':165} or manifest['val']['episodes']!={'3':46,'6':36}:
        raise ValueError('Frozen delta4 counts differ from inspected dataset')
    if set(manifest['train']['episodes'])!={'1','2','4','5','7','8'} or set(manifest['val']['episodes'])!={'3','6'}:
        raise ValueError('Wrong split')
    for split in raw:
        if not set(map(tuple,manifest[split]['window_keys'])) <= set(map(tuple,old['manifest'][split]['window_keys'])):
            raise ValueError('Sparse windows must be a subset of original starts')
    orders=[probe_order(raw['train'] if e==0 else loader(c,'train',e)) for e in range(8)]
    if orders!=[g4.epoch_order(len(raw['train'].dataset),e) for e in range(8)]:raise ValueError('Shuffle drift')
    if probe_order(raw['val'])!=list(range(len(raw['val'].dataset))):raise ValueError('Val order drift')
    return raw,dict(manifest=manifest,manifest_hash=g4.digest(manifest),epoch_orders=orders,
        epoch_order_hashes=[g4.digest(o) for o in orders],observed_epoch_order_hashes=[g4.digest(o) for o in orders],
        reference_sha=g4.G0_SHA,reference_hashes=g4.REFERENCES,
        original_delta1_manifest_hash=old['manifest_hash'],
        split_policy='unchanged full episodes, no contact filter; train stride1, val stride5; no test',
        time_policy='H15 delta4: conditions t+4*k, targets t+4*(k+1); original frame span60',
        order_policy='same DataLoader mechanism seed42+epoch; counts differ from delta1')


def paired_initialization(verify_outputs=True):
    torch.manual_seed(42);base=G4GlobalMotionPredictor(horizon=15,prediction_interval=4)
    torch.manual_seed(42);direct=build_model('direct',prediction_interval=4)
    torch.manual_seed(42);propagate=build_model('propagate',prediction_interval=4)
    shared=base.state_dict()
    for name,value in shared.items():
        if not torch.equal(value,direct.state_dict()[name]) or not torch.equal(value,propagate.state_dict()[name]):
            raise AssertionError('G4 shared initialization differs: '+name)
    for name,value in direct.direct_motion.state_dict().items():
        if not torch.equal(value,propagate.direct_motion.state_dict()[name]):raise AssertionError('Direct Mapping init differs')
    for model in (direct,propagate):
        if any(p.count_nonzero() for p in model.direct_motion.mapping[-1].parameters()):raise AssertionError('Local projection nonzero')
    exact=None
    if verify_outputs:
        from scripts.experiments.v4_stage1.fixtures import synthetic_batch
        from scripts.training.v4_runtime import prepare_batch
        x=prepare_batch(synthetic_batch(60,15,4),DataProcessor('001-rope'),base)['inputs']
        with torch.no_grad():
            baseline=base(x)
            exact={}
            for variant,model in (('direct',direct),('propagate',propagate)):
                prediction=model(x)
                exact[variant]={k:torch.equal(value,prediction[k]) for k,value in baseline.items()}
        if not all(all(v.values()) for v in exact.values()):raise AssertionError('Fresh H15 equality failed')
    return dict(seed=42,shared_exact=True,direct_mapping_exact=True,zero_local_projection=True,
        g4_shared_hash=state_hash(shared),direct_mapping_hash=state_hash(direct.direct_motion.state_dict()),
        variant_model_hashes={v:state_hash(m.state_dict()) for v,m in (('direct',direct),('propagate',propagate))},
        h15_all_attributes_exact=exact,forward_probe='CPU synthetic H15; no real data training',
        parameter_counts={k:sum(p.numel() for p in model.parameters()) for k,model in
                          (('g4',base),('direct',direct),('propagate',propagate))})


def fresh_model(variant):
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    model=build_model(variant,prediction_interval=4)
    if any(p.count_nonzero() for p in model.direct_motion.mapping[-1].parameters()):raise AssertionError('Not fresh')
    return model


def full_validation_score(validation,manifest):
    expected=manifest['val']['episodes']
    if sum(expected.values())!=manifest['val']['windows']:raise ValueError('Invalid manifest counts')
    for label in ('model','copy_initial'):
        result=validation[label];episodes=result['episodes']
        if set(episodes)!=set(expected):raise ValueError('Missing validation episode')
        scores=[]
        for ep,count in expected.items():
            row=episodes[ep]
            if row['samples']!=count or row['render_samples']!=count:raise ValueError('Incomplete Val/render samples')
            score=row['supervised_geometry']+row['rgb']+.1*row['alpha']
            if not math.isfinite(score):raise ValueError('Nonfinite validation')
            scores.append(score)
        equal_episode=sum(scores)/len(scores)
        if not math.isclose(result['objective'],equal_episode,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Selection must be episode-equal original Joint')
    return float(validation['model']['objective'])


def output_check(path,variant,*,resume=False,protected=()):
    raw=Path(path)
    if not raw.is_absolute() or raw.is_symlink():raise ValueError('Output must be an absolute non-symlink Run path')
    out=raw.resolve()
    if out.parent.name!=BUCKETS[variant]:raise ValueError('Output belongs to another variant/root')
    for other in (Path.cwd(),*(Path(p).resolve() for p in protected)):
        if out==other or other in out.parents:raise ValueError('Output overlaps protected code/data/reference')
    if resume:
        if not (out/'initial.pt').is_file():raise ValueError('No durable initial checkpoint')
    elif out.exists():raise FileExistsError(out)
    ancestor=out.parent
    while not ancestor.exists():ancestor=ancestor.parent
    usage=shutil.disk_usage(ancestor)
    if usage.free<4*1024**3:raise OSError('Less than 4GiB filesystem headroom for paired runs')
    return out,dict(output=str(out),exists=out.exists(),variant_root=str(out.parent),
                    filesystem_free_bytes=usage.free,budget_note='4GiB conservative pair budget; inspect user quota separately')


def pair_preflight(root,depth_root,sha,reference,outputs,verify_outputs=True):
    rows={}
    for variant in BUCKETS:
        c=configuration(root,depth_root,sha,variant);_,data=preflight(c,reference)
        _,storage=output_check(outputs[variant],variant,protected=(root,depth_root,reference))
        rows[variant]=dict(config=c,data=data,storage=storage)
    if rows['direct']['data']['observed_epoch_order_hashes']!=rows['propagate']['data']['observed_epoch_order_hashes']:
        raise AssertionError('Paired actual epoch orders differ')
    return dict(passed=True,variants=rows,initialization=paired_initialization(verify_outputs),
                training_started=False,actual_optimizer_updates=0,cuda_bit_reproducibility_claim=False)
