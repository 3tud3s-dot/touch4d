"""Frozen G0 H15 protocol; CPU preflight never reads model checkpoints."""
import hashlib
import json
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader
from scripts.experiments.full_training.v4_epochs import configuration as g0_configuration, loader, data_manifest
from scripts.training.v4_model import V4Stage1Predictor
from .model import G4GlobalMotionPredictor

G0_SHA='60b7e6044a139a92bf7fdad29c262bde437ae2f9'
REFERENCES={'config.json':'b3a4cff85fe6e8c3f9a4f29fc15fe63107908720555c3342163f77338b2e5401',
            'manifest.json':'cca3e5ace2e3f279871b89a9fee84ed0a1524c8178229d1f45497ec6dfa4e8a3'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def configuration(root,depth_root,sha):
    c=g0_configuration(root,depth_root,sha,horizon=15)
    c['architecture']='v4_g4_global_motion_v1'
    return c


def epoch_order(size,epoch,seed=42):
    # DataLoader consumes its base seed BEFORE RandomSampler's randperm.
    # Iterating sampler alone would silently give a different G0 order.
    dl=DataLoader(range(size),batch_size=1,shuffle=True,num_workers=0,
                  generator=torch.Generator().manual_seed(seed+epoch))
    return [int(i.item()) for i in dl]


def fresh_model():
    random.seed(42);torch.manual_seed(42)
    g0=V4Stage1Predictor(horizon=15,prediction_interval=1)
    torch.manual_seed(42)
    g4=G4GlobalMotionPredictor(horizon=15,prediction_interval=1)
    if not all(torch.equal(v,g4.state_dict()[k]) for k,v in g0.state_dict().items()):
        raise AssertionError('Shared initialization mismatch')
    if any(torch.count_nonzero(p) for p in g4.global_motion.mapping[-1].parameters()):
        raise AssertionError('Global projection must be zero')
    evidence=dict(seed=42,shared_exact=True,new_output_zero=True,
                  shared_parameters=sum(p.numel() for p in g0.parameters()),
                  new_parameters=sum(p.numel() for p in g4.global_motion.parameters()))
    return g4,evidence


def preflight(c,reference):
    ref=Path(reference)
    for name,expected in REFERENCES.items():
        if hashlib.sha256((ref/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen G0 evidence differs: '+name)
    old=json.loads((ref/'config.json').read_text())['training']
    expected=dict(old,source_sha=c['source_sha'],architecture=c['architecture'])
    if c!=expected: raise ValueError('Training config differs beyond architecture/source SHA')
    loaders={s:loader(c,s) for s in ('train','val')}
    manifest=json.loads(json.dumps(data_manifest(loaders)))  # JSON episode keys are strings.
    if manifest!=json.loads((ref/'manifest.json').read_text()):
        raise ValueError('Dataset window manifest differs from actual G0 H15 run')
    orders=[epoch_order(len(loaders['train'].dataset),ep,c['seed']) for ep in range(c['epochs'])]
    return loaders,dict(manifest=manifest,manifest_hash=digest(manifest),epoch_orders=orders,
        epoch_order_hashes=[digest(o) for o in orders],reference_sha=G0_SHA,reference_hashes=REFERENCES,
        split_policy='train ep1,2,4,5,7,8 stride1; val ep3,6 stride5; no test; no contact filter',
        order_policy='original DataLoader generator seed42+epoch including iterator base-seed consumption')


class CheckedStream:
    def __init__(self,raw_loader,expected):self.loader,self.expected,self.seen=raw_loader,expected,[]
    def __iter__(self):
        for batch in self.loader:
            key=[int(batch['meta']['episode_id'][0]),int(batch['meta']['frame_ids'][0,0])]
            i=len(self.seen)
            if i>=len(self.expected) or key!=self.expected[i]:raise ValueError('Window order drift')
            self.seen.append(key)
            yield batch
        if self.seen!=self.expected:raise ValueError('Incomplete window traversal')


def full_validation_score(validation,manifest):
    for mode in ('model','copy_initial'):
        got=validation[mode]['episodes'];expected=manifest['val']['episodes']
        if set(got)!=set(expected) or any(got[k]['samples']!=v for k,v in expected.items()):
            raise ValueError('Model selection requires complete validation episodes')
    score=float(validation['model']['objective'])
    if not torch.isfinite(torch.tensor(score)):raise ValueError('Nonfinite full validation joint score')
    return score
