"""G4 formal checkpoints: immutable completed-epoch commits, recoverable aliases."""
import copy
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch

VERSION='touch4d-g4-formal-h15-epochs-v1'


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if not torch.cuda.is_available():raise ValueError('CUDA RNG resume requires CUDA')
        torch.cuda.set_rng_state_all(state['cuda'])


def finite(value):
    if isinstance(value,torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError('Nonfinite checkpoint tensor')
    if isinstance(value,dict):
        for v in value.values():finite(v)
    if isinstance(value,(list,tuple)):
        for v in value:finite(v)


def validate_state(s,expected):
    if s.get('version')!=VERSION or s.get('identity')!=expected:
        raise ValueError('Wrong formal G4 format/run/config/processor/SHA/manifest/environment')
    ledger=s['ledger'];done=ledger['completed_epochs'];history=ledger['history']
    if type(done) is not int or not 0<=done<=8 or len(history)!=done:
        raise ValueError('Only complete epoch boundaries may resume')
    if [h['epoch'] for h in history]!=list(range(1,done+1)):
        raise ValueError('Epoch sequence is not contiguous')
    if ledger['successful_updates']!=sum(h['successful_updates'] for h in history):
        raise ValueError('Successful update ledger mismatch')
    keys=expected['data']['manifest']['train']['window_keys']
    for h in history:
        ep=h['epoch']-1
        if h['window_order']!=[keys[i] for i in expected['data']['epoch_orders'][ep]]:
            raise ValueError('Checkpoint window sequence mismatch')
        if not 0<=h['successful_updates']<=len(keys):raise ValueError('Invalid update count')
        from .protocol import full_validation_score
        full_validation_score(h['validation'],expected['data']['manifest'])
        if h['validation_window_order']!=expected['data']['manifest']['val']['window_keys']:
            raise ValueError('Validation window sequence mismatch')
    if history:
        best=min(history,key=lambda h:h['validation']['model']['objective'])
        if ledger['best_epoch']!=best['epoch'] or ledger['best_score']!=best['validation']['model']['objective']:
            raise ValueError('Best score ledger mismatch')
    elif ledger['best_score'] is not None or ledger['best_epoch'] is not None:
        raise ValueError('Initial checkpoint cannot have a best score')
    finite(s['model']);finite(s['optimizer'])


def atomic_save(path,state,replace=False):
    path=Path(path)
    if path.exists() and not replace:raise FileExistsError(path)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
        if replace:os.replace(tmp,path)
        else:os.link(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def save_boundary(directory,model,optimizer,identity,ledger):
    root=Path(directory);n=ledger['completed_epochs']
    if n:
        prior=root/('initial.pt' if n==1 else f'epoch_{n-1:02d}.pt')
        previous=torch.load(prior,map_location='cpu',weights_only=False)
        validate_state(previous,identity)
        if previous['ledger']['history']!=ledger['history'][:-1]:
            raise ValueError('Committed history changed')
    state=dict(version=VERSION,identity=copy.deepcopy(identity),ledger=copy.deepcopy(ledger),
               model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state())
    validate_state(state,identity)
    name='initial.pt' if n==0 else f'epoch_{n:02d}.pt'
    atomic_save(Path(directory)/name,state)
    publish_aliases(directory,state)
    return state


def publish_aliases(directory,state):
    root=Path(directory);identity=state['identity']
    for name in ('last.pt','best.pt'):
        if (root/name).exists():
            old=torch.load(root/name,map_location='cpu',weights_only=False)
            validate_state(old,identity)
            if old['ledger']['completed_epochs']>state['ledger']['completed_epochs']:
                raise ValueError('Refusing stale rollback over newer checkpoint')
    atomic_save(root/'last.pt',state,replace=True)
    best=state['ledger']['best_epoch']
    if best is not None:
        best_state=torch.load(root/f'epoch_{best:02d}.pt',map_location='cpu',weights_only=False)
        validate_state(best_state,identity)
        atomic_save(root/'best.pt',best_state,replace=True)


def restore(directory,model,optimizer,identity):
    root=Path(directory)
    initial=root/'initial.pt'
    if not initial.exists():raise ValueError('No durable initial checkpoint; refuse unsafe resume')
    files=sorted(root.glob('epoch_*.pt'))
    if [p.name for p in files]!=[f'epoch_{i:02d}.pt' for i in range(1,len(files)+1)]:
        raise ValueError('Missing or malformed durable epoch')
    path=files[-1] if files else initial
    state=torch.load(path,map_location='cpu',weights_only=False)
    validate_state(state,identity)
    if state['ledger']['completed_epochs']!=len(files):raise ValueError('Filename/state mismatch')
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
    publish_aliases(root,state)  # recovers crash between durable epoch and best/last publication
    restore_rng(state['rng'])
    return state['ledger']


def load_for_evaluation(path,model,processor):
    s=torch.load(path,map_location='cpu',weights_only=False)
    if s.get('version')!=VERSION:raise ValueError('Not a formal G4 checkpoint')
    identity=s['identity'];validate_state(s,identity)
    if identity['model']!=model.config or identity['processor']!=processor:
        raise ValueError('Architecture/normalization mismatch')
    model.load_state_dict(s['model'],strict=True)
    return s
