"""Finite, immutable checkpoint IO, RNG and identity helpers for the retained model."""
import hashlib
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch


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


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_tree(x,y) for x,y in zip(a,b))
    return a == b


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
