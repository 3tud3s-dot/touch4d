"""Small immutable head-only snapshots, strictly bound to frozen backbone."""
import os
from pathlib import Path
import tempfile
import torch
from scripts.experiments.v4_g5_signed_local.diagnostics import finite_tree, state_hash

VERSION = 'touch4d-g5-geometry-smoke-head-only-v1'


def save(path, model, optimizer, *, identity, updates):
    if type(updates) is not int or updates not in range(5):
        raise ValueError('Only zero through four diagnostic updates permitted')
    if any(p.requires_grad for p in model.backbone.parameters()) or model.backbone.training:
        raise ValueError('Backbone must remain frozen and eval')
    payload = dict(version=VERSION, config=model.geometry_config, identity=identity,
        head=model.head.state_dict(), optimizer=optimizer.state_dict() if optimizer else None,
        backbone_state_hash=state_hash(model.backbone.state_dict()), updates=updates,
        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    finite_tree(payload)
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name+'.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def load(path, model, optimizer=None, *, identity):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('version') != VERSION or payload.get('config') != model.geometry_config
            or payload.get('identity') != identity):
        raise ValueError('Incompatible geometry checkpoint identity/configuration/version')
    if payload.get('backbone_state_hash') != state_hash(model.backbone.state_dict()):
        raise ValueError('Frozen backbone hash mismatch')
    if type(payload.get('updates')) is not int or payload['updates'] not in range(5):
        raise ValueError('Invalid diagnostic update count')
    finite_tree(payload)
    model.head.load_state_dict(payload['head'], strict=True)
    if optimizer is not None:
        if payload['optimizer'] is None: raise ValueError('Missing optimizer')
        optimizer.load_state_dict(payload['optimizer'])
        torch.set_rng_state(payload['rng'])
        if payload['cuda_rng'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return payload
