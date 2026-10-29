"""Independent architecture checkpoints; not a formal eight-epoch resume runner."""
import os
from pathlib import Path
import tempfile
import torch
from .model import G5DirectPredictor,G5PropagatePredictor

VERSIONS={'v4_g5_signed_local_direct_v1':'touch4d-g5-direct-architecture-v1',
          'v4_g5_signed_local_propagate_v1':'touch4d-g5-propagate-architecture-v1'}


def build_model(variant,*,horizon=15,prediction_interval=1):
    if variant not in ('direct','propagate'):raise ValueError('variant must be direct or propagate')
    cls=G5DirectPredictor if variant=='direct' else G5PropagatePredictor
    return cls(horizon=horizon,prediction_interval=prediction_interval)


def _finite(value):
    if isinstance(value,torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError('Nonfinite checkpoint tensor')
    if isinstance(value,dict):
        for v in value.values():_finite(v)
    if isinstance(value,(list,tuple)):
        for v in value:_finite(v)


def save_checkpoint(path,model,optimizer=None,*,source_sha,processor_state=None,successful_steps=0):
    if model.config.get('architecture') not in VERSIONS:raise ValueError('Expected G5 architecture')
    if type(successful_steps) is not int or successful_steps<0:raise ValueError('Invalid successful_steps')
    state=dict(version=VERSIONS[model.config['architecture']],config=model.config,
        model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),
        source_sha=source_sha,processor=processor_state,successful_steps=successful_steps,
        scope='architecture/smoke snapshot only; formal epoch resume protocol not implemented here')
    _finite(state['model']);_finite(state['optimizer'])
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
        os.link(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def load_checkpoint(path,model,optimizer=None,*,processor_state=None,source_sha=None):
    if model.config.get('architecture') not in VERSIONS:raise ValueError('Expected G5 architecture')
    s=torch.load(path,map_location='cpu',weights_only=False)
    if s.get('version')!=VERSIONS.get(model.config.get('architecture')) or s.get('config')!=model.config:
        raise ValueError('Wrong architecture/variant/H/interval/checkpoint format')
    if processor_state is not None and s.get('processor')!=processor_state:
        raise ValueError('Processor mismatch')
    if source_sha is not None and s.get('source_sha')!=source_sha:raise ValueError('Source SHA mismatch')
    if type(s.get('successful_steps')) is not int or s['successful_steps']<0:
        raise ValueError('Invalid step count')
    if optimizer is not None and s.get('optimizer') is None:raise ValueError('No optimizer state')
    _finite(s['model']);_finite(s.get('optimizer'))
    model.load_state_dict(s['model'],strict=True)
    if optimizer is not None:optimizer.load_state_dict(s['optimizer'])
    return s
