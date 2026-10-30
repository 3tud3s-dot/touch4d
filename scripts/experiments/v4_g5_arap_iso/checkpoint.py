"""C/D full-state immutable Epoch commits with strict calibrated identities.

Only validation of the COMMON manifest/order/ledger schema is delegated to
the proven G5 checkpoint code. The validation view is never persisted: actual
C/D versions, architecture, calibration, lambda and weights stay independent.
"""
import copy
from pathlib import Path
import re

import torch

from scripts.experiments.v4_g4_global_motion.checkpoint import atomic_save, rng_state, restore_rng, finite
from scripts.experiments.v4_g5_signed_local import formal_checkpoint as original
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash
from .protocol import VARIANTS


VERSIONS = {variant: f'touch4d-{variant.replace("_", "-")}-formal-h15-epochs-v1' for variant in VARIANTS}
empty_ledger = original.empty_ledger
commit_epoch = original.commit_epoch


def _common_configuration(config):
    # Imported lazily so independent CPU checkpoint tools can inspect metadata
    # without constructing datasets or starting the formal runner.
    from .protocol import common_configuration
    return common_configuration(config)


def _parameter_signature(model):
    return [dict(name=name, shape=list(value.shape), dtype=str(value.dtype))
            for name, value in model.named_parameters()]


def _optimizer_matches(model, optimizer):
    parameters = list(model.parameters())
    actual = [p for group in optimizer.param_groups for p in group['params']]
    if any(not p.requires_grad for p in parameters):
        raise ValueError('Formal C/D training requires every model parameter trainable')
    if [id(p) for p in actual] != [id(p) for p in parameters]:
        raise ValueError('Optimizer must cover every C/D model parameter exactly once in model order')


def validate_state(state, identity):
    if not isinstance(state, dict) or not isinstance(identity, dict):
        raise ValueError('Formal checkpoint and identity must be dictionaries')
    variant = identity.get('variant')
    if variant not in VERSIONS or state.get('version') != VERSIONS[variant] or state.get('identity') != identity:
        raise ValueError('Wrong ARAP/ISO variant/version/Run/source/configuration')
    model = identity['model']
    if model.get('architecture') != 'v4_g5_signed_local_direct_v1' or model.get('horizon') != 15 or model.get('prediction_interval') != 1:
        raise ValueError('Wrong ARAP/ISO architecture identity')
    if identity['training']['variant'] != variant or identity['source_sha'] != identity['training']['source_sha']:
        raise ValueError('Training variant/source differs from checkpoint identity')
    calibration = identity.get('calibration_sha256')
    if not isinstance(calibration, str) or re.fullmatch('[0-9a-f]{64}', calibration) is None:
        raise ValueError('Frozen calibration SHA256 required')
    initial = identity['initialization']['variant_model_hashes']
    if variant not in initial or re.fullmatch('[0-9a-f]{64}', initial[variant]) is None:
        raise ValueError('Verified fresh paired initialization hash required')
    common_id = dict(identity, variant='direct', training=_common_configuration(identity['training']),
                     model=model)
    original.validate_state(dict(state, version=original.VERSIONS['direct'], identity=common_id), common_id)
    signature = state.get('parameter_signature')
    if not isinstance(signature, list) or not signature or len({row['name'] for row in signature}) != len(signature):
        raise ValueError('Complete parameter signature required')
    for row in signature:
        value = state['model'].get(row['name'])
        if not isinstance(value, torch.Tensor) or list(value.shape) != row['shape'] or str(value.dtype) != row['dtype']:
            raise ValueError('Parameter signature does not match model tensors')
    groups = state['optimizer']['param_groups']
    if len(groups) != 1:
        raise ValueError('Original single-group AdamW required')
    expected = dict(lr=1e-4, weight_decay=1e-4, betas=(.9,.999), eps=1e-8,
                    amsgrad=False, maximize=False, capturable=False, differentiable=False,
                    foreach=None, fused=None)
    if any(groups[0].get(name) != value for name, value in expected.items()):
        raise ValueError('Frozen AdamW hyperparameters differ')
    indices = groups[0]['params']
    if indices != list(range(len(signature))):
        raise ValueError('Incomplete or repeated optimizer parameter state')
    if state['ledger']['completed_epochs'] == 0 and state_hash(state['model']) != initial[variant]:
        raise ValueError('Initial weights are not the verified fresh paired initialization')


def read(path, identity):
    state = torch.load(path, map_location='cpu', weights_only=False)
    validate_state(state, identity)
    return state


def bind_directory(directory, identity):
    root = Path(directory).resolve()
    if str(root) != identity['run_id']:
        raise ValueError('Cross-Run checkpoint save/restore forbidden')
    return root


def _model_matches(model, identity):
    if model.config != identity['model'] or getattr(model, 'experiment_variant', None) != identity['variant']:
        raise ValueError('Checkpoint model/variant configuration mismatch')


def check_aliases(root, state):
    for name in ('best.pt', 'last.pt'):
        if (root/name).exists():
            old = read(root/name, state['identity'])
            count = old['ledger']['completed_epochs']
            if count > state['ledger']['completed_epochs']:
                raise ValueError('Refuse stale rollback over a newer alias')
            if old['ledger']['history'] != state['ledger']['history'][:count]:
                raise ValueError('Alias belongs to a forked Epoch history')


def publish_aliases(root, state):
    check_aliases(root, state)
    atomic_save(root/'last.pt', state, replace=True)
    best = state['ledger']['best_epoch']
    if best is not None:
        chosen = read(root/f'epoch_{best:02d}.pt', state['identity'])
        if chosen['ledger']['history'] != state['ledger']['history'][:best]:
            raise ValueError('Best is not on the durable Epoch chain')
        atomic_save(root/'best.pt', chosen, replace=True)


def save_boundary(directory, model, optimizer, identity, ledger):
    root = bind_directory(directory, identity)
    _model_matches(model, identity); _optimizer_matches(model, optimizer)
    count = ledger['completed_epochs']
    if count == 0:
        if list(root.glob('*.pt')):
            raise FileExistsError('Fresh Run already contains checkpoints')
    else:
        previous = read(root/('initial.pt' if count == 1 else f'epoch_{count-1:02d}.pt'), identity)
        if previous['ledger']['history'] != ledger['history'][:-1]:
            raise ValueError('Immutable committed history changed')
    state = dict(version=VERSIONS[identity['variant']], identity=copy.deepcopy(identity),
                 ledger=copy.deepcopy(ledger), parameter_signature=_parameter_signature(model),
                 model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                 rng=copy.deepcopy(rng_state()))
    validate_state(state, identity); check_aliases(root, state)
    atomic_save(root/('initial.pt' if count == 0 else f'epoch_{count:02d}.pt'), state)
    publish_aliases(root, state)
    return state


def restore(directory, model, optimizer, identity, *, rollback_info=None):
    """Restore last COMPLETE Epoch; caller separately records uncommitted work.

    ``rollback_info`` is an optional output dict, not part of the committed
    ledger. A boundary snapshot cannot infer how many interrupted steps ran;
    the launcher/runner must retain its attempt/update logs for that count.
    """
    root = bind_directory(directory, identity)
    _model_matches(model, identity); _optimizer_matches(model, optimizer)
    current = read(root/'initial.pt', identity)
    if current['ledger'] != empty_ledger():
        raise ValueError('Invalid initial boundary')
    files = sorted(root.glob('epoch_*.pt'))
    if len(files) > 8 or [p.name for p in files] != [f'epoch_{i:02d}.pt' for i in range(1,len(files)+1)]:
        raise ValueError('Noncontiguous immutable Epoch files')
    for count, path in enumerate(files, 1):
        state = read(path, identity)
        if state['ledger']['completed_epochs'] != count or state['ledger']['history'][:-1] != current['ledger']['history']:
            raise ValueError('Durable Epoch history/filename mismatch')
        current = state
    check_aliases(root, current)
    if current['parameter_signature'] != _parameter_signature(model):
        raise ValueError('Parameter signature differs from current model')
    model.load_state_dict(current['model'], strict=True)
    optimizer.load_state_dict(current['optimizer'])
    finite(model.state_dict()); finite(optimizer.state_dict())
    publish_aliases(root, current); restore_rng(current['rng'])
    if rollback_info is not None:
        rollback_info.update(restored_completed_epochs=current['ledger']['completed_epochs'],
            restored_successful_updates=current['ledger']['successful_updates'],
            restored_checkpoint=str(files[-1] if files else root/'initial.pt'),
            uncommitted_epoch_policy='rollback and replay; count discarded/replayed work from retained attempt logs',
            uncommitted_update_count_inferred_from_checkpoint=False)
    return copy.deepcopy(current['ledger'])


def load_for_evaluation(path, model, processor, *, source_sha, manifest, calibration_sha256=None):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(state, dict) or state.get('version') not in VERSIONS.values():
        raise ValueError('Not a formal ARAP/ISO checkpoint')
    identity = state['identity']; validate_state(state, identity); _model_matches(model, identity)
    if identity['processor'] != processor or identity['source_sha'] != source_sha or identity['data']['manifest'] != manifest:
        raise ValueError('Evaluation normalization/source/Manifest mismatch')
    if calibration_sha256 is not None and identity['calibration_sha256'] != calibration_sha256:
        raise ValueError('Evaluation calibration mismatch')
    if state['parameter_signature'] != _parameter_signature(model):
        raise ValueError('Evaluation parameter signature mismatch')
    model.load_state_dict(state['model'], strict=True)
    return state
