"""Full joint model/optimizer checkpoints; reject head-only/G5/G4 formats."""
import copy
from pathlib import Path
import torch
from scripts.experiments.v4_g4_global_motion.checkpoint import atomic_save, rng_state, restore_rng
from scripts.experiments.v4_g5_signed_local import formal_checkpoint as g5
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash, finite_tree
from .model import ARCHITECTURE
from .protocol import common_configuration
from scripts.experiments.v4_g5_geometry.geometry import LOG_SCALE_BOUND, ROT_COMPONENT_BOUND

VERSION = 'touch4d-g5-geo-joint-scratch-formal-h15-v1'
empty_ledger = g5.empty_ledger
commit_epoch = g5.commit_epoch


def validate(state, identity):
    if state.get('version') != VERSION or state.get('identity') != identity:
        raise ValueError('Wrong joint version, Run, source or configuration')
    if identity['model']['architecture'] != ARCHITECTURE:
        raise ValueError('Wrong joint model architecture')
    geo = identity['geometry_config']
    expected_geo = dict(architecture=ARCHITECTURE,widths=[131,128,64,6],activation='GELU',
        log_scale_bound=LOG_SCALE_BOUND,rotation_component_bound=ROT_COMPONENT_BOUND,
        quaternion='WXYZ_left_world',reference='initial',ar_feedback=False,
        detached_geometry_inputs=False,backbone_trainable=True)
    if geo != expected_geo:
        raise ValueError('Wrong joint gradient/feedback contract')
    # Only a validation VIEW is normalized to the original protocol. The actual
    # stored version, architecture and full weights remain independent and strict.
    common_id = dict(identity, variant='direct', training=common_configuration(identity['training']),
        model=dict(identity['model'], architecture='v4_g5_signed_local_direct_v1'))
    g5.validate_state(dict(state, version=g5.VERSIONS['direct'], identity=common_id), common_id)
    groups = state['optimizer']['param_groups']
    names = identity['optimizer_parameter_names']
    if len(groups)!=1 or groups[0]['lr']!=1e-4 or groups[0]['weight_decay']!=1e-4:
        raise ValueError('Joint AdamW configuration mismatch')
    if len(groups[0]['params'])!=len(names) or len(set(groups[0]['params']))!=len(names) or not any(n.startswith('backbone.motion.') for n in names) or not any(n.startswith('head.') for n in names):
        raise ValueError('Full G5 and Geometry optimizer state required')
    if state['ledger']['completed_epochs']==0 and state_hash(state['model']) != identity['initialization']['model_state_hash']:
        raise ValueError('Initial snapshot is not fresh verified weights')


def read(path, identity):
    state = torch.load(path, map_location='cpu', weights_only=False)
    validate(state, identity)
    return state


def bind(root, model, identity):
    root = Path(root).resolve()
    if str(root)!=identity['run_id'] or model.config!=identity['model'] or model.geometry_config!=identity['geometry_config']:
        raise ValueError('Cross-Run/model/config access forbidden')
    return root


def check_aliases(root, state):
    for name in ('best.pt','last.pt'):
        if (root/name).exists():
            old = read(root/name, state['identity'])
            n = old['ledger']['completed_epochs']
            if n>state['ledger']['completed_epochs'] or old['ledger']['history']!=state['ledger']['history'][:n]:
                raise ValueError('Newer/forked alias must not be overwritten')


def publish(root, state):
    check_aliases(root, state)
    atomic_save(root/'last.pt', state, replace=True)
    best = state['ledger']['best_epoch']
    if best is not None:
        chosen = read(root/f'epoch_{best:02d}.pt', state['identity'])
        if chosen['ledger']['history']!=state['ledger']['history'][:best]:
            raise ValueError('Best is not on the durable Epoch chain')
        atomic_save(root/'best.pt', chosen, replace=True)


def save_boundary(root, model, optimizer, identity, ledger):
    root = bind(root, model, identity)
    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError('Joint model unexpectedly frozen')
    if {id(p) for g in optimizer.param_groups for p in g['params']}!={id(p) for p in model.parameters()}:
        raise ValueError('Joint optimizer must cover every parameter')
    n = ledger['completed_epochs']
    if n==0:
        if list(root.glob('*.pt')):raise FileExistsError('Fresh Run already has snapshots')
    else:
        previous = read(root/('initial.pt' if n==1 else f'epoch_{n-1:02d}.pt'), identity)
        if previous['ledger']['history']!=ledger['history'][:-1]:raise ValueError('Immutable history changed')
    state = dict(version=VERSION, identity=copy.deepcopy(identity), ledger=copy.deepcopy(ledger),
        model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()), rng=rng_state())
    validate(state, identity); check_aliases(root, state)
    atomic_save(root/('initial.pt' if n==0 else f'epoch_{n:02d}.pt'), state)
    publish(root, state)
    return state


def restore(root, model, optimizer, identity):
    root = bind(root, model, identity)
    current = read(root/'initial.pt', identity)
    if current['ledger']!=empty_ledger():raise ValueError('Bad initial boundary')
    files = sorted(root.glob('epoch_*.pt'))
    if len(files)>8 or [p.name for p in files]!=[f'epoch_{i:02d}.pt' for i in range(1,len(files)+1)]:
        raise ValueError('Noncontiguous immutable Epoch chain')
    for i, path in enumerate(files, 1):
        state = read(path, identity)
        if state['ledger']['completed_epochs']!=i or state['ledger']['history'][:-1]!=current['ledger']['history']:
            raise ValueError('Epoch history mismatch')
        current = state
    check_aliases(root,current)
    model.load_state_dict(current['model'], strict=True)
    optimizer.load_state_dict(current['optimizer'])
    finite_tree((model.state_dict(),optimizer.state_dict()))
    publish(root,current);restore_rng(current['rng'])
    return copy.deepcopy(current['ledger'])
