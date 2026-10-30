"""Independent head-only formal Epoch commits; Smoke snapshots cannot resume."""
import copy
from pathlib import Path
import torch
from scripts.experiments.v4_g4_global_motion.checkpoint import atomic_save,rng_state,restore_rng
from scripts.experiments.v4_g5_signed_local.formal_checkpoint import empty_ledger,commit_epoch
from scripts.experiments.v4_g5_signed_local.protocol import full_validation_score,validate_configuration
from scripts.experiments.v4_g4_global_motion.protocol import digest
from scripts.experiments.v4_g5_signed_local.diagnostics import finite_tree,state_hash
from .model import ARCHITECTURE

VERSION='touch4d-g5-geometry-head-only-formal-epochs-v1'


def validate(state,identity):
    if state.get('version')!=VERSION or state.get('identity')!=identity:
        raise ValueError('Wrong Geometry formal format, source, Run or configuration')
    if identity['geometry_config']['architecture']!=ARCHITECTURE or identity['geometry_config']['ar_feedback'] is not False:
        raise ValueError('Wrong output-only geometry architecture')
    validate_configuration(identity['training'])
    if identity['training']['source_sha']!=identity['source_sha']:raise ValueError('Source mismatch')
    if state['backbone_state_hash']!=identity['backbone_state_hash']:raise ValueError('Frozen backbone changed')
    data=identity['data'];manifest=data['manifest'];ledger=state['ledger'];history=ledger['history']
    n=ledger['completed_epochs']
    if type(n) is not int or not 0<=n<=8 or len(history)!=n or [r['epoch'] for r in history]!=list(range(1,n+1)):
        raise ValueError('Only complete sequential Epoch boundaries may resume')
    if data['manifest_hash']!=digest(manifest) or data['epoch_order_hashes']!=[digest(v) for v in data['epoch_orders']]:
        raise ValueError('Manifest/order hashes differ')
    if len(data['epoch_orders'])!=8:raise ValueError('Eight shuffle orders required')
    train=manifest['train']['window_keys'];val=manifest['val']['window_keys']
    for row in history:
        order=data['epoch_orders'][row['epoch']-1]
        if sorted(order)!=list(range(len(train))) or row['window_order']!=[train[i] for i in order] or row['validation_window_order']!=val:
            raise ValueError('Window order drift')
        if row['visited_train_windows']!=len(train) or row['visited_val_windows']!=len(val):raise ValueError('Incomplete Epoch')
        flags=row['optimizer_update_flags']
        if len(flags)!=len(train) or any(type(v) is not bool for v in flags) or sum(flags)!=row['successful_updates']:
            raise ValueError('Visit/update count mismatch')
        if row['partial_train'] is not False or row['partial_validation'] is not False:raise ValueError('Partial Epoch')
        full_validation_score(row['validation'],manifest)
    for name in ('successful_updates','visited_train_windows','visited_val_windows'):
        if type(ledger[name]) is not int or ledger[name]!=sum(r[name] for r in history):raise ValueError('Bad cumulative counter')
    if history:
        best=min(history,key=lambda r:r['validation']['model']['objective'])
        if ledger['best_epoch']!=best['epoch'] or ledger['best_score']!=best['validation']['model']['objective']:raise ValueError('Wrong Best rule')
    elif ledger['best_epoch'] is not None or ledger['best_score'] is not None:raise ValueError('No initial Best')
    if set(state['rng'])!={'python','numpy','torch','cuda'}:raise ValueError('Incomplete RNG')
    groups=state['optimizer']['param_groups']
    if len(groups)!=1 or groups[0]['lr']!=1e-4 or groups[0]['weight_decay']!=1e-4:
        raise ValueError('Frozen head optimizer configuration changed')
    finite_tree((state['head'],state['optimizer']))


def read(path,identity):
    state=torch.load(path,map_location='cpu',weights_only=False);validate(state,identity);return state


def aliases(root,state):
    for name in ('best.pt','last.pt'):
        path=root/name
        if path.exists():
            old=read(path,state['identity'])
            if old['ledger']['completed_epochs']>state['ledger']['completed_epochs']:raise ValueError('Alias is newer than durable boundary')
    atomic_save(root/'last.pt',state,replace=True)
    if state['ledger']['best_epoch'] is not None:
        best=read(root/f"epoch_{state['ledger']['best_epoch']:02d}.pt",state['identity'])
        atomic_save(root/'best.pt',best,replace=True)


def save_boundary(root,model,optimizer,identity,ledger):
    root=Path(root)
    if str(root.resolve())!=identity['run_id'] or model.geometry_config!=identity['geometry_config']:
        raise ValueError('Wrong Run/model')
    if model.backbone.training or any(p.requires_grad for p in model.backbone.parameters()):raise ValueError('Backbone not frozen')
    if {id(p) for g in optimizer.param_groups for p in g['params']}!={id(p) for p in model.head.parameters()}:
        raise ValueError('Optimizer must contain head only')
    n=ledger['completed_epochs']
    if n:
        previous=read(root/('initial.pt' if n==1 else f'epoch_{n-1:02d}.pt'),identity)
        if previous['ledger']['history']!=ledger['history'][:-1]:raise ValueError('Immutable history changed')
    state=dict(version=VERSION,identity=copy.deepcopy(identity),ledger=copy.deepcopy(ledger),
        head=model.head.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),
        backbone_state_hash=state_hash(model.backbone.state_dict()))
    validate(state,identity)
    atomic_save(root/('initial.pt' if n==0 else f'epoch_{n:02d}.pt'),state)
    aliases(root,state)
    return state


def restore(root,model,optimizer,identity):
    root=Path(root)
    if str(root.resolve())!=identity['run_id'] or model.geometry_config!=identity['geometry_config']:
        raise ValueError('Wrong Run/model')
    if state_hash(model.backbone.state_dict())!=identity['backbone_state_hash']:raise ValueError('Wrong frozen backbone')
    current=read(root/'initial.pt',identity)
    if current['ledger']!=empty_ledger():raise ValueError('Invalid initial boundary')
    files=sorted(root.glob('epoch_*.pt'))
    if len(files)>8 or [p.name for p in files]!=[f'epoch_{i:02d}.pt' for i in range(1,len(files)+1)]:raise ValueError('Epoch gap')
    for i,path in enumerate(files,1):
        state=read(path,identity)
        if state['ledger']['completed_epochs']!=i or state['ledger']['history'][:-1]!=current['ledger']['history']:
            raise ValueError('Durable Epoch chain mismatch')
        current=state
    model.head.load_state_dict(current['head'],strict=True);optimizer.load_state_dict(current['optimizer'])
    aliases(root,current);restore_rng(current['rng'])
    return copy.deepcopy(current['ledger'])
