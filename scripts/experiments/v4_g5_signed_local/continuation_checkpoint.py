"""Separate Direct Epoch9-16 lineage. The original eight-Epoch format is frozen."""
import copy
from pathlib import Path
import torch
from scripts.experiments.v4_g4_global_motion.checkpoint import atomic_save,rng_state,restore_rng,finite
from scripts.experiments.v4_g4_global_motion.protocol import digest
from . import formal_checkpoint as original
from .protocol import configuration,full_validation_score
from .diagnostics import sha256,state_hash

PARENT_SHA='f978e0a29b6f0f0b6097c40684b949797a2f5d7b'
VERSION='touch4d-g5-direct-epoch9-16-continuation-v1'


def training_config(root,depth,source):
    return dict(configuration(root,depth,source,'direct'),epochs=16)


def load_parent(path,expected_hash,processor,manifest):
    if sha256(path)!=expected_hash:raise ValueError('Parent checkpoint SHA256 differs')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('version')!=original.VERSIONS['direct']:raise ValueError('Requires formal Direct Last8, never G4/Smoke/Propagate')
    identity=state['identity'];original.validate_state(state,identity)
    if identity['source_sha']!=PARENT_SHA or identity['processor']!=processor or identity['data']['manifest']!=manifest:
        raise ValueError('Parent source/processor/manifest differs')
    if state['ledger']['completed_epochs']!=8 or state['ledger']['best_epoch']!=8:
        raise ValueError('This frozen continuation starts from Last8 which is also Best8')
    if sha256(path)!=expected_hash:raise ValueError('Parent changed while reading')
    return state


def validate(state,identity):
    if state.get('version')!=VERSION or state.get('identity')!=identity:raise ValueError('Wrong continuation version/Run/source/config/lineage')
    c=identity['training']
    if c!=training_config(c['root'],c['depth_root'],identity['source_sha']):raise ValueError('Only budget and source can change')
    if identity.get('variant')!='direct' or identity['model'].get('architecture')!='v4_g5_signed_local_direct_v1' or identity['model'].get('horizon')!=15 or identity['model'].get('prediction_interval')!=1:
        raise ValueError('Direct H15/interval1 only')
    d=identity['data'];m=d['manifest'];orders=d['epoch_orders']
    if len(orders)!=16 or d['manifest_hash']!=digest(m) or d['epoch_order_hashes']!=[digest(v) for v in orders] or d['observed_epoch_order_hashes']!=d['epoch_order_hashes']:
        raise ValueError('Manifest/order provenance differs')
    parent=identity['parent_ledger'];ledger=state['ledger'];n=ledger['completed_epochs'];history=ledger['history']
    if identity['parent_source_sha']!=PARENT_SHA or parent['completed_epochs']!=8 or parent['best_epoch']!=8:
        raise ValueError('Not the frozen Last8 lineage')
    if type(n) is not int or not 8<=n<=16 or len(history)!=n or history[:8]!=parent['history']:
        raise ValueError('Missing or changed parent history / invalid boundary')
    if [v['epoch'] for v in history]!=list(range(1,n+1)):raise ValueError('Noncontiguous history')
    for row in history:
        if any(type(row[k]) is not int for k in ('successful_updates','visited_train_windows','visited_val_windows')):
            raise ValueError('Window/update counters must be integers')
        order=orders[row['epoch']-1]
        if sorted(order)!=list(range(m['train']['windows'])):raise ValueError('Not a permutation')
        if row['window_order']!=[m['train']['window_keys'][i] for i in order] or row['validation_window_order']!=m['val']['window_keys']:
            raise ValueError('Actual order differs')
        if row['partial_train'] is not False or row['partial_validation'] is not False:raise ValueError('Partial Epoch')
        flags=row['optimizer_update_flags']
        if len(flags)!=m['train']['windows'] or any(type(v) is not bool for v in flags) or sum(flags)!=row['successful_updates']:
            raise ValueError('Success/visit accounting differs')
        if row['visited_train_windows']!=m['train']['windows'] or row['visited_val_windows']!=m['val']['windows']:
            raise ValueError('Incomplete visits')
        full_validation_score(row['validation'],m)
    for key in ('successful_updates','visited_train_windows','visited_val_windows'):
        if type(ledger[key]) is not int or ledger[key]!=sum(v[key] for v in history):raise ValueError('Cumulative count differs')
    best=min(history,key=lambda v:v['validation']['model']['objective'])
    if ledger['best_epoch']!=best['epoch'] or ledger['best_score']!=best['validation']['model']['objective']:
        raise ValueError('Best not earliest full-Val minimum')
    if not {'python','numpy','torch','cuda'}<=state['rng'].keys():raise ValueError('Incomplete RNG')
    finite(state['model']);finite(state['optimizer'])


def read(path,identity):
    state=torch.load(path,map_location='cpu',weights_only=False);validate(state,identity);return state


def directory(root,identity):
    root=Path(root).resolve()
    if str(root)!=identity['run_id']:raise ValueError('Cross-Run write/restore')
    return root


def aliases(root,state):
    for name in ('last.pt','best.pt'):
        if (root/name).exists():
            old=read(root/name,state['identity']);n=old['ledger']['completed_epochs']
            if n>state['ledger']['completed_epochs'] or old['ledger']['history']!=state['ledger']['history'][:n]:
                raise ValueError('Newer/forked alias')
    best=state['ledger']['best_epoch']
    chosen=read(root/('initial.pt' if best==8 else f'epoch_{best:02d}.pt'),state['identity'])
    if chosen['ledger']['history']!=state['ledger']['history'][:best]:raise ValueError('Best chain mismatch')
    atomic_save(root/'last.pt',state,replace=True);atomic_save(root/'best.pt',chosen,replace=True)


def save(root,model,opt,identity,ledger):
    root=directory(root,identity);n=ledger['completed_epochs']
    if model.config!=identity['model']:raise ValueError('Model configuration changed')
    if n==8:
        if list(root.glob('*.pt')):raise FileExistsError('Continuation already initialized')
        if ledger!=identity['parent_ledger'] or state_hash(model.state_dict())!=identity['parent_model_hash']:
            raise ValueError('Not exact parent parameters/history')
    else:
        prior=read(root/('initial.pt' if n==9 else f'epoch_{n-1:02d}.pt'),identity)
        if prior['ledger']['history']!=ledger['history'][:-1]:raise ValueError('History fork')
    state=dict(version=VERSION,identity=copy.deepcopy(identity),ledger=copy.deepcopy(ledger),
        model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state())
    validate(state,identity);atomic_save(root/('initial.pt' if n==8 else f'epoch_{n:02d}.pt'),state);aliases(root,state)


def restore(root,model,opt,identity):
    root=directory(root,identity);state=read(root/'initial.pt',identity)
    if model.config!=identity['model']:raise ValueError('Restore model configuration differs')
    if state['ledger']!=identity['parent_ledger'] or state_hash(state['model'])!=identity['parent_model_hash']:
        raise ValueError('Continuation origin differs')
    files=sorted(root.glob('epoch_*.pt'))
    if [v.name for v in files]!=[f'epoch_{i:02d}.pt' for i in range(9,9+len(files))] or len(files)>8:
        raise ValueError('Incomplete/malformed Epoch sequence')
    for i,path in enumerate(files,9):
        nxt=read(path,identity)
        if nxt['ledger']['completed_epochs']!=i or nxt['ledger']['history'][:-1]!=state['ledger']['history']:
            raise ValueError('Noncontiguous Epoch commit')
        state=nxt
    model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
    aliases(root,state);restore_rng(state['rng'])
    return copy.deepcopy(state['ledger'])
