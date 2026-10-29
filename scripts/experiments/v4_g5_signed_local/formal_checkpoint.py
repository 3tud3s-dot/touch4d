"""Variant-specific immutable Epoch commits. Smoke/G4/G0 formats never resume."""
import copy
from pathlib import Path
import torch
from scripts.experiments.v4_g4_global_motion.checkpoint import atomic_save,rng_state,restore_rng,finite
from .protocol import validate_configuration,full_validation_score
from scripts.experiments.v4_g4_global_motion.protocol import digest
from .diagnostics import state_hash

VERSIONS={v:f'touch4d-g5-{v}-formal-h15-epochs-v1' for v in ('direct','propagate')}


def empty_ledger():
    return dict(completed_epochs=0,successful_updates=0,visited_train_windows=0,visited_val_windows=0,
                best_epoch=None,best_score=None,history=[])


def validate_state(state,identity):
    variant=identity['variant']
    if variant not in VERSIONS or state.get('version')!=VERSIONS[variant] or state.get('identity')!=identity:
        raise ValueError('Wrong formal variant/version/source/Run/config/manifest/environment')
    validate_configuration(identity['training'])
    if identity['source_sha']!=identity['training']['source_sha'] or variant!=identity['training']['variant']:
        raise ValueError('Identity source/variant disagrees')
    if identity['model']['architecture']!=f'v4_g5_signed_local_{variant}_v1' or identity['model']['horizon']!=15 or identity['model']['prediction_interval']!=1:
        raise ValueError('Not the frozen G5 H15 architecture')
    ledger=state['ledger'];done=ledger['completed_epochs'];history=ledger['history']
    if type(done) is not int or not 0<=done<=8 or len(history)!=done or [r['epoch'] for r in history]!=list(range(1,done+1)):
        raise ValueError('Only contiguous completed Epoch boundaries are durable')
    data=identity['data'];manifest=data['manifest'];train=manifest['train']['window_keys'];val=manifest['val']['window_keys']
    if len(data['epoch_orders'])!=8:raise ValueError('Eight frozen orders required')
    hashes=[digest(v) for v in data['epoch_orders']]
    if data.get('manifest_hash')!=digest(manifest) or data.get('epoch_order_hashes')!=hashes or data.get('observed_epoch_order_hashes')!=hashes:
        raise ValueError('Manifest/order hashes are inconsistent')
    for row in history:
        if any(type(row[key]) is not int for key in ('successful_updates','visited_train_windows','visited_val_windows')):
            raise ValueError('Counters must be integer values, not booleans')
        order=data['epoch_orders'][row['epoch']-1]
        if sorted(order)!=list(range(len(train))):raise ValueError('Invalid permutation')
        if row['window_order']!=[train[i] for i in order] or row['validation_window_order']!=val:
            raise ValueError('Actual window order differs from frozen order')
        if row['visited_train_windows']!=len(train) or row['visited_val_windows']!=len(val):raise ValueError('Partial Epoch')
        if row.get('partial_train') is not False or row.get('partial_validation') is not False:raise ValueError('Partial Epoch flags')
        flags=row['optimizer_update_flags']
        if len(flags)!=len(train) or any(type(v) is not bool for v in flags) or sum(flags)!=row['successful_updates']:
            raise ValueError('Visits and successful optimizer updates disagree')
        full_validation_score(row['validation'],manifest)
    for key in ('successful_updates','visited_train_windows','visited_val_windows'):
        if type(ledger[key]) is not int or ledger[key]!=sum(row[key] for row in history):raise ValueError('Ledger counter mismatch: '+key)
    if history:
        best=min(history,key=lambda r:r['validation']['model']['objective'])
        if ledger['best_epoch']!=best['epoch'] or ledger['best_score']!=best['validation']['model']['objective']:
            raise ValueError('Best must be earliest minimum complete-Val Joint')
    elif ledger['best_epoch'] is not None or ledger['best_score'] is not None:raise ValueError('Initial cannot be selected Best')
    finite(state['model']);finite(state['optimizer'])
    if not {'python','numpy','torch','cuda'}<=state['rng'].keys():raise ValueError('Incomplete RNG snapshot')


def bind_directory(directory,identity):
    root=Path(directory).resolve()
    if str(root)!=identity['run_id']:raise ValueError('Cross-Run directory restore/save forbidden')
    return root


def read(path,identity):
    state=torch.load(path,map_location='cpu',weights_only=False);validate_state(state,identity)
    return state


def check_aliases(root,state):
    for name in ('best.pt','last.pt'):
        if (root/name).exists():
            old=read(root/name,state['identity'])
            if old['ledger']['completed_epochs']>state['ledger']['completed_epochs']:
                raise ValueError('Refuse stale rollback over a newer alias')
            if old['ledger']['history']!=state['ledger']['history'][:old['ledger']['completed_epochs']]:
                raise ValueError('Alias history belongs to a fork, not the durable chain')


def publish_aliases(root,state):
    check_aliases(root,state)
    atomic_save(root/'last.pt',state,replace=True)
    best=state['ledger']['best_epoch']
    if best is not None:
        chosen=read(root/f'epoch_{best:02d}.pt',state['identity'])
        if chosen['ledger']['history']!=state['ledger']['history'][:best]:raise ValueError('Best history is not a prefix')
        atomic_save(root/'best.pt',chosen,replace=True)


def save_boundary(directory,model,optimizer,identity,ledger):
    root=bind_directory(directory,identity);n=ledger['completed_epochs']
    if model.config!=identity['model']:raise ValueError('Wrong model class/config')
    if n==0:
        if list(root.glob('*.pt')):raise FileExistsError('A fresh Run cannot contain checkpoints')
        if state_hash(model.state_dict())!=identity['initialization']['variant_model_hashes'][identity['variant']]:
            raise ValueError('Initial weights are not the verified fresh paired initialization')
    else:
        prior=read(root/('initial.pt' if n==1 else f'epoch_{n-1:02d}.pt'),identity)
        if prior['ledger']['history']!=ledger['history'][:-1]:raise ValueError('Committed history changed')
    state=dict(version=VERSIONS[identity['variant']],identity=copy.deepcopy(identity),ledger=copy.deepcopy(ledger),
               model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state())
    validate_state(state,identity);check_aliases(root,state)
    atomic_save(root/('initial.pt' if n==0 else f'epoch_{n:02d}.pt'),state)
    publish_aliases(root,state)
    return state


def restore(directory,model,optimizer,identity):
    root=bind_directory(directory,identity)
    if model.config!=identity['model']:raise ValueError('Wrong model class')
    current=read(root/'initial.pt',identity)
    if current['ledger']!=empty_ledger():raise ValueError('Invalid initial commit')
    files=sorted(root.glob('epoch_*.pt'))
    if len(files)>8 or [p.name for p in files]!=[f'epoch_{i:02d}.pt' for i in range(1,len(files)+1)]:
        raise ValueError('Missing/malformed immutable Epoch sequence')
    for i,path in enumerate(files,1):
        state=read(path,identity)
        if state['ledger']['completed_epochs']!=i or state['ledger']['history'][:-1]!=current['ledger']['history']:
            raise ValueError('Epoch history chain mismatch')
        current=state
    check_aliases(root,current)
    model.load_state_dict(current['model'],strict=True);optimizer.load_state_dict(current['optimizer'])
    publish_aliases(root,current);restore_rng(current['rng'])
    # Any interrupted in-memory Epoch is discarded. No files/logs are deleted.
    return copy.deepcopy(current['ledger'])


def commit_epoch(ledger,row,manifest):
    result=copy.deepcopy(ledger)
    if row['epoch']!=result['completed_epochs']+1:raise ValueError('Noncontiguous Epoch commit')
    score=full_validation_score(row['validation'],manifest)
    result['history'].append(row);result['completed_epochs']=row['epoch']
    for name in ('successful_updates','visited_train_windows','visited_val_windows'):result[name]+=row[name]
    if result['best_score'] is None or score<result['best_score']:
        result['best_score']=score;result['best_epoch']=row['epoch']
    return result


def load_for_evaluation(path,model,processor,*,source_sha,manifest):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('version') not in VERSIONS.values():raise ValueError('Not formal G5')
    identity=state['identity'];validate_state(state,identity)
    if identity['model']!=model.config or identity['processor']!=processor or identity['source_sha']!=source_sha:
        raise ValueError('Evaluation architecture/processor/source mismatch')
    if identity['data']['manifest']!=manifest:raise ValueError('Evaluation split/manifest mismatch')
    model.load_state_dict(state['model'],strict=True)
    return state
