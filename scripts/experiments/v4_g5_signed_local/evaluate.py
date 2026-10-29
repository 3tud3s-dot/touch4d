"""Read-only full Val100 + frozen window diagnostics for completed paired G5 Runs."""
import argparse
import gc
import os
from pathlib import Path
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.training.train import evaluate_loader,make_renderer,to_device,atomic_json
from scripts.training.v4_runtime import prepare_batch,v4_objective,load_checkpoint as load_g0
from scripts.training.v4_model import V4Stage1Predictor
from scripts.training.losses import nearest_distances
from scripts.experiments.full_training.v4_epochs import loader,PreparedLoader,PreparedProcessor
from scripts.experiments.v4_g4_global_motion.model import G4GlobalMotionPredictor
from scripts.experiments.v4_g4_global_motion.checkpoint import load_for_evaluation as load_g4
from scripts.experiments.v4_g4_global_motion.evaluate import fixed_regions,motion_metrics
from scripts.experiments.v4_g4_global_motion.smoke import trace_numbers
from .checkpoint import build_model
from .formal_checkpoint import load_for_evaluation
from .protocol import configuration,preflight,full_validation_score
from .diagnostics import sha256,state_hash,mapping_context,cpu_tree,collect,write_csv,finite_tree
from .smoke import source_preflight,G4_HASH


def validate_pair_metadata(identities,ledgers):
    for variant in ('direct','propagate'):
        best,last=ledgers[variant+'-best'],ledgers[variant+'-last']
        if last['completed_epochs']!=8 or best['completed_epochs']!=last['best_epoch'] or best['best_score']!=last['best_score']:
            raise ValueError('Best/Last selection or completed budget mismatch')
        if identities[variant+'-best']['run_id']!=identities[variant+'-last']['run_id']:
            raise ValueError('Best/Last cross-Run mix')
    left,right=identities['direct-last'],identities['propagate-last']
    if left['run_id']==right['run_id']:raise ValueError('Variants share a Run directory')
    for key in ('g4_shared_hash','direct_mapping_hash'):
        if left['initialization'][key]!=right['initialization'][key]:raise ValueError('Paired initial weights differ')
    if left['data']['epoch_order_hashes']!=right['data']['epoch_order_hashes']:raise ValueError('Paired orders differ')

WINDOWS=(('val',3,135),('val',6,110),('train',7,105))


def evaluate_full(model,raw,processor,renderer,c,device,manifest):
    before=state_hash(model.state_dict())
    from scripts.experiments.v4_g4_global_motion.protocol import CheckedStream
    stream=CheckedStream(raw,manifest['val']['window_keys'])
    result=evaluate_loader(model,PreparedLoader(stream,processor,model),PreparedProcessor(processor),renderer,c,device)
    full_validation_score(result,manifest)
    if state_hash(model.state_dict())!=before:raise AssertionError('Evaluation modified weights')
    return result


@torch.no_grad()
def window_diagnostics(model,batch,processor,renderer,label):
    before=state_hash(model.state_dict());x=batch['inputs'];g5=hasattr(model,'direct_motion');factor=processor.state_dict()['scale']*1000
    if g5:
        with mapping_context(model) as contexts:pred,trace=model(x,return_trace=True)
    else:pred,trace=model(x,return_trace=True);contexts=[]
    loss,metrics=v4_objective(pred,batch,processor,renderer);finite_tree((pred,metrics,loss))
    groups=fixed_regions(x);rows=motion_metrics(pred,x,processor.state_dict()['scale'])
    for row in rows:
        h=row['horizon']-1;position=pred['xyz'][0,h];ids=groups[row['region']]
        gt=batch['targets']['target_xyz'][h][0,batch['targets']['target_mask'][h][0]]
        row['pred_region_to_full_gt_mm']=float(nearest_distances(position[ids],gt,512).mean())*factor
        if row['region']=='all':
            valid=position[groups['all']]
            row['cd_mm']=float((nearest_distances(valid,gt,512).mean()+nearest_distances(gt,valid,512).mean())*.5)*factor
        row.update(model=label,rgb=float(metrics['rgb_by_horizon'][0,h]),alpha=float(metrics['alpha_by_horizon'][0,h]))
        for axis,j in zip('xyz',range(3)):row['signed_mean_'+axis+'_mm']=row['mean_xyz_mm'][j]
    report=dict(joint=float(loss),cd_mm=float(metrics['geometry'].mean())*factor,
                rgb=float(metrics['rgb'].mean()),alpha=float(metrics['alpha'].mean()))
    geometry=[];propagation=[];conditions=[]
    if g5:
        geometry,propagation=collect(model,cpu_tree(x),cpu_tree(trace),cpu_tree(contexts),
            variant=model.config['checkpoint_kind'].removeprefix('g5_').removesuffix('_v1'),window=label,stage='frozen_formal',factor=factor)
        for h,s in enumerate(trace['steps'],1):
            active=trace['field'].anchor_mask[0]
            for hand in (0,1):
                local=s['local_correction'][0,active];d=s['direct_seed_per_hand'][0,active,hand]
                p=s.get('propagated_seed_per_hand',s['direct_seed_per_hand'])[0,active,hand]
                row=dict(model=label,horizon=h,hand=('left','right')[hand],hand_valid=bool(s['hand_valid'][0,hand]),
                    direct_rms_mm=float(d.square().sum(-1).mean().sqrt())*factor,
                    propagated_rms_mm=float(p.square().sum(-1).mean().sqrt())*factor,
                    mix_difference_rms_mm=float((p-d).square().sum(-1).mean().sqrt())*factor,
                    local_anchor_rms_mm=float(local.square().sum(-1).mean().sqrt())*factor,
                    local_gs_demeaned_rms_mm=float(s['local_gs_demeaned_rms'][0])*factor)
                for axis,j in zip('xyz',range(3)):
                    row['g4_global_'+axis+'_mm']=float(s['g4_global_delta'][0,j])*factor
                    row['g4_hand_'+axis+'_mm']=float(s['hand_contribution'][0,hand,j])*factor
                    row['direct_mean_'+axis+'_mm']=float(d[:,j].mean())*factor
                    row['propagated_mean_'+axis+'_mm']=float(p[:,j].mean())*factor
                    row['local_gs_mean_'+axis+'_mm']=float(s['local_gs_mean'][0,j])*factor
                conditions.append(row)
    elif hasattr(model,'global_motion'):report['g4_contributions']=trace_numbers(trace,processor.state_dict()['scale'])
    if state_hash(model.state_dict())!=before:raise AssertionError('Frozen window evaluation changed weights')
    return report,rows,conditions,geometry,propagation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','depth-root','g0-reference','output','expected-sha','direct-best','direct-last',
                'propagate-best','propagate-last','g4-checkpoint','g0-checkpoint'):p.add_argument('--'+key,required=True)
    a=p.parse_args();source,_=source_preflight(a.expected_sha)
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Read-only evaluation requires approved Slurm GPU allocation')
    import json,gsplat
    environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gsplat=gsplat.__version__,gpu=torch.cuda.get_device_name(),tf32=False)
    old_env=json.loads((Path(a.g0_reference)/'config.json').read_text())
    if environment['cuda']!='12.8' or any(environment[k]!=old_env[k] for k in ('torch','gsplat','gpu')):
        raise ValueError('Evaluation environment differs from reference')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    c=configuration(a.root,a.depth_root,source,'direct');raw,data=preflight(c,a.g0_reference);processor=DataProcessor('001-rope')
    paths={key:Path(getattr(a,key.replace('-','_'))) for key in ('direct-best','direct-last','propagate-best','propagate-last','g4-checkpoint','g0-checkpoint')}
    out=Path(a.output).resolve()
    for protected in (Path.cwd(),Path(a.root).resolve(),Path(a.depth_root).resolve(),*(v.parent.resolve() for v in paths.values())):
        if out==protected or protected in out.parents:raise ValueError('Evaluation output overlaps protected code/data/checkpoint Run')
    renderer=make_renderer(c);out.mkdir(parents=True,exist_ok=False)
    hashes={key:sha256(path) for key,path in paths.items()};result={};motion=[];conditions=[];geometry=[];propagation=[];ledgers={};identities={}
    if hashes['g4-checkpoint']!=G4_HASH:raise ValueError('Not the frozen G4 Best')
    for label in ('direct-best','direct-last','propagate-best','propagate-last'):
        variant,selection=label.split('-');model=build_model(variant)
        if paths[label].name!=selection+'.pt':raise ValueError('Explicit Best/Last aliases required')
        state=load_for_evaluation(paths[label],model,processor.state_dict(),source_sha=source,manifest=data['manifest'])
        identities[label]=state['identity'];ledgers[label]=state['ledger']
        del model,state
    validate_pair_metadata(identities,ledgers)
    for label,path in paths.items():
        source_preflight(a.expected_sha)
        if label.startswith(('direct-','propagate-')):
            variant,selection=label.split('-');model=build_model(variant)
            if path.name!=selection+'.pt':raise ValueError('Best/Last alias explicitly required')
            state=load_for_evaluation(path,model,processor.state_dict(),source_sha=source,manifest=data['manifest'])
            ledgers[label]=state['ledger'];identities[label]=state['identity']
            if selection=='last' and state['ledger']['completed_epochs']!=8:raise ValueError('Formal comparison needs completed eight-Epoch Last')
        elif label=='g4-checkpoint':
            model=G4GlobalMotionPredictor(horizon=15);state=load_g4(path,model,processor.state_dict())
            if state['identity']['data']['manifest']!=data['manifest']:raise ValueError('G4 split differs')
            if state['ledger']['completed_epochs']!=state['ledger']['best_epoch']:raise ValueError('Requires selected G4 Best')
        else:
            model=V4Stage1Predictor(horizon=15);state=load_g0(path,model,processor_state=processor.state_dict())
            if state['config']['horizon']!=15 or state['config']['prediction_interval']!=1:raise ValueError('G0 is not H15/1')
            run=state.get('run_state',{})
            if run.get('best_epoch') is None or run.get('completed_epochs')!=run['best_epoch']:raise ValueError('Requires selected G0 Best')
            if state.get('source_sha')!='60b7e6044a139a92bf7fdad29c262bde437ae2f9' or json.loads(json.dumps(run.get('manifest')))!=data['manifest']:
                raise ValueError('G0 reference source/split differs')
        model.cuda().eval();result[label]=dict(full_val=evaluate_full(model,raw['val'],processor,renderer,c,'cuda',data['manifest']),
            checkpoint=str(path),checkpoint_sha256=hashes[label],windows={},
            training_source_sha=state.get('source_sha',state.get('identity',{}).get('source_sha')),
            completed_epochs=state.get('ledger',state.get('run_state',{})).get('completed_epochs'))
        for split,ep,start in WINDOWS:
            ds=raw[split].dataset;cpu=collate_fn([ds[ds.windows.index((ep,start))]])
            batch=to_device(prepare_batch(cpu,processor,model),'cuda')
            report,rows,cr,gr,pr=window_diagnostics(model,batch,processor,renderer,label)
            key=f'ep{ep}_start{start}';result[label]['windows'][key]=report
            for target,items in ((motion,rows),(conditions,cr),(geometry,gr),(propagation,pr)):
                for row in items:row.update(episode=ep,start=start,split=split)
                target.extend(items)
            del batch
        atomic_json(out/(label+'.json'),result[label])
        del state,model;gc.collect();torch.cuda.empty_cache()
    validate_pair_metadata(identities,ledgers)
    if any(sha256(paths[k])!=value for k,value in hashes.items()):raise AssertionError('Reference checkpoint changed')
    source_preflight(a.expected_sha)
    atomic_json(out/'results.json',dict(source_sha=source,environment=environment,models=result,g4_context_reference=dict(cd_mm=2.88409,joint=.119000),
        copy_initial=result['direct-best']['full_val']['copy_initial'],manifest=data['manifest'],
        notes='Full Val episode-equal Joint; local error is one-way region-to-full-GT; independent GS have no material ID. Spatial graph and 16NN ratios are structure proxies, not topology/strain. No optimizer or backward.'))
    for name,rows in (('motion',motion),('conditions',conditions),('geometry_diagnostics',geometry),('propagation_diagnostics',propagation)):
        write_csv(out/(name+'.csv'),rows)


if __name__=='__main__':main()
