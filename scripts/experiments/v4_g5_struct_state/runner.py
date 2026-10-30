"""Small experiment loop: original renderer/Loss/metrics + one weak GS term."""
import csv
import json
from pathlib import Path
import torch
from scripts.training.train import to_device, metric_tracker, copy_initial, loss_options
from scripts.training.v4_runtime import prepare_batch
from scripts.training.losses import objective
from scripts.training.rendering import GaussianRenderer
from scripts.experiments.v4_g5_signed_local.diagnostics import finite_tree
from .structure import build_structure_graph,structure_loss,structure_loss_per_window,structure_statistics


class WindowBoundaryStop(RuntimeError):
    """Only complete Epoch snapshots persist; current attempted Epoch rolls back."""


class CachedRenderer:
    """Cache only the same batch's immutable target; original camera/render math."""
    def __init__(self,base=None,capture=False):
        self.base=base or GaussianRenderer();self.capture=capture;self.clear()
        self.settings=self.base.settings
    def clear(self):self.batch=None;self.target=None;self.rendered=None
    def render_targets(self,batch,processor):
        if self.batch is not batch:
            self.clear();self.batch=batch
        if self.target is None:self.target=self.base.render_targets(batch,processor)
        return self.target
    def __call__(self,prediction,batch,processor):
        result=self.base(prediction,batch,processor)
        if self.capture:self.rendered=result
        return result


def prepared(raw,processor,model,device):
    return to_device(prepare_batch(raw,processor,model),device)


def assert_real_batch(batch,episode,start):
    x,y=batch['inputs'],batch['targets']
    if batch['meta']['frame_ids'].tolist()!=[list(range(start,start+16))] or x['condition_frame_ids'].tolist()!=[list(range(start,start+15))]:
        raise AssertionError('H15/interval1 real frame IDs differ')
    if tuple(y['target_depth'].shape)!=(1,15,3,720,1280):raise AssertionError('Original three 1280x720 cameras required')
    if x['operation_hands'].tolist()!=[[episode>=5,True]]:raise AssertionError('Wrong original hand policy')
    # Natural empty view/time masks retain the original reduction/skip semantics.
    return y['target_depth_valid'].flatten(3).any(-1)


def key(raw):
    return [int(raw['meta']['episode_id'][0]),int(raw['meta']['frame_ids'][0,0])]


def original_objective(pred,batch,processor,renderer,config):
    return objective(pred,batch,processor,renderer,config['depth_weight'],config['chamfer_chunk'],**loss_options(config))


def losses(pred,batch,processor,renderer,config,graph=None):
    graph=graph or build_structure_graph(batch['inputs']['gs_xyz'],batch['inputs']['gs_mask'])
    original,values=original_objective(pred,batch,processor,renderer,config)
    unweighted=structure_loss(pred['xyz'],graph)
    weighted=config['lambda_struct']*unweighted
    total=original+weighted
    if not torch.isfinite(total):raise FloatingPointError('Nonfinite original/struct/total objective')
    return total,values,dict(loss_original=original,loss_struct_unweighted=unweighted,
        loss_struct_weighted=weighted,loss_total=total),graph


def metric_record(values,processor):
    return dict(cd_mm=float(values['geometry'].detach().mean())*processor.state_dict()['scale']*1000,
        rgb=float(values['rgb'].detach().mean()),alpha=float(values['alpha'].detach().mean()),joint=float(values['data_score'].detach().mean()),
        rgb_by_horizon=values['rgb_by_horizon'].detach().cpu().tolist(),
        alpha_by_horizon=values['alpha_by_horizon'].detach().cpu().tolist(),
        render_available=values['render_available'].detach().cpu().tolist(),
        render_time_available=values['render_time_available'].detach().cpu().tolist())


def frozen_attributes(pred,x):
    for name,a,b in (('log_scales',0,3),('quaternions',3,7),('opacity_logits',7,8),('sh_dc',8,11)):
        expected=x['gs_attr'][:,None,:,a:b].expand_as(pred[name])
        mask=x['gs_mask'][:,None,:,None].expand_as(pred[name])
        if not torch.equal(pred[name][mask],expected[mask]):raise AssertionError('Initial 11D attribute changed: '+name)


def train_epoch(model,optimizer,raw,processor,renderer,config,device,expected,*,record=None,step_observer=None,pending=None,progress=None,stop_requested=None):
    model.train();tracker=metric_tracker(processor,config)
    seen=[];flags=[];components=[];graph_stats=[];norms=[]
    for visit,raw_batch in enumerate(raw,1):
        window=key(raw_batch);seen.append(window)
        if visit>len(expected) or window!=expected[visit-1]:raise ValueError('Training window order drift')
        batch=prepared(raw_batch,processor,model,device);x=batch['inputs']
        if str(device).startswith('cuda'):assert_real_batch(batch,*window)
        optimizer.zero_grad(set_to_none=True)
        pred=model(x);total,values,parts,graph=losses(pred,batch,processor,renderer,config)
        tracker.add(batch['meta']['episode_id'],values)
        item=dict(visit=visit,episode=window[0],start=window[1],**metric_record(values,processor),
            **{k:float(v.detach()) for k,v in parts.items()},graph=graph.stats[0],successful_update=False)
        if bool(values['render_available'].any()):
            total.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            if pending:pending(visit,sum(flags),True)
            before=None
            if step_observer:before={n:p.detach().clone() for n,p in model.named_parameters()}
            optimizer.step()
            flags.append(True)
            if pending:pending(visit,sum(flags),False)
            if step_observer:
                finite_tree((pred,model.state_dict(),optimizer.state_dict(),[p.grad for p in model.parameters()]))
                step_observer(visit,model,optimizer,before,parts,graph,batch)
            item.update(successful_update=True,gradient_norm=float(norm),clipped=bool(norm>1))
            norms.append(float(norm))
        else:flags.append(False)
        components.append({k:item[k] for k in parts});graph_stats.append(graph.stats[0])
        if record:record(item)
        if progress and visit%20==0:progress(visit,sum(flags),item)
        renderer.clear() if hasattr(renderer,'clear') else None
        del batch,pred,total,values,parts,graph
        if stop_requested and stop_requested():raise WindowBoundaryStop('Safe train-window stop; uncommitted Epoch will replay')
    if seen!=expected:raise ValueError('Incomplete train order')
    return dict(metrics=tracker.summary(),window_order=seen,optimizer_update_flags=flags,successful_updates=sum(flags),
        visited_train_windows=len(seen),loss_component_batch_means={k:sum(v[k] for v in components)/max(1,len(components)) for k in components[0]} if components else {},
        structure_graphs=graph_stats,gradient_norm_max=max(norms,default=0.))


@torch.no_grad()
def evaluate_epoch(model,raw,processor,renderer,config,device,expected,*,record=None,structure_record=None,state_readout_off=False,stop_requested=None):
    was_training=model.training;model.eval()
    trackers={name:metric_tracker(processor,config) for name in ('model','copy_initial')};seen=[]
    try:
        for raw_batch in raw:
            window=key(raw_batch);seen.append(window)
            if len(seen)>len(expected) or window!=expected[len(seen)-1]:raise ValueError('Val order drift')
            batch=prepared(raw_batch,processor,model,device)
            if str(device).startswith('cuda'):assert_real_batch(batch,*window)
            pred=model(batch['inputs'],state_readout_off=True) if state_readout_off else model(batch['inputs'])
            graph=build_structure_graph(batch['inputs']['gs_xyz'],batch['inputs']['gs_mask'])
            for label,prediction in (('model',pred),('copy_initial',copy_initial(batch['inputs'],'gs_geometry'))):
                _,values=original_objective(prediction,batch,processor,renderer,config)
                trackers[label].add(batch['meta']['episode_id'],values)
                if label=='model' and record:
                    record(dict(episode=window[0],start=window[1],**metric_record(values,processor),
                        loss_struct_unweighted=float(structure_loss(prediction['xyz'],graph)),graph=graph.stats[0]))
            if structure_record:
                for row in structure_statistics(pred['xyz'],graph):structure_record(dict(row,episode=window[0],start=window[1]))
            renderer.clear() if hasattr(renderer,'clear') else None
            if stop_requested and stop_requested():raise WindowBoundaryStop('Safe Val-window stop; uncommitted Epoch will replay')
        if seen!=expected:raise ValueError('Incomplete validation')
        return {k:t.summary() for k,t in trackers.items()},seen
    finally:model.train(was_training)


def perform_epoch(model,optimizer,processor,renderer,config,device,train,val,train_order,val_order,**callbacks):
    result=train_epoch(model,optimizer,train,processor,renderer,config,device,train_order,
        **{k:callbacks[k] for k in ('record','step_observer','pending','progress','stop_requested') if k in callbacks})
    validation,seen=evaluate_epoch(model,val,processor,renderer,config,device,val_order,
        record=callbacks.get('validation_record'),structure_record=callbacks.get('structure_record'),stop_requested=callbacks.get('stop_requested'))
    return dict(training=result.pop('metrics'),validation=validation,validation_window_order=seen,
        visited_val_windows=len(seen),partial_train=False,partial_validation=False,**result)


class JsonLines:
    def __init__(self,path):self.path=Path(path);self.stream=self.path.open('x',encoding='utf-8')
    def __call__(self,row):self.stream.write(json.dumps(row,allow_nan=False)+'\n');self.stream.flush()
    def close(self):self.stream.close()
