"""Exactly two diagnostic joint updates, then discard all weights before formal."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.training.train import atomic_json, make_renderer
from scripts.experiments.v4_g4_global_motion.checkpoint import atomic_save
from scripts.experiments.v4_g4_global_motion.smoke import same_tree
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash, finite_tree, sha256
from scripts.experiments.v4_g5_geometry.smoke import assigned_gpu_idle, prepared
from .model import fresh_model, initialization
from .protocol import configuration, preflight, source_guard
from .train import configure_runtime, environment, perform_epoch


def group_statistics(model, before):
    result={}
    for group,prefix,part in (('motion_head','backbone.motion.',None),
            ('cross','backbone.cross.',None),('global','backbone.global_motion.',None),
            ('local','backbone.direct_motion.',None),('geometry_scale','head.4.',slice(0,3)),
            ('geometry_rotation','head.4.',slice(3,6)),('geometry_hidden','head.',None)):
        gradient=update=0.
        for name,p in model.named_parameters():
            if not name.startswith(prefix) or (group=='geometry_hidden' and name.startswith('head.4.')):
                continue
            if p.grad is not None:
                g=p.grad if part is None else p.grad[part]
                gradient+=float(g.detach().double().square().sum())
            delta=p.detach()-before[name]
            if part is not None:delta=delta[part]
            update+=float(delta.double().square().sum())
        result[group]=dict(clipped_gradient_norm=gradient**.5,update_norm=update**.5)
    return result


def run(args):
    configure_runtime();source,files=source_guard(args.expected_sha)
    resource=assigned_gpu_idle();env=environment()
    out=Path(args.output)
    if not out.is_absolute() or out.exists() or out.parent.name!='v4-g5-geo-joint-scratch-v1-smoke':
        raise ValueError('New independent joint Smoke namespace required')
    config=configuration(args.root,args.depth_root,source)
    if config['scale_weight']!=0:raise AssertionError('Original Joint must not acquire scale regularization')
    raw,data=preflight(config,args.g0_reference)
    model=fresh_model();initial=initialization(model);model.cuda()
    processor=DataProcessor('001-rope')
    def get(split,ep,start):
        ds=raw[split].dataset
        if (ep,start) not in ds.windows:raise AssertionError('Window absent from required split')
        return collate_fn([ds[ds.windows.index((ep,start))]])
    train=get('train',7,105);vals=[get('val',3,135),get('val',6,110)]
    batch=prepared(train,processor,model,7,105)
    with torch.no_grad():
        original=model.backbone(batch['inputs'])
        zero=model(batch['inputs'],mode='geometry_on_zero')
        off=model(batch['inputs'],mode='geometry_off')
    if not same_tree(original,zero) or not same_tree(original,off):
        raise AssertionError('Fresh real H15 equality failed')
    del batch,original,zero,off
    out.mkdir(parents=True,exist_ok=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    atomic_save(out/'initial_smoke.pt',dict(version='g5-geo-joint-smoke-v1',source_sha=source,
        config=model.config,geometry_config=model.geometry_config,model=model.state_dict(),optimizer=optimizer.state_dict(),updates=0))
    before={};updates=[];validation_hash=[]
    def pre(opt,args,kwargs):
        before.clear();before.update({n:p.detach().clone() for n,p in model.named_parameters()})
    def post(opt,args,kwargs):
        finite_tree((model.state_dict(),[p.grad for p in model.parameters()],opt.state_dict()))
        updates.append(dict(update=len(updates)+1,groups=group_statistics(model,before)))
        atomic_json(out/'updates.json',updates)
    def validation_start(module,args):
        if not module.training and not validation_hash:validation_hash.append(state_hash(module.state_dict()))
    hooks=[optimizer.register_step_pre_hook(pre),optimizer.register_step_post_hook(post),model.register_forward_pre_hook(validation_start)]
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    try:
        row=perform_epoch(model,optimizer,processor,make_renderer(config),config,'cuda',
            [train,train],vals,[[7,105],[7,105]],[[3,135],[6,110]])
    finally:
        for h in hooks:h.remove()
    if row['successful_updates']!=2 or row['optimizer_update_flags']!=[True,True] or len(updates)!=2:
        raise AssertionError('Smoke did not perform exactly two updates')
    for update in updates:
        for group in ('motion_head','cross','geometry_scale','geometry_rotation'):
            if update['groups'][group]['clipped_gradient_norm']<=0 or update['groups'][group]['update_norm']<=0:
                raise AssertionError('Dead joint gradient/update: '+group)
    if updates[-1]['groups']['geometry_hidden']['clipped_gradient_norm']<=0:
        raise AssertionError('Geometry hidden layers did not activate')
    if validation_hash!=[state_hash(model.state_dict())]:raise AssertionError('Validation modified weights')
    state=dict(version='g5-geo-joint-smoke-v1',source_sha=source,config=model.config,geometry_config=model.geometry_config,
        model=model.state_dict(),optimizer=optimizer.state_dict(),updates=2)
    atomic_save(out/'after_2updates.pt',state)
    restored=torch.load(out/'after_2updates.pt',map_location='cpu',weights_only=False)
    other=fresh_model().cuda().eval();other.load_state_dict(restored['model'],strict=True)
    other_opt=torch.optim.AdamW(other.parameters(),lr=1e-4,weight_decay=1e-4);other_opt.load_state_dict(restored['optimizer'])
    if not same_tree(optimizer.state_dict(),other_opt.state_dict()):raise AssertionError('Optimizer restore mismatch')
    probe=prepared(vals[0],processor,model,3,135)['inputs'];model.eval()
    with torch.no_grad():
        if not same_tree(model(probe),other(probe)):raise AssertionError('Strict restored prediction mismatch')
    result=dict(passed=True,source_sha=source,initialization=initial,successful_updates=2,source_files=files,
        environment=env,resource=resource,job=os.environ['SLURM_JOB_ID'],step=os.environ.get('SLURM_STEP_ID'),
        command=sys.argv,seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        exact_zero_h15=True,strict_restore_exact=True,validation_weights_unchanged=True,
        updates=updates,runner_metrics=row,checkpoint_sha256={p.name:sha256(p) for p in out.glob('*.pt')},
        note='Diagnostic model/optimizer are discarded. Formal entry constructs a fresh seed42 model; never loads these snapshots.')
    atomic_json(out/'completion.json',result)
    print(json.dumps({k:result[k] for k in ('passed','successful_updates','seconds','peak_cuda_bytes')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','depth-root','g0-reference','output','expected-sha'):p.add_argument('--'+name,required=True)
    args=p.parse_args()
    try:run(args)
    except BaseException:
        if Path(args.output).is_dir():
            atomic_json(Path(args.output)/'failure.json',dict(traceback=__import__('traceback').format_exc()))
        raise


if __name__=='__main__':main()
