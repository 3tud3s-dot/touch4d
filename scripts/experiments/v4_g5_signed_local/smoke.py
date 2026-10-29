"""Future real CUDA/gsplat acceptance only. Not executed by CPU implementation tests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.training.v4_runtime import build_dataset,prepare_batch,v4_objective
from scripts.training.train import GS_CAMERAS,make_renderer,gs_render_settings,to_device,atomic_json
from scripts.experiments.v4_g4_global_motion.smoke import same_tree
from .checkpoint import build_model,save_checkpoint,load_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=('direct','propagate'),required=True)
    for k in ('root','depth-root','output','expected-sha'):p.add_argument('--'+k,required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires an authorized Slurm CUDA allocation; no CPU substitute')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source!=a.expected_sha or subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Source must be clean and pinned')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(4);torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=build_model(a.variant,horizon=15,prediction_interval=1).cuda()
        proc=DataProcessor('001-rope')
        ds=build_dataset('001-rope','train',a.root,horizon=15,prediction_interval=1,with_depth=True,
                         depth_root=a.depth_root,depth_cameras=GS_CAMERAS['001-rope'])
        raw=collate_fn([ds[ds.windows.index((7,105))]])
        if raw['meta']['frame_ids'].tolist()!=[list(range(105,121))]:raise ValueError('Wrong smoke window')
        batch=to_device(prepare_batch(raw,proc,model),'cuda')
        renderer=make_renderer(dict(output_mode='gs_geometry',render_settings=gs_render_settings()))
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats();clock=time.perf_counter();rows=[]
        for step in (1,2):
            opt.zero_grad(set_to_none=True)
            before=model.direct_motion.mapping[-1].weight.detach().clone()
            prop_before=({k:v.detach().clone() for k,v in model.propagation.named_parameters()}
                         if a.variant=='propagate' else {})
            pred,trace=model(batch['inputs'],return_trace=True)
            loss,m=v4_objective(pred,batch,proc,renderer)
            loss.backward()
            gradients={k:float(v.grad.norm()) if v.grad is not None else None for k,v in model.named_parameters()
                       if k.startswith(('direct_motion.','propagation.'))}
            for param in model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():raise FloatingPointError('Nonfinite gradients')
            direct_grad=model.direct_motion.mapping[-1].weight.grad
            if direct_grad is None or not float(direct_grad.abs().sum()):raise AssertionError('Direct head inactive')
            if a.variant=='propagate' and step==2:
                if not any(v.grad is not None and float(v.grad.abs().sum())>0 for v in model.propagation.parameters()):
                    raise AssertionError('Propagation score inactive after nonzero seed')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            update=float((model.direct_motion.mapping[-1].weight-before).norm())
            if update<=0:raise AssertionError('No local parameter update')
            prop_updates=({k:float((v.detach()-prop_before[k]).norm()) for k,v in model.propagation.named_parameters()}
                          if a.variant=='propagate' else {})
            if a.variant=='propagate' and step==2 and not any(v>0 for v in prop_updates.values()):
                raise AssertionError('No actual propagation parameter update')
            for value in model.state_dict().values():
                if value.is_floating_point() and not torch.isfinite(value).all():raise FloatingPointError('Nonfinite state')
            rows.append(dict(step=step,loss=float(loss.detach()),cd_mm=float(m['geometry'].mean())*proc.state_dict()['scale']*1000,
                rgb=float(m['rgb'].mean()),alpha=float(m['alpha'].mean()),raw_grad_norm=float(norm),
                local_gradients=gradients,local_head_update=update,propagation_parameter_updates=prop_updates,
                h15_local_gs_mean_mm=(trace['steps'][-1]['local_gs_mean'].detach()*proc.state_dict()['scale']*1000).cpu().tolist()))
            del pred,trace,loss,m
        save_checkpoint(out/'smoke.pt',model,opt,source_sha=source,processor_state=proc.state_dict(),successful_steps=2)
        other=build_model(a.variant,horizon=15,prediction_interval=1).cuda()
        opt2=torch.optim.AdamW(other.parameters(),lr=1e-4,weight_decay=1e-4)
        load_checkpoint(out/'smoke.pt',other,opt2,processor_state=proc.state_dict(),source_sha=source)
        if not same_tree(opt.state_dict(),opt2.state_dict()):raise AssertionError('Optimizer restore differs')
        model.eval();other.eval()
        with torch.no_grad():
            before=model(batch['inputs']);after=other(batch['inputs'])
            for key in before:torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
        import gsplat
        atomic_json(out/'results.json',dict(passed=True,variant=a.variant,source_sha=source,job_id=os.environ['SLURM_JOB_ID'],
            config=model.config,torch=torch.__version__,cuda=torch.version.cuda,gsplat=gsplat.__version__,gpu=torch.cuda.get_device_name(),
            successful_updates=2,initialization='fresh seed42; no checkpoint initialization',rows=rows,
            elapsed_seconds=time.perf_counter()-clock,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),checkpoint_restore_bit_exact=True,optimizer_restore_exact=True))
    except BaseException:
        atomic_json(out/'failure.json',dict(traceback=traceback.format_exc(),source_sha=source));raise


if __name__=='__main__':main()
