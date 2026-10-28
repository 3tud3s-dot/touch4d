"""Bounded real H15 CUDA acceptance; exactly two train updates, frozen val only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.training.v4_model import V4Stage1Predictor
from scripts.training.v4_runtime import build_dataset, prepare_batch, v4_objective, load_checkpoint
from scripts.training.train import GS_CAMERAS, gs_render_settings, make_renderer, to_device, atomic_json
from .model import G4GlobalMotionPredictor

VERSION = 'touch4d-g4-cuda-smoke-v1'
G0_SHA = '60b7e6044a139a92bf7fdad29c262bde437ae2f9'
G0_HASH = '1fa35ddcc4c62ca359d8491d77bdd7888152a795ffae49b07a8480e190ccd05f'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def transfer_shared(g0, g4):
    expected = dict(g0.config, architecture='v4_g4_global_motion_v1',
                    global_motion='signed_per_hand_global_translation_v1')
    if g4.config != expected:
        raise ValueError('G0/G4 configuration mismatch')
    extra = {k for k in g4.state_dict() if k.startswith('global_motion.')}
    loaded = g4.load_state_dict(g0.state_dict(), strict=False)
    if set(loaded.missing_keys) != extra or loaded.unexpected_keys:
        raise ValueError('Only the new global-motion keys may be absent')
    for name, value in g0.state_dict().items():
        if not torch.equal(value, g4.state_dict()[name]):
            raise ValueError('Shared parameter differs: '+name)
    if any(torch.count_nonzero(p) for p in g4.global_motion.mapping[-1].parameters()):
        raise ValueError('Transfer requires zero final projection')
    return sorted(extra)


def save_g4(path, model, optimizer, steps, processor, source_sha):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    state = dict(version=VERSION, config=model.config, model=model.state_dict(),
                 optimizer=optimizer.state_dict(), successful_steps=steps,
                 processor=processor, source_sha=source_sha,
                 initialization='G0 diagnostic transfer ONLY; not a formal training protocol')
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
        os.link(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def restore_g4(path, model, optimizer, processor, source_sha):
    s = torch.load(path, map_location='cpu', weights_only=False)
    if (s.get('version') != VERSION or s.get('config') != model.config
            or s.get('processor') != processor or s.get('source_sha') != source_sha
            or type(s.get('successful_steps')) is not int or not 0 <= s['successful_steps'] <= 2):
        raise ValueError('Incompatible G4 smoke checkpoint')
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in s['model'].values()):
        raise ValueError('Nonfinite checkpoint')
    model.load_state_dict(s['model'], strict=True)
    if optimizer is not None: optimizer.load_state_dict(s['optimizer'])
    return s


def finite_tree(x):
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        if not torch.isfinite(x).all(): raise FloatingPointError('Nonfinite tensor')
    elif isinstance(x, dict):
        for v in x.values(): finite_tree(v)
    elif isinstance(x, (list, tuple)):
        for v in x: finite_tree(v)


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_tree(x,y) for x,y in zip(a,b))
    return a == b


class CheckedRenderer:
    def __init__(self):
        self.renderer = make_renderer(dict(output_mode='gs_geometry', render_settings=gs_render_settings()))
        self.records = []
    def check(self, result, kind):
        finite_tree(result)
        if result['rgb'].shape[2] != 3: raise ValueError('Require original three cameras')
        self.records.append(dict(kind=kind, shapes={k:list(v.shape) for k,v in result.items() if isinstance(v,torch.Tensor)}))
        return result
    def __call__(self, *args): return self.check(self.renderer(*args), 'prediction')
    def render_targets(self, *args): return self.check(self.renderer.render_targets(*args), 'target')


def trace_numbers(trace, scale):
    rows=[]
    for h,s in enumerate(trace['steps'],1):
        finite_tree(s)
        torch.testing.assert_close(s['global_delta'], s['hand_contribution'].sum(1), atol=0, rtol=0)
        invalid=~s['hand_valid']
        if torch.count_nonzero(s['hand_contribution'][invalid]): raise ValueError('Invalid hand contribution')
        rows.append(dict(horizon=h, hand_valid=s['hand_valid'].tolist(),
                         hand_basis_mm=(s['hand_basis'].detach()*scale*1000).tolist(),
                         hand_contribution_mm=(s['hand_contribution'].detach()*scale*1000).tolist(),
                         global_delta_mm=(s['global_delta'].detach()*scale*1000).tolist()))
    return rows


def metrics(loss, m):
    finite_tree(m)
    if not m['render_time_available'].all(): raise ValueError('Missing rendering supervision')
    return dict(loss=float(loss.detach()), **{k:v.detach().cpu().tolist() for k,v in m.items()})


def run(args, out):
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source != args.expected_sha or subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Source must be clean and frozen')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires allocated Slurm CUDA')
    if sha(args.checkpoint) != G0_HASH: raise ValueError('Frozen checkpoint hash mismatch')
    processor=DataProcessor('001-rope'); g0=V4Stage1Predictor(horizon=15,prediction_interval=1)
    saved=load_checkpoint(args.checkpoint,g0,processor_state=processor.state_dict())
    if saved['source_sha'] != G0_SHA: raise ValueError('Wrong H15 training SHA')
    provenance=dict(path=args.checkpoint, sha256=G0_HASH, source_sha=saved['source_sha'],
                    config=saved['config'],successful_steps=saved['successful_steps'],
                    epoch=(saved.get('run_state') or {}).get('best_epoch'))
    del saved
    g4=G4GlobalMotionPredictor(horizon=15,prediction_interval=1)
    provenance['new_keys']=transfer_shared(g0,g4)
    atomic_json(out/'checkpoint_provenance.json',provenance)
    batches={}; manifest={}
    for split,ep,start in [('train',7,105),('val',3,135)]:
        ds=build_dataset('001-rope',split,args.root,horizon=15,prediction_interval=1,
                         with_depth=True,depth_root=args.depth_root,depth_cameras=GS_CAMERAS['001-rope'])
        raw=collate_fn([ds[ds.windows.index((ep,start))]])
        if raw['meta']['frame_ids'].tolist() != [list(range(start,start+16))]: raise ValueError('Frame mismatch')
        batches[split]=prepare_batch(raw,processor,g0)
        manifest[split]=dict(episode=ep,start=start,frame_ids=raw['meta']['frame_ids'].tolist(),
            cameras=GS_CAMERAS['001-rope'],gs_points=int(raw['inputs']['gs_mask'].sum()),
            operation_hands=raw['inputs']['operation_hands'].tolist())
    atomic_json(out/'data_manifest.json',manifest)
    import gsplat
    results=dict(source_sha=source,base='0bed0d46305284a750cf3fd2430cc3e9ca923274',
        command=sys.argv,job_id=os.environ['SLURM_JOB_ID'],torch=torch.__version__,
        cuda=torch.version.cuda,gsplat=gsplat.__version__,gpu=torch.cuda.get_device_name(),
        render_settings=gs_render_settings(),loss='CD + RGB + 0.1 Alpha',
        initialization='diagnostic G0 shared transfer, NOT formal G4 training initialization',
        updates=[],equivalence={},successful_updates=0)
    start_time=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    g0.cuda().eval(); g4.cuda().eval()
    for split in batches:
        batch=to_device(batches[split],'cuda')
        with torch.no_grad():
            p0=g0(batch['inputs']);p4=g4(batch['inputs'])
            finite_tree(p0);finite_tree(p4)
            errors={k:(p0[k]-p4[k]).abs().flatten(2).amax(2).cpu().tolist() for k in p0}
            if not all(torch.equal(p0[k],p4[k]) for k in p0):
                atomic_json(out/'equivalence_failure.json',errors)
                raise AssertionError('Real H15 zero-init not bit-exact')
            results['equivalence'][split]=dict(bit_exact=True,max_abs_by_horizon=errors)
        del batch,p0,p4
    g0.cpu();del g0
    renderer=CheckedRenderer();optimizer=torch.optim.AdamW(g4.parameters(),lr=1e-4,weight_decay=1e-4)
    batch=to_device(batches['train'],'cuda');g4.train()
    for step in range(1,3):
        optimizer.zero_grad(set_to_none=True)
        before={k:p.detach().clone() for k,p in g4.global_motion.named_parameters()}
        pred,trace=g4(batch['inputs'],return_trace=True);finite_tree(pred)
        loss,m=v4_objective(pred,batch,processor,renderer)
        row=dict(step=step,metrics=metrics(loss,m),trace=trace_numbers(trace,processor.state_dict()['scale']))
        loss.backward()
        grads={k:float(p.grad.norm()) if p.grad is not None else None for k,p in g4.global_motion.named_parameters()}
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in g4.parameters()):
            raise FloatingPointError('Nonfinite gradient')
        if not grads['mapping.3.weight'] or grads['mapping.3.weight'] <= 0: raise AssertionError('Output map has no gradient')
        norm=torch.nn.utils.clip_grad_norm_(g4.parameters(),1.,error_if_nonfinite=True)
        row.update(raw_global_gradients=grads,raw_total_grad_norm=float(norm),clipped=bool(norm>1))
        row['clipped_global_gradients']={k:float(p.grad.norm()) if p.grad is not None else None
                                         for k,p in g4.global_motion.named_parameters()}
        optimizer.step();results['successful_updates']+=1
        finite_tree(g4.state_dict())
        finite_tree(optimizer.state_dict())
        delta={k:float((p.detach()-before[k]).norm()) for k,p in g4.global_motion.named_parameters()}
        if delta['mapping.3.weight'] <= 0: raise AssertionError('No actual output map update')
        row['parameter_update_l2']=delta;results['updates'].append(row)
        atomic_json(out/'partial_results.json',results)
        print('UPDATE',step,'loss',row['metrics']['loss'],'grad',grads['mapping.3.weight'],flush=True)
        del pred,trace,loss,m,before
    del batch
    save_g4(out/'g4_smoke_2_updates.pt',g4,optimizer,2,processor.state_dict(),source)
    restored=G4GlobalMotionPredictor(horizon=15,prediction_interval=1).cuda()
    opt2=torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=1e-4)
    restored_state=restore_g4(out/'g4_smoke_2_updates.pt',restored,opt2,processor.state_dict(),source)
    if not same_tree(optimizer.state_dict(),opt2.state_dict()): raise AssertionError('Optimizer restore differs')
    finite_tree(opt2.state_dict());g4.eval();restored.eval()
    with torch.no_grad():
        batch=to_device(batches['val'],'cuda')
        pred,trace=g4(batch['inputs'],return_trace=True);p2=restored(batch['inputs'])
        if not all(torch.equal(pred[k],p2[k]) for k in pred): raise AssertionError('Restored output differs')
        loss,m=v4_objective(pred,batch,processor,renderer)
        results['validation']=dict(metrics=metrics(loss,m),trace=trace_numbers(trace,processor.state_dict()['scale']),
                                   optimizer_steps=0,restore_bit_exact=True)
    # Both cross-format directions must fail closed.
    for path,model,loader in [(args.checkpoint,restored,'g4'),(out/'g4_smoke_2_updates.pt',V4Stage1Predictor(horizon=15),'g0')]:
        try:
            if loader=='g4': restore_g4(path,model,None,processor.state_dict(),source)
            else: load_checkpoint(path,model,processor_state=processor.state_dict())
        except ValueError: pass
        else: raise AssertionError('Cross-format checkpoint accepted')
    if sha(args.checkpoint)!=G0_HASH: raise AssertionError('Original checkpoint changed')
    torch.cuda.synchronize()
    results.update(passed=True,checkpoint_original_unchanged=True,checkpoint_sha256=sha(out/'g4_smoke_2_updates.pt'),
                   optimizer_restore_exact=True,cross_format_rejected=True,renderer_records=renderer.records,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   elapsed_seconds=time.perf_counter()-start_time)
    atomic_json(out/'results.json',results)
    (out/'smoke_report.md').write_text(
        '# G4 CUDA/gsplat Smoke\n\nPASS: real H15/Δ1, three cameras, original CD + RGB + 0.1 Alpha.\n\n'
        f"Source: `{source}`; Slurm: {results['job_id']}; GPU: {results['gpu']}.\n\n"
        'G0 -> zero G4 is bit-exact on ep7/start105 and ep3/start135. '
        'Exactly 2 optimizer updates on ep7; ep3 frozen inference only. '
        'New output-map gradients and actual updates are finite and nonzero. '
        'Independent G4 checkpoint and optimizer restore exactly; G0 cross-format loads rejected.\n\n'
        'This is engineering acceptance only, not fitting/performance evidence. '
        'G0 initialization here is diagnostic and does not define future G4 training initialization.\n',encoding='utf8')
    print('PASS',json.dumps({k:results[k] for k in ('job_id','successful_updates','peak_allocated_bytes','elapsed_seconds')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','depth-root','checkpoint','output','expected-sha'):p.add_argument('--'+key,required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    try: run(args,out)
    except BaseException:
        atomic_json(out/'failure.json',dict(traceback=traceback.format_exc(),command=sys.argv,
                    job_id=os.environ.get('SLURM_JOB_ID'),source_sha=args.expected_sha))
        raise


if __name__=='__main__': main()
