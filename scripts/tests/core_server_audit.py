"""CPU-only, paired real-window audit of the retained E8 model and its relocation.

No optimizer is constructed, no backward pass is called and CUDA is hidden.
The two arms run in independent Python processes to isolate package imports.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


CORE_ROOT=Path(__file__).resolve().parents[2]
BASE_SHA='f978e0a29b6f0f0b6097c40684b949797a2f5d7b'
E8_SHA256='00fe8922bde7356e7290125813c2f0ca456604df03eafa2143158afd30073e57'
BUNDLE_SHA256='0e6dc18d5b527d07147a8592755cc8ef16592096b6441f7ca80ae00e2e07701a'
OVERRIDES={
    'dataset.py':'c934b8f2f3a8747e9cc5e081e6b420800274e4d4ca0267aa76fad2a6d14fe2e3',
    'processor.py':'caeeb56d1e4e996a2a016610344362d2fa3e67f9b4c311397cf5c7bc8783ed37',
    'cloth008.py':'1ce322053249a1112b2e3afb1ee0c62974ba3a6b601df21b6829bb1aae77747e',
    'cloth008_train.py':'b8f20dd966d1d09040af34fd8ed5810b3d2feb535f94ca862d2101ddf0a698ae',
    'cloth008_smoke.sbatch':'02b06aa12af5e4dde99799f40a776a47584bf30b7fb99701062874af2838eb5b',
}
TRACE_KEYS=('alpha','anchor_xyz','anchor_history','tool_motion','adjacent_valid',
            'head_history_mass','base_cumulative_d','global_delta','cumulative_d',
            'previous_d','state','local_correction','final_cumulative_anchor_displacement',
            'hand_basis12','direct_seed_per_hand')
MARKER='CORE_SERVER_AUDIT_RESULT='


def require(condition,message):
    if not condition:raise ValueError(message)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def git_head(path):
    return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()


def source_identity(args):
    require(git_head(args.oracle_source)==BASE_SHA,'Oracle checkout is not the frozen f978 source')
    tracked_status=subprocess.check_output(['git','-C',str(args.oracle_source),'status','--porcelain','--untracked-files=no'],text=True)
    require(not tracked_status.strip(),'Oracle tracked source is dirty; do not overwrite it for audit')
    values={name:sha256(args.frozen_overrides/name) for name in OVERRIDES}
    require(values==OVERRIDES,'Frozen source override identity differs')
    require(sha256(args.frozen_overrides/'source.sha256')==BUNDLE_SHA256,'Frozen bundle manifest identity differs')
    return dict(oracle_head=BASE_SHA,frozen_overrides=values,bundle_sha256=BUNDLE_SHA256)


class FrozenModules(importlib.abc.MetaPathFinder):
    names={'scripts.dataloader.dataset':'dataset.py',
           'scripts.data_process.processor':'processor.py',
           'scripts.experiments.v4_g5_signed_local.cloth008':'cloth008.py'}

    def __init__(self,root):self.root=root

    def find_spec(self,fullname,path=None,target=None):
        if fullname in self.names:
            return importlib.util.spec_from_file_location(fullname,self.root/self.names[fullname])
        return None


def tree_hash(value):
    """Canonical tensor-tree hash; source class/module names are deliberately excluded."""
    import dataclasses
    import numpy as np
    import torch
    digest=hashlib.sha256()

    def visit(item):
        if isinstance(item,torch.Tensor):
            require(item.device.type=='cpu','Audit attempted to hash a non-CPU tensor')
            digest.update(b'tensor:');digest.update(str(item.dtype).encode())
            digest.update(str(tuple(item.shape)).encode())
            digest.update(item.detach().contiguous().numpy().tobytes())
        elif isinstance(item,np.ndarray):
            digest.update(b'ndarray:');digest.update(str(item.dtype).encode())
            digest.update(str(item.shape).encode());digest.update(item.tobytes())
        elif isinstance(item,dict):
            digest.update(b'dict{')
            for key in sorted(item):digest.update(str(key).encode()+b':');visit(item[key])
            digest.update(b'}')
        elif isinstance(item,(list,tuple)):
            digest.update(('list:' if isinstance(item,list) else 'tuple:').encode())
            for member in item:visit(member)
        elif dataclasses.is_dataclass(item):
            visit({f.name:getattr(item,f.name) for f in dataclasses.fields(item)})
        elif item is None or isinstance(item,(str,bool,int,float)):
            digest.update(type(item).__name__.encode()+b':'+repr(item).encode())
        else:raise TypeError(f'Unsupported audit tree leaf: {type(item).__name__}')
    visit(value)
    return digest.hexdigest()


def tensor_summary(tensor):
    import torch
    require(isinstance(tensor,torch.Tensor),'Expected tensor output')
    require(tensor.device.type=='cpu','Non-CPU model output')
    if tensor.is_floating_point():require(bool(torch.isfinite(tensor).all()),'Nonfinite audit tensor')
    return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=tree_hash(tensor))


def worker(args):
    # Set visibility before importing torch or any project module.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(),'CUDA was initialized before the CPU audit')
    begin=time.monotonic()
    before_sources=source_identity(args)
    checkpoint_before=sha256(args.checkpoint)
    require(checkpoint_before==E8_SHA256,'Checkpoint is not the selected single-Episode E8')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    require(state.get('version')=='touch4d-g5-direct-008-episode0-v1','Unexpected checkpoint version')
    identity=state['identity'];history=state['history']
    require(identity['mode']=='formal' and identity['base_sha']==BASE_SHA,'Wrong checkpoint run/source identity')
    require(identity['source_bundle_sha256']==BUNDLE_SHA256,'Wrong checkpoint frozen bundle')
    orders=identity['epoch_orders']
    require(len(history)==len(orders)==8,'Expected exactly eight completed epochs')
    for epoch,(row,order) in enumerate(zip(history,orders),1):
        require(sorted(order)==list(range(113,253)),f'Epoch {epoch} protocol order differs')
        require(row['starts']==order and row['successful_updates']==140,
                f'Epoch {epoch} update or window coverage differs')
    updates=sum(row['successful_updates'] for row in history)
    require(updates==1120,'Expected exactly 1120 committed successful updates')
    require(identity['held_out_starts']==list(range(268,292)),'Held-out protocol differs')

    if args.worker=='oracle':
        sys.path.insert(0,str(args.oracle_source))
        sys.meta_path.insert(0,FrozenModules(args.frozen_overrides))
        from scripts.experiments.v4_g5_signed_local import cloth008 as data
        from scripts.experiments.v4_g5_signed_local.protocol import fresh_model
    else:
        sys.path.insert(0,str(CORE_ROOT))
        from scripts.training import cloth_data as data
        from scripts.training.cloth_protocol import fresh_model
    from scripts.dataloader.collate import collate_fn

    model=fresh_model('direct').cpu().eval()
    processor=data.Cloth008Processor(data.SCENE)
    require(identity['model']==model.config,'Model configuration differs from checkpoint identity')
    require(identity['processor']==processor.state_dict(),'Processor differs from checkpoint identity')
    original_config=identity['config']
    expected_config=data.configuration(original_config['root'],original_config['depth_root'],BASE_SHA)
    require(original_config==expected_config,'Frozen training configuration differs')
    runtime_config=data.configuration(args.root,args.depth_root,BASE_SHA)
    # Only input-root relocation is allowed at runtime, never a protocol change.
    require({k:v for k,v in runtime_config.items() if k not in ('root','depth_root')}
            =={k:v for k,v in original_config.items() if k not in ('root','depth_root')},
            'Runtime input relocation changed the frozen protocol')
    result=model.load_state_dict(state['model'],strict=True)
    require(not result.missing_keys and not result.unexpected_keys,'Strict state restoration failed')
    weights_before=tree_hash(model.state_dict())
    require(weights_before==tree_hash(state['model']),'Restored model tensors differ from checkpoint')
    require(all(p.device.type=='cpu' for p in model.parameters()),'Model parameters are not all CPU')

    datasets={split:data.make_dataset(runtime_config,split) for split in ('train','val')}
    require(datasets['train'].windows==tuple((0,s) for s in range(113,253)),'Train manifest differs')
    require(datasets['val'].windows==tuple((0,s) for s in range(268,292)),'Held-out manifest differs')
    index=datasets['train'].windows.index((0,225))
    raw=collate_fn([datasets['train'][index]])
    raw_hash=tree_hash(raw)
    prepared=data.prepare_batch(raw,processor,model)
    require(raw_hash==tree_hash(raw),'prepare_batch mutated the raw sample')
    require(raw['meta']['frame_ids'].tolist()==[list(range(225,241))],'Wrong real window')
    require(prepared['inputs']['condition_frame_ids'].tolist()==[list(range(225,240))],
            'Condition timeline includes future data or wrong offsets')
    with torch.no_grad():
        prediction,trace=model(prepared['inputs'],return_trace=True)
    require(len(trace['steps'])==15,'Trace does not cover the complete H15')
    require(set(prediction)=={'xyz','log_scales','quaternions','opacity_logits','sh_dc'},
            'Incomplete Gaussian output attributes')
    fields={key:tensor_summary(value) for key,value in prediction.items()}
    point_count=int(prepared['inputs']['gs_mask'].sum())
    require(fields['xyz']['shape']==[1,15,point_count,3],'Unexpected real H15 output shape')
    complete=torch.cat([prediction[k] for k in ('xyz','log_scales','quaternions','opacity_logits','sh_dc')],dim=-1)
    require(complete.shape[-1]==14,'Complete Gaussian output is not 14D')
    steps=[]
    for step,row in enumerate(trace['steps'],1):
        require(all(key in row for key in TRACE_KEYS),f'Missing comparable trace fields at H{step}')
        values={key:row[key] for key in TRACE_KEYS}
        summaries={key:tensor_summary(value) for key,value in values.items()}
        steps.append(dict(horizon=step,sha256=tree_hash(values),tensors=summaries))
    require(weights_before==tree_hash(model.state_dict()),'Inference changed model state')
    require(all(p.grad is None for p in model.parameters()),'Audit unexpectedly produced gradients')
    require(not torch.cuda.is_initialized(),'CPU audit initialized a CUDA context')
    checkpoint_after=sha256(args.checkpoint)
    require(checkpoint_after==checkpoint_before,'Checkpoint changed during audit')
    require(source_identity(args)==before_sources,'Oracle or frozen source changed during audit')
    snapshot=dict(checkpoint_sha256=checkpoint_before,version=state['version'],
                  completed_epochs=8,successful_updates=updates,train_windows=140,val_windows=24,
                  model_configuration=model.config,processor=processor.state_dict(),
                  frozen_configuration_sha256=tree_hash(original_config),
                  model_state_sha256=weights_before,raw_batch_sha256=raw_hash,
                  prepared_batch_sha256=tree_hash(prepared),input_sha256=tree_hash(prepared['inputs']),
                  targets_sha256=tree_hash(prepared['targets']),
                  point_count=point_count,frame_ids=raw['meta']['frame_ids'].tolist(),
                  condition_frame_ids=prepared['inputs']['condition_frame_ids'].tolist(),
                  prediction_sha256=tree_hash(prediction),complete_gaussian_sha256=tree_hash(complete),
                  prediction_fields=fields,trace=steps,
                  field_sha256=tree_hash(trace['field']),source_identity=before_sources)
    return dict(status='PASS',arm=args.worker,seconds=time.monotonic()-begin,snapshot=snapshot,
                cpu_threads=args.cpu_threads,torch=str(torch.__version__),
                cuda_initialized=False,checkpoint_before=checkpoint_before,
                checkpoint_after=checkpoint_after,optimizer_constructed=False,
                optimizer_steps=0,backward_calls=0)


def different_paths(left,right,prefix=''):
    if type(left) is not type(right):return [prefix or '<root>']
    if isinstance(left,dict):
        if left.keys()!=right.keys():return [prefix+'.keys']
        result=[]
        for key in left:
            result.extend(different_paths(left[key],right[key],f'{prefix}.{key}' if prefix else str(key)))
        return result
    if isinstance(left,list):
        if len(left)!=len(right):return [prefix+'.length']
        result=[]
        for index,(a,b) in enumerate(zip(left,right)):result.extend(different_paths(a,b,f'{prefix}[{index}]'))
        return result
    return [] if left==right else [prefix]


def run_parent(args):
    output=args.output.resolve()
    require(not output.exists(),'Output already exists; preserve the prior audit and choose a new output')
    for protected in (args.oracle_source,args.frozen_overrides,args.root,args.depth_root):
        require(not output.is_relative_to(protected.resolve()),f'Output must not be inside read-only input {protected}')
    require(output!=args.checkpoint.resolve(),'Output must not overwrite the checkpoint')
    begin=time.monotonic()
    report=dict(status='FAIL',kind='paired_cpu_real_window_relocation_audit',
                code=str(Path(__file__).resolve()),code_sha256=sha256(__file__),
                oracle_source=str(args.oracle_source),core_source=str(CORE_ROOT),
                checkpoint=str(args.checkpoint),window=dict(scene='008-pink-cloth',episode=0,start=225,horizon=15,delta=1),
                gpu_requested=0,optimizer_steps=0,backward_calls=0,arms={})
    try:
        source_identity(args)
        checkpoint_before=sha256(args.checkpoint)
        require(checkpoint_before==E8_SHA256,'Checkpoint is not the selected E8')
        for arm in ('oracle','core'):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
                     OMP_NUM_THREADS=str(args.cpu_threads),MKL_NUM_THREADS=str(args.cpu_threads),
                     OPENBLAS_NUM_THREADS='1')
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',arm,
                     '--oracle-source',str(args.oracle_source),'--frozen-overrides',str(args.frozen_overrides),
                     '--checkpoint',str(args.checkpoint),'--root',str(args.root),
                     '--depth-root',str(args.depth_root),'--output',str(args.output),
                     '--cpu-threads',str(args.cpu_threads),'--timeout',str(args.timeout)]
            process=subprocess.run(command,cwd=args.oracle_source if arm=='oracle' else CORE_ROOT,
                                   env=env,capture_output=True,text=True,timeout=args.timeout)
            entries=[line[len(MARKER):] for line in process.stdout.splitlines() if line.startswith(MARKER)]
            if process.returncode or len(entries)!=1:
                report['arms'][arm]=dict(status='FAIL',returncode=process.returncode,
                                         stdout_tail=process.stdout[-4000:],stderr_tail=process.stderr[-4000:])
                raise RuntimeError(f'{arm} CPU audit subprocess failed')
            report['arms'][arm]=json.loads(entries[0])
            if process.stderr.strip():report['arms'][arm]['stderr_tail']=process.stderr[-2000:]
        mismatches=different_paths(report['arms']['oracle']['snapshot'],report['arms']['core']['snapshot'])
        report['mismatch_paths']=mismatches
        require(not mismatches,'Oracle/core real-window outputs or states differ')
        report['checkpoint_sha256_before']=checkpoint_before
        report['checkpoint_sha256_after']=sha256(args.checkpoint)
        require(report['checkpoint_sha256_after']==checkpoint_before,'Checkpoint changed across paired audit')
        report.update(status='PASS',paired_prediction_exact=True,complete_attributes_exact=True,
                      key_state_trace_exact=True,shared_field_exact=True,
                      successful_training_updates_audited=1120,new_training_updates=0,
                      new_supervision_frames=0,cuda_initialized=False)
    except Exception as exc:
        report['error']=str(exc);report['traceback']=traceback.format_exc()
    report['seconds']=time.monotonic()-begin
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=report['status'],seconds=report['seconds'],output=str(output),
                          mismatch_paths=report.get('mismatch_paths'),error=report.get('error')),ensure_ascii=False))
    return 0 if report['status']=='PASS' else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('oracle-source','frozen-overrides','checkpoint','root','depth-root','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--timeout',type=int,default=1800,help='Maximum seconds per independent CPU arm')
    parser.add_argument('--worker',choices=('oracle','core'),help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    require(args.cpu_threads>0 and args.timeout>0,'Expected positive CPU thread count and timeout')
    for name in ('oracle_source','frozen_overrides','checkpoint','root','depth_root'):
        setattr(args,name,getattr(args,name).resolve())
    if args.worker:
        print(MARKER+json.dumps(worker(args),sort_keys=True,allow_nan=False))
        return 0
    return run_parent(args)


if __name__=='__main__':
    raise SystemExit(main())
