"""Fresh configurable-horizon V4/interval1, eight full epochs. Reuses production epoch loops."""
import argparse
from collections import Counter
import os
from pathlib import Path
import random
import subprocess
import traceback
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.loader import make_dataloader
from scripts.dataloader.hand_policy import HAND_POLICY
from scripts.training.v4_runtime import build_model, prepare_batch, save_checkpoint
from scripts.training.train import (GS_CAMERAS, gs_render_settings, make_renderer,
    train_epoch, evaluate_loader, atomic_json)


def configuration(root, depth_root, sha, horizon=10):
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    return dict(source_sha=sha, scene='001-rope', architecture='v4_stage1', protocol='epochs',
                root=str(Path(root).resolve()), depth_root=str(Path(depth_root).resolve()),
                epochs=8, single_epochs=0, horizon=horizon, prediction_interval=1,
                batch_size=1, seed=42, lr=1e-4, weight_decay=1e-4, workers=0,
                output_mode='gs_geometry', chamfer_weight=1., rgb_weight=1., alpha_weight=.1,
                scale_weight=0., depth_weight=0., chamfer_chunk=512,
                max_train_batches=0, max_val_batches=0, overfit_windows=0,
                depth_cameras=GS_CAMERAS['001-rope'], render_settings=gs_render_settings(),
                operation_hand_policy=HAND_POLICY, best_metric='validation.model.objective')


def loader(config, split, epoch=0):
    if split not in ('train', 'val'):
        raise ValueError('This experiment never opens the test split')
    return make_dataloader(config['scene'],split,root=config['root'],stage='all',
        batch_size=1,num_workers=0,seed=config['seed']+epoch,with_depth=True,
        depth_root=config['depth_root'],depth_cameras=config['depth_cameras'],
        horizon=config["horizon"],prediction_interval=1,operation_hand_policy=HAND_POLICY)


class PreparedLoader:
    """Validate/normalize on CPU exactly as the accepted V4 real-data path."""
    def __init__(self, raw_loader, processor, model):
        self.raw_loader,self.processor,self.model=raw_loader,processor,model
    def __iter__(self):
        for raw in self.raw_loader:
            yield prepare_batch(raw,self.processor,self.model)


class PreparedProcessor:
    """Production loops move prepared batches to CUDA; never normalize twice."""
    def __init__(self, processor): self.processor=processor
    def __call__(self,batch):
        if 'condition_frame_ids' not in batch['inputs']:
            raise ValueError('Expected validated, normalized V4 batch')
        return batch
    def __getattr__(self,key): return getattr(self.processor,key)


def data_manifest(loaders):
    return {name:dict(windows=len(value.dataset),
        episodes=dict(Counter(ep for ep,start in value.dataset.windows)),
        window_keys=[list(k) for k in value.dataset.windows]) for name,value in loaders.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','depth-root','output','expected-sha'): parser.add_argument('--'+key,required=True)
    parser.add_argument("--horizon",type=int,default=10)
    args=parser.parse_args()
    if args.horizon <= 0: parser.error("--horizon must be positive")
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if sha!=args.expected_sha or subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('Source must be clean and pinned to the submitted SHA')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires a Slurm CUDA allocation')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(4);torch.manual_seed(42);random.seed(42)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        config=configuration(args.root,args.depth_root,sha,args.horizon)
        processor=DataProcessor('001-rope'); adapter=PreparedProcessor(processor)
        model=build_model('v4_stage1',horizon=config['horizon'],prediction_interval=1).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        renderer=make_renderer(config)
        import gsplat
        raw_loaders={split:loader(config,split) for split in ('train','val')}
        manifest=data_manifest(raw_loaders)
        atomic_json(out/'manifest.json',manifest)
        atomic_json(out/'config.json',dict(training=config,model=model.config,processor=processor.state_dict(),
            torch=torch.__version__,gsplat=gsplat.__version__,gpu=torch.cuda.get_device_name(),job_id=os.environ['SLURM_JOB_ID']))
        ledger=dict(run_id=str(out),config=config,manifest=manifest,history=[],best_score=None,
                    successful_updates=0,completed_epochs=0)
        def counted_step(opt,unused_args,unused_kwargs):
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite parameters after optimizer update')
            ledger['successful_updates']+=1
        optimizer.register_step_post_hook(counted_step)
        def save(name):
            path=out/name
            save_checkpoint(path,model,optimizer,ledger['successful_updates'],source_sha=sha,
                processor_state=processor.state_dict(),run_state=ledger,replace_existing=path.exists())
        save('initial.pt')
        print('START',dict(source_sha=sha,windows={k:v['windows'] for k,v in manifest.items()},output=str(out)),flush=True)
        val=PreparedLoader(raw_loaders['val'],processor,model)
        for epoch in range(8):
            raw_train=raw_loaders['train'] if epoch==0 else loader(config,'train',epoch)
            before=ledger['successful_updates']
            training=train_epoch(model,optimizer,PreparedLoader(raw_train,processor,model),adapter,renderer,config,'cuda')
            validation=evaluate_loader(model,val,adapter,renderer,config,'cuda')
            row=dict(epoch=epoch+1,training=training,validation=validation,
                successful_updates=ledger['successful_updates']-before,partial_train=False,partial_validation=False)
            ledger['history'].append(row);ledger['completed_epochs']=epoch+1
            score=validation['model']['objective']
            improved=ledger['best_score'] is None or score<ledger['best_score']
            if improved:
                ledger['best_score']=score;ledger['best_epoch']=epoch+1;save('best.pt')
            save('last.pt')
            atomic_json(out/'metrics.json',ledger)
            print(__import__('json').dumps(row,allow_nan=False),flush=True)
        atomic_json(out/'status.json',dict(state='COMPLETED',epochs=8,successful_updates=ledger['successful_updates'],source_sha=sha))
    except BaseException:
        atomic_json(out/'failure.json',dict(source_sha=sha,traceback=traceback.format_exc()))
        raise


if __name__=='__main__': main()
