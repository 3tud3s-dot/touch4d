"""Serial CUDA preflight: warm private extension cache, immutable legacy regression."""
import argparse,json,os,subprocess
from pathlib import Path
from types import SimpleNamespace
import torch
from .runner import run,sha,dataset,batch_at,read_manifest,ARCH,WEIGHTS
from scripts.training.train import atomic_json,MODEL_KEYS,read_checkpoint,make_renderer
from scripts.training.model import PositionPredictor
from scripts.training.losses import objective
from scripts.data_process import DataProcessor


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--expected-sha',required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm CUDA required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    # Exactly the old default code, compiled in its original package namespace.
    source=subprocess.check_output(['git','show','64b1320:scripts/training/model.py'],text=True)
    scope={'__package__':'scripts.training'};exec(compile(source,'<main-64b1320>','exec'),scope)
    manifest=read_manifest(a.manifest);regression=[]
    old_paths=[('/data1/userdata/tcweng/projects/touch4d-worktrees/p2-001-h5-e8/runs/001-rope/p2-001-h5-e8-30695/best.pt',5),
               ('/data1/userdata/tcweng/projects/touch4d-worktrees/p2-baseline/runs/001-rope/p2-001-h15-e8-30666/best.pt',15)]
    torch.set_num_threads(4)
    for path,h in old_paths:
        before=sha(path);state=read_checkpoint(path);config=state['config'];processor=DataProcessor.from_state_dict(state['processor'])
        batch=batch_at(dataset(a.root,'val',h),manifest['windows'][str(h)]['val'][0],processor,'cuda')
        # Legacy models ignore operation_hands, exactly as their original protocol.
        m=PositionPredictor(**{k:config[k] for k in MODEL_KEYS}).cuda().eval()
        old=scope['PositionPredictor'](**{k:config[k] for k in MODEL_KEYS}).cuda().eval()
        m.load_state_dict(state['model']);old.load_state_dict(state['model']);renderer=make_renderer(config)
        with torch.no_grad():
            pred=m(batch['inputs']);reference=old(batch['inputs'])
            assert all(torch.equal(pred[k],reference[k]) for k in pred),'Legacy output drift'
            loss,_=objective(pred,batch,processor,renderer,0.,512,**WEIGHTS)
            reference_loss,_=objective(reference,batch,processor,renderer,0.,512,**WEIGHTS)
            assert torch.equal(loss,reference_loss),'Legacy loss drift'
        assert sha(path)==before
        regression.append(dict(path=path,sha256=before,horizon=h,output_exact=True,loss_exact=True,loss=float(loss)))
        del state,batch,m,old,pred,reference,loss,reference_loss
        torch.cuda.empty_cache()
    atomic_json(out/'legacy.json',regression)
    configs=[dict(mode=s) for s in ('S0','S1','S2-C0','S2-C1','S3-Pre','S3-Post','S4-C0','S4-C1')]
    configs += [dict(mode='S2-C1',decoder='D2'),dict(mode='S2-C1',feedback='F1'),
                dict(mode='S2-C1',regularization=.01),dict(mode='S4-C1',decoder='D1')]
    records=[]
    for index,cfg in enumerate(configs):
        args=SimpleNamespace(**dict(root=a.root,manifest=a.manifest,output=str(out/f'smoke-{index}'),
            expected_sha=a.expected_sha,command='smoke',mode='S0',decoder='D0',feedback='F0',regularization=0.,horizon=5,seed=42,updates=400))
        for k,v in cfg.items():setattr(args,k,v)
        run(args);records.append(dict(config=cfg,status=json.loads((Path(args.output)/'status.json').read_text())))
        torch.cuda.empty_cache()
    atomic_json(out/'acceptance.json',dict(legacy=regression,smokes=records,status='passed'))
if __name__=='__main__':main()
