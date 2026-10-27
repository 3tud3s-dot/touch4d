"""One fresh 400-update squared-CD run; frozen CD/MSE are read-only controls."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import traceback
import torch
from scripts.experiments.v4_action_loss_comparison import run as paired
from scripts.experiments.v4_action_identifiability.data import normalize
from scripts.experiments.fixed_window.fixed_fit import digest
from scripts.training.train import atomic_json, to_device
from scripts.training.v4_runtime import build_model, load_checkpoint
from .loss import squared_chamfer_samples

BASE='e32933f730590c2a0507a64689a43e5853a13576'


@contextmanager
def diagnostic_adapter():
    """Scoped experiment-only callbacks reuse the exact V2 training/update loop.

    No production callable changes. Always restore, including on exceptions.
    This runner is single-threaded at Python orchestration level.
    """
    old_loss,old_measure=paired.loss_samples,paired.measure
    def loss(pred,batch,mode):
        if mode=='squared_cd':return squared_chamfer_samples(pred['xyz'],batch,512)
        return old_loss(pred,batch,mode)
    @torch.no_grad()
    def measure(model,batch,processor,affected,mode):
        row,pred=old_measure(model,batch,processor,affected,'cd' if mode=='squared_cd' else mode)
        if mode=='squared_cd':
            values=loss(pred,batch,mode)
            row.update(raw_squared_cd_normalized=float(values.mean()),
                       branch_squared_cd_normalized=values.tolist(),training_loss=float(values.mean()))
        return row,pred
    paired.loss_samples,paired.measure=loss,measure
    try:yield
    finally:paired.loss_samples,paired.measure=old_loss,old_measure


def smoke(out,batch,processor,config,initial_hash):
    out.mkdir();torch.manual_seed(42)
    model=build_model('v4_stage1').cuda()
    assert digest(model.state_dict())==initial_hash
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    rows=[paired.update(model,opt,batch,'squared_cd') for _ in range(3)]
    for group in paired.GROUPS:
        if not any(r['raw_gradients'][group]['gradient_norm']>0 for r in rows[1:]):
            raise AssertionError('Missing nondegenerate gradient: '+group)
    paired.save(out/'smoke.pt',model,opt,3,processor,dict(config=config,mode='squared_cd'))
    clone=build_model('v4_stage1').cuda();co=torch.optim.AdamW(clone.parameters(),lr=1e-4,weight_decay=1e-4)
    state=load_checkpoint(out/'smoke.pt',clone,co,processor_state=processor.state_dict())
    if state['successful_steps']!=3:raise AssertionError('Checkpoint update count')
    with torch.no_grad():torch.testing.assert_close(model(batch['inputs'])['xyz'],clone(batch['inputs'])['xyz'],rtol=0,atol=0)
    paired.update(model,opt,batch,'squared_cd');paired.update(clone,co,batch,'squared_cd')
    for k,v in model.state_dict().items():torch.testing.assert_close(v,clone.state_dict()[k],rtol=0,atol=0)
    atomic_json(out/'acceptance.json',dict(passed=True,updates=rows,checkpoint_next_update_exact=True,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,python=platform.python_version()))


def hashes(root):
    return {str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(root.rglob('*')) if f.is_file()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('v1','v2','output','expected-sha'):p.add_argument('--'+arg,required=True)
    args=p.parse_args();sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if sha!=args.expected_sha or subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Source drift')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm GPU required')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        v2=Path(args.v2);before=hashes(v2)
        previous=json.loads((v2/'results.json').read_text());ids=json.loads((v2/'data_identity.json').read_text())
        if json.loads((v2/'config.json').read_text())['source_sha']!=BASE:raise AssertionError('Wrong V2 reference')
        frozen,identity=paired.read_frozen(args.v1)
        torch.manual_seed(42);model=build_model('v4_stage1');initial_hash=digest(model.state_dict())
        if initial_hash!=identity['v1_initial_model_sha256']:raise AssertionError('Initialization differs')
        batch,processor=normalize(frozen['raw'],model)
        identity.update(normalized_inputs_sha256=digest(batch['inputs']),normalized_targets_sha256=digest(batch['targets']))
        if identity!=ids:raise AssertionError('V2 input/target identity differs')
        if any(previous[m]['initial_parameter_sha256']!=initial_hash for m in ('cd','mse')):raise AssertionError('V2 model mismatch')
        config=dict(source_sha=sha,base_sha=BASE,v1=str(Path(args.v1).resolve()),v2=str(v2.resolve()),
            horizon=10,prediction_interval=1,seed=42,batch_size=2,optimizer='AdamW',lr=1e-4,weight_decay=1e-4,
            clip_norm=1.,updates=400,eval_every=20,loss='squared_cd',loss_multiplier=1.,
            formula='0.5*(mean NN squared L2 forward + mean NN squared L2 reverse), mean H, mean branches',
            scaling='none; coefficient1 fixed before training',selection='minimum own training loss over 0,20,...,400',
            job_id=os.environ['SLURM_JOB_ID'],historical_controls_retrained=False)
        atomic_json(out/'config.json',config);atomic_json(out/'data_identity.json',identity)
        batch=to_device(batch,'cuda');affected=frozen['affected'].cuda()
        with diagnostic_adapter():
            smoke(out/'smoke',batch,processor,config,initial_hash);torch.cuda.empty_cache()
            result=paired.train(out/'squared_cd','squared_cd',batch,processor,affected,config,initial_hash)
        if hashes(v2)!=before:raise AssertionError('Historical V2 changed')
        if paired.read_frozen(args.v1)[1]!={k:v for k,v in identity.items() if not k.startswith('normalized_')}:raise AssertionError('V1 changed')
        for name,value in identity['v1_checkpoint_hashes'].items():
            if hashlib.sha256((Path(args.v1)/'conditioned'/name).read_bytes()).hexdigest()!=value:raise AssertionError('V1 checkpoint changed')
        atomic_json(out/'historical_hashes.json',dict(before=before,after=hashes(v2),unchanged=True))
        atomic_json(out/'results.json',dict(cd=previous['cd'],squared_cd=result,mse=previous['mse']))
        from .report import write_report
        write_report(out,v2)
        atomic_json(out/'status.json',dict(state='COMPLETED',successful_updates=400,source_sha=sha))
    except BaseException:
        atomic_json(out/'failure.json',dict(source_sha=sha,traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
