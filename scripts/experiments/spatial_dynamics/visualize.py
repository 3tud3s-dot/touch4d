"""Reuse the formal GS exporter; label untrained-horizon diagnostics explicitly."""
import argparse,json,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scripts.training.train import MODEL_KEYS,atomic_json,to_device,gs_render_settings
from scripts.training.visualize import export_gaussians
from scripts.dataloader.collate import collate_fn
from scripts.data_process import DataProcessor
from .runner import dataset,read_manifest,sha,hand_inputs,make_renderer
from .model import SpatialPredictor
from .metrics import enriched_metrics


def load_research(path,horizon):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['version']!='touch4d-spatial-dynamics-v1':raise ValueError('Wrong research checkpoint')
    config=state['config'];old_h=config['horizon']
    options={k:config[k] for k in (*MODEL_KEYS,'spatial','decoder','feedback','regularization')}
    old=SpatialPredictor(**options).eval();old.load_state_dict(state['model'])
    if old_h==horizon:return old,old,None,state
    if old_h!=5 or horizon!=15:raise ValueError('Only preregistered H5-to-H15 hold-last diagnostic allowed')
    model=SpatialPredictor(**dict(options,horizon=horizon)).eval();weights=dict(state['model'])
    for name in ('condition_time.weight','prediction_time.weight'):
        weights[name]=torch.cat((weights[name],weights[name][-1:].expand(horizon-old_h,-1)),0)
    model.load_state_dict(weights)
    return model,old,'H5权重外推至H15；末项时间编码保持；h6–15未经训练，不是H15训练结果',state


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--checkpoints',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm CUDA required')
    torch.set_num_threads(4);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    manifest=read_manifest(a.manifest);spec=json.loads(Path(a.checkpoints).read_text());records=[]
    for label,path in spec.items():
        before=sha(path);model,old,warning,state=load_research(path,15);model=model.cuda();old=old.cuda()
        processor=DataProcessor.from_state_dict(state['processor']);renderer=make_renderer(dict(output_mode='gs_geometry',render_settings=gs_render_settings()))
        for row in manifest['windows']['15']['cases']:
            ep,start=row['episode'],row['start'];split='train' if ep in (1,2,4,5,7,8) else 'val'
            ds=dataset(a.root,split,15);key=(ep,start)
            if key not in ds.windows:ds.windows=ds.windows+(key,)
            raw=collate_fn([ds[ds.windows.index(key)]]);batch=processor(to_device(raw,'cuda'))
            batch['inputs']=hand_inputs(batch['inputs'],batch['meta']['episode_id'],batch['meta']['scene'])
            prediction=model(batch['inputs'])
            if warning:
                short=dict(batch['inputs'])
                for k in ('action_state','action_valid','controller_points','controller_mask','tactile','valid_mask','tactile_spatial_mask','tactile_age_us'):
                    short[k]=short[k][:,:5]
                ref=old(short)
                assert all(torch.equal(prediction[k][:,:5],ref[k]) for k in prediction),'Extrapolation changed trained H5 prefix'
            folder=out/f'{label}-ep{ep}-s{start}'
            args=SimpleNamespace(output=str(folder),checkpoint=path,episode=ep,start=start,split=split)
            config=dict(state['config'],scene='001-rope',horizon=15,render_settings=gs_render_settings())
            export_gaussians(args,config,raw,batch,prediction,processor)
            caption=f'{label} / {state["completed_updates"]} updates / {state["config"]["spatial"]} {state["config"]["decoder"]} {state["config"]["feedback"]}'
            if warning:caption+=' / '+warning
            for name in ('index.html','geometry.html'):
                target=folder/name;text=target.read_text(encoding='utf-8')
                text=text.replace('<body>','<body>')
                text=text.replace('<h1>',f'<p style="background:#fff3cd;padding:12px">{caption}</p><h1>',1)
                if state['config']['decoder']=='D2':
                    text=text.replace('P2 结构锚点 attention + P1a 位移 Decoder；尺度、旋转、颜色及不透明度继承初态。','D2 锚点 SE(3)；尺度、颜色及不透明度继承初态，四元数随合法混合旋转更新。')
                target.write_text(text,encoding='utf-8')
            metrics=enriched_metrics(prediction,batch,processor,renderer,model.trace)
            arrays=np.load(folder/'prediction.npz')
            distinct=[sha_bytes(arrays['predicted_xyz'][h]) for h in range(15)]
            record=dict(label=label,episode=ep,start=start,frames=row['frame_ids'],checkpoint=path,
                checkpoint_sha256=before,config=state['config'],updates=state['completed_updates'],warning=warning,
                folder=folder.name,metrics=metrics,prediction_frame_sha256=distinct,
                unique_prediction_states=len(set(distinct)),first_h5_exact=True if warning else None)
            atomic_json(folder/'evidence.json',record);records.append(record)
            del raw,batch,prediction;model.trace=[]
        assert sha(path)==before
        del model,old,state;torch.cuda.empty_cache()
    atomic_json(out/'visualizations.json',records)


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value.tobytes()).hexdigest()

if __name__=='__main__':main()
