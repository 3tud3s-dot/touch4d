"""Read-only full Val100 Best/Last acceptance after eight head-only epochs."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.training.train import atomic_json
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.protocol import configuration,preflight
from scripts.experiments.v4_g5_signed_local.formal_checkpoint import load_for_evaluation
from scripts.experiments.v4_g5_signed_local.diagnostics import sha256,state_hash,write_csv
from .model import GeometryAdapter
from .formal_checkpoint import read
from .smoke import BASE,BACKBONE_HASH,source_guard,prepared,PICTURES
from .evaluation import (CachedRenderer,equality,objective_record,geometry_rows,save_images,
    summarize,assert_metrics_equal,REFERENCE,TOLERANCE)


@torch.no_grad()
def run(args):
    source,files=source_guard(args.expected_sha);out=Path(args.output)
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm GPU required')
    if not out.is_absolute() or out.exists():raise FileExistsError('New absolute acceptance directory required')
    if sha256(args.checkpoint)!=BACKBONE_HASH:raise ValueError('Wrong G5 checkpoint')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(args.run).resolve();protocol=json.loads((root/'protocol.json').read_text());identity=protocol['identity']
    if identity['source_sha']!=source or identity['run_id']!=str(root):raise ValueError('Formal source/Run mismatch')
    raw,data=preflight(configuration(args.root,args.depth_root,source,'direct'),args.g0_reference)
    if identity['data']!=data:raise ValueError('Formal data Manifest changed')
    processor=DataProcessor('001-rope');backbone=G5DirectPredictor(horizon=15,prediction_interval=1)
    state=load_for_evaluation(args.checkpoint,backbone,processor.state_dict(),source_sha=BASE,manifest=data['manifest']);del state
    adapter=GeometryAdapter(backbone).cuda().eval().requires_grad_(False)
    if adapter.geometry_config!=identity['geometry_config'] or state_hash(backbone.state_dict())!=identity['backbone_state_hash']:
        raise ValueError('Formal geometry/backbone mismatch')
    heads={};evidence={}
    for label,name in (('geometry_best','best.pt'),('geometry_last','last.pt')):
        state=read(root/name,identity);heads[label]=copy.deepcopy(state['head'])
        evidence[label]=dict(path=str(root/name),sha256=sha256(root/name),epoch=state['ledger']['completed_epochs'],
            successful_updates=state['ledger']['successful_updates'],best_epoch=state['ledger']['best_epoch'])
    if evidence['geometry_last']['epoch']!=8:raise ValueError('Eight completed epochs required')
    out.mkdir(parents=True)
    for name in ('results','images','figures','logs'):(out/name).mkdir()
    renderer=CachedRenderer();tables={k:[] for k in ('window_metrics','horizons','camera_metrics','geometry_statistics','baseline_equivalence','val_summary')}
    before=state_hash(backbone.state_dict());begin=time.perf_counter()
    def one(raw_batch,split,ep,start):
        batch=prepared(raw_batch,processor,adapter,ep,start);x=batch['inputs']
        motion,trace,features=adapter.motion_forward(x);original=backbone(x)
        equality(original,motion,x['gs_mask'],all_attributes=True);base_row=None
        for mode in ('original','geometry_best','geometry_last'):
            if mode=='original':pred,geo=original,{}
            else:
                adapter.head.load_state_dict(heads[mode],strict=True)
                pred,geo=adapter.render_from_motion(motion,trace,features,x['gs_mask'])
            match=equality(original,pred,x['gs_mask'],all_attributes=mode=='original')
            _,row,horizons,cameras=objective_record(pred,batch,processor,renderer,mode=mode,split=split,episode=ep,start=start)
            if mode=='original':base_row=row
            elif abs(row['cd_mm']-base_row['cd_mm'])>TOLERANCE['cd_mm']:raise AssertionError('Formal training changed center CD')
            tables['window_metrics'].append(row);tables['horizons'].extend(horizons);tables['camera_metrics'].extend(cameras)
            tables['baseline_equivalence'].append(dict(mode=mode,split=split,episode=ep,start=start,**match))
            if geo:tables['geometry_statistics'].extend(geometry_rows(pred,motion,geo,batch,mode=mode,split=split,episode=ep,start=start))
            if (ep,start) in PICTURES:
                save_images(renderer,out/'images',f'ep{ep}_start{start}_{mode}',save_gt=mode=='original')
                if geo:
                    np.savez_compressed(out/'figures'/f'ep{ep}_start{start}_{mode}_attributes.npz',
                        xyz=processor.denormalize_xyz(original['xyz'],x['gs_mask'][:,None].expand(-1,15,-1)).cpu().numpy(),
                        mask=x['gs_mask'].cpu().numpy(),log_delta=geo['log_scale_delta'].cpu().numpy(),rotation_vector=geo['rotation_vector'].cpu().numpy())
        off,_=adapter.render_from_motion(motion,trace,features,x['gs_mask'],'geometry_off')
        equality(original,off,x['gs_mask'],all_attributes=True)
        renderer.clear()
    for i,raw_batch in enumerate(raw['val']):
        ep=int(raw_batch['meta']['episode_id'][0]);start=int(raw_batch['meta']['frame_ids'][0,0]);one(raw_batch,'val',ep,start)
        if (i+1)%10==0:
            for name,rows in tables.items():write_csv(out/'results'/f'{name}.csv',rows)
            print(json.dumps(dict(acceptance_val_completed=i+1)),flush=True)
    ds=raw['train'].dataset;one(collate_fn([ds[ds.windows.index((7,105))]]),'train',7,105)
    for mode in ('original','geometry_best','geometry_last'):tables['val_summary'].extend(summarize(tables['window_metrics'],mode))
    original=next(r for r in tables['val_summary'] if r['mode']=='original' and r['episode']=='episode_equal')
    assert_metrics_equal(REFERENCE,original)
    baseline={(r['episode'],r['start']):r for r in tables['window_metrics'] if r['mode']=='original' and r['split']=='val'}
    tables['paired_window_effects']=[dict(mode=r['mode'],episode=r['episode'],start=r['start'],
        **{'delta_'+k:r[k]-baseline[r['episode'],r['start']][k] for k in REFERENCE})
        for r in tables['window_metrics'] if r['mode']!='original' and r['split']=='val']
    for name,rows in tables.items():write_csv(out/'results'/f'{name}.csv',rows)
    if before!=state_hash(backbone.state_dict()) or sha256(args.checkpoint)!=BACKBONE_HASH:raise AssertionError('Frozen source changed')
    for label,name in (('geometry_best','best.pt'),('geometry_last','last.pt')):
        if sha256(root/name)!=evidence[label]['sha256']:raise AssertionError('Formal checkpoint changed during evaluation')
    atomic_json(out/'provenance.json',dict(source_sha=source,source_files=files,formal_run=str(root),
        checkpoints=evidence,backbone_sha256=BACKBONE_HASH,backbone_state_hash=before,
        job_id=os.environ['SLURM_JOB_ID'],step_id=os.environ.get('SLURM_STEP_ID'),seconds=time.perf_counter()-begin,
        weights_unchanged=True,optimizer_updates=0,xyz_bit_exact=True))
    atomic_json(out/'completion.json',dict(passed=True,full_val_windows=100,epochs=8,summary=tables['val_summary']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','depth-root','g0-reference','checkpoint','run','output','expected-sha'):parser.add_argument('--'+key,required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
