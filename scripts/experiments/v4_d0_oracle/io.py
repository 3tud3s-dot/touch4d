# Read-only provenance/data helpers reused from H15 audit 3994b716.
"""Frozen H15 inference only: no backward, optimizer, parameter updates or test split."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.experiments.full_training.v4_epochs import loader
from scripts.experiments.fixed_window.fixed_fit import digest, geometry_diagnostics
from scripts.training.v4_runtime import build_model, prepare_batch, load_checkpoint
from scripts.training.train import to_device, copy_initial, make_renderer
from scripts.training.losses import nearest_distances, render_samples
from scripts.training.visualize import png_bytes


BASE_SHA='60b7e6044a139a92bf7fdad29c262bde437ae2f9'
AUDIT_ROOT=Path('/data1/userdata/tcweng/experiments/touch4d/v4-001-h15-best-visual-v1')
CHECKPOINT_SHA='1fa35ddcc4c62ca359d8491d77bdd7888152a795ffae49b07a8480e190ccd05f'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(1024*1024),b''):h.update(v)
    return h.hexdigest()

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value

def metadata():
    return dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        command=[sys.executable,*sys.argv],python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        job=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name() if torch.cuda.is_initialized() else None,
        cwd=str(Path.cwd()),no_optimizer=True,no_backward=True)

def checked_model():
    provenance=json.loads((AUDIT_ROOT/'checkpoint_provenance.json').read_text(encoding='utf8'))
    checkpoint=AUDIT_ROOT/'best_snapshot.pt'
    if sha(checkpoint)!=CHECKPOINT_SHA or provenance['sha256']!=CHECKPOINT_SHA:raise ValueError('Frozen checkpoint hash mismatch')
    processor=DataProcessor('001-rope');model=build_model('v4_stage1',horizon=15,prediction_interval=1).eval()
    state=load_checkpoint(checkpoint,model,processor_state=processor.state_dict())
    if state['source_sha']!=BASE_SHA or state['run_state']['completed_epochs']!=6 or state['successful_steps']!=9084:
        raise ValueError('Wrong epoch/source/updates')
    config=json.loads((AUDIT_ROOT/'training_config.json').read_text(encoding='utf8'))['training']
    if config!=state['run_state']['config']:raise ValueError('Training config not equal checkpoint ledger')
    model.requires_grad_(False)
    return model,processor,state,config,dict(snapshot=str(checkpoint),sha256=CHECKPOINT_SHA,
        original_provenance=provenance,source_sha=state['source_sha'],config=state['config'],
        processor=state['processor'],epoch=6,successful_updates=9084,selection='best validation joint loss')

def batch_for(ds,key,processor,model):
    raw=collate_fn([ds[ds.windows.index(key)]])
    if raw['meta']['frame_ids'][0].tolist()!=list(range(key[1],key[1]+16)):raise ValueError('H15 frame mismatch')
    prepared=prepare_batch(raw,processor,model)
    if not prepared['targets']['target_depth_valid'].flatten(3).any(-1).all():raise ValueError('Missing camera supervision')
    for h in range(15):
        y=prepared['targets'];q=y['target_xyz'][h][0,y['target_mask'][h][0]]
        if not len(q) or not torch.isfinite(q).all():raise ValueError('Invalid target set')
    return raw,prepared
