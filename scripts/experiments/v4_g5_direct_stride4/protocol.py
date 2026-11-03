"""Frozen G5 graph, H15 interval1, rotating start-frame phase and full Val100."""
from collections import Counter
from pathlib import Path
import json
import copy
import torch
from torch.utils.data import DataLoader, Subset
from scripts.experiments.v4_g5_signed_local import protocol as original
from scripts.experiments.v4_g4_global_motion.protocol import digest

COUNTS = (381, 379, 377, 377, 381, 379, 377, 377)
SAMPLING = dict(window_start_stride=4, phase_by_epoch=[0,1,2,3,0,1,2,3],
                rule='start_frame % 4 == epoch_0based % 4', shuffle='seed42+epoch_0based')


def configuration(root, depth_root, sha):
    spec = json.loads((Path(__file__).parent/'config.json').read_text())
    expected = dict(schema='g5-direct-stride4-runtime-v1', epochs=8, horizon=15,
                    prediction_interval=1, batch_size=1, seed=42, lr=1e-4,
                    weight_decay=1e-4, gradient_clip_norm=1., train_window_stride=4,
                    phase_by_epoch=[0,1,2,3,0,1,2,3], val_window_stride=5, val_windows=100)
    if spec != expected:
        raise ValueError('Runtime config differs from approved stride4 plan')
    return dict(original.configuration(root, depth_root, sha, 'direct'),
                protocol='g5_direct_stride4_rotating_v1', sampling=copy.deepcopy(SAMPLING))


def validate_configuration(c):
    if c != configuration(c['root'], c['depth_root'], c['source_sha']):
        raise ValueError('Stride4 configuration differs from frozen protocol')


def selected_indices(keys, epoch):
    if type(epoch) is not int or not 0 <= epoch < 8:
        raise ValueError('Epoch must be zero-based in [0,7]')
    return [i for i, (_, start) in enumerate(keys) if start % 4 == epoch % 4]


def sampled_loader(raw, epoch):
    if raw.batch_size != 1 or raw.num_workers != 0 or raw.drop_last:
        raise ValueError('Original loader settings changed')
    indices = selected_indices(raw.dataset.windows, epoch)
    result = DataLoader(Subset(raw.dataset, indices), batch_size=1, shuffle=True,
                        num_workers=0, drop_last=False, collate_fn=raw.collate_fn,
                        pin_memory=raw.pin_memory,
                        generator=torch.Generator().manual_seed(42 + epoch))
    return result, indices


def validate_data(data):
    manifest = data['manifest']
    keys, val = manifest['train']['window_keys'], manifest['val']['window_keys']
    if len(keys) != 1514 or len(val) != 100 or len({tuple(k) for k in keys}) != 1514:
        raise ValueError('Original manifest changed')
    if Counter(ep for ep, _ in keys) != {1:176,2:277,4:349,5:292,7:210,8:210}:
        raise ValueError('Training Episode counts changed')
    if Counter(ep for ep, _ in val) != {3:55,6:45}:
        raise ValueError('Val100 Episode counts changed')
    if data['manifest_hash'] != digest(manifest) or data['sampling'] != SAMPLING:
        raise ValueError('Manifest or sampling identity changed')
    orders = data['epoch_orders']
    if len(orders) != 8:
        raise ValueError('Eight orders required')
    visits = Counter()
    for epoch, order in enumerate(orders):
        if any(type(i) is not int for i in order):
            raise ValueError('Order indices must be integers')
        if len(order) != COUNTS[epoch] or sorted(order) != selected_indices(keys, epoch):
            raise ValueError('Wrong phase, duplicate or incomplete epoch order')
        visits.update(order)
    if len(visits) != 1514 or set(visits.values()) != {2}:
        raise ValueError('Every original window must be covered exactly twice')
    hashes = [digest(order) for order in orders]
    if data['epoch_order_hashes'] != hashes or data['observed_epoch_order_hashes'] != hashes:
        raise ValueError('Actual DataLoader order hashes differ')


def preflight(c, reference):
    validate_configuration(c)
    base = original.configuration(c['root'], c['depth_root'], c['source_sha'], 'direct')
    raw, old = original.preflight(base, reference)
    orders = []
    for epoch in range(8):
        sampled, indices = sampled_loader(raw['train'], epoch)
        orders.append([indices[i] for i in original.probe_order(sampled)])
    data = dict(old, epoch_orders=orders,
                epoch_order_hashes=[digest(v) for v in orders],
                observed_epoch_order_hashes=[digest(v) for v in orders],
                sampling=copy.deepcopy(SAMPLING),
                order_policy='Subset by per-Episode start-frame phase, then real DataLoader seed42+epoch; iterator base seed included')
    validate_data(data)
    return raw, data


def output_check(path, *, resume=False, protected=()):
    raw = Path(path)
    if not raw.is_absolute() or raw.is_symlink():
        raise ValueError('Output must be an absolute non-symlink Run path')
    out = raw.resolve()
    if out.parent.name != 'v4-g5-direct-stride4-v1':
        raise ValueError('Wrong experiment output directory')
    for p in protected:
        p = Path(p).resolve()
        if out == p or p in out.parents or out in p.parents:
            raise ValueError('Output overlaps protected input/source')
    if resume:
        if not (out/'initial.pt').is_file():
            raise ValueError('Missing durable initial checkpoint')
    elif out.exists():
        raise FileExistsError(out)
    return out


def full_validation_score(validation, manifest):
    return original.full_validation_score(validation, manifest)
