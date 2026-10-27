"""Real-data V4 integration gate followed by exactly 400 single-window updates.

Uses the production processor, renderer and loss. No P2 fixed400 training loop,
no teacher forcing, no validation episodes, no automatic retries or tuning.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
from types import SimpleNamespace

import torch

from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.training.model import initial_knn
from scripts.training.rendering import GaussianRenderer
from scripts.training.train import GS_CAMERAS, atomic_json, copy_initial, gs_render_settings, to_device
from scripts.training.visualize import export_gaussians
from scripts.experiments.fixed_window.fixed_fit import TargetCache, digest, prediction_metrics
from scripts.training.v4_runtime import (build_dataset, build_model, prepare_batch, v4_objective,
                                        save_checkpoint, load_checkpoint)

UPDATES, EVAL_EVERY = 400, 20
GROUPS = ('gs_encoder', 'local', 'conditions.hand', 'conditions.controller', 'cross', 'motion')


def frozen_config(source_sha, root, depth_root):
    return dict(source_sha=source_sha, scene='001-rope', episode=2, start=73,
                horizon=10, prediction_interval=1, seed=42, batch_size=1, updates=UPDATES,
                optimizer='AdamW', lr=1e-4, weight_decay=1e-4, clip_norm=1., eval_every=EVAL_EVERY,
                loss=dict(cd=1., rgb=1., alpha=.1), root=str(Path(root).resolve()),
                depth_root=str(Path(depth_root).resolve()), cameras=GS_CAMERAS['001-rope'],
                render_settings=gs_render_settings(), precision='float32; TF32 disabled',
                selection='best_joint is joint objective; lowest_cd is a separate descriptive diagnostic')


def norm(value):
    return float(value.detach().double().square().sum().sqrt())


def group_stats(model, before=None, gradients=False):
    result = {}
    for group in GROUPS:
        params = [(n, p) for n, p in model.named_parameters() if n.startswith(group)]
        result[group] = dict(parameter_norm=sum(norm(p)**2 for _, p in params)**.5)
        if gradients:
            result[group].update(gradient_norm=sum(norm(p.grad)**2 for _, p in params if p.grad is not None)**.5,
                                 missing_gradients=sum(p.grad is None for _, p in params))
        if before is not None:
            change = sum(norm(p-before[n])**2 for n, p in params)**.5
            result[group].update(update_norm=change,
                                 relative_update=change/max(sum(norm(before[n])**2 for n, _ in params)**.5, 1e-30))
    return result


def read_window(config, model):
    dataset = build_dataset(config['scene'], 'train', config['root'], horizon=config['horizon'],
                            prediction_interval=config['prediction_interval'], with_depth=True,
                            depth_root=config['depth_root'], depth_cameras=config['cameras'])
    key = (config['episode'], config['start'])
    if dataset.windows.count(key) != 1:
        raise ValueError('Requested exact training window is not available')
    raw = collate_fn([dataset[dataset.windows.index(key)]])
    expected = list(range(73, 84))
    if raw['meta']['frame_ids'].tolist() != [expected]:
        raise ValueError('Expected initial73, conditions73..82, targets74..83')
    pixels = raw['targets']['target_depth_valid'].sum((-1, -2))[0]
    if not (pixels > 0).all():
        raise ValueError('Every horizon and supervision camera needs valid target pixels')
    processor = DataProcessor('001-rope')
    normalized = prepare_batch(raw, processor, model)
    if normalized['inputs']['operation_hands'].tolist() != [[False, True]]:
        raise ValueError('Episode 2 must use right hand only')
    return raw, normalized, processor, dict(
        frame_ids=expected, condition_frame_ids=expected[:-1], target_frame_ids=expected[1:],
        raw_data_sha256=digest(raw), normalized_data_sha256=digest(normalized),
        initial_points=int(raw['inputs']['gs_mask'].sum()),
        target_points=raw['targets']['target_lengths'][0].tolist(),
        positive_pixels=pixels.tolist(), cameras=raw['meta']['depth_camera_ids'][0],
        operation_hands=[[False, True]], processor=processor.state_dict())


def condition_audit(model, batch):
    rows = []
    for h in range(model.config['horizon']):
        _, _, features, valid, _ = model.conditions.features(batch['inputs'], h)
        if valid[0, :30].any() or torch.any(features[0, :30] != 0):
            raise AssertionError('Masked left hand leaked')
        rows.append(dict(frame_id=int(batch['inputs']['condition_frame_ids'][0, h]),
                         valid_controllers=int(valid.sum()), a0_rms=norm(features[..., 3:6])/max(int(valid.sum()), 1)**.5,
                         mean_valid=int(features[..., 11].sum()), max_valid=int(features[..., 12].sum()),
                         valid_zero_mean=int(((features[..., 7] == 0) & (features[..., 11] > 0)).sum()),
                         coverage_mean=float(features[..., 9][valid].mean()),
                         age_mean=float(features[..., 10][valid].mean())))
    return rows


@torch.no_grad()
def evaluate(model, batch, processor, renderer, neighbors):
    prediction, trace = model(batch['inputs'], return_trace=True)
    metrics = prediction_metrics(prediction, batch, processor, renderer, 512, neighbors)
    scale = processor.state_dict()['scale']
    metrics['loss'] = metrics['mean_cd_mm']/(scale*1000)+metrics['mean_rgb']+.1*metrics['mean_alpha']
    keep = batch['inputs']['gs_mask'][0]
    delta = (prediction['xyz'][0, :, keep]-batch['inputs']['gs_xyz'][0, keep])*scale*1000
    metrics['motion_rms_mm'] = delta.square().sum(-1).mean(-1).sqrt().tolist()
    metrics['attention'] = []
    for step in trace['steps']:
        weights = step['alpha'][trace['field'].anchor_mask]
        metrics['attention'].append(dict(null_mean_by_head=weights[..., -1].mean(0).tolist(),
                                         controller_mass_by_head=weights[..., :60].mean(0).tolist()))
    return metrics, prediction


def successful_update(model, optimizer, batch, processor, renderer):
    optimizer.zero_grad(set_to_none=True)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    prediction = model(batch['inputs'])
    loss, parts = v4_objective(prediction, batch, processor, renderer, 512)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite loss/gradient; no optimizer update performed')
    groups = group_stats(model, gradients=True)
    gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    clipped = group_stats(model, gradients=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite parameters after update')
    return dict(loss=float(loss.detach()), cd_normalized=float(parts['geometry'].mean()),
                rgb=float(parts['rgb'].detach().mean()), alpha=float(parts['alpha'].detach().mean()),
                gradient_norm=gradient, clipped=gradient > 1., raw_groups=groups,
                clipped_groups=clipped, actual_updates=group_stats(model, before=before))


def integration_smoke(directory, config, raw, batch, processor, manifest):
    directory.mkdir(exist_ok=False)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model = build_model('v4_stage1').cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    renderer = TargetCache(GaussianRenderer())
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    # Gate tests real full H10 rendering and loss, not just model instantiation.
    row = successful_update(model, optimizer, batch, processor, renderer)
    prediction, trace = model(batch['inputs'], return_trace=True)
    gradients = torch.autograd.grad(prediction['xyz'][:, -1].square().mean(),
                                    [s['state'] for s in trace['steps']], allow_unused=True)
    if any(g is None or not torch.isfinite(g).all() for g in gradients):
        raise AssertionError('Missing/nonfinite cross-step graph')
    checkpoint = directory/'one_update.pt'
    save_checkpoint(checkpoint, model, optimizer, 1, source_sha=config['source_sha'], processor_state=processor.state_dict())
    clone = build_model('v4_stage1').cuda(); clone_opt = torch.optim.AdamW(clone.parameters(), lr=1e-4, weight_decay=1e-4)
    load_checkpoint(checkpoint, clone, clone_opt, processor_state=processor.state_dict())
    with torch.no_grad():
        restored = clone(batch['inputs'])
        torch.testing.assert_close(restored['xyz'], prediction['xyz'], rtol=0, atol=0)
    report = dict(passed=True, config=config, manifest=manifest, condition_audit=condition_audit(model, batch),
                  update=row, shape=list(prediction['xyz'].shape), state_gradients=[dict(is_none=False, norm=norm(g)) for g in gradients],
                  allocated_peak=torch.cuda.max_memory_allocated(), reserved_peak=torch.cuda.max_memory_reserved(),
                  seconds=time.perf_counter()-start, checkpoint_restore_exact=True,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, python=platform.python_version(),
                  job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(directory/'acceptance.json', report)
    print('REAL_CUDA_SMOKE_PASSED', json.dumps({k: report[k] for k in ('gpu', 'shape', 'allocated_peak', 'seconds')}), flush=True)
    return report


def publish_checkpoint(path, model, opt, n, config, processor, run_state):
    save_checkpoint(path, model, opt, n, source_sha=config['source_sha'], processor_state=processor.state_dict(),
                    run_state=run_state, replace_existing=path.exists())


def check_resume(state, config, manifest, run_id):
    saved = state.get('run_state') or {}
    if (state.get('source_sha') != config['source_sha'] or saved.get('config') != config
            or saved.get('manifest') != manifest or saved.get('run_id') != run_id):
        raise ValueError('Resume needs identical source, window, data, optimizer, loss and run directory')
    if not 0 <= state['successful_steps'] <= UPDATES:
        raise ValueError('Invalid cumulative update count')
    history = saved.get('updates', [])
    if [r['update'] for r in history] != list(range(1, state['successful_steps']+1)):
        raise ValueError('Successful update ledger disagrees with checkpoint')
    return saved


def export_snapshot(directory, label, model, raw, batch, processor, config, prediction=None):
    destination = directory/'visualizations'/label
    if destination.exists():
        if not (destination/'index.html').is_file():
            raise RuntimeError(f'Incomplete prior export retained: {destination}')
        return
    destination.parent.mkdir(exist_ok=True)
    with torch.no_grad():
        prediction = model(batch['inputs']) if prediction is None else prediction
        args = SimpleNamespace(output=str(destination), checkpoint='', split='train', episode=2, start=73)
        export_gaussians(args, dict(config, gs_decoder='anchors', gs_structure='none'), raw, batch, prediction, processor)


def train(directory, config, raw, batch, processor, manifest, resume=False):
    directory.mkdir(exist_ok=resume)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model = build_model('v4_stage1').cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    renderer = TargetCache(GaussianRenderer())
    neighbors = initial_knn(batch['inputs']['gs_xyz'], batch['inputs']['gs_mask'], 16)
    run_id = str(directory.resolve())
    ledger = dict(run_id=run_id, config=config, manifest=manifest, updates=[], evaluations=[], lowest_cd_update=0, best_joint_update=0)
    progress = directory/'progress.pt'
    completed = 0
    if resume:
        state = load_checkpoint(progress, model, optimizer, processor_state=processor.state_dict())
        ledger = check_resume(state, config, manifest, run_id)
        completed = state['successful_steps']
    else:
        atomic_json(directory/'config.json', dict(config, model=model.config))
        atomic_json(directory/'manifest.json', manifest)
        with torch.no_grad():
            baseline = prediction_metrics(copy_initial(batch['inputs'], 'gs_geometry'), batch, processor, renderer, 512, neighbors)
        ledger['copy_initial'] = baseline
        ledger['gt_geometry_change'] = dict(copy_cd_mm_by_horizon=baseline['cd_mm'],
            centroid_change_mm=[((t[0, m[0]].mean(0)-batch['inputs']['gs_xyz'][0, batch['inputs']['gs_mask'][0]].mean(0))
                                *processor.state_dict()['scale']*1000).tolist()
                               for t, m in zip(batch['targets']['target_xyz'], batch['targets']['target_mask'])],
            note='Independent reconstructed point sets: centroid and CD changes are not material-particle trajectories.')
        publish_checkpoint(progress, model, optimizer, 0, config, processor, ledger)
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    while True:
        if completed % EVAL_EVERY == 0 and not any(row['update'] == completed for row in ledger['evaluations']):
            metrics, prediction = evaluate(model, batch, processor, renderer, neighbors)
            metrics['update'] = completed
            previous = ledger['evaluations']
            improve_cd = not previous or metrics['mean_cd_mm'] < min(r['mean_cd_mm'] for r in previous)
            improve_joint = not previous or metrics['loss'] < min(r['loss'] for r in previous)
            ledger['evaluations'].append(metrics)
            if improve_cd: ledger['lowest_cd_update'] = completed
            if improve_joint: ledger['best_joint_update'] = completed
            publish_checkpoint(directory/f'update_{completed:06d}.pt', model, optimizer, completed, config, processor, ledger)
            if improve_cd: publish_checkpoint(directory/'lowest_cd.pt', model, optimizer, completed, config, processor, ledger)
            if improve_joint: publish_checkpoint(directory/'best_joint.pt', model, optimizer, completed, config, processor, ledger)
            if completed == UPDATES: publish_checkpoint(directory/'last_400.pt', model, optimizer, completed, config, processor, ledger)
            publish_checkpoint(progress, model, optimizer, completed, config, processor, ledger)
            atomic_json(directory/'metrics.json', ledger)
            print(json.dumps({k: metrics[k] for k in ('update', 'loss', 'mean_cd_mm', 'mean_rgb', 'mean_alpha')}), flush=True)
            if completed in (0, 200, 400): export_snapshot(directory, f'update_{completed:03d}', model, raw, batch, processor, config, prediction)
            del prediction
        if completed == UPDATES: break
        started = time.perf_counter()
        row = successful_update(model, optimizer, batch, processor, renderer)
        completed += 1  # Only after a finite successful optimizer step.
        row.update(update=completed, seconds=time.perf_counter()-started)
        ledger['updates'].append(row)
        # Atomic recovery point for every successful update, not just evaluations.
        publish_checkpoint(progress, model, optimizer, completed, config, processor, ledger)
    for n in (0, 200, 400):
        load_checkpoint(directory/f'update_{n:06d}.pt', model, processor_state=processor.state_dict())
        export_snapshot(directory, f'update_{n:03d}', model, raw, batch, processor, config)
    load_checkpoint(directory/'lowest_cd.pt', model, processor_state=processor.state_dict())
    export_snapshot(directory, 'lowest_cd', model, raw, batch, processor, config)
    with torch.no_grad():
        real, prediction = evaluate(model, batch, processor, renderer, neighbors)
        diagnostics = {}
        for mode in ('freeze_action', 'zero_tactile'):
            altered = copy.deepcopy(batch)
            x = altered['inputs']
            if mode == 'freeze_action':
                for name in ('controller_points', 'action_state'):
                    x[name] = x[name][:, :1].expand_as(x[name]).clone()
                for name in ('controller_mask', 'action_valid'):
                    x[name] = x[name] & x[name][:, :1]
            else: x['tactile'].zero_()
            other, output = evaluate(model, altered, processor, renderer, neighbors)
            keep = batch['inputs']['gs_mask'][0]
            diff = (output['xyz'][0, :, keep]-prediction['xyz'][0, :, keep])*processor.state_dict()['scale']*1000
            diagnostics[mode] = dict(delta_cd_mm=other['mean_cd_mm']-real['mean_cd_mm'], metrics=other,
                                    output_change_rms_mm=diff.square().sum(-1).mean(-1).sqrt().tolist())
    ledger.update(successful_updates=completed, condition_diagnostics=diagnostics,
                  elapsed_this_invocation_seconds=time.perf_counter()-begin,
                  allocated_peak=torch.cuda.max_memory_allocated(), reserved_peak=torch.cuda.max_memory_reserved(),
                  checkpoints={name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                               for name in ('lowest_cd.pt', 'best_joint.pt', 'last_400.pt')})
    atomic_json(directory/'metrics.json', ledger)
    with (directory/'training_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['update', 'loss', 'cd_normalized', 'rgb', 'alpha', 'gradient_norm', 'clipped', 'seconds'])
        writer.writeheader(); writer.writerows({k: row[k] for k in writer.fieldnames} for row in ledger['updates'])
    from .report import write_report
    write_report(directory)
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--depth-root', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--stage', choices=['all', 'smoke'], default='all'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires an independently allocated Slurm CUDA GPU')
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if actual != args.expected_sha or subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Source SHA mismatch or dirty worktree')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=args.resume)
    import fcntl
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            config = frozen_config(actual, args.root, args.depth_root)
            raw, normalized, processor, manifest = read_window(config, build_model('v4_stage1'))
            batch = to_device(normalized, 'cuda')
            gate_path = output/'smoke/acceptance.json'
            if args.resume:
                gate = json.loads(gate_path.read_text())
                if not gate.get('passed') or gate['config'] != config or gate['manifest'] != manifest:
                    raise ValueError('Cannot resume without matching passed integration gate')
            else:
                atomic_json(output/'environment.json', dict(config=config, manifest=manifest, gpu=torch.cuda.get_device_name(),
                                                            job_id=os.environ['SLURM_JOB_ID'], torch=torch.__version__, python=platform.python_version()))
                integration_smoke(output/'smoke', config, raw, batch, processor, manifest)
            torch.cuda.empty_cache()
            if args.stage == 'all':
                result = train(output/'training', config, raw, batch, processor, manifest, args.resume)
                if result['successful_updates'] != UPDATES: raise AssertionError('Incorrect update count')
            atomic_json(output/'status.json', dict(status='completed', stage=args.stage, source_sha=actual,
                                                  successful_updates=UPDATES if args.stage == 'all' else 0))
        except BaseException:
            atomic_json(output/'failure.json', dict(traceback=traceback.format_exc(), source_sha=actual,
                                                   job_id=os.environ.get('SLURM_JOB_ID')))
            raise


if __name__ == '__main__': main()
