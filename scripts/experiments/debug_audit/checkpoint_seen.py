"""Read-only inference on the original 24 seen training windows, no optimization.

The additional diagnostic distinguishes membership in a training *episode* from
membership in the frozen 24-window screen. It performs no model selection and
does not export images or perturb conditions. Run only in a new one-GPU Slurm job.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from scripts.data_process import DataProcessor
from scripts.experiments.spatial_dynamics.runner import (
    dataset, batch_at, read_manifest, sha, WEIGHTS,
)
from scripts.experiments.spatial_dynamics.visualize import load_research
from scripts.training.train import atomic_json, copy_initial, make_renderer, gs_render_settings
from .common import CASES, measurements
from .runner import BASE, CHECKPOINTS


def validate_seen_rows(manifest):
    rows = manifest['windows']['5']['train']
    keys = [(row['episode'], row['start']) for row in rows]
    if len(keys) != 24 or len(set(keys)) != 24:
        raise ValueError('Exactly the original 24 distinct training windows are required')
    for row in rows:
        if row['episode'] not in (1, 2, 4, 5, 7, 8) or row['start'] < 0:
            raise ValueError('Seen-window audit may not read validation/test as training')
        if row['frame_ids'] != list(range(row['start'], row['start'] + 6)):
            raise ValueError('Original seen window must have initial plus five consecutive targets')
    return rows


def require_execution(expected_sha):
    job = os.environ.get('SLURM_JOB_ID')
    if not job or job == '30903':
        raise RuntimeError('A new Slurm job is required; protected user allocation 30903 is forbidden')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one dedicated CUDA device is required')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if source != expected_sha or dirty:
        raise RuntimeError('Execution source SHA drift or tracked changes')
    changed = subprocess.check_output(['git', 'diff', '--name-only', BASE, 'HEAD', '--',
        'scripts/training', 'scripts/dataloader', 'scripts/data_process',
        'scripts/experiments/spatial_dynamics'], text=True).strip()
    if changed:
        raise RuntimeError('Frozen model or data implementation changed: ' + changed)
    return source


def summarize(rows):
    """Paired equal-window H5 summaries; no threshold or winner selection."""
    result = dict(windows=len(rows))
    for name in ('model', 'copy_initial'):
        source = [row[name] for row in rows]
        entry = {}
        for metric in ('cd_mm', 'rgb', 'alpha'):
            array = np.asarray([row[metric] for row in source], dtype=float)
            entry[metric + '_by_horizon'] = array.mean(0).tolist()
            entry['mean_' + metric] = float(array.mean())
        for metric in ('displacement_rms_mm', 'centroid_error_mm', 'centered_cd_mm'):
            array = np.asarray([[h[metric] for h in row['motion']] for row in source])
            entry[metric + '_by_horizon'] = array.mean(0).tolist()
        result[name] = entry
    diff = np.asarray([row['model']['cd_mm'] for row in rows]) - np.asarray([
        row['copy_initial']['cd_mm'] for row in rows])
    result['pred_minus_copy_cd_mm_by_horizon'] = diff.mean(0).tolist()
    result['pred_minus_copy_cd_mm_mean'] = float(diff.mean())
    result['windows_with_lower_mean_cd_than_copy'] = int((diff.mean(1) < 0).sum())
    result['gt_independent_set_centroid_motion_norm_mm_by_horizon'] = np.asarray([
        [np.linalg.norm(h['gt_centroid_motion_mm']) for h in row['model']['motion']]
        for row in rows]).mean(0).tolist()
    return result


@torch.no_grad()
def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    source = require_execution(args.expected_sha)
    # This reads and verifies the old manifest's source hashes once; no mutation.
    manifest = read_manifest(args.manifest)
    rows = validate_seen_rows(manifest)
    manifest_hash = sha(args.manifest)
    paths = {label: CHECKPOINTS[label] for label in ('S0-last', 'S2-C1-last')}
    before = {label: sha(path) for label, path in paths.items()}
    keys = {(row['episode'], row['start']) for row in rows}
    payload = dict(
        status='running', source_sha=source, base_sha=BASE,
        command=os.sys.argv, slurm_job_id=os.environ['SLURM_JOB_ID'],
        protected_user_job='30903', cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        manifest=str(Path(args.manifest).resolve()), manifest_sha256=manifest_hash,
        historical_hashes_before=before, models={},
        seen_window_list=[dict(episode=row['episode'], start=row['start'], frame_ids=row['frame_ids'])
                          for row in rows],
        original_diagnostic_cases=[dict(split=split, episode=ep, start=start,
                                       in_original_training_24=(ep, start) in keys)
                                   for split, ep, start in CASES],
        interpretation='Training-episode membership is not original 24-window membership. '
            'These are in-sample diagnostics of frozen H5 weights, not new training or generalization. '
            'GT independent-set centroid changes are not material-particle trajectories.',
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation prevents accidental overwriting even concurrent starts.
    with output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    copy_cache = {}
    processor_state = None
    try:
        ds = dataset(args.root, 'train', 5)
        for label, path in paths.items():
            model, _, _, state = load_research(path, 5)
            if (state['manifest_sha256'] != manifest_hash or state['weights'] != WEIGHTS
                    or state['config']['spatial'] != label.removesuffix('-last')
                    or state['completed_updates'] != 1200):
                raise ValueError('Checkpoint disagrees with original 24-window H5 protocol')
            processor = DataProcessor.from_state_dict(state['processor'])
            if processor_state is None:
                processor_state = state['processor']
            elif state['processor'] != processor_state:
                raise ValueError('Paired checkpoints have different normalization')
            model = model.cuda().eval()
            renderer = make_renderer(dict(output_mode='gs_geometry', render_settings=gs_render_settings()))
            records = []
            payload['models'][label] = dict(checkpoint=path, checkpoint_sha256=before[label],
                completed_updates=state['completed_updates'], config=state['config'],
                processor=processor.state_dict(), windows=records)
            for row in rows:
                key = (row['episode'], row['start'])
                batch = batch_at(ds, row, processor, 'cuda')
                pred = model(batch['inputs'])
                predicted = measurements(pred, batch, processor, renderer, model.trace)
                if key not in copy_cache:
                    copy_cache[key] = measurements(
                        copy_initial(batch['inputs'], 'gs_geometry'), batch, processor, renderer)
                record = dict(episode=key[0], start=key[1], frame_ids=row['frame_ids'],
                    source_data_digest=row.get('data_digest'), in_original_training_24=True,
                    model=predicted, copy_initial=copy_cache[key])
                record['pred_minus_copy_cd_mm'] = (
                    np.asarray(predicted['cd_mm']) - np.asarray(copy_cache[key]['cd_mm'])).tolist()
                records.append(record)
                del pred, batch
                model.trace = []
            payload['models'][label]['summary'] = summarize(records)
            payload['models'][label]['by_episode'] = {
                str(ep): summarize([record for record in records if record['episode'] == ep])
                for ep in sorted({row['episode'] for row in rows})}
            atomic_json(output, payload)
            del model, state
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        payload.update(status='complete', elapsed_seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException:
        payload.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        after = {label: sha(path) for label, path in paths.items()}
        payload['historical_hashes_after'] = after
        payload['historical_hashes_unchanged'] = before == after
        if before != after:
            payload.update(status='failed', integrity_error='Historical checkpoint changed')
        atomic_json(output, payload)
        if before != after:
            raise RuntimeError('Historical checkpoint changed')
    return payload


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True, help='Fresh JSON file; existing path is rejected')
    p.add_argument('--expected-sha', required=True)
    return p


def main():
    run(parser().parse_args())


if __name__ == '__main__':
    main()
