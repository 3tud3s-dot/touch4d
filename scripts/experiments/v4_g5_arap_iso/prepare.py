"""One independent Slurm GPU preparation job; no formal optimization here.

Runs paired full-protocol preflight, the prescribed once-only calibration and
exactly four diagnostic updates for each variant.  Each subprocess uses the
same committed, clean source.  Partial evidence is retained and never erased.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import hashlib
import numpy as np

from scripts.training.train import atomic_json
from scripts.experiments.v4_g5_signed_local.diagnostics import sha256
from .protocol import (VARIANTS, source_guard, resource_guard, environment,
                       calibration_read, gate_read, configure_runtime)


def _read(path):
    return json.loads(Path(path).read_text())


def reviewed_graph_proof(reviewed_root, calibration_root):
    """Read-only exact payload comparison with previously pixel-reviewed graphs."""
    reviewed = Path(reviewed_root); current = Path(calibration_root)
    review_file = reviewed/'graph_review.json'; review = _read(review_file)
    if review.get('passed') is not True or review.get('calibration_sha256') != sha256(reviewed/'calibration'/'calibration.json'):
        raise ValueError('Prior graph review/calibration identity changed')
    expected_figures = {f'initial_graph_ep{ep}_start{start}.png' for ep, start in ((2, 141), (7, 105))}
    if set(review.get('figure_sha256', {})) != expected_figures:
        raise ValueError('Both reviewed training graph figures are required')
    for name, expected in review.get('figure_sha256', {}).items():
        if sha256(reviewed/'figures'/name) != expected:
            raise ValueError('Reviewed graph figure changed')
    arrays = {}
    for ep, start in ((2, 141), (7, 105)):
        name = f'graph_ep{ep}_start{start}.npz'
        old, new = reviewed/'calibration'/name, current/name
        if review.get('graph_array_sha256', {}).get(name) != sha256(old):
            raise ValueError('Reviewed initial graph array changed')
        payload = hashlib.sha256()
        with np.load(old, allow_pickle=False) as a, np.load(new, allow_pickle=False) as b:
            keys = ('initial_world', 'initial_normalized', 'mask', 'edges', 'rest_lengths')
            for key in keys:
                left, right = a[key], b[key]
                equal = np.array_equal(left, right, equal_nan=True) if np.issubdtype(left.dtype, np.floating) else np.array_equal(left, right)
                if left.dtype != right.dtype or left.shape != right.shape or not equal:
                    raise ValueError('Current graph differs from reviewed initial graph: '+name+':'+key)
                payload.update(key.encode()); payload.update(str(right.dtype).encode())
                payload.update(str(right.shape).encode()); payload.update(right.tobytes())
        arrays[name] = dict(reviewed_path=str(old.resolve()), reviewed_sha256=sha256(old),
            current_path=str(new.resolve()), current_sha256=sha256(new), payload_sha256=payload.hexdigest(), same_payload=True)
    return dict(passed=True, exact_initial_graph_payload=True, review_path=str(review_file.resolve()),
        review_sha256=sha256(review_file), reviewer=review.get('reviewer'), notes=review.get('notes'), arrays=arrays)


def _one(args, module, extra, label):
    source_guard(args.expected_sha)
    resource_guard(require_idle=True, inspect_cuda=False)
    command = [sys.executable, '-B', '-m', 'scripts.experiments.v4_g5_arap_iso.'+module,
        '--root', args.root, '--depth-root', args.depth_root, '--g0-reference', args.g0_reference,
        '--expected-sha', args.expected_sha, *extra]
    logs = Path(args.common)/'logs'; logs.mkdir(parents=True, exist_ok=True)
    path = logs/(label+'.log')
    if path.exists():
        raise FileExistsError('Retained preparation log already exists: '+str(path))
    with path.open('xb') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'{label} failed with exit {result.returncode}; inspect {path}')
    source_guard(args.expected_sha)
    return dict(command=command, log=str(path.resolve()), log_sha256=sha256(path), exit_code=result.returncode)


def preparation(args):
    common = Path(args.common)
    if not common.is_absolute() or common.is_symlink():
        raise ValueError('Independent absolute common directory required')
    common.mkdir(parents=True, exist_ok=True)
    if (common/'gate.json').exists():
        raise FileExistsError('Completed gate is immutable; do not rerun preparation')
    source, changes = source_guard(args.expected_sha)
    # Keep this controlling process free of a CUDA context while child GPU
    # tools require exclusive allocation ownership.  Query full runtime only
    # after every child has exited and released its own context.
    gpu = resource_guard(require_idle=True, inspect_cuda=False)
    started = time.monotonic(); commands = []
    preflight = common/'engineering'/'preflight.json'
    if not preflight.exists():
        commands.append(_one(args, 'train', ['--variant', VARIANTS[0], '--output', args.c_output,
            '--common', args.common, '--preflight-only'], 'preflight'))
    proof = _read(preflight)
    if proof.get('passed') is not True or proof.get('source_sha') != source:
        raise ValueError('Wrong committed-source paired preflight')
    data, initialization = proof['data'], proof['initialization']
    calibration = common/'calibration'/'calibration.json'
    if not calibration.exists():
        if calibration.parent.exists():
            raise FileExistsError('Partial calibration retained; inspect before a new preparation')
        commands.append(_one(args, 'calibrate', ['--output', str(calibration.parent)], 'calibration'))
    calibrated = calibration_read(calibration)
    graph_proof = reviewed_graph_proof(args.reviewed_graphs, calibration.parent)
    atomic_json(common/'reviewed_graph_equivalence.json', graph_proof)
    smoke = {}
    for variant in VARIANTS:
        out = common/'smoke'/variant; completion = out/'completion.json'
        if not completion.exists():
            if out.exists():
                raise FileExistsError('Partial Smoke retained; inspect before a new preparation')
            commands.append(_one(args, 'smoke', ['--variant', variant, '--calibration', str(calibration),
                '--output', str(out)], 'smoke-'+variant))
        record = _read(completion)
        if (record.get('passed') is not True or record.get('source_sha') != source
                or record.get('variant') != variant or type(record.get('successful_updates')) is not int
                or record['successful_updates'] != 4 or record.get('calibration_sha256') != sha256(calibration)):
            raise ValueError('Both variants require exactly four same-source Smoke updates')
        smoke[variant] = dict(completion_path=str(completion.resolve()), completion_sha256=sha256(completion))
    source_guard(source); runtime = environment()
    gate = dict(passed=True, source_sha=source, calibration_sha256=sha256(calibration),
        manifest_hash=data['manifest_hash'], epoch_order_hashes=data['epoch_order_hashes'],
        initialization=initialization, smoke=smoke,
        graph_review_path=str((common/'reviewed_graph_equivalence.json').resolve()),
        graph_review_sha256=sha256(common/'reviewed_graph_equivalence.json'),
        preparation=dict(commands=commands, runtime=runtime, resource=gpu,
            changed_files_sha256=changes, seconds=time.monotonic()-started,
            calibration_source=calibrated.get('source_sha'),
            graph_review='Each initial/world/mask/edge/rest-length array exactly matches the two previously pixel-reviewed training graphs. '
                         'Spatial neighbors are not material topology.'),
        note='Smoke states are diagnostic only. Formal C/D restart from the common fresh seed42 G5 state.')
    atomic_json(common/'gate.json', gate)
    gate_read(common, calibration, source, data, initialization)
    atomic_json(common/'preparation_completion.json', dict(passed=True, source_sha=source,
        gate_sha256=sha256(common/'gate.json'), job_id=os.environ.get('SLURM_JOB_ID'),
        step_id=os.environ.get('SLURM_STEP_ID'), seconds=time.monotonic()-started))
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'depth-root', 'g0-reference', 'common', 'c-output', 'd-output', 'reviewed-graphs', 'expected-sha'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(); configure_runtime(4)
    try:
        preparation(args)
    except BaseException:
        common = Path(args.common)
        if common.is_dir():
            atomic_json(common/f'preparation_failure-{os.getpid()}.json', dict(source_sha=args.expected_sha,
                job_id=os.environ.get('SLURM_JOB_ID'), step_id=os.environ.get('SLURM_STEP_ID'), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
