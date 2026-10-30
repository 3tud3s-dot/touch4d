"""Login-CPU-only, idempotent Slurm dependency submission for fixed C/D runs.

No GPU, training or SSH is invoked by this module.  Unknown submission results
are reconciled by an exact unique job-name AND source-bearing Comment match;
an ambiguous/missing result is blocked rather than blindly submitted twice.
The cluster may schedule the two independent single-GPU trains concurrently
or serially according to its ordinary resource/user limits.
"""
import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

from scripts.training.train import atomic_json
from scripts.experiments.v4_g4_global_motion.protocol import digest
from .protocol import BUCKETS, VARIANTS, source_guard


MIN_STORAGE_BYTES = 8*1024**3
COMMON_BUCKET = 'v4-g5-arap-iso-ablation-v1'
PHASES = ('prepare', 'g5_arap', 'g5_arap_iso', 'evaluate')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def _command(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def _quantity(token):
    token = token.rstrip('*')
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGTPE]?)', token, re.I)
    if not match:
        return None
    # quota -s numbers without a suffix are KiB blocks, not byte counts.
    value, suffix = match.groups()
    return float(value)*1024**('KMGTPE'.index(suffix.upper())+1 if suffix else 1)


def quota_headroom(raw):
    """Parse standard quota -s block columns; unsupported output fails closed."""
    headrooms = []
    for line in raw.splitlines():
        fields = line.split()
        start = next((i for i, value in enumerate(fields) if _quantity(value) is not None), None)
        if start is None or len(fields) < start+3:
            continue
        values = [_quantity(v) for v in fields[start:start+3]]
        if any(v is None for v in values):
            continue
        used, soft, hard = values
        limits = [v for v in (soft, hard) if v > 0]
        if limits:
            headrooms.append(min(limits)-used)
    if not headrooms:
        raise RuntimeError('Cannot establish quota headroom from quota -s; inspect retained output')
    return min(headrooms)


def _fields(line):
    return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)', line))


def precheck(args):
    if os.name != 'posix' or sys.platform != 'linux':
        raise RuntimeError('Submission is only run from the Linux cluster login CPU')
    if os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit from the login CPU, not inside Allocation32087 or another job')
    source, changes = source_guard(args.expected_sha)
    user = getpass.getuser()
    if user != 'tcweng':
        raise RuntimeError('Expected tcweng cluster owner')
    queue = _command(['squeue', '--user='+user, '--long'])
    partition = _command(['scontrol', 'show', 'partition', '5090', '-o'])
    config = _fields(partition)
    if config.get('PartitionName') != '5090' or config.get('State') != 'UP':
        raise RuntimeError('RTX5090 partition is unavailable')
    max_time = config.get('MaxTime')
    if max_time not in ('UNLIMITED', 'INFINITE'):
        match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)', max_time or '')
        if not match or int(match[1] or 0)*24+int(match[2])+int(match[3])/60 < 24:
            raise RuntimeError('Partition limit cannot support frozen 24-hour job budget')
    quota = _command(['quota', '-s'])
    headroom = quota_headroom(quota)
    # No cache cleanup or environment upgrade is ever done by submission.
    parent = Path(args.common).parent
    if not parent.is_dir():
        parent = next((p for p in parent.parents if p.is_dir()), None)
    if parent is None:
        raise RuntimeError('No existing independent output filesystem')
    disk = shutil.disk_usage(parent)
    if headroom < MIN_STORAGE_BYTES or disk.free < MIN_STORAGE_BYTES:
        raise RuntimeError('Less than8GiB safe quota/filesystem headroom; preserve existing results')
    for value in (args.root, args.depth_root, args.g0_reference, args.reviewed_graphs):
        if not Path(value).is_dir():
            raise FileNotFoundError(value)
    for name in ('config.json', 'manifest.json'):
        if not (Path(args.g0_reference)/name).is_file():
            raise FileNotFoundError(name)
    return dict(source_sha=source, user=user, current_jobs=queue, partition=partition,
        partition_limits=config, quota_output=quota, quota_headroom_bytes=int(headroom),
        filesystem_free_bytes=disk.free, min_headroom_bytes=MIN_STORAGE_BYTES,
        changed_files_sha256=changes, gpu_request='One independent5090GPU per job; at most C/D training overlap',
        allocation_32087_used=False, checked_utc=stamp())


def _identity(args):
    if not re.fullmatch(r'[0-9a-f]{40}', args.expected_sha):
        raise ValueError('Exact lowercase full40-character source SHA required')
    paths = {}
    for key in ('root', 'depth_root', 'g0_reference', 'common', 'c_output', 'd_output', 'reviewed_graphs'):
        raw = Path(getattr(args, key))
        if not raw.is_absolute() or raw.is_symlink() or any(v in str(raw) for v in ('\n', '\r', ',')):
            raise ValueError('Absolute nonsymlink output/data path required: '+key)
        paths[key] = str(raw.resolve())
    if Path(paths['common']).parent.name != COMMON_BUCKET:
        raise ValueError('Wrong independent comparison namespace')
    for key, variant in (('c_output', VARIANTS[0]), ('d_output', VARIANTS[1])):
        if Path(paths[key]).parent.name != BUCKETS[variant]:
            raise ValueError('Wrong independent C/D training namespace')
    values = [Path(paths[k]) for k in ('common', 'c_output', 'd_output')]
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(values) for b in values[i+1:]):
        raise ValueError('Comparison and variant runs must be disjoint')
    return dict(paths, source_sha=args.expected_sha, source=str(Path.cwd().resolve()),
        protocol='paired_h15_interval1_seed42_fresh8epochs_1514train100val', submission_version=1)


def validate_existing_engineering(path, source):
    """Only completed same-source CPU preflight/logs may precede submission."""
    root = Path(path); preflight = root/'preflight.json'
    if not root.is_dir() or root.is_symlink() or not preflight.is_file() or preflight.is_symlink():
        raise FileExistsError('Existing engineering directory lacks a completed canonical preflight')
    proof = json.loads(preflight.read_text())
    if proof.get('passed') is not True or proof.get('source_sha') != source:
        raise ValueError('Existing engineering preflight is failed or from another source')
    data = proof.get('data', {}); manifest = data.get('manifest', {}); orders = data.get('epoch_orders', [])
    if (manifest.get('train', {}).get('windows') != 1514 or manifest.get('val', {}).get('windows') != 100
            or len(manifest.get('train', {}).get('window_keys', [])) != 1514
            or len(manifest.get('val', {}).get('window_keys', [])) != 100 or len(orders) != 8
            or any(sorted(order) != list(range(1514)) for order in orders)):
        raise ValueError('Existing engineering preflight does not prove1514/100 and eight full window orders')
    hashes = [digest(order) for order in orders]
    if (data.get('manifest_hash') != digest(manifest) or data.get('epoch_order_hashes') != hashes
            or data.get('observed_epoch_order_hashes') != hashes):
        raise ValueError('Existing engineering preflight Manifest/window order hash changed')
    initial = proof.get('initialization', {}); model_hashes = initial.get('variant_model_hashes', {})
    exact = initial.get('h15_all_attributes_exact', {})
    if (initial.get('shared_exact') is not True or set(model_hashes) != set(VARIANTS)
            or len(set(model_hashes.values())) != 1
            or any(re.fullmatch('[0-9a-f]{64}', value or '') is None for value in model_hashes.values())
            or not exact or any(value is not True for value in exact.values())
            or proof.get('optimizer_created') is not False or proof.get('successful_updates') != 0):
        raise ValueError('Existing engineering preflight lacks exact fresh paired initialization proof')
    return proof


class Submission:
    def __init__(self, args):
        self.args = args; self.identity = _identity(args)
        self.common = Path(self.identity['common']); self.control = self.common/'submission'
        self.path = self.control/'state.json'
        self.control.mkdir(parents=True, exist_ok=True)
        (self.common/'logs').mkdir(exist_ok=True)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get('identity') != self.identity:
                raise ValueError('Pinned submission source/data/run identity changed')
        else:
            if any(Path(self.identity[k]).exists() for k in ('c_output', 'd_output')):
                raise FileExistsError('Fresh C/D run already exists; no implicit recovery or overwrite')
            existing = {p.name for p in self.common.iterdir()}
            if 'engineering' in existing:
                validate_existing_engineering(self.common/'engineering', args.expected_sha)
            if existing-{'submission', 'logs', 'engineering'}:
                raise FileExistsError('Unregistered comparison results already exist')
            self.state = dict(identity=self.identity, created_utc=stamp(), token=uuid.uuid4().hex[:12], jobs={})
            self.save()

    def save(self):
        self.state['updated_utc'] = stamp(); atomic_json(self.path, self.state)

    def environment(self, phase):
        value = dict(SOURCE=self.identity['source'], EXPECTED_SHA=self.identity['source_sha'],
            DATA_ROOT=self.identity['root'], DEPTH_ROOT=self.identity['depth_root'],
            G0_REFERENCE=self.identity['g0_reference'], COMMON=self.identity['common'],
            REVIEWED_GRAPHS=self.identity['reviewed_graphs'],
            C_OUTPUT=self.identity['c_output'], D_OUTPUT=self.identity['d_output'],
            EVALUATION_OUTPUT=str(self.common/'acceptance'), RESUME='0')
        if phase in VARIANTS:
            value.update(VARIANT=phase, RUN_OUTPUT=self.identity['c_output' if phase==VARIANTS[0] else 'd_output'])
        return value

    def find_accepted(self, record):
        """Exact identity only; missing/duplicate accounting is never a retry cue."""
        candidates = {}
        active = _command(['scontrol', 'show', 'jobs', '-o'])
        for line in active.splitlines():
            row = _fields(line)
            if row.get('JobName') == record['job_name'] and row.get('Comment') == record['comment']:
                if not row.get('UserId', '').startswith('tcweng('):
                    raise RuntimeError('Matched job has unexpected owner')
                candidates[row['JobId']] = row.get('JobState', 'UNKNOWN')
        history = _command(['sacct', '-X', '--noheader', '--parsable2', '--name='+record['job_name'],
            '--starttime='+self.state['created_utc'][:10], '--format=JobIDRaw,JobName%100,Comment%200,State%40,User%32'])
        for line in history.splitlines():
            fields = [v.strip() for v in line.split('|')]
            if len(fields) < 5:
                continue
            job, name, comment, state, user = fields[:5]
            if name == record['job_name'] and comment == record['comment']:
                if user != 'tcweng' or not job.isdigit():
                    raise RuntimeError('Matched accounting identity is invalid')
                candidates[job] = state
        if len(candidates) != 1:
            raise RuntimeError('Unknown submission result has no unique exact Slurm/accounting match; do not repeat sbatch')
        return next(iter(candidates.items()))

    def submit_phase(self, phase, dependencies=()):
        source_guard(self.args.expected_sha)
        existing = self.state['jobs'].get(phase)
        if existing:
            slug = {'prepare':'prep', 'g5_arap':'c', 'g5_arap_iso':'d', 'evaluate':'eval'}[phase]
            expected_name = 't4-arapiso-'+self.state['token']+'-'+slug
            expected_comment = 'touch4d-arapiso:'+self.state['token']+':'+phase+':'+self.args.expected_sha
            if (existing.get('job_name') != expected_name or existing.get('comment') != expected_comment
                    or existing.get('dependencies') != list(dependencies) or existing.get('export') != self.environment(phase)):
                raise ValueError('Stored job phase/dependency/source identity changed')
            if existing.get('job_id'):
                if not str(existing['job_id']).isdigit():
                    raise ValueError('Invalid stored Slurm job ID')
                return existing['job_id']
            job, observed = self.find_accepted(existing)
            existing.update(job_id=job, status='RECONCILED', observed_state=observed); self.save()
            return job
        slug = {'prepare':'prep', 'g5_arap':'c', 'g5_arap_iso':'d', 'evaluate':'eval'}[phase]
        job_name = 't4-arapiso-'+self.state['token']+'-'+slug
        comment = 'touch4d-arapiso:'+self.state['token']+':'+phase+':'+self.args.expected_sha
        module = Path(__file__).parent
        script = module/({'prepare':'prepare_smoke', 'evaluate':'evaluate'}.get(phase, 'train')+'.sbatch')
        command = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes', '--job-name='+job_name,
            '--comment='+comment, '--chdir='+self.identity['source'],
            '--output='+str(self.common/'logs'/(phase+'-%j.log')), '--export=ALL']
        if dependencies:
            if any(not str(v).isdigit() for v in dependencies):
                raise ValueError('Dependency must be an exact Slurm job ID')
            command.append('--dependency=afterok:'+':'.join(map(str, dependencies)))
        command.append(str(script.resolve()))
        env = self.environment(phase)
        record = dict(job_name=job_name, comment=comment, command=command, export=env,
            dependencies=list(dependencies), status='SUBMITTING', attempted_utc=stamp())
        self.state['jobs'][phase] = record; self.save()  # Intent BEFORE any external submission.
        try:
            raw = subprocess.check_output(command, env=dict(os.environ, **env), text=True, stderr=subprocess.STDOUT).strip()
            match = re.fullmatch(r'(\d+)(?:;[A-Za-z0-9_.-]+)?', raw)
            if not match:
                raise RuntimeError('Unexpected sbatch reply: '+raw)
        except BaseException as error:
            diagnostic = getattr(error, 'output', None)
            if isinstance(diagnostic, bytes):
                diagnostic = diagnostic.decode(errors='replace')
            record.update(status='SUBMISSION_UNKNOWN', error=str(error), error_type=type(error).__name__,
                stdout_stderr=diagnostic); self.save()
            raise
        record.update(job_id=match[1], status='SUBMITTED', accepted_utc=stamp()); self.save()
        return record['job_id']

    def submit_chain(self):
        preparation = self.submit_phase('prepare')
        c = self.submit_phase(VARIANTS[0], (preparation,))
        d = self.submit_phase(VARIANTS[1], (preparation,))
        evaluation = self.submit_phase('evaluate', (c, d))
        self.state.update(status='DEPENDENCY_CHAIN_SUBMITTED', job_ids=dict(prepare=preparation, c=c, d=d, evaluate=evaluation))
        self.save(); return self.state['job_ids']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'depth-root', 'g0-reference', 'common', 'c-output', 'd-output', 'reviewed-graphs', 'expected-sha'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    proof = precheck(args)
    control = Path(_identity(args)['common'])/'submission'; control.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (control/'execution.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        submission = Submission(args)
        atomic_json(submission.control/'precheck.json', proof)
        ids = submission.submit_chain()
    print(json.dumps(dict(job_ids=ids, source_sha=args.expected_sha, state_path=str(submission.path)), indent=2))


if __name__ == '__main__':
    main()
