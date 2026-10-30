"""Same G5 H15 data/Loss/order protocol; independent joint architecture/run."""
import subprocess
from pathlib import Path
from scripts.experiments.v4_g5_signed_local.protocol import configuration as g5_configuration, preflight as g5_preflight
from scripts.experiments.v4_g5_signed_local.diagnostics import sha256
from .model import ARCHITECTURE

BASE = '5d0e632c75a3742b2a9c56f298edbb01200788d2'
BUCKET = 'v4-g5-geo-joint-scratch-v1-training'


def configuration(root, depth_root, source):
    return dict(g5_configuration(root, depth_root, source, 'direct'), architecture=ARCHITECTURE,
                training_mode='joint_scratch_geometry_output_only')


def common_configuration(c):
    if c != configuration(c['root'], c['depth_root'], c['source_sha']):
        raise ValueError('Frozen joint protocol differs')
    return g5_configuration(c['root'], c['depth_root'], c['source_sha'], 'direct')


def preflight(c, reference):
    # Reuse actual manifest, split and eight DataLoader shuffle-order probes.
    return g5_preflight(common_configuration(c), reference)


def source_guard(expected):
    source = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    if source != expected or subprocess.check_output(['git','status','--porcelain'], text=True).strip():
        raise ValueError('Committed clean pinned source required')
    changes = subprocess.check_output(['git','diff','--name-only',BASE,'HEAD'], text=True).splitlines()
    if any(not (p.startswith('scripts/experiments/v4_g5_geometry_joint/') or
                p == 'scripts/tests/training/test_v4_g5_geometry_joint.py') for p in changes):
        raise ValueError('Original G5/GEO/production code changed: '+str(changes))
    return source, {p:sha256(p) for p in changes}


def output_check(path, *, resume=False):
    raw = Path(path)
    root = raw.resolve()
    if not raw.is_absolute() or raw.is_symlink() or root.parent.name != BUCKET:
        raise ValueError('Independent absolute joint training namespace required')
    if resume:
        if not (root/'initial.pt').is_file():
            raise ValueError('No durable initial boundary')
    elif root.exists():
        raise FileExistsError(root)
    return root
