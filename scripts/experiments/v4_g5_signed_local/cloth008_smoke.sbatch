#!/bin/bash
#SBATCH --job-name=g5-008-h15-smoke
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
run_root="$1"
shared=/data1/userdata/tcweng/projects/touch4d
base=/data1/userdata/tcweng/projects/touch4d-worktrees/v4-g5-runner-integration-smoke-v1
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$shared/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$shared/.cache/torch_extensions"
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$run_root/source"
sha256sum --check --strict source.sha256
source_id=$(sha256sum source.sha256 | cut -d ' ' -f 1)
cd "$base"
test "$(git rev-parse HEAD)" = f978e0a29b6f0f0b6097c40684b949797a2f5d7b
test -z "$(git status --porcelain)"
srun --exclusive --ntasks=1 "$shared/.venv/bin/python" -B - "$run_root" "$source_id" "$shared" <<'PY'
from pathlib import Path
import importlib.abc
import importlib.util
import sys

run, source_id, shared = sys.argv[1:]
files = {
    "scripts.dataloader.dataset": "dataset.py",
    "scripts.data_process.processor": "processor.py",
    "scripts.experiments.v4_g5_signed_local.cloth008": "cloth008.py",
    "scripts.experiments.v4_g5_signed_local.cloth008_train": "cloth008_train.py",
}

class RunSources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in files:
            return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(Path(run) / "source" / files[module.__name__])
        source = Path(module.__file__).read_bytes()
        exec(compile(source, module.__file__, "exec"), module.__dict__)

sys.meta_path.insert(0, RunSources())
from scripts.experiments.v4_g5_signed_local.cloth008_train import main

sys.argv = [
    "cloth008_train", "--mode", "smoke", "--source-id", source_id,
    "--root", shared + "/datasets/touch4d-v1",
    "--depth-root", shared + "/datasets/touch4d-depth-v1",
    "--output", run + "/result",
]
main()
PY
