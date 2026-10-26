#!/bin/bash
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${EXPECTED_SHA:?}" "${MANIFEST:?}" "${RUN_ROOT:?}" "${CONFIGS:?}"
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MAX_JOBS=4 PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/data1/userdata/tcweng/.cache/touch4d-p3p4-extensions
export XDG_CACHE_HOME=/data1/userdata/tcweng/.cache/touch4d-p3p4
PY=/data1/userdata/tcweng/projects/touch4d/.venv/bin/python
# Config is a JSON list; shlex quoting is supplied by Python, never eval shell input.
"$PY" -B -m scripts.experiments.spatial_dynamics.dispatch --configs "$CONFIGS" --index "${SLURM_ARRAY_TASK_ID:-0}" --source "$SOURCE" --sha "$EXPECTED_SHA" --manifest "$MANIFEST" --run-root "$RUN_ROOT"
