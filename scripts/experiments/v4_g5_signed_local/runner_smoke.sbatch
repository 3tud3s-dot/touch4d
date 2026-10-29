#!/bin/bash
#SBATCH --job-name=g5-runner-smoke
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${EXPECTED_SHA:?}" "${SMOKE_OUTPUT:?}" "${SLURM_JOB_ID:?}"
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
test ! -e "$SMOKE_OUTPUT"
SHARED=/data1/userdata/tcweng/projects/touch4d
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$SHARED/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$SHARED/.cache/torch_extensions"
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec "$SHARED/.venv/bin/python" -B -m scripts.experiments.v4_g5_signed_local.runner_smoke \
 --root "$SHARED/datasets/touch4d-v1" --depth-root "$SHARED/datasets/touch4d-depth-v1" \
 --g0-reference /data1/userdata/tcweng/experiments/touch4d/v4-001-h15-e8/run-30995 \
 --output "$SMOKE_OUTPUT" --expected-sha "$EXPECTED_SHA"
