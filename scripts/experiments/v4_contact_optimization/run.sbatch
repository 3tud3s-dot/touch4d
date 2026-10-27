#!/bin/bash
#SBATCH --job-name=v4-contact-opt
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${EXPECTED_SHA:?}" "${OUTPUT_ROOT:?}" "${STAGE:?}"
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
SHARED=/data1/userdata/tcweng/projects/touch4d
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$SHARED/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$SHARED/.cache/torch_extensions" TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ID_ARGS=()
if [[ "$STAGE" == short ]]; then ID_ARGS=(--id "S${SLURM_ARRAY_TASK_ID:?}"); fi
if [[ "$STAGE" == long ]]; then ID_ARGS=(--id "${RUN_ID:?}"); fi
exec "$SHARED/.venv/bin/python" -B -m scripts.experiments.v4_contact_optimization.run --root "$OUTPUT_ROOT" --stage "$STAGE" --expected-sha "$EXPECTED_SHA" "${ID_ARGS[@]}"
