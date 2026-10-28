#!/bin/bash
#SBATCH --job-name=g4-001-h15-e8
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${EXPECTED_SHA:?}" "${RUN_OUTPUT:?}" "${G0_REFERENCE:?}" "${SLURM_JOB_ID:?}"
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
readonly SHARED=/data1/userdata/tcweng/projects/touch4d
test -x "$SHARED/.venv/bin/python"
test -d "$SHARED/datasets/touch4d-v1"
test -d "$SHARED/datasets/touch4d-depth-v1"
test -f "$G0_REFERENCE/config.json"
test -f "$G0_REFERENCE/manifest.json"
extra=()
if [[ "${RESUME:-0}" == 1 ]]; then
 test -f "$RUN_OUTPUT/initial.pt"; extra=(--resume)
elif [[ "${RESUME:-0}" == 0 ]]; then
 test ! -e "$RUN_OUTPUT"
else
 echo 'RESUME must be 0 or 1' >&2; exit 1
fi
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$SHARED/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$SHARED/.cache/torch_extensions"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec "$SHARED/.venv/bin/python" -B -m scripts.experiments.v4_g4_global_motion.train \
 --root "$SHARED/datasets/touch4d-v1" --depth-root "$SHARED/datasets/touch4d-depth-v1" \
 --g0-reference "$G0_REFERENCE" --output "$RUN_OUTPUT" --expected-sha "$EXPECTED_SHA" "${extra[@]}"
