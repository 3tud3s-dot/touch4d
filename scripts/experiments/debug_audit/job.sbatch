#!/bin/bash
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${EXPECTED_SHA:?}" "${OUTPUT:?}" "${COMMAND:?}"
test "${SLURM_JOB_ID:?}" != 30903
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MAX_JOBS=4 PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/data1/userdata/tcweng/.cache/touch4d-debug-audit-extensions
export XDG_CACHE_HOME=/data1/userdata/tcweng/.cache/touch4d-debug-audit
PY=/data1/userdata/tcweng/projects/touch4d/.venv/bin/python
ARGS=("$COMMAND" --root /data1/userdata/tcweng/projects/touch4d/datasets/touch4d-v1 --output "$OUTPUT" --expected-sha "$EXPECTED_SHA")
if [[ "$COMMAND" == fit ]]; then ARGS+=(--mode "${MODE:?}"); fi
"$PY" -B -m scripts.experiments.debug_audit.runner "${ARGS[@]}"
