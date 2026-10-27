#!/bin/bash
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${EXPECTED_SHA:?}" "${OUTPUT:?}" "${MODE:?}" "${AUDIT:?}"
test "${SLURM_JOB_ID:?}" != 30903
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MAX_JOBS=4 PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0
export PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/data1/userdata/tcweng/.cache/touch4d-identifiability-extensions
export XDG_CACHE_HOME=/data1/userdata/tcweng/.cache/touch4d-identifiability
mkdir -p "$XDG_CACHE_HOME/torch/kernels"
ARGS=(--mode "$MODE" --root /data1/userdata/tcweng/projects/touch4d/datasets/touch4d-v1 --audit "$AUDIT" --output "$OUTPUT" --expected-sha "$EXPECTED_SHA")
if [[ "$MODE" != smoke ]]; then ARGS+=(--smoke "${SMOKE:?}"); fi
/data1/userdata/tcweng/projects/touch4d/.venv/bin/python -B -m scripts.experiments.condition_identifiability.run "${ARGS[@]}"
