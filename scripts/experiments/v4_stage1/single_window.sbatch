#!/bin/bash
#SBATCH --job-name=v4-h10-single400
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${EXPECTED_SHA:?}" "${OUTPUT:?}"
test "${SLURM_JOB_ID:?}" != 30903
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MAX_JOBS=4 PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0
export PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/data1/userdata/tcweng/.cache/touch4d-identifiability-extensions
export XDG_CACHE_HOME="$OUTPUT-cache"
mkdir -p "$XDG_CACHE_HOME/torch/kernels"
PYTHON=/data1/userdata/tcweng/projects/touch4d/.venv/bin/python
ARGS=(--expected-sha "$EXPECTED_SHA" --output "$OUTPUT"
 --root /data1/userdata/tcweng/projects/touch4d/datasets/touch4d-v1
 --depth-root /data1/userdata/tcweng/projects/touch4d/datasets/touch4d-depth-v1)
if [[ "${RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi
"$PYTHON" -B -m scripts.experiments.v4_stage1.single_window "${ARGS[@]}"
