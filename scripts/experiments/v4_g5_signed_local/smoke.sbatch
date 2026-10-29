#!/bin/bash
#SBATCH --job-name=g5-bounded-smoke
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${EXPECTED_SHA:?}" "${RUN_OUTPUT:?}" "${G4_CHECKPOINT:?}" "${SLURM_JOB_ID:?}"
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
test ! -e "$RUN_OUTPUT"
SHARED=/data1/userdata/tcweng/projects/touch4d
test -x "$SHARED/.venv/bin/python"
test -d "$SHARED/datasets/touch4d-v1"
test -d "$SHARED/datasets/touch4d-depth-v1"
test "$(sha256sum "$G4_CHECKPOINT" | cut -d' ' -f1)" = 240045e3ccb0a37bfcc954feabb2b61f4a1f828d63566eb856619ebe6335b92d
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$SHARED/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$SHARED/.cache/torch_extensions"
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
test ! -e "${RUN_OUTPUT}.cpu-tests.log"
echo "Server CPU regression (CUDA hidden); no real GPU updates yet"
CUDA_VISIBLE_DEVICES='' "$SHARED/.venv/bin/python" -B -m unittest discover -s scripts/tests -t . -v > "${RUN_OUTPUT}.cpu-tests.log" 2>&1
echo "Server CPU regression passed; begin real GPU smoke"
exec "$SHARED/.venv/bin/python" -B -m scripts.experiments.v4_g5_signed_local.smoke \
 --variant all --expected-sha "$EXPECTED_SHA" --g4-checkpoint "$G4_CHECKPOINT" \
 --root "$SHARED/datasets/touch4d-v1" --depth-root "$SHARED/datasets/touch4d-depth-v1" --output "$RUN_OUTPUT"
