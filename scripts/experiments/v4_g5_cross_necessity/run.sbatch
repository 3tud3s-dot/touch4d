#!/bin/bash
#SBATCH --job-name=g5-cross-frozen
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
: "$SOURCE" "$EXPECTED_SHA" "$RUN_OUTPUT" "$SLURM_JOB_ID"
test -n "$SOURCE"
test -n "$EXPECTED_SHA"
test -n "$RUN_OUTPUT"
SHARED=/data1/userdata/tcweng/projects/touch4d
REFERENCE=/data1/userdata/tcweng/experiments/touch4d/v4-001-h15-e8/run-30995
CHECKPOINT=/data1/userdata/tcweng/experiments/touch4d/v4-g5-direct-v1-training/run-001/best.pt
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
test ! -e "$RUN_OUTPUT"
test -x "$SHARED/.venv/bin/python"
test -f "$CHECKPOINT"
test "$(dirname "$RUN_OUTPUT" | xargs basename)" = v4-g5-cross-necessity-v1
test "$(sha256sum "$CHECKPOINT" | cut -d " " -f 1)" = 067984c4c286306094efeff2f28ed57e3dcdd0c333972ecc6feb4b982ee3a40c
test "${SLURM_GPUS_ON_NODE:-0}" = 1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm must assign the GPU}"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$SHARED/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$SHARED/.cache/torch_extensions"
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"$SHARED/.venv/bin/python" -B -m scripts.experiments.v4_g5_cross_necessity.run \
 --root "$SHARED/datasets/touch4d-v1" --depth-root "$SHARED/datasets/touch4d-depth-v1" \
 --reference "$REFERENCE" --checkpoint "$CHECKPOINT" --output "$RUN_OUTPUT" --expected-sha "$EXPECTED_SHA"
echo "Inference complete. Build plots/report with the existing local CPU matplotlib environment; never launch Stage 2 from this script."
