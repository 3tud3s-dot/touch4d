#!/bin/bash
#SBATCH --job-name=g5-geo-head-e8
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${EXPECTED_SHA:?}" "${RUN_OUTPUT:?}" "${SMOKE_EVIDENCE:?}" "${SLURM_JOB_ID:?}"
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
extra=()
case "${RESUME:-0}" in
0) test ! -e "$RUN_OUTPUT";;
1) test -f "$RUN_OUTPUT/initial.pt"; extra=(--resume);;
*) exit 2;;
esac
SHARED=/data1/userdata/tcweng/projects/touch4d
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$SHARED/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$SHARED/.cache/torch_extensions"
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"$SHARED/.venv/bin/python" -B -m scripts.experiments.v4_g5_geometry.train \
 --root "$SHARED/datasets/touch4d-v1" --depth-root "$SHARED/datasets/touch4d-depth-v1" \
 --g0-reference /data1/userdata/tcweng/experiments/touch4d/v4-001-h15-e8/run-30995 \
 --checkpoint /data1/userdata/tcweng/experiments/touch4d/v4-g5-direct-v1-training/run-001/best.pt \
 --output "$RUN_OUTPUT" --expected-sha "$EXPECTED_SHA" --smoke-evidence "$SMOKE_EVIDENCE" "${extra[@]}"
exec "$SHARED/.venv/bin/python" -B -m scripts.experiments.v4_g5_geometry.evaluate \
 --root "$SHARED/datasets/touch4d-v1" --depth-root "$SHARED/datasets/touch4d-depth-v1" \
 --g0-reference /data1/userdata/tcweng/experiments/touch4d/v4-001-h15-e8/run-30995 \
 --checkpoint /data1/userdata/tcweng/experiments/touch4d/v4-g5-direct-v1-training/run-001/best.pt \
 --run "$RUN_OUTPUT" --output "${ACCEPTANCE_OUTPUT:-$RUN_OUTPUT/acceptance}" --expected-sha "$EXPECTED_SHA"
