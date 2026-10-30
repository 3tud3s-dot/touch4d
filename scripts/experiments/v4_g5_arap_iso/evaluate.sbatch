#!/bin/bash
#SBATCH --job-name=g5-arap-evaluate
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${EXPECTED_SHA:?}" "${DATA_ROOT:?}" "${DEPTH_ROOT:?}" "${G0_REFERENCE:?}" "${COMMON:?}" "${C_OUTPUT:?}" "${D_OUTPUT:?}" "${EVALUATION_OUTPUT:?}" "${SLURM_JOB_ID:?}"
test "$SLURM_JOB_ID" != 32087
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
SHARED=/data1/userdata/tcweng/projects/touch4d
test -x "$SHARED/.venv/bin/python"
test -f "$COMMON/gate.json"
test -f "$C_OUTPUT/completion.json"
test -f "$D_OUTPUT/completion.json"
test ! -e "$EVALUATION_OUTPUT"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$SHARED/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$SHARED/.cache/torch_extensions"
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec srun --ntasks=1 --cpus-per-task=4 --kill-on-bad-exit=1 "$SHARED/.venv/bin/python" -B -m scripts.experiments.v4_g5_arap_iso.evaluate \
 --root "$DATA_ROOT" --depth-root "$DEPTH_ROOT" --g0-reference "$G0_REFERENCE" --common "$COMMON" \
 --c-run "$C_OUTPUT" --d-run "$D_OUTPUT" --calibration "$COMMON/calibration/calibration.json" \
 --output "$EVALUATION_OUTPUT" --expected-sha "$EXPECTED_SHA"
