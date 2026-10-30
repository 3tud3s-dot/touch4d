#!/bin/bash
# Run ONLY inside an independent step of the existing authorized Allocation32087.
# There are deliberately no sbatch directives or new GPU allocation commands.
set -euo pipefail
: "${SOURCE:?}" "${EXPECTED_SHA:?}" "${COMMON:?}" "${A_OUTPUT:?}" "${B_OUTPUT:?}"
test "${SLURM_JOB_ID:?}" = 32087
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
SHARED=/data1/userdata/tcweng/projects/touch4d
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$SHARED/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$SHARED/.cache/torch_extensions"
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec "$SHARED/.venv/bin/python" -B -m scripts.experiments.v4_g5_struct_state.launcher \
 --root "$SHARED/datasets/touch4d-v1" --depth-root "$SHARED/datasets/touch4d-depth-v1" \
 --g0-reference /data1/userdata/tcweng/experiments/touch4d/v4-001-h15-e8/run-30995 \
 --common "$COMMON" --a-output "$A_OUTPUT" --b-output "$B_OUTPUT" --expected-sha "$EXPECTED_SHA"
