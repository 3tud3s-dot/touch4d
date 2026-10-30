#!/bin/bash
# Run ONLY as a Slurm step in the allocation explicitly authorized by the user.
set -euo pipefail
: "$SOURCE" "$EXPECTED_SHA" "$RUN_OUTPUT" "$AUTHORIZED_ALLOCATION" "$SLURM_JOB_ID"
test "$SLURM_JOB_ID" = "$AUTHORIZED_ALLOCATION"
SHARED=/data1/userdata/tcweng/projects/touch4d
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
test ! -e "$RUN_OUTPUT"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$SHARED/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$SHARED/.cache/torch_extensions"
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec "$SHARED/.venv/bin/python" -B -m scripts.experiments.v4_g5_condition_attribution.run \
 --root "$SHARED/datasets/touch4d-v1" --depth-root "$SHARED/datasets/touch4d-depth-v1" \
 --reference /data1/userdata/tcweng/experiments/touch4d/v4-001-h15-e8/run-30995 \
 --checkpoint /data1/userdata/tcweng/experiments/touch4d/v4-g5-direct-v1-training/run-001/best.pt \
 --output "$RUN_OUTPUT" --expected-sha "$EXPECTED_SHA" --allocation-id "$AUTHORIZED_ALLOCATION"
