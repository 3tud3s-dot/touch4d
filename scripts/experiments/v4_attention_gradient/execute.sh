#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm required}" "${SOURCE:?}" "${EXPECTED_SHA:?}" "${OUTPUT_ROOT:?}"
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MAX_JOBS=4 PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0
export PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/data1/userdata/tcweng/.cache/touch4d-identifiability-extensions
export XDG_CACHE_HOME="$OUTPUT_ROOT/cache"
mkdir -p "$XDG_CACHE_HOME/torch/kernels"
PYTHON=/data1/userdata/tcweng/projects/touch4d/.venv/bin/python
"$PYTHON" -B -m scripts.experiments.v4_attention_gradient.run --root "$OUTPUT_ROOT" --expected-sha "$EXPECTED_SHA" --stage pilot > "$OUTPUT_ROOT/logs/cuda_pilot.txt" 2>&1
"$PYTHON" -B -m scripts.experiments.v4_attention_gradient.run --root "$OUTPUT_ROOT" --expected-sha "$EXPECTED_SHA" --stage full > "$OUTPUT_ROOT/logs/cuda_full.txt" 2>&1
"$PYTHON" -B -m scripts.experiments.v4_attention_gradient.run --root "$OUTPUT_ROOT" --expected-sha "$EXPECTED_SHA" --stage report > "$OUTPUT_ROOT/logs/report.txt" 2>&1
