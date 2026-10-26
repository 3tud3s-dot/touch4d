#!/bin/bash
#SBATCH --partition=5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=p2-cond-screen
#SBATCH --no-requeue
set -euo pipefail
# Explicit SHA, mode, manifest and unique output directory. No resume/retry.
sha="$1"; mode="$2"; manifest="$3"; output="$4"
command="${5:-run}"
case "$command" in run|smoke) ;; *) exit 2 ;; esac
test "$(git rev-parse HEAD)" = "$sha"
test -z "$(git status --porcelain --untracked-files=no)"
test ! -e "$output"
test -n "${SLURM_JOB_ID:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
project=/data1/userdata/tcweng/projects/touch4d
export TORCH_EXTENSIONS_DIR="$project/.cache/torch_extensions"
export MAX_JOBS=4
exec "$project/.venv/bin/python" -B -m scripts.experiments.condition_encoder.screen "$command" \
  --root "$project/datasets/touch4d-v1" --mode "$mode" --expected-sha "$sha" \
  --manifest "$manifest" --output "$output"
