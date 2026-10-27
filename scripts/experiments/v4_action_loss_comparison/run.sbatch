#!/bin/bash
#SBATCH --job-name=v4-action-cd-mse
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${EXPECTED_SHA:?}" "${OUTPUT:?}" "${SLURM_JOB_ID:?}"
cd "$SOURCE"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec /data1/userdata/tcweng/projects/touch4d/.venv/bin/python -B -m scripts.experiments.v4_action_loss_comparison.run \
 --v1 /data1/userdata/tcweng/experiments/touch4d/v4-action-identifiability-v1/run-20261027-01 \
 --output "$OUTPUT" --expected-sha "$EXPECTED_SHA"
