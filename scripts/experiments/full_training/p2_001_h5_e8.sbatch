#!/bin/bash
#SBATCH --partition=5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=p2-001-h5-e8
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/data1/userdata/tcweng/projects/touch4d-worktrees/p2-001-h5-e8
#SBATCH --output=/data1/userdata/tcweng/projects/touch4d/audit/p2-001-h5-e8-submission-v1/slurm-%j.out
#SBATCH --error=/data1/userdata/tcweng/projects/touch4d/audit/p2-001-h5-e8-submission-v1/slurm-%j.err

# Submit once from the synchronized checkout, after creating the log directory:
# sbatch --parsable scripts/experiments/full_training/p2_001_h5_e8.sbatch "$(git rev-parse HEAD)"
# Fresh epoch training only. No resume, fixed-window subset, test evaluation or retries.
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9a-f]{40}$ ]]; then
    printf '%s\n' 'Expected one argument: the reviewed, synchronized commit SHA.' >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
readonly EXPECTED_SHA="$1"
readonly PROJECT=/data1/userdata/tcweng/projects/touch4d-worktrees/p2-001-h5-e8
readonly SHARED=/data1/userdata/tcweng/projects/touch4d
readonly PYTHON="$SHARED/.venv/bin/python"
readonly RUN_ID="p2-001-h5-e8-${SLURM_JOB_ID}"
readonly RUN_DIR="$PROJECT/runs/001-rope/$RUN_ID"

cd "$PROJECT"
test "$(git rev-parse --show-toplevel)" = "$PROJECT"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
# Keep this execution checkout detached so H15 retains its checked-out branch.
test -z "$(git branch --show-current)"
test -z "$(git status --porcelain)"
test ! -e "$RUN_DIR"
test -x "$PYTHON"
test -d "$SHARED/datasets/touch4d-v1/001-rope"
test -d "$SHARED/datasets/touch4d-depth-v1/001-rope"

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$SHARED/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$SHARED/.cache/torch_extensions"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

printf 'Source SHA: %s\nRun ID: %s\nOutput: %s\n' "$EXPECTED_SHA" "$RUN_ID" "$RUN_DIR"
date -Is
"$PYTHON" -B - <<'PY'
import sys
import gsplat
import h5py
import torch

assert sys.version_info[:3] == (3, 10, 20), sys.version
assert torch.cuda.is_available(), "CUDA is required"
assert torch.cuda.device_count() == 1, "Exactly one allocated GPU is required"
assert "5090" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print({"python": sys.version, "torch": torch.__version__, "gsplat": gsplat.__version__,
       "h5py": h5py.__version__, "gpu": torch.cuda.get_device_name(0)}, flush=True)
PY

# Normal train episodes 1,2,4,5,7,8; validation 3,6. All eight epochs use stage=all.
# The three audited cameras are explicit; depth validity still masks RGB/alpha.
exec "$PYTHON" -B -m scripts.training.train \
    --scene 001-rope \
    --protocol epochs \
    --run-id "$RUN_ID" \
    --root "$SHARED/datasets/touch4d-v1" \
    --depth-root "$SHARED/datasets/touch4d-depth-v1" \
    --depth-cameras brics-odroid-001_cam0 brics-odroid-011_cam0 brics-odroid-021_cam0 \
    --epochs 8 \
    --single-epochs 0 \
    --horizon 5 \
    --gs-decoder anchors \
    --gs-structure anchor_attention \
    --gs-local-attention knn \
    --gs-local-k 16 \
    --anchor-count 128 \
    --anchor-pool-k 16 \
    --anchor-bandwidth 1 \
    --controller-features 7 \
    --prediction-mode autoregressive \
    --output-mode gs_geometry \
    --chamfer-weight 1 \
    --rgb-weight 1 \
    --alpha-weight 0.1 \
    --scale-weight 0.01 \
    --depth-weight 0 \
    --batch-size 1 \
    --seed 42 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --width 128 \
    --heads 4 \
    --layers 2 \
    --workers 0 \
    --chamfer-chunk 512 \
    --overfit-windows 0 \
    --max-train-batches 0 \
    --max-val-batches 0 \
    --device cuda
