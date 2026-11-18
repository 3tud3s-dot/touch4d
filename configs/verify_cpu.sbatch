#!/usr/bin/env bash
#SBATCH --job-name=touch4d-core-verify
#SBATCH --partition=5090
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
source_root=$1
output_root=$2
oracle_root=$3
frozen_overrides=$4
checkpoint=$5
data_root=$6
depth_root=$7
raw_root=$8
python_bin=$9
test -n "${SLURM_JOB_ID:-}"
test ! -e "$output_root"
mkdir -p "$output_root"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$source_root"
"$python_bin" -B -m unittest scripts.tests.test_core_data scripts.tests.test_core_data_depth -v > "$output_root/data_tests.log" 2>&1
"$python_bin" -B scripts/rebuild_dataset/validate.py --scene 008-pink-cloth --episodes 0,1 --source-root "$raw_root" --data-root "$data_root" > "$output_root/v1_validation.jsonl" 2> "$output_root/v1_validation.stderr"
"$python_bin" -B -m scripts.rebuild_dataset.depth --mode validate --scene 008-pink-cloth --episodes 0,1 --source-root "$raw_root" --core-root "$data_root" --depth-root "$depth_root" > "$output_root/depth_validation.jsonl" 2> "$output_root/depth_validation.stderr"
"$python_bin" -B scripts/tests/core_server_audit.py --oracle-source "$oracle_root" --frozen-overrides "$frozen_overrides" --checkpoint "$checkpoint" --root "$data_root" --depth-root "$depth_root" --output "$output_root/model_regression.json" > "$output_root/model_regression.log" 2>&1
printf 'PASS: CPU-only data/full-source/depth/checkpoint/real-window regression\n'
