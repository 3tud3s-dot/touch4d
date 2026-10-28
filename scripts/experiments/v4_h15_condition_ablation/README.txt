Frozen H15 condition ablation, no training

Run from the repository root. Full commands and execution SHA are also in result provenance/status.
Shared server interpreter: /data1/userdata/tcweng/projects/touch4d/.venv/bin/python

CPU tests:
python -B -m unittest scripts.tests.training.test_v4_h15_condition_ablation -v

Use a NEW output directory; do not overwrite an attempted stage:
python -B -m scripts.experiments.v4_h15_condition_ablation.run --stage preflight --root OUTPUT --expected-sha SHA

Inside a valid single-GPU Slurm allocation, with clean frozen checkout:
SOURCE=WORKTREE OUTPUT_ROOT=OUTPUT EXPECTED_SHA=SHA bash scripts/experiments/v4_h15_condition_ablation/execute.sh

After copying the complete results locally (CPU matplotlib available):
python -B -m scripts.experiments.v4_h15_condition_ablation.report --root OUTPUT
python -B -m scripts.experiments.v4_h15_condition_ablation.package --root OUTPUT --zip OUTPUT_PARENT/v4-h15-condition-ablation-v1-review.zip

Open report.html directly offline; no web server/CDN needed. Graphs and renders are relative paths.
Full CSV and JSON contain all 15 horizons; figures show H1/H5/H10/H15, three calibrated cameras.
Raw tensor bundles are gzip-compressed torch.save containers. Only load trusted local files:
with gzip.open(path, 'rb') as f: data = torch.load(f, map_location='cpu', weights_only=False)
geometry.npz is portable NumPy world-coordinate evidence.

Intervention functions are AST-identical to audited commit 0934d8c core.py.
Freeze Action intersects current masks with initial validity; no tactile mutation.
Zero 12D only replaces motion-input columns 128:140; it does not freeze subsequent feedback.
Near/Far are initial nearest-controller distance quartiles, not physical-contact labels.
Local error is directed prediction-subset to complete GT, not region-wise symmetric CD.
No optimizer, backward, or parameter update is performed by the inference tools.
Checkpoint file hash, model-state hash and input hash are checked before/after.
Normal must match historical frozen checkpoint metrics and uninstrumented output.

Packaging includes complete tracked source and all numeric/input/output evidence.
Historical checkpoint weights stay at their recorded server location with SHA256 provenance.
Every ZIP member is decompressed and SHA256-checked; file_manifest.json is inside,
ZIP digest and verification receipt are beside the ZIP to avoid self-referential hashes.
