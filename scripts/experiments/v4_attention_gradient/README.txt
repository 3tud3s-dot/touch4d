V4 frozen-weight Attention / Loss gradient audit

Base: d81d04544ccfd68f548ed3406342c6bb11e95bf9.
Checkpoint training source: 0baaedf5fb62ef8c9b2fb4cc9c3cc7ed7fbe2a26.
No production source changes; no optimizer creation, step, gradient clipping,
parameter update, new training, test-set evaluation, or architecture intervention.
Model eval() preserves the frozen numerical state while parameters require grad.

Stages (from repository root, existing audited Python environment):
python -B -m unittest scripts.tests.training.test_v4_attention_gradient -v
python -B -m scripts.experiments.v4_attention_gradient.run --root OUTPUT --expected-sha SHA --stage preflight
SOURCE=WORKTREE EXPECTED_SHA=SHA OUTPUT_ROOT=OUTPUT bash scripts/experiments/v4_attention_gradient/execute.sh
python -B -m scripts.experiments.v4_attention_gradient.run --root OUTPUT --expected-sha SHA --stage package

Preflight uses CPU and the original datasets/checkpoints. Pilot/full run only
inside the explicitly authorized live Slurm allocation. execute.sh sequences
pilot, full and CPU report; any failure stops the sequence. Package is separate,
after the execution_exit.txt receipt equals EXIT=0. All stages refuse to overwrite
an attempted stage; no automatic retries. Record manual engineering recovery
separately if needed, preserving failed evidence.

G0 windows: train ep2/73, ep2/141, ep7/107, ep2/20; val ep3/140, ep6/115.
G2 and G3: ep2/73, ep2/141, ep3/140. G1 excluded to bound work.
G0: four loss gradients, all ten internal states, same-coordinate parameter
cosines, per-point direction histograms, temporary zero-12D ablation.
G2/G3: same signal and intervention measurements, joint-only gradients.
H1/H5/H10 are highlighted in reports; all ten steps remain in raw tables.

Loss decomposition matches V4 CD+RGB+0.1alpha including sample availability,
camera/pixel/time reductions. All target times/cameras must have observations.
Output gradient is the direct H-mean gradient. Multiplying by ten gives the
standalone horizon loss gradient. Internal next_xyz gradient includes later AR
paths; future contribution is total_AR minus direct_Hmean.
Parameter gradient groups overlap (motion vs last layer, cross vs score/MLPs),
so their norms must not be added as independent modules. Missing and zero
gradients are distinct; zero-denominator cosines are null, never invented as 0.

Finite differences: G0 ep2/73 and ep2/141, standalone H10 loss. Fixed directions:
negative joint output gradient, +/- mean current-minus-initial valid active
Controller displacement, CPU generator seed1729 random per-GS vector field.
Normalize each field to unit valid-GS Euclidean RMS. Probe .001,.01,.1,.5,1 mm;
central differences at the two smallest scales. Full H10-mean delta equals
H10-only delta/10. Directions do not establish physically realizable motion.
No scale/direction search, no target correspondence assumption.

Temporal hooks observe the last Cross output and Motion input. Zero-tool only
replaces columns128:140; AR is rerun so later states can change. Hooks are always
removed in finally. No checkpoint rewriting. Before/after state and file hashes
must match. Original G2 always-active A2A Null is intentionally preserved.

Read final_report.md, results/conclusion.json and all CSV/JSON together.
Source archive includes all tracked scripts/ and AGENTS for import reproduction;
weights, datasets, environment caches and repeated render images are excluded.
CPU-only offline report recovery
-------------------------------
The actual CUDA preflight/pilot/full ran at 6d9eb12. All 12 records passed.
The server report stage failed only because its environment has no matplotlib.
Keep status/report.json, logs/report.txt and provenance/execution_exit.txt as-is.
On a local copy with matplotlib, run from this committed repository:
  python -B -m scripts.experiments.v4_attention_gradient.run --root <LOCAL_RESULT_ROOT> --expected-sha <FINAL_SHA> --stage report --postprocess-attempt local_recovery
  python -B -m scripts.experiments.v4_attention_gradient.run --root <LOCAL_RESULT_ROOT> --expected-sha <FINAL_SHA> --stage package --postprocess-attempt local_recovery
The recovery has its own status files. It cannot repeat a GPU/data stage.
The offline report checks the server checkpoint-integrity receipt against frozen
provenance; it does not pretend that Linux checkpoint paths exist on Windows.
Both GPU execution and final postprocessing source snapshots are included.
The final package gate requires successful preflight/pilot/full and recovered
report, and accepts EXIT=1 only for the recorded missing-matplotlib failure.
