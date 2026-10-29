Frozen G4 attribution; base 8cba442. No production code or checkpoint edits.

Full rollout: output-only temporary hook suppresses all/left/right NEW global
contribution. Original controllers, masks, cross attention and base motion head
are untouched. Removing one hand does not remove that hand's input observations.

Single-step: uses normal saved base_d and fixed RBF at the SAME normal history;
compares current-step on/off at H1/5/10/15. A real only_step hook re-forward on
ep3/start135 and ep7/start105 verifies equality. Later outputs from that single
step intervention are discarded; full AR is evaluated separately.

Full Val100 + exactly Train ep7/start105; Val ep3/start135 is within Val100.
All counts and hashes fail closed. No optimizer, backward or test split.

CPU:
python -B -m unittest scripts.tests.training.test_v4_g4_attribution -v
GPU: run.sbatch, one normally allocated 5090; source pinned to diagnostic SHA.
Report on a CPU environment with numpy/matplotlib/Pillow (local Windows):
python -B -m scripts.experiments.v4_g4_attribution.report --root OUTPUT

Original G4/G0 checkpoints remain on server. Download only result data/plots.
Preserved improvement ratios are signed, unbounded, and undefined for zero
denominator. They are frozen intervention contrasts, not training-causal proof.
