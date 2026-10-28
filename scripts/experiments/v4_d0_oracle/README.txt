D0 Oracle capacity V1 (frozen H15 V4 network)

Base: 60b7e6044a139a92bf7fdad29c262bde437ae2f9.
No production model, dataset, renderer or loss implementation is modified.
Only new diagnostic scripts/tests are added.

Exact objects: ep7/start105 (train), ep3/start135 (validation diagnostic only).
Each independently fits h5, h10, h15 using production fixed 128-anchor RBF.
CD-only and original CD+RGB+0.1Alpha; each with zero and Normal cumulative-D initialization.
Adam lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0, no clipping.
300 updates each, 24 fits; no learning-rate search or early selection of budgets.
Every step logs all loss components. Best minimizes that fit's own objective,
not cherry-picked CD for the joint fit. Both initializations and last are retained.

Tests:
python -B -m unittest scripts.tests.training.test_v4_d0_oracle -v

Within a normally allocated Slurm/tmux GPU shell:
SOURCE=FROZEN_WORKTREE EXPECTED_SHA=COMMIT OUTPUT_ROOT=NEW_OUTPUT bash scripts/experiments/v4_d0_oracle/execute.sh

Execution refuses an existing config. All equivalence gates must pass before pilot:
zero displacement = Copy, trace cumulative_d = Normal, independently rebuilt field = model field.
Pilot is the first successful step of the first fit (not additional training budget).
The optimizer receives only D. All model weights requires_grad=False; model/field/checkpoint hashes checked.

Report on local CPU with matplotlib and Pillow:
python -B -m scripts.experiments.v4_d0_oracle.report --root OUTPUT

Open OUTPUT/report.html offline. Single-condition original-resolution RGB/alpha plus
three-camera sheets are in images/. Fixed-range geometry, distortion heatmaps,
Near/Far vector fields and optimization curves are in figures/.
All raw metrics, both starts and best/last are in results/.
arrays/*.pt contains ONLY fitted anchor variables and optimizer state, not model weights.
arrays/*.npz has world-metre GS sets and local geometric proxies, no invented GT identity.

The first GPU attempt (cbf05b8) stopped before any optimization because a diagnostic
world-coordinate export omitted a required mask. Its logs are preserved at the parent
v4-d0-oracle-capacity-v1/ directory. The corrected run uses run-v2/ and frozen 376d790.
The CPU-discovered neighbor-indexing error is also recorded in the initial local test log.
Neither diagnostic fix changes the loss, budget, production mapping or network.

Bounded optimization is not a proof of the theoretical representation upper bound.
Small Chamfer does not establish physically plausible strain; initial kNN statistics
are only geometric distortion proxies, and GT GS is independently reconstructed.
