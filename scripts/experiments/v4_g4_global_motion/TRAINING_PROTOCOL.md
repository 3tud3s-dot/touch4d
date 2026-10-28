# G4-v1 formal H15 protocol

This is an implementation/protocol freeze, **not a training result**. No G5/G6.
Base implementation: `f5ddd2bdb6b40a1c3a506c5043d40b9301e48781`.
Original G0: `60b7e6044a139a92bf7fdad29c262bde437ae2f9`, actual run
`/data1/userdata/tcweng/experiments/touch4d/v4-001-h15-e8/run-30995/`.

## Frozen comparison

The original `config.json` SHA256 is
`b3a4cff85fe6e8c3f9a4f29fc15fe63107908720555c3342163f77338b2e5401`;
`manifest.json` SHA256 is
`cca3e5ace2e3f279871b89a9fee84ed0a1524c8178229d1f45497ec6dfa4e8a3`.
Preflight checks these exact files, all configuration keys (only architecture
and source SHA differ), and the complete current Dataset window lists.

| Setting | Frozen value |
|---|---|
| Model | Fresh G4GlobalMotionPredictor, H15, interval 1 |
| Scene / epochs | 001-rope / 8, single_epochs=0 |
| Train | ep1/2/4/5/7/8, stride1, all 1514 windows |
| Train per episode | 176 / 277 / 349 / 292 / 210 / 210 |
| Validation | ep3/6, stride5, 55+45=100 windows |
| Test | Never opened |
| Batch / workers / seed | 1 / 0 / 42 |
| Epoch shuffle | Original DataLoader generator seed=42+zero-based epoch |
| Optimizer / clipping | AdamW lr=1e-4, decay=1e-4 / norm1 |
| Loss | CD + RGB + 0.1 Alpha; scale=depth=0 |
| Rendering | Original 3 calibrated cameras, 1280x720, target-depth-valid mask |
| Selection | Lowest full-validation episode-equal Joint Loss; strict less-than |
| Expected updates | Up to 1514 per epoch, 12112 total; log actual successful steps |

No contact-only filter, curriculum, batch limit, overfit window subset or Oracle
label is introduced. The existing production train_epoch/evaluate_loader,
PreparedLoader/PreparedProcessor, loader, renderer and losses are reused.
As in G0, samples without rendering availability do not perform an optimizer
step; visited windows and successful steps are therefore recorded separately.

Fresh G0 and G4 with seed42 are constructed on CPU and every shared tensor is
checked exactly equal; the new G4 output layer must be zero. No pretrained or
Smoke checkpoint is loaded by the fresh training path. Extra G4 initialization
consumes RNG, but data sampling uses an independent generator and these models
have no dropout. All eight window orders are reconstructed from the original
DataLoader algorithm, including its iterator base-seed draw, and checked against
every actually visited window. The historical G0 run did not save an empirical
per-batch order log; this is a source/runtime-based reconstruction, not a claim
to have recovered such a historical log.

## Formal checkpoint and recovery boundary

`checkpoint.py` uses `touch4d-g4-formal-h15-epochs-v1`; G0 and the two-update
Smoke format are rejected. Full architecture/config, source SHA, processor,
runtime, complete manifest and epoch orders, model/Adam state, Python/NumPy/
CPU/CUDA RNG, successful updates, completed epochs, metrics and best score are
saved. No model factory or production checkpoint reader is changed.

`initial.pt` and `epoch_01.pt` ... `epoch_08.pt` are immutable durable commits.
`last.pt` and `best.pt` are atomically published aliases containing full states.
Run identity and existing history are checked before replacing aliases. Resume
selects the latest contiguous durable epoch and repairs aliases if publication
was interrupted. It refuses different run paths, source/config/manifest/runtime,
missing epochs, inconsistent metrics or unsafe initial states. Linux flock
prevents concurrent writers. A changed source SHA needs a separate audit, not
an automatic permissive resume.

**Resume is supported at completed epoch boundaries only.** If interrupted
mid-epoch or during validation before commit, those unsaved updates are rolled
back and that uncommitted epoch is replayed from the saved model/optimizer/RNG.
They are not added twice to the committed optimizer history. This does not
recover the latest in-memory batch and may repeat wall-clock computation.
CPU deterministic state/RNG continuation is tested; GPU bitwise continuation
is not promised because gsplat/CUDA kernels may be nondeterministic. No real
full-epoch resume or full training is executed in this implementation phase.

## Diagnostics, without selection leakage

Detached hooks record per-horizon left/right hand contributions, global_delta,
valid hand counts, signed mean XYZ and RMS in mm (normalization scale x1000).
Missing hands contribute zero; means include all samples, with valid counts
reported separately. These are diagnostics only, not a loss or best criterion.

`evaluate.py` is a separate read-only GPU entry after training. It loads a
formal G4 checkpoint and a compatible G0 H15 checkpoint, evaluates only Train
ep7/start105 and Val ep3/start135, and compares G4/G0/Copy with identical cameras,
Loss and masks. It records full per-horizon CD, RGB/Alpha, signed XYZ/RMS,
global-translation-removed motion, fixed Near/Middle/Far regions, and local
16-non-self-neighbor distance ratios/P95 distortion. Near/Far are initial
controller-distance bottom/top quartiles; Middle is the remainder. Local
pred-to-GT error is one-way; independent reconstructed GT has no material IDs.
These structural proxies are not physical strain. Different G0/G4 training
budgets must be disclosed; this entry does not make them automatically fair.

## Commands (examples only; do not submit without authorization)

Run from the pinned clean server G4 worktree. Replace SHA with the accepted
commit and RUN_OUTPUT with a new unique directory. Existing Smoke and G0
directories remain read-only.

```bash
SHARED=/data1/userdata/tcweng/projects/touch4d
SOURCE=/data1/userdata/tcweng/projects/touch4d-worktrees/v4-g4-global-motion-v1
G0_REFERENCE=/data1/userdata/tcweng/experiments/touch4d/v4-001-h15-e8/run-30995
EXPECTED_SHA=<accepted-commit>
cd "$SOURCE"
CUDA_VISIBLE_DEVICES='' "$SHARED/.venv/bin/python" -B -m scripts.experiments.v4_g4_global_motion.train \
 --root "$SHARED/datasets/touch4d-v1" --depth-root "$SHARED/datasets/touch4d-depth-v1" \
 --g0-reference "$G0_REFERENCE" --expected-sha "$EXPECTED_SHA" \
 --output <new-preflight-directory> --preflight-only

# NOT EXECUTED during implementation acceptance:
export SOURCE G0_REFERENCE EXPECTED_SHA
export RUN_OUTPUT=/data1/userdata/tcweng/experiments/touch4d/v4-g4-global-motion-v1-training/<unique-run-id>
export RESUME=0
sbatch --output=<existing-log-directory>/g4-%j.log scripts/experiments/v4_g4_global_motion/train.sbatch
# Explicit recovery only: same SHA, same RUN_OUTPUT, RESUME=1. No auto-requeue.
```

The Slurm script preserves the original 5090, one GPU, 4 CPUs, 48GB RAM,
24-hour limit, CUDA12.8/gsplat environment. It validates paths and cleanliness;
Python verifies manifests/runtime before training. Time-limit interruption
requires an explicit later resume; this stage submits nothing.

```bash
# Later, in an authorized GPU allocation, evaluation only:
"$SHARED/.venv/bin/python" -B -m scripts.experiments.v4_g4_global_motion.evaluate \
 --g4-checkpoint "$RUN_OUTPUT/best.pt" --g0-checkpoint <frozen-G0-H15-checkpoint> \
 --root "$SHARED/datasets/touch4d-v1" --depth-root "$SHARED/datasets/touch4d-depth-v1" \
 --expected-sha "$EXPECTED_SHA" --output <new-evaluation-directory>
```
