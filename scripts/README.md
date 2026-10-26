# touch4d scripts

This tree promotes the accepted **P2** baseline at `fdd4013c7f166daae12379f4ee020b4da5724965` without changing its model or experiment semantics. P3-v0 is not adopted; its implementation remains on its historical branch.

```text
scripts/
├── data_process/             # scene normalization and checks
├── dataloader/               # data contracts, splits, collation and checks
├── rebuild_dataset/         # rebuild and validation
├── training/                # P2 model and production entry points
│   ├── model.py             # PositionPredictor, unchanged import path
│   ├── gs_structure.py      # Structural GS Encoder
│   ├── gs_decoder.py        # shared anchors + legacy decoder modes
│   ├── train.py / evaluate.py
│   ├── losses.py / rendering.py / visualize.py
│   └── fixed_fit.py / oracle.py / oracle_fit.py  # import/CLI shims only
├── experiments/
│   ├── fixed_window/fixed_fit.py
│   ├── oracle/oracle.py / oracle_fit.py
│   └── archive/             # version indexes, no copied historical sources
└── tests/
    ├── data_process/
    ├── dataloader/
    └── training/
```

Run commands from this worktree's repository root using the existing environment. A worktree has its own project root: pass the existing server data paths explicitly when needed. Do not copy datasets or checkpoints into the checkout.

```bash
python -B -m scripts.training.train --help
python -B -m scripts.training.evaluate --help
python -B -m scripts.training.visualize --help
python -B -m scripts.training.train --protocol fixed400 --help
python -B -m scripts.experiments.oracle.oracle_fit --help
```

The fixed-window dispatcher remains `scripts.training.train --protocol fixed400`; its implementation is under `experiments/fixed_window`. The legacy `python -m scripts.training.oracle_fit` entry still works. The three old experiment import paths alias the canonical modules (including globals used by patching), rather than maintaining a second implementation. Fixed window selection, 400 updates, checkpoint recovery and report generation are unchanged. Importing a module or requesting `--help` never starts training.

`experiments/fixed_window/provenance.py` records the exact reviewed source-layout migration. Fixed400 resume retains strict configuration/data checks and accepts the frozen P2 H5/H15 source hashes only while the migrated implementation inventory is unchanged. Unknown source hashes or later implementation edits fail closed. New checkpoints hash the actual migrated tools; a resumed checkpoint records the source migration. Older unrelated experiments retain their original worktree for exact reproduction.

The P2 model uses 14D GS, LocalGSBlock, 128 FPS anchors, structural anchor attention and isotropic shared-anchor displacement. H=5/10/15, action/tactile causality and existing checkpoint versions remain supported. Compatibility modes in `model.py` and `gs_decoder.py` intentionally retain earlier free-head and anisotropic models; their presence does not make those modes the adopted baseline. Existing checkpoint configs select their original architecture. This refactor does not alter CLI defaults to silently select P2.

## Starting the adopted P2 model

Always specify **`--gs-decoder anchors --gs-structure anchor_attention`** for a new P2 run. Omitting these flags starts the legacy Free Head (`gs_decoder=free`, `gs_structure=none`); specifying only `anchors` selects P1a, not P2. This compatibility default is intentional. The paired P2 experiments also use `--chamfer-weight 1`; the generic epoch CLI defaults to `0.1`, so specify the loss weights explicitly.

The following is a complete **epoch-training invocation**, for use only after training is separately authorized and a GPU has been allocated through Slurm. It is documentation, not a new training or full-dataset validation result. Run from the reviewed P2 checkout; the current server candidate is shown below. After PR #1 is merged and the main server checkout is fast-forwarded, that checkout can be used instead. The Python environment, rebuilt data and depth data remain in the original server project.

```bash
cd /data1/userdata/tcweng/projects/touch4d-worktrees/p2-baseline
git branch --show-current
git rev-parse HEAD

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR=/data1/userdata/tcweng/projects/touch4d/.cache/torch_extensions
export TORCH_CUDA_ARCH_LIST=12.0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1

/data1/userdata/tcweng/projects/touch4d/.venv/bin/python -B -m scripts.training.train \
  --protocol epochs --scene 001-rope \
  --root /data1/userdata/tcweng/projects/touch4d/datasets/touch4d-v1 \
  --depth-root /data1/userdata/tcweng/projects/touch4d/datasets/touch4d-depth-v1 \
  --gs-decoder anchors --gs-structure anchor_attention \
  --anchor-count 128 --anchor-pool-k 16 --anchor-bandwidth 1 \
  --gs-local-attention knn --gs-local-k 16 \
  --controller-features 7 --prediction-mode autoregressive --output-mode gs_geometry \
  --width 128 --heads 4 --layers 2 --horizon 5 \
  --chamfer-weight 1 --rgb-weight 1 --alpha-weight 0.1 --scale-weight 0.01 --depth-weight 0 \
  --epochs 20 --single-epochs 1 --batch-size 1 --workers 0 \
  --seed 42 --lr 1e-4 --weight-decay 1e-4 --chamfer-chunk 512 --device cuda
```

- Train `051-cube` separately by changing `--scene`; do not mix scene normalization or checkpoints. The audited three-camera order is selected automatically for each scene. H defaults to 5; explicitly select `--horizon 10` or `15` when that experiment is intended. Conditions arrive online through `t+H-1`; this is not open-loop prediction with future tactile input available at the initial state.
- Epoch training uses the existing train/validation splits, one initial single-hand epoch and then all training episodes. Omitting `--run-id` creates a unique directory under **the checkout running the command**: `runs/<scene>/<timestamp>-<suffix>/`. A supplied run ID must be new; never reuse a historical experiment directory.
- To reproduce the **fixed-window 400-update diagnostic protocol**, replace `--protocol epochs --epochs 20 --single-epochs 1` with `--protocol fixed400 --fit-diagnostic`, retaining the explicit P2 model/loss flags. Two exact windows alternate for 200 updates each; this is not full training. H=10/15 additionally requires a previously audited `--window-manifest` with matching H. The existing H15 manifest is `/data1/userdata/tcweng/projects/touch4d/audit/structural-gs-h15-v1/stage0_frozen_windows.json`. Do not claim fit acceptance without reviewed tolerances.
- Resume with `--resume /absolute/path/to/last.pt`, the same scene/protocol/data/model/loss options, and no `--run-id`. Epoch resumes require `--epochs` to exceed the completed count. The loader inherits checkpoint architecture/H and rejects conflicting overrides; it does not convert an old Free Head checkpoint into P2. Use `best.pt` for epoch evaluation, or `lowest_cd.pt`/`last_400.pt` with `scripts.training.evaluate --split fit` for diagnostic fixed400 results.

Before launching, verify the parsed configuration contains `gs_decoder=anchors`, `gs_structure=anchor_attention`, `prediction_mode=autoregressive`, `output_mode=gs_geometry`, the intended horizon and `chamfer_weight=1`. Inspect the `training` section of the new run's `config.json` before relying on its results.

## Tests

```bash
python -B -m unittest discover -s scripts/tests -t . -v
python -B -m unittest discover -s scripts/tests/training -t . -v
python -B -m unittest discover -s scripts/tests/dataloader -t . -v
python -B -m unittest discover -s scripts/tests/data_process -t . -v
```

On an allocated CUDA node with the existing gsplat environment:

```bash
TOUCH4D_TEST_CUDA_RENDER=1 python -B -m unittest discover -s scripts/tests -t . -v
```

All ten original test modules are retained under `tests/`; their historical locations remain available at the pre-refactor Git commit. The refactor adds compatibility tests without removing assertions. CPU environments may skip h5py/CUDA-dependent cases; report these separately from failures.

Historical experiment commits and report locations: [experiment index](experiments/README.md). Data/model protocols remain in the existing root and `docs/` documents; this change does not rewrite those historical documents or evidence paths.
