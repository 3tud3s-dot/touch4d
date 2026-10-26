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
