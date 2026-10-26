# P1a: isotropic shared-anchor displacement

- Goal: reduce independent Gaussian center divergence with a coherent displacement field.
- Source: `ea0e79ce2bb2004c87f978adaee9ff1a010119c0` (P1a implementation, retained as an ancestor of `codex/structural-gs-h15`).
- Status: **isotropic decoder retained in adopted P2**; P1a's pre-structural encoder remains a historical comparison.
- Changes: initial FPS anchors, fixed isotropic RBF interpolation, cumulative displacement relative to initial GS; other GS attributes inherited. No covariance transport.
- Existing evidence: `audit/p1a-anchor-v1/report.html`; corresponding checkpoints/configs remain in the original server workspace.
- Original modules: `scripts/training/gs_decoder.py`, `model.py`, `train.py` and fixed-window tools at the commit.

```bash
git worktree add --detach ../touch4d-p1a ea0e79ce2bb2004c87f978adaee9ff1a010119c0
```

Use an unused external worktree path and the historical evidence's exact commands/configuration. Do not copy the old sources over P2 or rerun into existing results.
