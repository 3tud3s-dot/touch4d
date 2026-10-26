# P3-v0: two-frame temporal dynamics and template refinement

- Source: branch `codex/p3-two-frame-h10`, commit `391188acf3ed62dfceb91d3b402012c16929b42b`.
- Status: **archived architecture experiment, not adopted, not merged into P2**. Preserve the branch and checkpoint evidence.
- Goal: separate one-step recurrence, two-frame temporal fusion and bounded scale/opacity refinement using M0/M1/M2/M3 paired H10 experiments.
- Historical changes: TemporalAnchorFusion, P3 One-Step Predictor, Template Refiner, temporal-pair loader and P3 fit/acceptance/report tools. These implementations are intentionally absent from the current P2 production tree.
- Existing evidence: `audit/p3-two-frame-h10-v0/report.html`, `results.json`, `acceptance.json`, `run_manifest.json`, `template_audit.json`; existing server runs retain checkpoints.

```bash
git worktree add --detach ../touch4d-p3 391188acf3ed62dfceb91d3b402012c16929b42b
```

Use an unused external path. The local browsing worktree prepared during this refactor is `D:\Amax\Projects\touch4d-worktrees\p3-experiment` at that commit. Original CLI paths and tests remain available there; no P3-specific reorganization or new training was performed. Historical outputs remain in the original project workspace and are not copied into the worktree.
