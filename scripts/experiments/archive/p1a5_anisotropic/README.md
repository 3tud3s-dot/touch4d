# P1a.5: learnable anisotropic RBF

- Goal: learn direction-dependent anchor interpolation from initial GS features.
- Source/result snapshot: `7bc78bbce69879afb53fee0dc0b6fee540621c5f` (historical main before the P2 promotion PR); implementation introduced at `880972f` in its ancestry.
- Status: ablation, **not adopted**. P2 uses the isotropic P1a decoder. The bounded-metric compatibility implementation remains solely to load/reproduce existing configured models.
- Changes: zero-initialized learned 3x3 metric, bounded SPD anisotropy, shared across the horizon; otherwise paired P1a protocol.
- Existing evidence: `audit/p1a5-anisotropic-v1/report.html` and its experiment outputs in the original workspace.

```bash
git worktree add --detach ../touch4d-p1a5 7bc78bbce69879afb53fee0dc0b6fee540621c5f
```

Use the historical recorded configs and checkpoints; the commit is the immutable reference even after main advances. Do not change the RBF/loss or extend training under the name of reproduction.
