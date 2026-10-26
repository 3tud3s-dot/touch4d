# P0: free GS head baseline

- Goal: establish exact fixed-window 400-update fitting and trustworthy geometry/rendering diagnostics before changing the decoder.
- Source: `e412b6f5ec14b4279f3a401b3b07c78c2ae1615a` (accepted Python 3.10-compatible P0 implementation; ancestor retained in P2 history).
- Status: historical baseline; the free-head architecture is not the current adopted P2 decoder. Legacy checkpoint support remains.
- Changes: exact window selection, common metrics, best-fit/lowest-CD versus last-400 management, geometry/ellipsoid visualization; no target point-ID assumption.
- Existing evidence: `audit/p0-free-head-v1/report.html` and its associated files in the original project workspace.
- Original code: `scripts/training/fixed_fit.py` and the contemporary training modules at this commit.

From an existing clone, create a new external path (choose an unused location):

```bash
git worktree add --detach ../touch4d-p0 e412b6f5ec14b4279f3a401b3b07c78c2ae1615a
```

Use the recorded config, commands, environment and checkpoints from the evidence to reproduce. Checkout alone does not fetch datasets or authorize training/overwriting historical outputs.
