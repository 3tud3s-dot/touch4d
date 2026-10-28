# G4-v1: signed per-hand global translation

Isolated experimental `G4GlobalMotionPredictor`, based on formal V4 H15 source
`60b7e6044a139a92bf7fdad29c262bde437ae2f9`. No edits to production
V4; no training script or GPU run in this first implementation stage.

## Mechanism

One per-hand 12D signed-motion basis:
`[root(t)-root(0), root(t)-root(t-1), mean_cp(t)-mean_cp(0), mean_cp(t)-mean_cp(t-1)]`.
Positions, displacements and outputs use V4's **normalized** world XYZ;
12D here is **per hand**, not G0's four-head 12D motion feature. Missing
endpoints are masked; adjacent motion is zero at H1. Left and right retain
physical identity and are never averaged before conditioning. A shared MLP,
conditioned on the post-Cross object query and existing per-hand token, outputs
one `3x12` matrix for each valid hand. The two per-hand vector contributions
are summed (not averaged) into a global correction `delta_T`.

Every valid Anchor receives the same correction: `d_G4 = d_G0 + delta_T`.
For fixed initial RBF weights that sum to one at every valid GS, the current
prediction has an exact additional translation `delta_T`, with no additional
within-step GS relative displacement. Full AR can of course affect later
local states through feedback.

The mapping's final linear layer is exactly zero-initialized, producing the
same function as G0 at initialization and allowing nonzero first-step gradients
on moving hands. With invalid hands or zero motion basis the new branch
contributes exactly zero, regardless of contextual token values. Null attention
continues unchanged in G0 Cross, but does not attenuate the direct raw signed
basis of this additional global branch.

## Testing / scope

CPU test command:

```
python -m pytest scripts/tests/training/test_v4_g4_global_motion.py -q
```

Tests cover equal shared initialization, bit-exact G0 outputs before training,
world signed axis and per-hand masks, missing/invalid controllers and H1,
strict translation invariant on H2, causal prefix and nonzero gradients.
**This branch has not undergone real-data H15 training or CUDA/gsplat validation.**
It is an architecture implementation candidate, not a performance result.
