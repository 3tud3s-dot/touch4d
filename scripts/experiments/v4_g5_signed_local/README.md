# G5-v1 signed local motion: Direct and Propagate

Implementation candidates based on frozen G4 source
`8cba442736b81d05e9785a5963345d3fe2c35131`. No G4/V4, Dataset, Loss,
renderer or fixed RBF edits. No G5 performance claim is made by CPU tests.

## Shared rollout and initialization

Both constructors call the original G4 constructor first. Both then construct
the same `direct_motion` module; only Propagate appends `propagation` afterward.
The same manual seed therefore yields identical G4 shared tensors and identical
Direct Mapping tensors. Both new local output projections are exactly zero.
Full H15 zero-local outputs match G4 bit-for-bit, including nontrivial existing
Motion Head and G4 global correction weights in the equivalence test.

The common Forward retains 14D GS, LocalGSBlock16, FPS128, the two original
Controller Cross blocks, the original 151D Head, G4 per-hand global correction,
and the initial fixed RBF. No original module is replaced. Each step computes

```
g4_d = base_cumulative_d + g4_global_delta
final_d = g4_d + local_correction
GS_next = initial_GS + initial_RBF @ final_d
```

`previous_d` is this final cumulative control; Anchor history is the actual
movement of the FPS GS centers after RBF, not a control-vector difference.
Predicted GS feeds the next step without detach or target substitution.
The other11D GS attributes inherit the initial state. H/interval metadata are
explicit, with H15/interval1 defaults. No future GS is read by the model.

## Direct: 18D per Anchor and hand

`hand_basis12 [B,2,12]` comes directly from the original G4 module: root
cumulative/adjacent XYZ and within-hand mean Controller cumulative/adjacent XYZ.
For each Anchor and hand, two independent geometric normalizations aggregate
Controller cumulative and adjacent displacement over their respective valid
endpoint intersections. Scores are `-.5 * squared_distance / sigma0**2`, with
the unchanged initial field bandwidth in normalized-world units.

The weights are computed in stable log/softmax space. There is no Null, learned
contact label, Controller Cross softmax sharing, distance threshold or Far
cutoff. No valid endpoint gives exactly zero output. Valid zero motion remains
valid. Inputs are sanitized before subtraction/distances, and full13D Action
validity gates the initial/current/previous hands as in G4. H1 adjacent motion
is zero. With missing Controller data, the two new spatial motion vectors are
zero; a valid observed root can still contribute through the G4 basis.

```
anchor_hand_basis18 [B,A,2,18] = [hand_basis12, local_cumulative3, local_adjacent3]
```

Each `[Anchor,hand]` is independently evaluated by the same Mapping:

```
query128 + corresponding_hand_token128
+ geometric_relativeXYZ3_and_distance1
+ signed_basis18 + actual_anchor_history3 + previous_final_d3
+ corresponding_G4_global_hand_contribution3
= 287 features
LayerNorm(287) -> Linear(287,128) -> GELU -> Linear(128,54)
54 -> matrix [3,18]; seed_hand = matrix @ signed_basis18
```

Geometric context is the Gaussian-weighted current Controller relative XYZ and
distance, divided by the same sigma0. The last Linear's weight and bias are
zero. The two hand seeds are summed only at the end. No new cross-Anchor
reduction or attention exists in Direct; existing G4 shared/context features
remain the original ones. Signed channels are preserved; a nonlinear Mapping
conditioned on them does not impose an odd or physically correct response.

The local output is **not centered**. It can have nonzero GS mean and can
counteract G4 translation in one region without a forced opposite correction.
`local_gs_mean` and `local_gs_demeaned_rms` measure the actual fixed-RBF result,
not an assumed pure deformation component. They never modify the prediction.

## Propagate: one signed-vector A2A layer

Only Propagate builds the initial spatial graph, once per rollout. Each valid
Anchor has Self first plus its nearest8 other valid Anchors, up to9 neighbors.
Distance ties use ascending Anchor row index with a stable sort. Padding has
no legal edges; small sets and singletons retain the available neighbors/Self.
The graph is fixed across H15; current relative coordinates stay differentiable.

For each hand separately, a shared score MLP reads
`qa128 + qb128 + hand_token128 + initial/current relativeXYZ_and_distance8 + basis18`,
giving 410 features. Its shape is LayerNorm410 -> Linear410,128 -> GELU ->
Linear128,1 (no bias). There is one layer and one independent row softmax per
hand, with no Null or Controller tokens. Active Anchor-and-hand rows sum to1;
inactive/padded rows are zero.

```
P [B,A,2,K]
seed_propagated[a,h] = seed_direct[a,h]
                    + 0.5 * sum_b P[a,h,b] * (seed_direct[b,h] - seed_direct[a,h])
```

Only signed3D vectors are propagated. The difference form preserves constant
seeds exactly. There is no extra zero output gate: while all Direct seeds are
zero, score gradients can be zero; after heterogeneous seeds become nonzero,
the score has a finite nonzero gradient. The hands are propagated before summing.
Conditional module independence does not mean the original shared G4 Query is
causally independent of the other hand.

This fixed spatial graph is not a material topology. Folded rope segments may
become neighbors. Softmax mixing can smooth/attenuate local signed seeds and
does not by itself establish physical validity.

## Trace and checkpoint contracts

`return_trace=True` preserves the ordinary prediction dictionary and returns
live tensors in `steps`. New names include `g4_global_delta`, `hand_basis12`,
`anchor_hand_basis18`, `direct_seed_per_hand`, `local_correction`,
`final_cumulative_anchor_displacement`, `local_gs_mean`, `local_gs_demeaned_rms`.
Propagate additionally returns `propagated_seed_per_hand`,
`anchor_attention_weights [B,A,2,K]`, `anchor_neighbor_indices` and
`anchor_neighbor_mask`. Graph indices/masks also appear at trace root.
Trace makes no full-tensor CPU transfers. Positions/displacements are in the
original normalized world frame; multiply displacement by DataProcessor scale
and1000 for mm. The RMS trace uses a norm with a defined derivative at zero.

`checkpoint.py` has independent Direct/Propagate versions and configs, strict
state loading, optional optimizer state, processor/source/step metadata and
atomic no-clobber saves. G0/G4 and wrong G5 variant/H/interval are rejected.
This is an architecture/Smoke snapshot facility, **not** a completed formal
epoch/RNG/window-cursor recovery system. A later formal runner needs its own
reviewed protocol; the production factory is unchanged.

Parameters at H15: G4=576107; Direct=620511; Propagate=674067.
Direct adds44404 parameters; Propagate adds53556 beyond Direct. A future
parameter-matched control is not implemented in this round.

## CPU acceptance and future training

```
python -B -m unittest scripts.tests.training.test_v4_g5_signed_local scripts.tests.training.test_v4_g5_propagation -v
python -B -m unittest discover -s scripts/tests -t . -v
```

Future formal comparison remains H15/interval1, eight epochs, seed42, batch1,
AdamW1e-4/decay1e-4, original CD+RGB+0.1Alpha, full1514Train/100Val, original
three-camera gsplat, full-Val Joint best selection. Both models must start
fresh with paired G4 and Direct initialization, never from trained G4/Smoke/
attribution weights. Existing G4 Best CD2.88409mm/Joint0.119000 is a reference,
not a G5 result. No formal G5 training entry is introduced in this stage.

The following **future** Smoke command is supported by `smoke.py` but is not
executed during CPU implementation acceptance. It uses fresh seed42, exactly
two diagnostic updates on real Train ep7/start105, the original loss/renderer,
and a separate no-clobber architecture snapshot. Run variants sequentially in
a normally allocated single GPU session; inspect resources first.

```bash
SOURCE=/data1/userdata/tcweng/projects/touch4d-worktrees/v4-g5-signed-local-v1
SHARED=/data1/userdata/tcweng/projects/touch4d
SHA='REPLACE_WITH_ACCEPTED_G5_COMMIT'
cd "$SOURCE"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$SHARED/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$SHARED/.cache/torch_extensions"
export TORCH_CUDA_ARCH_LIST=12.0 OMP_NUM_THREADS=4 MAX_JOBS=4
# Example for direct; after it finishes, use --variant propagate and a new output.
srun -p 5090 --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=00:20:00 \
 "$SHARED/.venv/bin/python" -B -m scripts.experiments.v4_g5_signed_local.smoke \
 --variant direct --expected-sha "$SHA" \
 --root "$SHARED/datasets/touch4d-v1" --depth-root "$SHARED/datasets/touch4d-depth-v1" \
 --output /data1/userdata/tcweng/experiments/touch4d/v4-g5-signed-local-v1-smoke/smoke-direct-REPLACE_WITH_UNIQUE_ID
```

Replace both REPLACE_* values and first sync the accepted commit to that clean
server worktree. No command in the example is executed by this acceptance.

No GPU run, H15 training, hyperparameter search, G6, contact label, new Loss or
deformation-gradient decoder is authorized by this implementation acceptance.
