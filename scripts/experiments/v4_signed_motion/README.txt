Controller-Object Signed Motion Audit V1
Base eed58d859bdeb60dfdcf33c268c56bc8410cf653; production H15 60b7e604.
No production changes, network forward, optimizer, backward or new Oracle fit.
Only ep7/start105 and ep3/start135. CPU reads raw_batch/outputs_and_trace/zero_tool_inputs
from H15 frozen ablation, and fixed motion fields from the preceding Oracle audit.
Zero-tool input snapshot is the unchanged input (prior intervention acts at MLP hook).
features() and DataProcessor replay are deterministic CPU algebra, not learned inference.

Before running, copy existing metadata.json, controller.npz and timestamps.npz for
episode_3 and episode_7 into OUTPUT/provenance/episode_3 and episode_7. Verify against
server source hashes. Do not modify Dataset or copy GS frames.

From repository root:
python -B -m unittest scripts.tests.training.test_v4_signed_motion -v
python -B -m scripts.experiments.v4_signed_motion.audit --ablation ABLATION_ROOT --motion MOTION_FIELD_ROOT --output OUTPUT
python -B -m scripts.experiments.v4_signed_motion.visualize --root OUTPUT
python -B -m scripts.experiments.v4_signed_motion.report --root OUTPUT
python -B -m scripts.experiments.v4_signed_motion.verify --root OUTPUT

All original source files remain read-only. Audit refuses a completed output directory.
No full training-test suite is invoked in this task because it contains optimizer/backward
tests outside the explicit read-only constraint; eight new CPU-only tests are used.

H1 reads t for t+1; H15 reads t+14 for t+15. A0 cumulative differs from 12D adjacent.
All four heads retained; null mass not renormalized away. Compare per-hand and per-finger,
never invent a common two-hand direction. Near/Middle/Far fixed from previous audit.
World metres internally, mm in plots; direction undefined below 1e-6 m. This threshold
only defines numerically meaningful angles, not contact or a motion-selection criterion.
Residual RMS within each region subtracts that region's mean; all-region residual is
the requested globally de-translated motion. Residual may include rigid rotation.

Original ep7 openings below 40mm are clipped in Controller geometry but not Action State.
This existing calibrated-range behavior is reported as a design limit, not fixed here.
External calibration and original device clock synchronization are not independently measured.

Initial diagnostic-tool failures are retained in logs: depth fields/metadata contract,
float32 world export versus original coordinates, and overly exact synthetic delta assertion.
Corrected with full batch contract and explicit round-trip/ULP arithmetic checks.
No old production tests or expected results were changed.
