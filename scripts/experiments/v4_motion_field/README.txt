Normal-Oracle Motion Field Audit V1
Base: 753aaacb545b05e979d9b41b9f82200d5ca52d2f

CPU only. Saved arrays and anchor controls from v4-d0-oracle-capacity-v1/run-v2.
No model construction, checkpoint inference, optimizer, backward, or fitting.
Only ep7/start105 and ep3/start135, h5/h10/h15, both initializations, best and last.

Run from repository root (OUTPUT is a fresh directory):
python -B -m unittest scripts.tests.training.test_v4_motion_field -v
python -B -m unittest discover -s scripts/tests -t .
python -B -m scripts.experiments.v4_motion_field.analyze --source SOURCE --output OUTPUT
python -B -m scripts.experiments.v4_motion_field.visualize --root OUTPUT
python -B -m scripts.experiments.v4_motion_field.report --root OUTPUT
python -B -m scripts.experiments.v4_motion_field.verify --source SOURCE --root OUTPUT

Requires CPU torch, numpy, matplotlib, Pillow. No GPU or gsplat execution.
analyze refuses to overwrite a completed audit. Figure/report regeneration affects
only this new output directory; source hashes must remain unchanged.

Production FPS/RBF rebuilt from saved initial world XYZ after inverse normalization.
All 54 saved GS fields must match mapped controls within max 0.0001 mm. This is
float32 round-trip tolerance, not a scientific quality threshold or an exact-bit claim.
Otherwise fail closed and request original tensor export; never adjust tolerance to fit.

Direction undefined when either vector norm <=1e-6 m. No contact threshold inferred.
World metres for calculation, mm for displays. Vector RMS means sqrt(mean(sum(xyz^2))).
Initial saved near/far quarter indexes remain fixed; middle is remainder.
Initial production 16-NN slots include self, then self/zero edges are removed as before.
No GT row identities. D control variables are distinct from actual FPS movement W D.
Global translation removed residual still includes rotation; not pure material strain.

Both losses x both starts x best/last are retained, with own-objective best selection.
23 fits have best=last300. ep7/h10 Joint/Zero best296 differs from last300 slightly.
This was checked against original saved D rather than assumed from similar curves.

HTML uses relative images and no fetch/CDN. All 12 selector states are statically
validated; report does not claim real-browser interaction testing.

No modifications to production scripts, original checkpoint, Oracle files or main.
