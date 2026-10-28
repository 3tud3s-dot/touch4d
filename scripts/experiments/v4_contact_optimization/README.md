# Contact-only V4: frozen execution protocol

Base: 2e8afae7a74a1a4138f45e49dc8add83df6b579e. Squared nearest-neighbour implementation: acfe977fd6a734f1507d6423fd2147983f5e423f. No production model/default/loss file changes.

The manual v1 labels are external immutable inputs. Contact windows require s <= t and t+10 < e. Original episode-based stride and splits remain: 709 train, 52 val, 51 test. Test GT is never loaded by this runner.

Eight short configurations S1–S8 are fixed in protocol.py, seed42 and 800 successful updates, validation every200. Long A is original CD/window sampling; Long B is chosen from S6–S8 against S5 using the frozen eligibility rule, or S6 exploratory fallback. Both long runs use12352 successful updates, validation every1544. No seed43 replication or extra tuning is planned.

Squared geometry weight is the median of six train-only per-window XYZ gradient norm ratios at the SAME seed42 initialization. Exactly one calibration; no optimizer step. Calibration and visualization windows are the median contact start in each applicable episode, frozen before fitting.

Checkpoint selection uses the same episode-equal normalized ordinary CD + RGB + 0.1Alpha for all groups (`best_joint.pt`); each group's actual training loss is reported separately. `lowest_cd.pt` is diagnostic. This common metric avoids comparing losses with different dimensions. No threshold is fitted to validation results.

Sampling uses a stored deterministic Python RNG sequence: shuffled window passes, or uniform episode then uniform window. Every CD/squared pair has the same initialization and sample sequence. A different data distribution/repetition is an explicit variable.

CPU: `python -B -m unittest discover -s scripts/tests -t . -v`.
CUDA: submit run.sbatch with SOURCE, EXPECTED_SHA, OUTPUT_ROOT and STAGE=preflight. Only after gate.json passes submit STAGE=short array1-8%4. Long jobs require all eight completed records and selection.json, at most two GPUs. Existing30947 is read-only and never canceled.

No automatic numerical retries. All outputs are new; frozen inputs, main, other worktrees and historical checkpoints stay unchanged. GPU work requires Slurm. HTML and final scientific conclusions are generated only from actual records. Geometry file validity and manual overall contact intervals do not establish frame-level physical contact or reconstruction completeness.

Recovery is explicit and only for an approved infrastructure interruption. Atomic progress checkpoints are written after each successful update, before validation. A pending validation is repeated without repeating its optimizer step. Source SHA, complete experiment configuration, run identity and the contiguous update ledger must match; best-checkpoint selection history is retained. No numerical failure changes configuration automatically.

Actual frozen training source: `c608815f52529d30220893d2a31ca881cc0bca3d`. Exact submitted commands and job IDs are in the external output root's `pipeline_submission.json`. Later report-only commits do not change training provenance. Do not fast-forward an execution worktree while a job is still bound to its SHA.

After the real GPU comparison completes, rebuild the offline summary without inference using `python -B -m scripts.experiments.v4_contact_optimization.report --root /data1/userdata/tcweng/experiments/touch4d/v4-contact-optimization-v1 --summarize`. It requires completed acceptance and reads saved numerical results only. The existing GPU execution SHA is retained alongside the summary SHA.

`selected_camera_audit.py` is a read-only CPU census of the actual 001/011/021 supervision cameras on train/validation target frames. It never reads test GS, modifies HDF5, or infers physical contact. Its output is created exclusively under `provenance/selected_camera_audit.json`; an existing audit is not overwritten. The older 010-camera warning must not be attributed to 011.
