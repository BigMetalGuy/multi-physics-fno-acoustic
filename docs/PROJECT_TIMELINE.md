# CS 153 Project Timeline — Multi-Physics FNO Surrogates for Cylindrical Acoustic Inverse Design

← Back to [main README](../README.md) · Companion: [Pipeline mapping](PIPELINE_MAPPING.md) · [Receipts index](../receipts/README.md)

**Chronological log** of every meaningful decision, architecture change, success, failure, and training run from project genesis through current state (2026-06-04). Organized by era; cross-references to specific code paths, commits, and receipts inline.

---

## ERA 1 — Foundation & Pipeline Build (≤ 2026-05-03)

### Problem framing (predates the public repo)
- **Problem statement**: Build a real-time-capable forward surrogate for the acoustic Helmholtz equation in a cylindrical 40 kHz chamber driven by a 120-element phased ultrasonic array, so that PDE-constrained inverse design (target trajectory → required phases) can run in milliseconds instead of seconds–minutes.
- **Architectural bet**: Three forward physics tracks — analytical (1/r superposition), j-Wave (spectral Helmholtz), FEM-coupled (full Helmholtz + heat + Eckart streaming) — each with its own FNO surrogate. Their pairwise disagreement maps to a known missing-physics term and weights an adversarial training objective for a combined teacher, which then distills to a real-time student.
- **Implication**: For each track, build the forward solver, generate a dataset, train an FNO surrogate, validate via mean-prediction sanity gate, then move up the stack.

### Pre-PR-#1: ml_inverse pipeline Phases 1–5 (cleared in single PR on 2026-05-03)
- **Phase 1 — Data pipeline**: HDF5 schema (`phases`, `field`, `source`, `slice_z` columns); 80/20 uniform/rollout sampling; trajectory-id split via deterministic SHA1 hash with disjointness assertion; field as 2-channel (Re, Im) to avoid 2π wrap-learning; per-sample `slice_z_m` column.
- **Phase 2 — Forward surrogate** evolved through **three full iterations** (v1 → v2 → v2.1), see § Phase 2 Evolution below.
- **Phase 3 — Differentiable inverse**: forward-model-agnostic gradient descent through whichever surrogate is plugged in. Validated at field error 5.29e-4 on test sample idx=42. Multi-modality empirically confirmed across 3 restarts.
- **Phase 4 v1 — Block integration**: PD baseline. Landing error 24.9 mm — "wrong-architecture, not budget" → motivated Phase 4 v2.
- **Phase 4 v2 — Controller quality fix**: `ZeroResidualFNOController` in `controller.py`. Landing error 0.0072 mm (**1.97× PD baseline — stretch goal hit**). Sharpened framing: FNO+PID's optimal free-field residual is zero by construction; FNO would add value only outside free-field. 42 ml_inverse + 310 drip_physics tests all pass.
- **Phase 5 — Class deliverable**: video, README, FAQ, FINAL_REPORT. All 7 visual assets produced. Locks at 2026-05-25 / 29 final window.

### Phase 2 evolution (the crucial "data dominates network" lesson)

| Version | Result | Cause | Receipt |
|---|---|---|---|
| **v1** | mean-pred FAIL (ratio 1.17) | Pattern A broadcast conditioning collapsed to predicting train mean (~zero). LR sweep at `notebooks/04_mps_lr_investigation.md` ruled out hyperparameter cause. | `STATUS.md` |
| **v2** | mean-pred FAIL (ratio **1.76** — *worse than v1*) | Implementer added analytical-prior input channels; FNO learned ~zero residual against a *wrong prior*. Decorrelated prediction → larger raw L2 than v1. | `phases/PHASE2_PROMPT_REVIEW.md` |
| **v2.1** | mean-pred **PASS (ratio 0.0018)** | Domain-expert intuition ("the data is artificial, could this be a data error?") led to audit. v2 prior was wrong in 4 specific ways: hardcoded `slice_z=0.09` (data has variable z), wrong sign on k·r, missing `p_ref·r_ref` scale, missing sinc directivity. With correct prior, FNO learns ~zero residual against a *correct* baseline → trivially correct. | `research/V2_PRIOR_ATTRIBUTION.md` |

**The headline lesson — repeated throughout the project**: prior-correctness was a **30× lever** on model quality, vs. architecture variant being a **13% lever** (`research/FNO_ARCH_SWEEP.md` 4-variant sweep). **Data dominates network.**

### j-Wave backend discovery & fix (2026-05-10 prep work, surfaced earlier)
- `drip_physics/backends/jwave_backend.py` was discovered to already exist (April 2026) claiming JAX-native differentiable Helmholtz. Audit (`research/JWAVE_BACKEND_AUDIT.md`) found it broken in **4 ways**: source calibration (700× magnitude error), coordinate alignment (focal at boundary), SciPy interpolator (broke `jax.grad`), source-path differentiability. Fixer (~50 min) closed all four.
- Validated: `|p_jwave|` at focal 1091 Pa vs 1309 Pa analytical (ratio 0.833, within 20% spec), local argmax at focus, `jax.grad(loss)(phases)` finite, ‖grad‖ ≈ 2.5e5, full `drip_physics/` 310/310 tests pass.

### Disagreement-framework calibration
- FNO v2.1 vs analytical: 0.31% mean rel L2 across 50 configs.
- FNO_jwave vs analytical: 133% rel L2.
- 430× separation between "noise floor" and "regime divergence" → calibrated signal-to-noise.
- High-disagreement pixels concentrate on focal axis at 4× random rate.
- `disagreement_analysis.py` + `research/DISAGREEMENT_ANALYSIS.md`.

### FEM build (Phases 0–5, multi-week)
- **Phase 0 — Discovery**: JAX-FEM verified, `jax.grad` clean at AD/FD = 2.55e-7. Receipt `research/FEM_DISCOVERY_REPORT.md`.
- **Phase 1 — Mesh**: `mesh.py` + 14 tests; 239k tetrahedra; 120 transducer faces.
- **Phase 1.1 — Orphan-node cleanup**: `load_jax_fem_mesh` public API.
- **Phase 2.0 — Helmholtz Sommerfeld**: 5/5 tests, AD/FD 1.55e-9, focal spot 0.95 mm.
- **Phase 2.1 — Helmholtz Bermúdez 2007 PML**: 11/11 tests, reflection −18.45 dB, MUMPS solver path. Indefinite Helmholtz solve → BiCGSTAB diverges → `umfpack_solver` (scipy spsolve under JAX-FEM dispatcher) is default.
- **Phase 3 — HeatTransientProblem**: 7/7 tests, AD/FD 1.25e-5, trajectory 0.11s/step.
- **Phase 4 — StaggeredCouplingDriver (two-way)**: `coupling.py` ~620 lines + 7/7 tests. Eckart streaming + heat advection iterated staggered. AD/FD coupled-gradient < 1e-3. Inner-loop converges in ~2 iterations.
- **Phase 5 — Drop-in backend**: `femcoupled_backend.py` matches `compute_pressure_from_phases` signature → existing `inverse.py` runs through FEM with **no caller-side code changes**. The architectural-bet payoff in concrete form.

---

## ERA 2 — Phase 6.x: FNO_F Iteration (the "is the model broken?" arc) — 2026-05-04 → 05-15

This is the deepest single arc of the project. **Six iterations** of FNO_F before the first PASS.

### Phase 6.1 — Initial FNO_F training (2026-05-04)
- First FNO_F surrogate on Phase 5 FEM-coupled forward.
- Receipt: `Layer 3 disagreement entry`.

### Phase 6.2 — FNO_F production retrain receipts (2026-05-09)
- **1000 configs × 100 epochs, CPU run**.
- Status: training completed but mean-prediction sanity FAILED.

### Phase 6.3 — Scaled-prior diagnostic (2026-05-09)
- **Hypothesis**: prior magnitude is the problem.
- **Test**: scale prior by α=1/300.
- **Result**: mean_pred ratio went from 2.47 → 0.996. **Improvement, but still FAIL** (threshold < 0.5).
- **Conclusion**: prior scale was a real lever but not sufficient.

### Phase 6.4 — Per-sample H1 loss diagnostic (2026-05-09)
- **Hypothesis**: the batch-pooled `neuralop.H1Loss` has a "predict-the-mean" attractor at ratio_mean ≈ 1.0; per-sample relative H1 would prevent that.
- **Test**: train with `PerSampleRelativeH1Loss`.
- **Result**: **rejects the loss-formulation hypothesis**. Per-sample loss did not fix Phase 6.x failures.
- **Conclusion**: the issue was upstream of the loss function.

### Phase 6.5 — Design-pressure diagnostic + FNO_F decision log (2026-05-10)
- Investigated whether the target field magnitudes (and the design operating pressure) matched expectations.
- Built the FNO_F decision log to track all variants tried.

### Phase 6.6 — first attempt
- Status: **FAIL**.

### Phase 6.6b — FIRST FNO_F PASS (2026-05-10)
- **Receipt** (`receipts/_phase66_cloud_production_run/mean_pred.json`):
  - **`ratio_mean = 0.144`** (PASS, threshold < 0.5)
  - `ratio_median = 0.117`
  - `best_val_h1 = 9.947`
  - `median_model_err = 30.76` vs `median_mean_err = 261.07` → model 8.5× better than predict-mean
  - Grid: 32³ cubic
- **Significance**: first time the FNO_F architecture cleared the mean-prediction sanity gate. Validated that the v2.1 lessons (correct prior, residual architecture) transferred to FEM-coupled data.

### Architecture additions during Phase 6 (parallel work)
- **JWaveBackend + j-Wave gen with anisotropic grid support** (2026-05-10): the second forward track operationalized.

---

## ERA 3 — Disagreement Framework + Focal-Zone Gate (2026-05-15 → 05-17)

### Phase 7a — FNO_J first cloud production run
- **Receipt** (`receipts/_phase7a_cloud_production_run/mean_pred.json`):
  - `ratio_mean = 0.0936` (PASS)
  - `best_val_h1 = 1.179`
  - Grid: 32×32×96 mini-array
- First j-Wave-based FNO with clean PASS.

### 2026-05-15 — Pairwise FNO disagreement script (public repo commit `3886e0b`)
- Computed the FNO_F vs FNO_J L1 pairwise residual the disagreement matrix needs.
- Both models loaded from best-only checkpoints (no embedded `model_config` — config reconstructed from `state_dict` shapes + L1-array transducer positions baked into prior buffer).
- Each model evaluates at its native grid; FNO_J L1's chamber-scale output is trilinearly resampled onto FNO_F's 32³ focal-zone grid for direct Pa comparison.
- **N=20 result**: FNO_F vs FNO_J L1 = mean rel_l2 **1.071**, median 1.064, range [1.05, 1.10] — strikingly consistent.
- Context vs analytical baseline:
  - analytical vs FNO_F = 0.725
  - analytical vs FNO_J L1 = 0.693
  - analytical vs FNO_J mini = 0.998
- **Initial reading**: each FNO captures ~70% non-analytical structure, but they disagree at ~1.07 → "complementary physics components beyond analytical." Exactly the regime where disagreement-weighted adversarial training has the most signal.

### 2026-05-15 — Focal-zone signal quality metric + reveal Phase 7c undertraining (commit `b169eb0`)
- **The CRITICAL plot twist of the project.**
- The 1.07 pairwise disagreement initially looked like "two physics tracks learning complementary components." **Visual inspection of mid-z slices flipped the story**: FNO_F produces clean focal-spot structure (peak ~800 Pa at center), FNO_J L1's focal-zone output is essentially noise.
- **Built `focal_zone_signal_quality.py`**: measures whether a surrogate produces structured predictions inside the chamber interior (where droplets traverse) vs boundary regions.
  - `peak/mean > 4` if focused (vs ~4 for white noise)
  - `db_dynamic_range > 12` dB if real signal
  - `energy_in_focal > 0.05` (fraction of |P|² in focal zone)
- **Baseline N=10 results**:
  - FNO_F (Phase 6.6): peak/mean 5.99, dyn 15.5 dB, E_focal **0.747** → **PASS**
  - FNO_J L1 (Phase 7c): peak/mean 3.69, dyn 11.3 dB, E_focal **0.0487** → **FAIL**
- **The reveal**: 4.87% energy_in_focal is *below* uniform distribution (focal zone is 9.6% of chamber volume) → FNO_J L1 was *actively pushing energy away* from the interior toward transducer walls.
- **Diagnosis**: model learned the Robin BC forcing at transducer surfaces but not the interior wave-propagation physics that produces focal spots.
- **Phase 7c's mean_pred = 0.193 PASS was a FALSE POSITIVE.** Wall-dominated predictions reduce aggregate error enough to clear the 0.5 gate, but the load-bearing interior physics is missing. **Aggregate metrics are insufficient — focal_zone_signal_quality is a stricter gate.**

### Phase 7c v2 — FNO_J L1 retrain after focal-zone gate failure (2026-05-15)
- Retrained with focal-zone PASS as target.
- Locked retrain targets:
  - peak/mean ≥ 4.0 (preferably 5+)
  - db_dynamic_range ≥ 12 dB (preferably 15+)
  - energy_in_focal ≥ 0.05 (preferably 0.5+)

### Phase 7c — FNO_J production retrain (receipt)
- **Receipt** (`receipts/_phase7c_cloud_production_run/mean_pred.json`):
  - `ratio_mean = 0.193` (PASS by mean_pred, but FAILED focal_zone)
  - `best_val_h1 = 1.213`
  - `focal_peak_fraction = 0.7`
  - Grid: 44×44×144 L1 cylinder
- This was the run later flagged as false-positive PASS.

### 2026-05-17 — Methodology audit
- **Four eval-pipeline bugs fixed**, model verified operational.
- Documented the focal-zone story as a methodology lesson: **mean_pred_sanity has a known false-positive failure mode (wall-dominated learning); focal_zone gate added as second-layer check**.

---

## ERA 4 — Era-2 Audit Pass (2026-05-19 → 05-22)

- **13 audits, 30+ fixes** across physics + pipeline + sprint infrastructure.
- **2026-05-20 Pass 13**: 2 CRIT + 4 HIGH from reproducibility + sprint-ops audits.
- **2026-05-20 Pass 13b**: Closed deferred sim-to-hardware findings C-1, C-2, C-3, C-4.
- **2026-05-20**: POST_REGEN_RUNBOOK for upcoming Phase 7d/c cloud cycle.
- **2026-05-22**: CLOUD_PROVIDER toggle for Lambda → DigitalOcean kill-switch.
- **2026-05-22**: 5 physics-accuracy metrics + p5/p95 worst-case bars in eval.
- **2026-05-22**: `mean_pred_ratio` + `circ_phase_variance` (informational by default).
- **2026-05-22**: Document each metric's MEANS / DOES NOT MEAN to block false +/−.
- **2026-05-24**: `train.py` writes sibling `_config.json`; loader falls back to it.

---

## ERA 5 — Cloud Sprint: Stanford CS 153 / Omniva Cluster (2026-05-27 → 05-31)

### Allocation
- **4× H100 80GB HBM3 single-node**.
- **250 GPU-hours total**.
- Single-node only (all Stanford partitions `MaxNodes=1`).
- Access: kubectl exec into LoginSet pod (no SSH).

### 2026-05-27 — Phase 7d FEM gen on Stanford — **8 jobs in a row, ALL FAILED, then ABANDONED**
The most painful debug arc of the cloud phase. ~4.5 hours of debugging across 8 jobs.

| Job | Wall | State | Failure mode / fix attempted |
|---|---|---|---|
| **63** | 23:35 | CANCELLED | pip stalled 14+ min fetching rich-15.0.0 metadata. Worker network restricted. Fix: pre-stage wheels on login pod. |
| **64** | 23:49 | CANCELLED | Same pip stall — pre-staging not yet in place. |
| **71** | 6:31 | FAILED | Missing `gmsh` transitive. Fix: add to wheel set. |
| **73** | 0:01 | FAILED | Self-inflicted: stale empty h5 from job 71 blocked restart sentinel. Fix: `rm` before resubmit. |
| **74** | 6:19 | FAILED | Missing `libGLU.so.1`. Fix: apt-install libglu1-mesa libxrender1. |
| **76** | 5:49 | FAILED (100) | apt-get can't locate packages (stale index). Fix: `apt-get update` before install. |
| **77** | 5:58 | FAILED | Non-idempotent petsc4py patch re-applied on venv reuse + missing libGL.so.1. Fix: idempotent patch + libgl1. |
| **79** | 8:40 | FAILED | AttributeError `'NoneType' has no attribute 'Mat'` — monkey-patch insufficient. Tried real petsc4py install. |
| **80** | 26:16 | FAILED | **OpenMPI not built with SLURM PMI support → architectural mismatch.** *Gave up cluster path; gen ran on Mac M2 Max instead via `phase7d_local_gen.sh`.* |

**Outcome**: Phase 7d FEM dataset (16.7 GB, 5000 configs) ultimately generated on Mac M2 Max in ~7h wall, later uploaded to cluster for training.

Saved as memory: `omniva-startup-gotchas.md`.

### Phase 7c j-Wave gen (cluster — SUCCESS, 3 jobs)

| Job | Wall | Outcome | Notes |
|---|---|---|---|
| 83 | 21:51 | smoke 1 | **CPU jaxlib bug**: default PyPI wheel is CPU-only → 94 s/config (**23× slower than GPU**). Diagnosed and fixed. |
| 95 | 8:24 | smoke 2 | `jax[cuda12]` installed → 4.04 s/config. CUDA engagement confirmed. |
| **106** | **6:04:28** | **REAL — 5000 configs, 18 GB** | Mean 4.11 s/config (within 2% of smoke). gen_git_sha `f7dc1d1`. |

### FNO_J training smoke arc (2026-05-29 → 05-30, **11 attempts** before success)
Most failures were env setup (numpy 1.x vs 2.x + torch + tensorly + drip-physics-core interaction).

| Job | Wall | State | Failure mode / fix |
|---|---|---|---|
| 114 | 5:40 | FAILED | pip needs `--no-build-isolation` (setuptools-scm fetch). |
| 115 | 15:08 | FAILED | `tensorly-torch` PyPI ≠ `tltorch` Python import. |
| 116 | 6:12 | FAILED | h5 attrs `slice_extent_m` z-range asymmetric (`-0.24, +0.237`); `centred_grid` validator rejected. Fix: in-place attrs patch. |
| 117 | 6:06 | FAILED | torch compiled against numpy 1.x, container ships 2.2.6 → `from_numpy()` runtime fail. |
| 118 | 15:50 | FAILED | `--system-site-packages` venv inherited login pod's numpy 2.x. |
| 125 | 15:42 | FAILED | Pinned `numpy<2` in install — pip kept picking 2.2.6 (existing satisfied). |
| 126 | 15:07 | FAILED | `--ignore-installed --no-deps 'numpy==1.26.4'` — same outcome. |
| 128 | 15:40 | FAILED | `rm -rf /usr/local/.../numpy*` before install — STILL 2.2.6 (venv pollution from 115). |
| 156 | 0:00 | FAILED | sbatch syntax error: f-string in numpy guard broke outer `srun bash -c` quoting. |
| 157 | 6:08 | FAILED | **ROOT CAUSE FOUND**: `drip-physics-core` pyproject pins `numpy<3,>=2.2`. Its separate install undid numpy 1.26.4. Fix: `--no-deps` on dpc install. |
| **159** | **22:18** | **COMPLETED ✅** | Smoke 11 PASSED. val_h1=1.44, all imports cleared, checkpoint written (1.7 GB). |

Per-smoke wall 5–22 min; total debug burn ~2 GPU-hr. SLURM fail-fast (GPU released on FAILED) meant no idle waste.

Memory: `omniva-training-gotchas.md`.

### FNO_F + FNO_A smoke arc (2026-05-30, 4 jobs)
Discovered **SLURM rate-limit pattern**: rapid back-to-back submissions cause second job to instantly fail "Invalid TRES" — space by ~30s.

| Job | Wall | State | Notes |
|---|---|---|---|
| 161 | 0:01 | FAILED (TRES) | Submitted within seconds of 159 still finalizing. |
| **165** | 22:55 | **COMPLETED ⚠️** | **FNO_F smoke: mechanically passes, predictively dead.** Loss flatlined at val_h1=2.0000 (predict-zero baseline). Confirms SW-43: Phase 7d FEM dataset signal too weak for FNO (heat coupling is a no-op in single-shot gen). |
| 166 | 0:00 | FAILED (TRES) | Submitted same second as 165. |
| 167 | 17:35 | COMPLETED ✅ | **FNO_A smoke**: val_h1=1.50, loss descended cleanly. |

### Real training runs (2026-05-30 → 05-31)

| Job | Wall | State | Final val_h1 | Notes |
|---|---|---|---|---|
| **183** | **8:00:01** | **TIMEOUT** at ep 43/50 | 1.084 (still descending @ −0.010/ep) | sbatch `--time=08:00:00` too tight. Per-epoch wall = 10.8 min (estimated 6 min). Best.pt at ep42 — usable but not converged. **Bumped sbatch to `--time=12:00:00`.** |
| **205** | **7:32:03** | **COMPLETED ✅** | **1.192** | **FNO_A real, full 50 epochs.** Descent: 3.23 (ep1) → 1.95 (ep10) → 1.54 (ep20) → 1.35 (ep30) → 1.23 (ep40) → 1.19 (ep50). Nearly plateaued. |
| 240 | RUNNING | — | — | FNO_J real rerun (fair comparison vs 183 timeout). |

### Cross-cutting lessons banked as memory (8 distinct gotchas)
1. **`drip-physics-core` pyproject pins `numpy<3,>=2.2`** — but works fine with 1.x. Install with `--no-deps` when other deps need 1.x.
2. **PyPI distribution names ≠ Python import names**: `neuraloperator`/`neuralop`, `tensorly-torch`/`tltorch`, `jax-fem`/`jax_fem`.
3. **jax-fem solver rename**: upstream renamed `umfpack_solver`→`spsolve` 2026-04-28. Pin commit `a749ca20` (pre-rename).
4. **Default jax wheel is CPU-only** — must use `jax[cuda12]`. 23× silent slowdown otherwise.
5. **`--system-site-packages` venv on container** inherits container's `/usr/local/...` site-packages → can mask/conflict with venv installs. Surgical numpy installs need `--force-reinstall --no-deps`.
6. **SLURM rate-limit on rapid submissions** — space by ~30s.
7. **Per-epoch wall ≠ smoke-extrapolation**: BATCH=2 with bigger batches → 10.8 min not 6 min. Need actual epoch-1 measurement.
8. **Container pull (pyxis squashfs build)** is ~6 min first time per worker, then cached.

---

## ERA 6 — SW-43: Thermal-Aware Extension (2026-05-31 → 06-02)

The diagnosis that motivated the whole v3 line.

### 2026-05-31 — SW-43 v2: bed-heat initial T wired into FEM gen pipeline
- Recognition: FEM coupling means ρ(T) and c₀(T) vary with bed temperature, but Phase 7d v1 had fixed bed temp 800 K.

### 2026-05-31 — SW-43 v2 fix: bed thickness above floor (frame-robust)

### 2026-05-31 — SW-43 v2.5: per-config bed_temp randomization
- Generate dataset with bed_temp ∈ [400, 1000] K sampled per config.
- **Voxel coefficient-of-variation**: v2 = 21%, v2.5 = 43% (validated). Goal: double the signal diversity.

### 2026-06-01 — SW-43 v3 model: `prior_version='v2.5'` thermal-aware FNO
- Add bed_temp as 121st input dim to FNO conditioning.
- `cond_mlp` rebuilt with 121 inputs.

### 2026-06-01 — SW-43 v3 data plumbing
- Adapter `pack_bed_temp` parameter, normalizes (bt − 600) / 300.
- Dataset class adds `bed_temp_K` to sample dict when present.

### 2026-06-01 — SW-43 v3 train.py: `--thermal-aware` flag

### 2026-06-01 — SW-43 v3 gen: write `bed_temp_K` natively at gen time
- Replaces post-hoc reconstruct script.

### 2026-06-01 — Cluster sbatch `TRAIN_EXTRA_FLAGS` knob

### v3 production dataset gen (multi-machine, 2026-06-01)
- **5000 configs on Ryzen 9 5900X workstation (rig-b)**, 12 parallel FEM workers, ~17 s/config sequential.
- **2000 configs on M2 Max** (Apple Accelerate `spsolve` ran **~3× faster per core than predicted** — a result we did not expect).
- Disjoint base seeds (43 + 2,000,000) guarantee no duplicate configs.
- Total 7000 configs in ~2 h wall (vs ~5 h single-machine extrapolation).
- Final dataset: 27 GB at `v3_thermal_aware_fem_3d_phase7d_L1_FINAL.h5`.

### Transport: kubectl truncation discovery
- `kubectl cp` for a 16 GB file: **truncated 2 MB short, silent**, sha256 mismatch.
- `kubectl exec ... cat` retry: **truncated 5 MB short on attempt 1**.
- **Conclusion**: kubectl streaming layer unreliable for multi-GB files via Stanford's Teleport API server.
- **Fix**: Cloudflare R2 intermediary (multipart upload at 64 MB chunks).
- Memory: `kubectl-streaming-unreliable.md`.

---

## ERA 7 — v3 Hyperparameter Sweep (2026-06-02)

### Job 446 — v3 thermal-aware smoke at LR=1e-3
- **FAILED**: val_h1 = 2.0000 **exactly** for 5 epochs.
- **Diagnostic**: model output collapsed to `mean|.| = 0.003` (350× smaller than target).
- **Diagnosis**: phases are raw radians [0, 2π] (not unit-normalized) → `cond_mlp` output std ≈ 6 → at LR=1e-3, gradient explosion → predict-zero attractor.
- Memory: `fno-lr-collapse.md`.

### Job 469 — v3 retry smoke at LR=1e-4
- **COMPLETED ✅**: val_h1 1.65 → 1.09 over 5 epochs. Clean monotonic descent.
- **Hypothesis A validated**: LR=1e-4 stays in descent basin.

### Job 520 — v3 thermal-aware 50ep production at LR=1e-4
- **OVERFIT at epoch 14**:
  - ep10: val_h1 = 2.01
  - ep14: val_h1 = 1.940 ← **BEST**
  - ep15–25: val rose to 2.13 while train kept dropping 1.66 → 1.31 → gap doubled
- **User scancelled at ep25** when smokes 565/566 confirmed Hypothesis A.
- best.pt frozen at ep14, val_h1 = 1.940.

### Job 565 — LR=5e-5 DDP smoke (Hypothesis A — LR halved)
- **20 epochs, 4× H100, BATCH=2, LR=5e-5**.
- Slower descent than 520 (val 2.18 at ep14 vs 520's 1.94 at ep14) but **monotonic — no uptick**.
- TIMEOUT at ep19 (4 h walltime). Still descending.
- **Confirmed**: LR=5e-5 avoids the ep14 overfit pattern.
- Memory: `fno-ddp-lr-overfit.md`.

### Job 566 — `--per-sample-loss` DDP smoke (Hypothesis B — different loss)
- Same DDP setup as 520 but with PerSampleRelativeH1Loss.
- **20 epochs COMPLETED.**
- val_h1 1.94 at ep14, **same overfit pattern at ep15+ as 520**. Final ep20 = 1.979.
- **Hypothesis B REJECTED**: loss landscape isn't the dominant variable.

### Job 603 — LR=5e-5 50ep production
- **COMPLETED, 50/50 epochs, 6h 52min.**
- Best val_h1 = **2.011 at ep22** — *worse* than 520's ep14 best (1.940).
- ep30–50 plateaued at val ~2.05, train kept dropping to 1.555 → train-val gap 0.5.
- **Result**: LR fix avoided 520's specific overfit pattern but produced worse absolute val. Different path, same ceiling around val_h1 ~ 2.0.

---

## ERA 8 — Post-mortem: Truncation Ceiling Discovery (2026-06-03)

### Round 1 — initial deep eval of 520 ep14 best.pt
- **Check 1 mean_pred_sanity**: ratio = 0.857 → **FAIL** (PASS < 0.5)
- **Check 2 focal_zone_signal_quality**: E_focal = 0.0000 → **FAIL** (PASS > 0.05)
- **Check 3 thermal_sensitivity**: rel diff 5.4% (model uses bed_temp but weakly)
- **Initial read**: model is wall-dominated, hasn't learned interior physics. Same false-positive pattern as Phase 7c v1.

### R2 + MODEL_MANIFEST update
- Uploaded 520 ep14 best.pt + norm + config to Cloudflare R2.
- Updated Railway env var MODEL_MANIFEST (16 entries, was 13).
- Memory: `r2-creds-location.md`.

### Round 2 — alternate hypothesis testing (THE CRITICAL FIND)
**The user pushed back**: "validate it hard first, explore the dataset, explore the model, find other explanations and disprove them."

Tested 5 alternate hypotheses against the focal-zone FAIL:

| # | Hypothesis | Result |
|---|---|---|
| **B** | Targets don't have focal-zone energy | **CONFIRMED**: 100 sampled targets median E_focal = 0.003, range [0.0023, 0.0039]. Uniform baseline 0.0395. **Targets have less focal energy than uniform.** |
| D | Normalization destroys focal-zone signal | REJECTED: focal/walls std ratio 0.30 (not crushed) |
| **E** | Truncated Fourier modes can't represent | **PARTIALLY CONFIRMED**: projecting targets onto 8×8×24 modes loses **83% of target structure** (rel L² = 0.83) |
| G | Gen-pipeline bug | REJECTED: targets tightly consistent |
| H | Prior contamination | INCONCLUSIVE (forward errors with prior disabled) |

**Two paradigm-shifting findings:**

1. **The focal_zone_signal_quality gate was misapplied.** It was designed for *inverse-design output fields*, not *forward-training targets on random-phase data*. Random phases produce diffuse interference; focal energy only emerges under focused (inverse-designed) phases. The model matching ~0 focal-zone energy is correct behavior on this evaluation.

2. **The val_h1 = 1.94 ceiling is architectural, not training.** Even the targets themselves can't be reconstructed beyond 17% relative-L² when projected onto the FNO's 8×8×24 mode budget. The trained model captures **~70% of what the architecture allows** (rel L² = 0.73 vs ceiling 0.83). **Adding more training data or epochs cannot break this ceiling.**

The math (sanity check):
- Predict-mean baseline rel L²: 0.85 → captures ~15% of signal
- Truncation ceiling: 0.83 → max architecturally representable ~17%
- Actual model: 0.73 → captures ~27% of signal, **70% of the architectural max**
- mean_pred ratio 0.857 = model / predict-mean baseline → model is 14% better than predict-mean (within noise of truncation ceiling)

Memory: `fno-fourier-truncation-ceiling.md`.

### Methodological lesson recorded
- **Distinguish representational ceiling from optimization ceiling** before scaling compute.
- Representational ceiling is measurable in minutes by projecting targets through the model's mode budget.
- Optimization ceiling is what hyperparameter sweeps actually address.
- We spent ~80 GPU-hr sweeping LR (520 → 565 → 603) before measuring the representational ceiling that explained everything. **Order should have been reversed.**

---

## ERA 9 — Mode-Scaling Experiment (2026-06-03 → 06-04 — COMPLETED)

### Sbatch knob added
- `N_MODES_X`/`Y`/`Z` and `HIDDEN` made env-overridable in `phase7c_train_omniva.sbatch`.
- Commit `33b62b9`, pushed.

### Job 800 — 12×12×36 modes smoke (~3.4× more spectral params)
- 5 epochs, gpu=1, BATCH=1, LR=1e-4, --thermal-aware.
- Wall 52:50.

| epoch | 469 (8×8×24) | 800 (12×12×36) | gap |
|---|---|---|---|
| 1 | 1.65 | 1.30 | −0.35 |
| 2 | 1.28 | 1.08 | −0.20 |
| 3 | 1.17 | 1.02 | −0.15 |
| 4 | 1.12 | 1.00 | −0.12 |
| 5 | 1.09 | **0.98** | −0.11 |

- **val_h1 broke below the 1.0 predict-zero baseline for the first time across any v3 run.**
- Test_all_h1 at ep5 = 0.952. Smoke hypothesis confirmed; fired production.

### Job 866 — 12×12×36 modes 50ep production (COMPLETED)
- 4× H100 DDP, BATCH=2, LR=1e-4, --thermal-aware.
- Wall 10:08:01 (~40 GPU-hr actual).
- **Best: val_h1 = 1.799 at ep25 (test_all_h1 = 1.754).** Final ep50 = 1.826.
- Curves descended cleanly from 4.0 → 1.94 by ep14 (same as 520's
  former best), then continued past the prior ceiling to ~1.80 as the
  cosine LR schedule decayed below the overfit regime.

### Job 866 deep evaluation vs 520 baseline

| Metric | 520 (8×8×24) | 866 (12×12×36) | Δ |
|---|---|---|---|
| best val_h1 | 1.940 | **1.799** | −7.3% |
| best test_all_h1 | 1.903 | **1.754** | −7.8% |
| mean_pred ratio (PASS < 0.5) | 0.833 ❌ FAIL | **0.281** ✅ **PASS** | −66% |
| model rel L² (normalized) | 0.730 | **0.247** | −66% |
| Pa magnitude ratio pred/target | 0.476 | **0.724** | +52% |
| normalized std ratio pred/target | 0.84 | **0.95** | +14% |
| thermal sensitivity (rel diff) | 4.1% | **8.1%** | +97% |
| truncation ceiling rel L² | 0.826 | 0.735 | −11% |

**Headline: 866 PASSED mean_pred (0.281 vs 520's 0.833) — first
thermal-aware FNO_F to clear the sanity gate.** Comparable to
FNO_J L1's 0.193 production result.

### Disagreement matrix recompute with 866 F

```
                  520 F (val=1.94, FAIL)         866 F (val=1.80, PASS)
          A         J         F          A         J         F
     A  0.000     1.296     4.405      0.000     1.296     3.768
     J  1.296     0.000     5.270      1.296     0.000     4.495
     F  4.405     5.270     0.000      3.768     4.495     0.000
```

- A↔J unchanged (1.296 → 1.296) — sanity ✓ (A and J weren't retrained).
- F's row: A↔F dropped 4.40 → 3.77 (−14%); J↔F dropped 5.27 → 4.50 (−14%).
- F's Pa magnitude recovered 7.6 → 11.6 Pa (target ~16) — physical
  improvement confirmed.
- **F still ~3× the regime-divergence calibration of A↔J = 1.30** (the
  project's prior-calibrated "complementary physics" signal level).
  FNO_combined adversarial training would still be dominated by F's
  representational deficit rather than complementary-physics signal —
  not yet justified.

### Honest narrative correction recorded

The original "Fourier truncation is the architectural ceiling" framing
turned out to be a *loose upper bound*, not a tight one. The truncation
test moved 11% (0.826 → 0.735) while the actual model loss moved 66%
(0.730 → 0.247). FNOs have hidden_channels, real-space convolutions, and
prior pathways beyond the strict truncated-mode subspace; the truncation
projection underestimates real capacity. The lesson — *distinguish
representational from optimization ceiling before scaling compute* —
still holds, but the quantitative ceiling estimate via truncation
projection requires a bigger-architecture comparison to validate.

---

## ERA 10 — Post-866 axis-exploration smokes (2026-06-04)

After 866 demonstrated the FAIL → PASS transition with the
8×8×24 → 12×12×36 mode change, we ran two single-axis smokes to test
which axis (mode count vs hidden channels) is the binding lever.

### Jobs 974, 975, 981 — single-axis variations + combined-axes test

All three are 5-epoch smokes, gpu=1, BATCH=1, LR=1e-4, --thermal-aware,
same v3 dataset.

| Job | Modes | Hidden | ep5 val_h1 | ep5 test_all_h1 | Checkpoint size |
|---|---|---|---|---|---|
| 866 (baseline smoke 800 ref) | 12×12×36 | 128 | 0.98 | 0.952 | 4.79 GB |
| 974 | **16×16×48** | 128 | **0.914** | 0.883 | 10.5 GB |
| 975 | 12×12×36 | **192** | **0.911** | 0.880 | 10.2 GB |
| 981 (combined) | **16×16×48** | **192** | running | — | — |

### What we measured

- Both 974 (more modes) and 975 (more hidden channels) improved on the
  866 smoke baseline by ~7% at ep5.
- 974 vs 975 at ep5: 0.914 vs 0.911 — 0.003 difference. Below the
  bootstrap CI widths we measured for 866's full eval (~0.005), so we
  can't distinguish them as separate winners.
- Checkpoint sizes ~2.2× the 866 baseline in both cases.

### What we don't know

- Which axis (modes vs hidden) is the better lever to scale further.
  Single seed, smoke scale only; the difference is below measurement
  noise.
- Whether the smoke-vs-production gap (smokes landed ~0.85 better than
  production in both 469→520 and 800→866 pairings) is the same
  magnitude for these bigger models or different.
- Whether the DDP cosine-LR overfit pattern that showed up at production
  scale in 520 and 866 (val rising mid-training before recovering) would
  appear in 974/975/981 production runs.
- Whether axes compound — that's what job 981 is testing. As of this
  writing, 981 is queued/running.

### Decision after the smokes landed

We did not fire a 50-epoch production run of either 974 or 975. The
information value of one 50-ep production at this point relative to the
GPU-hr cost (each ~15–20 GPU-hr based on smoke wall-time scaling) was
judged not worth it inside the submission window. The smokes themselves
are the data; a single 50-ep production wouldn't distinguish which axis
matters more without a controlled multi-seed comparison.

---

## Headline numbers as of 2026-06-04

| Phase / Model | mean_pred ratio | val_h1 | Verdict |
|---|---|---|---|
| Phase 6.6b FNO_F (32³) | **0.144** | 9.95 | PASS — first FNO_F mean_pred PASS |
| Phase 7a FNO_J (32×32×96) | **0.094** | 1.18 | PASS |
| Phase 7c FNO_J L1 (44×44×144) | 0.193 | 1.21 | PASS (mean_pred) but FAIL focal_zone — false positive |
| **Phase 7d v1/v2 FNO_F fixed bed_temp** | — | **4.0 flatline** | FAIL — diagnosed as hidden-conditioning failure |
| **Phase 7d v3 thermal-aware (8×8×24)** | 0.833 | 1.940 at ep14 | FAIL — initially read as architectural ceiling; was training-dynamics limit |
| **Phase 7d v3 thermal-aware (12×12×36) smoke** | — | 0.98 at ep5 | Single-GPU smoke broke below predict-zero baseline |
| **Phase 7d v3 thermal-aware (12×12×36) production (866)** | **0.240 PASS** (N=32 CI [0.238, 0.241]) | **1.799 at ep25** | mode-change coincided with FAIL→PASS; controlled ablation (modes vs param count) not yet done |
| Phase 7d v3 axis smokes (974, 975) | — | both ~0.91 at ep5 | More-modes and more-hidden-channels each improve over 866 baseline by ~7%; tied at smoke scale (within bootstrap noise) |
| Phase 7d v3 axis-combined smoke (981) | — | running | tests whether axes compound or saturate |
| Student v1 distilled from FNO_J | — | 0.479 ms inference (CPU, batch=1; reference hardware not specified in this submission's artifacts) | Distillation referenced for latency only — quality not validated in this submission |

---

## Reusable lessons & memory artifacts

| Memory file | Lesson |
|---|---|
| `fno-lr-collapse.md` | LR=1e-3 collapses FNO_F to predict-zero on FEM datasets; use 1e-4. Symptom: val_h1 stuck at exact constant. |
| `fno-ddp-lr-overfit.md` | LR=1e-4 + cosine T_max=50 overfits at ep14 on DDP (effective batch 8); smoke at T_max=5 hides this. |
| `fno-fourier-truncation-ceiling.md` | 8×8×24 modes lose 83% of target structure; focal_zone gate misapplied to forward-trained models on random-phase data. |
| `kubectl-streaming-unreliable.md` | kubectl cp + exec cat both truncate 16 GB files; use R2 intermediary for >10 GB cluster transfers. |
| `r2-creds-location.md` | R2 write creds at `~/.config/drip/r2_creds.env`; gotcha re docstring placeholder in extraction. |
| `omniva-startup-gotchas.md` | 5 distinct startup crashes burned before sbatch worked; pre-stage wheels. |
| `omniva-training-gotchas.md` | 5 training-side issues; tltorch import, h5 extent patch, numpy 1.x force-install. |
| `jax-fem-solver-rename.md` | Upstream renamed `umfpack_solver`→`spsolve` 2026-04-28; pin pre-rename or update code. |
| `neuralop-pypi-name.md` | Distribution is `neuraloperator`, not `neuralop` (import alias). |
| `sw43-thermal-aware-fno.md` | Why v2.5 dataset alone failed and the v3 thermal-aware fix. |

---

## Five takeaway threads for the presentation narrative

1. **Data dominates network (×30).** Phase 2 v1 → v2.1 (prior-correctness, 30× lever) vs Phase 3 arch sweep (13% lever). Repeated in Phase 6.3 (prior scaling) and Phase 7c v2 (focal-zone-targeted retraining).

2. **PASS gates are necessary but not sufficient.** Phase 7c v1's `mean_pred = 0.193 PASS` was a false positive — wall-dominated learning. Added `focal_zone_signal_quality` as second-layer gate. But then misapplied it to a forward-trained random-phase model in 2026-06-03 — gates have **scope of validity** too.

3. **Distinguish representational ceiling from optimization ceiling.** The val_h1 = 1.94 "ceiling" was architectural (truncated Fourier) all along; ~80 GPU-hr of LR sweeps could not have found it. Order of debug should be: representational check first (minutes), then optimization (hours).

4. **MLOps multi-machine orchestration unlocks speed.** 7000 configs in 2 h wall via parallel Mac M2 Max + Ryzen 5900X workstation gen. Apple Accelerate's spsolve was ~3× faster per core than expected — an empirical surprise worth capturing.

5. **Persistent agent memory across sessions changes the workflow.** Lessons (`fno-lr-collapse`, `kubectl-streaming-unreliable`, etc.) accumulate across sessions; a session N+1 picks up with the operational context a senior eng would have. The agent-as-MLOps-operator pattern is the meta-contribution.
