# CS 153 Pipeline Mapping — Canonical ML Stages × Our Multi-Physics Surrogate Stack

**Companion to `cs153_project_timeline.md`.** Each section maps our concrete work onto the standard ML training pipeline stages so the narrative becomes "this is what a modern ML pipeline looks like for surrogate-based PDE-constrained control" rather than "here is a list of training jobs."

Standard ML stages (LLM-era convention):

```
DATA → PRETRAIN → TRAIN → MID-TRAIN → POST-TRAIN → DEPLOYMENT → ONLINE FEEDBACK
```

Our project equivalent (multi-physics surrogate stack):

```
DATASET GEN (3 physics tracks) → ARCHITECTURE + PRIOR DESIGN → TRAIN 3 FNOs →
DISAGREEMENT-WEIGHTED COMBINE → DISTILL STUDENT → DEPLOY IN SIM / HARDWARE →
TELEMETRY FEEDBACK (future)
```

---

## STAGE 0 — DATA / DATASET GENERATION

**Standard ML**: collect raw corpus, clean, tokenize, split into train/val/test.

**Our equivalent**: generate forward-physics training data from three physics solvers, each at multiple grid resolutions and parameter regimes. The three solvers have different fidelity / cost / coverage trade-offs.

### The three forward physics tracks

| Track | Solver | Captures | Misses | Cost |
|---|---|---|---|---|
| **Analytical** | 1/r superposition (closed-form) | Free-field interference | Diffraction, reflection, coupling | Microseconds |
| **j-Wave** | JAX-native spectral Helmholtz | Diffraction + reflection in free field | Streaming, heat-coupled dispersion | ~4 s/config (GPU) |
| **FEM-coupled** | jax-fem + scipy spsolve (Helmholtz + heat + Eckart streaming) | Full multi-physics | — | ~17 s/config (CPU) |

### Per-dataset evolution (every dataset version we shipped)

| Dataset | Era | Solver | Grid | Configs | Notes |
|---|---|---|---|---|---|
| v1 (Phase 1) | Era 1 | analytical, 2D | 64×64 | small | Field as 2-channel (Re, Im) to avoid 2π wrap. First HDF5 schema. |
| v2 (Phase 1, 2D) | Era 1 | analytical, 2D | 64×64 | small | Adds `slice_z_m` column. |
| Phase 6.6 FEM | Era 2 | FEM-coupled | 32³ cubic | ~1000 | First Phase 6.6b PASS dataset (mean_pred 0.144). |
| Phase 7a j-Wave | Era 5 | j-Wave | 32×32×96 mini | ~5000 | FNO_J first cloud PASS (ratio 0.094). |
| **Phase 7c j-Wave L1** | Era 5 | j-Wave | 44×44×144 cylinder | **5000** | 6 h cluster gen (job 106, 18 GB). |
| Phase 7d v1 FEM | Era 5 | FEM-coupled | 56×56×160 cylinder | 5000 | Fixed bed_temp 800 K. Voxel CV 21%. FNO_F flatlined. |
| Phase 7d v2 FEM | Era 6 | FEM-coupled | 56×56×160 | 5000 | Expanded sweep of other vars. Still flatlined. |
| Phase 7d v2.5 FEM | Era 6 | FEM-coupled | 56×56×160 | 5000 | Per-config random bed_temp ∈ [400, 1000] K. Voxel CV 43%. |
| **Phase 7d v3 thermal-aware FEM** | Era 6 | FEM-coupled | 56×56×160 | **7000** | bed_temp written natively as h5 column. Multi-machine gen (Mac 2000 + Ryzen 5000) in 2 h wall. 27 GB. |

### What this stage taught us
- **Data dominates network ~30×** — prior-correctness fixed a Phase 2 v1 → v2.1 jump from FAIL to PASS at 0.0018. Architecture variants gave only ~13% improvements.
- **Voxel coefficient-of-variation (CV) as a data-quality scalar** — v1 21% → v2.5 43%, predicted to break the flatline. (It didn't, alone — exposed the next-layer hidden-conditioning issue.)
- **Hidden conditioning variables make the task non-functional** — same input, different output → optimal predictor is conditional mean. Drove the v3 thermal-aware fix.
- **Multi-machine parallel gen unlocks wall-time gains** — Apple Accelerate's `spsolve` was empirically 3× faster per core than Ryzen + scipy.

### Where this stage is in the timeline doc
- Era 1 (Phase 1 data pipeline), Era 5 (Phase 7c j-Wave gen, Phase 7d cluster gen failure → Mac fallback), Era 6 (SW-43 v2 → v2.5 → v3 dataset evolution).

---

## STAGE 1 — PRETRAIN / ARCHITECTURE + PRIOR DESIGN

**Standard ML**: train the base architecture (e.g., language modeling) on raw corpus to acquire general representations.

**Our equivalent**: design the inductive biases — the **FNO architecture choices** and the **physical prior** that gets concatenated to inputs. This is the "what does the network start knowing for free?" step.

### Architecture decisions (the inductive bias)
- **FNO topology**: spectral conv layers in truncated Fourier basis. Modes 8×8×24 (later 12×12×36 in v3 mode-scaling experiment). Hidden channels = 128. `n_layers = 4`.
- **Output format**: 2 real channels (Re, Im) for the complex pressure field. NOT magnitude/phase (avoids 2π wrap-learning).
- **Conditioning**: Pattern A — small `cond_mlp` lifts the input phase vector to `cond_channels`, broadcast spatially to inject into FNO.
- **Loss**: H1 (= sqrt(L² + gradient L²)) with `measure=axis_lengths_m` for anisotropic-grid correctness. The H1 gradient term matters because downstream uses ∇|P|² for the radiation force.
- **Optimizer**: torch.optim.AdamW (NOT neuralop's AdamW — that produced NaN params after first step on MPS). Cosine LR schedule with `eta_min = lr × 0.01`.

### Physical prior evolution (the inductive bias's other half)

| Prior version | Form | Status |
|---|---|---|
| **v2** prior | Hardcoded slice_z=0.09, wrong sign on k·r, missing p_ref·r_ref scale, missing sinc directivity | **Wrong in 4 ways**. Drove Phase 2 v2 to mean-pred ratio 1.76 (worse than v1). |
| **v2.1** prior | Full physical formula matching `compute_pressure_field`: per-sample slice_z, correct k·r sign, correct scale, sinc(ka·sinθ) directivity | **Correct**. Drove Phase 2 to mean-pred ratio 0.0018 — 30× lever. |
| **v2.1_3d** prior | 3D variant (no slice_z packing — z is a grid axis) | Used for Phase 7c L1 production. |
| **v2.5** prior | v2.1_3d + bed_temp packed as 121st input channel through `cond_mlp` | The v3 thermal-aware extension. |

### Residual-prior toggle (the architectural switch)
- `residual_prior=True`: model output = `normalize(prior) + fno_output` — FNO learns small correction over the prior.
- `residual_prior=False`: model output = `fno_output` — FNO must learn the full field.
- v3 thermal-aware uses `residual_prior=False` (FNO learns whole field, prior is just a conditioning channel).

### Where this stage is in the timeline doc
- Era 1 (Phase 2 v1→v2→v2.1 evolution, 4-variant arch sweep), Era 2 (Phase 6.3 prior-scaling diagnostic).

---

## STAGE 2 — TRAIN (the three forward surrogates)

**Standard ML**: train the primary task model on the curated dataset to convergence.

**Our equivalent**: train one FNO surrogate per physics track. Three independent training runs producing three "expert" forward surrogates with overlapping but distinct competence.

### The three FNOs

| Surrogate | Dataset | Final val_h1 | mean_pred PASS | focal_zone PASS | Best state |
|---|---|---|---|---|---|
| **FNO_A** (analytical) | Phase 7-analytical L1 | 1.192 (job 205, 50 ep) | ✓ | n/a (analytical has no real focal physics) | Era 5 |
| **FNO_J** (j-Wave) | Phase 7c L1 | 1.05 (job 240) | ✓ (0.193 — false positive) | **FAIL** (E_focal 0.0487 below uniform) | Era 5; retrained Era 3 → still focal_zone-marginal |
| **FNO_F** (FEM-coupled) | Phase 6.6b → v1 → v2 → v2.5 → v3 | 1.94 (v3 ep14) | 0.857 (FAIL, at architectural ceiling) | n/a in the random-phase regime | Era 7; mode-scaling to 12×12×36 in Era 9 (job 866 queued) |

### Training-side lessons (each is its own "compound bug" the team had to debug to PASS)

| Lesson | Source | Memory file |
|---|---|---|
| LR=1e-3 collapses FNO_F to predict-zero on FEM data | v3 job 446 | `fno-lr-collapse.md` |
| LR=1e-4 + cosine T_max=50 overfits at ep14 on DDP (effective batch 8) | v3 job 520 | `fno-ddp-lr-overfit.md` |
| Smoke at T_max=N hides overfit dynamics that only appear at T_max ≫ N | v3 jobs 469 vs 520 | (same memory) |
| Container ships numpy 2.x but torch needs 1.x → force-reinstall 1.26.4 with --no-deps | FNO_J smokes 117–157 | `omniva-training-gotchas.md` |
| `drip-physics-core` pinned numpy<3 but works with 1.x — install with --no-deps | FNO_J smoke 157 | (same memory) |
| Per-epoch wall ≠ smoke-extrapolation — measure epoch-1 before extrapolating | FNO_J 183 timeout | (same memory) |
| FNO_F representational ceiling on cylindrical FEM is the 8×8×24 truncated Fourier basis — 83% loss before model | Era 8 alternate-hypothesis test | `fno-fourier-truncation-ceiling.md` |

### Where this stage is in the timeline doc
- Era 2 (Phase 6 FNO_F iterations), Era 5 (FNO_J/A smoke + real on cluster), Era 7 (v3 hyperparameter sweep), Era 9 (mode-scaling experiment, in flight).

---

## STAGE 3 — MID-TRAIN / DISAGREEMENT-WEIGHTED COMBINATION

**Standard ML**: additional training objectives, supervised fine-tuning, multi-task objectives.

**Our equivalent**: build the **disagreement matrix** between the three FNOs and use it as a calibrated uncertainty signal for an adversarial training objective on a combined teacher.

### The disagreement framework

| Pair | Measures | Result |
|---|---|---|
| FNO_A vs analytical | Noise floor (should be near zero — same physics) | **0.31% mean rel L²** across 50 configs |
| FNO_J vs analytical | Regime divergence (where j-Wave adds beyond analytical) | **133% rel L²** |
| FNO_F vs FNO_J L1 | Multi-physics signal beyond j-Wave | 1.071 (focal zone) |
| analytical vs FNO_F | Reference benchmark | 0.725 |
| analytical vs FNO_J L1 | Reference benchmark | 0.693 |
| analytical vs FNO_J mini | Reference benchmark | 0.998 |

**430× separation** between noise floor (0.31%) and regime divergence (133%) calibrates the signal-to-noise of the disagreement signal.

### The intended combine step (FNO_combined adversarial training)

- After all three FNOs pass their respective sanity gates, **build the pairwise disagreement matrix**.
- Each pairwise residual maps to a known *missing physics term*:
  - analytical-vs-j-Wave residual isolates *diffraction + reflection*.
  - j-Wave-vs-FEM residual isolates *coupled-physics terms (streaming, heat-dispersion)*.
- Use the disagreement matrix to **weight an adversarial loss** training a combined teacher `FNO_combined`.
- High-disagreement regions get more weight → the combined model learns to be best in the regions where the experts disagree most.

### Status of this stage
- **Receipts framework**: complete. `disagreement_analysis.py` + `research/DISAGREEMENT_ANALYSIS.md` + `research/DISAGREEMENT_FNO_J_VS_ANALYTICAL.md`.
- **Pairwise residuals**: computed for analytical, FNO_J, FNO_F.
- **`focal_zone_signal_quality.py` gate**: built as the *second-layer* gate during Era 3 after the disagreement matrix revealed FNO_J L1's wall-dominated false-positive PASS.
- **`FNO_combined` adversarial training**: scaffolded but NOT yet executed. Gated on FNO_F clearing its sanity gates (currently in Era 9 mode-scaling experiment).

### Where this stage is in the timeline doc
- Era 3 (pairwise disagreement script + focal-zone gate discovery), Era 4 (era-2 audit added 5 physics-accuracy metrics + worst-case bars).

---

## STAGE 4 — POST-TRAIN / DISTILLATION TO A REAL-TIME STUDENT

**Standard ML**: alignment (RLHF, DPO), distillation to smaller production model, quantization.

**Our equivalent**: distill the 118M-parameter `FNO_combined` teacher into a **small student model** that runs at video frame rate for closed-loop control.

### Distillation status
- **Student v1**: distilled from `FNO_J` directly (Phase 7a era, pre-FNO_combined).
  - **23,288× speedup over the teacher**.
  - **0.479 ms inference**.
  - This is the artifact already on disk.
- **Student v2** (planned): distilled from `FNO_combined` once mid-train completes.
- **Cloud sprint scaffolding** for distillation: complete (`ml_inverse/cloud_sprint/`). Pre-staged wheels, sbatch templates, idempotent venv setup all proven on FNO_J training.

### Why this matters
- Production inference needs ≤ 16 ms (60 fps) for the closed-loop droplet controller. Teacher at ~15 ms forward + autograd through it for inverse = too slow.
- Student at 0.479 ms gives 33× margin → can take 30+ gradient steps per control frame inside the inverse-design loop.

### Where this stage is in the timeline doc
- Era 1 (Phase 5 student v1 distillation, mentioned in STATUS.md "Layer 4 partial distillation, cloud sprint $5.01").

---

## STAGE 5 — DEPLOYMENT / IMPLANT INTO SIM AND HARDWARE

**Standard ML**: containerize, deploy to inference servers, A/B test, monitor.

**Our equivalent**: TWO parallel deployment surfaces.

### 5a — Simulation deployment (live now)

- **Plotly Dash dashboard** (`simulations/dashboard/dash_app_v6.py` + 18 page modules).
- **Block diagram editor**: drag-drop control blocks (PID, transfer fn, **ml_model**, trajectory_optimizer) → compiled via Kahn topological sort + Tustin discretization → executable `CompiledController`.
- **`ml_model` block**: drops any trained FNO checkpoint into a closed-loop simulation as a block, with same execution interface as classical control blocks. Hybrid PID+FNO controllers are first-class.
- **Cloudflare R2 model registry**: dashboard reads `MODEL_MANIFEST` env var at startup, fetches checkpoints by URL, verifies sha256, registers. 16 models currently registered as of 2026-06-03.
- **Railway production deployment**: live now.
- **`model_downloader.py`**: handles cold-start fetch + caching to volume mount.

### 5b — Hardware deployment (gated on L1 prototype, Nov 2026)

- **Phase command wire protocol**: complete. `simulations/integration/phase_protocol/` — 39/39 tests. JSON encoding (723 B for focal trap), binary encoding stubbed (558 B). **8 MOLLY_DECIDES items** in handoff to EE board owner. Zero new deps.
- **Sim-to-real comparison + telemetry**: complete. `simulations/integration/sim_to_real/` — 25/25 tests. 5-band residual classifier; JAX-autograd calibration loop; mock FEM for Stage 0 + Phase 5 swap-in is one-line. End-to-end demo: 5 m/s drift recovered at 0.008% error.

### Where this stage is in the timeline doc
- Era 4 (sim-to-real integration packages landed in Pass 13b deferred-findings cleanup), Era 8 (R2 upload + MODEL_MANIFEST update).

---

## STAGE 6 — ONLINE FEEDBACK / TELEMETRY → DATA LOOP (Future)

**Standard ML**: production logs → retraining corpus → continual learning, RLHF rounds.

**Our equivalent**: machine telemetry + acoustic sensors → calibration loop → updated training dataset → re-distill student.

### The intended feedback loop

```
HARDWARE (running machine)
  → in-situ acoustic sensors (FFT / wavelet pipeline)
  → 5-band residual classifier (sim-to-real telemetry, deployed)
  → JAX-autograd calibration loop adjusts FEM parameters to match real
  → updated FEM forward = updated training data
  → retrain student head (or full FNO_combined) on residuals
  → ship new student via MODEL_MANIFEST update (already wired)
```

### Status by stage of the feedback loop

| Component | Status |
|---|---|
| Telemetry classifier (5-band residual) | ✅ deployed |
| Calibration loop (JAX-autograd through mock FEM) | ✅ deployed |
| Swap mock FEM → real Phase 5 backend | one-line change, ready |
| Acoustic sensor pipeline integration | Drip-side hardware, gated on L1 |
| Continual retraining of student | Cloud sprint pattern ready (used for v1 student) |
| MODEL_MANIFEST hot-swap | Already wired; dashboard re-fetches on env-var change |

### Where this stage is in the timeline doc
- Era 4 (sim-to-real package), Era 1 (Phase 5 architectural-bet drop-in backend that makes the FEM swap-in trivial).

---

## VISUAL SUMMARY — current state across the pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE                       STATE              WHAT WE HAVE             │
├─────────────────────────────────────────────────────────────────────────┤
│  DATA                        ✅ DONE             4 grid resolutions × 3   │
│  (3 physics tracks)                              physics tracks, 7000-   │
│                                                  config v3 thermal-aware │
│                                                  is current state of art │
│                                                                          │
│  PRETRAIN                    ✅ DONE             Pattern A + v2.5 prior, │
│  (arch + prior design)                           cosine LR, H¹ loss,     │
│                                                  thermal-aware 121-dim   │
│                                                                          │
│  TRAIN (3 surrogates)        🟡 IN PROGRESS     FNO_A ✅ val 1.19        │
│                                                  FNO_J ✅ val 1.05       │
│                                                  FNO_F ⏳ at architec-   │
│                                                       tural ceiling      │
│                                                       (job 866 testing   │
│                                                       12×12×36 modes)    │
│                                                                          │
│  MID-TRAIN                   ⏸ GATED            Disagreement matrix     │
│  (combine via                ON FNO_F            framework ✅            │
│   disagreement)              READINESS           Pairwise residuals ✅   │
│                                                  FNO_combined NOT FIRED  │
│                                                                          │
│  POST-TRAIN                  🟡 PARTIAL          Student v1 (from FNO_J) │
│  (distillation)                                  ✅ 23,288× speedup     │
│                                                  Student v2 (from        │
│                                                  combined) gated        │
│                                                                          │
│  DEPLOY (sim)                ✅ LIVE             Dashboard + R2 +        │
│                                                  Railway + block         │
│                                                  diagram editor + 16     │
│                                                  models in manifest      │
│                                                                          │
│  DEPLOY (hardware)           ⏸ GATED            Phase protocol ✅       │
│                              ON L1 PROTO         Sim-to-real ✅          │
│                              NOV 2026                                    │
│                                                                          │
│  ONLINE FEEDBACK             ⏸ GATED            Calibration loop ✅     │
│                              ON HARDWARE         Classifier ✅           │
│                                                  Hot-swap wired ✅       │
│                                                  No live data yet        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## How the CS 153 work specifically fits this pipeline

CS 153 timeline (the 1-month sprint): the project focuses on **Stages 0 → 2** (the dataset gen + train-3-FNOs portion). Stages 3 → 6 are scaffolded but executed beyond the class window.

- **Class deliverable scope**: Stages 0–2, the three forward surrogates each clearing their respective sanity gates with receipts.
- **Class deliverable secondary**: Stage 3 disagreement framework calibrated, ready for FNO_combined firing.
- **Class deliverable tertiary**: Stage 4 student v1 already distilled (from FNO_J) as proof-of-concept for the 23,288× speedup.
- **Beyond class**: Stages 5b + 6 are the hardware-integration arc, gated on L1 prototype.

**The story arc for the presentation** is therefore:
1. Why we need this pipeline (PDE-constrained control needs real-time inverse → surrogates).
2. The clean canonical mapping (gen → train → combine → distill → deploy → feedback).
3. Our state at each stage (most complete: gen, train; in progress: train FNO_F; ready: combine; demoed: distill; live: sim deploy; staged: hardware deploy; future: feedback).
4. The reusable methodology lessons (data dominates network, representational vs optimization ceilings, sanity-gate scope-of-validity, agent-as-MLOps-operator).

---

## Cross-reference table back to the timeline doc

| Pipeline stage | Timeline doc eras |
|---|---|
| DATA | Era 1 (Phase 1), Era 5 (Phase 7c gen, Phase 7d-cluster failure → Mac fallback), Era 6 (v2 → v2.5 → v3) |
| PRETRAIN (arch + prior) | Era 1 (Phase 2 v1→v2.1, FNO_ARCH_SWEEP), Era 2 (Phase 6.3 prior-scale) |
| TRAIN | Era 2 (Phase 6 FNO_F iterations), Era 5 (FNO_J + FNO_A on cluster), Era 7 (v3 LR sweep), Era 9 (mode scaling) |
| MID-TRAIN | Era 3 (disagreement framework + focal-zone gate), Era 4 (additional eval metrics) |
| POST-TRAIN | Era 1 (student v1 distillation), Era 8 (R2 + MODEL_MANIFEST) |
| DEPLOY (sim) | Era 4 (sim-to-real package), Era 8 (R2 + Railway manifest update) |
| DEPLOY (hardware) | Era 1 (phase_protocol, sim_to_real integration) |
| ONLINE FEEDBACK | Era 1 (sim-to-real telemetry classifier + calibration loop) |
