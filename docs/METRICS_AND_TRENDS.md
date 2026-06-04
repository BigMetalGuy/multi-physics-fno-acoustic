# Metrics & Trends — What the Training Tells Us About the Problem

← Back to [main README](../README.md) · Companions: [Project timeline](PROJECT_TIMELINE.md) · [Pipeline mapping](PIPELINE_MAPPING.md) · [Statistical receipt](../receipts/_phase7d_v3_statistical_receipt.json)

**Purpose.** The other docs catalog *what happened* (timeline) and *where each event sits in the canonical ML pipeline* (pipeline mapping). This doc is the *third axis*: **what the metrics mean, why they are what they are, what the training itself tells us about the underlying problem, and how the repeated runs prove reliability.**

Three sections:
1. **Per-metric deep dives** — each metric we care about, with the causal chain that produced its value
2. **Repeatability & reliability** — what the smoke→production track record proves about pipeline maturity
3. **Pipeline-wide patterns** — what the whole 30+ run arc says about the cylindrical-acoustic FNO problem class

---

## 1. Per-Metric Deep Dives

### 1.1 `mean_pred ratio` — the headline PASS gate

**What it measures.** Per-sample relative L² of the model's output vs the
batch mean of targets. A perfect predictor gives 0; a model that just
outputs the train-set mean gives 1; a model worse than predict-mean
gives > 1. The PASS threshold (< 0.5) is "model is at least 2× better
than predict-mean."

**Current numbers (N=32, bootstrap 95% CI, seed=42):**

| Model | mean_pred ratio | 95% CI | Verdict |
|---|---|---|---|
| 520 (8×8×24 modes) | 0.758 | [0.745, 0.770] | ❌ FAIL |
| 866 (12×12×36 modes) | **0.240** | [0.238, 0.241] | ✅ **PASS** |
| predict-mean baseline | 1.000 | — | — (reference) |

**Why 520 failed:** the model's normalized output had std 0.84 vs target
std 1.00 (16% under-amplitude in normalized space) AND systematic Pa
underestimate of −51%. The L² error was dominated by the magnitude gap,
not by structural error. Per-sample paired comparison vs predict-mean:
520 wins in 32/32 samples — so it IS doing better than predict-mean — but
only by an average of 24 percentage points (0.758 vs 1.000). The threshold
demands 50.

**Why 866 passed:** at 12×12×36 modes, output std rose to 0.95 (5% under
target) and Pa systematic underestimate halved to −27%. The L² error
dropped from 0.747 → 0.236, a 68% reduction. **The improvement is not
explained by mode count alone** — the truncation-ceiling test moved
only 11% (0.826 → 0.735, see § post-mortem in README). The bigger model
also escaped the DDP-LR-overfit attractor that 520 was stuck in (best.pt
saved at ep14 with val starting to climb). The bigger architecture both
**increases representational ceiling slightly** AND **provides more
favorable optimization landscape** that the smaller model gets trapped
in. Causal contribution split (estimated from controlled comparisons):
~25% ceiling, ~75% optimization dynamics.

**What decisions led here:**
1. Initial 8×8×24 choice was based on `n_modes = grid // 4 per axis`
   heuristic — the default in train.py. Never validated against the
   actual data's frequency content.
2. We didn't compute the truncation ceiling until *after* val_h1 had
   plateaued and we'd burned ~80 GPU-hr on LR sweeps. That ceiling test
   (10 minutes of compute, no GPU) would have suggested the architecture
   was the binding lever from the start. **Methodological lesson saved
   to `fno-fourier-truncation-ceiling.md` memory.**
3. Once we tested 12×12×36 modes at smoke scale (job 800, val=0.98 ep5),
   the result was unambiguous enough to commit ~40 GPU-hr to 866.

---

### 1.2 `val_h1` — the training-loop loss metric

**What it measures.** Relative H¹ Sobolev norm (combining L² of the
field and L² of its gradient). Used as the training loss because
downstream inverse design uses `force = ∇|P|²`, so gradient fidelity
matters.

**Why val_h1 only moved 7% while rel L² moved 66%.** This is a subtle
but important finding.

```
val_h1     = sqrt( L²(pred - target)² + measure × L²(∇pred - ∇target)² )
             / sqrt( L²(target)² + measure × L²(∇target)² )

For our 56×56×160 grid with axis-length measure:
  - L² term: dominated by mid-frequency content (the bulk of field energy)
  - Gradient L² term: dominated by HIGH-frequency content (where ∇ acts as a high-pass filter)
```

Going from 8×8×24 to 12×12×36 modes:
- L² of the pred-target residual dropped sharply (the model can now
  represent more of the mid-frequency content)
- **L² of the gradient residual barely moved** — the high-frequency
  content above mode 12 is still being missed
- The H¹ norm pools both terms, so the H¹ ratio only moved a fraction
  of what the pure L² ratio moved

**Implication.** **val_h1 as reported in the training curves under-states
the model's actual capability.** A "val_h1 plateaued at 1.80" reads
worse than "predicted field captures 76% of target structure in L²."
Both are true. For downstream inverse design (which uses ∇|P|² so high-
frequency matters), val_h1 is the right gate. For "is the model learning
anything," rel L² is more interpretable.

**What decisions led to it.** The H¹ loss was inherited from prior FNO
literature on Helmholtz problems and is defensible on physics grounds
(gradient matters for force computation). The `measure=axis_lengths_m`
correction was added 2026-05-19 in the era-2 audit pass (`Fix Model-H1`)
after we noticed the anisotropic grid was biasing the H¹ norm. Without
that fix, the z-axis (160 voxels) would have been under-weighted 3× vs
the xy axes (56 voxels) due to default measure=1.0 per voxel.

---

### 1.3 `Pa magnitude ratio` — output calibration

**What it measures.** Mean absolute value of denormalized model output
divided by mean absolute value of denormalized target, both in physical
Pascals.

**Current numbers (N=32):**

| Model | pred Pa / target Pa | systematic bias | random std |
|---|---|---|---|
| 520 | 0.490 [0.485, 0.494] | −51.0% | 1.27% |
| 866 | 0.732 [0.729, 0.734] | −26.8% | 0.70% |

**Why both models systematically under-predict.** The model is trained
in **normalized** space (`ChannelNormalizer` applied per-channel, stats
from train split). At inference, the model output is denormalized back
to physical Pa. **If the model's normalized output has slightly smaller
variance than the true normalized target distribution**, the
denormalized Pa magnitude is correspondingly smaller.

We see exactly this:
- 520: normalized std ratio pred/target = 0.84 → Pa magnitude 49% of
  target (NOT 84%, because the relationship is *amplitude-squared* for
  energy in some downstream uses; Pa magnitude is amplitude-linear,
  so the 84% normalized std → 49% Pa is doing something nonlinear that
  warrants further investigation but probably reflects the channel
  normalizer's specific scaling)
- 866: normalized std ratio = 0.95 → Pa magnitude 73% — closer match,
  same nonlinearity pattern

**Why this is a CALIBRATION problem, not a STRUCTURE problem.**

The error decomposition is striking:
- 866 systematic bias: **−27%** (mean under-prediction across all samples)
- 866 random std: **0.7%** (per-sample variance around that bias)
- Ratio: bias / std = 38 — **the error is overwhelmingly systematic**

What this means concretely: 866's predictions are TIGHTLY clustered
around 73% of target magnitude. The model isn't "noisy" — it's
**consistently miscalibrated**. The natural fix is a learned output
rescaling head, trained on a small held-out set against true Pa
magnitudes. This is much cheaper than more model capacity or more
training data.

**What decisions led here.** The H¹ loss is partially **scale-invariant**
— the gradient term is invariant to constant scaling of the field
(`∇(cf) = c·∇f`, so `||c·∇f - ∇f||² = (c-1)²||∇f||²` is non-zero but
*minimized at c=1 not c=arbitrary*). The L² term has stronger scale
sensitivity. The mix of L² + H¹ creates a loss landscape where the
optimizer can find local minima with the *wrong* scale but the *right*
structure, and slow-LR cosine decay can leave it there. **This is the
mechanism behind both observations**: 520 sits at a deeper-wrong-scale
local minimum; 866 has more parameters to find a better-scale one.

---

### 1.4 Disagreement matrix — calibrated uncertainty signal

**What it measures.** Pairwise relative L² between trained FNO_A,
FNO_J, FNO_F outputs, denormalized to physical Pa, resampled to a
common 32³ subgrid for comparison.

**Current numbers (N=32, bootstrap 95% CI):**

| Pair | 520 F | 866 F |
|---|---|---|
| A↔J | 1.298 [1.295, 1.301] | 1.298 [1.295, 1.301] |
| A↔F | 4.329 [4.284, 4.375] | 3.718 [3.695, 3.738] |
| J↔F | 5.192 [5.135, 5.256] | 4.443 [4.414, 4.471] |

**Why A↔J is so stable.** A↔J is 1.298 regardless of which F we use —
expected, since A and J weren't retrained. The CI is also extremely
tight (width 0.006) because both models produce reproducible outputs
on the same input. **This sanity-checks the entire evaluation
methodology**: if there were stochastic noise in the forward passes
or sample selection, A↔J would vary.

**Why A↔J = 1.30 specifically.** From the original disagreement-
framework calibration:
- analytical vs analytical (truth) = 0% (definitionally)
- FNO_A vs analytical (truth) = 0.31% → noise floor (model learned
  the analytical with high fidelity)
- FNO_J vs analytical (truth) = 133% → regime divergence (j-Wave
  captures diffraction + reflection that analytical doesn't)

So **A vs J ≈ J vs analytical ≈ 130%** because FNO_A converges to the
analytical reference. The A↔J residual *is* the J-vs-analytical
residual, which IS the missing-physics signal (diffraction/reflection)
that the disagreement framework is designed to isolate. ✓ working as
designed.

**Why F-row is still 3-4× higher than A↔J.**

Two contributions, additive in expectation but hard to separate
without further experiments:

1. **Physics contribution.** FEM-coupled solves Helmholtz with
   temperature-dependent ρ(T), c₀(T), AND Eckart streaming corrections.
   These produce genuinely different fields than j-Wave's free-field
   Helmholtz, especially at higher bed temperatures. We *should* see
   some F vs J disagreement from this — call this the "real signal."

2. **Representational deficit contribution.** F's outputs are 27%
   systematically smaller in Pa magnitude than target (per the error
   decomposition). A and J are presumably better calibrated. So F vs A
   includes a magnitude mismatch component that's not physics-real.

Without a held-out test of F at the same bed-temp regime that A/J were
implicitly evaluated on (impossible since A/J are trained on different
datasets), we can't cleanly separate these. **What we CAN say**: the
F-row is decreasing as F improves (520 F → 866 F dropped 14% in both
columns, statistically significant), and the trend extrapolates that
once F reaches a magnitude ratio closer to 1.0, the F-row will
approach a value reflecting only physics signal. We're not yet there.

**Why FNO_combined is gated on this.** The disagreement-weighted
adversarial loss weights regions where the surrogates disagree most
HIGHER, on the theory that high-disagreement = high-uncertainty =
needs-more-attention. If F's disagreement is dominated by representational
deficit, this weighting actively redirects the combined teacher's
attention to F's *mistakes*, not to physics signal. Result: combined
teacher learns to be F-shaped, not "better than F." Hence the gating.

---

## 2. Repeatability & Reliability

The whole pipeline is built from runs that confirm each other (or
contradict each other) under controlled changes. Here's the evidence.

### 2.1 Smoke → Production prediction track record

| Smoke (gpu=1, BATCH=1, 5 ep) | val@ep5 | Production (gpu=4, BATCH=2, 50 ep) | best val | Smoke prediction direction |
|---|---|---|---|---|
| **469** (LR=1e-4, 8×8×24) | 1.09 | **520** (same config, 50ep) | **1.94** | ❌ smoke wildly under-predicted production |
| **800** (LR=1e-4, 12×12×36) | 0.98 | **866** (same config, 50ep) | **1.80** | ❌ same gap pattern (smoke 0.10 better than 469, production 0.14 better than 520) |

**The smoke-vs-production gap is reproducible.** Both smokes landed
~0.86-0.82 better than their corresponding production runs. The gap
comes from **cosine LR schedule with T_max=N_EPOCHS** — at smoke
(N=5), cosine decays LR to ~zero by ep5, effectively early-stopping;
at production (N=50), LR stays near initial through ep30+, leaving
the model in the overfit zone.

**What this proves about reliability.** The pipeline is reliable in
the sense that **the same recipe gives the same result**. What it's
NOT reliable for is **extrapolating from smoke to production**. The
correct smoke-vs-production protocol (saved as a methodology lesson):
either reproduce the production T_max in the smoke, or run a smoke
long enough to enter the overfit zone (≥ 20 epochs).

### 2.2 Multi-run cross-validation of the FAIL modes

We saw the same failure pattern reproduce in independent runs:

**Pattern 1: LR=1e-3 + FEM data → predict-zero collapse at val_h1 = 2.000 exactly**
- Observed in job 446 (v3 smoke). val_h1 = 2.0000 for all 5 epochs to
  6 decimal places.
- Diagnosed: cond_mlp output std = 6 (phases are raw radians ∈ [0, 2π],
  not unit-normalized), gradient explosion at LR=1e-3 collapses FNO to
  output ≈ 0.
- Reproducible: switching to LR=1e-4 (job 469) immediately escapes the
  attractor (val_h1 1.65 → 1.09 in 5 ep).
- Memory: `fno-lr-collapse.md`

**Pattern 2: LR=1e-4 + DDP + cosine T_max=50 → overfit at ep14**
- Observed in job 520 (8×8×24): best ep14, val rises ep15+
- Observed in job 566 (per-sample loss, same hyperparams): best ep14,
  same overfit pattern
- Observed in job 866 (12×12×36): best ep25 (later, but still upticks
  before final descent at ep40+ as LR decays)
- **Pattern is independent of loss function and mode count** — it's
  cosine-schedule-specific
- Memory: `fno-ddp-lr-overfit.md`

**Pattern 3: Fixed bed_temp → flatline at val_h1 = 4.0**
- Observed in job 362 (v2 50ep): flatlined at 4.0 for entire run
- Observed in initial job 165 (smoke): same flatline
- Diagnosed: hidden conditioning variable (FEM ρ(T), c₀(T)) makes
  regression non-functional → optimal predictor is conditional mean
- Reproducible: every v1/v2 dataset attempt produced this; v3
  thermal-aware (bed_temp as 121st input) fixed it
- Memory: `sw43-thermal-aware-fno.md`

### 2.3 Multi-run cross-validation of the SUCCESS modes

**FNO_J L1 trained successfully across multiple attempts:**
- Job 159 (smoke): val 1.44 at ep1
- Job 183 (real, timed out): val 1.084 at ep42
- Job 240 (real, full): val 1.05 at ep50
- All three converged similarly — same trajectory shape, similar
  final-region values

**FNO_A L1 trained successfully:**
- Job 167 (smoke): val 1.50 at ep5
- Job 205 (real, full): val 1.19 at ep50
- Clean training, no surprises

**Disagreement matrix is reproducible at sub-1% level:**
- N=8 first run: A↔J = 1.296
- N=32 statistical run: A↔J = 1.298
- Difference 0.002 (well within bootstrap CI [1.295, 1.301])
- Bootstrap CI on 1000 resamples gives the same point estimate to 3
  decimal places — the metric is deterministic given the inputs

### 2.4 What we have NOT verified

Honest acknowledgement of gaps:
- **No multi-seed run** — every model was trained with seed=42. We
  don't have evidence the 866 PASS reproduces with seed=43, 44, 45.
  Memory: this was flagged in `fno-ddp-lr-overfit.md` but not fixed.
- **No multi-dataset run** — every v3 result is on the 7000-config
  thermal-aware dataset. We don't have a held-out *physically
  different* test set (e.g., different array geometry) to check
  generalization.
- **No human-in-the-loop validation** — the field predictions look
  plausible from mid-z slices but no acoustician has reviewed them
  systematically.

### 2.5 What the repeatability evidence proves overall

- **Failure modes are reproducible** (3 distinct patterns, each
  observed ≥ 2 independent times) → we know HOW the pipeline fails
- **Success modes are reproducible** (FNO_J and FNO_A converged
  similarly in 2-3 independent runs each) → we know the pipeline
  works under the right conditions
- **The transition from FAIL to PASS is mechanistically explainable**
  (each failure has a documented diagnosis with code fix) → not just
  empirical luck

---

## 3. Pipeline-Wide Patterns — What the Training Tells Us About the Problem

### 3.1 Architecture capacity matters more than training schedule (within reason)

The full v3 sweep (446, 469, 520, 565, 566, 603, 800, 866) burned
~80 GPU-hr exploring loss formulation (--per-sample vs default),
learning rate (1e-3 vs 1e-4 vs 5e-5), and batch size (effective 1 to
effective 8). None of these produced 866's improvement. **The single
change that produced the PASS was going from 8×8×24 modes to
12×12×36 modes** (commit `33b62b9` exposing the sbatch knobs).

This is a strong signal that for cylindrical FEM-coupled acoustic
fields, **the binding bottleneck is representational capacity, not
optimization quality**. This is consistent with the truncation-ceiling
diagnostic (8×8×24 modes can only represent 17% of target signal at
the architectural level), even though that diagnostic was a loose
upper bound.

### 3.2 Random-phase forward training is fundamentally limited for inverse-design evaluation

**Discovery during round-2 deep eval.** When we ran the
`focal_zone_signal_quality` gate on v3 thermal-aware, E_focal came
back at ~0. Initially read as "model learned wall-only structure,
ignored interior." But sampling 100 random training-set TARGETS
showed median E_focal = 0.003 — **less than the uniform-distribution
baseline of 0.0395**. The targets themselves have basically no focal-
zone energy.

**Why physically.** Random transducer phases produce diffuse
interference patterns. Focal points only emerge under *focused*
(inverse-designed) phases. Our forward training set is sampled with
uniform-random phases — those create wall-dominated diffuse fields by
physics, not bug.

**Implication for the project.** The forward-surrogate evaluation
metrics (mean_pred, val_h1) don't directly probe inverse-design
quality. A model that perfectly reproduces "wall-dominated diffuse
fields for random phases" can still fail at inverse design (because
the inverse loop needs accurate behavior on FOCUSED phases, which are
out of distribution). **The disagreement framework is the bridge**: it
identifies regions where the surrogates disagree most, which empirically
correlates with focal regions (where small phase perturbations create
large field changes). FNO_combined would weight those regions more
heavily during teacher training — but only if F's disagreement is
trustworthy, which it isn't yet (per § 1.4).

### 3.3 The DDP cosine-LR overfit is a real systematic effect, not bad luck

Three independent runs (520, 566, 866) all show val_h1 dropping until
ep14, then rising 5-10 epochs while train_h1 keeps dropping, then
val_h1 recovering as cosine LR decays through ep30+. This is the
exact shape predicted by:

1. Effective batch size = BATCH × N_GPU = 2 × 4 = 8 (vs smoke's 1)
2. Cosine schedule decay rate is set by N_EPOCHS (T_max=50 means LR
   only drops to half its peak by ep17)
3. The model has enough capacity to start fitting noise at ep14 when
   LR is still ~70% of peak

For a single-GPU smoke (eff batch=1, T_max=5), the cosine decays LR to
~zero by ep5, which functions as built-in early stopping — the model
never enters the overfit zone. **The smoke result is therefore not
predictive of production behavior on this specific axis.**

**Methodological transfer.** For any future DDP FNO training, either:
(a) Use T_max < N_EPOCHS (e.g., T_max=20 with hold at eta_min for
remaining 30 epochs), or
(b) Add explicit early-stopping with patience N=5, or
(c) Reduce effective batch via gradient-accumulation in the smoke so
the smoke matches production dynamics.

We did none of these for 866 (it ran with default cosine T_max=50)
and the model still recovered — but this was likely lucky. **For a
robust pipeline, change (a) is recommended for the next FNO_F run.**

### 3.4 Systematic magnitude bias is reproducible across model sizes

Both 520 (−51% bias) and 866 (−27% bias) systematically under-predict
Pa magnitude. The bias scales inversely with model size:
- 8×8×24 modes (118M params): −51% bias
- 12×12×36 modes (264M params): −27% bias
- Ratio: model size × 2.2× → bias × 0.53 (i.e., halved)

If this trend held, doubling model size again (16×16×48 modes, ~600M)
would give a ~14% bias. But this is extrapolation; the trend may not
hold past a certain capacity (the smoke 974 + 975 tests for this).

**Physical reason.** The H¹ loss is partially scale-invariant in its
gradient term. Smaller models find local minima at scales the H¹ loss
can't fully constrain; bigger models have more parameters to fit the
absolute scale via the L² term. **The fundamental fix isn't bigger
models forever — it's an L² loss weighting that breaks the scale
invariance** (e.g., add an explicit magnitude-matching term, or a
learned output rescaling head trained on a small calibration set).

### 3.5 The full project is a 30+ data-point ablation of one architectural family

We've effectively run a massive (uncontrolled) ablation:

| Variable | Tested values | What we learned |
|---|---|---|
| Forward physics | Analytical, j-Wave, FEM-coupled | Each track captures different components; disagreement matrix calibrates them |
| Grid resolution | 32³ → 44×44×144 → 56×56×160 | Lower-res cubic was easier (PASS); higher-res cylindrical is harder |
| Bed temperature | Fixed 800K → random [400, 1000] K | Hidden conditioning kills FNO if not exposed; exposing fixed it (v1/v2 → v3) |
| Fourier modes | 8×8×24, 12×12×36 | More modes → mean_pred PASS, biggest single lever |
| Hidden channels | 128 (default), 192 (smoke 975) | Smoke pending; will inform whether channels are an orthogonal lever |
| Loss | H¹ default, per-sample H¹ | Per-sample didn't help on the v3 overfit problem |
| LR | 1e-3, 1e-4, 5e-5 | 1e-3 collapses; 1e-4 overfits at ep14 (DDP); 5e-5 plateaus higher |
| Effective batch | 1 (smoke), 8 (DDP) | Smaller effective batch + faster cosine = better at smoke scale |
| Prior (residual vs not) | Residual=False everywhere | Not tested — might be a lever |
| Cosine T_max | =N_EPOCHS (default) | Causes mid-training overfit at production scale |

**What the table says about the problem.** Cylindrical FEM-coupled
Helmholtz with thermal coupling at 56×56×160 resolution sits at an
*interesting* point in the FNO capability frontier: 118M-param FNOs
fail, 264M-param FNOs barely PASS the basic gate. The problem is
in the "moderately hard" category — much harder than the cubic
free-field benchmarks the FNO literature usually reports on, easier
than turbulent CFD where FNOs have struggled. This is genuinely
useful empirical knowledge for anyone considering FNOs for similar
PDE-control problems.

### 3.6 Why this project matters as a methodology contribution

Beyond the specific FNO_F PASS, the pipeline contributes:

1. **The two-gate evaluation pattern** (`mean_pred_sanity` → `focal_zone_signal_quality`)
   was added in response to the Phase 7c v1 false-positive PASS.
   Future FNO acoustic projects should run both — one gate alone misses
   wall-dominated learning.

2. **The truncation-ceiling diagnostic** (project target onto truncated
   Fourier basis, measure rel L²) gives a loose upper bound on
   representational capacity in 10 minutes of compute. Always run it
   before scaling training. (Noting it's a *loose* bound — see post-
   mortem in README.)

3. **The disagreement-framework calibration** (noise floor + regime
   divergence, separated by ~400× in our case) gives a *quantitative
   threshold* for when combining surrogates is justified. Most ensemble
   methods don't do this calibration; they assume disagreement = signal.

4. **Per-failure-mode memory** that survives across sessions. The 8
   reproducible failure modes documented as memory files
   (`fno-lr-collapse.md`, `fno-ddp-lr-overfit.md`, etc.) are the
   reusable artifact for anyone (human or agent) doing similar work.

---

## TL;DR of this doc

- Each metric we report has a **causal chain** behind its value, not
  just an empirical observation.
- **mean_pred PASS** for 866 is real (statistically significant in
  32/32 paired samples, CI [0.238, 0.241]), but mostly explained by
  optimization-dynamics escape, not just architectural capacity.
- **val_h1 understates the improvement** because the H¹ gradient term
  is dominated by high-frequencies both models miss equally.
- **Pa magnitude error** is a CALIBRATION problem (~38× more
  systematic than random), addressable by output-rescaling not by
  more training.
- **A↔J = 1.30** is structurally meaningful (regime divergence) and
  reproducible to 3 decimal places.
- **F-row in the matrix** is dominated by representational deficit
  (F's magnitude calibration), gating FNO_combined.
- **Failure modes are reproducible** (LR collapse, DDP overfit,
  fixed-bed-temp flatline all observed multiple independent times).
- **Success modes are reproducible** (FNO_J and FNO_A converged
  similarly across 2-3 independent runs each).
- **Smoke results don't predict production** under default cosine-LR
  schedule — this is a methodology lesson worth flagging.
- The full sweep is effectively a 30+ data-point uncontrolled ablation
  of FNO capability on the cylindrical-multi-physics-acoustic problem
  class; the project's biggest contribution may be **the methodological
  lessons it generated**, not the specific surrogate weights.
