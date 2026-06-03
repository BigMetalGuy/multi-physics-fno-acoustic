# Multi-Physics FNO Surrogates for Cylindrical Acoustic Inverse Design

Training cross-physics Fourier Neural Operator (FNO) surrogates for the
40 kHz acoustic forward problem in cylindrical chambers, then using
their pairwise disagreement as a calibrated uncertainty signal for
inverse design and distillation.

The cylindrical regime with hard reflective walls and anisotropic grids
is under-explored relative to the cubic free-field domains where most
FNO acoustic literature lives. Applications include droplet manipulation
for additive manufacturing, HIFU thermal focusing, acoustic tweezer
trapping in microfluidic cylinders, and ultrasonic NDT.

## Architecture

Three forward physics tracks, each producing a learned surrogate:

| Surrogate | Physics | Captures | Misses |
|---|---|---|---|
| **FNO_A** | Analytical 1/r superposition | Free-field interference | Diffraction, reflection, coupling |
| **FNO_J** | j-Wave spectral Helmholtz | Diffraction + reflection in free field | Streaming, heat-coupled dispersion |
| **FNO_F** | FEM-coupled (Helmholtz + heat + Eckart streaming) | Full multi-physics | — |

After all three pass a mean-prediction sanity gate, build the pairwise
disagreement matrix. Each pairwise residual maps to a known missing
physics term — analytical-vs-j-Wave isolates diffraction, j-Wave-vs-FEM
isolates the coupled-physics terms. Use the matrix to weight an
adversarial loss training a combined teacher (`FNO_combined`), then
distill a student model for real-time inference.

## Current state

| Phase | Model | Geometry | mean_pred / val_h1 | Verdict |
|---|---|---|---|---|
| 6.6b | FNO_F | 32³ cubic | 0.144 | **PASS** |
| 7a   | FNO_J | 32×32×96 mini-array | 0.094 | **PASS** |
| 7c   | FNO_J | 44×44×144 L1 cylinder | 0.193 | **PASS** |
| —    | student v1 (distilled from FNO_J) | — | — | **23,288× speedup, 0.479 ms inference** |
| 7d v1/v2 | FNO_F (fixed bed_temp) | 56×56×160 L1 cylinder | val_h1 ≈ 4.0 | **FLATLINE** — diagnosed as hidden-conditioning failure |
| 7d v3 *(in progress)* | FNO_F thermal-aware | 56×56×160 L1 cylinder | smoke val_h1 1.65→1.09 in 5 ep | descending; 50-ep real underway |

All PASS receipts (mean_pred.json, disagreement images, training logs)
live in `receipts/`.

### The v2 → v3 thermal-aware extension

Phase 7d's first FNO_F attempts on the L1 cylinder dataset (5000 configs,
bed_temp fixed at 800 K) **never trained**: val_h1 sat at exactly 4.0 for
50 epochs. Voxel CV was only 21%. We doubled it to 43% by re-generating
with per-config random bed_temp ∈ [400, 1000] K (the v2.5 dataset) — but
val_h1 stayed at exactly 4.0 again.

Diagnosis: bed_temp is a **hidden conditioning variable**. The FEM
forward solves Helmholtz with temperature-dependent density ρ(T) and
speed of sound c₀(T); varying bed_temp produces different fields for the
same transducer phases. Without bed_temp as a model input, the same
phase vector maps to multiple correct targets — a non-functional regression
task. The optimal predictor under L² loss is the conditional mean, which
for this dataset happens to be ≈ zero. Val_h1 = 4.0 = the predict-zero
ratio. The model wasn't failing — it was correctly predicting the conditional
mean of an ill-posed regression.

**v3 fix.** Pack `bed_temp_K` as the 121st input channel; reshape the
conditioning MLP to ingest the full vector; carry thermal context all the
way into the FNO's spatial conditioning. Gen pipeline writes `bed_temp_K`
as a first-class HDF5 column at generation time. 7000-config dataset.

**The MLOps wrinkle**: v3 with LR=1e-3 also flatlined, but at a different
constant (val_h1 = 2.000 exactly for 5 epochs). A diagnostic forward pass
on real training samples revealed the model output had collapsed to
`mean|.| = 0.003` against a unit-norm target — a degenerate predict-zero
attractor. Cause: conditioning MLP output had std ≈ 6 (phases are raw
radians ∈ [0, 2π], not unit-normalized), gradient explosion at LR=1e-3
crushed the FNO into the zero attractor. LR=1e-4 stays in the descent
basin: val_h1 1.65 → 1.09 in 5 epochs. 50-ep real fired with LR=1e-4.

### Phase 7d post-mortem: representational ceiling, not training ceiling

Three 50-epoch runs (520, 603) plus four hyperparameter smokes (446, 469,
565, 566) converged on **val_h1 ≈ 1.94** as an apparent floor. The natural
read was "model is overfitting" or "needs more compute." Both wrong.

A direct test of the architecture's representational capacity revealed the
actual ceiling: **projecting the target fields onto the FNO's 8×8×24
truncated Fourier basis recovers only ~17% of the target signal**
(relative L² = 0.83 between target and truncated-target). That's the
architectural floor before any model is involved. The trained model's
relative L² = 0.73 sits at ~70% of that representable maximum.

Independently, when running the existing `focal_zone_signal_quality` gate
on 520's best.pt we initially read E_focal = 0.000 as evidence of "wall-
dominated learning" — the same false-positive failure mode the public
repo flagged for Phase 7c v1. But sampling 100 random training-set
*targets* showed median E_focal = 0.003 (vs uniform-distribution baseline
0.0395). The targets themselves put almost no energy in the focal zone
because forward FEM with *random* transducer phases produces diffuse
interference patterns. Focal points only emerge under *focused* (inverse-
designed) phases. The gate is built for inverse-design outputs, not
forward-trained surrogates evaluated on random-phase splits. The model
matching ~0 focal-zone energy on random-phase targets is correct
behavior, not failure.

**The actually informative next axes:**

- **Higher Fourier mode counts.** 12×12×36 (3× more spectral params,
  ~400M total) is a reasonable first step; 16×16×48 is 8× more
  (~1B params). Test how much the truncation ceiling moves.
- **Cylindrical-harmonics basis** instead of Cartesian Fourier. The
  chamber geometry has azimuthal symmetry the model is currently
  forced to learn from scratch.
- **Hybrid FNO + CNN/MLP head** so high-frequency local structure
  (which the truncated Fourier basis misses) gets a dedicated decoder.

Higher hidden_channels would add optimization capacity without addressing
the truncation ceiling, so it's not the lever. Same with more training
data — the model is already capturing 70% of the representable space.

**Methodological lesson:** when val plateaus, distinguish *optimization
ceiling* from *representational ceiling* before scaling compute. The
representational ceiling is measurable in minutes by projecting targets
through the model's mode budget; the optimization ceiling is what
hyperparameter sweeps actually address. We spent ~80 GPU-hours sweeping
LR (520 → 565 → 603) before measuring the representational ceiling
that explained everything.

### Dataset generation pipeline

7000 configs at 56×56×160 production grid, sharded across two machines
in parallel: 5000 configs on a 12-core Ryzen 9 5900X workstation
(12 parallel FEM workers, ~17 s/config), 2000 configs on an Apple M2
Max laptop (8 workers — Apple Accelerate's sparse spsolve ran ~3×
faster per core than the Ryzen + scipy path, a result we didn't predict).
Each box wrote its own per-worker HDF5 chunks; final dataset assembled
by axis-0 concatenation of per-config datasets plus shallow copy of
shared groups (grid coordinates). Disjoint base seeds (43 and 2,000,000)
guarantee no duplicate configs. ~2 h wall vs. ~5 h single-machine
extrapolation.

For cluster delivery, the dataset goes through Cloudflare R2 (S3-
compatible, multipart upload at 64 MB chunks). `kubectl cp` and
`kubectl exec ... cat` both silently truncated 16 GB files on first
attempts (2–5 MB short, undetected without a size check), so the R2
intermediary is the canonical transport.

## Repository layout

```
ml_inverse/                   # training, inversion, distillation pipeline
  train.py                    # FNO training entry point (residual-prior, H¹ loss)
  inverse.py                  # gradient-descent inverse design through any surrogate
  disagreement_analysis.py    # pairwise residual + attribution
  generate_*_dataset.py       # dataset gen for each physics track
  scripts/mean_pred_sanity.py # the PASS gate
  model.py, dataset.py, ...

drip_physics/
  backends/
    jwave_backend.py          # j-Wave spectral solver wrapper
    femcoupled_backend.py     # drop-in FEM backend, matches jwave signature
    femcoupled/
      helmholtz.py            # parametric Robin BC + Bermúdez PML
      heat.py                 # transient heat with advection
      coupling.py             # StaggeredCouplingDriver — fixed-point Helmholtz↔heat
      mesh.py                 # L1 chamber mesh builder (gmsh)
      geometry.py             # cylinder-geometry helpers

drip_physics_core/            # shared config schemas (PressureConfig, EnvironmentConfig)

cloud_sprint/
  lambda_launch.sh            # generic Lambda instance launch
  phase7c_jwave_L1_cloud.sh   # j-Wave gen + train + eval at L1
  phase7d_fem_L1_gen_only.sh  # FEM gen at L1 (gen-only; verify dataset before training)

receipts/                     # PASS-gate evidence per phase (JSON + PNG)
  _phase66_cloud_production_run/
  _phase7a_cloud_production_run/
  _phase7c_cloud_production_run/
```

## Forward solver detail

The FEM forward (`drip_physics/backends/femcoupled/`) solves
`∫ ∇p·∇v - k² p v dV` with parametric Robin transducer BC
`∂_n p = i k ρ₀ c₀ v_n(φᵢ)` and either Sommerfeld or Bermúdez 2007 PML
on the outer boundary. The Helmholtz solve is **indefinite** (saddle-
point structure for k·L large) — iterative methods like BiCGSTAB
diverge on it. The default solver is `umfpack_solver` (scipy spsolve
under JAX-FEM's dispatcher) which is bulletproof on CPU at the L1 mesh
size (≤ a few × 10⁵ DOFs). PETSc + MUMPS is available as an opt-in for
CUDA-cluster runs.

## Setup

```bash
git clone https://github.com/USER/multi-physics-fno-acoustic.git
cd multi-physics-fno-acoustic
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e drip_physics_core/
```

System dependencies for the FEM path (Ubuntu):
```bash
sudo apt-get install libglu1-mesa libxrender1 libxcursor1 \
                     libxft2 libxinerama1 libxi6 libxext6
```

## Running

Mean-prediction sanity gate on an existing checkpoint:
```bash
PYTHONPATH=. python -m ml_inverse.scripts.mean_pred_sanity \
    --data-path path/to/dataset.h5 \
    --ckpt-best path/to/best.pt \
    --norm path/to/norm.npz \
    --device cuda \
    --out mean_pred.json
```

Disagreement analysis FNO-vs-analytical:
```bash
PYTHONPATH=. python -m ml_inverse.disagreement_analysis \
    --mode fno-vs-analytical \
    --fno-ckpt path/to/best.pt --norm path/to/norm.npz \
    --dataset path/to/dataset.h5 \
    --n-samples 50 --n-render 5 \
    --output-dir ./disagreement_out --device cuda
```

The cloud-sprint scripts under `cloud_sprint/` are the full launchers
(dataset gen → train → eval) used for the Phase 6.6 / 7a / 7c PASS runs.

## AI use

All code in this repository was written by AI coding assistants. Neither
author is a programmer by training — our backgrounds are in mechanical
engineering, acoustics, thermal/fluids systems, and chemical/materials
science.

What we own and are responsible for:

* **Problem formulation.** The decision to build a multi-physics surrogate
  stack (analytical / j-Wave / FEM-coupled) and the disagreement-weighted
  distillation strategy is ours. The cylindrical-chamber regime, the
  choice of focal zone (z ∈ [100, 300] mm, r < 30 mm) and which physics
  terms each track captures or misses are domain decisions grounded in
  the actual hardware.
* **Physics correctness.** The Helmholtz weak form, the Bermúdez 2007
  PML formulation, the Eckart streaming approximation, the Gor'kov
  radiation force, and the residual-prior architecture choices were
  reviewed against published physics; the validation cascade
  (analytical / j-Wave / FEM three-way residual + AM-Bench thermal
  benchmark) was designed to catch physics errors in any single solver.
* **Evaluation methodology.** The choice to use `mean_pred_sanity` as a
  PASS gate, then to add the `focal_zone_signal_quality` gate after it
  surfaced a false-positive PASS, are our calls. The interpretation of
  the disagreement matrix (FNO_F vs FNO_J L1 = 1.07 turning out to be
  "one model converged, the other was undertrained at the interior" not
  "complementary physics" — until the retrain) is our reading of the
  rendered evidence.
* **Compute decisions.** Cloud spend ($~80 to date, ~$30 per training
  sprint), instance-type tradeoffs (A100 vs A10 vs CPU), kill-switch
  caps, dataset gen / pull / verify gates, and the choice to retrain
  rather than build FNO_combined on top of a broken FNO_J L1 are all
  human judgment calls.
* **Bug triage and root-cause analysis.** The discovery that
  `DEFAULT_SOLVER_OPTIONS = {"spsolve_solver": {}}` silently fell
  through to BiCGSTAB on indefinite Helmholtz systems (real bug, fixed
  in commit `d184ba4`), and the realization that Phase 7c v1's
  mean_pred=0.193 PASS was a false positive masking transducer-wall-
  only learning, were debugging calls made by reading rendered output
  and reasoning about the physics — not by the AI proposing it
  unprompted.

What the AI did:

* Wrote essentially all the Python (training loop, FNO model wrapping
  `neuraloperator`, j-Wave and FEM forward backends, dataset generators,
  inverse-design loop, distillation pipeline, all evaluation scripts,
  this README).
* Wrote the cloud-sprint shell runners, set up the multi-layer vigilance
  pattern (mirror loops, checkpoint backup with snapshot-then-scp,
  ETA-vs-deadline monitors), and orchestrated the long-running Lambda
  sessions.
* Drafted the public-repo file structure, the documentation in this
  README, and the public-vs-private code split.

The AI is a tool we use because it's faster than learning Rust-style
type systems and PyTorch idioms from scratch. The decisions above are
where the domain knowledge actually mattered.

— Jamie Marwell & Emma Blemaster

## License

MIT — see LICENSE.
