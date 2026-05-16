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

| Phase | Model | Geometry | mean_pred ratio | Verdict |
|---|---|---|---|---|
| 6.6b | FNO_F | 32³ cubic | 0.144 | **PASS** |
| 7a   | FNO_J | 32×32×96 mini-array | 0.094 | **PASS** |
| 7c   | FNO_J | 44×44×144 L1 cylinder | 0.193 | **PASS** |
| —    | student v1 (distilled from FNO_J) | — | — | **23,288× speedup, 0.479 ms inference** |
| 7d *(in progress)* | FNO_F retrain | 44×44×144 L1 cylinder | — | Pending |

All PASS receipts (mean_pred.json, disagreement images, training logs)
live in `receipts/`.

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
