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

## License

MIT — see LICENSE.
