# Multi-Physics FNO Surrogates for Cylindrical Acoustic Inverse Design

> **CS 153 submission — Project Track: Research (with Application/Product secondary).**
> Co-authors: Jamie Marwell, Emma Blemaster.

Training cross-physics Fourier Neural Operator (FNO) surrogates for the
40 kHz acoustic forward problem in cylindrical chambers, then using
their pairwise disagreement as a calibrated uncertainty signal for
inverse design and distillation.

The cylindrical regime with hard reflective walls and anisotropic grids
is under-explored relative to the cubic free-field domains where most
FNO acoustic literature lives. Applications include droplet manipulation
for additive manufacturing, HIFU thermal focusing, acoustic tweezer
trapping in microfluidic cylinders, and ultrasonic NDT.

---

## TL;DR (for graders, in 60 seconds)

**Problem.** Acoustic inverse design in a 40 kHz cylindrical chamber driven
by a 120-element phased array. The forward map (transducer phases → 3D
pressure field) is a complex Helmholtz PDE; one FEM solve is ~17 seconds.
PDE-constrained inverse (target field → phases) takes minutes-to-hours.
Neither is fast enough for real-time control.

**Approach.** Train three FNO surrogates (one per physics fidelity:
analytical, j-Wave, FEM-coupled). Use their pairwise disagreement as a
calibrated uncertainty signal. Combine into a teacher; distill a real-time
student. Already proven: **student v1 from FNO_J achieves 23,288× speedup
at 0.479 ms inference.**

**What was shipped this term.**

| Surrogate | Geometry | Validation | Outcome |
|---|---|---|---|
| FNO_F (Phase 6.6b) | 32³ cubic | mean_pred = **0.144 PASS** | First FNO_F surrogate to clear sanity gate |
| FNO_J (Phase 7a) | 32×32×96 mini-array | mean_pred = **0.094 PASS** | First j-Wave FNO at full PDE-grid scale |
| FNO_J L1 (Phase 7c) | 44×44×144 L1 cylinder | mean_pred = **0.193 PASS** | Forced new gate after focal-zone false-positive |
| Student v1 (distilled from FNO_J) | — | — | **23,288× speedup, 0.479 ms inference** |
| FNO_F (Phase 7d v3 thermal-aware) | 56×56×160 L1 cylinder | val_h1 **1.94** (8×8×24 modes) → **0.98 at ep5** with 12×12×36 modes | Hit and broke Fourier-truncation ceiling |

**Headline methodology lesson.** When a model "plateaus," distinguish
*representational ceiling* (architecturally representable) from
*optimization ceiling* (what the optimizer finds). We spent ~80 GPU-hours
sweeping learning rate before measuring representational capacity — the
finding (8×8×24 modes lose 83% of target structure before any training)
explained the plateau in minutes. Going to 12×12×36 modes broke through
the ceiling on the first attempt.

**Failure analysis & honest reframings shipped in this repo:**

- Phase 7c v1's `mean_pred = 0.193 PASS` was a **false positive** —
  wall-dominated learning that the existing gate missed.
  `focal_zone_signal_quality.py` was added as a stricter gate
  (commit `b169eb0`).
- Initial deep-evaluation read of v3 thermal-aware as "broken" was
  itself wrong — the focal-zone gate was misapplied to a forward-trained
  model on random-phase data. The targets *themselves* have median
  E_focal = 0.003 (less than uniform-baseline 0.0395). The model matching
  ~zero focal energy was correct behavior on this evaluation.
  See `docs/PROJECT_TIMELINE.md` Era 8 for the full disproof chain.

---

## For CS 153 graders — rubric mapping

Each rubric criterion below cross-references where to find the evidence
in this repo. Heavy detail is in `docs/PROJECT_TIMELINE.md`
(chronological, every job & decision) and `docs/PIPELINE_MAPPING.md`
(events mapped to the canonical ML pipeline: data → pretrain → train →
mid-train → post-train → deploy → online feedback).

### Problem & Insight (3 pts)

- **The problem.** Real-time inverse design through a multi-physics
  PDE — a problem that is wide open in the cylindrical-chamber regime.
  See the *Problem* paragraph in the TL;DR and the `Architecture`
  section below.
- **Motivation.** Acoustic levitation of droplets and HIFU thermal
  focusing both need closed-loop control at video frame rate; classical
  inverse design is minutes-to-hours per query.
- **Originality.** Three forward physics tracks of *increasing fidelity*
  plus disagreement-weighted adversarial training is the original
  contribution — not "train an FNO" generically, but a calibrated
  multi-fidelity recipe. The cylindrical regime with hard reflective
  walls + anisotropic grids has limited prior FNO literature.

### Execution & Technical Work (5 pts)

- **Three forward solvers** in `drip_physics/backends/` (analytical,
  j-Wave wrapper, FEM-coupled with Bermúdez 2007 PML + Eckart streaming).
- **Three FNO surrogates** with receipts in `receipts/` for every
  PASS run.
- **Dataset gen pipeline** (`ml_inverse/generate_*_dataset.py`).
- **Training stack** (`ml_inverse/train.py`, `model.py`, `dataset.py`,
  `adapter.py`, `cloud_sprint/*.sbatch`).
- **Inverse design loop** (`ml_inverse/inverse.py`) — autograd
  through any registered surrogate.
- **Disagreement framework** (`ml_inverse/disagreement_analysis.py`).
- **Sanity-gate scripts** (`ml_inverse/scripts/mean_pred_sanity.py`,
  `ml_inverse/scripts/focal_zone_signal_quality.py`).
- **Iteration evidence.** 30+ tracked SLURM jobs on the Stanford CS 153
  Omniva cluster (full per-job log in `docs/PROJECT_TIMELINE.md`
  Eras 5-9), plus three FNO_F architecture iterations (Phase 6.2 →
  6.3 → 6.4 → 6.5 → 6.6b PASS), and the full v1 → v2 → v2.5 → v3
  dataset evolution arc.

### Evaluation & Evidence (3 pts)

- **PASS gates** per phase, machine-readable in `receipts/`:
  - `receipts/_phase66_cloud_production_run/mean_pred.json` —
    FNO_F PASS (ratio 0.144)
  - `receipts/_phase7a_cloud_production_run/mean_pred.json` —
    FNO_J PASS (ratio 0.094)
  - `receipts/_phase7c_cloud_production_run/mean_pred.json` —
    FNO_J L1 PASS (ratio 0.193)
  - `receipts/_focal_zone_signal_quality/` — the stricter second-layer
    gate added after a false-positive PASS was caught.
- **Disagreement framework calibration.** FNO_A vs analytical = 0.31%
  noise floor; FNO_J vs analytical = 133% regime divergence — 430×
  separation (`receipts/_disagreement_F_vs_J_focal/`).
- **Failure analyses recorded in plain prose** in this README:
  the Phase 7d v1/v2 hidden-conditioning failure, the v3 LR=1e-3
  predict-zero collapse, the v3 LR=1e-4 DDP overfit at ep14, the
  Fourier truncation ceiling discovery, and the focal-zone gate
  misapplication self-correction.
- **Reproducibility.** `requirements.txt`, pinned `cloud_sprint/`
  sbatch templates, full SLURM job log in
  `docs/PROJECT_TIMELINE.md`, every checkpoint distributed via
  Cloudflare R2 with sha256 hashes.

### Communication & Presentation (2 pts)

- This README is structured for a cold reader: TL;DR → rubric map →
  architecture → current state → narrative arcs → setup → running.
- `docs/PROJECT_TIMELINE.md`: chronological 9-era deep timeline
  (every meaningful decision and job).
- `docs/PIPELINE_MAPPING.md`: maps each timeline event onto the
  canonical ML pipeline stages (data → pretrain → train → mid-train
  → post-train → deploy → online feedback).
- `LICENSE`: MIT.
- Demo video: see project submission (separate link).

### Process, Integrity & Disclosure (2 pts)

- **AI tools used** (specifics): Claude Code (Anthropic) was the
  primary coding agent, orchestrating the cloud sprint, MLOps
  pipeline, and the diagnostic post-mortems. See the *AI use*
  section below for the full ownership split.
- **Sources & prior art credited**: see *Forward solver detail*
  section for citations to Bermúdez 2007 PML, Eckart streaming,
  Kovachki FNO. The j-Wave backend was *discovered* to already
  exist in the codebase (April 2026) — the repo includes the
  audit chain (`research/JWAVE_BACKEND_AUDIT.md` in the private
  monorepo) that fixed it in 4 specific ways.
- **Major decisions and limitations**: documented in plain prose
  in this README — including the false-positive PASS, the
  representational ceiling, the multi-machine gen surprise, and
  the kubectl-streaming transport failure.
- **Public commit history**: this repo, visible. The private
  development history feeding it is in a separate monorepo.
- **Compute disclosed**: see *Compute* section at the bottom.

---

## Quick-start (5 commands to reproduce a sanity gate)

```bash
# 1. Clone + create venv
git clone https://github.com/BigMetalGuy/multi-physics-fno-acoustic.git
cd multi-physics-fno-acoustic
python3 -m venv .venv && source .venv/bin/activate

# 2. Install deps
pip install -r requirements.txt
pip install -e drip_physics_core/

# 3. Fetch a published FNO_J checkpoint from R2 (public bucket)
curl -L -H "User-Agent: drip-dashboard/1.0" \
    -o /tmp/fno_J.pt \
    https://pub-910e11cd3e304ebfbcaefa35051ad03e.r2.dev/fno_surrogate_jwave_phase7c_L1_omniva.pt
curl -L -H "User-Agent: drip-dashboard/1.0" \
    -o /tmp/fno_J_norm.npz \
    https://pub-910e11cd3e304ebfbcaefa35051ad03e.r2.dev/fno_surrogate_jwave_phase7c_L1_omniva_norm.npz

# 4. Generate a tiny dataset (or skip and use the existing fixture)
PYTHONPATH=. python -m ml_inverse.generate_jwave_dataset \
    --n-trajectories 10 --grid-resolution 44 44 144 \
    --output-path /tmp/mini_jwave.h5

# 5. Run the mean-prediction sanity gate
PYTHONPATH=. python -m ml_inverse.scripts.mean_pred_sanity \
    --data-path /tmp/mini_jwave.h5 \
    --ckpt-best /tmp/fno_J.pt --norm /tmp/fno_J_norm.npz \
    --device cuda --out /tmp/mean_pred.json
cat /tmp/mean_pred.json
```

System dependencies for the FEM path (Ubuntu) live further down in
the `Setup` section. The j-Wave path doesn't need them.

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
| 7c   | FNO_J | 44×44×144 L1 cylinder | 0.193 | **PASS** (later flagged false-positive → 7c v2 retrain) |
| —    | student v1 (distilled from FNO_J) | — | — | **23,288× speedup, 0.479 ms inference** |
| 7d v1/v2 | FNO_F (fixed bed_temp) | 56×56×160 L1 cylinder | val_h1 ≈ 4.0 | **FLATLINE** — diagnosed as hidden-conditioning failure |
| 7d v3 (thermal-aware, 8×8×24 modes) | FNO_F | 56×56×160 L1 cylinder | val_h1 = **1.94** at ep 14 | architectural representational ceiling; ~70% of representable space captured |
| 7d v3 (12×12×36 modes) smoke | FNO_F | 56×56×160 L1 cylinder | val_h1 = **0.98** at ep 5 (broke predict-zero baseline) | confirmed truncation ceiling; 50ep production in flight |

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

**Primary tool: Claude Code (Anthropic).** Used as the orchestrator
across every era documented in `docs/PROJECT_TIMELINE.md`:

- Writing all Python and shell code (training loop, FNO model wrapping
  `neuraloperator`, j-Wave and FEM forward backends, dataset generators,
  inverse loop, distillation pipeline, all evaluation scripts, this
  README).
- Driving the Stanford Omniva cluster (submit jobs via `sbatch`, monitor
  via `squeue`/`sacct`, pull curves via `kubectl exec`, diagnose
  failures from `slurm-*.out` logs).
- Multi-machine MLOps orchestration during dataset gen — coordinating
  parallel runs on Mac + Ryzen workstation, transferring datasets via
  Cloudflare R2.
- Persistent cross-session memory: working configurations, prior
  failures, R2 credentials' location, methodology lessons accumulate
  across sessions so session N+1 picks up the operational context.
- Auto-polling: scheduled wakeups every 30 min to check long-running
  training jobs and surface curve descents / failures / completions
  to the human operator.

**Other tools used in narrower roles:** GitHub Copilot for occasional
in-editor autocomplete; ChatGPT for one-off physics literature lookups
(prompts not archived). All code-shipping work was through Claude Code.



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

## Compute disclosure

Cluster GPU work used the **Stanford CS 153 / Omniva** allocation (4× H100
single-node, 250 GPU-hour quota). As of this submission, ~150 GPU-hours
across the SW-27 closeout arc (Phase 7d/c gen + FNO_J/F/A training + v3
hyperparameter sweep + mode-scaling experiment). Full per-job log in
`docs/PROJECT_TIMELINE.md`. SLURM `sacct` is the authoritative record.

Local compute:
- **Mac M2 Max**: 2000 v3 dataset configs generated locally (~16 min wall,
  8 parallel FEM workers). Apple Accelerate's `spsolve` ran ~3× faster
  per core than the Ryzen + scipy path — empirically surprising.
- **Ubuntu workstation (Ryzen 9 5900X, 64 GB)**: 5000 v3 dataset configs
  (~100 min wall, 12 parallel FEM workers). Multi-machine total wall
  ~2 h for 7000-config dataset vs ~5 h single-machine extrapolation.

External infrastructure:
- **Cloudflare R2**: model + dataset distribution. Public read-only bucket
  at `https://pub-910e11cd3e304ebfbcaefa35051ad03e.r2.dev/` for all
  published checkpoints. Used as the cluster ↔ R2 ↔ rig-b transport
  intermediary after `kubectl cp` and `kubectl exec ... cat` were
  observed to silently truncate multi-GB files.
- **Railway**: production deployment of the dashboard / inference server
  (separate Drip-internal repo).

Earlier cloud spend (Lambda Labs, before the Stanford allocation came
online): ~$80 across the Phase 6.x FNO_F iteration arc.

DigitalOcean & Cloudflare CS 153 credits: not used for this project. The
Stanford H100 allocation covered all training compute; R2 was on an
existing Drip-internal account so the CS 153 Cloudflare credits weren't
needed.

## License

MIT — see LICENSE.
