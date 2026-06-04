# Metrics & Trends — Measured Values + Known Limitations

← Back to [main README](../README.md) · Companions: [Project timeline](PROJECT_TIMELINE.md) · [Pipeline mapping](PIPELINE_MAPPING.md) · [Statistical receipt](../receipts/_phase7d_v3_statistical_receipt.json)

## TL;DR of this doc
- The measured numbers (with bootstrap 95% CIs at N=32) for every metric we report.
- Reproducibility evidence from the runs we ran — what *did* repeat and what we *didn't test* for repeatability.
- A frank list of what we don't know and what the data does NOT support.

This doc is intentionally not interpretive. Earlier drafts of it speculated about *mechanisms* — why a number was what it was, what the model was "really" doing. Several of those speculations couldn't be verified; we've removed them rather than dress guesses as findings. For the longer-term work this surrogate stack feeds into, knowing **what we actually measured + where the gaps are** is more useful than a confident narrative we can't defend.

---

## 1. Per-Metric Reported Values (with CIs)

All numbers from N=32 samples of the v3 dataset, bootstrap 95% CI (1000 resamples, seed=42). Raw JSON: [`../receipts/_phase7d_v3_statistical_receipt.json`](../receipts/_phase7d_v3_statistical_receipt.json).

### 1.1 `mean_pred ratio` — the headline PASS gate

| Model | mean_pred ratio | 95% CI | vs threshold |
|---|---|---|---|
| 520 (8×8×24 modes) | 0.758 | [0.745, 0.770] | FAIL (> 0.5) |
| 866 (12×12×36 modes) | 0.240 | [0.238, 0.241] | PASS (< 0.5) |
| predict-mean baseline | 1.000 | [reference] | — |

**What we measured.** Going from 8×8×24 to 12×12×36 modes (the only deliberate change between 520 and 866) coincided with mean_pred dropping from FAIL to PASS. Paired per-sample test: 866 has lower rel L² than 520 in 32/32 samples (CI on the paired difference doesn't cross zero).

**What we don't know.**
- Whether the improvement came from "more representational capacity" or "different optimization landscape from more parameters" — both changed simultaneously. A clean ablation would hold one fixed; we didn't do that.
- Whether the result reproduces with different random seeds — every model was trained with seed=42.
- Whether the 12×12×36 PASS holds out-of-distribution (different array geometries, different bed-temp ranges, focused-vs-random phases).

### 1.2 `val_h1` — the training-loop loss metric

| Model | best val_h1 | best test_all_h1 |
|---|---|---|
| 520 | 1.940 (ep 14) | 1.903 (ep 15) |
| 866 | 1.799 (ep 25) | 1.754 (ep 25) |

**What we measured.** val_h1 dropped 7% from 520 to 866. Rel L² (normalized space, on the same samples) dropped 66%.

**What we don't know.**
- Why val_h1 moved much less than rel L². One possible reason is that val_h1 includes a gradient term (`||∇pred - ∇target||²`) that emphasizes high-frequency content both models may still miss equally — but we didn't separately measure the L² and gradient components of the val_h1, so this is conjecture.
- Whether val_h1 at ~1.8 is "good" for this problem class. We don't have an external benchmark for cylindrical FEM-coupled Helmholtz at this resolution.

### 1.3 `Pa magnitude ratio` — output calibration

| Model | pred Pa / target Pa | 95% CI | systematic bias | random std |
|---|---|---|---|---|
| 520 | 0.490 | [0.485, 0.494] | −51.0% | 1.27% |
| 866 | 0.732 | [0.729, 0.734] | −26.8% | 0.70% |

**What we measured.** Both models systematically under-predict Pa magnitude. For 866 the systematic bias is 38× larger than the per-sample random standard deviation.

**What we don't know.**
- The mechanism. The H¹ loss has different scale sensitivities for its L² and gradient terms, which *could* produce systematic-bias floors at imperfect scales, but we haven't tested this — e.g., we haven't trained an L²-only version of 866 to see if the bias changes. The mechanism explanation is speculation, not measurement.
- Whether the systematic bias would survive a learned output-rescaling head. That fix is plausible (because the per-sample random std is small relative to the bias) but untested.

### 1.4 Disagreement matrix — pairwise rel L² in physical Pa

N=32, resampled to common 32³ subgrid for cross-grid comparison.

| Pair | 520 F | 866 F |
|---|---|---|
| A↔J | 1.298 [1.295, 1.301] | 1.298 [1.295, 1.301] |
| A↔F | 4.329 [4.284, 4.375] | 3.718 [3.695, 3.738] |
| J↔F | 5.192 [5.135, 5.256] | 4.443 [4.414, 4.471] |

**What we measured.**
- A↔J is identical between the two F-variants (sanity: A and J weren't retrained).
- F-row dropped 14% in both columns going 520 → 866. Paired per-sample test: drop is consistent in 32/32 samples for both A↔F and J↔F.
- F-row remains 2.9× to 3.4× higher than A↔J.

**Calibration context.** The original disagreement-framework calibration (from earlier work in this project, in the private monorepo, not in this submission's receipts/) reported:
- FNO_A vs analytical-truth = 0.31% rel L² ("noise floor")
- FNO_J vs analytical-truth = 133% rel L² ("regime divergence")

Our current measurement of A↔J = 130% matches that earlier "regime divergence" value. F-row at 370–449% is above either of those calibration points.

**What we don't know.**
- How much of the F-vs-A and F-vs-J disagreement is "real" missing-physics signal (FEM-coupled physics has terms — Eckart streaming, ρ(T), c₀(T) — that analytical and j-Wave don't) vs "fake" signal from F's own representational deficit (the 27% Pa magnitude bias from § 1.3 contaminates pairwise comparison).
- Whether the FNO_combined adversarial-training pitch the project was sold on would actually work given the current F. We did not fire FNO_combined.

---

## 2. Repeatability Evidence

### 2.1 What did repeat across independent runs

**Three failure modes each observed in ≥2 independent runs:**
- LR=1e-3 + FEM data → predict-zero collapse at val_h1 = 2.000 exactly. Observed in v3 smoke 446; the LR=1e-4 retry (smoke 469) escaped immediately.
- LR=1e-4 + DDP gpu=4 + cosine T_max=N_EPOCHS → val_h1 rises mid-training while train_h1 keeps falling. Observed in 520 (8×8×24, peaked at ep14), 566 (per-sample loss, same hyperparams, peaked at ep14), 866 (12×12×36, peaked at ep25).
- FEM v1/v2 dataset with fixed bed_temp → val_h1 flatline at 4.0 for 50 epochs. Observed in jobs 165 and 362.

**Two success modes each observed in ≥2 independent runs:**
- FNO_J L1 (Phase 7c dataset) trained successfully across smoke (159, val=1.44 ep1), timed-out real (183, val=1.084 ep42), and full real (240, val=1.05 ep50). All three on the same trajectory shape.
- FNO_A L1 trained successfully across smoke (167, val=1.50 ep5) and full real (205, val=1.19 ep50).

**Deterministic measurement infrastructure.**
- The disagreement matrix gave A↔J = 1.296 at N=8 and 1.298 at N=32 — different sample subsets, same point estimate to 3 decimal places. The bootstrap CI on N=32 has width 0.006, so the difference between N=8 and N=32 estimates is within bootstrap noise.

### 2.2 Smoke→production prediction track record

| Smoke (gpu=1, BATCH=1, 5 ep) | val@ep5 | Production (gpu=4, BATCH=2, 50 ep) | best val |
|---|---|---|---|
| 469 (LR=1e-4, 8×8×24) | 1.09 | 520 (same config, 50 ep) | 1.94 |
| 800 (LR=1e-4, 12×12×36) | 0.98 | 866 (same config, 50 ep) | 1.80 |

In both cases the smoke landed substantially *below* what the production run achieved. The gap was similar in both pairs (~0.85–0.90). The direction of the smoke-to-smoke comparison (800 better than 469 by 0.11) and the direction of the production-to-production comparison (866 better than 520 by 0.14) point the same way.

**One known mechanism** that *would* produce this pattern: the cosine LR schedule has T_max=N_EPOCHS, so at smoke (N=5) the LR decays to eta_min by ep5, while at production (N=50) the LR stays near its initial value through ep30+. We didn't run a controlled experiment to confirm the mechanism — this is a plausible but unverified explanation.

### 2.3 What we did NOT test for repeatability

- **No multi-seed validation.** Every model trained with seed=42. The mode-scaling PASS at 866 could in principle be seed-dependent. We don't have evidence either way.
- **No held-out architectural test.** The PASS gates are computed on the same dataset the model was trained on (different splits within the same data distribution). We don't have a physically different test set (different array geometry, different droplet regime).
- **No human review of field predictions.** Mid-z slice PNGs in `receipts/` exist; no qualified acoustician has reviewed them systematically. The "FNO_J L1 was undertrained at the chamber interior" finding from 2026-05-15 was based on visual inspection by the project authors, not by a domain reviewer.

### 2.4 Known limitations

1. **Single-seed training.** Every result is one seed.
2. **N=32 bootstrap.** Tight CIs, but a small sample. Bootstrap CIs measure the bootstrap procedure's uncertainty at a given N; they don't bound the underlying sampling variance with the rigor that larger N would.
3. **Same-distribution train/test.** All evaluation is within the v3 dataset's distribution.
4. **No inverse-design loop evaluation.** The pitch is "real-time inverse design" but we only validated forward surrogates. Whether the surrogates are good enough for autodiff-through-the-model inverse design is untested in this submission.
5. **No deployment-readiness evaluation.** Model inference latency was measured; OOD behavior, monitoring, fallback paths are not addressed.
6. **No FNO_combined or student-from-combined.** The disagreement-weighted combine + distill steps the project was sold on were not executed. Disagreement matrix evidence (§ 1.4) is consistent with that decision but the *combined* model itself was never trained.
7. **AI-saturated content production.** The volume of documentation and prose was generated heavily by an AI coding assistant. AI-vs-human contribution split documented in README § Author contributions; verifying which intellectual decisions were human-original is left to the reader.

---

## 3. Observations About the Whole Training Arc

These are patterns observed across the run set. They are observations, not conclusions.

### 3.1 The single change that produced the FNO_F PASS — and what follow-on smokes added

We ran ~9 training jobs in the v3 thermal-aware arc varying loss (default H¹ vs per-sample H¹), LR (1e-3, 1e-4, 5e-5), and batch (BATCH=1 single-GPU vs BATCH=2 DDP-gpu=4). None of these alone produced a mean_pred PASS. The transition from FAIL to PASS coincided with the mode-count change (8×8×24 → 12×12×36 in commit `33b62b9`).

We did not run a controlled ablation isolating "more modes" from "more parameters" (mode count change took the model from 118M to 264M params, ~2.2× — both axes changed simultaneously).

**Post-866 single-axis smokes (jobs 974, 975).** After the PASS, we ran two 5-epoch smokes each varying one axis:

| | Modes | Hidden | ep5 val_h1 | vs 866 smoke (0.98) |
|---|---|---|---|---|
| 866 smoke baseline | 12×12×36 | 128 | 0.98 | — |
| 974 (more modes) | **16×16×48** | 128 | 0.914 | −6.7% |
| 975 (more hidden) | 12×12×36 | **192** | 0.911 | −7.0% |

**What we measured:** both axes improved over 866's smoke baseline at ep5 by ~7%. 974 vs 975 differed by 0.003 — below the bootstrap CI width we measured at full eval (~0.005), so we cannot distinguish them at this scale.

**What we don't know:**
- Which axis is the better scaling lever. The smokes don't tell us; same seed, same depth, indistinguishable result.
- Whether the smoke-vs-production gap (smokes landed ~0.85 below production in the two pairings we measured: 469→520 and 800→866) is the same magnitude for these bigger models, or whether the bigger models would close the smoke-vs-production gap differently.
- Whether the DDP cosine-LR overfit at production scale would apply equally to both larger variants.

A combined-axes smoke (job 981: 16×16×48 modes AND HIDDEN=192) was fired to test whether the two axes compound or saturate at the same ~0.91 ceiling. Result not yet captured in this writing of the doc.

**Decision on 50-ep production for either variant.** Not fired before submission. The smokes themselves are the data; a single 50-ep production wouldn't distinguish which axis matters more without controlled multi-seed comparison.

### 3.2 Forward training on random phases doesn't probe focal-zone behavior

Sampled 100 random targets from the v3 dataset: median E_focal = 0.003 (vs uniform-distribution baseline 0.0395). The targets themselves put almost no energy in the focal zone when transducer phases are randomly sampled.

This means: the `focal_zone_signal_quality` gate that was added in 2026-05-15 (to catch the Phase 7c v1 false-positive PASS) **does not apply** to forward surrogates evaluated on random-phase splits. We had a brief period during the v3 deep-eval when we read E_focal ≈ 0 as failure; the targets themselves have E_focal ≈ 0, so the model matching that is correct behavior on this evaluation.

The implication is that **forward surrogates trained on random phases cannot be evaluated for inverse-design quality using the focal-zone gate.** The appropriate inverse-design evaluation would run the autodiff-through-the-model inverse loop and check the resulting field — we did not do this.

### 3.3 Systematic magnitude bias persisted across both 8×8×24 and 12×12×36 architectures

Both 520 and 866 systematically under-predict Pa magnitude. 520 by 51% on average, 866 by 27% on average. The per-sample random standard deviation is small (1.3% for 520, 0.7% for 866), so the prediction is *tightly clustered around a biased mean*.

The bias dropped roughly in half going from 118M params (520) to 264M params (866) — a 2.2× parameter increase. We don't know if the trend would continue with more parameters, whether it's an artifact of the H¹ loss landscape, or whether it's a data-normalization bug. We did not isolate.

### 3.4 What the run set is and isn't

The 30+ training jobs in this project span variation along 8 axes: dataset version (v1/v2/v2.5/v3), modes, hidden channels, layers, LR, schedule, batch, loss. Most jobs vary multiple axes simultaneously — they were exploratory, run to make progress on a specific failure, not to ablate.

Calling this an "ablation study" overstates the experimental design. It is an *exploration* whose net effect was to produce three trained surrogates (A, J, F) and a documented set of failure modes. The honest framing of the contribution is empirical engineering on the problem class, not a controlled experimental study.

---

## What we'd want to do next (for Drip's longer-term planning)

To convert the current empirical observations into things we'd defend with confidence, the experiments worth running are:

1. **Multi-seed validation of 866's PASS.** Three more runs with seed=43, 44, 45. ~12 GPU-hr each = ~36 GPU-hr. Would tell us whether the PASS is robust or seed-lucky.
2. **L²-only loss variant of 866.** One additional production run with L² only (no H¹). Tests whether the systematic magnitude bias comes from the H¹ loss landscape (would expect bias to shrink if so).
3. **Held-out test on different array geometry.** Generate a small dataset with a different transducer-positioning (e.g., 12-ring vs 10-ring), evaluate FNO_F there. Tests generalization.
4. **Inverse-design loop evaluation.** Run the autodiff-through-FNO_F inverse loop on 10 target trajectories, check the resulting fields. Whether the surrogate is good enough for inverse design is unknown until this is done.
5. **Output-rescaling head.** Train a small calibration head on the systematic magnitude bias. If it closes the bias without breaking structure, our predicted Pa magnitudes become deployment-grade.

If any of these come back negative, the surrogate is less ready than the current docs suggest. If they come back positive, we have strong evidence to fire FNO_combined and the student-v2 distillation. **Either outcome is more useful than continuing to polish the current docs.**
