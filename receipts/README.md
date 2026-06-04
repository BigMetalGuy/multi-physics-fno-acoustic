# Receipts Index

Machine-readable evidence per phase. Each folder corresponds to a
production run; PASS gate JSONs are at the top, attribution/disagreement
PNGs are visualizations of the residual.

← Back to [main README](../README.md) · [Project timeline](../docs/PROJECT_TIMELINE.md) · [Pipeline mapping](../docs/PIPELINE_MAPPING.md)

---

## `_phase66_cloud_production_run/`

**Phase 6.6b — first FNO_F (FEM-coupled) to clear the mean_pred sanity gate.**
Cubic 32³ grid, ~1000 configs.

| File | What it is |
|---|---|
| `mean_pred.json` | PASS gate result. `ratio_mean = 0.144` (PASS, threshold < 0.5). `best_val_h1 = 9.947`. |
| `fno_vs_analytical_summary.json` | Pairwise residual FNO_F vs analytical 1/r baseline across N=20 configs. |
| `fno_vs_analytical_attribution.json` | Per-axis attribution of where the residual concentrates. `focal_peak_fraction = 0.7`. |
| `attribution_avg_residual_*.png` | Visualization: average residual heat map vs analytical baseline. |
| `disagreement_idx*_midz.png` | 5 per-sample mid-z slice plots showing FNO output vs analytical. |

**Headline finding:** model produces clean focal-spot structure at peak ~800 Pa center. `peak/mean = 5.99`, `dB dynamic range = 15.5 dB`, `E_focal = 0.747` — PASS on the focal-zone gate that was added later.

## `_phase7a_cloud_production_run/`

**Phase 7a — first FNO_J (j-Wave spectral Helmholtz) at production grid.**
32×32×96 mini-array, 5000 configs.

| File | What it is |
|---|---|
| `mean_pred.json` | `ratio_mean = 0.094` (PASS). `best_val_h1 = 1.179`. |
| `fno_vs_analytical_summary.json` | Pairwise residual FNO_J vs analytical. |
| `fno_vs_analytical_attribution.json` | Per-axis attribution. |
| `disagreement_idx*_midz.png` | 5 mid-z slice visualizations. |

## `_phase7c_cloud_production_run/`

**Phase 7c — FNO_J at full L1 cylinder scale (44×44×144).**

| File | What it is |
|---|---|
| `mean_pred.json` | `ratio_mean = 0.193` (PASS). `best_val_h1 = 1.213`. |
| `fno_vs_analytical_summary.json` + `attribution.json` | Same format as 6.6 / 7a. |
| `disagreement_idx*_midz.png` | 5 mid-z slices. |

⚠️ **This PASS was later flagged as a false positive** by the
`focal_zone_signal_quality` gate (see next folder). The model passed
mean_pred only because wall-dominated predictions reduced aggregate
error enough to clear the 0.5 threshold; interior wave physics was
missing. Forced a retrain. See [README § Phase 7d post-mortem](../README.md#phase-7d-post-mortem-from-ceiling-to-pass-via-architecture-scaling) and [PROJECT_TIMELINE Era 3](../docs/PROJECT_TIMELINE.md).

## `_focal_zone_signal_quality/`

**The stricter second-layer gate**, added after the 7c v1 false-positive
was discovered. Measures whether the predicted field has realistic
focal-spot structure (peak/mean, dB dynamic range, E_focal) inside the
chamber interior vs the boundary.

| File | What it is |
|---|---|
| `signal_quality_FNO_F_phase66.json` | FNO_F (Phase 6.6) — PASS (`peak/mean 5.99`, `dyn 15.5dB`, `E_focal 0.747`). |
| `signal_quality_FNO_J_L1_phase7c.json` | FNO_J L1 (Phase 7c) — **FAIL** (`peak/mean 3.69`, `dyn 11.3dB`, **`E_focal 0.0487`** — below uniform-distribution baseline of 0.096). |

The FAIL revealed wall-dominated learning: the model put energy at
transducer surfaces (the Robin BC source), not in the chamber interior
where droplets traverse. This is the "false-positive PASS" pattern
caught by adding this gate after `mean_pred_sanity`.

## `_disagreement_F_vs_J_focal/`

**Pairwise FNO_F vs FNO_J L1 residual**, both denormalized to physical Pa,
both resampled onto FNO_F's 32³ focal-zone grid for direct comparison.

| File | What it is |
|---|---|
| `summary.json` | Pairwise rel L² statistics over N=20 samples. Result: `mean = 1.071`, `median = 1.064`, range `[1.05, 1.10]` — strikingly consistent. |
| `compare_slice_idx*.png` | 5 side-by-side mid-z slice renderings showing FNO_F and FNO_J L1 outputs at the same input phases. |

**The story this told** (chronological): initial read was "two physics
tracks learning complementary components beyond analytical (~70% each)."
Visual inspection of the mid-z slices flipped the story: FNO_F produces
clean focal-spot structure (peak ~800 Pa center), FNO_J L1's interior is
essentially noise. The 1.07 disagreement wasn't complementary physics —
it was FNO_J L1 being undertrained at the chamber interior. **This is
exactly why the disagreement matrix is gated on each track passing the
focal-zone gate.** Documented in [PROJECT_TIMELINE Era 3](../docs/PROJECT_TIMELINE.md).

---

## Receipts NOT yet in this directory

The Phase 7d v3 thermal-aware results (job 866, val_h1 = 1.799, mean_pred
0.281 PASS) and the updated disagreement matrix (520 F vs 866 F) live
inline in [README § Phase 7d post-mortem](../README.md#phase-7d-post-mortem-from-ceiling-to-pass-via-architecture-scaling)
as tables rather than separate JSON receipts. The raw JSON of the
deep-eval comparison is at `/tmp/v3_866_eval/deep_eval_comparison.json`
on the development machine; a sanitized copy could be added here.
