"""Focal-zone signal quality diagnostic.

Measures whether a trained surrogate produces *structured* acoustic
predictions in the chamber interior (the focal zone where droplets
actually traverse) vs essentially noise.

Catches the failure mode `mean_pred` misses: a model can pass
`mean_pred < 0.5` by learning the boundary-condition forcing at the
transducer surfaces (high pressure at the walls) while the interior
is uniform noise.

Metrics per sample:
    p_max_focal       — peak |P| inside focal zone
    p_mean_focal      — mean |P| inside focal zone
    p_std_focal       — std |P| inside focal zone
    peak_to_mean      — p_max / p_mean (high = focused, low = uniform)
    coeff_var         — std / mean (high = structured, low = flat/noise)
    energy_in_focal   — fraction of total |P|² energy inside focal zone
    db_dynamic_range  — 20·log10(p_max / p_mean)

Verdict heuristics:
    db_dynamic_range > 12 dB  → focused signal
    peak_to_mean > 4          → structured (not noise)
    energy_in_focal > 0.05    → meaningful focal mass (vs walls)

Run against analytical (closed-form 1/r) for free since the model's
residual-prior already exposes it as `_amplitude_3d` and `_kr_3d`
buffers — we can synthesize the prior-only field for comparison.

Compare:
    python -m ml_inverse.scripts.focal_zone_signal_quality \
        --ckpt simulations/ml_inverse/models/fno_surrogate_fem_phase66_cloud_best.pt \
        --norm simulations/ml_inverse/models/fno_surrogate_fem_phase66_cloud_norm.npz \
        --phases-h5 simulations/ml_inverse/data/inverse_dataset_fem_3d_phase66_5000.h5 \
        --grid-extent-m -0.03 0.03 -0.03 0.03 0.10 0.30 \
        --grid-shape 32 32 32 \
        --n-modes 15 15 15 \
        --label FNO_F_phase66 \
        --out-dir /tmp/fz_fno_f
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ml_inverse.scripts.pairwise_fno_overlap import load_fno_manual, denormalize  # noqa: E402


# Focal zone: where droplets actually traverse + are controlled.
# Interior cylinder of radius 30mm (well inside the 50mm L1 array radius),
# z ∈ [100, 300] mm (between bottom of array and end of control region).
FOCAL_R_M = 0.030
FOCAL_Z_MIN_M = 0.100
FOCAL_Z_MAX_M = 0.300


def build_focal_mask(
    grid_shape: Tuple[int, int, int],
    extent_m: Tuple[float, float, float, float, float, float],
    focal_r: float = FOCAL_R_M,
    focal_z: Tuple[float, float] = (FOCAL_Z_MIN_M, FOCAL_Z_MAX_M),
) -> np.ndarray:
    """Boolean mask of shape (nx, ny, nz) selecting focal-zone voxels.

    Returns the integer count and physical-volume coverage to help
    interpret the resulting metrics.
    """
    nx, ny, nz = grid_shape
    xmin, xmax, ymin, ymax, zmin, zmax = extent_m

    x = np.linspace(xmin, xmax, nx, dtype=np.float32)
    y = np.linspace(ymin, ymax, ny, dtype=np.float32)
    z = np.linspace(zmin, zmax, nz, dtype=np.float32)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2)

    mask = (r < focal_r) & (Z >= focal_z[0]) & (Z <= focal_z[1])
    return mask


def signal_quality_metrics(
    p_mag: np.ndarray,  # |P| at (nx, ny, nz)
    mask: np.ndarray,   # boolean focal-zone mask
) -> dict:
    """Compute structural-signal metrics inside the focal zone."""
    interior = p_mag[mask]
    if interior.size == 0:
        return {"error": "empty focal mask"}

    p_max = float(interior.max())
    p_mean = float(interior.mean())
    p_std = float(interior.std())
    peak_to_mean = p_max / max(p_mean, 1e-12)
    coeff_var = p_std / max(p_mean, 1e-12)
    db_dyn_range = 20.0 * np.log10(p_max / max(p_mean, 1e-12))

    e_total = float((p_mag ** 2).sum())
    e_focal = float((p_mag[mask] ** 2).sum())
    energy_in_focal = e_focal / max(e_total, 1e-12)

    return {
        "p_max_focal":      p_max,
        "p_mean_focal":     p_mean,
        "p_std_focal":      p_std,
        "peak_to_mean":     float(peak_to_mean),
        "coeff_var":        float(coeff_var),
        "db_dynamic_range": float(db_dyn_range),
        "energy_in_focal":  float(energy_in_focal),
        "n_focal_voxels":   int(mask.sum()),
        "n_total_voxels":   int(mask.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--phases-h5", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True, help="Short tag for the output summary")
    ap.add_argument("--grid-shape", type=int, nargs=3, required=True)
    ap.add_argument("--grid-extent-m", type=float, nargs=6, required=True)
    ap.add_argument("--n-modes", type=int, nargs=3, required=True)
    ap.add_argument("--hidden-channels", type=int, default=128)
    ap.add_argument("--prior-version", default="v2.1_3d")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    grid_shape = tuple(args.grid_shape)
    extent = tuple(args.grid_extent_m)
    n_modes = tuple(args.n_modes)

    # Load model
    model, mean, std, eps = load_fno_manual(
        args.ckpt, args.norm,
        grid_shape=grid_shape, extent_m=extent,
        n_modes=n_modes, hidden_channels=args.hidden_channels,
        prior_version=args.prior_version, device=device,
    )
    print(f"[load] {args.label}: grid={grid_shape} extent={extent}")

    # Build focal-zone mask
    mask = build_focal_mask(grid_shape, extent)
    print(f"[mask] focal-zone voxels: {mask.sum()}/{mask.size} "
          f"({100*mask.sum()/mask.size:.1f}% of grid)")
    if mask.sum() == 0:
        print("[ERROR] focal mask is empty — your extent doesn't intersect z=[100,300]mm × r<30mm")
        return 1

    # Pull phases
    with h5py.File(args.phases_h5, "r") as h5:
        all_phases = h5["phases"][:].astype(np.float32)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(all_phases), size=min(args.n_samples, len(all_phases)), replace=False)
    idx.sort()
    phases = torch.from_numpy(all_phases[idx]).to(device)

    # Forward + metrics
    per_sample = []
    with torch.no_grad():
        for i in range(len(phases)):
            x = phases[i:i+1]
            out_n = model(x)
            out = denormalize(out_n, mean, std, eps)
            field = out[0].cpu().numpy()  # (2, *grid)
            p_mag = np.sqrt(field[0] ** 2 + field[1] ** 2)
            m = signal_quality_metrics(p_mag, mask)
            per_sample.append(m)
            print(f"  sample {i+1:2d}/{len(phases)} (idx={idx[i]:5d})  "
                  f"p_max={m['p_max_focal']:.0f}  p_mean={m['p_mean_focal']:.1f}  "
                  f"peak/mean={m['peak_to_mean']:.2f}  CV={m['coeff_var']:.2f}  "
                  f"E_focal/E_total={m['energy_in_focal']:.3f}  "
                  f"dyn_range={m['db_dynamic_range']:.1f}dB")

    # Aggregate
    def agg(key):
        vals = np.array([s[key] for s in per_sample])
        return {
            "mean":   float(vals.mean()),
            "median": float(np.median(vals)),
            "min":    float(vals.min()),
            "max":    float(vals.max()),
            "std":    float(vals.std()),
        }

    summary = {
        "label": args.label,
        "n_samples": len(per_sample),
        "grid_shape": list(grid_shape),
        "grid_extent_m": list(extent),
        "focal_zone_def": {
            "r_max_m": FOCAL_R_M,
            "z_range_m": [FOCAL_Z_MIN_M, FOCAL_Z_MAX_M],
        },
        "n_focal_voxels": int(mask.sum()),
        "n_total_voxels": int(mask.size),
        "aggregate": {
            "peak_to_mean":     agg("peak_to_mean"),
            "coeff_var":        agg("coeff_var"),
            "energy_in_focal":  agg("energy_in_focal"),
            "db_dynamic_range": agg("db_dynamic_range"),
            "p_max_focal":      agg("p_max_focal"),
            "p_mean_focal":     agg("p_mean_focal"),
        },
        "verdict": {
            "structured_threshold": "peak_to_mean > 4 AND db_dynamic_range > 12 dB",
            "focused_threshold":    "energy_in_focal > 0.05",
        },
        "per_sample": per_sample,
        "indices": idx.tolist(),
    }
    (out_dir / f"signal_quality_{args.label}.json").write_text(json.dumps(summary, indent=2))

    pt_m = np.array([s["peak_to_mean"] for s in per_sample])
    db_m = np.array([s["db_dynamic_range"] for s in per_sample])
    e_m  = np.array([s["energy_in_focal"] for s in per_sample])

    structured = (pt_m.mean() > 4) and (db_m.mean() > 12)
    focused = e_m.mean() > 0.05

    print(f"\n{'=' * 60}")
    print(f"  {args.label}")
    print(f"{'=' * 60}")
    print(f"  peak/mean          mean={pt_m.mean():.2f}   (>4 = structured)")
    print(f"  db_dynamic_range   mean={db_m.mean():.1f} dB   (>12 dB = focused signal)")
    print(f"  energy_in_focal    mean={e_m.mean():.4f}   (>0.05 = mass actually in focal zone)")
    print(f"")
    print(f"  STRUCTURED:  {'YES' if structured else 'NO'}")
    print(f"  FOCUSED:     {'YES' if focused else 'NO'}")
    print(f"  -> {'PASS' if (structured and focused) else 'FAIL'}: focal-zone signal quality")
    print(f"\n[done] wrote {out_dir/('signal_quality_'+args.label+'.json')}")
    return 0 if (structured and focused) else 1


if __name__ == "__main__":
    sys.exit(main())
