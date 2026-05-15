"""Pairwise FNO disagreement on the focal-zone overlap region.

Compares two FNO surrogates trained on different physics tracks at
different native grids. Resamples the larger-grid surrogate onto the
smaller-grid surrogate's voxel coordinates via trilinear interpolation,
then computes relative-L2 residuals in physical units.

Specifically built for:
    FNO_F  — Phase 6.6b FEM-coupled at 32³ over (-30..30, -30..30, 100..300) mm
    FNO_J  — Phase 7c j-Wave L1 at (44, 44, 144) over (-66..66, -66..66, 0..432) mm

The 32³ focal-zone region of FNO_F is the physically meaningful overlap
where droplets traverse. FNO_J L1's larger chamber is downsampled to
that region; downsampling is well-defined (vs upsampling which is the
out-of-distribution risk).
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

from ml_inverse.model import PhaseConditionedFNO  # noqa: E402


def build_grid_coords(
    extent_m: Tuple[float, float, float, float, float, float],
    shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-axis voxel center coordinates."""
    xmin, xmax, ymin, ymax, zmin, zmax = extent_m
    nx, ny, nz = shape
    return (
        np.linspace(xmin, xmax, nx, dtype=np.float32),
        np.linspace(ymin, ymax, ny, dtype=np.float32),
        np.linspace(zmin, zmax, nz, dtype=np.float32),
    )


def load_fno_manual(
    ckpt_path: str,
    norm_path: str,
    *,
    grid_shape: Tuple[int, int, int],
    extent_m: Tuple[float, float, float, float, float, float],
    n_modes: Tuple[int, int, int],
    hidden_channels: int = 128,
    cond_channels: int = 32,
    out_channels: int = 2,
    n_transducers: int = 120,
    prior_version: str = "v2.1_3d",
    wavenumber: float = 732.27,
    slice_z: float = 0.09,
    device: torch.device | None = None,
) -> Tuple[PhaseConditionedFNO, np.ndarray, np.ndarray, float]:
    """Hand-construct an FNO and load weights from a best-only checkpoint.

    Returns ``(model, mean, std, eps)`` where mean/std are the
    normalizer's per-channel mean/std for denormalization.
    """
    device = device or torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]

    # Pull tx positions from the state_dict (baked in by the prior)
    tx_positions = sd["_tx_positions"].cpu().numpy().astype(np.float32)
    assert tx_positions.shape == (n_transducers, 3)

    grid_coords = build_grid_coords(extent_m, grid_shape)

    # Normalizer
    norm = np.load(norm_path)
    mean = norm["mean"].astype(np.float32)
    std = norm["std"].astype(np.float32)
    eps = float(norm["eps"])

    model = PhaseConditionedFNO(
        field_dims=3,
        grid_shape=grid_shape,
        n_transducers=n_transducers,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        cond_channels=cond_channels,
        out_channels=out_channels,
        transducer_positions=tx_positions,
        grid_coords=grid_coords,
        wavenumber=wavenumber,
        slice_z=slice_z,
        residual_prior=True,
        prior_norm_mean=mean,
        prior_norm_std=std,
        prior_version=prior_version,
    ).to(device)

    state_dict = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    model.load_state_dict(state_dict)
    model.eval()
    model._warned_once = True
    return model, mean, std, eps


def denormalize(field_norm: torch.Tensor, mean: np.ndarray, std: np.ndarray, eps: float) -> torch.Tensor:
    """Convert from normalized-space (B, 2, *grid) to physical (Pa)."""
    m = torch.from_numpy(mean).view(1, 2, *([1] * (field_norm.dim() - 2))).to(field_norm.device)
    s = torch.from_numpy(np.maximum(std, eps)).view(1, 2, *([1] * (field_norm.dim() - 2))).to(field_norm.device)
    return field_norm * s + m


def resample_to_focal_zone(
    field_j: torch.Tensor,  # (B, 2, 44, 44, 144) at L1 extent
    j_extent_m: Tuple[float, float, float, float, float, float],
    f_extent_m: Tuple[float, float, float, float, float, float],
    f_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Trilinear-interpolate FNO_J's full-chamber output onto FNO_F's
    focal-zone grid. Uses torch.nn.functional.grid_sample.

    The trick: build a sampling grid in normalized [-1, 1] coordinates
    relative to FNO_J's extent, where each sample point corresponds to
    a voxel of FNO_F's focal-zone grid.
    """
    import torch.nn.functional as F

    j_xmin, j_xmax, j_ymin, j_ymax, j_zmin, j_zmax = j_extent_m
    f_xmin, f_xmax, f_ymin, f_ymax, f_zmin, f_zmax = f_extent_m
    fx, fy, fz = f_shape

    # FNO_F's focal-zone voxel coordinates in physical space
    x_f = np.linspace(f_xmin, f_xmax, fx, dtype=np.float32)
    y_f = np.linspace(f_ymin, f_ymax, fy, dtype=np.float32)
    z_f = np.linspace(f_zmin, f_zmax, fz, dtype=np.float32)

    # Normalize to FNO_J's extent ([-1, 1]) for grid_sample
    nx = 2.0 * (x_f - j_xmin) / (j_xmax - j_xmin) - 1.0
    ny = 2.0 * (y_f - j_ymin) / (j_ymax - j_ymin) - 1.0
    nz = 2.0 * (z_f - j_zmin) / (j_zmax - j_zmin) - 1.0

    # grid_sample expects (N, D_out, H_out, W_out, 3) with order (x, y, z)
    # Note: PyTorch's grid_sample 3D uses (W, H, D) layout for axes 0,1,2;
    # here we interpret FNO axes as (x, y, z) → grid_sample's (W, H, D).
    Z, Y, X = np.meshgrid(nz, ny, nx, indexing="ij")  # (fz, fy, fx)
    # Build (D_out, H_out, W_out, 3) = (fz, fy, fx, 3) with last dim (x,y,z)
    grid = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    grid_t = torch.from_numpy(grid).unsqueeze(0).to(field_j.device)  # (1, fz, fy, fx, 3)

    # field_j is (B, 2, jx, jy, jz). grid_sample expects (B, C, D, H, W).
    # Map (jx, jy, jz) → (D=jx, H=jy, W=jz)? Need to think carefully about axis order.
    # We built grid with last-dim order (x, y, z) and grid normalized [-1,1] over jx, jy, jz.
    # By PyTorch convention, grid_sample 3D treats input as (B, C, D, H, W); the last
    # element of grid is W (which maps to input's W axis = -1 = z), middle is H = y,
    # first is D = x. So our grid order matches: (x, y, z) → (W, H, D)? No — last is W.
    # Convention: grid[..., 0] = W coord, grid[..., 1] = H coord, grid[..., 2] = D coord.
    # We want grid[..., 0] to index along x (input axis -3 = D), and we want grid[..., 2]
    # to index along z (input axis -1 = W). So we need order (z, y, x) on the last dim.
    grid_t = torch.from_numpy(np.stack([Z, Y, X], axis=-1).astype(np.float32))
    grid_t = grid_t.unsqueeze(0).to(field_j.device)  # (1, fz, fy, fx, 3)

    # Replicate batch
    B = field_j.shape[0]
    grid_t = grid_t.expand(B, -1, -1, -1, -1)

    # grid_sample: input (B, C, D, H, W). FNO_J output is (B, 2, jx, jy, jz)
    # → treat (jx, jy, jz) as (D, H, W). Output shape (B, 2, fz, fy, fx).
    out = F.grid_sample(field_j, grid_t, mode="bilinear", align_corners=True)
    # out is (B, 2, fz, fy, fx). FNO_F's output is (B, 2, fx, fy, fz).
    # Permute to match FNO_F's axis convention.
    out = out.permute(0, 1, 4, 3, 2).contiguous()  # (B, 2, fx, fy, fz)
    return out


def rel_l2_per_sample(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Per-sample relative L2 = ||a-b||_2 / ||b||_2 (using b as reference)."""
    diff = a - b
    num = torch.sqrt((diff ** 2).flatten(1).sum(dim=1))
    den = torch.sqrt((b ** 2).flatten(1).sum(dim=1)).clamp_min(1e-12)
    return (num / den).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-f", required=True)
    ap.add_argument("--norm-f", required=True)
    ap.add_argument("--ckpt-j", required=True)
    ap.add_argument("--norm-j", required=True)
    ap.add_argument("--phases-h5", required=True, help="HDF5 with 'phases' (N, 120)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # FNO_F: 32³ over focal zone (-30..30, -30..30, 100..300) mm
    f_extent = (-0.03, 0.03, -0.03, 0.03, 0.10, 0.30)
    f_shape = (32, 32, 32)
    model_f, mean_f, std_f, eps_f = load_fno_manual(
        args.ckpt_f, args.norm_f,
        grid_shape=f_shape, extent_m=f_extent,
        n_modes=(15, 15, 15), hidden_channels=128,
        prior_version="v2.1_3d", device=device,
    )
    print(f"[load] FNO_F (32³ focal-zone): ok")

    # FNO_J L1: (44, 44, 144) over (-66..66, -66..66, 0..432) mm
    j_extent = (-0.066, 0.066, -0.066, 0.066, 0.0, 0.432)
    j_shape = (44, 44, 144)
    model_j, mean_j, std_j, eps_j = load_fno_manual(
        args.ckpt_j, args.norm_j,
        grid_shape=j_shape, extent_m=j_extent,
        n_modes=(8, 8, 24), hidden_channels=128,
        prior_version="v2.1_3d", device=device,
    )
    print(f"[load] FNO_J L1 (44×44×144): ok")

    # Pull test phases
    with h5py.File(args.phases_h5, "r") as h5:
        all_phases = h5["phases"][:].astype(np.float32)  # (N_total, 120)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(all_phases), size=min(args.n_samples, len(all_phases)), replace=False)
    idx.sort()
    phases = torch.from_numpy(all_phases[idx]).to(device)  # (N, 120)
    print(f"[data] {len(phases)} test phases from {args.phases_h5} (indices: {idx[:5].tolist()}...)")

    # Forward both (batched)
    rl2_values = []
    with torch.no_grad():
        for i in range(len(phases)):
            x = phases[i:i+1]  # (1, 120)
            out_f_n = model_f(x)  # (1, 2, 32, 32, 32) normalized
            out_j_n = model_j(x)  # (1, 2, 44, 44, 144) normalized
            # Denormalize to physical Pa
            out_f = denormalize(out_f_n, mean_f, std_f, eps_f)
            out_j = denormalize(out_j_n, mean_j, std_j, eps_j)
            # Resample FNO_J to FNO_F's focal-zone grid
            out_j_focal = resample_to_focal_zone(out_j, j_extent, f_extent, f_shape)
            # Relative L2 (using FNO_F as reference)
            rl2 = rel_l2_per_sample(out_j_focal, out_f)
            rl2_values.append(float(rl2[0]))
            print(f"  sample {i+1}/{len(phases)}  rel_l2(FNO_J_focal vs FNO_F) = {rl2[0]:.4f}")

    arr = np.array(rl2_values)
    summary = {
        "mode": "fno_F_vs_fno_J_focal_overlap",
        "n_samples": len(arr),
        "rel_l2_phys_mean":   float(arr.mean()),
        "rel_l2_phys_median": float(np.median(arr)),
        "rel_l2_phys_p90":    float(np.quantile(arr, 0.9)),
        "rel_l2_phys_max":    float(arr.max()),
        "rel_l2_phys_min":    float(arr.min()),
        "indices": idx.tolist(),
        "per_sample_rel_l2_phys": rl2_values,
        "fno_F_extent_m": f_extent,
        "fno_F_grid_shape": f_shape,
        "fno_J_extent_m": j_extent,
        "fno_J_grid_shape": j_shape,
        "comparison_region": "FNO_F native focal zone (60×60×200mm at 32³)",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] mean rel_l2 = {arr.mean():.4f}  median = {np.median(arr):.4f}")
    print(f"[done] wrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    sys.exit(main())
