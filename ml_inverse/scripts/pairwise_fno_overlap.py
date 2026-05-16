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


def rel_l2_scale_normalized(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Per-sample rel_l2 after L2-normalizing both fields.

    Factors out amplitude / scale mismatch: if a and b are the same
    *shape* but different magnitudes, this returns ~0. If they're
    genuinely different shapes, this stays near the unnormalized rel_l2.
    """
    norm_a = torch.sqrt((a ** 2).flatten(1).sum(dim=1)).clamp_min(1e-12)
    norm_b = torch.sqrt((b ** 2).flatten(1).sum(dim=1)).clamp_min(1e-12)
    a_hat = a / norm_a.view(-1, *[1] * (a.dim() - 1))
    b_hat = b / norm_b.view(-1, *[1] * (b.dim() - 1))
    diff = a_hat - b_hat
    return torch.sqrt((diff ** 2).flatten(1).sum(dim=1)).cpu().numpy()


def field_norm(x: torch.Tensor) -> np.ndarray:
    """||x||_2 per sample (B, 2, ...)  -> (B,)."""
    return torch.sqrt((x ** 2).flatten(1).sum(dim=1)).cpu().numpy()


def render_compare_slice(
    out_f: torch.Tensor,  # (1, 2, fx, fy, fz)
    out_j_focal: torch.Tensor,  # (1, 2, fx, fy, fz)
    sample_idx: int,
    real_idx: int,
    out_dir: Path,
) -> None:
    """Save a 3-panel mid-z slice of |P| for FNO_F, FNO_J_focal, and |diff|."""
    import matplotlib.pyplot as plt

    z_mid = out_f.shape[-1] // 2
    p_f = torch.sqrt(out_f[0, 0, :, :, z_mid] ** 2 + out_f[0, 1, :, :, z_mid] ** 2).cpu().numpy()
    p_j = torch.sqrt(out_j_focal[0, 0, :, :, z_mid] ** 2 + out_j_focal[0, 1, :, :, z_mid] ** 2).cpu().numpy()
    diff = np.abs(p_f - p_j)
    # Shared colorbar for the two intensity panels for easy visual comparison
    vmax = max(p_f.max(), p_j.max())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    im0 = axes[0].imshow(p_f.T, origin="lower", vmin=0, vmax=vmax, cmap="viridis")
    axes[0].set_title(f"|P| FNO_F  ||={np.linalg.norm(p_f):.1e}")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(p_j.T, origin="lower", vmin=0, vmax=vmax, cmap="viridis")
    axes[1].set_title(f"|P| FNO_J_focal  ||={np.linalg.norm(p_j):.1e}")
    plt.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(diff.T, origin="lower", cmap="magma")
    axes[2].set_title(f"||P_F| - |P_J||  max={diff.max():.1e}")
    plt.colorbar(im2, ax=axes[2])
    fig.suptitle(f"sample idx={real_idx}, z mid-slice")
    out_path = out_dir / f"compare_slice_idx{real_idx:05d}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-f", required=True)
    ap.add_argument("--norm-f", required=True)
    ap.add_argument("--ckpt-j", required=True)
    ap.add_argument("--norm-j", required=True)
    ap.add_argument("--phases-h5", required=True, help="HDF5 with 'phases' (N, 120)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--n-render", type=int, default=0,
                    help="Render mid-z compare slice for the first N samples (0 = no rendering).")
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

    # Forward both, with sanity-check metrics + optional rendering
    rl2_values = []
    rl2_scale_norm_values = []
    norm_f_values = []
    norm_j_values = []
    with torch.no_grad():
        for i in range(len(phases)):
            x = phases[i:i+1]  # (1, 120)
            out_f_n = model_f(x)  # (1, 2, 32, 32, 32) normalized
            out_j_n = model_j(x)  # (1, 2, 44, 44, 144) normalized
            out_f = denormalize(out_f_n, mean_f, std_f, eps_f)
            out_j = denormalize(out_j_n, mean_j, std_j, eps_j)
            out_j_focal = resample_to_focal_zone(out_j, j_extent, f_extent, f_shape)

            rl2 = rel_l2_per_sample(out_j_focal, out_f)
            rl2_scaled = rel_l2_scale_normalized(out_j_focal, out_f)
            nf = field_norm(out_f)
            nj = field_norm(out_j_focal)

            rl2_values.append(float(rl2[0]))
            rl2_scale_norm_values.append(float(rl2_scaled[0]))
            norm_f_values.append(float(nf[0]))
            norm_j_values.append(float(nj[0]))

            print(f"  sample {i+1:2d}/{len(phases)}  rl2={rl2[0]:.4f}  rl2_scale_norm={rl2_scaled[0]:.4f}  "
                  f"||P_F||={nf[0]:.2e}  ||P_J||={nj[0]:.2e}  ratio_J/F={nj[0]/max(nf[0], 1e-12):.3f}")

            if i < args.n_render:
                render_compare_slice(out_f, out_j_focal, sample_idx=i, real_idx=int(idx[i]), out_dir=out_dir)

    arr = np.array(rl2_values)
    arr_sn = np.array(rl2_scale_norm_values)
    nf_arr = np.array(norm_f_values)
    nj_arr = np.array(norm_j_values)
    ratio = nj_arr / np.clip(nf_arr, 1e-12, None)

    summary = {
        "mode": "fno_F_vs_fno_J_focal_overlap",
        "n_samples": len(arr),
        "rel_l2_phys_mean":   float(arr.mean()),
        "rel_l2_phys_median": float(np.median(arr)),
        "rel_l2_phys_p90":    float(np.quantile(arr, 0.9)),
        "rel_l2_phys_max":    float(arr.max()),
        "rel_l2_phys_min":    float(arr.min()),
        "rel_l2_scale_normalized_mean":   float(arr_sn.mean()),
        "rel_l2_scale_normalized_median": float(np.median(arr_sn)),
        "rel_l2_scale_normalized_p90":    float(np.quantile(arr_sn, 0.9)),
        "field_norm_F_mean": float(nf_arr.mean()),
        "field_norm_J_focal_mean": float(nj_arr.mean()),
        "amplitude_ratio_J_over_F_mean": float(ratio.mean()),
        "amplitude_ratio_J_over_F_median": float(np.median(ratio)),
        "indices": idx.tolist(),
        "per_sample_rel_l2_phys": rl2_values,
        "per_sample_rel_l2_scale_normalized": rl2_scale_norm_values,
        "per_sample_field_norm_F": norm_f_values,
        "per_sample_field_norm_J_focal": norm_j_values,
        "per_sample_amplitude_ratio_J_over_F": ratio.tolist(),
        "fno_F_extent_m": f_extent,
        "fno_F_grid_shape": f_shape,
        "fno_J_extent_m": j_extent,
        "fno_J_grid_shape": j_shape,
        "comparison_region": "FNO_F native focal zone (60×60×200mm at 32³)",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f"[done] rl2_phys                mean={arr.mean():.4f}  median={np.median(arr):.4f}")
    print(f"[done] rl2_scale_normalized    mean={arr_sn.mean():.4f}  median={np.median(arr_sn):.4f}")
    print(f"[done] ||P_F||                 mean={nf_arr.mean():.2e}  median={np.median(nf_arr):.2e}")
    print(f"[done] ||P_J_focal||           mean={nj_arr.mean():.2e}  median={np.median(nj_arr):.2e}")
    print(f"[done] amplitude_ratio J/F     mean={ratio.mean():.3f}  median={np.median(ratio):.3f}")
    print(f"[done] wrote {out_dir/'summary.json'}")
    if args.n_render > 0:
        print(f"[done] rendered {min(args.n_render, len(phases))} mid-z slice PNGs in {out_dir}/")


if __name__ == "__main__":
    sys.exit(main())
