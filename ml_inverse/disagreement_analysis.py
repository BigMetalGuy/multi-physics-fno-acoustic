"""Two-surrogate disagreement analysis.

Framework for "disagreement = signal" research direction. Two
forward-pressure surrogates trained against different physics regimes
should produce identical outputs in regions where the physics agrees,
and diverge where one model captures something the other does not.
The disagreement field, computed in a common normalized space, is a
proxy for "where does physics matter most?"

Two operating modes
===================

* ``fno-vs-analytical`` — FNO checkpoint vs. the analytical free-field
  formula (the same prior used at training time). Used as a sanity
  probe: we expect tiny disagreement because the v2.1 model learned a
  near-zero residual against the analytical prior.

* ``fno-vs-fno`` — Two FNO checkpoints. The intended downstream use is
  ``FNO_v2.1`` (analytical-trained) vs. ``FNO_jwave`` (PDE-trained).
  Currently stubbed: a follow-up agent will run it after Track A's
  cloud sprint.

Outputs
=======

* JSON manifest: per-config disagreement statistics (relative L2,
  per-pixel mean/median/max, focal-region attribution, in-array fraction).
* Aggregate stats CSV row.
* 4-panel heatmap PNGs: ``model_a |P|``, ``model_b |P|``,
  ``|residual| (Pa)``, ``log10(|residual| + eps)``.

CLI
===

::

    python -m ml_inverse.disagreement_analysis \\
        --mode fno-vs-analytical \\
        --fno-ckpt ml_inverse/models/fno_surrogate_v2_1.pt \\
        --norm ml_inverse/models/fno_surrogate_v2_1_norm.npz \\
        --dataset ml_inverse/data/inverse_dataset_v3.h5 \\
        --output-dir ml_inverse/research/_disagreement/ \\
        --n-samples 50 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Module-level logger; CLI configures the root handler.
logger = logging.getLogger("ml_inverse.disagreement")


# --------------------------------------------------------------------------
# Constants — keep in one place rather than scattered "magic numbers".
# --------------------------------------------------------------------------

#: Numerical floor used for log/log10 plotting and divide-by-zero guards.
LOG_EPSILON: float = 1e-9

#: Threshold defining a "high disagreement pixel" — top quantile of |residual|.
HIGH_QUANTILE: float = 0.95

#: Phase 1 v3 dataset extent (m) — used to compute physical pixel coordinates.
DEFAULT_EXTENT_M: Tuple[float, float, float, float] = (-0.025, 0.025, -0.025, 0.025)

#: L1 array cylinder radius (m). Used for "in-array" attribution.
DEFAULT_ARRAY_RADIUS_M: float = 0.05

#: Focal-region detection: pixel is considered "near-focal" if it is within
#: this fraction of |peak| in the model_a magnitude field.
FOCAL_PEAK_FRACTION: float = 0.7


def _json_safe(value: Any) -> Any:
    """Coerce numpy / tuple values into JSON-serializable primitives."""
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


# --------------------------------------------------------------------------
# Data containers — frozen dataclasses, immutable by convention.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PerSampleStats:
    """Disagreement summary for a single test config.

    Attributes
    ----------
    sample_index
        Row in the source HDF5 dataset.
    relative_l2_normalized
        ``||a_norm - b_norm||_2 / ||b_norm||_2``. Computed in the
        common normalized space so the two channels are comparable.
    relative_l2_physical
        Same but in physical (Pa) units; computed on the magnitude
        field ``|P|``. Useful for human interpretability.
    abs_residual_mean_pa
        Per-pixel ``mean |a - b|`` in physical units (Pa).
    abs_residual_median_pa
        Per-pixel ``median |a - b|`` (Pa).
    abs_residual_max_pa
        Per-pixel ``max |a - b|`` (Pa).
    focal_overlap_fraction
        Fraction of high-disagreement pixels that fall inside the
        focal region (defined by ``FOCAL_PEAK_FRACTION``).
    in_array_fraction
        Fraction of high-disagreement pixels whose physical location
        is inside the cylindrical array footprint.
    """

    sample_index: int
    relative_l2_normalized: float
    relative_l2_physical: float
    abs_residual_mean_pa: float
    abs_residual_median_pa: float
    abs_residual_max_pa: float
    focal_overlap_fraction: float
    in_array_fraction: float


@dataclass(frozen=True)
class AggregateStats:
    """Per-run aggregate over all sampled configs."""

    n_samples: int
    rel_l2_norm_mean: float
    rel_l2_norm_median: float
    rel_l2_norm_max: float
    rel_l2_norm_p95: float
    rel_l2_phys_mean: float
    rel_l2_phys_median: float
    rel_l2_phys_max: float
    abs_residual_max_pa_aggregate: float
    focal_overlap_mean: float
    in_array_mean: float


@dataclass(frozen=True)
class DisagreementResult:
    """Full result envelope for a comparison run."""

    mode: str
    aggregate: AggregateStats
    per_sample: Tuple[PerSampleStats, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Device selection — MPS-aware with graceful CPU fallback.
# --------------------------------------------------------------------------


def auto_device(arg: Optional[str] = None, *, mps_busy_threshold_bytes: int = 4 * 1024**3) -> torch.device:
    """Select compute device with MPS-busy detection.

    If MPS is requested or auto-selected and ``torch.mps.current_allocated_memory()``
    exceeds ``mps_busy_threshold_bytes``, fall back to CPU. This keeps Track B's
    MPS work uncontested.
    """
    if arg is not None:
        return torch.device(arg)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            allocated = int(torch.mps.current_allocated_memory())
        except Exception:  # pragma: no cover - depends on torch version
            allocated = 0
        if allocated > mps_busy_threshold_bytes:
            logger.warning(
                "MPS allocated memory %.2f GB exceeds threshold %.2f GB; "
                "falling back to CPU.",
                allocated / 1024**3,
                mps_busy_threshold_bytes / 1024**3,
            )
            return torch.device("cpu")
        return torch.device("mps")

    return torch.device("cpu")


# --------------------------------------------------------------------------
# Core disagreement computation — model-agnostic.
# --------------------------------------------------------------------------


PressurePredictor = Callable[[torch.Tensor], torch.Tensor]
"""Callable signature: takes ``x`` ``(B, input_dim)``, returns
``(B, 2, *grid_shape)`` field in normalized space."""

Denormalizer = Callable[[torch.Tensor], torch.Tensor]
"""Callable signature: takes a single ``(2, *grid)`` tensor, returns
the field in physical (Pa) units."""


def compute_disagreement(
    model_a: PressurePredictor,
    model_b: PressurePredictor,
    test_inputs: torch.Tensor,
    *,
    denormalize: Denormalizer,
    grid_extent_m: Optional[Tuple[float, ...]] = None,
    array_radius_m: float = DEFAULT_ARRAY_RADIUS_M,
    high_quantile: float = HIGH_QUANTILE,
    focal_peak_fraction: float = FOCAL_PEAK_FRACTION,
    device: Optional[torch.device] = None,
) -> Tuple[
    Tuple[PerSampleStats, ...],
    AggregateStats,
    np.ndarray,  # all_normalized_residuals  (N, 2, *spatial)
    np.ndarray,  # all_physical_residuals    (N, *spatial) magnitude diff
    np.ndarray,  # all_a_phys                (N, *spatial) magnitude
    np.ndarray,  # all_b_phys                (N, *spatial) magnitude
]:
    """Compute per-sample and aggregate disagreement statistics.

    Both predictors must produce normalized fields on the same grid; the
    ``denormalize`` callback is used to convert each pair to physical
    units before computing the magnitude residual.

    ``test_inputs`` is the full batch (already in the input format each
    predictor expects, e.g. ``(B, n_transducers + 1)`` for v2.1). For
    fairness we run each predictor on the same input once.

    Spatial dimensionality is inferred from the predictor outputs: 2D
    (``(B, 2, Nx, Ny)``) and 3D (``(B, 2, Nx, Ny, Nz)``) are both
    supported. The "in-array" mask uses only the in-plane (x, y) axes;
    a 3D field is checked column-wise (a (z, *) column is "in array" if
    its (x, y) projection is inside the cylindrical footprint).

    ``grid_extent_m`` is ``(x_min, x_max, y_min, y_max)`` for 2D or
    ``(x_min, x_max, y_min, y_max, z_min, z_max)`` for 3D. Defaults to
    the v3 dataset extent for 2D inputs.

    Returns six values; ``all_a_phys`` and ``all_b_phys`` are the
    magnitude fields (Pa), and ``all_physical_residuals`` is
    ``|a_phys| - |b_phys|`` (signed, useful for diverging colormaps).
    """
    if test_inputs.ndim != 2:
        raise ValueError(f"test_inputs must be 2D (B, D); got {tuple(test_inputs.shape)}")
    if device is None:
        device = test_inputs.device

    test_inputs = test_inputs.to(device)
    n = int(test_inputs.shape[0])

    # Default extent: 2D v3 dataset extent; for 3D the caller should pass extent.
    if grid_extent_m is None:
        grid_extent_m = DEFAULT_EXTENT_M

    per_sample: List[PerSampleStats] = []
    res_norm_list: List[np.ndarray] = []
    res_phys_list: List[np.ndarray] = []
    a_phys_list: List[np.ndarray] = []
    b_phys_list: List[np.ndarray] = []

    rel_l2_norm: List[float] = []
    rel_l2_phys: List[float] = []
    abs_res_mean: List[float] = []
    abs_res_median: List[float] = []
    abs_res_max: List[float] = []
    focal_overlap: List[float] = []
    in_array_frac: List[float] = []

    for i in range(n):
        x_i = test_inputs[i : i + 1]  # keep batch dim (1, D)
        with torch.no_grad():
            a_norm = model_a(x_i).detach()
            b_norm = model_b(x_i).detach()

        if a_norm.shape != b_norm.shape:
            raise ValueError(
                f"model outputs differ in shape: a={tuple(a_norm.shape)}, "
                f"b={tuple(b_norm.shape)}"
            )

        # Strip batch dim → (2, *grid)
        a_n = a_norm[0]
        b_n = b_norm[0]

        # Normalized-space relative L2: invariant to channel scale.
        diff_n = (a_n - b_n).reshape(-1)
        denom_n = b_n.reshape(-1).norm().clamp(min=LOG_EPSILON)
        rl2_n = float((diff_n.norm() / denom_n).item())

        # Physical-space (denormalize → magnitude → relative L2).
        a_phys = denormalize(a_n)
        b_phys = denormalize(b_n)
        a_mag = torch.sqrt(a_phys[0] ** 2 + a_phys[1] ** 2)
        b_mag = torch.sqrt(b_phys[0] ** 2 + b_phys[1] ** 2)
        diff_p = (a_mag - b_mag).reshape(-1)
        denom_p = b_mag.reshape(-1).norm().clamp(min=LOG_EPSILON)
        rl2_p = float((diff_p.norm() / denom_p).item())

        abs_res = (a_mag - b_mag).abs()
        a_mag_np = a_mag.cpu().numpy()
        b_mag_np = b_mag.cpu().numpy()
        abs_res_np = abs_res.cpu().numpy()

        # ---- High-disagreement pixel attribution ----
        threshold = float(np.quantile(abs_res_np, high_quantile))
        high_mask = abs_res_np >= threshold

        # Focal region defined on model_a magnitude field.
        peak_a = float(a_mag_np.max())
        focal_mask = a_mag_np >= focal_peak_fraction * peak_a
        if high_mask.sum() > 0:
            focal_overlap_i = float((high_mask & focal_mask).sum() / high_mask.sum())
        else:
            focal_overlap_i = 0.0

        # In-array attribution by physical pixel coordinates. The footprint
        # is (x, y) cylindrical; for 3D fields we broadcast across z.
        in_array_mask = _build_in_array_mask(
            grid_shape=a_mag_np.shape,
            extent_m=grid_extent_m,
            array_radius_m=array_radius_m,
        )
        if high_mask.sum() > 0:
            in_array_i = float((high_mask & in_array_mask).sum() / high_mask.sum())
        else:
            in_array_i = 0.0

        per_sample.append(
            PerSampleStats(
                sample_index=i,
                relative_l2_normalized=rl2_n,
                relative_l2_physical=rl2_p,
                abs_residual_mean_pa=float(abs_res_np.mean()),
                abs_residual_median_pa=float(np.median(abs_res_np)),
                abs_residual_max_pa=float(abs_res_np.max()),
                focal_overlap_fraction=focal_overlap_i,
                in_array_fraction=in_array_i,
            )
        )

        rel_l2_norm.append(rl2_n)
        rel_l2_phys.append(rl2_p)
        abs_res_mean.append(float(abs_res_np.mean()))
        abs_res_median.append(float(np.median(abs_res_np)))
        abs_res_max.append(float(abs_res_np.max()))
        focal_overlap.append(focal_overlap_i)
        in_array_frac.append(in_array_i)

        # Stack for downstream plotting / aggregate analysis.
        res_norm_list.append((a_n - b_n).cpu().numpy())
        res_phys_list.append((a_mag_np - b_mag_np))
        a_phys_list.append(a_mag_np)
        b_phys_list.append(b_mag_np)

    aggregate = AggregateStats(
        n_samples=n,
        rel_l2_norm_mean=float(np.mean(rel_l2_norm)),
        rel_l2_norm_median=float(np.median(rel_l2_norm)),
        rel_l2_norm_max=float(np.max(rel_l2_norm)),
        rel_l2_norm_p95=float(np.percentile(rel_l2_norm, 95)),
        rel_l2_phys_mean=float(np.mean(rel_l2_phys)),
        rel_l2_phys_median=float(np.median(rel_l2_phys)),
        rel_l2_phys_max=float(np.max(rel_l2_phys)),
        abs_residual_max_pa_aggregate=float(np.max(abs_res_max)),
        focal_overlap_mean=float(np.mean(focal_overlap)),
        in_array_mean=float(np.mean(in_array_frac)),
    )

    return (
        tuple(per_sample),
        aggregate,
        np.stack(res_norm_list, axis=0),
        np.stack(res_phys_list, axis=0),
        np.stack(a_phys_list, axis=0),
        np.stack(b_phys_list, axis=0),
    )


def _build_in_array_mask(
    *,
    grid_shape: Tuple[int, ...],
    extent_m: Tuple[float, ...],
    array_radius_m: float,
) -> np.ndarray:
    """True where the pixel center sits inside the cylindrical array footprint.

    Handles 2D ``(Nx, Ny)`` and 3D ``(Nx, Ny, Nz)`` grids. The cylindrical
    footprint is in the (x, y) plane only — for 3D, the 2D footprint mask
    is broadcast across the z axis (every voxel in a column shares the
    same in/out classification).

    ``extent_m`` is ``(x_min, x_max, y_min, y_max)`` for 2D or
    ``(x_min, x_max, y_min, y_max, z_min, z_max)`` for 3D. Only the
    (x, y) bounds are used; the z bounds are accepted for symmetry but
    ignored.
    """
    if len(grid_shape) == 2:
        nx, ny = grid_shape
        if len(extent_m) < 4:
            raise ValueError(
                f"2D grid requires 4-tuple extent (x_min,x_max,y_min,y_max); got {extent_m}"
            )
        x_min, x_max, y_min, y_max = extent_m[:4]
        xs = np.linspace(x_min, x_max, nx, dtype=np.float32)
        ys = np.linspace(y_min, y_max, ny, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys, indexing="ij")
        return (xv**2 + yv**2) <= array_radius_m**2

    if len(grid_shape) == 3:
        nx, ny, nz = grid_shape
        if len(extent_m) < 4:
            raise ValueError(
                f"3D grid requires at least 4-tuple extent for (x,y); got {extent_m}"
            )
        x_min, x_max, y_min, y_max = extent_m[:4]
        xs = np.linspace(x_min, x_max, nx, dtype=np.float32)
        ys = np.linspace(y_min, y_max, ny, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys, indexing="ij")
        mask_xy = (xv**2 + yv**2) <= array_radius_m**2  # (Nx, Ny)
        # Broadcast across z.
        return np.broadcast_to(mask_xy[:, :, None], (nx, ny, nz)).copy()

    raise ValueError(
        f"unsupported grid_shape ndim={len(grid_shape)} (expected 2 or 3)"
    )


def attribution_heatmap(
    residual_phys: np.ndarray,
    *,
    a_magnitude: np.ndarray,
    grid_extent_m: Optional[Tuple[float, ...]] = None,
    array_radius_m: float = DEFAULT_ARRAY_RADIUS_M,
    high_quantile: float = HIGH_QUANTILE,
    focal_peak_fraction: float = FOCAL_PEAK_FRACTION,
) -> Dict[str, Any]:
    """Categorize where high-disagreement pixels land.

    Spatially-N-D: accepts ``(N, *spatial)`` for 2D ``(Nx, Ny)`` or 3D
    ``(Nx, Ny, Nz)`` magnitude fields. The "focal region" mask is the
    set of voxels where ``a_magnitude >= focal_peak_fraction * peak_a``;
    the "in-array" mask uses the (x, y) cylindrical footprint only and
    broadcasts across z for 3D.

    Parameters
    ----------
    residual_phys
        Per-sample magnitude residual ``(N, *spatial)``.
    a_magnitude
        Per-sample magnitude field of model A, same shape.
    """
    if a_magnitude.shape != residual_phys.shape:
        raise ValueError(
            f"shape mismatch: residual_phys={residual_phys.shape}, "
            f"a_magnitude={a_magnitude.shape}"
        )
    if residual_phys.ndim < 3:
        raise ValueError(
            f"residual_phys must be (N, *spatial); got ndim={residual_phys.ndim}"
        )
    if grid_extent_m is None:
        grid_extent_m = DEFAULT_EXTENT_M

    abs_res = np.abs(residual_phys)
    n = abs_res.shape[0]
    spatial_shape = abs_res.shape[1:]

    in_array_mask = _build_in_array_mask(
        grid_shape=spatial_shape,
        extent_m=grid_extent_m,
        array_radius_m=array_radius_m,
    )  # (*spatial)

    # Per-sample threshold: top-quantile of that sample's |residual|.
    flat = abs_res.reshape(n, -1)
    thresholds = np.quantile(flat, high_quantile, axis=1)  # (N,)
    # Broadcast (N,) -> (N, 1, 1, ...) over all spatial axes.
    threshold_view = (slice(None),) + (None,) * len(spatial_shape)
    high_masks = abs_res >= thresholds[threshold_view]  # (N, *spatial)

    peak_per_sample = a_magnitude.reshape(n, -1).max(axis=1)
    focal_masks = a_magnitude >= (
        focal_peak_fraction * peak_per_sample[threshold_view]
    )

    # Stack-level fractions. Broadcast in_array (no leading N axis) by
    # adding the batch axis up front.
    in_array_batched = in_array_mask[None, ...]  # (1, *spatial)
    high_total = high_masks.sum()
    focal_count = (high_masks & focal_masks).sum()
    in_array_count = (high_masks & in_array_batched).sum()
    out_array_count = (high_masks & ~in_array_batched).sum()

    # Aggregate "where do high-disagreement pixels live, on average?"
    avg_residual_field = abs_res.mean(axis=0)  # (*spatial)
    return {
        "high_quantile": float(high_quantile),
        "focal_peak_fraction": float(focal_peak_fraction),
        "high_pixels_total": int(high_total),
        "high_pixels_focal_overlap": int(focal_count),
        "high_pixels_in_array": int(in_array_count),
        "high_pixels_out_array": int(out_array_count),
        "fraction_focal": float(focal_count / max(high_total, 1)),
        "fraction_in_array": float(in_array_count / max(high_total, 1)),
        "fraction_out_array": float(out_array_count / max(high_total, 1)),
        "avg_residual_field_max_pa": float(avg_residual_field.max()),
        "avg_residual_field_mean_pa": float(avg_residual_field.mean()),
        "avg_residual_field": avg_residual_field,
        "spatial_shape": tuple(int(s) for s in spatial_shape),
    }


# --------------------------------------------------------------------------
# Rendering — matplotlib heatmaps. Lazy imports to avoid Agg side effects
# in callers that don't need rendering.
# --------------------------------------------------------------------------


def render_disagreement_map(
    *,
    sample_index: int,
    a_magnitude: np.ndarray,
    b_magnitude: np.ndarray,
    residual_phys: np.ndarray,
    output_dir: Path,
    extent_m: Optional[Tuple[float, ...]] = None,
    label_a: str = "model A",
    label_b: str = "model B",
) -> Path:
    """Produce a 4-panel PNG: A, B, |residual|, log10|residual|.

    Accepts 2D ``(Nx, Ny)`` or 3D ``(Nx, Ny, Nz)`` magnitude fields. For
    3D inputs we render the mid-z slice (``z = Nz // 2``); the slice
    suffix ``_midz`` is added to the output filename.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if a_magnitude.shape != b_magnitude.shape or residual_phys.shape != a_magnitude.shape:
        raise ValueError(
            f"shape mismatch: a={a_magnitude.shape}, b={b_magnitude.shape}, "
            f"residual={residual_phys.shape}"
        )
    if extent_m is None:
        extent_m = DEFAULT_EXTENT_M

    is_3d = a_magnitude.ndim == 3
    if is_3d:
        zmid = a_magnitude.shape[-1] // 2
        a_magnitude = a_magnitude[:, :, zmid]
        b_magnitude = b_magnitude[:, :, zmid]
        residual_phys = residual_phys[:, :, zmid]

    abs_res = np.abs(residual_phys)
    extent_mm = (
        extent_m[0] * 1000,
        extent_m[1] * 1000,
        extent_m[2] * 1000,
        extent_m[3] * 1000,
    )

    vmax = float(max(a_magnitude.max(), b_magnitude.max()))
    rmax = float(abs_res.max() + LOG_EPSILON)
    log_res = np.log10(abs_res + LOG_EPSILON)

    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2))

    panels: List[Tuple[np.ndarray, str, str, Tuple[float, float]]] = [
        (a_magnitude, f"{label_a} |P| (Pa)", "viridis", (0.0, vmax)),
        (b_magnitude, f"{label_b} |P| (Pa)", "viridis", (0.0, vmax)),
        (abs_res, "|residual| (Pa)", "magma", (0.0, rmax)),
        (
            log_res,
            "log10|residual| (Pa)",
            "magma",
            (float(log_res.min()), float(log_res.max())),
        ),
    ]
    for c, (img, title, cmap, vlim) in enumerate(panels):
        im = axs[c].imshow(
            img.T,
            origin="lower",
            extent=extent_mm,
            aspect="equal",
            cmap=cmap,
            vmin=vlim[0],
            vmax=vlim[1],
        )
        axs[c].set_title(title, fontsize=10)
        axs[c].set_xlabel("x (mm)")
        axs[c].set_ylabel("y (mm)")
        plt.colorbar(im, ax=axs[c], fraction=0.046)

    suffix = " (mid-z slice)" if is_3d else ""
    fig.suptitle(
        f"Disagreement: {label_a} vs {label_b} (sample idx={sample_index}){suffix}",
        fontsize=11,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    name_suffix = "_midz" if is_3d else ""
    out_path = output_dir / f"disagreement_idx{sample_index:04d}{name_suffix}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_attribution_summary(
    *,
    attribution: Dict[str, Any],
    output_dir: Path,
    label: str = "fno_vs_analytical",
    extent_m: Optional[Tuple[float, ...]] = None,
) -> Path:
    """Render the average residual field across all samples — the
    "where does disagreement consistently land?" map. For 3D fields,
    renders the mid-z slice."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if extent_m is None:
        extent_m = DEFAULT_EXTENT_M

    avg_field = attribution["avg_residual_field"]
    if avg_field.ndim == 3:
        avg_field = avg_field[:, :, avg_field.shape[-1] // 2]
    extent_mm = (
        extent_m[0] * 1000,
        extent_m[1] * 1000,
        extent_m[2] * 1000,
        extent_m[3] * 1000,
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(
        avg_field.T,
        origin="lower",
        extent=extent_mm,
        aspect="equal",
        cmap="magma",
    )
    ax.set_title(f"avg |residual| over N samples — {label}", fontsize=11)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, fraction=0.046, label="Pa")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"attribution_avg_residual_{label}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Predictor builders — wrap the FNO and the analytical formula behind
# the same ``PressurePredictor`` interface so ``compute_disagreement``
# stays model-agnostic.
# --------------------------------------------------------------------------


def _resolve_model_config(
    *,
    ckpt: Dict[str, Any],
    config_source: Optional[str],
    dataset_h5: Optional[str],
    device: torch.device,
) -> Dict[str, Any]:
    """Reconstruct a ``model_config`` for a best-only checkpoint.

    Strategy:
    1. If ``config_source`` is given and the file has ``model_config``,
       use that directly.
    2. Else if ``dataset_h5`` is given, read ``transducer_positions``,
       ``grid_coords``, frequency, sound speed from the dataset and
       infer the FNO architecture (``n_modes``, ``hidden_channels``,
       ``cond_channels``, ``out_channels``) from the state_dict shapes.
       ``prior_version`` is inferred from spatial dimensionality
       (3D → ``'v2.1_3d'``; 2D → ``'v2.1'`` if a slice_z column exists,
       otherwise ``'v2'``).
    3. Else raise.
    """
    if config_source is not None:
        ref = torch.load(config_source, map_location=device, weights_only=False)
        if "model_config" in ref:
            logger.info("loaded model_config from sibling ckpt: %s", config_source)
            return ref["model_config"]
        raise RuntimeError(
            f"--config-source ckpt does not contain 'model_config': {config_source}"
        )

    if dataset_h5 is None:
        raise RuntimeError(
            "checkpoint has no embedded model_config; pass --config-source "
            "(sibling full ckpt) or --dataset-config (HDF5 to derive geometry from)."
        )

    import h5py  # local import

    state_dict = ckpt["model_state_dict"]
    with h5py.File(dataset_h5, "r") as fh:
        attrs = dict(fh.attrs)
        tx = np.asarray(fh["transducer_positions"][()], dtype=np.float32)
        grid_coords_grp = fh["grid_coords"]
        # grid_coords is a group with /x, /y[, /z] datasets.
        coord_keys = sorted(grid_coords_grp.keys())
        grid_coords = [
            np.asarray(grid_coords_grp[k][()], dtype=np.float32) for k in coord_keys
        ]
    grid_shape = [int(c.shape[0]) for c in grid_coords]
    field_dims = int(attrs.get("field_dims", len(grid_shape)))
    if len(grid_shape) != field_dims:
        raise RuntimeError(
            f"dataset grid_coords ndim ({len(grid_shape)}) != field_dims attr "
            f"({field_dims}); dataset is malformed"
        )

    # Infer wavenumber from frequency / sound speed.
    freq = float(attrs.get("frequency_hz", 40_000.0))
    c0 = float(attrs.get("sound_speed_m_s", 343.0))
    wavenumber = 2.0 * np.pi * freq / c0

    # Infer architecture from state_dict shapes.
    # Convention (from PhaseConditionedFNO):
    #   cond_mlp.0.weight      : (cond_channels_proj, n_transducers)
    #   The FNO block lifts conditioning -> hidden_channels per stage.
    cond_w = state_dict.get("cond_mlp.0.weight")
    if cond_w is None:
        raise RuntimeError("state_dict missing 'cond_mlp.0.weight'; cannot infer arch")
    n_tx_inferred = int(cond_w.shape[1])
    if int(tx.shape[0]) != n_tx_inferred:
        logger.warning(
            "n_transducers from state_dict (%d) differs from dataset attr (%d); using state_dict",
            n_tx_inferred,
            int(tx.shape[0]),
        )
    n_transducers = n_tx_inferred

    # Hidden channels: read from FNO spectral-conv weight (most reliable).
    # Convention: convs.<i>.weight.tensor has shape (hidden, hidden, *modes).
    hidden_channels = None
    for k, v in state_dict.items():
        if "fno.fno_blocks.convs." in k and k.endswith(".weight.tensor"):
            hidden_channels = int(v.shape[0])
            break
        if "fno_blocks.convs." in k and "weight" in k and v.ndim >= 4:
            hidden_channels = int(v.shape[0])
            break
    if hidden_channels is None:
        # Fallback to lifting output (last linear in lifting MLP).
        for k, v in state_dict.items():
            if k.endswith("fno.lifting.fcs.1.weight"):
                hidden_channels = int(v.shape[0])
                break
    if hidden_channels is None:
        raise RuntimeError("could not infer hidden_channels from state_dict")

    # cond_channels: output of cond_mlp final linear.
    cond_channels = None
    cond_keys = sorted(
        [k for k in state_dict.keys() if k.startswith("cond_mlp.") and k.endswith(".weight")]
    )
    if cond_keys:
        cond_channels = int(state_dict[cond_keys[-1]].shape[0])
    if cond_channels is None:
        cond_channels = hidden_channels  # safe-ish default

    # n_modes: inferred from spectral conv weight shape.
    # neuralop's SpectralConv weight has shape (hidden, hidden, *modes) where
    # the LAST axis is r-FFT compressed (n_modes_last // 2 + 1). All other
    # mode axes are full. We undo the compression on the last axis.
    n_modes_inferred = None
    for k, v in state_dict.items():
        if "fno.fno_blocks.convs." in k and k.endswith(".weight.tensor"):
            modes_raw = tuple(int(s) for s in v.shape[2:])
            if len(modes_raw) >= 1:
                # Undo r-FFT compression on the last axis.
                last_full = (modes_raw[-1] - 1) * 2
                n_modes_inferred = tuple(list(modes_raw[:-1]) + [last_full])
            break
    if n_modes_inferred is None:
        # Conservative default — half grid Nyquist.
        n_modes_inferred = tuple(int(s // 4) for s in grid_shape)

    # prior_version: 3D → v2.1_3d; 2D defaults to v2.1.
    prior_version = "v2.1_3d" if field_dims == 3 else "v2.1"

    # Slice z scalar (only used by v2.1 2D path).
    slice_z = float(attrs.get("slice_z_m", 0.09))
    if not np.isfinite(slice_z):
        slice_z = 0.09

    # Detect residual_prior from state_dict buffer keys.
    has_prior_mean = "_prior_mean" in state_dict
    has_prior_std = "_prior_std" in state_dict
    residual_prior = bool(has_prior_mean and has_prior_std)
    if residual_prior:
        prior_norm_mean = state_dict["_prior_mean"].cpu().numpy().tolist()
        prior_norm_std = state_dict["_prior_std"].cpu().numpy().tolist()
    else:
        prior_norm_mean = [0.0, 0.0]
        prior_norm_std = [1.0, 1.0]

    cfg: Dict[str, Any] = {
        "field_dims": field_dims,
        "grid_shape": list(grid_shape),
        "n_transducers": n_transducers,
        "n_modes": list(n_modes_inferred),
        "hidden_channels": hidden_channels,
        "cond_channels": cond_channels,
        "out_channels": 2,  # Re/Im — invariant across all FNOs in this codebase
        "prior_enabled": True,
        "residual_prior": residual_prior,
        "prior_version": prior_version,
        "wavenumber": float(wavenumber),
        "slice_z": float(slice_z),
        "transducer_positions": tx.tolist(),
        "grid_coords": [g.tolist() for g in grid_coords],
        "prior_norm_mean": prior_norm_mean,
        "prior_norm_std": prior_norm_std,
    }
    logger.info(
        "reconstructed model_config from dataset: field_dims=%d grid=%s prior=%s "
        "n_tx=%d hidden=%d modes=%s",
        field_dims,
        list(grid_shape),
        prior_version,
        n_transducers,
        hidden_channels,
        list(n_modes_inferred),
    )
    return cfg


def _load_fno_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    *,
    config_source: Optional[str] = None,
    dataset_h5: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Load a ``PhaseConditionedFNO`` and return ``(model, model_config)``.

    Two checkpoint formats are supported:

    1. **Full ckpt** (``train.py`` saves this as ``<name>.pt``): contains
       ``model_state_dict`` plus ``model_config`` with the geometry to
       rebuild the constructor.
    2. **Best-only ckpt** (``train.py`` saves this as ``<name>_best.pt``):
       contains only ``{best_val_h1, epoch, model_state_dict}``. To
       rebuild the constructor, we need either:
       - ``config_source``: path to a sibling full ckpt with embedded
         ``model_config``, OR
       - ``dataset_h5``: path to the HDF5 file used at training time;
         we read ``transducer_positions``, ``grid_coords``, attrs, and
         infer the rest from the state_dict shapes.

    The dataset-fallback path is what the prior `/tmp/disagreement_3d.py`
    used (with a separate ``--ckpt-config`` arg). Folding it in lets the
    in-repo command work for the FNO_jwave checkpoint without external
    helper scripts.
    """
    from ml_inverse.model import PhaseConditionedFNO  # local import (heavy)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config")
    if cfg is None:
        cfg = _resolve_model_config(
            ckpt=ckpt,
            config_source=config_source,
            dataset_h5=dataset_h5,
            device=device,
        )
    kwargs: Dict[str, Any] = {
        "field_dims": cfg["field_dims"],
        "grid_shape": tuple(cfg["grid_shape"]),
        "n_transducers": cfg["n_transducers"],
        "n_modes": tuple(cfg["n_modes"]),
        "hidden_channels": cfg["hidden_channels"],
        "cond_channels": cfg["cond_channels"],
        "out_channels": cfg["out_channels"],
    }
    if cfg.get("prior_enabled"):
        kwargs.update(
            transducer_positions=np.asarray(cfg["transducer_positions"], dtype=np.float32),
            grid_coords=cfg["grid_coords"],
            wavenumber=cfg["wavenumber"],
            slice_z=cfg.get("slice_z", 0.09),
        )
    if cfg.get("residual_prior"):
        kwargs.update(
            residual_prior=True,
            prior_norm_mean=np.asarray(cfg["prior_norm_mean"], dtype=np.float32),
            prior_norm_std=np.asarray(cfg["prior_norm_std"], dtype=np.float32),
        )
    kwargs["prior_version"] = cfg.get("prior_version", "v2")

    model = PhaseConditionedFNO(**kwargs).to(device)
    state_dict = {
        k: v for k, v in ckpt["model_state_dict"].items() if torch.is_tensor(v)
    }
    model.load_state_dict(state_dict)
    model.eval()
    model._warned_once = True  # avoid the first-batch UserWarning guard cost
    return model, cfg


def make_fno_predictor(model: Any) -> PressurePredictor:
    """Wrap an FNO in the ``PressurePredictor`` callable interface."""

    def _predict(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    return _predict


def make_analytical_predictor(
    model_for_geometry: Any,
    *,
    normalizer_mean: np.ndarray,
    normalizer_std: np.ndarray,
    eps: float,
    device: torch.device,
) -> PressurePredictor:
    """Build the analytical free-field predictor in normalized space.

    Reuses the FNO model's own ``_physical_prior_v21`` (or v2) so the
    geometry / wavenumber / aperture all match. The result is then
    normalized using the provided ``(mean, std)`` so it's directly
    comparable to the FNO output (which is also in normalized space).

    Returns a callable taking ``x`` of shape ``(B, n_transducers + 1)``
    for v2.1 or ``(B, n_transducers)`` for v2.
    """
    mean_t = torch.from_numpy(np.asarray(normalizer_mean, dtype=np.float32)).to(device)
    std_t = torch.from_numpy(np.asarray(normalizer_std, dtype=np.float32)).to(device)
    std_t = torch.clamp(std_t, min=eps)

    def _predict(x: torch.Tensor) -> torch.Tensor:
        # x is already on device; mirror PhaseConditionedFNO.forward parsing.
        prior_ver = model_for_geometry.prior_version
        if prior_ver == "v2.1":
            phases = x[:, : model_for_geometry.n_transducers]
            slice_z = x[:, -1]
            prior = model_for_geometry._physical_prior_v21(phases, slice_z)
        elif prior_ver == "v2.1_3d":
            # 3D path — phases-only input, no per-sample slice_z.
            phases = x[:, : model_for_geometry.n_transducers]
            prior = model_for_geometry._physical_prior_v21_3d(phases)
        else:
            phases = x
            prior = model_for_geometry._physical_prior(phases)
        # Normalize per channel: prior shape (B, 2, *grid).
        view = (None, slice(None)) + (None,) * (prior.ndim - 2)
        return (prior - mean_t[view]) / std_t[view]

    return _predict


def make_denormalizer(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float,
    device: torch.device,
) -> Denormalizer:
    """Returns a callable that maps a single normalized field back to Pa."""
    mean_t = torch.from_numpy(np.asarray(mean, dtype=np.float32)).to(device)
    std_t = torch.from_numpy(np.asarray(std, dtype=np.float32)).to(device)
    std_t = torch.clamp(std_t, min=eps)

    def _denorm(field: torch.Tensor) -> torch.Tensor:
        # field shape: (C, *grid)
        view = (slice(None),) + (None,) * (field.ndim - 1)
        return field * std_t[view] + mean_t[view]

    return _denorm


# --------------------------------------------------------------------------
# Sample selection — deterministic via seed.
# --------------------------------------------------------------------------


def sample_test_indices(
    *,
    h5_path: str,
    n_samples: int,
    seed: int,
    pack_slice_z: bool,
    n_transducers: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Deterministic random sample from the test split.

    Uses the same trajectory split logic as ``InverseSurrogateDataset``.

    Returns
    -------
    inputs
        Tensor ``(n_samples, n_transducers + 1)`` (v2.1) or
        ``(n_samples, n_transducers)`` (v2). float32, CPU.
    chosen_indices
        Row indices in the HDF5 file (for traceability).
    """
    from ml_inverse.dataset import InverseSurrogateDataset  # local import (heavy)

    test_raw = InverseSurrogateDataset(h5_path, split="test", seed=seed)
    n_test = len(test_raw)
    if n_test == 0:
        raise RuntimeError(f"empty test split in {h5_path}")
    if n_samples > n_test:
        logger.warning("requested %d samples but test split has %d; using all", n_samples, n_test)
        n_samples = n_test

    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_test, size=n_samples, replace=False)
    chosen.sort()  # nicer for downstream logging

    rows: List[torch.Tensor] = []
    chosen_h5_rows: List[int] = []
    for idx in chosen:
        sample = test_raw[int(idx)]
        phases = sample["phases"].to(dtype=torch.float32)
        if pack_slice_z:
            sz = sample["slice_z"]
            if not torch.isfinite(sz).item():
                raise ValueError(
                    f"sample idx={idx} has NaN slice_z; v2.1 prior requires finite z"
                )
            x = torch.cat([phases, sz.reshape(1).to(torch.float32)], dim=0)
        else:
            x = phases
        rows.append(x)
        chosen_h5_rows.append(int(test_raw._row_indices[int(idx)]))

    inputs = torch.stack(rows, dim=0)  # (N, D)
    if inputs.shape[1] != (n_transducers + (1 if pack_slice_z else 0)):
        raise RuntimeError(
            f"input width mismatch: got {inputs.shape[1]}, "
            f"expected {n_transducers + (1 if pack_slice_z else 0)}"
        )
    return inputs, np.asarray(chosen_h5_rows, dtype=np.int64)


# --------------------------------------------------------------------------
# Convenience drivers — the CLI uses these.
# --------------------------------------------------------------------------


def compare_fno_vs_analytical(
    *,
    fno_checkpoint: str,
    norm_path: str,
    dataset_h5: str,
    output_dir: Path,
    n_samples: int = 50,
    seed: int = 42,
    n_render: int = 5,
    device: Optional[torch.device] = None,
    config_source: Optional[str] = None,
) -> DisagreementResult:
    """Run the FNO vs. analytical comparison end-to-end.

    Loads the v2.1 FNO checkpoint, builds the analytical predictor from
    the model's own embedded geometry/prior code, samples ``n_samples``
    test configs deterministically, computes disagreement, and renders
    ``n_render`` 4-panel PNGs plus an aggregate-attribution PNG.
    """
    from ml_inverse.adapter import ChannelNormalizer

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = auto_device()
    logger.info("device: %s", device)

    model, cfg = _load_fno_from_checkpoint(
        fno_checkpoint,
        device,
        config_source=config_source,
        dataset_h5=dataset_h5,
    )
    norm = ChannelNormalizer.load(norm_path)
    # v2.1 (2D) packs slice_z as the last input column; v2 and v2.1_3d
    # use plain (B, n_transducers) inputs.
    pack_slice_z = cfg.get("prior_version", "v2") == "v2.1"
    field_dims = int(cfg.get("field_dims", 2))

    # FNO predictor.
    fno_pred = make_fno_predictor(model)

    # Analytical predictor — uses model's own geometry buffers.
    analytical_pred = make_analytical_predictor(
        model,
        normalizer_mean=norm.mean,
        normalizer_std=norm.std,
        eps=norm.eps,
        device=device,
    )

    denorm = make_denormalizer(
        mean=norm.mean,
        std=norm.std,
        eps=norm.eps,
        device=device,
    )

    inputs, h5_rows = sample_test_indices(
        h5_path=dataset_h5,
        n_samples=n_samples,
        seed=seed,
        pack_slice_z=pack_slice_z,
        n_transducers=int(cfg["n_transducers"]),
    )
    inputs = inputs.to(device)
    logger.info("sampled %d configs (seed=%d) from test split of %s", inputs.shape[0], seed, dataset_h5)

    # Derive grid extent from the model's grid_coords for accurate physical-
    # space attribution (esp. relevant for 3D where the v3 default is wrong).
    grid_coords = cfg.get("grid_coords")
    if grid_coords is not None and len(grid_coords) >= 2:
        gx = np.asarray(grid_coords[0])
        gy = np.asarray(grid_coords[1])
        if field_dims == 3 and len(grid_coords) >= 3:
            gz = np.asarray(grid_coords[2])
            grid_extent = (
                float(gx.min()), float(gx.max()),
                float(gy.min()), float(gy.max()),
                float(gz.min()), float(gz.max()),
            )
        else:
            grid_extent = (
                float(gx.min()), float(gx.max()),
                float(gy.min()), float(gy.max()),
            )
    else:
        grid_extent = DEFAULT_EXTENT_M

    t0 = time.perf_counter()
    per_sample, aggregate, _res_norm, res_phys, a_phys, b_phys = compute_disagreement(
        model_a=fno_pred,
        model_b=analytical_pred,
        test_inputs=inputs,
        denormalize=denorm,
        grid_extent_m=grid_extent,
        device=device,
    )
    elapsed = time.perf_counter() - t0
    logger.info("disagreement compute: %.2fs (%.3fs/sample)", elapsed, elapsed / max(len(per_sample), 1))

    # Attribution roll-up.
    attribution = attribution_heatmap(
        residual_phys=res_phys,
        a_magnitude=a_phys,
        grid_extent_m=grid_extent,
    )

    # Render 4-panel maps for the worst-N (most informative) and a couple
    # of median/best so the user sees the full distribution.
    rel_l2_norm = np.array([s.relative_l2_normalized for s in per_sample])
    sort_idx = np.argsort(rel_l2_norm)
    n_render = min(n_render, len(per_sample))
    if n_render >= 5:
        # 1 best, 1 25th, 1 median, 1 75th, rest worst.
        special = [sort_idx[0], sort_idx[len(sort_idx) // 4], sort_idx[len(sort_idx) // 2], sort_idx[3 * len(sort_idx) // 4]]
        worst = sort_idx[::-1][: max(n_render - len(special), 1)]
        chosen = list(dict.fromkeys(list(special) + list(worst)))[:n_render]
    else:
        chosen = sort_idx[::-1][:n_render].tolist()

    label_a = f"FNO ({cfg.get('prior_version', 'v?')})"
    rendered_paths: List[str] = []
    for i in chosen:
        h5_idx = int(h5_rows[i])
        path = render_disagreement_map(
            sample_index=h5_idx,
            a_magnitude=a_phys[i],
            b_magnitude=b_phys[i],
            residual_phys=res_phys[i],
            output_dir=output_dir,
            extent_m=grid_extent,
            label_a=label_a,
            label_b="analytical",
        )
        rendered_paths.append(str(path))
        logger.info("rendered %s (rel_l2_norm=%.4e)", path.name, rel_l2_norm[i])

    attribution_path = render_attribution_summary(
        attribution=attribution,
        output_dir=output_dir,
        label="fno_vs_analytical",
        extent_m=grid_extent,
    )
    rendered_paths.append(str(attribution_path))

    # Persist receipts.
    metadata = {
        "mode": "fno-vs-analytical",
        "fno_checkpoint": str(fno_checkpoint),
        "norm_path": str(norm_path),
        "dataset_h5": str(dataset_h5),
        "n_samples": int(len(per_sample)),
        "seed": int(seed),
        "device": str(device),
        "elapsed_seconds": float(elapsed),
        "h5_indices": [int(r) for r in h5_rows.tolist()],
        "rendered_paths": rendered_paths,
        "attribution": {
            k: _json_safe(v)
            for k, v in attribution.items()
            if k != "avg_residual_field"
        },
    }

    summary = {
        "mode": "fno-vs-analytical",
        "metadata": metadata,
        "aggregate": asdict(aggregate),
        "per_sample": [asdict(s) for s in per_sample],
    }
    summary_path = output_dir / "fno_vs_analytical_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("summary saved: %s", summary_path)

    # Aggregate attribution as separate JSON for easy diffing.
    attribution_path_json = output_dir / "fno_vs_analytical_attribution.json"
    with open(attribution_path_json, "w") as fh:
        json.dump(
            {k: v for k, v in metadata["attribution"].items()},
            fh,
            indent=2,
        )

    return DisagreementResult(
        mode="fno-vs-analytical",
        aggregate=aggregate,
        per_sample=tuple(per_sample),
        metadata=metadata,
    )


def compare_fno_vs_fno(
    *,
    checkpoint_a: str,
    checkpoint_b: str,
    norm_a: str,
    norm_b: str,
    dataset_h5: str,
    output_dir: Path,
    n_samples: int = 50,
    seed: int = 42,
    n_render: int = 5,
    device: Optional[torch.device] = None,
    label_a: str = "FNO_a",
    label_b: str = "FNO_b",
) -> DisagreementResult:
    """Run FNO vs. FNO comparison.

    Both checkpoints must produce fields on the same grid. Each is
    de-normalized using its own normalizer for the physical-residual
    metric, then we compare ``|P|_a`` vs ``|P|_b`` directly. For the
    normalized-space metric we use ``norm_b`` as the common reference
    so the relative-L2 comparison is well-defined.
    """
    from ml_inverse.adapter import ChannelNormalizer

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = auto_device()
    logger.info("device: %s", device)

    model_a, cfg_a = _load_fno_from_checkpoint(
        checkpoint_a, device, dataset_h5=dataset_h5
    )
    model_b, cfg_b = _load_fno_from_checkpoint(
        checkpoint_b, device, dataset_h5=dataset_h5
    )
    norm_a_obj = ChannelNormalizer.load(norm_a)
    norm_b_obj = ChannelNormalizer.load(norm_b)

    if tuple(cfg_a["grid_shape"]) != tuple(cfg_b["grid_shape"]):
        raise ValueError(
            f"grid mismatch: a={cfg_a['grid_shape']}, b={cfg_b['grid_shape']}"
        )
    pack_slice_z = (
        cfg_a.get("prior_version", "v2") == "v2.1"
        or cfg_b.get("prior_version", "v2") == "v2.1"
    )

    # Each FNO is normalized in its own coordinate space. To compare in a
    # common normalized space we de-normalize then re-normalize using
    # model_b's normalizer.
    denorm_a = make_denormalizer(
        mean=norm_a_obj.mean, std=norm_a_obj.std, eps=norm_a_obj.eps, device=device
    )
    mean_b_t = torch.from_numpy(norm_b_obj.mean.astype(np.float32)).to(device)
    std_b_t = torch.clamp(
        torch.from_numpy(norm_b_obj.std.astype(np.float32)).to(device),
        min=norm_b_obj.eps,
    )

    def fno_a_predictor(x: torch.Tensor) -> torch.Tensor:
        y_norm_a = model_a(x)
        # Denorm with A's stats, re-norm with B's.
        view_dn = (slice(None), slice(None)) + (None,) * (y_norm_a.ndim - 2)
        # Denorm using closure over norm_a_obj.
        m_a = torch.from_numpy(norm_a_obj.mean.astype(np.float32)).to(device)[view_dn]
        s_a = torch.clamp(
            torch.from_numpy(norm_a_obj.std.astype(np.float32)).to(device)[view_dn],
            min=norm_a_obj.eps,
        )
        phys = y_norm_a * s_a + m_a
        view_b = (slice(None), slice(None)) + (None,) * (phys.ndim - 2)
        return (phys - mean_b_t[view_b]) / std_b_t[view_b]

    def fno_b_predictor(x: torch.Tensor) -> torch.Tensor:
        return model_b(x)

    # Use B's denormalizer for the physical-residual computation.
    denorm_b = make_denormalizer(
        mean=norm_b_obj.mean, std=norm_b_obj.std, eps=norm_b_obj.eps, device=device
    )
    inputs, h5_rows = sample_test_indices(
        h5_path=dataset_h5,
        n_samples=n_samples,
        seed=seed,
        pack_slice_z=pack_slice_z,
        n_transducers=int(cfg_a["n_transducers"]),
    )
    inputs = inputs.to(device)
    logger.info("sampled %d configs (seed=%d)", inputs.shape[0], seed)

    t0 = time.perf_counter()
    per_sample, aggregate, _res_norm, res_phys, a_phys, b_phys = compute_disagreement(
        model_a=fno_a_predictor,
        model_b=fno_b_predictor,
        test_inputs=inputs,
        denormalize=denorm_b,
        device=device,
    )
    elapsed = time.perf_counter() - t0
    logger.info("disagreement compute: %.2fs", elapsed)

    attribution = attribution_heatmap(
        residual_phys=res_phys,
        a_magnitude=a_phys,
    )

    rel_l2_norm = np.array([s.relative_l2_normalized for s in per_sample])
    sort_idx = np.argsort(rel_l2_norm)
    n_render = min(n_render, len(per_sample))
    if n_render >= 5:
        special = [sort_idx[0], sort_idx[len(sort_idx) // 4], sort_idx[len(sort_idx) // 2], sort_idx[3 * len(sort_idx) // 4]]
        worst = sort_idx[::-1][: max(n_render - len(special), 1)]
        chosen = list(dict.fromkeys(list(special) + list(worst)))[:n_render]
    else:
        chosen = sort_idx[::-1][:n_render].tolist()

    rendered_paths: List[str] = []
    for i in chosen:
        h5_idx = int(h5_rows[i])
        path = render_disagreement_map(
            sample_index=h5_idx,
            a_magnitude=a_phys[i],
            b_magnitude=b_phys[i],
            residual_phys=res_phys[i],
            output_dir=output_dir,
            label_a=label_a,
            label_b=label_b,
        )
        rendered_paths.append(str(path))

    attribution_path = render_attribution_summary(
        attribution=attribution,
        output_dir=output_dir,
        label=f"{label_a}_vs_{label_b}".replace(" ", "_"),
    )
    rendered_paths.append(str(attribution_path))

    metadata = {
        "mode": "fno-vs-fno",
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "norm_a_path": str(norm_a),
        "norm_b_path": str(norm_b),
        "label_a": label_a,
        "label_b": label_b,
        "dataset_h5": str(dataset_h5),
        "n_samples": int(len(per_sample)),
        "seed": int(seed),
        "device": str(device),
        "elapsed_seconds": float(elapsed),
        "h5_indices": [int(r) for r in h5_rows.tolist()],
        "rendered_paths": rendered_paths,
        "attribution": {
            k: _json_safe(v)
            for k, v in attribution.items()
            if k != "avg_residual_field"
        },
    }
    summary = {
        "mode": "fno-vs-fno",
        "metadata": metadata,
        "aggregate": asdict(aggregate),
        "per_sample": [asdict(s) for s in per_sample],
    }
    summary_path = output_dir / f"{label_a}_vs_{label_b}_summary.json".replace(" ", "_")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("summary saved: %s", summary_path)

    # Suppress unused-name warning on denorm_a (it's used inside fno_a_predictor's closure).
    _ = denorm_a

    return DisagreementResult(
        mode="fno-vs-fno",
        aggregate=aggregate,
        per_sample=tuple(per_sample),
        metadata=metadata,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-surrogate disagreement analysis.",
    )
    p.add_argument(
        "--mode",
        choices=("fno-vs-analytical", "fno-vs-fno"),
        default="fno-vs-analytical",
    )
    p.add_argument(
        "--fno-ckpt",
        type=str,
        default="ml_inverse/models/fno_surrogate_v2_1.pt",
        help="Path to FNO checkpoint (mode=fno-vs-analytical) or FNO_A "
        "(mode=fno-vs-fno).",
    )
    p.add_argument(
        "--fno-ckpt-b",
        type=str,
        default=None,
        help="Path to FNO_B checkpoint (mode=fno-vs-fno only).",
    )
    p.add_argument(
        "--norm",
        type=str,
        default=None,
        help="Path to FNO normalizer .npz (default: derive from --fno-ckpt by "
        "replacing .pt with _norm.npz).",
    )
    p.add_argument(
        "--norm-b",
        type=str,
        default=None,
        help="Path to FNO_B normalizer .npz (mode=fno-vs-fno only).",
    )
    p.add_argument(
        "--label-a",
        type=str,
        default="FNO_a",
        help="Plot label for model A (mode=fno-vs-fno).",
    )
    p.add_argument(
        "--label-b",
        type=str,
        default="FNO_b",
        help="Plot label for model B (mode=fno-vs-fno).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="ml_inverse/data/inverse_dataset_v3.h5",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="ml_inverse/research/_disagreement/",
    )
    p.add_argument(
        "--config-source",
        type=str,
        default=None,
        help="Path to a sibling full ckpt with embedded model_config; only "
        "needed for train.py-format best-only checkpoints whose config "
        "cannot be inferred from the dataset (rare).",
    )
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--n-render", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def _derive_default_norm(ckpt_path: str) -> str:
    """Default normalizer path: ``foo.pt`` → ``foo_norm.npz``."""
    p = Path(ckpt_path)
    if p.suffix == ".pt":
        return str(p.with_suffix("").as_posix() + "_norm.npz")
    return str(p) + "_norm.npz"


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = None
    if args.device is not None:
        device = torch.device(args.device)

    if args.mode == "fno-vs-analytical":
        norm_path = args.norm or _derive_default_norm(args.fno_ckpt)
        result = compare_fno_vs_analytical(
            fno_checkpoint=args.fno_ckpt,
            norm_path=norm_path,
            dataset_h5=args.dataset,
            output_dir=output_dir,
            n_samples=args.n_samples,
            n_render=args.n_render,
            seed=args.seed,
            device=device,
            config_source=args.config_source,
        )
    elif args.mode == "fno-vs-fno":
        if args.fno_ckpt_b is None:
            logger.error(
                "mode=fno-vs-fno requires --fno-ckpt-b. "
                "(FNO_jwave is not yet trained — see DISAGREEMENT_ANALYSIS.md "
                "stub section for the command to run after Track A completes.)"
            )
            return 2
        norm_a = args.norm or _derive_default_norm(args.fno_ckpt)
        norm_b = args.norm_b or _derive_default_norm(args.fno_ckpt_b)
        result = compare_fno_vs_fno(
            checkpoint_a=args.fno_ckpt,
            checkpoint_b=args.fno_ckpt_b,
            norm_a=norm_a,
            norm_b=norm_b,
            dataset_h5=args.dataset,
            output_dir=output_dir,
            n_samples=args.n_samples,
            n_render=args.n_render,
            seed=args.seed,
            device=device,
            label_a=args.label_a,
            label_b=args.label_b,
        )
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    agg = result.aggregate
    print()
    print("=" * 60)
    print(f"AGGREGATE — mode={result.mode}, n={agg.n_samples}")
    print("=" * 60)
    print(f"rel_l2_norm  mean   = {agg.rel_l2_norm_mean:.6e}")
    print(f"rel_l2_norm  median = {agg.rel_l2_norm_median:.6e}")
    print(f"rel_l2_norm  p95    = {agg.rel_l2_norm_p95:.6e}")
    print(f"rel_l2_norm  max    = {agg.rel_l2_norm_max:.6e}")
    print(f"rel_l2_phys  mean   = {agg.rel_l2_phys_mean:.6e}")
    print(f"rel_l2_phys  max    = {agg.rel_l2_phys_max:.6e}")
    print(f"abs_residual max Pa = {agg.abs_residual_max_pa_aggregate:.4e}")
    print(f"focal_overlap_mean  = {agg.focal_overlap_mean:.3f}")
    print(f"in_array_mean       = {agg.in_array_mean:.3f}")
    print(f"output_dir          = {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
