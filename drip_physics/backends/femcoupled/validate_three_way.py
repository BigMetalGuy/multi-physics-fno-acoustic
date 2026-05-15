"""Phase 6.1 three-way residual study — analytical vs j-Wave vs FEM-coupled.

Mirrors the structure of ``ULTRAINO_VALIDATION.md`` but for the ``femcoupled``
backend.  Picks 5 representative free-field configurations from the v3 inverse
dataset (test split), projects each backend's pressure field onto the same
coarse voxel grid, and computes pairwise relative-L2 residuals on |p|.

Receipts land at ``simulations/ml_inverse/research/_fem_three_way_v0/`` by
default; pass ``--out-dir`` to redirect (e.g. for the v1 resolved-grid rerun).

Caveat (per FEM_V2_DESIGN §3.7):
  * Analytical (``drip_physics.pressure``) is unbounded free-field 1/r
    superposition with sinc-piston directivity — *no chamber walls*.
  * j-Wave is a Cartesian Helmholtz solve with PML on all faces — *no
    chamber-wall reflection*; effectively free-field-in-a-box.
  * FEM-coupled solves Helmholtz on the L1 chamber mesh with finite-
    aperture Robin transducer faces and (optionally) Bermúdez PML.  Even
    with ``pml_width=0`` the indefinite Helmholtz system + finite-area
    BC structure differs from the analytical 1/r superposition.

So we expect FEM-coupled to disagree with analytical *more* than j-Wave
disagrees with analytical (chamber + finite-area BC vs delta sources +
Cartesian PML), and FEM-coupled-vs-j-Wave to be lower than
FEM-vs-analytical because both solve some flavour of Helmholtz.

Usage::

    cd simulations
    PYTHONPATH=. ml_inverse/_jaxfem_discovery/.venv/bin/python \\
        drip_physics/backends/femcoupled/validate_three_way.py \\
        [--j-grid-n N] [--j-cube-side-m S] [--out-dir DIR]

Wall budget (Phase 6 / v0, j_grid_n=64, dx=λ/1.4): ~5–15 min on the coarse
16³ probe grid.  v1 (j_grid_n≥192, dx≤λ/3.9) is roughly 4-10x slower in
the j-Wave column.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from drip_physics.backends.femcoupled.mesh import build_l1_chamber_mesh
from drip_physics.backends.femcoupled_backend import (
    DEFAULT_GRID_EXTENT_M,
    compute_pressure_from_phases as fem_compute,
)
from drip_physics.config import (
    L1_ARRAY,
    MA40S4S_40KHZ,
    PressureConfig,
    default_environment,
)
from drip_physics.core import ArrayGeometry, PhaseArray, SimulationParams
from drip_physics.pressure import compute_pressure_field
from drip_physics.backends.jwave_backend import JWaveBackend


logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parents[3]
    / "ml_inverse"
    / "research"
    / "_fem_three_way_v0"
)

# Small grid for receipts — production would be 64³.
GRID_SHAPE: Tuple[int, int, int] = (16, 16, 16)
GRID_EXTENT_M: Tuple[float, float, float, float, float, float] = DEFAULT_GRID_EXTENT_M

# j-Wave grid defaults.  Phase 6 (v0) used n=64 / cube_side=0.40 m → dx=6.25
# mm = λ/1.37, which under-resolved the 8.575 mm wavelength and caused the
# anomaly ``jWave-vs-FEM > analytical-vs-FEM`` in 5/5 configs.  v1 default
# bumps to n=128 / cube=0.42 m → dx=3.28 mm = λ/2.61 — a 1.91x resolution
# increase that brings the j-Wave Helmholtz solve into a regime where it
# can resolve the focal lobe.  λ/5 (the audit standard) at the same cube
# would require n≥244; n=128 is the wall-time-feasible compromise on CPU
# (~100 s per first call vs ~10 min at n=192 vs ~1 hour at n=256).  The
# cube cannot shrink below 0.40 m because the L1 array spans 0.36 m in z;
# any smaller cube clamps transducers to the boundary and is unphysical.
DEFAULT_J_GRID_N: int = 128
DEFAULT_J_CUBE_SIDE_M: float = 0.42


# ---------------------------------------------------------------------------
# Configuration selection
# ---------------------------------------------------------------------------


def select_configs(n_configs: int = 5) -> List[Dict]:
    """Pick ``n_configs`` representative phase vectors.

    We synthesise them analytically rather than reading the v3 dataset because:
      (a) v3 stores 2D slices, not full 3D fields — we'd recompute anyway.
      (b) Analytic configs (focal trap, twin trap, off-axis, etc.) span the
          design regime better than 5 sequential test-split frames.

    Each config returns a phase vector + label + diagnostic target point
    so the residual heatmaps can ground-truth the focal location.
    """
    env = default_environment()
    n_t = L1_ARRAY.total_transducers
    array = _build_array_geometry()
    centers = array.positions  # (N, 3)
    k = float(env.wavenumber)

    configs: List[Dict] = []

    # Config 1: Centred focal trap at chamber centre.
    target = np.array([0.0, 0.0, 0.20], dtype=np.float64)
    distances = np.linalg.norm(centers - target[None, :], axis=1)
    phases = (-k * distances) % (2 * np.pi)
    configs.append({
        "label": "focal_trap_centre",
        "phases": phases.astype(np.float64),
        "target": target.tolist(),
        "description": "Single focal point at chamber centroid (canonical focusing).",
    })

    # Config 2: Twin trap (focusing + pi step in x).
    target = np.array([0.0, 0.0, 0.20], dtype=np.float64)
    distances = np.linalg.norm(centers - target[None, :], axis=1)
    pi_step = np.pi * (centers[:, 0] < 0).astype(np.float64)
    phases = (-k * distances + pi_step) % (2 * np.pi)
    configs.append({
        "label": "twin_trap",
        "phases": phases.astype(np.float64),
        "target": target.tolist(),
        "description": "Twin trap: focusing + pi step antisymmetric in x (Marzo 2018).",
    })

    # Config 3: Off-axis trap (target shifted +5 mm in x).
    target = np.array([0.005, 0.0, 0.20], dtype=np.float64)
    distances = np.linalg.norm(centers - target[None, :], axis=1)
    phases = (-k * distances) % (2 * np.pi)
    configs.append({
        "label": "off_axis_trap",
        "phases": phases.astype(np.float64),
        "target": target.tolist(),
        "description": "Off-axis focal trap (target +5 mm in x).",
    })

    # Config 4: Multi-droplet (two adjacent focal points).
    target_a = np.array([+0.004, 0.0, 0.20], dtype=np.float64)
    target_b = np.array([-0.004, 0.0, 0.20], dtype=np.float64)
    # Average phases of two single-trap configurations weighted equally.
    da = np.linalg.norm(centers - target_a[None, :], axis=1)
    db = np.linalg.norm(centers - target_b[None, :], axis=1)
    pa = -k * da
    pb = -k * db
    # Coherent superposition then take phase angle.
    z = np.exp(1j * pa) + np.exp(1j * pb)
    phases = np.angle(z) % (2 * np.pi)
    configs.append({
        "label": "multi_droplet",
        "phases": phases.astype(np.float64),
        "target": [target_a.tolist(), target_b.tolist()],
        "description": "Two-target coherent superposition (multi-droplet pattern).",
    })

    # Config 5: Edge case — random phases (tests off-resonance / noise floor).
    rng = np.random.default_rng(0)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_t)
    configs.append({
        "label": "random_phases",
        "phases": phases.astype(np.float64),
        "target": None,
        "description": "Random phases — edge case, no focal structure expected.",
    })

    return configs


# ---------------------------------------------------------------------------
# Backend wrappers — return uniform (nz, ny, nx) complex voxel field
# ---------------------------------------------------------------------------


def _build_array_geometry() -> ArrayGeometry:
    """Build the L1 ArrayGeometry to match the FEM-mesh transducer layout.

    The FEM mesh places transducers at z = bottom_ring_z + ring*spacing
    (chamber-frame, z=0 at substrate).  We mirror that exactly so all
    three backends consume the same transducer positions.
    """
    from drip_physics.backends.femcoupled.geometry import (
        resolve_chamber_geometry,
        transducer_centers,
    )

    geom = resolve_chamber_geometry(L1_ARRAY, MA40S4S_40KHZ)
    positions = transducer_centers(geom)  # (120, 3)
    params = SimulationParams()
    return ArrayGeometry(positions=positions, params=params)


def _voxel_grid_points(
    grid_shape: Tuple[int, int, int],
    grid_extent_m: Tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Build (nz*ny*nx, 3) array of voxel-centre coordinates in (x, y, z).

    Same indexing convention as ``femcoupled_backend._build_voxel_grid``.
    """
    nx, ny, nz = grid_shape
    xmin, xmax, ymin, ymax, zmin, zmax = grid_extent_m
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    zs = np.linspace(zmin, zmax, nz, dtype=np.float64)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.stack([X, Y, Z], axis=-1).reshape(-1, 3)


def run_analytical(
    phases: np.ndarray,
    grid_shape: Tuple[int, int, int],
    grid_extent_m: Tuple[float, float, float, float, float, float],
) -> Tuple[np.ndarray, float]:
    """Analytical ``pressure.compute_pressure_field`` evaluated on the voxel grid."""
    env = default_environment()
    array = _build_array_geometry()
    n_t = array.num_transducers
    pa = PhaseArray(
        phases=np.asarray(phases, dtype=np.float64),
        amplitudes=np.ones(n_t, dtype=np.float64),
    )
    points = _voxel_grid_points(grid_shape, grid_extent_m)
    t0 = time.time()
    field = compute_pressure_field(array, pa, points, env=env)
    wall = time.time() - t0
    nx, ny, nz = grid_shape
    return field.reshape(nz, ny, nx), float(wall)


def run_jwave(
    phases: np.ndarray,
    grid_shape: Tuple[int, int, int],
    grid_extent_m: Tuple[float, float, float, float, float, float],
    backend: JWaveBackend,
) -> Tuple[np.ndarray, float]:
    """j-Wave Helmholtz on the same voxel grid.

    The j-Wave backend solves on its own internal grid; we point-query at
    the voxel centres to project to the shared grid.
    """
    env = default_environment()
    array = _build_array_geometry()
    n_t = array.num_transducers
    pa = PhaseArray(
        phases=np.asarray(phases, dtype=np.float64),
        amplitudes=np.ones(n_t, dtype=np.float64),
    )
    points = _voxel_grid_points(grid_shape, grid_extent_m)
    t0 = time.time()
    backend.invalidate_cache()
    field = backend.compute_pressure_field(array, pa, points, env=env)
    wall = time.time() - t0
    nx, ny, nz = grid_shape
    return np.asarray(field, dtype=np.complex128).reshape(nz, ny, nx), float(wall)


def run_femcoupled(
    phases: np.ndarray,
    grid_shape: Tuple[int, int, int],
    grid_extent_m: Tuple[float, float, float, float, float, float],
    state_holder: Dict,
    mesh_path: Path,
) -> Tuple[np.ndarray, float]:
    """FEM-coupled Helmholtz + heat on the voxel grid.

    The state is shared across the 5 configs to amortise the ~20 s mesh+
    driver build cost.  ``state_holder`` is a mutable dict because the
    state is rebuilt lazily inside the backend.
    """
    env = default_environment()
    phases_jnp = jnp.asarray(phases, dtype=jnp.float64)
    t0 = time.time()
    field_real_imag, new_state = fem_compute(
        phases=phases_jnp,
        env=env,
        grid_shape=grid_shape,
        grid_extent_m=grid_extent_m,
        mesh_path=mesh_path,
        backend_state=state_holder.get("state"),
        pml_width=0.0,
        max_inner_iters=2,
        convergence_tol=1e-2,
        return_state=True,
    )
    wall = time.time() - t0
    state_holder["state"] = new_state
    arr = np.asarray(field_real_imag)
    field_complex = arr[..., 0] + 1j * arr[..., 1]
    return field_complex, float(wall)


# ---------------------------------------------------------------------------
# Residual statistics
# ---------------------------------------------------------------------------


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 error on |·| after peak-normalising each field separately.

    The three backends use different absolute-pressure calibrations:
      * analytical: ``p_ref · r_ref / r``  (Murata datasheet → Pa)
      * j-Wave:     ``S₀·exp(iφ)`` with ``S₀ = p_ref · r_ref · λ / dx²``
                    (calibrated at construction; see jwave_backend.py)
      * FEM:        Robin BC at ``v_amp = 0.01 m/s`` (no datasheet cal)

    To compare *spatial structure* rather than absolute Pa, each field is
    normalised to its own peak before differencing.  This is the same
    convention used in ``ULTRAINO_VALIDATION.md`` for the magnitude-L2.
    """
    am = np.abs(a)
    bm = np.abs(b)
    a_peak = am.max() or 1e-30
    b_peak = bm.max() or 1e-30
    am_n = am / a_peak
    bm_n = bm / b_peak
    norm = np.sqrt((am_n ** 2 + bm_n ** 2).sum() / 2.0)
    return float(np.sqrt(((am_n - bm_n) ** 2).sum()) / max(norm, 1e-30))


def focal_alignment(
    field: np.ndarray,
    grid_extent_m: Tuple[float, float, float, float, float, float],
) -> Tuple[Tuple[float, float, float], float]:
    """Return (peak_xyz_m, peak_magnitude_pa) of the field's argmax."""
    nx, ny, nz = field.shape[2], field.shape[1], field.shape[0]
    xmin, xmax, ymin, ymax, zmin, zmax = grid_extent_m
    mag = np.abs(field)
    iz, iy, ix = np.unravel_index(np.argmax(mag), mag.shape)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = np.linspace(zmin, zmax, nz)
    return (
        (float(xs[ix]), float(ys[iy]), float(zs[iz])),
        float(mag[iz, iy, ix]),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_three_way_slice(
    field_an: np.ndarray,
    field_jw: np.ndarray,
    field_fc: np.ndarray,
    config_label: str,
    out_path: Path,
    grid_extent_m: Tuple[float, float, float, float, float, float],
) -> None:
    """6-panel mid-z heatmap: 3 fields + 3 pairwise residuals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nz = field_an.shape[0]
    iz_mid = nz // 2
    # Peak-normalise each field for comparable visualisation.
    a_n = np.abs(field_an) / max(np.abs(field_an).max(), 1e-30)
    j_n = np.abs(field_jw) / max(np.abs(field_jw).max(), 1e-30)
    f_n = np.abs(field_fc) / max(np.abs(field_fc).max(), 1e-30)
    fields = [
        ("analytical (normalised)", a_n),
        ("j-Wave (normalised)", j_n),
        ("FEM-coupled (normalised)", f_n),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    xmin, xmax, ymin, ymax = grid_extent_m[0], grid_extent_m[1], grid_extent_m[2], grid_extent_m[3]

    # Top row: field magnitudes, shared 0..1 colourmap.
    mags = [f[iz_mid] for _, f in fields]
    vmax_field = 1.0
    for j, ((name, _), m) in enumerate(zip(fields, mags)):
        ax = axes[0, j]
        im = ax.imshow(
            m, origin="lower", extent=[xmin * 1e3, xmax * 1e3, ymin * 1e3, ymax * 1e3],
            vmin=0, vmax=vmax_field, cmap="viridis", aspect="equal",
        )
        ax.set_title(f"{name}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Bottom row: residuals between pairs (on normalised fields).
    pairs = [
        ("an − jW (normalised)", a_n[iz_mid] - j_n[iz_mid]),
        ("an − FC (normalised)", a_n[iz_mid] - f_n[iz_mid]),
        ("jW − FC (normalised)", j_n[iz_mid] - f_n[iz_mid]),
    ]
    vmax_resid = max(np.abs(r).max() for _, r in pairs) or 1.0
    for j, (label, r) in enumerate(pairs):
        ax = axes[1, j]
        im = ax.imshow(
            r, origin="lower", extent=[xmin * 1e3, xmax * 1e3, ymin * 1e3, ymax * 1e3],
            vmin=-vmax_resid, vmax=vmax_resid, cmap="RdBu_r", aspect="equal",
        )
        ax.set_title(label)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"Config: {config_label}  —  mid-z slice z={(grid_extent_m[4]+grid_extent_m[5])/2*1e3:.0f} mm",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_attribution(
    aggregated: Dict[str, np.ndarray],
    out_path: Path,
    grid_extent_m: Tuple[float, float, float, float, float, float],
) -> None:
    """Aggregate-over-configs heatmap of where residuals concentrate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    nz = aggregated["an_vs_jw"].shape[0]
    iz_mid = nz // 2
    xmin, xmax, ymin, ymax = grid_extent_m[0], grid_extent_m[1], grid_extent_m[2], grid_extent_m[3]

    panels = [
        ("an vs jW (mean abs resid)", aggregated["an_vs_jw"][iz_mid]),
        ("an vs FC (mean abs resid)", aggregated["an_vs_fc"][iz_mid]),
        ("jW vs FC (mean abs resid)", aggregated["jw_vs_fc"][iz_mid]),
    ]
    vmax = max(p[1].max() for p in panels) or 1.0
    for ax, (label, m) in zip(axes, panels):
        im = ax.imshow(
            m, origin="lower",
            extent=[xmin * 1e3, xmax * 1e3, ymin * 1e3, ymax * 1e3],
            vmin=0, vmax=vmax, cmap="hot", aspect="equal",
        )
        ax.set_title(label)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Residual attribution — mean |Δ| across configs", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    out_dir: Optional[Path] = None,
    j_grid_n: int = DEFAULT_J_GRID_N,
    j_cube_side_m: float = DEFAULT_J_CUBE_SIDE_M,
) -> None:
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now():%H:%M:%S}] Three-way validation — receipts → {out_dir}")

    configs = select_configs(n_configs=5)
    print(f"  {len(configs)} configurations selected")

    # Pre-build a single mesh + jWave backend for all 5 configs.
    print(f"[{datetime.now():%H:%M:%S}] pre-building L1 chamber mesh ...")
    mesh_path, info = build_l1_chamber_mesh(
        array_config=L1_ARRAY,
        transducer_config=MA40S4S_40KHZ,
        focus_h_mm=2.0,
        bulk_h_mm=10.0,
        cache_dir=out_dir / "_mesh_cache",
        overwrite=False,
        verbose=False,
    )
    print(f"  mesh: {info['element_count']} tets, {info['node_count']} nodes")

    # j-Wave backend with grid sized to cover the chamber array.
    # Array fits in ±0.05 m (radial) × 0.005-0.365 (z); cube must be ≥ 0.4 m.
    j_grid_dx = float(j_cube_side_m) / int(j_grid_n)
    env_for_check = default_environment()
    lambda_m = float(env_for_check.wavelength)
    print(
        f"  j-Wave grid: n={j_grid_n} cube={j_cube_side_m*1e3:.1f} mm  "
        f"dx={j_grid_dx*1e3:.3f} mm  λ/dx={lambda_m/j_grid_dx:.2f}"
    )
    jwave_cfg = PressureConfig(
        backend="jwave", grid_n=j_grid_n, grid_dx=j_grid_dx,
    )
    jwave_backend = JWaveBackend(config=jwave_cfg, env=default_environment())

    # Shared FEM state holder (lazily built on first call).
    fem_state_holder: Dict = {"state": None}

    per_config: List[Dict] = []
    aggregated_residuals = {
        "an_vs_jw": np.zeros((GRID_SHAPE[2], GRID_SHAPE[1], GRID_SHAPE[0])),
        "an_vs_fc": np.zeros((GRID_SHAPE[2], GRID_SHAPE[1], GRID_SHAPE[0])),
        "jw_vs_fc": np.zeros((GRID_SHAPE[2], GRID_SHAPE[1], GRID_SHAPE[0])),
    }

    for idx, cfg in enumerate(configs):
        print(
            f"[{datetime.now():%H:%M:%S}] config {idx+1}/{len(configs)} "
            f"'{cfg['label']}' ..."
        )
        phases = cfg["phases"]
        # 1. Analytical
        f_an, t_an = run_analytical(phases, GRID_SHAPE, GRID_EXTENT_M)
        print(f"  analytical:    {t_an:.2f}s   peak={np.abs(f_an).max():.2e} Pa")
        # 2. j-Wave
        f_jw, t_jw = run_jwave(phases, GRID_SHAPE, GRID_EXTENT_M, jwave_backend)
        print(f"  j-Wave:        {t_jw:.2f}s   peak={np.abs(f_jw).max():.2e} Pa")
        # 3. FEM-coupled
        f_fc, t_fc = run_femcoupled(
            phases, GRID_SHAPE, GRID_EXTENT_M, fem_state_holder, mesh_path,
        )
        print(f"  FEM-coupled:   {t_fc:.2f}s   peak={np.abs(f_fc).max():.2e} Pa")

        # Pairwise residuals
        rl2_an_jw = relative_l2(f_an, f_jw)
        rl2_an_fc = relative_l2(f_an, f_fc)
        rl2_jw_fc = relative_l2(f_jw, f_fc)

        peak_an, mag_an = focal_alignment(f_an, GRID_EXTENT_M)
        peak_jw, mag_jw = focal_alignment(f_jw, GRID_EXTENT_M)
        peak_fc, mag_fc = focal_alignment(f_fc, GRID_EXTENT_M)
        # focal-shift for first 3 configs that have a single target
        target = cfg.get("target")
        if isinstance(target, list) and len(target) == 3 and not isinstance(target[0], list):
            target_xyz = np.array(target, dtype=np.float64)
            shift_an = float(np.linalg.norm(np.array(peak_an) - target_xyz) * 1e3)
            shift_jw = float(np.linalg.norm(np.array(peak_jw) - target_xyz) * 1e3)
            shift_fc = float(np.linalg.norm(np.array(peak_fc) - target_xyz) * 1e3)
        else:
            shift_an = shift_jw = shift_fc = None

        print(
            f"  rel L2:  an↔jW {rl2_an_jw:.3f}  an↔FC {rl2_an_fc:.3f}  "
            f"jW↔FC {rl2_jw_fc:.3f}"
        )

        # Aggregate residual heatmap (sum |Δ| across configs).  Each field
        # is peak-normalised before differencing so the heatmap reflects
        # *structural* disagreement, not absolute-Pa calibration.
        a_n = np.abs(f_an) / max(np.abs(f_an).max(), 1e-30)
        j_n = np.abs(f_jw) / max(np.abs(f_jw).max(), 1e-30)
        f_n = np.abs(f_fc) / max(np.abs(f_fc).max(), 1e-30)
        aggregated_residuals["an_vs_jw"] += np.abs(a_n - j_n)
        aggregated_residuals["an_vs_fc"] += np.abs(a_n - f_n)
        aggregated_residuals["jw_vs_fc"] += np.abs(j_n - f_n)

        slice_path = out_dir / f"mid_z_slice_{idx}_{cfg['label']}.png"
        plot_three_way_slice(
            f_an, f_jw, f_fc, cfg["label"], slice_path, GRID_EXTENT_M,
        )

        per_config.append({
            "idx": idx,
            "label": cfg["label"],
            "description": cfg["description"],
            "target": cfg.get("target"),
            "rel_l2": {
                "analytical_vs_jwave": rl2_an_jw,
                "analytical_vs_femcoupled": rl2_an_fc,
                "jwave_vs_femcoupled": rl2_jw_fc,
            },
            "peak": {
                "analytical": {"xyz_m": list(peak_an), "mag_pa": mag_an},
                "jwave": {"xyz_m": list(peak_jw), "mag_pa": mag_jw},
                "femcoupled": {"xyz_m": list(peak_fc), "mag_pa": mag_fc},
            },
            "focal_shift_mm": {
                "analytical": shift_an,
                "jwave": shift_jw,
                "femcoupled": shift_fc,
            },
            "wall_s": {
                "analytical": t_an,
                "jwave": t_jw,
                "femcoupled": t_fc,
            },
            "slice_png": slice_path.name,
        })

    # ---------------------------------------------------------------------
    # Differentiability sanity check — jax.grad through the FEM path used
    # by this validation.  Phase 5 already proved this for the backend
    # entry point; Phase 6 verifies it stays true after running through
    # the validation pipeline.
    # ---------------------------------------------------------------------
    print(f"\n[{datetime.now():%H:%M:%S}] differentiability sanity check ...")
    try:
        from drip_physics.backends.femcoupled_backend import (
            compute_pressure_from_phases as fem_compute_inner,
        )
        env_diff = default_environment()
        n_t = L1_ARRAY.total_transducers
        small_state = fem_state_holder["state"]

        def loss_fn(phi):
            field = fem_compute_inner(
                phases=phi, env=env_diff,
                grid_shape=GRID_SHAPE, grid_extent_m=GRID_EXTENT_M,
                backend_state=small_state,
                pml_width=0.0, max_inner_iters=1, convergence_tol=1.0,
            )
            # Probe at chamber-centroid voxel.
            re = field[GRID_SHAPE[2] // 2, GRID_SHAPE[1] // 2, GRID_SHAPE[0] // 2, 0]
            im = field[GRID_SHAPE[2] // 2, GRID_SHAPE[1] // 2, GRID_SHAPE[0] // 2, 1]
            return re * re + im * im

        phases_diff = jnp.linspace(0.0, jnp.pi, n_t, dtype=jnp.float64)
        grad_arr = np.asarray(jax.grad(loss_fn)(phases_diff))
        diff_ok = bool(np.all(np.isfinite(grad_arr)))
        diff_norm = float(np.linalg.norm(grad_arr))
        print(
            f"  jax.grad(|p|²)(phases): finite={diff_ok} | "
            f"||grad||={diff_norm:.3e}"
        )
        diff_summary = {
            "method": "jax.grad on |p|² at chamber-centre voxel",
            "finite": diff_ok,
            "grad_norm": diff_norm,
            "shape": list(grad_arr.shape),
        }
    except Exception as e:  # noqa: BLE001
        diff_summary = {"method": "jax.grad on femcoupled forward", "finite": False, "error": str(e)}
        print(f"  differentiability check FAILED: {e}")

    # Mean attribution heatmap (divide by n_configs)
    for k in aggregated_residuals:
        aggregated_residuals[k] /= len(configs)
    plot_attribution(aggregated_residuals, out_dir / "attribution.png", GRID_EXTENT_M)

    # Aggregate stats
    rl2_an_jw_arr = np.array([c["rel_l2"]["analytical_vs_jwave"] for c in per_config])
    rl2_an_fc_arr = np.array([c["rel_l2"]["analytical_vs_femcoupled"] for c in per_config])
    rl2_jw_fc_arr = np.array([c["rel_l2"]["jwave_vs_femcoupled"] for c in per_config])

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_configs": len(configs),
            "grid_shape": list(GRID_SHAPE),
            "grid_extent_m": list(GRID_EXTENT_M),
            "femcoupled_pml_width": 0.0,
            "femcoupled_max_inner_iters": 2,
            "femcoupled_focus_h_mm": 2.0,
            "femcoupled_bulk_h_mm": 10.0,
            "jwave_grid_n": j_grid_n,
            "jwave_grid_dx_m": j_grid_dx,
            "jwave_cube_side_m": float(j_cube_side_m),
            "jwave_lambda_over_dx": float(lambda_m / j_grid_dx),
            "jwave_grid_rationale": (
                f"n={j_grid_n}, cube={j_cube_side_m*1e3:.1f} mm → "
                f"dx={j_grid_dx*1e3:.3f} mm = λ/{lambda_m/j_grid_dx:.2f}. "
                f"v0 used n=64 / dx=6.25 mm = λ/1.37 which under-resolved the "
                f"40 kHz wavelength (λ=8.575 mm) and caused the "
                f"jWave-vs-FEM > analytical-vs-FEM anomaly in 5/5 configs. "
                f"Cube must enclose the L1 array bbox (0.10×0.10×0.36 m) "
                f"with ≥4-cell margin per face, so cube ≥ 0.40 m."
            ),
            "mesh": {
                "element_count": int(info["element_count"]),
                "node_count": int(info["node_count"]),
                "path": str(mesh_path),
            },
        },
        "rel_l2_aggregate": {
            "analytical_vs_jwave": {
                "mean": float(rl2_an_jw_arr.mean()),
                "median": float(np.median(rl2_an_jw_arr)),
                "p90": float(np.percentile(rl2_an_jw_arr, 90)),
                "min": float(rl2_an_jw_arr.min()),
                "max": float(rl2_an_jw_arr.max()),
            },
            "analytical_vs_femcoupled": {
                "mean": float(rl2_an_fc_arr.mean()),
                "median": float(np.median(rl2_an_fc_arr)),
                "p90": float(np.percentile(rl2_an_fc_arr, 90)),
                "min": float(rl2_an_fc_arr.min()),
                "max": float(rl2_an_fc_arr.max()),
            },
            "jwave_vs_femcoupled": {
                "mean": float(rl2_jw_fc_arr.mean()),
                "median": float(np.median(rl2_jw_fc_arr)),
                "p90": float(np.percentile(rl2_jw_fc_arr, 90)),
                "min": float(rl2_jw_fc_arr.min()),
                "max": float(rl2_jw_fc_arr.max()),
            },
        },
        "anomalies": {
            "fem_jw_exceeds_an_fc_count": int(
                (rl2_jw_fc_arr > rl2_an_fc_arr).sum()
            ),
        },
        "differentiability": diff_summary,
        "per_config": per_config,
        "caveat": (
            "Analytical is unbounded free-field (no chamber walls); j-Wave is "
            "Cartesian-PML cube (no chamber walls); FEM-coupled is the L1 chamber "
            "mesh with finite-area Robin transducer faces (with pml_width=0, "
            "Sommerfeld first-order radiation BC). Disagreement vs analytical "
            "should rank: FEM > jWave because FEM additionally has chamber "
            "geometry + finite-area BC structure. jWave-vs-FEM should be "
            "smaller than analytical-vs-FEM (both solve Helmholtz) — if "
            "reversed, flag for investigation."
        ),
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{datetime.now():%H:%M:%S}] summary written: {summary_path}")

    print("\n=== Three-way residual aggregate (5 configs) ===")
    print(
        f"  analytical vs jWave:        mean={rl2_an_jw_arr.mean():.3f} "
        f"median={np.median(rl2_an_jw_arr):.3f} p90={np.percentile(rl2_an_jw_arr, 90):.3f}"
    )
    print(
        f"  analytical vs FEM-coupled:  mean={rl2_an_fc_arr.mean():.3f} "
        f"median={np.median(rl2_an_fc_arr):.3f} p90={np.percentile(rl2_an_fc_arr, 90):.3f}"
    )
    print(
        f"  jWave vs FEM-coupled:       mean={rl2_jw_fc_arr.mean():.3f} "
        f"median={np.median(rl2_jw_fc_arr):.3f} p90={np.percentile(rl2_jw_fc_arr, 90):.3f}"
    )
    if summary["anomalies"]["fem_jw_exceeds_an_fc_count"] > 0:
        print(
            "  *** ANOMALY: jWave-vs-FEM > analytical-vs-FEM in "
            f"{summary['anomalies']['fem_jw_exceeds_an_fc_count']} of "
            f"{len(configs)} configs — investigate ***"
        )


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
        help="Receipts output directory (default: _fem_three_way_v0)",
    )
    p.add_argument(
        "--j-grid-n", type=int, default=DEFAULT_J_GRID_N,
        help="j-Wave grid points per axis (default: %(default)s; v0 used 64)",
    )
    p.add_argument(
        "--j-cube-side-m", type=float, default=DEFAULT_J_CUBE_SIDE_M,
        help=(
            "j-Wave cube side length in metres (default: %(default)s). "
            "Must enclose the L1 array (≥0.40 m)."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    main(
        out_dir=Path(args.out_dir),
        j_grid_n=args.j_grid_n,
        j_cube_side_m=args.j_cube_side_m,
    )
