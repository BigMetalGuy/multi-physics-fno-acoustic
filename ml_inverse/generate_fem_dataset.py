#!/usr/bin/env python3
"""Generate a FEM-coupled-physics-grounded (phases -> 3D pressure field) dataset.

Mirror of ``generate_jwave_dataset.py`` but with the j-Wave Helmholtz solve
replaced by the Phase 5 FEM-coupled backend
(:func:`drip_physics.backends.femcoupled_backend.compute_pressure_from_phases`).

Key differences from the j-Wave generator
-----------------------------------------
* Uses the L1 chamber mesh (radius 5 cm, height 40 cm) instead of a
  miniaturised cylinder. The FEM-coupled backend operates on the actual
  L1 geometry — no need to fit a cube around it like j-Wave does.
* The voxel grid is the FEM backend's default ``DEFAULT_GRID_EXTENT_M``
  (a 6×6×20 cm box around the focal zone) projected from the mesh nodes.
* The backend's ``backend_state`` is threaded across configs so the
  ~17–25 s mesh + Problem build amortises across the corpus. Saves ~50
  minutes vs naive per-call construction over 200 configs.
* Stores the FEM mesh + voxel-extent metadata in HDF5 attrs so downstream
  consumers (FNO training, disagreement analysis) can rebuild the grid.
* ``physics_backend`` attr is set to ``femcoupled``; ``source`` field is
  set to ``2`` (per ``FNO_F_TRAINING_PLAN.md``: 0=analytical, 1=jwave,
  2=femcoupled).

Output schema mirrors ``inverse_dataset_jwave_3d_full.h5`` so
``InverseSurrogateDataset`` reads it transparently.

Usage
-----
::

    /path/to/jaxfem-venv/python -m ml_inverse.generate_fem_dataset \
        --n-trajectories 200 \
        --grid-resolution 32 32 32 \
        --output-path data/inverse_dataset_fem_3d.h5 \
        --seed 42

Runtime
-------
Per-config wall on the coarse L1 mesh with ``max_inner_iters=2`` is
~2–6 s after the first call (which pays the ~22 s mesh + Problem build).
Expect ~12–20 min for 200 configs on a single CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Allow running as a script: drip_physics is the package one level up.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from drip_physics import __version__ as _DRIP_PHYSICS_VERSION  # noqa: E402
from drip_physics.config import (  # noqa: E402
    EnvironmentConfig,
    default_environment,
)
from drip_physics.core import ArrayGeometry, SimulationParams  # noqa: E402
from drip_physics.geometry import create_cylindrical_array  # noqa: E402

logger = logging.getLogger(__name__)


# =============================================================================
# Generation config
# =============================================================================


@dataclass(frozen=True)
class FemDataConfig:
    """Configuration for a single FEM-coupled dataset generation run.

    All physical units are SI.

    Attributes
    ----------
    n_trajectories
        Number of independent uniform-random phase configs to sample.
    grid_resolution
        ``(Nx, Ny, Nz)`` voxel grid resolution. Default 32^3 for the
        coarse-mesh L1 chamber.
    grid_extent_m
        Voxel-grid extent in metres ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
        Default matches the FEM backend's ``DEFAULT_GRID_EXTENT_M``.
    pml_width
        FEM Helmholtz PML width (m). 0.0 = first-order Sommerfeld.
    focus_h_mm, bulk_h_mm
        Mesh-density targets passed to ``build_l1_chamber_mesh``.
    max_inner_iters
        Inner fixed-point iteration cap. 2 is the speed default for the
        coarse-mesh forward; 3 is Phase 4 default.
    output_path
        Output HDF5 path.
    seed
        RNG seed; reproducibility.
    write_every
        Flush incremental rows to disk every ``write_every`` configs.
    """

    n_trajectories: int = 200
    grid_resolution: Tuple[int, int, int] = (32, 32, 32)
    grid_extent_m: Tuple[float, float, float, float, float, float] = (
        -0.03, 0.03, -0.03, 0.03, 0.10, 0.30,
    )
    pml_width: float = 0.0
    focus_h_mm: float = 2.0
    bulk_h_mm: float = 10.0
    max_inner_iters: int = 2
    velocity_amp: Optional[float] = None  # None = backend default (0.01 m/s)
    output_path: str = "ml_inverse/data/inverse_dataset_fem_3d.h5"
    seed: int = 42
    write_every: int = 5

    def __post_init__(self) -> None:
        if len(self.grid_resolution) != 3:
            raise ValueError(
                f"grid_resolution must have length 3; got {self.grid_resolution}"
            )
        if self.n_trajectories < 1:
            raise ValueError(
                f"n_trajectories must be >= 1; got {self.n_trajectories}"
            )
        if self.write_every < 1:
            raise ValueError(f"write_every must be >= 1; got {self.write_every}")
        if len(self.grid_extent_m) != 6:
            raise ValueError(
                f"grid_extent_m must have length 6; got {self.grid_extent_m}"
            )


# =============================================================================
# Geometry helpers
# =============================================================================


def _build_array(env: EnvironmentConfig) -> ArrayGeometry:
    """Construct the L1 cylindrical array from env config (5 cm × 40 cm)."""
    arr_cfg = env.array
    positions = create_cylindrical_array(
        radius=arr_cfg.cylinder_radius,
        height=arr_cfg.height,
        rings=arr_cfg.ring_count,
        per_ring=arr_cfg.transducers_per_ring,
    )
    params = SimulationParams()
    return ArrayGeometry(positions=positions, params=params)


def _build_grid_axes(
    cfg: FemDataConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the FEM backend's voxel-grid axes.

    Backend uses ``np.linspace(min, max, n)`` for each axis; we mirror it
    here so the saved coordinate grid lines up with the field cells.
    """
    nx, ny, nz = cfg.grid_resolution
    xmin, xmax, ymin, ymax, zmin, zmax = cfg.grid_extent_m
    x = np.linspace(xmin, xmax, nx, dtype=np.float32)
    y = np.linspace(ymin, ymax, ny, dtype=np.float32)
    z = np.linspace(zmin, zmax, nz, dtype=np.float32)
    return x, y, z


# =============================================================================
# Incremental HDF5 writer
# =============================================================================


def _open_or_create_hdf5(
    cfg: FemDataConfig,
    array: ArrayGeometry,
    env: EnvironmentConfig,
    coord_axes: Tuple[np.ndarray, np.ndarray, np.ndarray],
):
    """Open or create the HDF5 file with resizable datasets."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py required for dataset generation. Install: pip install h5py"
        ) from exc

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_t = int(array.num_transducers)
    nx, ny, nz = cfg.grid_resolution
    # FEM backend stores fields as (nz, ny, nx, 2). We persist them as
    # (nx, ny, nz) (HDF5 grid_shape attr matches the v3 dataset's
    # transposition convention used by InverseSurrogateDataset). The
    # field_real / field_imag datasets are sized (n, nz, ny, nx) which
    # downstream code already accepts (j-Wave dataset uses the same).
    grid_shape = (nz, ny, nx)

    f = h5py.File(output_path, "w")

    f.create_dataset(
        "phases",
        shape=(0, n_t),
        maxshape=(None, n_t),
        dtype=np.float32,
        chunks=(1, n_t),
    )
    f.create_dataset(
        "field_real",
        shape=(0, *grid_shape),
        maxshape=(None, *grid_shape),
        dtype=np.float32,
        chunks=(1, *grid_shape),
        compression="gzip",
        compression_opts=4,
    )
    f.create_dataset(
        "field_imag",
        shape=(0, *grid_shape),
        maxshape=(None, *grid_shape),
        dtype=np.float32,
        chunks=(1, *grid_shape),
        compression="gzip",
        compression_opts=4,
    )
    f.create_dataset(
        "trajectory_id",
        shape=(0,),
        maxshape=(None,),
        dtype=np.int64,
    )
    f.create_dataset(
        "source",
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint8,
    )
    f.create_dataset(
        "frame_idx",
        shape=(0,),
        maxshape=(None,),
        dtype=np.int64,
    )
    f.create_dataset(
        "slice_z_m",
        shape=(0,),
        maxshape=(None,),
        dtype=np.float32,
    )

    f.create_dataset(
        "transducer_positions",
        data=np.asarray(array.positions, dtype=np.float32),
    )

    coords_grp = f.create_group("grid_coords")
    coords_grp.create_dataset("x", data=coord_axes[0])
    coords_grp.create_dataset("y", data=coord_axes[1])
    coords_grp.create_dataset("z", data=coord_axes[2])

    xmin, xmax, ymin, ymax, zmin, zmax = cfg.grid_extent_m
    f.attrs["field_dims"] = 3
    f.attrs["grid_resolution"] = np.asarray(grid_shape, dtype=np.int64)
    f.attrs["slice_extent_m"] = np.asarray(
        [
            [xmin, xmax],
            [ymin, ymax],
            [zmin, zmax],
        ],
        dtype=np.float64,
    )
    f.attrs["slice_z_m"] = float("nan")
    f.attrs["frequency_hz"] = float(env.transducer.frequency)
    f.attrs["sound_speed_m_s"] = float(env.speed_of_sound)
    f.attrs["n_transducers"] = int(n_t)
    f.attrs["seed"] = int(cfg.seed)
    f.attrs["drip_physics_version"] = str(_DRIP_PHYSICS_VERSION)
    f.attrs["physics_backend"] = "femcoupled"
    f.attrs["pml_width_m"] = float(cfg.pml_width)
    f.attrs["focus_h_mm"] = float(cfg.focus_h_mm)
    f.attrs["bulk_h_mm"] = float(cfg.bulk_h_mm)
    f.attrs["max_inner_iters"] = int(cfg.max_inner_iters)
    f.attrs["uniform_z_mode"] = "centred_grid"
    f.attrs["uniform_z_range_m"] = np.asarray([zmin, zmax], dtype=np.float64)
    f.attrs["rollout_initial_z_m"] = float("nan")

    f.attrs["n_trajectories"] = 0
    f.attrs["n_skipped_nonfinite"] = 0

    return f


def _append_row(
    f,
    *,
    phases: np.ndarray,
    field_real: np.ndarray,
    field_imag: np.ndarray,
    trajectory_id: int,
    source: int,
    frame_idx: int,
    slice_z_m: float,
) -> None:
    """Append a single row to all parallel datasets in the open HDF5 file."""
    for name, value, dtype in (
        ("phases", phases.astype(np.float32), np.float32),
        ("field_real", field_real.astype(np.float32), np.float32),
        ("field_imag", field_imag.astype(np.float32), np.float32),
        ("trajectory_id", np.int64(trajectory_id), np.int64),
        ("source", np.uint8(source), np.uint8),
        ("frame_idx", np.int64(frame_idx), np.int64),
        ("slice_z_m", np.float32(slice_z_m), np.float32),
    ):
        ds = f[name]
        cur = ds.shape[0]
        ds.resize((cur + 1,) + ds.shape[1:])
        if name in ("phases", "field_real", "field_imag"):
            ds[cur] = np.asarray(value, dtype=dtype)
        else:
            ds[cur] = value


# =============================================================================
# Driver
# =============================================================================


def _is_finite_complex(field_real: np.ndarray, field_imag: np.ndarray) -> bool:
    return bool(np.isfinite(field_real).all() and np.isfinite(field_imag).all())


def generate(cfg: FemDataConfig) -> str:
    """Run the FEM-coupled dataset generation pipeline. Returns HDF5 path."""
    # Lazy JAX import (only required for the actual gen).
    import jax  # noqa: F401  — used implicitly by the backend
    import jax.numpy as jnp

    from drip_physics.backends.femcoupled_backend import (
        compute_pressure_from_phases,
    )

    rng = np.random.default_rng(cfg.seed)

    env = default_environment()
    array = _build_array(env)
    n_t = array.num_transducers

    coord_axes = _build_grid_axes(cfg)

    logger.info(
        "Generating FEM-coupled dataset: %d configs, grid=%s, extent=%s, "
        "pml=%.4fm, focus_h=%.1fmm, bulk_h=%.1fmm, max_inner_iters=%d, "
        "n_transducers=%d",
        cfg.n_trajectories,
        cfg.grid_resolution,
        cfg.grid_extent_m,
        cfg.pml_width,
        cfg.focus_h_mm,
        cfg.bulk_h_mm,
        cfg.max_inner_iters,
        n_t,
    )

    f = _open_or_create_hdf5(cfg, array, env, coord_axes)

    skip_log: List[Tuple[int, str]] = []
    n_written = 0
    backend_state = None  # threaded across configs to amortise mesh build

    t_start = time.time()
    per_config_times: List[float] = []

    try:
        for i in range(cfg.n_trajectories):
            traj_id = i
            phase_vec = rng.uniform(0.0, 2.0 * np.pi, size=n_t).astype(np.float64)

            t0 = time.time()
            try:
                phases_jnp = jnp.asarray(phase_vec, dtype=jnp.float64)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    field, backend_state = compute_pressure_from_phases(
                        phases_jnp,
                        env,
                        grid_shape=cfg.grid_resolution,
                        grid_extent_m=cfg.grid_extent_m,
                        pml_width=cfg.pml_width,
                        focus_h_mm=cfg.focus_h_mm,
                        bulk_h_mm=cfg.bulk_h_mm,
                        max_inner_iters=cfg.max_inner_iters,
                        velocity_amp=cfg.velocity_amp,
                        backend_state=backend_state,
                        return_state=True,
                    )
                # field shape: (nz, ny, nx, 2). Split into Re/Im stacks
                # of shape (nz, ny, nx) each.
                field_np = np.asarray(field)
                fr = field_np[..., 0].astype(np.float32)
                fi = field_np[..., 1].astype(np.float32)
            except Exception as exc:  # pragma: no cover — defensive
                skip_log.append(
                    (
                        traj_id,
                        f"femcoupled solve exception: "
                        f"{type(exc).__name__}: {exc!r}",
                    )
                )
                logger.warning(
                    "config %d: skipped (exception %s: %r)",
                    traj_id,
                    type(exc).__name__,
                    exc,
                )
                continue
            t_elapsed = time.time() - t0
            per_config_times.append(t_elapsed)

            if not _is_finite_complex(fr, fi):
                skip_log.append((traj_id, "femcoupled non-finite field"))
                logger.warning(
                    "config %d: skipped (non-finite field, t=%.2fs)",
                    traj_id,
                    t_elapsed,
                )
                continue

            _append_row(
                f,
                phases=phase_vec.astype(np.float32),
                field_real=fr,
                field_imag=fi,
                trajectory_id=traj_id,
                source=2,  # 0=analytical, 1=jwave, 2=femcoupled
                frame_idx=0,
                slice_z_m=float("nan"),
            )
            n_written += 1

            if n_written % cfg.write_every == 0:
                f.flush()
                cur_mean = float(np.mean(per_config_times[-cfg.write_every:]))
                elapsed = time.time() - t_start
                eta_min = (
                    (cfg.n_trajectories - (i + 1)) * cur_mean / 60.0
                    if cur_mean > 0
                    else float("nan")
                )
                logger.info(
                    "[%4d/%d]  t_iter_recent=%.2fs  elapsed=%.1f min  "
                    "eta=%.1f min  skipped=%d",
                    i + 1,
                    cfg.n_trajectories,
                    cur_mean,
                    elapsed / 60.0,
                    eta_min,
                    len(skip_log),
                )

    finally:
        try:
            f.attrs["n_trajectories"] = int(n_written)
            f.attrs["n_skipped_nonfinite"] = int(len(skip_log))
            elapsed_total = time.time() - t_start
            f.attrs["wall_time_s"] = float(elapsed_total)
            if per_config_times:
                f.attrs["per_config_time_mean_s"] = float(np.mean(per_config_times))
                f.attrs["per_config_time_median_s"] = float(
                    np.median(per_config_times)
                )
            f.flush()
        finally:
            f.close()

    elapsed = time.time() - t_start

    receipt_path = Path(cfg.output_path).with_suffix(".gen_receipt.json")
    receipt = {
        "output_path": str(cfg.output_path),
        "n_requested": int(cfg.n_trajectories),
        "n_written": int(n_written),
        "n_skipped": int(len(skip_log)),
        "wall_time_s": float(elapsed),
        "wall_time_min": float(elapsed / 60.0),
        "per_config_time_mean_s": (
            float(np.mean(per_config_times)) if per_config_times else None
        ),
        "per_config_time_median_s": (
            float(np.median(per_config_times)) if per_config_times else None
        ),
        "grid_resolution": list(cfg.grid_resolution),
        "grid_extent_m": list(cfg.grid_extent_m),
        "pml_width_m": float(cfg.pml_width),
        "focus_h_mm": float(cfg.focus_h_mm),
        "bulk_h_mm": float(cfg.bulk_h_mm),
        "max_inner_iters": int(cfg.max_inner_iters),
        "n_transducers": int(n_t),
        "physics_backend": "femcoupled",
        "seed": int(cfg.seed),
        "skip_log_first_5": skip_log[:5],
    }
    with open(receipt_path, "w") as fh:
        json.dump(receipt, fh, indent=2)
    logger.info("wrote receipt %s", receipt_path)

    if skip_log:
        logger.warning(
            "Skipped %d non-finite/exception configs. First 5 reasons:",
            len(skip_log),
        )
        for tid, msg in skip_log[:5]:
            logger.warning("  trajectory_id=%s: %s", tid, msg)

    logger.info(
        "TOTAL runtime: %.1f s (%.1f min) for %d/%d configs (%d skipped)",
        elapsed,
        elapsed / 60.0,
        n_written,
        cfg.n_trajectories,
        len(skip_log),
    )
    print(
        f"[generate_fem_dataset] runtime_s={elapsed:.1f} "
        f"runtime_min={elapsed/60.0:.1f} "
        f"n_written={n_written} "
        f"n_requested={cfg.n_trajectories} "
        f"skipped={len(skip_log)} "
        f"output={cfg.output_path}"
    )
    return cfg.output_path


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-trajectories", type=int, default=200)
    p.add_argument(
        "--grid-resolution",
        type=int,
        nargs="+",
        default=[32, 32, 32],
    )
    p.add_argument(
        "--grid-extent-m",
        type=float,
        nargs=6,
        default=[-0.03, 0.03, -0.03, 0.03, 0.10, 0.30],
        help="(xmin xmax ymin ymax zmin zmax) in metres.",
    )
    p.add_argument(
        "--pml-width",
        type=float,
        default=0.0,
        help="FEM Helmholtz PML width (m). 0.0 = first-order Sommerfeld.",
    )
    p.add_argument("--focus-h-mm", type=float, default=2.0)
    p.add_argument("--bulk-h-mm", type=float, default=10.0)
    p.add_argument(
        "--max-inner-iters",
        type=int,
        default=2,
        help="Inner fixed-point iteration cap. 2=speed default; 3=Phase 4.",
    )
    p.add_argument(
        "--velocity-amp",
        type=float,
        default=None,
        help=(
            "Piston velocity amplitude (m/s) at each transducer face. "
            "None (default) lets the backend use its spec-sheet default "
            "(~0.01 m/s, ~1 Pa peak fields). Use 1.0 to reach the "
            "FEM_V2_DESIGN §3.4 design pressure (~1500 Pa) where coupling "
            "effects (Eckart streaming + heat advection) become "
            "nontrivial — see validate_coupled.py and Phase 6.5 receipt."
        ),
    )
    p.add_argument(
        "--output-path",
        type=str,
        default="ml_inverse/data/inverse_dataset_fem_3d.h5",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--write-every",
        type=int,
        default=5,
        help="Flush incremental rows every N configs.",
    )
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if len(args.grid_resolution) == 1:
        grid_res = (args.grid_resolution[0],) * 3
    elif len(args.grid_resolution) == 3:
        grid_res = tuple(args.grid_resolution)
    else:
        raise ValueError(
            "--grid-resolution must be 1 or 3 ints; got "
            f"{args.grid_resolution}"
        )

    cfg = FemDataConfig(
        n_trajectories=int(args.n_trajectories),
        grid_resolution=grid_res,
        grid_extent_m=tuple(args.grid_extent_m),
        pml_width=float(args.pml_width),
        focus_h_mm=float(args.focus_h_mm),
        bulk_h_mm=float(args.bulk_h_mm),
        max_inner_iters=int(args.max_inner_iters),
        velocity_amp=float(args.velocity_amp) if args.velocity_amp is not None else None,
        output_path=args.output_path,
        seed=int(args.seed),
        write_every=int(args.write_every),
    )

    if not os.path.isabs(cfg.output_path):
        cfg = FemDataConfig(
            **{
                **cfg.__dict__,
                "output_path": str(Path(cfg.output_path).resolve()),
            }
        )

    generate(cfg)


if __name__ == "__main__":
    main()
