#!/usr/bin/env python3
"""Generate a j-Wave-grounded (phases -> 3D pressure field) mini-dataset.

Mirrors ``generate_dataset.py`` but uses the JAX-traceable Helmholtz PDE
solve from ``drip_physics.backends.jwave_backend.JWaveBackend`` to produce
the ground-truth fields, instead of the analytical 1/r superposition.

Differences from the analytical generator
-----------------------------------------
* Uniform phase configs only — no controller rollouts. Keeps the per-config
  cost predictable (~10-20 s on CPU for a 64³ Helmholtz solve).
* True 3D fields (no 2D slice mode). Output grid is ``(grid_n, grid_n, grid_n)``
  where the cube is centred on the array centroid in j-Wave's centred frame.
* A miniaturised cylindrical array (default radius 2 cm, height 2 cm,
  10 rings × 12 transducers = 120) is used so the array fits inside the 64³
  cube at ``dx = lambda/10``. The L1 array (radius 5 cm, height 40 cm) does
  NOT fit at this resolution; running a 64³ PDE through the L1 cube would
  require either a coarser dx (Nyquist-violating) or a much bigger grid_n
  (out of CPU budget). The miniaturised array preserves ``n_transducers=120``
  so the inverse problem dimensionality matches the v3 analytical dataset.
* Incremental HDF5 writes: every config is appended to a resizable dataset
  so a partial run produces a usable file. A KeyboardInterrupt or
  out-of-budget kill leaves the file consistent.

Output schema mirrors ``data/inverse_dataset_v3.h5`` so ``dataset.py`` reads
it transparently. The new attribute ``attrs.physics_backend`` is set to
``'jwave'`` for downstream filtering.

Usage
-----
::

    python -m ml_inverse.generate_jwave_dataset \
        --n-trajectories 200 \
        --field-dims 3 \
        --grid-resolution 64 64 64 \
        --output-path data/inverse_dataset_jwave_3d.h5 \
        --seed 42

Runtime
-------
On CPU/JAX (M2 Max), per-config wall-clock is ~10-15 s including the JIT
recompile (j-Wave's ``Domain`` static shape forces a retrace whenever
``phases`` changes — but the actual matrix factorisation is much faster
than the trace itself). Expect ~50-70 min for 200 configs end-to-end.
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
from typing import List, Optional, Tuple

import numpy as np

# Allow running as a script: drip_physics is the package one level up.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from drip_physics import __version__ as _DRIP_PHYSICS_VERSION  # noqa: E402
from drip_physics.config import (  # noqa: E402
    EnvironmentConfig,
    PressureConfig,
    default_environment,
)
from drip_physics.core import ArrayGeometry, SimulationParams  # noqa: E402
from drip_physics.geometry import create_cylindrical_array  # noqa: E402

logger = logging.getLogger(__name__)


# =============================================================================
# Generation config
# =============================================================================


@dataclass(frozen=True)
class JWaveDataConfig:
    """Configuration for a single j-Wave dataset generation run.

    All physical units are SI.

    Attributes
    ----------
    n_trajectories
        Number of independent uniform-random phase configs to sample.
    field_dims
        Output field dimensionality (3 only — j-Wave is genuinely 3D).
    grid_resolution
        ``(Nx, Ny, Nz)`` — must all equal ``grid_n``.
    grid_dx_m
        Cell spacing in meters. Default lambda/10 at 40 kHz ~ 0.858 mm.
    array_radius_m, array_height_m
        Miniaturised cylindrical array dimensions chosen so the array
        fits inside the j-Wave cube at the requested ``grid_dx_m``.
    rings, per_ring
        Transducer count = ``rings * per_ring``. Default 10x12 = 120 to
        keep parity with the v3 analytical dataset.
    output_path
        Output HDF5 path.
    seed
        RNG seed; reproducibility.
    write_every
        Flush incremental rows to disk every ``write_every`` configs.
    """

    n_trajectories: int = 200
    field_dims: int = 3
    grid_resolution: Tuple[int, int, int] = (64, 64, 64)
    grid_dx_m: float = 0.858e-3
    array_radius_m: float = 0.02
    array_height_m: float = 0.02
    rings: int = 10
    per_ring: int = 12
    output_path: str = "ml_inverse/data/inverse_dataset_jwave_3d.h5"
    seed: int = 42
    write_every: int = 5  # incremental flush cadence (configs)

    def __post_init__(self) -> None:  # noqa: D401 - dataclass hook
        if self.field_dims != 3:
            raise ValueError(
                f"field_dims must be 3 (j-Wave is 3D); got {self.field_dims}"
            )
        if len(self.grid_resolution) != 3:
            raise ValueError(
                f"grid_resolution must have length 3; got {self.grid_resolution}"
            )
        if self.grid_dx_m <= 0.0:
            raise ValueError(f"grid_dx_m must be > 0; got {self.grid_dx_m}")
        if self.n_trajectories < 1:
            raise ValueError(
                f"n_trajectories must be >= 1; got {self.n_trajectories}"
            )
        if self.write_every < 1:
            raise ValueError(f"write_every must be >= 1; got {self.write_every}")
        # Per-axis sanity: each axis must contain the array bbox + 4-cell margin.
        # Anisotropic-aware (Nx, Ny, Nz may differ).
        bbox_per_axis = (
            2.0 * self.array_radius_m,  # x bbox
            2.0 * self.array_radius_m,  # y bbox
            self.array_height_m,        # z bbox
        )
        axes = ("x", "y", "z")
        for k, ax in enumerate(axes):
            side = self.grid_resolution[k] * self.grid_dx_m
            required = bbox_per_axis[k] + 4.0 * self.grid_dx_m
            if side < required:
                raise ValueError(
                    f"j-Wave {ax}-axis side {side*1e3:.2f} mm cannot contain "
                    f"array bbox {bbox_per_axis[k]*1e3:.2f} mm + 4-cell margin "
                    f"({required*1e3:.2f} mm required). Increase "
                    f"grid_resolution[{k}] or grid_dx_m."
                )


# =============================================================================
# Geometry helpers
# =============================================================================


def _build_array(cfg: JWaveDataConfig) -> ArrayGeometry:
    """Construct the miniaturised cylindrical array used for j-Wave gen."""
    positions = create_cylindrical_array(
        radius=cfg.array_radius_m,
        height=cfg.array_height_m,
        rings=cfg.rings,
        per_ring=cfg.per_ring,
    )
    params = SimulationParams()
    return ArrayGeometry(positions=positions, params=params)


def _build_grid_axes(
    array: ArrayGeometry, cfg: JWaveDataConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce JWaveBackend's centred-grid axes in array frame.

    Cell i on axis k sits at ``offset[k] + i*dx − n[k]*dx/2`` where
    ``offset`` is the array centroid (matches JWaveBackend._compute_origin_offset).
    Anisotropic-aware: per-axis n.
    """
    nx, ny, nz = cfg.grid_resolution
    dx = cfg.grid_dx_m
    offset = np.asarray(array.positions, dtype=np.float64).mean(axis=0)
    x = (np.arange(nx, dtype=np.float64) * dx - 0.5 * nx * dx + offset[0]).astype(np.float32)
    y = (np.arange(ny, dtype=np.float64) * dx - 0.5 * ny * dx + offset[1]).astype(np.float32)
    z = (np.arange(nz, dtype=np.float64) * dx - 0.5 * nz * dx + offset[2]).astype(np.float32)
    return x, y, z


# =============================================================================
# Incremental HDF5 writer
# =============================================================================


def _open_or_create_hdf5(
    cfg: JWaveDataConfig,
    array: ArrayGeometry,
    env: EnvironmentConfig,
    coord_axes: Tuple[np.ndarray, np.ndarray, np.ndarray],
):
    """Open or create the HDF5 file with resizable datasets.

    Returns the open ``h5py.File`` handle. Caller is responsible for
    closing it.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py required for dataset generation. Install: pip install h5py"
        ) from exc

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_t = int(array.num_transducers)
    grid_shape = tuple(cfg.grid_resolution)

    f = h5py.File(output_path, "w")

    # Resizable datasets (axis 0 grows as configs are appended).
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

    nx, ny, nz = cfg.grid_resolution
    dx = cfg.grid_dx_m
    offset = np.asarray(array.positions, dtype=np.float64).mean(axis=0)
    half_side = (0.5 * nx * dx, 0.5 * ny * dx, 0.5 * nz * dx)

    # Static attrs at file creation; non-static attrs (n_trajectories,
    # n_skipped_nonfinite) are written at finalize time.
    f.attrs["field_dims"] = int(cfg.field_dims)
    f.attrs["grid_resolution"] = np.asarray(grid_shape, dtype=np.int64)
    f.attrs["slice_extent_m"] = np.asarray(
        [
            [offset[0] - half_side[0], offset[0] + half_side[0]],
            [offset[1] - half_side[1], offset[1] + half_side[1]],
            [offset[2] - half_side[2], offset[2] + half_side[2]],
        ],
        dtype=np.float64,
    )
    # Legacy attr — j-Wave is 3D, no single slice z.
    f.attrs["slice_z_m"] = float("nan")
    f.attrs["frequency_hz"] = float(env.transducer.frequency)
    f.attrs["sound_speed_m_s"] = float(env.speed_of_sound)
    f.attrs["n_transducers"] = int(n_t)
    f.attrs["seed"] = int(cfg.seed)
    f.attrs["drip_physics_version"] = str(_DRIP_PHYSICS_VERSION)
    f.attrs["physics_backend"] = "jwave"
    f.attrs["grid_dx_m"] = float(cfg.grid_dx_m)
    f.attrs["array_radius_m"] = float(cfg.array_radius_m)
    f.attrs["array_height_m"] = float(cfg.array_height_m)
    f.attrs["uniform_z_mode"] = "centred_grid"  # n/a for 3D j-Wave
    f.attrs["uniform_z_range_m"] = np.asarray(
        [offset[2] - half_side[2], offset[2] + half_side[2]], dtype=np.float64
    )
    f.attrs["rollout_initial_z_m"] = float("nan")  # rollouts not used here

    # Initialised at finalize time.
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


def generate(cfg: JWaveDataConfig) -> str:
    """Run the j-Wave dataset generation pipeline. Returns the HDF5 path."""
    # Lazy JAX import: only required for the actual gen.
    import jax
    import jax.numpy as jnp

    from drip_physics.backends.jwave_backend import JWaveBackend

    rng = np.random.default_rng(cfg.seed)

    env = default_environment()
    array = _build_array(cfg)
    n_t = array.num_transducers

    coord_axes = _build_grid_axes(array, cfg)

    press_cfg = PressureConfig(
        backend="jwave",
        grid_shape=tuple(cfg.grid_resolution),
        grid_dx=cfg.grid_dx_m,
    )
    backend = JWaveBackend(config=press_cfg, env=env)

    nx, ny, nz = cfg.grid_resolution
    logger.info(
        "Generating j-Wave dataset: %d configs, grid=%s, dx=%.4e m, "
        "side=(%.1f, %.1f, %.1f) mm, array=(r=%.2f mm, h=%.2f mm, %d tx)",
        cfg.n_trajectories,
        cfg.grid_resolution,
        cfg.grid_dx_m,
        nx * cfg.grid_dx_m * 1e3,
        ny * cfg.grid_dx_m * 1e3,
        nz * cfg.grid_dx_m * 1e3,
        cfg.array_radius_m * 1e3,
        cfg.array_height_m * 1e3,
        n_t,
    )
    logger.info("JAX devices: %s", jax.devices())

    # Open file (truncate-and-create); incremental rows from here.
    f = _open_or_create_hdf5(cfg, array, env, coord_axes)

    skip_log: List[Tuple[int, str]] = []
    n_written = 0
    amps_jnp = jnp.ones(n_t, dtype=jnp.float32)

    t_start = time.time()
    per_config_times: List[float] = []
    last_log_time = t_start

    try:
        for i in range(cfg.n_trajectories):
            traj_id = i
            phase_vec = rng.uniform(0.0, 2.0 * np.pi, size=n_t).astype(np.float64)

            t0 = time.time()
            try:
                phases_jnp = jnp.asarray(phase_vec, dtype=jnp.float32)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    field_jnp = backend._solve_field_jax(
                        array, phases_jnp, amps_jnp, env
                    )
                    field_jnp.block_until_ready()
                field_np = np.asarray(field_jnp)
                fr = field_np.real.astype(np.float32)
                fi = field_np.imag.astype(np.float32)
            except Exception as exc:  # pragma: no cover — defensive
                skip_log.append(
                    (traj_id, f"jwave solve exception: {type(exc).__name__}: {exc!r}")
                )
                logger.warning(
                    "config %d: skipped (exception %s: %r)", traj_id, type(exc).__name__, exc
                )
                continue
            t_elapsed = time.time() - t0
            per_config_times.append(t_elapsed)

            if not _is_finite_complex(fr, fi):
                skip_log.append((traj_id, "jwave non-finite field"))
                logger.warning(
                    "config %d: skipped (non-finite field, t=%.2fs)",
                    traj_id,
                    t_elapsed,
                )
                continue

            # slice_z_m for 3D = NaN (no single slice plane in 3D j-Wave).
            _append_row(
                f,
                phases=phase_vec.astype(np.float32),
                field_real=fr,
                field_imag=fi,
                trajectory_id=traj_id,
                source=0,
                frame_idx=0,
                slice_z_m=float("nan"),
            )
            n_written += 1

            # Periodic flush + log.
            if n_written % cfg.write_every == 0:
                f.flush()
                now = time.time()
                cur_mean = float(np.mean(per_config_times[-cfg.write_every:]))
                elapsed = now - t_start
                eta_min = (
                    (cfg.n_trajectories - (i + 1)) * cur_mean / 60.0
                    if cur_mean > 0
                    else float("nan")
                )
                logger.info(
                    "[%4d/%d]  t_iter_recent=%.2fs  elapsed=%.1f min  eta=%.1f min  "
                    "skipped=%d",
                    i + 1,
                    cfg.n_trajectories,
                    cur_mean,
                    elapsed / 60.0,
                    eta_min,
                    len(skip_log),
                )
                last_log_time = now

    finally:
        # Always finalize the file even if we crashed/got interrupted.
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

    # Write a sidecar JSON receipt.
    receipt_path = Path(cfg.output_path).with_suffix(".gen_receipt.json")
    receipt = {
        "output_path": str(cfg.output_path),
        "n_requested": int(cfg.n_trajectories),
        "n_written": int(n_written),
        "n_skipped": int(len(skip_log)),
        "wall_time_s": float(elapsed),
        "wall_time_min": float(elapsed / 60.0),
        "per_config_time_mean_s": float(np.mean(per_config_times))
        if per_config_times
        else None,
        "per_config_time_median_s": float(np.median(per_config_times))
        if per_config_times
        else None,
        "grid_resolution": list(cfg.grid_resolution),
        "grid_dx_m": float(cfg.grid_dx_m),
        "n_transducers": int(n_t),
        "array_radius_m": float(cfg.array_radius_m),
        "array_height_m": float(cfg.array_height_m),
        "physics_backend": "jwave",
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
        f"[generate_jwave_dataset] runtime_s={elapsed:.1f} "
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
    p.add_argument("--field-dims", type=int, default=3, choices=(3,))
    p.add_argument(
        "--grid-resolution",
        type=int,
        nargs="+",
        default=[64, 64, 64],
    )
    p.add_argument(
        "--grid-dx-mm",
        type=float,
        default=0.858,
        help="Grid spacing in mm. Default 0.858 = lambda/10 at 40 kHz.",
    )
    p.add_argument(
        "--array-radius-mm",
        type=float,
        default=20.0,
        help="Cylindrical array radius (mm). Default 20 fits inside 64^3 cube.",
    )
    p.add_argument(
        "--array-height-mm",
        type=float,
        default=20.0,
        help="Cylindrical array height (mm). Default 20 fits inside 64^3 cube.",
    )
    p.add_argument("--rings", type=int, default=10)
    p.add_argument("--per-ring", type=int, default=12)
    p.add_argument(
        "--output-path",
        type=str,
        default="ml_inverse/data/inverse_dataset_jwave_3d.h5",
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
            "--grid-resolution must be either 1 or 3 ints; got "
            f"{args.grid_resolution}"
        )

    cfg = JWaveDataConfig(
        n_trajectories=int(args.n_trajectories),
        field_dims=int(args.field_dims),
        grid_resolution=grid_res,
        grid_dx_m=float(args.grid_dx_mm) * 1e-3,
        array_radius_m=float(args.array_radius_mm) * 1e-3,
        array_height_m=float(args.array_height_mm) * 1e-3,
        rings=int(args.rings),
        per_ring=int(args.per_ring),
        output_path=args.output_path,
        seed=int(args.seed),
        write_every=int(args.write_every),
    )

    if not os.path.isabs(cfg.output_path):
        cfg = JWaveDataConfig(
            **{
                **cfg.__dict__,
                "output_path": str(Path(cfg.output_path).resolve()),
            },
        )

    generate(cfg)


if __name__ == "__main__":
    main()
