#!/usr/bin/env python3
"""Generate the (phases -> pressure field) FNO surrogate dataset.

Two sampling modes are mixed in one HDF5:

1. **Uniform** - random phases ~ Uniform[0, 2*pi) per transducer. Field
   computed by ``pressure.compute_pressure_grid`` (3D) or
   ``pressure.compute_pressure_slice`` (2D). Each config is its own
   ``trajectory_id``.

2. **Controller rollout** - run ``simulate_path_tracking`` on a randomized
   lateral target inside +/- 15 mm with ``styrofoam_bead`` defaults. Harvest
   the issued phase commands at every ``rollout_stride``-th controller
   timestep and compute the pressure field at the droplet's z-plane (in 2D
   mode) or on the full 3D grid (3D mode). Each rollout shares one
   ``trajectory_id``.

The output schema is documented in ``simulations/ml_inverse/SCHEMA.md`` and
non-finite phase configs are skipped (count recorded in
``attrs.n_skipped_nonfinite``).

Runtime
-------
On the workstation used for first development run (Apple silicon, no GPU,
Python 3.13, numpy stock BLAS) the default 500-trajectory invocation
(400 uniform + 100 rollouts, 2D, 64x64 grid, rollout_stride=8) ran end-to-end
in ~1928 s (~32 min). Uniform sampling: ~12 s. Controller-rollout sampling +
field computation: ~1912 s (the dominant cost — ~30 ms per 64x64 slice times
~626 slices per rollout times 100 rollouts).
See the ``[generate_dataset] runtime_s=...`` line printed at end of run for
exact timing on the current machine.

Usage
-----
::

    python simulations/ml_inverse/generate_dataset.py
    python simulations/ml_inverse/generate_dataset.py --n-trajectories 10000

"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Allow running as a script: drip_physics is the package one level up.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from drip_physics import __version__ as _DRIP_PHYSICS_VERSION  # noqa: E402
from drip_physics.api import _make_array, _make_droplet  # noqa: E402
from drip_physics.config import (  # noqa: E402
    EnvironmentConfig,
    default_environment,
)
from drip_physics.core import ArrayGeometry, PhaseArray  # noqa: E402
from drip_physics.pathtrack import simulate_path_tracking  # noqa: E402
from drip_physics.pressure import (  # noqa: E402
    compute_pressure_grid,
    compute_pressure_slice,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Generation config (frozen dataclass; mirrors batch_generator.GenerationConfig)
# =============================================================================


@dataclass(frozen=True)
class InverseDataConfig:
    """Configuration for a single FNO surrogate dataset generation run.

    All physical units are SI. ``slice_extent_m`` and ``slice_z_m`` are in
    meters. The grid is symmetric about the origin in xy.

    ``uniform_z_mode`` controls per-sample slice z for the uniform branch:

    - ``"fixed"``: every uniform sample uses ``slice_z_m`` (legacy v1 behavior).
    - ``"random"``: each uniform sample's slice z is sampled uniformly from
      ``uniform_z_range_m`` (v2 default). This matches the z-range that the
      droplet visits during rollouts so uniform and rollout slices are drawn
      from overlapping z distributions, eliminating the slice-z ambiguity
      identified in the v1 audit.

    ``uniform_z_range_m`` is the inclusive ``[z_min, z_max]`` range (meters)
    used when ``uniform_z_mode == "random"``. The v3 default ``(0.0, 0.18)``
    matches the in-array region the droplet visits during rollouts when
    ``rollout_initial_z_m = 0.18`` m (just below the array top at z = 0.20 m),
    where the Im distribution is naturally symmetric. The v2 default
    ``(0.30, 0.40)`` placed the slice in the off-axis far-field above the
    array and produced a pathological Im asymmetry — see
    ``notebooks/02_rollout_diagnostics_SUMMARY.md``.

    ``rollout_initial_z_m`` overrides the droplet's initial z-position when
    seeding ``simulate_path_tracking``. The default ``0.18 m`` keeps the
    rollout slice plane inside the array (where the controller has full force
    authority and Im is symmetric) instead of above it. The styrofoam droplet
    factory ships an initial z of ``0.40 m`` for L1 hardware realism; that
    default is wrong for FNO training because the slice plane sits in the
    deep far-field where Im is artificially narrow.
    """

    n_trajectories: int = 500
    uniform_fraction: float = 0.8
    field_dims: int = 2
    grid_resolution: Tuple[int, ...] = (64, 64)
    slice_extent_m: float = 0.025
    slice_z_m: float = 0.10
    uniform_z_mode: str = "random"
    uniform_z_range_m: Tuple[float, float] = (0.0, 0.18)
    output_path: str = "ml_inverse/data/inverse_dataset.h5"
    seed: int = 42

    # Controller-rollout knobs.
    rollout_target_radius_m: float = 0.015
    rollout_max_duration_s: float = 0.5
    rollout_stride: int = 8  # subsample frames; 5001 -> ~625 at default dt.
    rollout_kp: float = 3.0
    rollout_kd: float = 1.0
    rollout_control_mode: str = "pid"
    rollout_z_extent_m: float = 0.030  # +/- around droplet z, only used in 3D.
    rollout_initial_z_m: float = 0.18  # v3: drop into array region.

    # Smart-regen knob: if set, copy rollout rows from this v1 HDF5 instead of
    # re-running fields. The rollout slice_z_m column is reconstructed by
    # re-running simulate_path_tracking with the same seed (no field compute).
    reuse_rollouts_from: Optional[str] = None

    def __post_init__(self) -> None:  # noqa: D401 - dataclass hook
        if self.field_dims not in (2, 3):
            raise ValueError(f"field_dims must be 2 or 3, got {self.field_dims}")
        if len(self.grid_resolution) != self.field_dims:
            raise ValueError(
                "grid_resolution must have length == field_dims "
                f"(field_dims={self.field_dims}, "
                f"grid_resolution={self.grid_resolution})"
            )
        if not 0.0 <= self.uniform_fraction <= 1.0:
            raise ValueError(
                f"uniform_fraction must be in [0, 1], got {self.uniform_fraction}"
            )
        if self.rollout_stride < 1:
            raise ValueError(f"rollout_stride must be >= 1, got {self.rollout_stride}")
        if self.uniform_z_mode not in ("fixed", "random"):
            raise ValueError(
                f"uniform_z_mode must be 'fixed' or 'random', "
                f"got {self.uniform_z_mode!r}"
            )
        z_lo, z_hi = self.uniform_z_range_m
        if not (z_lo < z_hi):
            raise ValueError(
                f"uniform_z_range_m must be (lo, hi) with lo < hi, "
                f"got {self.uniform_z_range_m}"
            )
        if not np.isfinite(self.rollout_initial_z_m):
            raise ValueError(
                f"rollout_initial_z_m must be finite, "
                f"got {self.rollout_initial_z_m}"
            )
        if self.rollout_initial_z_m <= 0.0:
            raise ValueError(
                f"rollout_initial_z_m must be > 0 (droplet must fall toward "
                f"z_floor=0), got {self.rollout_initial_z_m}"
            )


# =============================================================================
# Helpers
# =============================================================================


def _slice_bounds_2d(
    extent_m: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return ((-extent_m, extent_m), (-extent_m, extent_m))


def _grid_bounds_3d(
    extent_m: float, z_center: float, z_extent: float
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    return (
        (-extent_m, extent_m),
        (-extent_m, extent_m),
        (z_center - z_extent, z_center + z_extent),
    )


def _compute_field(
    *,
    array: ArrayGeometry,
    phases: PhaseArray,
    env: EnvironmentConfig,
    cfg: InverseDataConfig,
    droplet_z: float,
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """Compute a complex pressure field for one phase config.

    Returns
    -------
    field : np.ndarray
        Complex field with shape ``cfg.grid_resolution``.
    coords : tuple
        ``(x,)``, ``(x, y)`` or ``(x, y, z)`` 1D coord arrays. In 2D the
        slice is in the xy plane and ``coords == (x, y)``; in 3D
        ``coords == (x, y, z)``.
    """
    if cfg.field_dims == 2:
        bounds = _slice_bounds_2d(cfg.slice_extent_m)
        a1, a2, P = compute_pressure_slice(
            array,
            phases,
            slice_axis="z",
            slice_position=cfg.slice_z_m,
            bounds=bounds,
            resolution=cfg.grid_resolution,
            env=env,
        )
        # a1, a2 are 2D meshgrids; recover 1D axes.
        x_coords = a1[:, 0].astype(np.float32)
        y_coords = a2[0, :].astype(np.float32)
        return P, (x_coords, y_coords)

    # 3D path - centered on droplet_z so rollout fields are relevant.
    bounds = _grid_bounds_3d(cfg.slice_extent_m, droplet_z, cfg.rollout_z_extent_m)
    X, Y, Z, P = compute_pressure_grid(
        array,
        phases,
        bounds=bounds,
        resolution=cfg.grid_resolution,
        env=env,
    )
    x_coords = X[:, 0, 0].astype(np.float32)
    y_coords = Y[0, :, 0].astype(np.float32)
    z_coords = Z[0, 0, :].astype(np.float32)
    return P, (x_coords, y_coords, z_coords)


def _is_finite_field(field: np.ndarray) -> bool:
    return bool(np.isfinite(field.real).all() and np.isfinite(field.imag).all())


# =============================================================================
# Sampler
# =============================================================================


@dataclass
class _SampleBuffers:
    phases: List[np.ndarray] = field(default_factory=list)
    field_real: List[np.ndarray] = field(default_factory=list)
    field_imag: List[np.ndarray] = field(default_factory=list)
    trajectory_id: List[int] = field(default_factory=list)
    source: List[int] = field(default_factory=list)
    frame_idx: List[int] = field(default_factory=list)
    # Per-sample slice z (meters). For uniform samples this is the random or
    # fixed slice z; for rollout samples this is the droplet z at the harvested
    # frame. NaN means "unknown" (used for v1-imported rollouts when the seed
    # cannot be replayed).
    slice_z_m: List[float] = field(default_factory=list)


def _sample_uniform(
    *,
    cfg: InverseDataConfig,
    array: ArrayGeometry,
    env: EnvironmentConfig,
    rng: np.random.Generator,
    n_uniform: int,
    buffers: _SampleBuffers,
    skip_log: List[Tuple[int, str]],
    next_traj_id: int,
) -> int:
    """Append n_uniform uniform-random configs into buffers.

    Returns the next available trajectory_id.
    """
    n_t = array.num_transducers
    z_lo, z_hi = cfg.uniform_z_range_m
    for i in range(n_uniform):
        traj_id = next_traj_id + i
        phase_vec = rng.uniform(0.0, 2.0 * np.pi, size=n_t).astype(np.float64)

        # Determine per-sample slice z. CRITICAL: draw from the same rng
        # stream so the dataset is reproducible from --seed alone. We always
        # consume one rng draw per uniform sample (even in 'fixed' mode and
        # in 3D mode, where slice_z is unused for the field call) so changing
        # field_dims or uniform_z_mode does not desync downstream rollout
        # phase/target sampling against v1.
        z_draw = float(rng.uniform(z_lo, z_hi))
        if cfg.uniform_z_mode == "random":
            sample_slice_z = z_draw
        else:  # "fixed"
            sample_slice_z = float(cfg.slice_z_m)

        # In 2D mode, override the cfg's slice_z_m for this sample's field
        # computation. In 3D mode all uniform samples share a single fixed
        # 3D grid centered at cfg.slice_z_m (z is one of the grid axes — the
        # per-sample z draw above is irrelevant for uniform 3D, kept only for
        # rng-stream alignment with the 2D / rollout paths). This makes the
        # grid_coords/z stored in the HDF5 a faithful coordinate frame for
        # every sample, which the v2.1_3d analytical prior requires (see
        # research/3D_RETRY.md). The per-sample slice_z_m column records the
        # fixed grid center so downstream consumers don't have to reach into
        # the file attrs.
        local_cfg = cfg
        if cfg.field_dims == 2 and sample_slice_z != cfg.slice_z_m:
            local_cfg = InverseDataConfig(
                **{**cfg.__dict__, "slice_z_m": sample_slice_z}
            )
        if cfg.field_dims == 3:
            grid_center_z = float(cfg.slice_z_m)
        else:
            grid_center_z = sample_slice_z

        phases = PhaseArray(phase_vec.copy())
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                field, _coords = _compute_field(
                    array=array,
                    phases=phases,
                    env=env,
                    cfg=local_cfg,
                    droplet_z=grid_center_z,
                )
        except Exception as exc:  # pragma: no cover - defensive
            skip_log.append((traj_id, f"uniform exception: {exc!r}"))
            continue

        if not _is_finite_field(field):
            skip_log.append((traj_id, "uniform non-finite field"))
            continue

        # Record the field's actual z-grid center for downstream code. In 2D
        # this is the per-sample slice z; in 3D it's the fixed grid center
        # (so all uniform 3D rows share one value, matching the stored
        # grid_coords/z).
        recorded_slice_z = (
            float(cfg.slice_z_m) if cfg.field_dims == 3 else float(sample_slice_z)
        )

        buffers.phases.append(phases.phases.astype(np.float32))
        buffers.field_real.append(field.real.astype(np.float32))
        buffers.field_imag.append(field.imag.astype(np.float32))
        buffers.trajectory_id.append(int(traj_id))
        buffers.source.append(0)
        buffers.frame_idx.append(0)
        buffers.slice_z_m.append(recorded_slice_z)

    return next_traj_id + n_uniform


def _sample_rollouts(
    *,
    cfg: InverseDataConfig,
    array: ArrayGeometry,
    env: EnvironmentConfig,
    rng: np.random.Generator,
    n_rollouts: int,
    buffers: _SampleBuffers,
    skip_log: List[Tuple[int, str]],
    next_traj_id: int,
) -> int:
    """Run controller rollouts and harvest phase frames into buffers."""
    droplet = _make_droplet("styrofoam", None)
    # v3: override default initial_position (z=0.40m, in far-field above array)
    # with cfg.rollout_initial_z_m (default z=0.18m, inside array region) so
    # the rollout slice plane is where the controller has force authority and
    # the Im distribution is naturally symmetric. Build a fresh array (not
    # mutated) so the frozen droplet config remains immutable.
    initial_position = np.array([0.0, 0.0, float(cfg.rollout_initial_z_m)])

    for r in range(n_rollouts):
        traj_id = next_traj_id + r
        # Sample target uniformly in a box [-R, +R]^2 (the spec says square,
        # not disk). Using a box yields the corners and matches the prompt.
        tx = float(
            rng.uniform(-cfg.rollout_target_radius_m, cfg.rollout_target_radius_m)
        )
        ty = float(
            rng.uniform(-cfg.rollout_target_radius_m, cfg.rollout_target_radius_m)
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                traj = simulate_path_tracking(
                    array=array,
                    target=(tx, ty),
                    initial_position=initial_position.copy(),
                    env_config=env,
                    droplet_config=droplet,
                    control_mode=cfg.rollout_control_mode,
                    kp=cfg.rollout_kp,
                    kd=cfg.rollout_kd,
                    max_duration=cfg.rollout_max_duration_s,
                    z_floor=0.0,
                    store_phases=True,
                )
        except Exception as exc:  # pragma: no cover - defensive
            skip_log.append((traj_id, f"rollout exception: {exc!r}"))
            continue

        # Subsample by stride; harvest each step's phases + field at droplet z.
        kept_local = 0
        for k, point in enumerate(traj.points):
            if k % cfg.rollout_stride != 0:
                continue
            if point.phases is None:
                continue
            phase_vec = np.asarray(point.phases.phases, dtype=np.float64)
            if not np.all(np.isfinite(phase_vec)):
                skip_log.append(
                    (traj_id, f"rollout traj_id={traj_id} frame={k} non-finite phases")
                )
                continue
            # Field z-plane: in 2D mode follow the droplet down. In 3D mode
            # the grid is centered on the droplet z.
            droplet_z = float(point.position[2])
            local_cfg = cfg
            if cfg.field_dims == 2:
                local_cfg = InverseDataConfig(
                    **{**cfg.__dict__, "slice_z_m": droplet_z}
                )
            phases = PhaseArray(phase_vec.copy())
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    field, _coords = _compute_field(
                        array=array,
                        phases=phases,
                        env=env,
                        cfg=local_cfg,
                        droplet_z=droplet_z,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                skip_log.append(
                    (traj_id, f"rollout traj_id={traj_id} frame={k} exception: {exc!r}")
                )
                continue
            if not _is_finite_field(field):
                skip_log.append(
                    (traj_id, f"rollout traj_id={traj_id} frame={k} non-finite field")
                )
                continue

            buffers.phases.append(phase_vec.astype(np.float32))
            buffers.field_real.append(field.real.astype(np.float32))
            buffers.field_imag.append(field.imag.astype(np.float32))
            buffers.trajectory_id.append(int(traj_id))
            buffers.source.append(1)
            buffers.frame_idx.append(int(kept_local))
            buffers.slice_z_m.append(float(droplet_z))
            kept_local += 1

        if kept_local == 0:
            skip_log.append((traj_id, "rollout produced 0 valid frames"))

    return next_traj_id + n_rollouts


# =============================================================================
# HDF5 writer
# =============================================================================


def _write_hdf5(
    *,
    cfg: InverseDataConfig,
    array: ArrayGeometry,
    env: EnvironmentConfig,
    buffers: _SampleBuffers,
    skip_log: List[Tuple[int, str]],
    coord_axes: Tuple[np.ndarray, ...],
) -> None:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py required for dataset generation. Install: pip install h5py"
        ) from exc

    if len(buffers.phases) == 0:
        raise RuntimeError(
            "No samples generated - everything was skipped. Check skip_log."
        )

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(buffers.phases)
    n_t = int(buffers.phases[0].shape[0])
    grid_shape = tuple(buffers.field_real[0].shape)
    n_unique_traj = int(np.unique(np.array(buffers.trajectory_id)).size)

    phases_arr = np.stack(buffers.phases, axis=0)
    field_real_arr = np.stack(buffers.field_real, axis=0)
    field_imag_arr = np.stack(buffers.field_imag, axis=0)
    traj_arr = np.asarray(buffers.trajectory_id, dtype=np.int64)
    source_arr = np.asarray(buffers.source, dtype=np.uint8)
    frame_idx_arr = np.asarray(buffers.frame_idx, dtype=np.int64)
    # Per-sample slice z. NaN-fill to length n if the buffer wasn't populated
    # (defensive — every code path should append an entry; this guards against
    # drift if a future sampler forgets).
    if len(buffers.slice_z_m) == n:
        slice_z_arr = np.asarray(buffers.slice_z_m, dtype=np.float32)
    else:
        logger.warning(
            "slice_z_m buffer length %d != n_samples %d; padding with NaN. "
            "This indicates a sampler bug — please file an issue.",
            len(buffers.slice_z_m),
            n,
        )
        slice_z_arr = np.full(n, np.nan, dtype=np.float32)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("phases", data=phases_arr)
        f.create_dataset("field_real", data=field_real_arr)
        f.create_dataset("field_imag", data=field_imag_arr)
        f.create_dataset("trajectory_id", data=traj_arr)
        f.create_dataset("source", data=source_arr)
        f.create_dataset("frame_idx", data=frame_idx_arr)
        f.create_dataset("slice_z_m", data=slice_z_arr)
        f.create_dataset(
            "transducer_positions",
            data=array.positions.astype(np.float32),
        )

        coords_grp = f.create_group("grid_coords")
        coords_grp.create_dataset("x", data=coord_axes[0])
        coords_grp.create_dataset("y", data=coord_axes[1])
        if cfg.field_dims == 3:
            coords_grp.create_dataset("z", data=coord_axes[2])

        # attrs.
        f.attrs["field_dims"] = int(cfg.field_dims)
        f.attrs["grid_resolution"] = np.asarray(grid_shape, dtype=np.int64)
        if cfg.field_dims == 2:
            f.attrs["slice_extent_m"] = np.asarray(
                [
                    [-cfg.slice_extent_m, cfg.slice_extent_m],
                    [-cfg.slice_extent_m, cfg.slice_extent_m],
                ],
                dtype=np.float64,
            )
            # In 'fixed' uniform-z mode this is the single uniform-branch
            # slice z. In 'random' mode it's the legacy default kept around
            # for reproducible single-field debugging — the per-sample slice
            # z lives in the slice_z_m DATASET, not this attr.
            f.attrs["slice_z_m"] = float(cfg.slice_z_m)
        else:
            f.attrs["slice_extent_m"] = np.asarray(
                [
                    [-cfg.slice_extent_m, cfg.slice_extent_m],
                    [-cfg.slice_extent_m, cfg.slice_extent_m],
                    [-cfg.rollout_z_extent_m, cfg.rollout_z_extent_m],
                ],
                dtype=np.float64,
            )
            f.attrs["slice_z_m"] = float("nan")
        f.attrs["frequency_hz"] = float(env.transducer.frequency)
        f.attrs["sound_speed_m_s"] = float(env.speed_of_sound)
        f.attrs["n_transducers"] = int(n_t)
        f.attrs["n_trajectories"] = int(n_unique_traj)
        f.attrs["seed"] = int(cfg.seed)
        f.attrs["n_skipped_nonfinite"] = int(len(skip_log))
        f.attrs["drip_physics_version"] = str(_DRIP_PHYSICS_VERSION)
        f.attrs["uniform_z_mode"] = str(cfg.uniform_z_mode)
        f.attrs["uniform_z_range_m"] = np.asarray(
            cfg.uniform_z_range_m, dtype=np.float64
        )
        f.attrs["rollout_initial_z_m"] = float(cfg.rollout_initial_z_m)

    logger.info("Wrote %s (%d samples, %d trajectories)", output_path, n, n_unique_traj)


# =============================================================================
# Driver
# =============================================================================


def _reuse_rollouts(
    *,
    cfg: InverseDataConfig,
    array: ArrayGeometry,
    env: EnvironmentConfig,
    rng: np.random.Generator,
    n_rollouts: int,
    buffers: _SampleBuffers,
    skip_log: List[Tuple[int, str]],
    next_traj_id: int,
) -> int:
    """Copy rollout rows from an existing v1 HDF5, reconstructing slice_z_m.

    The expensive part of a rollout sample is the per-frame field computation
    (~30 ms x ~625 frames x 100 rollouts ~ 30 min). The cheap part is the
    rollout itself (~2 s without store_phases). This function:

    1. Opens ``cfg.reuse_rollouts_from`` read-only and copies all source==1
       rows verbatim (phases, field_real, field_imag, frame_idx).
    2. Re-runs ``simulate_path_tracking`` for each rollout with the same seed
       (no field compute) to recover the per-frame droplet z, which becomes
       the per-sample slice_z_m.
    3. Burns ``rng`` for the same number of (tx, ty) draws so the downstream
       random stream stays aligned with a from-scratch v2 run for any
       additional sampling steps.

    The trajectory_id values in the new file start from ``next_traj_id`` and
    are renumbered, NOT taken from the source file. This keeps trajectory ids
    contiguous in v2 even though we're stitching from two sources.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py required to reuse rollouts. Install: pip install h5py"
        ) from exc

    src_path = cfg.reuse_rollouts_from
    if src_path is None:
        return next_traj_id

    droplet = _make_droplet("styrofoam", None)
    # Use cfg.rollout_initial_z_m for replay consistency. If reusing v1 fields,
    # the source file was generated with the styrofoam default z=0.40m; in
    # that case callers should pass --rollout-initial-z-mm 400 to match the
    # source rollouts, otherwise replay z values won't line up with the
    # source-file frame z. For v3 the recommended path is fresh rollouts
    # (no --reuse-rollouts-from), since v1's z=0.40m start is the bug v3
    # exists to fix.
    initial_position = np.array([0.0, 0.0, float(cfg.rollout_initial_z_m)])

    with h5py.File(src_path, "r") as fsrc:
        src_source = fsrc["source"][...]
        src_traj = fsrc["trajectory_id"][...]
        src_frame_idx = fsrc["frame_idx"][...]
        rollout_mask = src_source == 1
        rollout_rows = np.flatnonzero(rollout_mask)

        # Ordered unique rollout trajectory ids in the source file.
        src_rollout_traj_ids = np.unique(src_traj[rollout_mask])
        n_src_rollouts = int(src_rollout_traj_ids.size)

        if n_rollouts != n_src_rollouts:
            logger.warning(
                "reuse_rollouts: cfg.n_trajectories implies %d rollouts but "
                "source file has %d. Using source's count.",
                n_rollouts,
                n_src_rollouts,
            )

        # Step 1 — replay the rollouts (no fields) to get per-frame z. The
        # source file used cfg.seed and burned 400 uniform-phase draws first.
        # We must reproduce that burn here. NOTE: this RNG burn matches v1
        # uniform-branch behavior even though v2 will overwrite the uniform
        # rows; the rollout (tx, ty) sequence in v1 was determined by what
        # the v1 sampler consumed, not by what v2 would consume. We replay
        # the v1 sequence faithfully here.
        replay_rng = np.random.default_rng(cfg.seed)
        # Burn 400 uniform phase draws (v1 uniform_fraction=0.8 * 500).
        n_v1_uniform = int(round(0.8 * 500))
        for _ in range(n_v1_uniform):
            replay_rng.uniform(0.0, 2.0 * np.pi, size=array.num_transducers)

        # Map from src trajectory_id -> per-frame z (in stride order, so the
        # k-th row of frame_idx corresponds to the k-th harvested frame).
        traj_z_map: dict[int, list[float]] = {}
        t_replay0 = time.time()
        for r, src_traj_id in enumerate(src_rollout_traj_ids):
            tx = float(
                replay_rng.uniform(
                    -cfg.rollout_target_radius_m, cfg.rollout_target_radius_m
                )
            )
            ty = float(
                replay_rng.uniform(
                    -cfg.rollout_target_radius_m, cfg.rollout_target_radius_m
                )
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                traj = simulate_path_tracking(
                    array=array,
                    target=(tx, ty),
                    initial_position=initial_position.copy(),
                    env_config=env,
                    droplet_config=droplet,
                    control_mode=cfg.rollout_control_mode,
                    kp=cfg.rollout_kp,
                    kd=cfg.rollout_kd,
                    max_duration=cfg.rollout_max_duration_s,
                    z_floor=0.0,
                    store_phases=False,
                )
            zs = [
                float(p.position[2])
                for k, p in enumerate(traj.points)
                if k % cfg.rollout_stride == 0
            ]
            traj_z_map[int(src_traj_id)] = zs
        logger.info(
            "reuse_rollouts: replayed %d rollouts (no fields) in %.1fs",
            n_src_rollouts,
            time.time() - t_replay0,
        )

        # Burn the parent rng with matching draws so caller's rng stream is
        # consistent with a from-scratch run that would have done the same
        # uniform target sampling here. (We treat 'reuse' as if the rollout
        # phase consumed n_src_rollouts pairs from rng.)
        for _ in range(n_src_rollouts):
            rng.uniform(-cfg.rollout_target_radius_m, cfg.rollout_target_radius_m)
            rng.uniform(-cfg.rollout_target_radius_m, cfg.rollout_target_radius_m)

        # Step 2 — copy rollout rows to buffers, attaching reconstructed z.
        # Rollout rows in v1 are stored in a contiguous block (uniform first,
        # then rollouts). We iterate per-trajectory so we can index into
        # traj_z_map[src_traj_id][frame_idx] safely.
        n_copied = 0
        for r, src_traj_id in enumerate(src_rollout_traj_ids):
            new_traj_id = next_traj_id + r
            traj_rows = np.flatnonzero(src_traj == int(src_traj_id))
            # Filter to rollout rows only (in case any uniform sample
            # happened to share an id — defensive; v1 guarantees disjoint).
            traj_rows = traj_rows[src_source[traj_rows] == 1]
            zs_for_traj = traj_z_map.get(int(src_traj_id), [])
            for row in traj_rows:
                fidx = int(src_frame_idx[row])
                if 0 <= fidx < len(zs_for_traj):
                    z_for_row = float(zs_for_traj[fidx])
                else:
                    z_for_row = float("nan")
                    skip_log.append(
                        (
                            new_traj_id,
                            f"reuse_rollouts traj_id={src_traj_id} frame={fidx} "
                            "missing z; replay length mismatch",
                        )
                    )
                # Lazy single-row reads keep peak memory low (~2 GB file).
                phase_vec = np.asarray(fsrc["phases"][row], dtype=np.float32)
                fr = np.asarray(fsrc["field_real"][row], dtype=np.float32)
                fi = np.asarray(fsrc["field_imag"][row], dtype=np.float32)
                buffers.phases.append(phase_vec)
                buffers.field_real.append(fr)
                buffers.field_imag.append(fi)
                buffers.trajectory_id.append(int(new_traj_id))
                buffers.source.append(1)
                buffers.frame_idx.append(fidx)
                buffers.slice_z_m.append(z_for_row)
                n_copied += 1

        logger.info(
            "reuse_rollouts: copied %d rollout rows (%d trajectories) from %s",
            n_copied,
            n_src_rollouts,
            src_path,
        )

    return next_traj_id + n_src_rollouts


def generate(cfg: InverseDataConfig) -> str:
    """Run the full generation pipeline. Returns the HDF5 path."""
    rng = np.random.default_rng(cfg.seed)

    env = default_environment()
    array = _make_array()

    n_uniform = int(round(cfg.uniform_fraction * cfg.n_trajectories))
    n_rollouts = cfg.n_trajectories - n_uniform

    logger.info(
        "Generating dataset: %d trajectories (%d uniform + %d rollouts), "
        "field_dims=%d, grid=%s, uniform_z_mode=%s, uniform_z_range_m=%s, "
        "rollout_initial_z_m=%.4f",
        cfg.n_trajectories,
        n_uniform,
        n_rollouts,
        cfg.field_dims,
        cfg.grid_resolution,
        cfg.uniform_z_mode,
        cfg.uniform_z_range_m,
        cfg.rollout_initial_z_m,
    )

    buffers = _SampleBuffers()
    skip_log: List[Tuple[int, str]] = []

    t0 = time.time()
    next_traj_id = 0
    next_traj_id = _sample_uniform(
        cfg=cfg,
        array=array,
        env=env,
        rng=rng,
        n_uniform=n_uniform,
        buffers=buffers,
        skip_log=skip_log,
        next_traj_id=next_traj_id,
    )
    t_uniform = time.time() - t0
    logger.info(
        "Uniform sampling done: %d samples in %.1fs",
        len([s for s in buffers.source if s == 0]),
        t_uniform,
    )

    t1 = time.time()
    if cfg.reuse_rollouts_from is not None:
        next_traj_id = _reuse_rollouts(
            cfg=cfg,
            array=array,
            env=env,
            rng=rng,
            n_rollouts=n_rollouts,
            buffers=buffers,
            skip_log=skip_log,
            next_traj_id=next_traj_id,
        )
    else:
        next_traj_id = _sample_rollouts(
            cfg=cfg,
            array=array,
            env=env,
            rng=rng,
            n_rollouts=n_rollouts,
            buffers=buffers,
            skip_log=skip_log,
            next_traj_id=next_traj_id,
        )
    t_rollout = time.time() - t1
    logger.info(
        "Rollout sampling done: %d samples in %.1fs",
        len([s for s in buffers.source if s == 1]),
        t_rollout,
    )

    # Recover the coordinate axes used by the grid (compute one stub field).
    stub_phases = PhaseArray(np.zeros(array.num_transducers))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, coord_axes = _compute_field(
            array=array,
            phases=stub_phases,
            env=env,
            cfg=cfg,
            droplet_z=cfg.slice_z_m,
        )

    _write_hdf5(
        cfg=cfg,
        array=array,
        env=env,
        buffers=buffers,
        skip_log=skip_log,
        coord_axes=coord_axes,
    )

    elapsed = time.time() - t0
    if skip_log:
        logger.warning(
            "Skipped %d non-finite/exception configs. First 5 reasons:",
            len(skip_log),
        )
        for tid, msg in skip_log[:5]:
            logger.warning("  trajectory_id=%s: %s", tid, msg)

    logger.info(
        "TOTAL runtime: %.1fs (uniform=%.1fs, rollout=%.1fs)",
        elapsed,
        t_uniform,
        t_rollout,
    )
    print(
        f"[generate_dataset] runtime_s={elapsed:.1f} "
        f"uniform_samples={sum(1 for s in buffers.source if s == 0)} "
        f"rollout_samples={sum(1 for s in buffers.source if s == 1)} "
        f"skipped={len(skip_log)} "
        f"output={cfg.output_path}"
    )
    return cfg.output_path


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-trajectories", type=int, default=500)
    p.add_argument("--uniform-fraction", type=float, default=0.8)
    p.add_argument("--field-dims", type=int, default=2, choices=(2, 3))
    p.add_argument("--grid-resolution", type=int, nargs="+", default=[64, 64])
    p.add_argument(
        "--slice-extent-mm",
        type=float,
        default=25.0,
        help="Half-extent in millimeters (grid is +/-extent in xy).",
    )
    p.add_argument(
        "--slice-z-mm",
        type=float,
        default=100.0,
        help="Slice z position in mm (only used in 2D).",
    )
    p.add_argument(
        "--output-path",
        type=str,
        default="ml_inverse/data/inverse_dataset.h5",
        help="Path for the HDF5 output. Relative paths resolve against cwd; "
        "run the script from `simulations/` so the default lands in "
        "`simulations/ml_inverse/data/`.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rollout-stride", type=int, default=8)
    p.add_argument(
        "--uniform-z-mode",
        type=str,
        default="random",
        choices=("random", "fixed"),
        help="'random' (default) draws per-sample slice z uniformly from "
        "--uniform-z-range-mm; 'fixed' uses --slice-z-mm for every uniform "
        "sample (legacy v1 behavior).",
    )
    p.add_argument(
        "--uniform-z-range-mm",
        type=float,
        nargs=2,
        default=[0.0, 180.0],
        metavar=("Z_MIN", "Z_MAX"),
        help="Inclusive z-range in mm for --uniform-z-mode=random. v3 default "
        "(0, 180) covers the in-array region the droplet visits when "
        "--rollout-initial-z-mm=180. v1/v2 used (300, 400) which placed "
        "the slice in the off-axis far-field above the array — see "
        "notebooks/02_rollout_diagnostics_SUMMARY.md.",
    )
    p.add_argument(
        "--rollout-initial-z-mm",
        type=float,
        default=180.0,
        help="Initial z-position (mm) for the rollout droplet. v3 default "
        "180 mm puts the droplet just below the array top (z=200 mm), so the "
        "rollout slice plane stays inside the array region where the "
        "controller has force authority and the Im distribution is "
        "symmetric. v1/v2 used the styrofoam default 400 mm (above the "
        "array, in the far-field where Im is artificially narrow).",
    )
    p.add_argument(
        "--reuse-rollouts-from",
        type=str,
        default=None,
        help="Path to an existing v1 HDF5 whose rollout rows should be copied "
        "into the new file (saves ~30 min field-recompute). If set, the "
        "rollout sampling stage replays trajectories with the same seed to "
        "reconstruct per-frame slice_z_m without recomputing fields.",
    )
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if len(args.grid_resolution) != args.field_dims:
        # Be permissive: a single-int grid_resolution gets broadcast.
        if len(args.grid_resolution) == 1:
            grid_res = tuple([args.grid_resolution[0]] * args.field_dims)
        else:
            raise ValueError(
                "--grid-resolution length must equal --field-dims "
                "(or be a single int)"
            )
    else:
        grid_res = tuple(args.grid_resolution)

    z_lo_m = float(args.uniform_z_range_mm[0]) * 1e-3
    z_hi_m = float(args.uniform_z_range_mm[1]) * 1e-3

    cfg = InverseDataConfig(
        n_trajectories=args.n_trajectories,
        uniform_fraction=args.uniform_fraction,
        field_dims=args.field_dims,
        grid_resolution=grid_res,
        slice_extent_m=args.slice_extent_mm * 1e-3,
        slice_z_m=args.slice_z_mm * 1e-3,
        uniform_z_mode=args.uniform_z_mode,
        uniform_z_range_m=(z_lo_m, z_hi_m),
        output_path=args.output_path,
        seed=args.seed,
        rollout_stride=args.rollout_stride,
        reuse_rollouts_from=args.reuse_rollouts_from,
        rollout_initial_z_m=float(args.rollout_initial_z_mm) * 1e-3,
    )

    # Resolve relative output_path against cwd. Callers should run from the
    # ``simulations`` directory so the default ``ml_inverse/data/...`` lands
    # in the right place; absolute paths are honored as-is.
    if not os.path.isabs(cfg.output_path):
        cfg = InverseDataConfig(
            **{
                **cfg.__dict__,
                "output_path": str(Path(cfg.output_path).resolve()),
            },
        )

    generate(cfg)


if __name__ == "__main__":
    main()
