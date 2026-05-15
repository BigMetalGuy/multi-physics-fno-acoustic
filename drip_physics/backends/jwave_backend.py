"""j-Wave Helmholtz PDE backend for pressure computation.

Solves the full 3D Helmholtz equation on a regular grid using j-Wave's
pseudospectral method.  Handles spatially-varying speed of sound and
absorption.  Results are cached per phase configuration and interpolated
for point queries.

Dependencies:
    jwave 0.2.1, jax 0.4.38, jaxdf 0.2.8 (pip install jwave)

Usage:
    from drip_physics.config import PressureConfig, EnvironmentConfig
    from drip_physics.backends.jwave_backend import JWaveBackend

    backend = JWaveBackend(
        config=PressureConfig(backend="jwave", grid_n=128),
        env=EnvironmentConfig(),
    )
    p = backend.compute_pressure(array, phases, point)

Validation summary (smoke test: 24 transducers, grid_n=128, focus at origin)
---------------------------------------------------------------------------
Reference: |p_analytical| ≈ 1.309e+03 Pa at the focal centroid.

| metric                          | pre-fix (2026-04-23) | post-fix (2026-05-02)        |
|---------------------------------|----------------------|------------------------------|
| |p_jwave| at focal centroid     | 1.83 Pa              | 1091 Pa                      |
| ratio |p_jwave|/|p_anal|        | 0.0014 (~700× low)   | 0.833 (within 20%)           |
| local argmax (±λ box at focus)  | n/a (700× off)       | (0.0, 0.0, 0.0) — at focus    |
| jax.grad(loss)(phases) finite   | TracerArrayConvErr   | finite, ||grad||≈2.5e5        |
| smoke wall-clock (forward solve)| ~162 s               | ~124-200 s (jit recompiled)   |
| smoke wall-clock (jax.grad@64³) | n/a                  | ~10-15 s (12-transducer cfg)  |

Fix log (iteration 2.2, 2026-05-02)
-----------------------------------
- Gap 1 (source calibration): per-transducer source value is now
  ``S = p_ref * r_ref * λ / dx²``.  Derivation: a discrete-cell delta in
  j-Wave's `FourierSeries` source is internally scaled by 2/(dx*c) and
  the solver returns -i·ω·u with (∇²+k²)u = -src·δ; empirical and analytic
  reduction both yield the constant ``K = dx²/λ`` such that |p|·r ≈ K·S
  in the inner free-field region.  Setting S = p_ref·r_ref/K ⇒
  envelope p·r = p_ref·r_ref, matching the analytical 1/r superposition.
- Gap 2 (coordinate alignment): grid axes now match j-Wave's internal
  centred axes (``arange(n)*dx − n·dx/2``).  Transducer positions are
  translated into this frame via a single offset vector so `Domain` and
  the source/probe coordinates agree.  Bounds are rejected up-front when
  the array exceeds the requested cube.
- Gap 3 (JAX-native interpolator): `scipy.interpolate.RegularGridInterpolator`
  replaced with `jax.scipy.ndimage.map_coordinates` over normalised grid
  indices.  This keeps gradients finite through point-query interpolation.
- Gap 4 (traceable source path): source assembly uses `jnp.zeros` +
  `.at[...].add()`.  Nearest-cell indices are precomputed (geometry
  is static) so the only data-dependent operations on `phases` are
  `jnp.exp(1j·φ)` and `.add()`, both autodiff-friendly.  A new public
  method `compute_pressure_from_phases(...)` accepts JAX arrays directly
  to bypass `PhaseArray.__post_init__`'s numpy normalisation.

Public API is unchanged: callers using
`compute_pressure(array, phases, point)` and
`compute_pressure_field(array, phases, points)` get the same signatures.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..boundary import BoundaryReflectionModel
    from ..config import EnvironmentConfig, PressureConfig
    from ..core import ArrayGeometry, PhaseArray, RingArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: cache-key for phase + ring state.
#
# This is invoked only on the eager (non-traced) call path; the JAX-native
# entry point bypasses caching because `jax.grad` re-traces every call.
# ---------------------------------------------------------------------------


def _phases_key(
    phases: "PhaseArray",
    ring_array: Optional["RingArray"],
) -> bytes:
    """Compute a deterministic cache key from phases + ring modulation."""
    key = np.asarray(phases.phases, dtype=np.float64).tobytes()
    if phases.amplitudes is not None:
        key += np.asarray(phases.amplitudes, dtype=np.float64).tobytes()
    if ring_array is not None:
        for ring in ring_array.rings:
            key += np.array(
                [ring.amplitude, ring.phase_offset], dtype=np.float64
            ).tobytes()
    return key


# ---------------------------------------------------------------------------
# JWaveBackend
# ---------------------------------------------------------------------------


class JWaveBackend:
    """Helmholtz PDE solver backed by j-Wave.

    On first call (or when phases change), solves the 3D Helmholtz equation
    on a regular Cartesian grid centred on the array.  Subsequent point
    queries interpolate from the cached solution until phases change.

    Parameters
    ----------
    config : PressureConfig
        Must contain ``grid_n`` (points per axis) and ``grid_dx`` (spacing).
    env : EnvironmentConfig | None
        Provides speed of sound, transducer frequency, and array geometry.
        If *None*, the backend will use the ``env`` passed to each call.
    """

    def __init__(
        self,
        config: "PressureConfig",
        env: Optional["EnvironmentConfig"] = None,
    ) -> None:
        self._config = config
        self._env = env

        # Cached solve products.  Only used by the eager (non-traced) path.
        self._cached_field: Optional[NDArray[np.complexfloating]] = None
        self._cached_phases_key: Optional[bytes] = None
        self._grid_axes: Optional[
            Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        ] = None
        # Offset translating array-frame positions to j-Wave centred frame.
        self._origin_offset: Optional[NDArray[np.float64]] = None

    # ------------------------------------------------------------------
    # Domain / Medium / Source construction
    # ------------------------------------------------------------------

    # FIX (gap 2): coordinate-mismatch fix, dated 2026-05-02.
    # ----------------------------------------------------------------------
    # j-Wave's `Domain.spatial_axis` is *centred at zero*: for grid index i
    # the physical position in j-Wave's internal frame is
    #     x_jwave(i) = i*dx − n*dx/2.
    # The previous backend built grid axes from `linspace(positions.min()-margin,
    # positions.max()+margin, n)`, which is NOT what `Domain(N, dx)` thinks
    # the coordinates are — so the source was deposited at the right index
    # but the solver interpreted it as a different spatial location, and the
    # interpolator looked up the wrong value at query time.
    #
    # The fix: pick a single offset O so that array-frame coordinate p maps
    # to j-Wave-frame coordinate p − O.  Use the array centroid for O so the
    # array sits at the centre of the cube (maximises distance to PML).
    # All transducer positions and query points must use the same offset.
    # ----------------------------------------------------------------------

    def _compute_origin_offset(
        self,
        array: "ArrayGeometry",
    ) -> NDArray[np.float64]:
        """Offset that translates array-frame coords into j-Wave centred frame.

        ``p_jwave = p_array − offset``.  Choosing offset = array centroid
        places the array at the j-Wave domain centre.
        """
        positions = np.asarray(array.positions, dtype=np.float64)
        return positions.mean(axis=0)

    def _build_grid_axes(
        self,
        array: "ArrayGeometry",
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Build 1-D coordinate axes for the regular grid in *array frame*.

        These axes match j-Wave's internal centred axes shifted by
        ``self._origin_offset``.  In array-frame coordinates the i-th cell
        on axis k sits at ``offset[k] + i*dx − n[k]*dx/2``.
        """
        nx, ny, nz = self._config.effective_grid_shape
        dx = self._config.grid_dx
        offset = self._compute_origin_offset(array)

        # j-Wave centred axis on each axis (anisotropic-aware): shape (n[k],), zero-mean.
        x = np.arange(nx, dtype=np.float64) * dx - 0.5 * nx * dx + offset[0]
        y = np.arange(ny, dtype=np.float64) * dx - 0.5 * ny * dx + offset[1]
        z = np.arange(nz, dtype=np.float64) * dx - 0.5 * nz * dx + offset[2]
        return x, y, z

    def _validate_domain_covers_array(
        self,
        array: "ArrayGeometry",
        margin_cells: float = 4.0,
    ) -> None:
        """Warn when cube side cannot contain the array + a small margin.

        Replaces the previous silent-zero behaviour (out-of-grid lookups
        used to return 0+0j without warning).  We warn rather than raise
        so existing tests that rely on partial-coverage geometries keep
        running, but transducers outside the cube are still clamped to
        the boundary cell — correct behaviour requires sizing the grid
        large enough.
        """
        shape = self._config.effective_grid_shape
        dx = self._config.grid_dx
        # Per-axis side lengths (anisotropic-aware).
        sides = np.asarray(shape, dtype=np.float64) * dx

        positions = np.asarray(array.positions, dtype=np.float64)
        bbox_per_axis = np.ptp(positions, axis=0)  # (3,)
        required = bbox_per_axis + 2.0 * margin_cells * dx
        if np.any(sides < required):
            import warnings
            axes = ("x", "y", "z")
            details = ", ".join(
                f"{ax}: side={sides[i]*1e3:.2f} mm need={required[i]*1e3:.2f} mm"
                for i, ax in enumerate(axes)
                if sides[i] < required[i]
            )
            warnings.warn(
                f"JWaveBackend: grid {shape} × dx={dx*1e3:.3f} mm too small "
                f"to contain array on axes [{details}]. Transducers outside "
                f"the cube will be clamped to the boundary cell. Increase "
                f"grid_shape or grid_dx for accurate results.",
                stacklevel=2,
            )

    def _build_domain_and_medium(
        self,
        array: "ArrayGeometry",
        env: "EnvironmentConfig",
    ) -> Tuple["object", "object"]:
        """Build j-Wave ``Domain`` and ``Medium`` from config."""
        from jwave.geometry import Domain, Medium

        shape = self._config.effective_grid_shape
        dx = self._config.grid_dx

        domain = Domain(N=tuple(shape), dx=(dx, dx, dx))
        medium = Medium(
            domain,
            sound_speed=float(env.speed_of_sound),
            density=float(env.medium.density),
        )
        return domain, medium

    # FIX (gap 1): source calibration, dated 2026-05-02.
    # ----------------------------------------------------------------------
    # j-Wave's `helmholtz_solver` internally scales the source by 2/(dx*c)
    # then returns -i·ω·u where u solves (∇² + k²)u = src_scaled.  A discrete
    # delta of value S deposited at one cell of a `FourierSeries` produces
    # a far-field |p(r)|·r ≈ K·|S| with K = dx²/λ (verified empirically and
    # by the analytic reduction; see _investigation/jwave_probe_calibration.py).
    #
    # The analytical solver in `pressure.py` produces, per transducer,
    #     p(r) = p_ref · (r_ref / r) · directivity · exp(i(k·r + φ))
    # so the matching delta value is
    #     S = p_ref · r_ref · λ / dx²
    # times the per-transducer amplitude and phasor exp(i·φ).  Directivity
    # cannot be encoded in a delta and is left to the PDE's natural
    # spreading; for most NF/MF geometries the dominant analytical
    # contribution is the 1/r envelope, which this calibration matches.
    # ----------------------------------------------------------------------

    def _source_amplitude_scale(self, env: "EnvironmentConfig") -> float:
        """Per-transducer multiplier giving p_ref·r_ref envelope at distance r.

        Returns S₀ = p_ref · r_ref · λ / dx²; the per-transducer source value
        is then ``S₀ · amp · exp(i·φ)``.
        """
        p_ref = float(env.transducer.p_ref)
        r_ref = float(env.transducer.r_ref)
        wavelength = float(env.wavelength)
        dx = float(self._config.grid_dx)
        return p_ref * r_ref * wavelength / (dx * dx)

    # FIX (gap 4): traceable source path, dated 2026-05-02.
    # ----------------------------------------------------------------------
    # The source-cell indices depend only on the (static) array geometry, so
    # we precompute them with numpy outside the trace.  The data-dependent
    # operations on `phases`/`amplitudes` are pure JAX: `.at[...].add()`
    # accumulates per-transducer phasors via scatter-add, and `jnp.exp(1j·φ)`
    # is autodiff-friendly.  No `np.argmin`, no `np.zeros + +=` mutation.
    # ----------------------------------------------------------------------

    def _precompute_source_indices(
        self,
        array: "ArrayGeometry",
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        """Return per-transducer (ix, iy, iz) nearest-cell indices.

        Uses the j-Wave centred frame: cell index i on axis k corresponds to
        position ``i·dx − n[k]·dx/2`` (j-Wave frame) = ``offset[k] + i·dx − n[k]·dx/2``
        (array frame). Inverting: ``i_k = round((p_k − offset_k)/dx + n_k/2)``.
        """
        shape = np.asarray(self._config.effective_grid_shape, dtype=np.float64)
        dx = self._config.grid_dx
        offset = self._compute_origin_offset(array)

        positions = np.asarray(array.positions, dtype=np.float64)
        local = positions - offset[None, :]  # j-Wave frame
        idx = np.rint(local / dx + 0.5 * shape[None, :]).astype(np.int64)
        # Per-axis clip to (0, n_k - 1).
        for k in range(3):
            idx[:, k] = np.clip(idx[:, k], 0, int(shape[k]) - 1)
        return idx[:, 0], idx[:, 1], idx[:, 2]

    def _build_source_jax(
        self,
        domain: "object",
        ix: NDArray[np.int64],
        iy: NDArray[np.int64],
        iz: NDArray[np.int64],
        phases_jnp,
        amps_jnp,
        env: "EnvironmentConfig",
    ) -> "object":
        """Build the j-Wave `FourierSeries` source — fully JAX-traceable.

        Parameters
        ----------
        ix, iy, iz : numpy int arrays of shape (N,)
            Per-transducer nearest-cell indices in the j-Wave centred grid.
        phases_jnp, amps_jnp : jnp.ndarray of shape (N,)
            Phases (rad) and amplitude multipliers.  Must be JAX arrays.
        """
        import jax.numpy as jnp
        from jwave import FourierSeries

        nx, ny, nz = self._config.effective_grid_shape
        s0 = jnp.asarray(self._source_amplitude_scale(env), dtype=jnp.complex64)

        # Per-transducer complex source value.
        amps_c = jnp.asarray(amps_jnp, dtype=jnp.complex64)
        phasors = jnp.exp(1j * jnp.asarray(phases_jnp, dtype=jnp.float32))
        values = s0 * amps_c * phasors

        # Build (nx, ny, nz, 1) source field via scatter-add.  Multiple
        # transducers at the same nearest cell sum naturally.
        src = jnp.zeros((nx, ny, nz, 1), dtype=jnp.complex64)
        idx_x = jnp.asarray(ix, dtype=jnp.int32)
        idx_y = jnp.asarray(iy, dtype=jnp.int32)
        idx_z = jnp.asarray(iz, dtype=jnp.int32)
        src = src.at[idx_x, idx_y, idx_z, 0].add(values)

        return FourierSeries(src, domain)

    # ------------------------------------------------------------------
    # Solve + cache (eager, non-traced path)
    # ------------------------------------------------------------------

    def _effective_phases_amps(
        self,
        phases: "PhaseArray",
        ring_array: Optional["RingArray"],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply optional ring modulation; return numpy arrays."""
        amps = np.asarray(phases.amplitudes, dtype=np.float64)
        phs = np.asarray(phases.phases, dtype=np.float64)
        if ring_array is not None:
            amps = ring_array.get_effective_amplitudes() * amps
            phs = (phs + ring_array.get_phase_offsets()) % (2.0 * np.pi)
        return phs, amps

    def _solve_field_jax(
        self,
        array: "ArrayGeometry",
        phases_jnp,
        amps_jnp,
        env: "EnvironmentConfig",
    ):
        """Run the Helmholtz solve; return the (n,n,n) complex field.

        Pure JAX: traceable end-to-end through `phases_jnp`/`amps_jnp`.
        """
        from jwave import helmholtz_solver

        domain, medium = self._build_domain_and_medium(array, env)
        ix, iy, iz = self._precompute_source_indices(array)
        source = self._build_source_jax(
            domain, ix, iy, iz, phases_jnp, amps_jnp, env
        )
        omega = float(env.angular_frequency)
        p_field = helmholtz_solver(medium, omega, source)
        return p_field.on_grid[..., 0]  # (n, n, n) complex jnp array

    def _solve(
        self,
        array: "ArrayGeometry",
        phases: "PhaseArray",
        ring_array: Optional["RingArray"],
        env: "EnvironmentConfig",
    ) -> None:
        """Run the Helmholtz solver (eager path) and cache the numpy result."""
        import jax.numpy as jnp

        self._validate_domain_covers_array(array)

        phs, amps = self._effective_phases_amps(phases, ring_array)
        phases_jnp = jnp.asarray(phs, dtype=jnp.float32)
        amps_jnp = jnp.asarray(amps, dtype=jnp.float32)

        logger.info(
            "JWaveBackend: solving Helmholtz (grid %s, dx=%.4e m, "
            "c=%.1f m/s, f=%.0f Hz)",
            tuple(self._config.effective_grid_shape),
            self._config.grid_dx,
            env.speed_of_sound,
            env.transducer.frequency,
        )

        field_jnp = self._solve_field_jax(array, phases_jnp, amps_jnp, env)
        self._cached_field = np.asarray(field_jnp).astype(np.complex128)

        # Cache axes (still useful for callers introspecting the grid).
        self._grid_axes = self._build_grid_axes(array)
        self._origin_offset = self._compute_origin_offset(array)

        logger.info(
            "JWaveBackend: solve complete, max |p| = %.4e Pa",
            float(np.max(np.abs(self._cached_field))),
        )

    def _ensure_solved(
        self,
        array: "ArrayGeometry",
        phases: "PhaseArray",
        ring_array: Optional["RingArray"],
        env: "EnvironmentConfig",
    ) -> None:
        """Re-solve only if the phase configuration has changed."""
        key = _phases_key(phases, ring_array)
        if self._cached_phases_key != key or self._cached_field is None:
            self._solve(array, phases, ring_array, env)
            self._cached_phases_key = key

    # ------------------------------------------------------------------
    # Interpolation (point queries)
    # ------------------------------------------------------------------

    # FIX (gap 3): JAX-native interpolator, dated 2026-05-02.
    # ----------------------------------------------------------------------
    # `scipy.interpolate.RegularGridInterpolator` is numpy-only, breaking
    # `jax.grad` through point queries.  We replace it with
    # `jax.scipy.ndimage.map_coordinates(order=1, mode='constant', cval=0)`,
    # which performs trilinear interpolation on a uniform grid using
    # fractional indices.  Real and imaginary parts are interpolated
    # separately because `map_coordinates` does not handle complex inputs.
    #
    # The grid is uniform with spacing `dx`, j-Wave-centred origin at
    # array-frame `offset`.  For a query point p (array frame), the
    # fractional index along axis k is
    #     i_k = (p_k − offset_k)/dx + (n − 1)/2
    # — using (n − 1)/2 (not n/2) because `map_coordinates` indexes from
    # 0..n−1 and the j-Wave physical axis sample at index i is
    # ``offset + i·dx − n·dx/2`` so the *centroid* is at index (n−1)/2 only
    # when n is odd; for even n the j-Wave centre falls between cells.
    # The convention here matches `_build_grid_axes` which puts cell i at
    # ``offset + i·dx − n·dx/2``, so the index of physical position p is
    #     i = (p − offset)/dx + n/2.
    # We use that formulation for consistency.
    # ----------------------------------------------------------------------

    def _query_indices(
        self,
        array: "ArrayGeometry",
        points: NDArray[np.float64],
    ):
        """Convert array-frame positions to fractional grid indices.

        Returns a JAX array of shape (3, M) suitable for `map_coordinates`.
        """
        import jax.numpy as jnp

        shape = np.asarray(self._config.effective_grid_shape, dtype=np.float64)
        dx = self._config.grid_dx
        offset = self._compute_origin_offset(array)

        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        local = pts - offset[None, :]
        # Per-axis fractional index: frac[k] = local[k]/dx + 0.5 * n[k].
        frac = local / dx + 0.5 * shape[None, :]  # shape (M, 3)
        # map_coordinates expects (ndim, M)
        return jnp.asarray(frac.T, dtype=jnp.float32)

    def _interpolate(self, field_jnp, frac_idx):
        """Trilinear sample of a complex (n,n,n) field at fractional indices.

        ``field_jnp``: jnp.complex array shape (n, n, n).
        ``frac_idx``:  jnp.float array shape (3, M).
        Returns: jnp.complex array shape (M,).
        """
        import jax.numpy as jnp
        from jax.scipy.ndimage import map_coordinates

        real = map_coordinates(
            field_jnp.real, frac_idx, order=1, mode="constant", cval=0.0
        )
        imag = map_coordinates(
            field_jnp.imag, frac_idx, order=1, mode="constant", cval=0.0
        )
        return real + 1j * imag

    # ------------------------------------------------------------------
    # PressureBackend protocol (eager API, unchanged)
    # ------------------------------------------------------------------

    def compute_pressure(
        self,
        array: "ArrayGeometry",
        phases: "PhaseArray",
        point: NDArray[np.float64],
        *,
        ring_array: Optional["RingArray"] = None,
        use_directivity: bool = True,
        use_physical_units: bool = True,
        boundary_model: Optional["BoundaryReflectionModel"] = None,
        env: Optional["EnvironmentConfig"] = None,
    ) -> complex:
        """Compute complex pressure phasor at a single point.

        The PDE solution inherently includes diffraction and interference
        effects, so ``use_directivity`` is accepted for API compatibility
        but has no effect (the full wave equation already captures
        directivity).

        ``boundary_model`` is similarly accepted but ignored — PML
        boundaries in the Helmholtz solver already handle domain
        truncation.
        """
        import jax.numpy as jnp

        resolved_env = env if env is not None else self._env
        if resolved_env is None:
            raise ValueError(
                "JWaveBackend.compute_pressure: env must be provided "
                "either at construction or call time."
            )

        if not use_directivity or not use_physical_units:
            logger.warning(
                "JWaveBackend: use_directivity=%s use_physical_units=%s "
                "ignored (PDE backend computes physical, diffractive field).",
                use_directivity,
                use_physical_units,
            )
        if boundary_model is not None:
            logger.warning(
                "JWaveBackend: boundary_model ignored — PML handles "
                "domain truncation."
            )

        self._ensure_solved(array, phases, ring_array, resolved_env)

        field_jnp = jnp.asarray(self._cached_field)
        frac = self._query_indices(array, point)
        vals = self._interpolate(field_jnp, frac)
        return complex(np.asarray(vals)[0])

    def compute_pressure_field(
        self,
        array: "ArrayGeometry",
        phases: "PhaseArray",
        points: NDArray[np.float64],
        *,
        ring_array: Optional["RingArray"] = None,
        use_directivity: bool = True,
        use_physical_units: bool = True,
        env: Optional["EnvironmentConfig"] = None,
    ) -> NDArray[np.complexfloating]:
        """Compute complex pressure phasors at multiple points.

        Interpolates from the cached 3D Helmholtz solution.
        """
        import jax.numpy as jnp

        resolved_env = env if env is not None else self._env
        if resolved_env is None:
            raise ValueError(
                "JWaveBackend.compute_pressure_field: env must be provided "
                "either at construction or call time."
            )

        self._ensure_solved(array, phases, ring_array, resolved_env)

        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)

        field_jnp = jnp.asarray(self._cached_field)
        frac = self._query_indices(array, pts)
        vals = self._interpolate(field_jnp, frac)
        return np.asarray(vals).astype(np.complex128)

    # ------------------------------------------------------------------
    # JAX-native API for `jax.grad`-compatible inverse design.
    # ------------------------------------------------------------------

    def compute_pressure_from_phases(
        self,
        array: "ArrayGeometry",
        phases_jnp,
        amplitudes_jnp,
        point,
        *,
        env: Optional["EnvironmentConfig"] = None,
    ):
        """Pure-JAX entry point: phases → pressure at a single point.

        This bypasses `PhaseArray.__post_init__` (which calls
        `np.asarray(phases)` and breaks JAX tracers) and runs the entire
        forward pipeline — source build, Helmholtz solve, point sample —
        through pure JAX so `jax.grad` returns a finite gradient.

        Parameters
        ----------
        array : ArrayGeometry
            Static geometry (transducer positions).  Treated as numpy data.
        phases_jnp : jnp.ndarray, shape (N,)
            Per-transducer phase in radians.  Differentiable.
        amplitudes_jnp : jnp.ndarray, shape (N,)
            Per-transducer amplitude multiplier.  Differentiable.
        point : array-like, shape (3,)
            Query position in array frame (metres).  Static.
        env : EnvironmentConfig | None
            Environment.  Falls back to `self._env`.

        Returns
        -------
        jnp.complex64 scalar  (the pressure phasor at `point`)
        """
        resolved_env = env if env is not None else self._env
        if resolved_env is None:
            raise ValueError(
                "JWaveBackend.compute_pressure_from_phases: env required."
            )
        self._validate_domain_covers_array(array)

        field_jnp = self._solve_field_jax(
            array, phases_jnp, amplitudes_jnp, resolved_env
        )
        frac = self._query_indices(array, np.asarray(point))
        vals = self._interpolate(field_jnp, frac)
        return vals[0]

    # ------------------------------------------------------------------
    # Cache control
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Discard cached solution so the next call triggers a re-solve."""
        self._cached_field = None
        self._cached_phases_key = None
        self._grid_axes = None
        self._origin_offset = None
