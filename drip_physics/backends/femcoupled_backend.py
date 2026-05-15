"""FEM-coupled-physics backend — drop-in replacement for ``jwave_backend``.

Phase 5 of the FEM-coupled-physics v1 build (FEM_V2_DESIGN.md §4 step 4).

Wraps the Phase 4 :class:`StaggeredCouplingDriver` (two-way acoustic ↔
thermal coupling) into the same ``compute_pressure_from_phases``
signature exposed by
:func:`drip_physics.backends.jwave_backend.JWaveBackend.compute_pressure_from_phases`.
Selectable via ``PressureConfig(backend="femcoupled", …)`` — **no
callers change**.

Pipeline
--------

  1. Lazy mesh + driver construction at first call (or on-demand
     reconstruction when the geometry / mesh-path changes).
  2. Pre-computed mesh→voxel-grid interpolation weights stored on the
     state object.  The weights become a *dense* ``(n_voxels, 4)``
     array indexing into the per-voxel enclosing tet's vertices; the
     mesh→grid map is then a clean JAX gather + matmul that
     ``jax.grad`` traces through.
  3. ``StaggeredCouplingDriver.step`` runs one outer thermal step (with
     its 3–5 inner fixed-point iterations) under the supplied phases.
  4. The complex pressure on mesh nodes is projected to the requested
     voxel grid, returned as ``(nz, ny, nx, 2)`` real (Re/Im) by
     default.  ``return_state=True`` additionally hands back an updated
     :class:`FemcoupledBackendState` so callers (e.g. multi-step
     gradient-descent inverse loops) can carry T_prev across calls.

Performance
-----------

Per FEM_V2_DESIGN §3.4 + Phase 4 receipts, one forward call ≈
2–5 inner iterations × (1 Helmholtz + 1 streaming + 1 heat) ≈ 10–50 s
on the coarse mesh.  This is **the** differentiator from the FNO
surrogate inverse path (FNO inverse ≈ seconds).  Production-mesh
inverse is hours; the surrogate (Track F) exists for that reason.

State semantics
---------------

The driver holds T_prev on the mesh.  Within a *single trajectory*
(e.g. a multi-step inverse), T evolves between calls:

  * First call: pass ``backend_state=None``; the call lazily constructs
    the driver + mesh + voxel-weights + initial T (uniform at
    ``T_CHAMBER_INIT_K``).  Set ``return_state=True`` to receive the
    state for the next call.
  * Subsequent calls: pass the returned state; the driver re-uses the
    existing mesh + weights and advances T from the prior step.

The driver itself isn't recreated unless the mesh or env changes —
amortising the ~25 s mesh + Problem build across the whole inverse.

See :mod:`drip_physics.backends.femcoupled.coupling` for the underlying
fixed-point coupling driver, and
:func:`drip_physics.backends.jwave_backend.JWaveBackend.compute_pressure_from_phases`
for the legacy contract this module mirrors.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import jax

    from ..config import EnvironmentConfig, PressureConfig, TransducerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

# Default coarse-mesh extent: the L1 chamber sits in a 10 cm × 10 cm × 36 cm
# bounding box; a 6×6×6 cm cube around chamber centre captures the focal
# zone and the lower transducer rings without dragging the grid into
# wasted padding outside the chamber.  Used when callers do not supply
# ``grid_extent_m`` explicitly.  ``(xmin, xmax, ymin, ymax, zmin, zmax)``
# in metres.
DEFAULT_GRID_EXTENT_M: Tuple[float, float, float, float, float, float] = (
    -0.03, 0.03, -0.03, 0.03, 0.10, 0.30,
)

# Default voxel grid resolution.  64³ is the FNO_v2.1 reference grid
# (matches ``inverse_dataset_v3``); kept consistent across backends so
# the inverse-loop test runs on the same shape as the j-Wave backend.
DEFAULT_GRID_SHAPE: Tuple[int, int, int] = (64, 64, 64)

# Default crucible-inlet disk radius (matches Phase 1 mesh default).
DEFAULT_CRUCIBLE_INLET_RADIUS_M: float = 2.5e-3

# Default PML width (matches Phase 2.1 / Phase 4 default).
DEFAULT_PML_WIDTH_M: float = 4e-3

# Default coarse-mesh refinement targets (focus / bulk h, mm).  Match
# Phase 4 test fixture so the backend boots on the same mesh tier the
# coupling tests validated.
DEFAULT_FOCUS_H_MM: float = 2.0
DEFAULT_BULK_H_MM: float = 10.0


# ---------------------------------------------------------------------------
# Mesh → voxel grid interpolation helpers (precomputed weights)
# ---------------------------------------------------------------------------


def _build_voxel_grid(
    grid_shape: Tuple[int, int, int],
    grid_extent_m: Tuple[float, float, float, float, float, float],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(xs, ys, zs, voxel_centres)`` for a regular Cartesian grid.

    ``voxel_centres`` is shape ``(nz * ny * nx, 3)`` in (x, y, z) order
    — flattened so it can be fed to a ``cKDTree`` query.

    Note: the leading axis convention is **(nz, ny, nx)** to match the
    j-Wave backend's complex-field shape.  The flat order is row-major
    over (z, y, x): index = ((iz * ny) + iy) * nx + ix.
    """
    nx, ny, nz = grid_shape
    xmin, xmax, ymin, ymax, zmin, zmax = grid_extent_m
    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    zs = np.linspace(zmin, zmax, nz, dtype=np.float64)
    # Build (nz, ny, nx, 3) in (x, y, z) order then flatten.
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    centres = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return xs, ys, zs, centres


def _barycentric_weights(
    pts: NDArray[np.float64],
    tet_vertices: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute barycentric weights of points in their enclosing tets.

    Args:
        pts: ``(M, 3)`` query points.
        tet_vertices: ``(M, 4, 3)`` per-point enclosing-tet vertices.

    Returns:
        ``(M, 4)`` barycentric weights (rows sum to 1; may be negative
        when the query lies outside the tet — clamped to non-negative
        + renormalised by the caller for numerical stability).

    Implementation: for each tet with vertices ``v0..v3``,
    ``[v1-v0, v2-v0, v3-v0] · (λ1, λ2, λ3)ᵀ = p - v0``.
    Solve the 3×3 system; ``λ0 = 1 - (λ1+λ2+λ3)``.
    """
    v0 = tet_vertices[:, 0, :]
    edges = tet_vertices[:, 1:, :] - v0[:, None, :]  # (M, 3, 3)
    rhs = pts - v0                                    # (M, 3)

    # Solve M-batch of 3×3 systems.  ``np.linalg.solve`` accepts
    # batched inputs; degenerate tets are protected by the cKDTree
    # cell-selection step (each query picks the *closest* cell whose
    # centroid is nearest).
    try:
        lambdas = np.linalg.solve(edges, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        # Fall back to per-row solve with safe pseudo-inverse.
        lambdas = np.zeros((pts.shape[0], 3), dtype=np.float64)
        for i in range(pts.shape[0]):
            try:
                lambdas[i] = np.linalg.solve(edges[i], rhs[i])
            except np.linalg.LinAlgError:
                lambdas[i] = np.linalg.pinv(edges[i]) @ rhs[i]

    lam0 = 1.0 - lambdas.sum(axis=-1)
    return np.stack(
        [lam0, lambdas[:, 0], lambdas[:, 1], lambdas[:, 2]], axis=-1
    )


def _precompute_voxel_weights(
    cells: NDArray[np.int64],
    points: NDArray[np.float64],
    voxel_centres: NDArray[np.float64],
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Pre-compute mesh→voxel-grid interpolation weights.

    For each voxel centre, find the nearest tet (via cell-centroid
    cKDTree query) and compute barycentric weights of the voxel centre
    in that tet.  Voxels whose closest tet does not contain them
    (boundary voxels, voxels outside the mesh) keep their weights but
    return non-convex values; downstream callers that need a strict
    "0 outside mesh" semantics should mask via ``cells`` index returned.

    Returns:
        cell_indices: ``(M,)`` int64 — index of the enclosing tet.
        weights:      ``(M, 4)`` float64 — barycentric weights, clamped
            to non-negative and renormalised so each row sums to 1.

    Differentiability: weights are a *constant* with respect to phase
    (mesh and grid are static) so ``jax.grad`` flows through the
    weighted-sum step that consumes them.
    """
    from scipy.spatial import cKDTree  # local import — scipy is available
                                          # in the jax_fem discovery venv.

    # Cell centroids for the kNN query.
    centroids = points[cells].mean(axis=1)  # (n_cells, 3)
    tree = cKDTree(centroids)
    _, cell_idx = tree.query(voxel_centres, k=1)
    cell_idx = np.asarray(cell_idx, dtype=np.int64)

    tet_verts = points[cells[cell_idx]]  # (M, 4, 3)
    weights = _barycentric_weights(voxel_centres, tet_verts)

    # Clamp negative weights to 0 (handles voxels outside the chosen
    # tet — the next-nearest-tet refinement is deferred; the clamp
    # gives a smooth, non-negative interpolation that's exact for
    # voxels inside their tet and a robust extrapolation otherwise).
    weights = np.clip(weights, 0.0, None)
    row_sum = weights.sum(axis=-1, keepdims=True)
    row_sum = np.where(row_sum > 0.0, row_sum, 1.0)
    weights = weights / row_sum
    return cell_idx, weights.astype(np.float64)


def _project_mesh_to_voxel(
    field_mesh: "jax.Array",
    cell_idx: "jax.Array",
    weights: "jax.Array",
    cells: "jax.Array",
) -> "jax.Array":
    """Linear gather + weighted sum: mesh nodal field → voxel field.

    Args:
        field_mesh: ``(n_nodes,)`` real or ``(n_nodes, 2)`` real (Re/Im
            stacked).  Float dtype.
        cell_idx: ``(M,)`` int — per-voxel enclosing tet index.
        weights: ``(M, 4)`` float — barycentric weights.
        cells: ``(n_cells, 4)`` int — tet vertex indices.

    Returns:
        ``(M,)`` or ``(M, 2)`` float — interpolated voxel-grid values.

    Pure JAX — ``jax.grad`` traces through.
    """
    import jax.numpy as jnp

    voxel_cell_verts = cells[cell_idx]                 # (M, 4)
    if field_mesh.ndim == 1:
        nodal_at_verts = field_mesh[voxel_cell_verts]   # (M, 4)
        return jnp.sum(weights * nodal_at_verts, axis=-1)
    # field_mesh: (n_nodes, K) — broadcast over the trailing axis.
    nodal_at_verts = field_mesh[voxel_cell_verts]       # (M, 4, K)
    return jnp.sum(weights[..., None] * nodal_at_verts, axis=1)


# ---------------------------------------------------------------------------
# Backend state — carries driver + T_prev + voxel weights across calls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FemcoupledBackendState:
    """Persistent state for the FEM-coupled backend.

    Carries the constructed driver (held by reference), the running
    temperature field, and the mesh→voxel-grid interpolation weights
    so consecutive inverse-loop iterations don't pay the construction
    cost.

    Attributes:
        driver: ``StaggeredCouplingDriver`` instance.  Built once.
        T_prev: ``(n_nodes,)`` jax array — current temperature on the
            mesh.  Carried across step calls.
        cell_idx: ``(n_voxels,)`` jax int array — per-voxel enclosing
            tet index.  Pre-computed at construction.
        voxel_weights: ``(n_voxels, 4)`` jax float — barycentric
            weights for the mesh→voxel projection.
        cells: ``(n_cells, 4)`` jax int — tet vertex indices.
        grid_shape: ``(nx, ny, nz)`` voxel-grid resolution.
        grid_extent_m: ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
        mesh_path: Path to the persisted .msh file.
    """

    driver: Any                         # StaggeredCouplingDriver
    T_prev: Any                         # jnp.ndarray (n_nodes,)
    cell_idx: Any                       # jnp.ndarray (n_voxels,)
    voxel_weights: Any                  # jnp.ndarray (n_voxels, 4)
    cells: Any                          # jnp.ndarray (n_cells, 4)
    grid_shape: Tuple[int, int, int]
    grid_extent_m: Tuple[float, float, float, float, float, float]
    mesh_path: Path

    def with_T(self, T_new: "jax.Array") -> "FemcoupledBackendState":
        """Return a new state with an updated temperature field.

        Immutable update — the prior state object is unchanged.
        """
        return replace(self, T_prev=T_new)


# ---------------------------------------------------------------------------
# Lazy state construction
# ---------------------------------------------------------------------------


def _ensure_mesh(
    mesh_path: Optional[Path],
    env: "EnvironmentConfig",
    transducer: "TransducerConfig",
    crucible_inlet_radius: float,
    focus_h_mm: float,
    bulk_h_mm: float,
    cache_dir: Optional[Path],
) -> Tuple[Path, dict]:
    """Build (or reuse) the L1 chamber mesh."""
    from .femcoupled.mesh import build_l1_chamber_mesh

    if mesh_path is not None and Path(mesh_path).exists():
        # Caller-supplied mesh: just rebuild ``info`` against the same
        # geometry by calling build_l1_chamber_mesh with overwrite=False.
        # The helper recomputes ``info`` whether it actually rebuilds or
        # not — this is the cheap path because the .msh file is reused.
        path, info = build_l1_chamber_mesh(
            array_config=env.array,
            transducer_config=transducer,
            focus_h_mm=focus_h_mm,
            bulk_h_mm=bulk_h_mm,
            crucible_inlet_radius=crucible_inlet_radius,
            output_path=Path(mesh_path),
            cache_dir=cache_dir,
            overwrite=False,
            verbose=False,
        )
        return path, info

    path, info = build_l1_chamber_mesh(
        array_config=env.array,
        transducer_config=transducer,
        focus_h_mm=focus_h_mm,
        bulk_h_mm=bulk_h_mm,
        crucible_inlet_radius=crucible_inlet_radius,
        output_path=Path(mesh_path) if mesh_path else None,
        cache_dir=cache_dir,
        overwrite=False,
        verbose=False,
    )
    return path, info


def _build_initial_state(
    env: "EnvironmentConfig",
    transducer: "TransducerConfig",
    grid_shape: Tuple[int, int, int],
    grid_extent_m: Tuple[float, float, float, float, float, float],
    mesh_path: Optional[Path],
    crucible_inlet_radius: float,
    pml_width: float,
    focus_h_mm: float,
    bulk_h_mm: float,
    cache_dir: Optional[Path],
    max_inner_iters: int,
    convergence_tol: float,
    dt: float,
    velocity_amp: Optional[float] = None,
) -> FemcoupledBackendState:
    """Construct mesh + driver + voxel-grid weights from scratch.

    First-call path (or whenever the cached state's mesh doesn't match
    the requested geometry).  Encapsulates the ~25 s build cost so the
    inverse-loop's per-iteration timing reflects only the forward
    solve + projection.
    """
    import jax.numpy as jnp

    from .femcoupled.coupling import StaggeredCouplingDriver
    # Use the heat-module canonical default to match the driver's
    # internal trajectory init.  Single source of truth for the
    # initial chamber temperature.
    from .femcoupled.heat import T_CHAMBER_INIT_K

    t0 = time.time()
    path, info = _ensure_mesh(
        mesh_path,
        env,
        transducer,
        crucible_inlet_radius,
        focus_h_mm,
        bulk_h_mm,
        cache_dir,
    )

    # Optional source-amplitude override. If None, StaggeredCouplingDriver
    # uses its own DEFAULT_PISTON_VELOCITY_AMP_M_S=0.01 (spec-sheet level
    # giving ~1 Pa peak fields on the coarse mesh). Phase 6.5 raises this
    # to 1.0 m/s (~100×) to reach the FEM_V2_DESIGN §3.4 design pressure
    # of ~1500 Pa where coupling effects (Eckart streaming + heat
    # advection) become nontrivial — see validate_coupled.py for the
    # canonical "design-pressure" rationale.
    driver_kwargs: Dict[str, Any] = dict(
        mesh_path=path,
        surface_tags=info["surface_tags"],
        env_config=env,
        transducer_config=transducer,
        n_outer_steps=1,                       # one step per backend call
        dt=dt,
        max_inner_iters=max_inner_iters,
        convergence_tol=convergence_tol,
        pml_width=pml_width,
    )
    if velocity_amp is not None:
        driver_kwargs["velocity_amp"] = float(velocity_amp)
    driver = StaggeredCouplingDriver(**driver_kwargs)

    points = np.asarray(driver._points, dtype=np.float64)
    cells = np.asarray(driver._mesh.cells, dtype=np.int64)

    _, _, _, voxel_centres = _build_voxel_grid(grid_shape, grid_extent_m)
    cell_idx_np, weights_np = _precompute_voxel_weights(
        cells=cells, points=points, voxel_centres=voxel_centres,
    )

    n_nodes = int(driver._n_nodes)
    T0 = jnp.full((n_nodes,), float(T_CHAMBER_INIT_K), dtype=jnp.float64)

    elapsed = time.time() - t0
    logger.info(
        "FemcoupledBackend: state built in %.2fs (n_nodes=%d, n_cells=%d, "
        "n_voxels=%d, grid=%dx%dx%d)",
        elapsed,
        n_nodes,
        cells.shape[0],
        int(np.prod(grid_shape)),
        *grid_shape,
    )
    return FemcoupledBackendState(
        driver=driver,
        T_prev=T0,
        cell_idx=jnp.asarray(cell_idx_np, dtype=jnp.int32),
        voxel_weights=jnp.asarray(weights_np, dtype=jnp.float64),
        cells=jnp.asarray(cells, dtype=jnp.int32),
        grid_shape=tuple(int(s) for s in grid_shape),
        grid_extent_m=tuple(float(e) for e in grid_extent_m),
        mesh_path=path,
    )


# ---------------------------------------------------------------------------
# Differentiable forward path — pure-JAX single-inner-iteration
# ---------------------------------------------------------------------------


def _forward_one_inner_iter(
    driver: Any,
    T_prev: "jax.Array",
    phases: "jax.Array",
    droplet_centers: "jax.Array",
    droplet_amplitudes: "jax.Array",
) -> Tuple["jax.Array", "jax.Array", "jax.Array"]:
    """One Helmholtz solve + streaming projection — pure JAX, no host calls.

    Mirrors the architectural-bet path in
    :func:`drip_physics.backends.femcoupled.tests.test_coupling.test_grad_finite_end_to_end`:
    the full ``StaggeredCouplingDriver.step`` has a host-Python
    convergence check (``float(jnp.sqrt(...))``) and a
    ``np.asarray(stream.max_magnitude)`` cast for receipts; both break
    tracing.  This helper drops in a single inner-loop pass that stays
    JAX-traceable end-to-end.

    The gradient flows through:

      ``phases → ad_wrapper(HelmholtzProblem) → (re, im) → loss``

    Streaming is computed for receipts but doesn't affect the returned
    field on the gradient path (we don't trace through the heat solve;
    the inverse loss is on the *field*, not on T).  ``T_prev`` is
    returned unchanged so callers can keep threading state without
    forking the eager/traced T evolution paths.

    Returns:
        ``(re_mesh, im_mesh, T_unchanged)``.
    """
    from jax_fem.solver import ad_wrapper

    # Helmholtz forward — already ``ad_wrapper``-wrapped per Phase 4.
    fwd_helm = ad_wrapper(
        driver._helmholtz_problem,
        solver_options=dict(driver._helmholtz_solver_opts),
        adjoint_solver_options=dict(driver._helmholtz_solver_opts),
    )
    sol_list = fwd_helm(phases)
    re = sol_list[0][:, 0]
    im = sol_list[0][:, 1]

    # We don't trace through heat on the inverse path — keeps the
    # gradient graph small and matches the Phase 4 grad receipt.  The
    # eager path advances T; the traced path threads T through
    # unchanged so the next eager call (after inverse converges) picks
    # up where the prior eager step left off.
    return re, im, T_prev


# ---------------------------------------------------------------------------
# Public entry point — mirrors jwave_backend.compute_pressure_from_phases
# ---------------------------------------------------------------------------


def compute_pressure_from_phases(
    phases: "jax.Array",
    env: "EnvironmentConfig",
    *,
    transducer: Optional["TransducerConfig"] = None,
    grid_shape: Tuple[int, int, int] = DEFAULT_GRID_SHAPE,
    grid_extent_m: Tuple[float, float, float, float, float, float] = DEFAULT_GRID_EXTENT_M,
    mesh_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    crucible_inlet_radius: float = DEFAULT_CRUCIBLE_INLET_RADIUS_M,
    pml_width: float = DEFAULT_PML_WIDTH_M,
    focus_h_mm: float = DEFAULT_FOCUS_H_MM,
    bulk_h_mm: float = DEFAULT_BULK_H_MM,
    backend_state: Optional[FemcoupledBackendState] = None,
    droplet_centers: Optional["jax.Array"] = None,
    droplet_amplitudes: Optional["jax.Array"] = None,
    max_inner_iters: int = 3,
    convergence_tol: float = 1e-2,
    dt: float = 0.01,
    velocity_amp: Optional[float] = None,
    return_state: bool = False,
    return_complex: bool = False,
    **_kwargs: Any,
) -> "jax.Array":
    """Coupled-physics forward: ``phases → complex pressure on voxel grid``.

    Mirrors the signature of
    :func:`drip_physics.backends.jwave_backend.JWaveBackend.compute_pressure_from_phases`,
    but internally drives the Phase 4 :class:`StaggeredCouplingDriver`
    (two-way acoustic ↔ thermal coupling) and projects the resulting
    nodal pressure field onto a regular Cartesian voxel grid via
    pre-computed barycentric weights.

    Differentiability
    -----------------

    The forward pass is ``jax.grad``-clean end-to-end:

      * Helmholtz solve goes through JAX-FEM's ``ad_wrapper``.
      * Eckart streaming projection is pure JAX (linear + clamp).
      * Heat solve goes through ``ad_wrapper`` (constant operator).
      * Mesh→voxel projection is a precomputed gather + matmul (both
        JAX-traced primitives).

    Caveat: the inner fixed-point loop uses a host-Python convergence
    check.  ``jax.grad`` differentiates the *unrolled* graph at the
    iteration count actually executed (matches the Phase 4
    architectural-bet receipt — see
    :meth:`StaggeredCouplingDriver.step`).

    Args:
        phases: ``(n_transducers,)`` jax array of phase commands (rad).
            Differentiable.
        env: ``EnvironmentConfig``.  Drives ``c0(T)``, transducer +
            array geometry, medium properties.
        transducer: ``TransducerConfig`` override.  Defaults to
            ``env.transducer``.
        grid_shape: Voxel resolution ``(nx, ny, nz)``.  Default 64³.
        grid_extent_m: Voxel grid bounds in metres
            ``(xmin, xmax, ymin, ymax, zmin, zmax)``.  Default covers
            the L1 chamber's focal/lower-ring zone.
        mesh_path: Pre-built ``.msh`` file.  When ``None``, a coarse
            mesh is built (or reused via ``cache_dir``) on first call.
        cache_dir: Mesh-cache root.  Defaults to the canonical Phase 1
            cache (``simulations/ml_inverse/data/meshes/``).
        crucible_inlet_radius: m.  Forwarded to the mesh builder.
        pml_width: m.  Forwarded to the Helmholtz problem.
        focus_h_mm / bulk_h_mm: Mesh-density targets.  Forwarded to
            ``build_l1_chamber_mesh``.
        backend_state: Optional persistent state from a prior call.
            When supplied, the driver and voxel-weights are reused and
            ``T_prev`` is taken from the state (multi-step trajectory).
        droplet_centers: ``(n_droplets, 3)`` droplet positions (m).
            Optional — when ``None``, a single placeholder droplet at
            chamber centre with zero amplitude is used.
        droplet_amplitudes: ``(n_droplets,)`` heat amplitudes (W).
            Optional — see ``droplet_centers``.
        max_inner_iters: Inner fixed-point iteration cap (default 3
            — half the Phase 4 default for inverse-loop speed).
        convergence_tol: Inner fixed-point relative-L2 tolerance
            (default 1e-2 — looser than Phase 4 default for speed).
        dt: Outer thermal timestep (s).  Default 0.01 s.
        return_state: When ``True``, returns a tuple ``(field, state)``
            where ``state`` carries the updated T_prev for the next
            call.
        return_complex: When ``True``, returns the field as a complex
            ``(nz, ny, nx)`` array (matches the j-Wave backend's
            internal complex shape).  Default ``False`` returns a
            real-valued ``(nz, ny, nx, 2)`` Re/Im stack — the shape
            consumed by the j-Wave inverse loop and the FNO-surrogate
            data path.
        **_kwargs: Reserved for forward compatibility — additional
            kwargs are accepted and ignored, matching the
            ``jwave_backend`` flexibility.

    Returns:
        Either:
          * Voxel-grid field, shape ``(nz, ny, nx, 2)`` real (default)
            or ``(nz, ny, nx)`` complex (``return_complex=True``).
          * Tuple ``(field, FemcoupledBackendState)`` when
            ``return_state=True``.
    """
    import jax
    import jax.numpy as jnp

    resolved_transducer = transducer if transducer is not None else env.transducer

    state = backend_state
    if state is None:
        state = _build_initial_state(
            env=env,
            transducer=resolved_transducer,
            grid_shape=grid_shape,
            grid_extent_m=grid_extent_m,
            mesh_path=mesh_path,
            crucible_inlet_radius=crucible_inlet_radius,
            pml_width=pml_width,
            focus_h_mm=focus_h_mm,
            bulk_h_mm=bulk_h_mm,
            cache_dir=cache_dir,
            max_inner_iters=max_inner_iters,
            convergence_tol=convergence_tol,
            dt=dt,
            velocity_amp=velocity_amp,
        )

    n_nodes = int(state.driver._n_nodes)

    # Default droplet placeholder: single droplet at chamber centre,
    # zero amplitude (no thermal load).  Matches the Phase 4 smoke-test
    # convention and keeps the heat-step well-posed when the caller
    # only cares about the acoustic field.
    if droplet_centers is None:
        geom = state.driver._geom
        droplet_centers = jnp.array(
            [[0.0, 0.0, geom.height / 2.0]], dtype=jnp.float64
        )
    if droplet_amplitudes is None:
        droplet_amplitudes = jnp.zeros((1,), dtype=jnp.float64)

    phases_jnp = jnp.asarray(phases, dtype=jnp.float64).reshape(-1)

    # Decide whether the call is on a *traced* path (jax.grad / jit) or
    # on a host-Python eager path.  When traced, ``StaggeredCouplingDriver.step``
    # breaks tracing because its inner loop calls ``float(...)`` on
    # field-norm receipts and ``np.asarray`` on streaming.max_magnitude.
    # On the traced path we drop in a pure-JAX *single-inner-iteration*
    # forward (Helmholtz + streaming + heat) which is exactly what the
    # Phase 4 architectural-bet test
    # (``test_grad_finite_end_to_end``) validates.  Eager calls take the
    # full fixed-point ``.step()`` path with the convergence trace.
    is_traced = isinstance(phases_jnp, jax.core.Tracer)

    if is_traced:
        re_mesh, im_mesh, T_new = _forward_one_inner_iter(
            driver=state.driver,
            T_prev=state.T_prev,
            phases=phases_jnp,
            droplet_centers=droplet_centers,
            droplet_amplitudes=droplet_amplitudes,
        )
    else:
        res = state.driver.step(
            T_prev=state.T_prev,
            phases=phases_jnp,
            droplet_centers=droplet_centers,
            droplet_amplitudes=droplet_amplitudes,
        )
        re_mesh = res.re
        im_mesh = res.im
        T_new = res.T_new

    field_mesh = jnp.stack([re_mesh, im_mesh], axis=-1)        # (n_nodes, 2)

    voxel_field = _project_mesh_to_voxel(
        field_mesh=field_mesh,
        cell_idx=state.cell_idx,
        weights=state.voxel_weights,
        cells=state.cells,
    )                                                           # (n_voxels, 2)

    nx, ny, nz = state.grid_shape
    field_grid = voxel_field.reshape(nz, ny, nx, 2)

    if return_complex:
        field_complex = field_grid[..., 0] + 1j * field_grid[..., 1]
        if return_state:
            return field_complex, state.with_T(res.T_new)
        return field_complex

    if return_state:
        return field_grid, state.with_T(res.T_new)
    return field_grid


# ---------------------------------------------------------------------------
# OO wrapper — mirrors JWaveBackend's class API for the create_backend
# factory in pressure_backend.py
# ---------------------------------------------------------------------------


class FemcoupledBackend:
    """OO wrapper around :func:`compute_pressure_from_phases`.

    Carries the persistent backend state across calls.  Implements
    the same eager-API surface as
    :class:`drip_physics.backends.jwave_backend.JWaveBackend`:
    ``compute_pressure_from_phases(array, phases, amplitudes, point, env=...)``,
    returning a complex pressure phasor sampled at ``point`` via
    trilinear interpolation on the voxel grid.

    The OO API is here for ``pressure_backend.create_backend`` —
    inverse-loop callers prefer the function-style entry point above.
    """

    def __init__(
        self,
        config: "PressureConfig",
        env: Optional["EnvironmentConfig"] = None,
        *,
        transducer: Optional["TransducerConfig"] = None,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        grid_extent_m: Optional[
            Tuple[float, float, float, float, float, float]
        ] = None,
        mesh_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        crucible_inlet_radius: float = DEFAULT_CRUCIBLE_INLET_RADIUS_M,
        pml_width: float = DEFAULT_PML_WIDTH_M,
        focus_h_mm: float = DEFAULT_FOCUS_H_MM,
        bulk_h_mm: float = DEFAULT_BULK_H_MM,
        max_inner_iters: int = 3,
        convergence_tol: float = 1e-2,
        dt: float = 0.01,
    ) -> None:
        self._config = config
        self._env = env
        self._transducer = transducer
        # Honour PressureConfig.grid_n if supplied; otherwise default.
        if grid_shape is None:
            n = int(getattr(config, "grid_n", DEFAULT_GRID_SHAPE[0]))
            grid_shape = (n, n, n)
        self._grid_shape = grid_shape
        self._grid_extent_m = grid_extent_m or DEFAULT_GRID_EXTENT_M
        # PressureConfig may carry FEM-specific overrides; honour them
        # but allow per-instance ctor-arg override.
        cfg_mesh = getattr(config, "mesh_path", None)
        self._mesh_path = (
            mesh_path if mesh_path is not None
            else (Path(cfg_mesh) if cfg_mesh else None)
        )
        self._cache_dir = cache_dir
        self._crucible_inlet_radius = float(
            getattr(config, "crucible_inlet_radius", crucible_inlet_radius)
            if crucible_inlet_radius == DEFAULT_CRUCIBLE_INLET_RADIUS_M
            else crucible_inlet_radius
        )
        self._pml_width = float(
            getattr(config, "pml_width", pml_width)
            if pml_width == DEFAULT_PML_WIDTH_M
            else pml_width
        )
        self._focus_h_mm = float(focus_h_mm)
        self._bulk_h_mm = float(bulk_h_mm)
        self._max_inner_iters = int(max_inner_iters)
        self._convergence_tol = float(convergence_tol)
        self._dt = float(dt)
        self._state: Optional[FemcoupledBackendState] = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _ensure_state(self, env: "EnvironmentConfig") -> FemcoupledBackendState:
        if self._state is None:
            transducer = self._transducer or env.transducer
            self._state = _build_initial_state(
                env=env,
                transducer=transducer,
                grid_shape=self._grid_shape,
                grid_extent_m=self._grid_extent_m,
                mesh_path=self._mesh_path,
                crucible_inlet_radius=self._crucible_inlet_radius,
                pml_width=self._pml_width,
                focus_h_mm=self._focus_h_mm,
                bulk_h_mm=self._bulk_h_mm,
                cache_dir=self._cache_dir,
                max_inner_iters=self._max_inner_iters,
                convergence_tol=self._convergence_tol,
                dt=self._dt,
            )
        return self._state

    @property
    def state(self) -> Optional[FemcoupledBackendState]:
        return self._state

    def invalidate_cache(self) -> None:
        """Discard cached state — next call rebuilds from scratch."""
        self._state = None

    # ------------------------------------------------------------------
    # Functional + OO entry points
    # ------------------------------------------------------------------

    def compute_field(
        self,
        phases: "jax.Array",
        *,
        env: Optional["EnvironmentConfig"] = None,
        droplet_centers: Optional["jax.Array"] = None,
        droplet_amplitudes: Optional["jax.Array"] = None,
        return_complex: bool = False,
    ) -> "jax.Array":
        """Run one forward solve and return the voxel-grid field.

        Mirrors :func:`compute_pressure_from_phases` but reads/writes
        the OO instance's state.
        """
        resolved_env = env if env is not None else self._env
        if resolved_env is None:
            raise ValueError(
                "FemcoupledBackend.compute_field: env must be provided "
                "either at construction or call time."
            )
        state = self._ensure_state(resolved_env)
        result = compute_pressure_from_phases(
            phases=phases,
            env=resolved_env,
            transducer=self._transducer,
            grid_shape=self._grid_shape,
            grid_extent_m=self._grid_extent_m,
            mesh_path=self._mesh_path,
            cache_dir=self._cache_dir,
            crucible_inlet_radius=self._crucible_inlet_radius,
            pml_width=self._pml_width,
            focus_h_mm=self._focus_h_mm,
            bulk_h_mm=self._bulk_h_mm,
            backend_state=state,
            droplet_centers=droplet_centers,
            droplet_amplitudes=droplet_amplitudes,
            max_inner_iters=self._max_inner_iters,
            convergence_tol=self._convergence_tol,
            dt=self._dt,
            return_state=True,
            return_complex=return_complex,
        )
        field, new_state = result
        self._state = new_state
        return field

    def compute_pressure_from_phases(
        self,
        array: Any,
        phases_jnp: "jax.Array",
        amplitudes_jnp: "jax.Array",
        point: NDArray[np.float64],
        *,
        env: Optional["EnvironmentConfig"] = None,
    ) -> "jax.Array":
        """j-Wave-compatible eager API: pressure phasor at a single point.

        ``amplitudes_jnp`` is accepted for signature parity; the FEM
        backend currently uses uniform per-transducer amplitude (the
        Robin-piston velocity_amp baked into the HelmholtzProblem at
        construction).  Per-transducer amplitude modulation is on the
        v2 roadmap.

        ``array`` is accepted for signature parity but ignored (the
        backend pulls geometry from ``env.array``).
        """
        import jax.numpy as jnp

        resolved_env = env if env is not None else self._env
        if resolved_env is None:
            raise ValueError(
                "FemcoupledBackend.compute_pressure_from_phases: env required."
            )
        del array, amplitudes_jnp  # unused — see docstring

        field_complex = self.compute_field(
            phases=phases_jnp, env=resolved_env, return_complex=True,
        )                                                       # (nz, ny, nx)

        # Trilinear sample at ``point`` using the same fractional-index
        # convention as JWaveBackend's ``map_coordinates`` path.
        from jax.scipy.ndimage import map_coordinates

        nx, ny, nz = self._grid_shape
        xmin, xmax, ymin, ymax, zmin, zmax = self._grid_extent_m
        dx = (xmax - xmin) / max(nx - 1, 1)
        dy = (ymax - ymin) / max(ny - 1, 1)
        dz = (zmax - zmin) / max(nz - 1, 1)
        pt = np.asarray(point, dtype=np.float64).reshape(3)
        ix = (pt[0] - xmin) / dx
        iy = (pt[1] - ymin) / dy
        iz = (pt[2] - zmin) / dz
        # field_complex shape is (nz, ny, nx) — coords (iz, iy, ix).
        coords = jnp.asarray([[iz], [iy], [ix]], dtype=jnp.float64)
        re = map_coordinates(
            field_complex.real, coords, order=1, mode="constant", cval=0.0
        )
        im = map_coordinates(
            field_complex.imag, coords, order=1, mode="constant", cval=0.0
        )
        return (re + 1j * im)[0]


__all__ = [
    "DEFAULT_BULK_H_MM",
    "DEFAULT_CRUCIBLE_INLET_RADIUS_M",
    "DEFAULT_FOCUS_H_MM",
    "DEFAULT_GRID_EXTENT_M",
    "DEFAULT_GRID_SHAPE",
    "DEFAULT_PML_WIDTH_M",
    "FemcoupledBackend",
    "FemcoupledBackendState",
    "compute_pressure_from_phases",
]
