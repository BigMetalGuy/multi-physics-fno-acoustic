"""HelmholtzProblem — parametric Robin transducer BC + selectable absorber.

Phase 2 of the FEM-coupled-physics v1 build (FEM_V2_DESIGN.md §3.2/§3.7).

Phase 2.0 shipped:
  * :class:`HelmholtzProblem` — JAX-FEM ``Problem`` subclass with
    ``vec=2`` (Re/Im of complex pressure) and ``dim=3``.
  * Bulk weak form ``∫ ∇u · ∇v - k² u v dV`` (component-wise; Re and Im
    decouple in the bulk for a real-coefficient operator).
  * Per-transducer parametric Robin BC ``∂_n p = i k ρ₀ c₀ v_n(φᵢ)``
    where ``v_n(φᵢ) = v_amp · exp(i·φᵢ)``. Phases enter via
    :meth:`HelmholtzProblem.set_params` so ``jax.grad`` flows through.
  * First-order Sommerfeld radiation BC ``∂_n p = i k p`` on the
    outer boundary (walls + lid + substrate).

Phase 2.1 adds (FEM_DISCOVERY_REPORT §4 Q1 resolution):
  * **Bermúdez 2007 unbounded-coefficient PML** — opt-in via
    ``pml_width > 0``.  The bulk weak form picks up a per-quadrature-
    point complex stretching ``s(x) = 1 - i·σ(x)/ω`` modulating the
    gradient and mass terms within a ramped layer of width ``L_pml``
    near the outer boundary.  Inside the PML the wavenumber is
    effectively ``k → k/s`` and the weak form becomes::

        ∫ (1/s)·∇p·∇v - k²·s·p·v dV.

    The "unbounded" coefficient ``σ(d) = α/(L - d)`` (where ``d`` is
    the distance into the layer) is singular at the outer wall — the
    feature that gives Bermúdez its faster convergence than Berenger
    split-field PML for time-harmonic problems and avoids UPML's
    parameter-tuning fragility (FEM_V2_DESIGN §3.7).  Outside the PML
    zone ``s = 1`` exactly, so bulk physics is identity-modulated.
  * Sommerfeld surface BC becomes a **no-op** when PML is enabled
    (the layer absorbs; the outer wall reflection survives only
    ``e^{-α}`` attenuated, well below 1% of incident at α = 1).
  * Optional **PETSc-MUMPS** solver path via
    ``solver_options={"petsc_solver": {...}}`` — direct LU through
    ``petsc4py``.  Falls back to ``scipy.sparse.linalg.spsolve`` if
    PETSc isn't importable.  Tests use the spsolve fallback; MUMPS is
    a production-deploy optional path.

The Sommerfeld BC (Phase 2.0 path) couples Re and Im across the
boundary integral:

    ∂_n(Re p) = -k · Im p
    ∂_n(Im p) = +k · Re p

so the surface map for Re returns ``-k · u[1]`` and for Im
``+k · u[0]`` (in the JAX-FEM "flux" sign convention from
``helmholtz_grad_check.py``).

Example
-------

::

    from drip_physics.backends.femcoupled import build_l1_chamber_mesh
    from drip_physics.backends.femcoupled.helmholtz import solve_helmholtz
    from drip_physics.config import L1_ARRAY, MA40S4S_40KHZ, default_environment

    path, info = build_l1_chamber_mesh(L1_ARRAY, MA40S4S_40KHZ)
    phases = jnp.zeros(120)
    # Sommerfeld v0 (Phase 2.0 backward-compat):
    result_v0 = solve_helmholtz(..., pml_width=0.0)
    # Bermúdez PML (Phase 2.1 default):
    result_pml = solve_helmholtz(..., pml_width=4e-3, pml_alpha=1.0)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import meshio
import numpy as np

from drip_physics.config import (
    ArrayConfig,
    EnvironmentConfig,
    TransducerConfig,
)

from .geometry import ChamberGeometry, resolve_chamber_geometry, transducer_centers
from .mesh import load_jax_fem_mesh


# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

# Default piston velocity amplitude. Calibrated from MA40S4S spec
# ``p_ref=10 Pa @ r_ref=0.3 m`` ⇒ free-field amplitude at the face is
# of order 10 Pa·(0.3/a) ≈ 1200 Pa, giving v_amp ≈ p/(ρ₀ c₀) ≈ 3 mm/s.
# We use 0.01 m/s (= 1 cm/s) as a conservative spec-sheet level so the
# resulting field magnitudes are comparable to the j-Wave backend audit
# under default phases.
DEFAULT_PISTON_VELOCITY_AMP_M_S: float = 0.01

# Geometric tolerances for surface predicate discrimination. The chamber
# uses cylindrical/disk geometry; tolerance is in metres.
SURFACE_TOL_M: float = 5e-4  # 0.5 mm

# Default linear-solver options. The Helmholtz weak form
# ``∫ ∇u · ∇v - k² u v`` is *indefinite* (saddle-point structure for
# k·L large), so JAX-FEM's default JAX BiCGSTAB does not converge. We
# route to JAX-FEM's ``umfpack_solver`` path — scipy ``spsolve`` under
# the hood, bullet-proof on CPU for the L1 mesh sizes (≤ a few × 10⁵
# DOFs). For production CUDA-cluster runs we'll swap to the PETSc +
# MUMPS path.
#
# Note: prior versions of this constant used the key ``spsolve_solver``,
# which JAX-FEM 0.0.9+ does not recognize. JAX-FEM's dispatcher
# (``solver.linear_solver``) accepts only
# ``{'jax_solver','amgx_solver','umfpack_solver','petsc_solver','custom_solver'}``;
# any other key silently falls through to ``jax_solver`` (BiCGSTAB),
# which diverges on this indefinite system. Use ``umfpack_solver``.
DEFAULT_SOLVER_OPTIONS: Dict[str, Any] = {"umfpack_solver": {}}

# Phase 2.1 PML defaults (FEM_DISCOVERY_REPORT §4 Q1).
# Default ``pml_width = 0.0`` keeps Phase 2.0 backward-compat: callers
# that don't request PML continue to get the Sommerfeld v0 path
# unchanged.  Recommended PML width is ~λ/2 at 40 kHz in air (8.575 mm
# wavelength → 4.3 mm), the canonical Bermúdez range; α = 1.0 gives
# ~-30 dB end-to-end reflection in practice (the unbounded-coefficient
# integral compounds beyond the e^{-α} amplitude estimate).
DEFAULT_PML_WIDTH_M: float = 0.0
DEFAULT_PML_ALPHA: float = 1.0
# Recommended PML width when callers opt in.
RECOMMENDED_PML_WIDTH_M: float = 4e-3
# Numerical floor for the singular denominator (L - d). Without this,
# nodes that land exactly on the singular wall produce inf · 0 NaNs in
# the assembled bilinear form. 1e-9 m is well below mesh resolution.
PML_DENOM_FLOOR_M: float = 1e-9


# ---------------------------------------------------------------------------
# Solver-side helpers (location functions and lazy JAX-FEM imports)
# ---------------------------------------------------------------------------


def _import_jax_fem():
    """Lazy import of JAX-FEM (avoids a hard dependency at module-import time).

    JAX-FEM is GPL-3.0 and lives in the discovery sidecar venv. The
    Phase 5 ``femcoupled_backend`` shim will surface a clear error if
    JAX-FEM isn't importable; here we import on first use.
    """
    from jax_fem.problem import Problem
    from jax_fem.generate_mesh import Mesh
    from jax_fem.solver import ad_wrapper, solver

    return Problem, Mesh, ad_wrapper, solver


def _make_transducer_location_fn(
    center: np.ndarray,
    face_radius: float,
    cylinder_radius: float,
    tol: float = SURFACE_TOL_M,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a location_fn that selects faces lying on the given disk.

    A point ``p`` is on transducer disk ``i`` iff:

      * ``sqrt(p_x² + p_y²) ≈ cylinder_radius`` (on the cylindrical wall),
      * ``||p - center|| ≤ face_radius + tol`` (within the disk footprint).

    JAX-FEM applies the predicate to every node of every face and
    requires ALL nodes to satisfy it before the face is selected, so
    this conservatively pulls in only triangles whose three vertices
    all lie on the disk patch (matching the Gmsh fragment output).
    """
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    r = float(cylinder_radius)
    a = float(face_radius)

    def loc_fn(point: jnp.ndarray) -> jnp.ndarray:
        rho = jnp.sqrt(point[0] ** 2 + point[1] ** 2)
        on_cyl = jnp.isclose(rho, r, atol=tol)
        d = jnp.sqrt(
            (point[0] - cx) ** 2
            + (point[1] - cy) ** 2
            + (point[2] - cz) ** 2
        )
        within_disk = d <= (a + tol)
        return on_cyl & within_disk

    return loc_fn


def _pml_stretching(
    x: jnp.ndarray,
    cyl_radius: float,
    height: float,
    L_pml: float,
    alpha: float,
    k_val: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the Bermúdez 2007 unbounded-coefficient PML stretching.

    Returns ``(s_re, s_im)`` per quadrature point.  Outside the PML
    zone the stretching is identity (``s_re=1, s_im=0``) so the weak
    form falls back to the standard Helmholtz operator.  Within the
    PML the stretching is

        s(x) = 1 + i · α / (k · (L_pml - d))

    where ``d`` is the distance into the layer (``d=0`` at the inner
    edge, ``d=L_pml`` at the singular outer wall) and ``k`` is the
    bulk wavenumber.  The ``+i`` sign matches the ``e^{-iωt}`` time
    convention used by the Robin BC at the transducer faces (flux
    convention ``∂_n p = +ik·ρ₀c₀·v_n``): outgoing waves ``e^{ikr}``
    get exponentially damped under the coordinate stretch
    ``r → r + i·∫σ/k dl``.

    Note on units: ``α`` is dimensionless and the ``k``-normalisation
    keeps ``s`` dimensionless.  The dispersion relation in the bulk
    PML weak form ``∫∇p·∇v - k²s²pv dV`` becomes ``K = k·s``, giving
    decay ``Im(K)/k = α/(k(L-d))``.  The unbounded coefficient is per
    Bermúdez et al. 2007 — the singularity at the wall makes the PML
    reflectionless without the auxiliary fields of Berenger split-
    field PML (the integral ``∫₀^L α/(k(L-d)) dd`` diverges, giving
    unbounded decay over the layer).

    The PML zones are:

      * radial:  ρ > R - L_pml      (annular shell at the cylinder wall)
      * bottom:  z < L_pml          (slab above the substrate)
      * top:     z > H - L_pml      (slab below the lid)

    Where multiple zones overlap (the corners), the ``σ`` contributions
    add — this is the simplified isotropic Bermúdez approximation
    (FEM_V2_DESIGN §3.7) trading the full anisotropic tensor stretching
    for ~30 lines of physics code with no auxiliary fields.  Practical
    receipts: tests below show -20+ dB reflection at this layer width.

    The denominator ``(L_pml - d)`` is floored at ``PML_DENOM_FLOOR_M``
    to keep the assembled bilinear form free of inf·0 NaNs at nodes
    that land exactly on the singular wall.

    Args:
        x: ``(num_quads, 3)`` quadrature point coordinates (metres).
        cyl_radius: chamber radius ``R``.
        height: chamber height ``H``.
        L_pml: PML layer width.
        alpha: PML decay strength (dimensionless).
        k_val: bulk wavenumber ``2π·f/c`` (1/m); used to non-
            dimensionalise the stretching coefficient.

    Returns:
        ``(s_re, s_im)`` each shape ``(num_quads,)``.
    """
    # Distance into each PML zone (zero if not inside).
    rho = jnp.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
    d_radial = jnp.maximum(0.0, rho - (cyl_radius - L_pml))
    d_bot = jnp.maximum(0.0, L_pml - x[..., 2])
    d_top = jnp.maximum(0.0, L_pml - (height - x[..., 2]))
    d_axial = jnp.maximum(d_bot, d_top)

    # Singular σ contributions; floored to avoid div-by-zero at the wall.
    # σ has units of 1/m through the (L - d) denominator; we then
    # divide by k to make s dimensionless.
    denom_radial = jnp.maximum(L_pml - d_radial, PML_DENOM_FLOOR_M)
    denom_axial = jnp.maximum(L_pml - d_axial, PML_DENOM_FLOOR_M)
    sigma_radial = jnp.where(d_radial > 0.0, alpha / denom_radial, 0.0)
    sigma_axial = jnp.where(d_axial > 0.0, alpha / denom_axial, 0.0)

    # s(x) = 1 + i · σ_total / k_val (Bermúdez unbounded-coefficient;
    # e^{-iωt} convention; outgoing e^{ikr} waves decay under the
    # coordinate stretch).
    sigma_total = sigma_radial + sigma_axial
    s_re = jnp.ones_like(sigma_total)
    s_im = sigma_total / k_val
    return s_re, s_im


def _make_sommerfeld_location_fn(
    geom: ChamberGeometry,
    transducer_centers_arr: np.ndarray,
    face_radius: float,
    tol: float = SURFACE_TOL_M,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a location_fn for the Sommerfeld outer boundary.

    Includes:
      * substrate (z = 0),
      * lid (z = height),
      * cylinder side wall NOT covered by any transducer disk.

    Excludes the transducer disks themselves (those carry the Robin
    piston BC).
    """
    r = float(geom.cylinder_radius)
    h = float(geom.height)
    a = float(face_radius)
    centers_jax = jnp.asarray(transducer_centers_arr, dtype=jnp.float64)

    def loc_fn(point: jnp.ndarray) -> jnp.ndarray:
        # Distance from each transducer centre.
        dxy_sq = (
            (point[0] - centers_jax[:, 0]) ** 2
            + (point[1] - centers_jax[:, 1]) ** 2
            + (point[2] - centers_jax[:, 2]) ** 2
        )
        d_min = jnp.sqrt(jnp.min(dxy_sq))
        in_any_disk = d_min <= (a + tol)

        rho = jnp.sqrt(point[0] ** 2 + point[1] ** 2)
        on_substrate = jnp.isclose(point[2], 0.0, atol=tol)
        on_lid = jnp.isclose(point[2], h, atol=tol)
        on_side_wall = jnp.isclose(rho, r, atol=tol)

        on_outer = on_substrate | on_lid | (on_side_wall & ~in_any_disk)
        return on_outer

    return loc_fn


# ---------------------------------------------------------------------------
# HelmholtzProblem
# ---------------------------------------------------------------------------


def make_helmholtz_problem(
    mesh: "Any",
    geom: ChamberGeometry,
    env_config: EnvironmentConfig,
    transducer_config: TransducerConfig,
    velocity_amp: float = DEFAULT_PISTON_VELOCITY_AMP_M_S,
    pml_width: float = DEFAULT_PML_WIDTH_M,
    pml_alpha: float = DEFAULT_PML_ALPHA,
) -> "HelmholtzProblem":
    """Construct a :class:`HelmholtzProblem` on the given JAX-FEM mesh.

    The factory builds the per-transducer + outer location_fns,
    instantiates the ``Problem``, and seeds zero phases via
    :meth:`HelmholtzProblem.set_params`. The instance is ready for
    ``solve_helmholtz`` or direct use under ``ad_wrapper``.

    Args:
        mesh: JAX-FEM ``Mesh`` instance (from :func:`load_jax_fem_mesh`).
        geom: resolved chamber geometry.
        env_config: environment (medium, frequency, array). Determines
            ``k = 2π f / c0`` and ``ρ₀, ω``.
        transducer_config: transducer face geometry (aperture radius).
        velocity_amp: piston velocity amplitude (m/s) at each face.
        pml_width: width of the Bermúdez PML layer in metres. Set to
            ``0.0`` to disable PML and fall back to Sommerfeld v0
            radiation BCs (Phase 2.0 backward-compat path).
        pml_alpha: PML decay strength (dimensionless). Larger ⇒ more
            aggressive absorption but more sensitive to mesh resolution
            in the PML zone. ``α = 1.0`` is the canonical Bermúdez
            value for time-harmonic problems.
    """
    Problem, _, _, _ = _import_jax_fem()

    centers = transducer_centers(geom)
    a = geom.transducer_face_radius

    # Build per-disk location_fns (120 entries) followed by ONE Sommerfeld
    # BC. ordering: 0..n_total-1 = transducers, n_total = sommerfeld.
    location_fns: List[Callable[[jnp.ndarray], jnp.ndarray]] = [
        _make_transducer_location_fn(centers[i], a, geom.cylinder_radius)
        for i in range(geom.total_transducers)
    ]
    location_fns.append(
        _make_sommerfeld_location_fn(geom, centers, a)
    )

    c0 = float(env_config.speed_of_sound)
    omega = float(env_config.angular_frequency)
    k = float(env_config.wavenumber)
    rho0 = float(env_config.medium.density)

    pml_enabled = float(pml_width) > 0.0
    L_pml = float(pml_width)
    alpha_pml = float(pml_alpha)
    cyl_R = float(geom.cylinder_radius)
    H = float(geom.height)

    # Bind the constants the closures need so the subclass body is small.
    additional = {
        "n_transducers": int(geom.total_transducers),
        "k_val": k,
        "omega": omega,
        "rho0": rho0,
        "v_amp": float(velocity_amp),
        "pml_enabled": pml_enabled,
        "L_pml": L_pml,
        "alpha_pml": alpha_pml,
        "cyl_R": cyl_R,
        "H": H,
    }

    class HelmholtzProblem(Problem):
        """Time-harmonic Helmholtz field with vec=2 (Re, Im).

        Implements:
          * Bulk (Sommerfeld v0 path, ``pml_width=0``):
            ``∫ ∇u · ∇v - k² u v dV`` per component (Re/Im decoupled).
          * Bulk (Bermúdez PML path, ``pml_width>0``):
            ``∫ (1/s)·∇u·∇v - k²·s·u·v dV`` where ``s(x)`` is the
            per-quadrature-point complex stretching from
            :func:`_pml_stretching`. Re/Im couple inside the PML zone.
          * Robin (transducer face i): ``∂_n p = iρ₀ω·v_amp·exp(iφᵢ)``.
            In real form: flux to (Re, Im) = (-ρ₀ω·v_amp·sin(φᵢ),
            +ρ₀ω·v_amp·cos(φᵢ)). Same in both bulk paths.
          * Outer surface (Sommerfeld path): ``∂_n p = ik p`` ⇒ flux to
            (Re, Im) = (-k·u[1], +k·u[0]).
          * Outer surface (PML path): no surface contribution — the PML
            absorbs and the outer wall is implicitly hard-reflective.
        """

        # NOTE: When ``pml_enabled=False`` we expose ``get_tensor_map``
        # and ``get_mass_map`` so the (Re, Im)-decoupled bulk form
        # routes through the legacy laplace+mass kernels (Phase 2.0
        # contract).  When PML is on we expose ``get_universal_kernel``
        # only — JAX-FEM dispatches to whichever method ``hasattr``s.
        if not pml_enabled:

            def get_tensor_map(self):
                def tensor_fn(u_grad: jnp.ndarray) -> jnp.ndarray:
                    # Identity flux ⇒ ∫ ∇u·∇v dV.
                    return u_grad

                return tensor_fn

            def get_mass_map(self):
                k2 = additional["k_val"] ** 2

                def mass_fn(u: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
                    # Helmholtz mass: residual contribution -∫ k² u v dV.
                    return -k2 * u

                return mass_fn

        else:

            def get_universal_kernel(self):
                """PML-modulated bulk weak form (Bermúdez 2007, scalar form).

                Within the PML zone the wavenumber is complex-shifted
                ``k → k·s`` where ``s(x) = 1 + i·σ(x)/k``.  In the weak
                form this becomes

                    ∫ ∇p · ∇v - k²·s²·p·v dV

                with the gradient term unchanged and the mass term
                picking up a complex factor ``s²``.  The dispersion
                relation in the bulk gives ``K = k·s``; the imaginary
                part of ``K`` is exactly ``σ`` (in 1/m), so outgoing
                waves decay as ``e^{-σ·dr}``.  Integrating over the
                layer with the unbounded-coefficient ``σ ∝ 1/(L-d)``
                gives unbounded total decay — the audit Q1 promise of
                a reflectionless absorber without auxiliary fields.

                Outside the PML zone ``s = 1`` exactly and Re/Im
                decouple, recovering the v0 weak form bit-for-bit.
                The s² mass coupling preserves ``jax.grad`` cleanliness
                because the complex weight is real-analytic in the
                stretching parameter (verified by
                ``test_pml_grad_finite_and_clean`` — AD vs FD agreement
                in the same envelope as Phase 2.0).
                """
                k_val = additional["k_val"]
                k2 = k_val * k_val
                L = additional["L_pml"]
                alpha = additional["alpha_pml"]
                R_cyl = additional["cyl_R"]
                H_chamber = additional["H"]

                def universal_kernel(
                    cell_sol_flat,
                    x,                      # (num_quads, dim)
                    cell_shape_grads,       # (num_quads, num_nodes+..., dim)
                    cell_JxW,               # (num_vars, num_quads)
                    cell_v_grads_JxW,       # (num_quads, num_nodes+..., 1, dim)
                    *cell_internal_vars,
                ):
                    cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
                    n_nodes = self.fes[0].num_nodes
                    cell_shape_grads = cell_shape_grads[:, :n_nodes, :]
                    cell_v_grads_JxW = cell_v_grads_JxW[:, :n_nodes, :, :]
                    cell_sol = cell_sol_list[0]                  # (num_nodes, 2)
                    cell_JxW0 = cell_JxW[0]                       # (num_quads,)
                    shape_vals = self.fes[0].shape_vals            # (num_quads, num_nodes)

                    # Per-quadrature complex stretching s = s_re + i·s_im.
                    s_re, s_im = _pml_stretching(
                        x, R_cyl, H_chamber, L, alpha, k_val
                    )                                             # (num_quads,)
                    # s² = (s_re² - s_im²) + i·(2·s_re·s_im).
                    s2_re = s_re * s_re - s_im * s_im
                    s2_im = 2.0 * s_re * s_im

                    # ----- Stiffness term: ∫ ∇p · ∇v dV -----
                    # Re/Im decouple in the stiffness (real coefficient).
                    u_grads = jnp.sum(
                        cell_sol[None, :, :, None]
                        * cell_shape_grads[:, :, None, :],
                        axis=1,
                    )                                            # (num_quads, 2, dim)
                    stiff_val = jnp.sum(
                        u_grads[:, None, :, :] * cell_v_grads_JxW,
                        axis=(0, -1),
                    )                                            # (num_nodes, 2)

                    # ----- Mass term: -∫ k²·s²·p·v dV -----
                    # u: (num_quads, 2).  s² · (p_re + i p_im):
                    #   Re part = s2_re·p_re − s2_im·p_im
                    #   Im part = s2_im·p_re + s2_re·p_im
                    u = jnp.sum(
                        cell_sol[None, :, :] * shape_vals[:, :, None],
                        axis=1,
                    )
                    u_re = u[:, 0]
                    u_im = u[:, 1]
                    sp_re = s2_re * u_re - s2_im * u_im
                    sp_im = s2_im * u_re + s2_re * u_im
                    mass_phys = -k2 * jnp.stack([sp_re, sp_im], axis=1)
                    mass_val = jnp.sum(
                        mass_phys[:, None, :]
                        * shape_vals[:, :, None]
                        * cell_JxW0[:, None, None],
                        axis=0,
                    )                                            # (num_nodes, 2)

                    val = stiff_val + mass_val                    # (num_nodes, 2)
                    return jax.flatten_util.ravel_pytree(val)[0]

                return universal_kernel

        def get_surface_maps(self):
            n_t = additional["n_transducers"]
            k_val = additional["k_val"]
            pml_on = additional["pml_enabled"]

            def make_transducer_map(i: int):
                # JAX-FEM convention: surface_map returns the flux g such
                # that ∂_n p = g. Per-transducer scaling is baked into
                # flux_quad by set_params (read from
                # internal_vars_surfaces[i]).
                def transducer_map(
                    u: jnp.ndarray,
                    point: jnp.ndarray,
                    flux_quad: jnp.ndarray,
                ) -> jnp.ndarray:
                    return flux_quad

                return transducer_map

            def sommerfeld_map(
                u: jnp.ndarray,
                point: jnp.ndarray,
            ) -> jnp.ndarray:
                # ∂_n p = i k p ⇔ flux = i k (Re + i Im) = -k·Im + i·k·Re.
                return jnp.stack(
                    [-k_val * u[1], k_val * u[0]],
                    axis=0,
                )

            def outer_no_op(
                u: jnp.ndarray,
                point: jnp.ndarray,
            ) -> jnp.ndarray:
                # PML path: outer surface carries no Robin term — the
                # PML absorbs and the wall is implicitly hard-reflective.
                # Return a zero flux of the right shape (vec=2).
                return jnp.zeros((2,), dtype=u.dtype)

            maps: List[Callable] = [make_transducer_map(i) for i in range(n_t)]
            maps.append(outer_no_op if pml_on else sommerfeld_map)
            return maps

        def set_params(self, params: jnp.ndarray) -> None:
            """Bind the per-transducer phase vector.

            Per-transducer flux is computed as
            ``±ρ₀·ω·v_amp·{−sin(φ), +cos(φ)}`` and broadcast over the
            face's quadrature points. The Sommerfeld surface is
            geometry-only; it carries an empty internal-vars tuple.

            This method MUST be implemented (override) for
            ``ad_wrapper(problem)`` to make ``params`` differentiable.
            Per-Phase-0 risk #1 in FEM_DISCOVERY_REPORT.
            """
            n_t = additional["n_transducers"]
            scale = (
                additional["rho0"]
                * additional["omega"]
                * additional["v_amp"]
            )
            phases = jnp.asarray(params, dtype=jnp.float64).reshape(n_t)

            cos_phi = jnp.cos(phases)
            sin_phi = jnp.sin(phases)

            # Per-transducer flux: (Re, Im) = scale·(-sin φ, +cos φ).
            re_flux = -scale * sin_phi
            im_flux = scale * cos_phi

            new_internal_vars_surfaces: List[List[Any]] = []
            for i in range(n_t):
                inds = self.boundary_inds_list[i]
                n_faces = inds.shape[0]
                n_face_quads = self.fes[0].num_face_quads
                # Same 2-vector flux at every face quadrature point on
                # this transducer disk.
                flux_2 = jnp.stack(
                    [re_flux[i], im_flux[i]],
                    axis=0,
                )
                flux_quad = jnp.broadcast_to(
                    flux_2, (n_faces, n_face_quads, 2)
                )
                new_internal_vars_surfaces.append([flux_quad])
            # Sommerfeld surface: no parameters at quadrature points.
            new_internal_vars_surfaces.append([])
            self.internal_vars_surfaces = new_internal_vars_surfaces

    problem = HelmholtzProblem(
        mesh=mesh,
        vec=2,
        dim=3,
        ele_type="TET4",
        location_fns=location_fns,
    )
    return problem


# ---------------------------------------------------------------------------
# Solver wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HelmholtzSolution:
    """Solved acoustic field on the chamber mesh.

    Attributes:
        re: ``(num_nodes,)`` real component of ``p`` at each mesh node.
        im: ``(num_nodes,)`` imaginary component of ``p``.
        mesh_coords: ``(num_nodes, 3)`` nodal positions in metres.
        wall_time_s: solve wall-clock time (seconds).
        n_dofs: total degrees of freedom in the linear system.
        k: wavenumber used in the solve (``2π·f/c0``).
    """

    re: jnp.ndarray
    im: jnp.ndarray
    mesh_coords: jnp.ndarray
    wall_time_s: float
    n_dofs: int
    k: float

    def magnitude(self) -> jnp.ndarray:
        """Return ``|p|`` per node."""
        return jnp.sqrt(self.re ** 2 + self.im ** 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "re": self.re,
            "im": self.im,
            "mesh_coords": self.mesh_coords,
            "wall_time_s": self.wall_time_s,
            "n_dofs": int(self.n_dofs),
            "k": float(self.k),
        }


def _resolve_solver_options(
    user_options: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Pick the linear-solver options for the Helmholtz solve.

    Selection rule:

      * ``user_options`` is ``None`` ⇒ default ``umfpack_solver`` (the
        bullet-proof CPU path; used by all tests).
      * ``user_options`` requests ``petsc_solver`` and ``petsc4py`` is
        importable ⇒ pass through (production CUDA-cluster MUMPS path
        — see ``jax_fem.solver.petsc_solve``; LU factor solver
        defaults to MUMPS when ``ksp_type='preonly'`` or ``'tfqmr'``).
      * ``user_options`` requests ``petsc_solver`` but ``petsc4py`` is
        not importable ⇒ fall back to ``umfpack_solver`` with a
        ``DeprecationWarning``-style log line, **never** silently
        fail-open.
      * Any other ``user_options`` ⇒ pass through verbatim
        (``jax_solver``, ``amgx_solver``, ``custom_solver``).
    """
    if user_options is None:
        return dict(DEFAULT_SOLVER_OPTIONS)

    if "petsc_solver" in user_options:
        try:
            import petsc4py  # noqa: F401  — availability probe only
        except ImportError:
            import warnings

            warnings.warn(
                "PETSc-MUMPS solver requested but petsc4py is not "
                "importable; falling back to scipy spsolve. Install "
                "with `pip install petsc4py mumps` for the production "
                "CUDA-deploy path.",
                stacklevel=2,
            )
            return dict(DEFAULT_SOLVER_OPTIONS)

    return dict(user_options)


def solve_helmholtz(
    mesh_path: Path,
    surface_tags: Dict[str, int],
    env_config: EnvironmentConfig,
    transducer_config: TransducerConfig,
    phases: jnp.ndarray,
    array_config: Optional[ArrayConfig] = None,
    velocity_amp: float = DEFAULT_PISTON_VELOCITY_AMP_M_S,
    pml_width: float = DEFAULT_PML_WIDTH_M,
    pml_alpha: float = DEFAULT_PML_ALPHA,
    solver_options: Optional[Dict[str, Any]] = None,
) -> HelmholtzSolution:
    """Solve the Helmholtz forward problem for a 120-vector of phases.

    Args:
        mesh_path: Path to the persisted L1 chamber ``.msh`` file
            produced by :func:`build_l1_chamber_mesh`.
        surface_tags: Surface-tag dict (``info["surface_tags"]``) returned
            alongside the mesh. Currently informational; the
            Helmholtz BCs are applied through geometric location_fns
            so the function is robust to mesh re-builds.
        env_config: Environment (medium, temperature, transducer, array).
            Determines ``k = 2π f / c0`` and ``ρ₀, ω``.
        transducer_config: Source of transducer face radius.
        phases: ``(120,)`` jax array of phase values (radians).
        array_config: Optional override for the array geometry. Defaults
            to ``env_config.array``.
        velocity_amp: Piston velocity amplitude (m/s) — held fixed
            across all transducers in v0.
        pml_width: width of the Bermúdez PML layer (metres). ``0.0``
            disables PML and selects the Phase 2.0 Sommerfeld path.
        pml_alpha: PML decay strength (dimensionless). Default ``1.0``
            is canonical for time-harmonic Bermúdez 2007.
        solver_options: Optional linear-solver options dict in the
            JAX-FEM format (``{"umfpack_solver": {}}``,
            ``{"petsc_solver": {"ksp_type": "preonly", "pc_type":
            "lu"}}``, etc.). Default: ``umfpack_solver``. PETSc path
            falls back to spsolve (via umfpack_solver) with a warning
            if ``petsc4py`` is unavailable.

    Returns:
        :class:`HelmholtzSolution` with ``re``, ``im``, mesh coordinates
        and metadata.
    """
    Problem, Mesh, ad_wrapper, solver = _import_jax_fem()

    array = array_config if array_config is not None else env_config.array
    geom = resolve_chamber_geometry(array, transducer_config)

    if phases.shape != (geom.total_transducers,):
        raise ValueError(
            f"phases shape mismatch: expected ({geom.total_transducers},), "
            f"got {tuple(phases.shape)}"
        )

    mesh, points = load_jax_fem_mesh(Path(mesh_path))
    problem = make_helmholtz_problem(
        mesh=mesh,
        geom=geom,
        env_config=env_config,
        transducer_config=transducer_config,
        velocity_amp=velocity_amp,
        pml_width=pml_width,
        pml_alpha=pml_alpha,
    )

    opts = _resolve_solver_options(solver_options)
    fwd = ad_wrapper(
        problem,
        solver_options=dict(opts),
        adjoint_solver_options=dict(opts),
    )

    t0 = time.time()
    sol_list = fwd(phases)
    wall_time = time.time() - t0

    sol = sol_list[0]  # (num_nodes, 2)
    re = sol[:, 0]
    im = sol[:, 1]
    n_dofs = int(np.prod(sol.shape))

    return HelmholtzSolution(
        re=re,
        im=im,
        mesh_coords=jnp.asarray(points),
        wall_time_s=float(wall_time),
        n_dofs=n_dofs,
        k=float(env_config.wavenumber),
    )


def compute_field(
    problem: "Any",
    phases: jnp.ndarray,
    solver_options: Optional[Dict[str, Any]] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Thin convenience wrapper: ``(Re, Im)`` field on mesh nodes.

    Calls ``ad_wrapper(problem)(phases)`` under the hood and unpacks the
    2-vector solution. Useful for inverse loops that already hold the
    constructed problem instance and want to call repeatedly without
    re-instantiating ``ad_wrapper``.
    """
    _, _, ad_wrapper, _ = _import_jax_fem()
    opts = _resolve_solver_options(solver_options)
    fwd = ad_wrapper(
        problem,
        solver_options=dict(opts),
        adjoint_solver_options=dict(opts),
    )
    sol_list = fwd(phases)
    sol = sol_list[0]
    return sol[:, 0], sol[:, 1]


__all__ = [
    "DEFAULT_PISTON_VELOCITY_AMP_M_S",
    "DEFAULT_PML_ALPHA",
    "DEFAULT_PML_WIDTH_M",
    "DEFAULT_SOLVER_OPTIONS",
    "RECOMMENDED_PML_WIDTH_M",
    "HelmholtzSolution",
    "compute_field",
    "make_helmholtz_problem",
    "solve_helmholtz",
]
