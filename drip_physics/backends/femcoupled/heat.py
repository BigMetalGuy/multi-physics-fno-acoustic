"""HeatTransientProblem v0 — implicit-Euler transient heat equation.

Phase 3 of the FEM-coupled-physics v1 build (FEM_V2_DESIGN.md §3.2/§3.6/§3.7).
This module ships the v0 transient heat forward model on the same JAX-FEM
substrate as :mod:`helmholtz`:

  * :class:`HeatTransientProblem` — JAX-FEM ``Problem`` subclass with
    ``vec=1`` (scalar temperature ``T``) and ``dim=3``.
  * Bulk weak form (implicit Euler):
        ``∫ (ρcₚ/Δt)(T - T_old) v dV + ∫ k ∇T·∇v dV
            - ∫ Q v dV - ∫ u_stream·∇T v dV = 0``
    where ``Q`` is the volumetric droplet source (Gaussian per droplet)
    and ``u_stream`` is the optional acoustic-streaming advection velocity
    field (zero in v0; non-zero is the Phase 4 hook for two-way coupling
    per FEM_V2_DESIGN §3.4).
  * Surface contributions per the corrected BC split (FEM_V2_DESIGN §3.7):
      - ``substrate``: Dirichlet ``T = T_substrate`` (≈ 573 K, heated bed
        memory). Applied through JAX-FEM's ``dirichlet_bc_info``.
      - ``chamber_walls``: Robin ``−k ∇T·n = h(T − T_∞)`` (Al outer wall
        with engineered cooling channel; ``h ≈ 50 W/m²K``,
        ``T_∞ ≈ 290 K``). Applied as a Neumann surface map.
      - ``transducer_faces``: thermally insulating Neumann
        (``∂T/∂n = 0``). Transducers are internally cooled and dump heat
        through the outer wall, not into the chamber. Implemented by
        simply not registering a surface map for the transducer
        location_fns (a missing surface_map ⇒ zero flux).

Phase change (latent heat / Voller-Cross enthalpy) is **deferred to
Phase 3.1**; v0 is sensible heat only. ``k``, ``ρ``, ``cₚ`` are constant.

This module uses :func:`drip_physics.backends.femcoupled.mesh.load_jax_fem_mesh`
to filter Gmsh orphan nodes.

Example
-------

::

    from drip_physics.backends.femcoupled import build_l1_chamber_mesh
    from drip_physics.backends.femcoupled.heat import solve_heat_trajectory
    from drip_physics.config import L1_ARRAY, MA40S4S_40KHZ, default_environment

    path, info = build_l1_chamber_mesh(L1_ARRAY, MA40S4S_40KHZ)
    droplet_traj = [
        {"centers": jnp.array([[0.0, 0.0, 0.40 - 0.01 * i]]),
         "amplitudes": jnp.array([1.0])}
        for i in range(50)
    ]
    result = solve_heat_trajectory(
        mesh_path=path,
        surface_tags=info["surface_tags"],
        env_config=default_environment(),
        droplet_trajectory=droplet_traj,
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from drip_physics.config import (
    EnvironmentConfig,
    MaterialConfig,
)

from .geometry import ChamberGeometry, resolve_chamber_geometry, transducer_centers
from .helmholtz import _import_jax_fem
from .mesh import load_jax_fem_mesh


# ---------------------------------------------------------------------------
# Default thermal parameters (v0; document deviations against L1 hardware)
# ---------------------------------------------------------------------------

# Substrate (heated bed) — project memory: 300 °C heated bed at L1.
T_SUBSTRATE_K: float = 573.15

# Crucible inlet temperature (project memory: 850 °C, L1 crucible setpoint).
T_CRUCIBLE_K: float = 1123.15

# Chamber-air initial temperature (FEM_V2_DESIGN §3.7: lid-as-quasi-isothermal
# at chamber-air temperature is the v0 default).
T_CHAMBER_INIT_K: float = 473.15  # 200 °C interior chamber

# Outer-wall coolant loop setpoint and film coefficient. Numerical values are
# placeholders pending mechanical-design input (FEM_V2_DESIGN §3.7).
H_CHANNEL_W_M2K: float = 50.0
T_COOLANT_K: float = 290.0

# Default droplet heat-source sigma — FEM_V2_DESIGN §3.7 / brief specifies
# 1.5 × droplet radius. L1 droplet diameter is 600 μm ⇒ radius 300 μm,
# so sigma = 450 μm.
DEFAULT_DROPLET_SIGMA_M: float = 450e-6

# Default thermal time step (FEM_V2_DESIGN §3.6).
DEFAULT_THERMAL_DT_S: float = 0.01  # 10 ms

# Default solver options. Heat is positive-definite for implicit Euler so
# JAX BiCGSTAB should converge cleanly; we still default to the direct LU
# path for parity with Phase 2's Helmholtz solver wrapper. Setting
# ``USE_DIRECT_SOLVER = True`` gives spsolve; ``False`` selects JAX bicgstab
# (the JAX-FEM default in solver.py:172). Empirically (test runs) BiCGSTAB
# converges in single-digit iterations for this operator.
USE_DIRECT_SOLVER: bool = True
# JAX-FEM dispatcher key for scipy spsolve (direct sparse) — see helmholtz.py
# DEFAULT_SOLVER_OPTIONS note. Prior ``spsolve_solver`` key silently fell
# through to BiCGSTAB and was happening to work for the well-conditioned
# parabolic heat operator, but is still the wrong dispatcher key.
DIRECT_SOLVER_OPTIONS: Dict[str, Any] = {"umfpack_solver": {}}
ITERATIVE_SOLVER_OPTIONS: Dict[str, Any] = {"jax_solver": {}}


def _default_solver_options() -> Dict[str, Any]:
    return (
        dict(DIRECT_SOLVER_OPTIONS)
        if USE_DIRECT_SOLVER
        else dict(ITERATIVE_SOLVER_OPTIONS)
    )


# Geometric tolerance for surface predicate selection (mirrors helmholtz.py).
SURFACE_TOL_M: float = 5e-4


# ---------------------------------------------------------------------------
# Material extraction with explicit defaults
# ---------------------------------------------------------------------------


def _resolve_thermal_properties(
    env_config: EnvironmentConfig,
) -> Tuple[float, float, float]:
    """Pull (k, ρ, cₚ) for the chamber medium out of ``env_config``.

    Returns:
        ``(k, rho, cp)`` in SI units (W/(m·K), kg/m³, J/(kg·K)).

    Raises:
        ValueError: if the medium config does not provide positive
            thermal_conductivity, specific_heat, or density. Use a
            temperature-appropriate profile such as ``AIR_200C`` for
            chamber-temperature simulations.
    """
    medium: MaterialConfig = env_config.medium
    if medium.thermal_conductivity <= 0.0:
        raise ValueError(
            f"medium '{medium.name}' has no thermal_conductivity; "
            f"thermal solver requires k > 0. Use a profile that defines "
            f"thermal_conductivity (e.g. AIR_200C for ~200 °C chamber)."
        )
    if medium.specific_heat <= 0.0:
        raise ValueError(
            f"medium '{medium.name}' has no specific_heat; "
            f"thermal solver requires cp > 0. Use a profile that defines "
            f"specific_heat (e.g. AIR_200C for ~200 °C chamber)."
        )
    if medium.density <= 0.0:
        raise ValueError(
            f"medium '{medium.name}' has non-positive density "
            f"({medium.density})."
        )
    return (
        float(medium.thermal_conductivity),
        float(medium.density),
        float(medium.specific_heat),
    )


# ---------------------------------------------------------------------------
# Volumetric heat source
# ---------------------------------------------------------------------------


def gaussian_heat_source(
    nodes: jnp.ndarray,
    centers: jnp.ndarray,
    amplitudes: jnp.ndarray,
    sigma: float = DEFAULT_DROPLET_SIGMA_M,
) -> jnp.ndarray:
    """Gaussian volumetric heat source ``Q(x)`` evaluated at mesh nodes.

    For each droplet at ``centers[i]`` with amplitude ``amplitudes[i]`` (W),
    contributes a Gaussian profile centred at the droplet position::

        Q_i(x) = amplitudes[i] * exp(-||x - centers[i]||² / (2 σ²))
                 / ((2π)^{3/2} σ³)

    The 3D-Gaussian normalisation makes ``∫ Q_i dV ≈ amplitudes[i]`` for
    σ ≪ chamber size.

    Args:
        nodes: ``(n_nodes, 3)`` mesh node coordinates (metres).
        centers: ``(n_droplets, 3)`` droplet centre positions.
        amplitudes: ``(n_droplets,)`` heat amplitudes (Watts).
        sigma: Gaussian std-dev (metres). Default 450 μm = 1.5 × 300 μm
            droplet radius (FEM_V2_DESIGN §3.7).

    Returns:
        ``(n_nodes,)`` jax array — total volumetric heat source on each node.
    """
    nodes = jnp.asarray(nodes, dtype=jnp.float64)
    centers = jnp.asarray(centers, dtype=jnp.float64).reshape(-1, 3)
    amplitudes = jnp.asarray(amplitudes, dtype=jnp.float64).reshape(-1)
    sigma = float(sigma)

    if centers.shape[0] != amplitudes.shape[0]:
        raise ValueError(
            f"centers/amplitudes shape mismatch: "
            f"{centers.shape[0]} != {amplitudes.shape[0]}"
        )

    # Pairwise squared distances: (n_nodes, n_droplets).
    diff = nodes[:, None, :] - centers[None, :, :]
    d2 = jnp.sum(diff * diff, axis=-1)
    # Gaussian profile with 3D normalisation.
    norm = (2.0 * jnp.pi * sigma * sigma) ** 1.5
    profile = jnp.exp(-d2 / (2.0 * sigma * sigma)) / norm
    # Sum contributions.
    return jnp.sum(profile * amplitudes[None, :], axis=-1)


# ---------------------------------------------------------------------------
# Location functions for the four surface groups
# ---------------------------------------------------------------------------


def _make_substrate_location_fn(
    geom: ChamberGeometry,
    tol: float = SURFACE_TOL_M,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Predicate selecting nodes on the substrate (z=0)."""

    def loc_fn(point: jnp.ndarray) -> jnp.ndarray:
        return jnp.isclose(point[2], 0.0, atol=tol)

    return loc_fn


def _make_chamber_walls_location_fn(
    geom: ChamberGeometry,
    transducer_centers_arr: np.ndarray,
    face_radius: float,
    tol: float = SURFACE_TOL_M,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Predicate: outer cylindrical wall (between transducers) + lid (z=h).

    Transducer disks themselves are excluded — they carry the insulating
    Neumann ``∂T/∂n = 0`` BC (no surface map registered).
    """
    r = float(geom.cylinder_radius)
    h = float(geom.height)
    a = float(face_radius)
    centers_jax = jnp.asarray(transducer_centers_arr, dtype=jnp.float64)

    def loc_fn(point: jnp.ndarray) -> jnp.ndarray:
        d = jnp.sqrt(
            jnp.min(
                (point[0] - centers_jax[:, 0]) ** 2
                + (point[1] - centers_jax[:, 1]) ** 2
                + (point[2] - centers_jax[:, 2]) ** 2
            )
        )
        in_any_disk = d <= (a + tol)

        rho = jnp.sqrt(point[0] ** 2 + point[1] ** 2)
        on_lid = jnp.isclose(point[2], h, atol=tol)
        on_side_wall = jnp.isclose(rho, r, atol=tol)
        return on_lid | (on_side_wall & ~in_any_disk)

    return loc_fn


# ---------------------------------------------------------------------------
# HeatTransientProblem
# ---------------------------------------------------------------------------


def make_heat_problem(
    mesh: "Any",
    geom: ChamberGeometry,
    env_config: EnvironmentConfig,
    *,
    h_channel: float = H_CHANNEL_W_M2K,
    t_coolant: float = T_COOLANT_K,
    t_substrate: float = T_SUBSTRATE_K,
    droplet_sigma: float = DEFAULT_DROPLET_SIGMA_M,
    dt: float = DEFAULT_THERMAL_DT_S,
) -> "HeatTransientProblem":
    """Construct a :class:`HeatTransientProblem` on the given JAX-FEM mesh.

    The factory builds the location_fns for the chamber walls (Robin BC
    surface), wires the Dirichlet substrate BC, and instantiates the
    ``Problem`` subclass. The instance is initialized with a no-droplet
    parameter dict via :meth:`HeatTransientProblem.set_params`; callers
    must call ``set_params`` again per-timestep before each solve.
    """
    Problem, _, _, _ = _import_jax_fem()

    centers = transducer_centers(geom)
    a = geom.transducer_face_radius

    substrate_fn = _make_substrate_location_fn(geom)
    chamber_walls_fn = _make_chamber_walls_location_fn(geom, centers, a)

    # JAX-FEM expects two parallel lists: ``location_fns`` for surface
    # integrals (Neumann/Robin), and ``dirichlet_bc_info`` for essential
    # BCs. The chamber_walls surface is a Robin term ⇒ enters via
    # location_fns + a get_surface_maps entry. The substrate is Dirichlet
    # ⇒ enters via dirichlet_bc_info.
    location_fns: List[Callable[[jnp.ndarray], jnp.ndarray]] = [
        chamber_walls_fn,
    ]

    def substrate_dirichlet_value(point: jnp.ndarray) -> float:  # noqa: D401
        return float(t_substrate)

    dirichlet_bc_info = [
        [substrate_fn],          # location predicates
        [0],                     # which vec component (vec=1 ⇒ index 0)
        [substrate_dirichlet_value],  # value functions
    ]

    k, rho, cp = _resolve_thermal_properties(env_config)

    additional = {
        "k": k,
        "rho": rho,
        "cp": cp,
        "h_channel": float(h_channel),
        "t_coolant": float(t_coolant),
        "t_substrate": float(t_substrate),
        "droplet_sigma": float(droplet_sigma),
        "dt": float(dt),
    }

    class HeatTransientProblem(Problem):
        """Transient heat equation with implicit-Euler time stepping.

        Implements:
          * Bulk conduction: ``∫ k ∇T·∇v dV`` via :meth:`get_tensor_map`.
          * Mass term + droplet source + advection by streaming velocity::
                ``∫ [ρcₚ(T − T_old)/Δt − Q − u_stream·∇T] v dV``
            via :meth:`get_mass_map`. The advection is folded into the
            mass-map kernel because the kernel has access to ``u`` and
            ``T_old`` at quad points; ``∇T`` at quadrature points is
            approximated by ``(T - T_old)/Δt · 0`` not directly. The
            advection term uses ``u_stream`` evaluated at the quad point
            and the *previous*-step temperature gradient stored in
            internal_vars (semi-implicit advection: lagged ∇T multiplies
            current u).
          * Robin chamber-wall flux: ``q = h(T_coolant - T)`` (heat into
            domain when coolant hotter). Implemented in
            :meth:`get_surface_maps`.

        Because v0 advection is fully zero by default and the flagship
        validation case is conduction-only, the bulk advection term is
        registered but contributes 0 unless ``streaming_velocity`` is
        non-zero.
        """

        def get_tensor_map(self):
            k_val = additional["k"]

            def tensor_fn(
                u_grad: jnp.ndarray,
                T_old: jnp.ndarray,
                Q_quad: jnp.ndarray,
                adv_quad: jnp.ndarray,
            ) -> jnp.ndarray:
                # JAX-FEM passes the same ``internal_vars`` to both
                # tensor_map and mass_map positionally; tensor_map only
                # consumes u_grad.
                del T_old, Q_quad, adv_quad
                return k_val * u_grad

            return tensor_fn

        def get_mass_map(self):
            rho_val = additional["rho"]
            cp_val = additional["cp"]
            dt_val = additional["dt"]

            def mass_fn(
                T: jnp.ndarray,
                x: jnp.ndarray,
                T_old: jnp.ndarray,
                Q_quad: jnp.ndarray,
                adv_quad: jnp.ndarray,
            ) -> jnp.ndarray:
                inertia = rho_val * cp_val * (T - T_old) / dt_val
                # Q is a *source* (positive into the domain) ⇒ subtract
                # from the residual; same sign convention for advection.
                return inertia - jnp.array([Q_quad]) - jnp.array([adv_quad])

            return mass_fn

        def get_surface_maps(self):
            h = additional["h_channel"]
            t_inf = additional["t_coolant"]

            def chamber_walls_robin(
                u: jnp.ndarray,
                point: jnp.ndarray,
                T_old_quad: jnp.ndarray,
            ) -> jnp.ndarray:
                # JAX-FEM convention (thermal_mechanical/example.py:54):
                # surface_map returns -q where q is the heat flux INTO
                # the domain, so Robin q = h*(T_inf - T) gives flux entry
                # h*(T - T_inf).
                #
                # Use the *current* T (u[0]) rather than the lagged
                # T_old_quad to keep the Robin BC implicit-stable on a
                # low-thermal-mass air medium where the lagged form is
                # only conditionally stable.
                del T_old_quad
                q = h * (t_inf - u[0])
                return -jnp.array([q])

            return [chamber_walls_robin]

        def set_params(self, params: Dict[str, Any]) -> None:
            """Bind the per-timestep parameter dict.

            This method MUST be implemented (override) for
            ``ad_wrapper(problem)`` to make params differentiable per
            FEM_DISCOVERY_REPORT risk #1.

            Args:
                params: dict with keys
                    * ``T_prev``: ``(n_nodes,)`` previous-step T (float64).
                    * ``Q_droplet_centers``: ``(n_droplets, 3)`` positions.
                    * ``Q_droplet_amplitudes``: ``(n_droplets,)`` Watts.
                    * ``streaming_velocity`` (optional):
                      ``(n_nodes, 3)`` advection velocity in m/s.
                      Defaults to zero.
            """
            T_prev = jnp.asarray(params["T_prev"], dtype=jnp.float64)
            centers_arr = jnp.asarray(
                params["Q_droplet_centers"], dtype=jnp.float64
            ).reshape(-1, 3)
            amps_arr = jnp.asarray(
                params["Q_droplet_amplitudes"], dtype=jnp.float64
            ).reshape(-1)
            sigma = additional["droplet_sigma"]

            Q_nodes = gaussian_heat_source(
                self.fes[0].points, centers_arr, amps_arr, sigma=sigma
            )

            stream = params.get("streaming_velocity", None)
            if stream is None:
                stream_arr = jnp.zeros_like(self.fes[0].points)
            else:
                stream_arr = jnp.asarray(stream, dtype=jnp.float64).reshape(
                    self.fes[0].points.shape
                )

            # u_stream · ∇T_prev at each cell quadrature point via the
            # interpolated-gradient approach: ∇T_prev = sum_n T_prev[cell_n]
            # * shape_grads[c, q, n, :]. Computing it here (rather than in
            # the kernel) lets us reuse the FE shape data once per step.
            cells = self.fes[0].cells
            shape_grads_phys = self.fes[0].shape_grads
            shape_vals = self.fes[0].shape_vals

            T_prev_cells = T_prev[cells]
            stream_cells = stream_arr[cells]

            grad_T_prev_q = jnp.einsum(
                "cn,cqnd->cqd", T_prev_cells, shape_grads_phys
            )
            u_stream_q = jnp.einsum("cnd,qn->cqd", stream_cells, shape_vals)
            adv_q = jnp.sum(u_stream_q * grad_T_prev_q, axis=-1)

            # JAX-FEM's get_mass_kernel expects internal vars shaped
            # (num_cells, num_quads, ...) and unpacks each entry positionally
            # into the mass_map. T_old_quad mirrors the thermal_mechanical
            # convention with a trailing vec=1 axis.
            T_prev_q = jnp.einsum("cn,qn->cq", T_prev_cells, shape_vals)
            T_prev_q_v = T_prev_q[..., None]

            Q_cells = Q_nodes[cells]
            Q_q = jnp.einsum("cn,qn->cq", Q_cells, shape_vals)

            self.internal_vars = [T_prev_q_v, Q_q, adv_q]

            # Surface internal var: T_prev at face quad points on
            # chamber_walls (boundary index 0).
            T_old_face = self.fes[0].convert_from_dof_to_face_quad(
                T_prev[:, None], self.boundary_inds_list[0]
            )
            self.internal_vars_surfaces = [[T_old_face]]

    problem = HeatTransientProblem(
        mesh=mesh,
        vec=1,
        dim=3,
        ele_type="TET4",
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
    )

    # Initialize with a uniform T_prev and no droplets so ad_wrapper has
    # something to introspect on first construction.
    problem.set_params({
        "T_prev": jnp.full(
            (problem.fes[0].num_total_nodes,), float(t_substrate),
            dtype=jnp.float64,
        ),
        "Q_droplet_centers": jnp.zeros((1, 3), dtype=jnp.float64),
        "Q_droplet_amplitudes": jnp.zeros((1,), dtype=jnp.float64),
        "streaming_velocity": None,
    })
    return problem


# ---------------------------------------------------------------------------
# Solver wrapper (single step) and trajectory loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeatStepSolution:
    """Single thermal step result.

    Attributes:
        T: ``(n_nodes,)`` temperature at the new step (Kelvin).
        wall_time_s: wall-clock for the linear solve.
        n_dofs: total degrees of freedom.
    """

    T: jnp.ndarray
    wall_time_s: float
    n_dofs: int


def solve_heat_step(
    problem: "Any",
    T_prev: jnp.ndarray,
    droplet_centers: jnp.ndarray,
    droplet_amplitudes: jnp.ndarray,
    streaming_velocity: Optional[jnp.ndarray] = None,
) -> HeatStepSolution:
    """Advance the temperature field by one thermal timestep.

    Reuses the constructed ``problem`` instance; binds new params via
    :meth:`HeatTransientProblem.set_params` and re-solves.
    """
    _, _, _, solver = _import_jax_fem()

    problem.set_params({
        "T_prev": T_prev,
        "Q_droplet_centers": droplet_centers,
        "Q_droplet_amplitudes": droplet_amplitudes,
        "streaming_velocity": streaming_velocity,
    })

    t0 = time.time()
    sol_list = solver(problem, solver_options=_default_solver_options())
    wall = time.time() - t0
    T_new = sol_list[0][:, 0]
    return HeatStepSolution(
        T=T_new,
        wall_time_s=float(wall),
        n_dofs=int(np.prod(sol_list[0].shape)),
    )


@dataclass(frozen=True)
class HeatTrajectorySolution:
    """Full trajectory result: T history + diagnostics."""

    T_history: jnp.ndarray            # (n_steps + 1, n_nodes)
    mesh_coords: jnp.ndarray          # (n_nodes, 3)
    wall_time_s: float
    per_step_wall_time_s: List[float]
    n_dofs: int


def solve_heat_trajectory(
    mesh_path: Path,
    surface_tags: Dict[str, int],
    env_config: EnvironmentConfig,
    droplet_trajectory: List[Dict[str, jnp.ndarray]],
    *,
    n_steps: Optional[int] = None,
    dt: float = DEFAULT_THERMAL_DT_S,
    t_initial: float = T_CHAMBER_INIT_K,
    t_substrate: float = T_SUBSTRATE_K,
    h_channel: float = H_CHANNEL_W_M2K,
    t_coolant: float = T_COOLANT_K,
    droplet_sigma: float = DEFAULT_DROPLET_SIGMA_M,
    streaming_velocity: Optional[jnp.ndarray] = None,
) -> HeatTrajectorySolution:
    """Run the full transient heat trajectory.

    Args:
        mesh_path: Persisted ``.msh`` produced by ``build_l1_chamber_mesh``.
        surface_tags: Surface-tag dict (informational; BCs use
            geometric location_fns).
        env_config: Environment for thermal property extraction.
        droplet_trajectory: List of per-step dicts with keys
            ``centers`` (n_droplets, 3) and ``amplitudes`` (n_droplets,).
            Length determines the number of timesteps unless ``n_steps``
            is set explicitly.
        n_steps: Override for number of steps. Defaults to len(trajectory).
        dt: Thermal timestep (seconds). Default 10 ms (FEM_V2_DESIGN §3.6).
        t_initial: Initial uniform temperature (K). Default chamber air
            temp 473 K.
        t_substrate, h_channel, t_coolant: BC parameters.
        droplet_sigma: Gaussian sigma for the droplet source profile (m).
        streaming_velocity: Optional ``(n_nodes, 3)`` advection field.
            Held constant across steps in v0.

    Returns:
        :class:`HeatTrajectorySolution` with ``T_history`` (including the
        initial state at index 0), ``mesh_coords``, and timing.
    """
    _, _, _, _ = _import_jax_fem()  # ensure JAX-FEM importable

    mesh, points = load_jax_fem_mesh(Path(mesh_path))
    geom = resolve_chamber_geometry(env_config.array, env_config.transducer)
    problem = make_heat_problem(
        mesh=mesh,
        geom=geom,
        env_config=env_config,
        h_channel=h_channel,
        t_coolant=t_coolant,
        t_substrate=t_substrate,
        droplet_sigma=droplet_sigma,
        dt=dt,
    )

    n_nodes = problem.fes[0].num_total_nodes
    n = n_steps if n_steps is not None else len(droplet_trajectory)

    T_curr = jnp.full((n_nodes,), float(t_initial), dtype=jnp.float64)
    history = [T_curr]
    per_step: List[float] = []
    t0 = time.time()
    for i in range(n):
        if i < len(droplet_trajectory):
            entry = droplet_trajectory[i]
            centers = jnp.asarray(entry["centers"], dtype=jnp.float64)
            amps = jnp.asarray(entry["amplitudes"], dtype=jnp.float64)
        else:
            centers = jnp.zeros((1, 3), dtype=jnp.float64)
            amps = jnp.zeros((1,), dtype=jnp.float64)

        sol = solve_heat_step(
            problem=problem,
            T_prev=T_curr,
            droplet_centers=centers,
            droplet_amplitudes=amps,
            streaming_velocity=streaming_velocity,
        )
        T_curr = sol.T
        history.append(T_curr)
        per_step.append(sol.wall_time_s)
    wall = time.time() - t0

    T_history = jnp.stack(history, axis=0)
    return HeatTrajectorySolution(
        T_history=T_history,
        mesh_coords=jnp.asarray(points),
        wall_time_s=float(wall),
        per_step_wall_time_s=per_step,
        n_dofs=int(n_nodes),
    )


__all__ = [
    "DEFAULT_DROPLET_SIGMA_M",
    "DEFAULT_THERMAL_DT_S",
    "H_CHANNEL_W_M2K",
    "HeatStepSolution",
    "HeatTrajectorySolution",
    "T_CHAMBER_INIT_K",
    "T_COOLANT_K",
    "T_CRUCIBLE_K",
    "T_SUBSTRATE_K",
    "gaussian_heat_source",
    "make_heat_problem",
    "solve_heat_step",
    "solve_heat_trajectory",
]

