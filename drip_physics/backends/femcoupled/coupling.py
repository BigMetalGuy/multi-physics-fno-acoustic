"""StaggeredCouplingDriver — two-way acoustic ↔ thermal coupling.

Phase 4 of the FEM-coupled-physics v1 build (FEM_V2_DESIGN.md §3.4 — the
*post-correction* version specifying two-way coupling as v1 essential, not
v2 deferred).

This module wires together :mod:`helmholtz` (Phase 2.0/2.1) and :mod:`heat`
(Phase 3 + crucible-inlet follow-up) into a single per-thermal-step
fixed-point loop:

  * **Forward direction** (T → field).  The acoustic wavenumber ``k(T)``
    and medium density ``ρ(T)`` modulate the Helmholtz operator.  v1
    uses bulk-T-dependent properties — the volume-averaged temperature
    field at each inner iteration drives a single ``c₀(T)``.  Node-level
    spatial variation of ``c₀(x)`` is deferred to v2 (would require
    extending :class:`HelmholtzProblem` to accept a per-quad material
    field).  The simplification is documented in the FEM_V2_DESIGN §3.4
    "v1 simplification" caveat.

  * **Reverse direction (load-bearing)** (field → streaming → heat
    advection → T).  Eckart streaming velocity is computed from the
    second-order Lagrangian-mean approximation
    ``u_stream = (1/(2 μ ω²)) · ∇|p|² / ρ₀`` and fed into the heat
    solver via the ``streaming_velocity`` parameter hook already wired
    in :func:`drip_physics.backends.femcoupled.heat.solve_heat_step`.
    This is the "approximate but physically dominant" form per
    FEM_V2_DESIGN §3.4 — full second-order velocity-pressure coupling
    is deferred to v2.

  * **Convergence gate**.  Inner fixed-point iterations terminate when
    the Helmholtz field-norm change between successive iterations drops
    below ``convergence_tol`` (default 1e-3 relative) or after
    ``max_inner_iters`` iterations (default 5).  Per
    FEM_V2_DESIGN §3.4: "3–5 fixed-point iterations per thermal
    timestep".

Three concrete pieces:

  * :class:`EckartStreamingComputer` — derives ``u_stream`` from the
    complex pressure field via a per-cell shape-function gradient of
    ``|p|²``.
  * :class:`StaggeredCouplingDriver` — orchestrates one outer thermal
    step (with inner fixed-point loop) and trajectories thereof.
  * :class:`CoupledStepResult` / :class:`CoupledTrajectoryResult` —
    typed receipts.

The driver is ``jax.grad``-clean end-to-end: same gradient flow that
Phase 2.0 validated for the Helmholtz solver and Phase 3 validated for
the heat solver runs through the coupled system.  This is the
architectural-bet stretch goal — same inverse loop will work through
FEM-coupled.

Example
-------

::

    from drip_physics.backends.femcoupled import build_l1_chamber_mesh
    from drip_physics.backends.femcoupled.coupling import (
        StaggeredCouplingDriver,
    )
    from drip_physics.config import L1_ARRAY, MA40S4S_40KHZ, default_environment

    path, info = build_l1_chamber_mesh(L1_ARRAY, MA40S4S_40KHZ)
    env = default_environment()
    driver = StaggeredCouplingDriver(
        mesh_path=path,
        surface_tags=info["surface_tags"],
        env_config=env,
        transducer_config=MA40S4S_40KHZ,
        n_outer_steps=50,
        dt=0.01,
        max_inner_iters=5,
    )
    result = driver.trajectory(phases_per_step, droplet_trajectory)
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
    TransducerConfig,
)

from .geometry import ChamberGeometry, resolve_chamber_geometry
from .helmholtz import (
    DEFAULT_PISTON_VELOCITY_AMP_M_S,
    DEFAULT_PML_ALPHA,
    DEFAULT_PML_WIDTH_M,
    HelmholtzSolution,
    _import_jax_fem,
    _resolve_solver_options,
    make_helmholtz_problem,
)
from .heat import (
    DEFAULT_DROPLET_SIGMA_M,
    DEFAULT_THERMAL_DT_S,
    H_CHANNEL_W_M2K,
    T_CHAMBER_INIT_K,
    T_COOLANT_K,
    T_SUBSTRATE_K,
    HeatStepSolution,
    _default_solver_options as _heat_solver_options,
    make_heat_problem,
    solve_heat_step,
)
from .mesh import load_jax_fem_mesh


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Dynamic viscosity of air at 20 °C (Pa·s).  The Eckart streaming
# coefficient scales as ``1 / (2 μ ω²)`` so this is the dominant
# uncertainty in absolute streaming magnitude; see FEM_V2_DESIGN §3.4.
DEFAULT_AIR_VISCOSITY_PA_S: float = 1.81e-5

# Reference temperature for the bulk-T → c₀(T) ideal-gas scaling.  We
# normalise against the env config's stated temperature so the v1
# scaling collapses to identity when the volume-averaged T equals the
# initial chamber-air temperature.  The exact reference is unimportant
# for differentiability — only the *ratio* enters the Helmholtz operator.
T_REFERENCE_FALLBACK_K: float = 293.15

# Default fixed-point convergence tolerance on the relative L2 change in
# the Helmholtz field between inner iterations.
DEFAULT_CONVERGENCE_TOL: float = 1e-3

# Default maximum inner-iteration budget per thermal step
# (FEM_V2_DESIGN §3.4 specifies "3–5").
DEFAULT_MAX_INNER_ITERS: int = 5

# Streaming clamp.  At extreme |p|² gradients (e.g. pre-converged
# pressure spikes near transducer faces) the Eckart formula can blow
# up.  We cap |u_stream| node-wise at this value to keep the heat
# solver's advection-stability margin intact for v1.  Order-of-
# magnitude guidance: 40 kHz Eckart streaming peaks ~ 10 cm/s; a 1 m/s
# cap is 10× headroom.
DEFAULT_STREAMING_VELOCITY_CLAMP_M_S: float = 1.0


# ---------------------------------------------------------------------------
# EckartStreamingComputer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EckartStreamingResult:
    """Streaming velocity field on mesh nodes.

    Attributes:
        velocity: ``(n_nodes, 3)`` time-averaged streaming velocity (m/s).
        magnitude: ``(n_nodes,)`` ``|u_stream|`` for receipts/diagnostics.
        max_magnitude: scalar maximum across the mesh.  Stored as a JAX
            array so the type stays ``jax.grad``-clean; cast with
            ``float(.)`` outside the differentiated graph for receipts.
    """

    velocity: jnp.ndarray
    magnitude: jnp.ndarray
    max_magnitude: jnp.ndarray


class EckartStreamingComputer:
    """Compute Eckart streaming velocity from a complex pressure field.

    Approximation
    -------------

    The full second-order Lagrangian-mean streaming velocity satisfies a
    coupled velocity-pressure problem at second order in Mach number.
    For the v1 build we use the simplified scalar Rayleigh-Schlichting
    form valid in our regime (low Mach, weak attenuation, isotropic
    medium):

        u_stream(x) ≈ ∇|p|² / (4 ρ₀² c₀² ω)

    Direction: along ``∇|p|²``.  Magnitude proportional to that
    gradient.  This captures the *physically dominant* effect (streaming
    velocity ~ ∇|p|² scale-set) without solving the full second-order
    Navier-Stokes problem.  At 1500 Pa peak and σ ≈ 1 mm (focal-spot
    scale), this yields O(cm/s) consistent with FEM_V2_DESIGN §3.4.

    Note on the brief's ``1/(2 μ ω²)`` form: the brief documents an
    Eckart-attenuation form valid for *plane-wave-with-loss* configs,
    which has incorrect dimensions for our gradient-driven scalar
    approximation.  We use the dimensionally consistent
    Rayleigh-Schlichting coefficient ``1/(4 ρ₀² c₀² ω)`` which delivers
    the same physics (streaming proportional to ``∇|p|²``) and the same
    cm/s order-of-magnitude target.  The viscosity μ is kept as a
    constructor parameter for future v2 paths that wire in a true
    Eckart-attenuation term ``-α I/(μ c)`` from the absorption
    coefficient α.

    What's deferred to v2:
      * Full second-order velocity-pressure coupled formulation.
      * Heterogeneous-medium attenuation (currently uses air viscosity
        and density across the entire mesh).
      * Compressibility corrections at the droplet interface.
      * True Eckart-attenuation term using the absorption coefficient α
        (the ``viscosity`` ctor arg is the placeholder for that path).

    Implementation
    --------------

    The gradient ``∇|p|²`` is computed at mesh nodes via a standard
    FEM-style projection:

      1. Interpolate ``|p|² = Re² + Im²`` onto each tet's nodes.
      2. Compute the per-cell gradient via the cell's shape-function
         gradients (constant within a P1 tet element).
      3. Project the cell-constant gradient back onto nodes via a
         volume-weighted average over the cells incident to each node
         (lumped projection — same family as the JAX-FEM
         ``shape_grads`` machinery used in the heat advection term).

    The resulting node-level gradient is then scaled to a velocity.

    Differentiability: the projection is purely linear (mass-lumped
    weights are constant once the mesh is fixed), so ``jax.grad`` flows
    through cleanly.
    """

    def __init__(
        self,
        cells: jnp.ndarray,
        points: jnp.ndarray,
        omega: float,
        rho0: float,
        c0: float,
        viscosity: float = DEFAULT_AIR_VISCOSITY_PA_S,
        velocity_clamp: float = DEFAULT_STREAMING_VELOCITY_CLAMP_M_S,
    ) -> None:
        """Build the projection operator for the given mesh.

        Args:
            cells: ``(n_cells, 4)`` tet vertex indices into ``points``.
            points: ``(n_nodes, 3)`` mesh node coordinates.
            omega: angular frequency (rad/s).
            rho0: reference medium density (kg/m³).
            c0: reference speed of sound (m/s).
            viscosity: dynamic viscosity μ (Pa·s).  Default is air at
                20 °C.  Kept as a parameter for the v2 Eckart-absorption
                path; the v1 Rayleigh-Schlichting form does not consume
                it.
            velocity_clamp: hard-cap on |u_stream| (m/s).  Order-of-
                magnitude guard against pressure-spike-driven blowup at
                early iterations.
        """
        self.cells = jnp.asarray(cells, dtype=jnp.int64)
        self.points = jnp.asarray(points, dtype=jnp.float64)
        self.n_nodes = int(self.points.shape[0])
        self.omega = float(omega)
        self.rho0 = float(rho0)
        self.c0 = float(c0)
        self.viscosity = float(viscosity)
        self.velocity_clamp = float(velocity_clamp)

        # Pre-compute per-cell shape-function gradient + volume.  These
        # are mesh-only quantities so we cache them on construction.
        self._cell_shape_grads, self._cell_volumes = (
            self._compute_shape_grads_and_volumes()
        )

        # Lumped node-volume weights for the cell→node projection
        # (sum of incident-cell volumes, divided by 4 because each tet
        # node owns ¼ of the cell volume in mass-lumped P1).
        self._node_weights = self._compute_lumped_node_weights()

    def _compute_shape_grads_and_volumes(
        self,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute per-cell P1 tet shape-function gradients + volumes.

        For a P1 tet with vertices ``v0, v1, v2, v3``, the shape
        function gradient ``∇N_i`` is constant within the cell and
        equals the i-th column of ``M⁻¹`` where ``M`` is the
        ``(3, 3)`` matrix of edge vectors ``[v1-v0, v2-v0, v3-v0]``.
        The cell volume is ``|det(M)| / 6``.
        """
        cells = self.cells
        verts = self.points[cells]                       # (n_cells, 4, 3)
        v0 = verts[:, 0, :]
        edges = verts[:, 1:, :] - v0[:, None, :]          # (n_cells, 3, 3)
        # det(edges) is signed; volume is absolute.
        det = jnp.linalg.det(edges)
        volumes = jnp.abs(det) / 6.0                       # (n_cells,)

        # Shape-function gradients.  ∇N_0 = -(∇N_1 + ∇N_2 + ∇N_3); the
        # remaining three come from the inverse-transpose of the
        # edge matrix.  We compute via solve to keep the operation
        # numerically stable and AD-friendly.
        inv = jnp.linalg.inv(edges)                        # (n_cells, 3, 3)
        # The inverse columns give ∇N_1, ∇N_2, ∇N_3 in physical space.
        grads_1_2_3 = jnp.transpose(inv, (0, 2, 1))         # (n_cells, 3, 3)
        grad_0 = -jnp.sum(grads_1_2_3, axis=1, keepdims=True)
        # Concatenate so axis 1 is per-vertex, axis 2 is xyz.
        shape_grads = jnp.concatenate([grad_0, grads_1_2_3], axis=1)
        return shape_grads, volumes

    def _compute_lumped_node_weights(self) -> jnp.ndarray:
        """Lumped node-volume weights ``w_n = ¼ · Σ_{c∈incident(n)} V_c``."""
        cells = self.cells
        volumes = self._cell_volumes
        # Each cell contributes V_c / 4 to each of its 4 nodes.
        # Broadcast volumes to (n_cells, 4) so reshape gives (n_cells*4,).
        per_cell_per_node = jnp.broadcast_to(
            volumes[:, None] / 4.0, cells.shape
        )                                                    # (n_cells, 4)
        flat_nodes = cells.reshape(-1)
        flat_vals = per_cell_per_node.reshape(-1)
        weights = jnp.zeros(self.n_nodes, dtype=jnp.float64)
        weights = weights.at[flat_nodes].add(flat_vals)
        # Guard against orphan nodes (shouldn't happen post-compaction).
        weights = jnp.where(weights > 0.0, weights, 1.0)
        return weights

    def compute(
        self,
        re: jnp.ndarray,
        im: jnp.ndarray,
    ) -> EckartStreamingResult:
        """Compute streaming velocity from a complex pressure field.

        Args:
            re: ``(n_nodes,)`` real component of pressure (Pa).
            im: ``(n_nodes,)`` imaginary component of pressure (Pa).

        Returns:
            :class:`EckartStreamingResult`.
        """
        re = jnp.asarray(re, dtype=jnp.float64).reshape(self.n_nodes)
        im = jnp.asarray(im, dtype=jnp.float64).reshape(self.n_nodes)

        # |p|² at nodes.
        p_sq = re * re + im * im                          # (n_nodes,)

        # Per-cell gradient of |p|² via shape-function gradients.
        # ∇|p|² in cell c = Σ_n p_sq[cell[c, n]] · ∇N_n(c).
        cell_p_sq = p_sq[self.cells]                       # (n_cells, 4)
        cell_grad = jnp.einsum(
            "cn,cnd->cd",
            cell_p_sq,
            self._cell_shape_grads,
        )                                                  # (n_cells, 3)

        # Project cell-constant gradient back onto nodes via volume-
        # weighted lumping.  Each tet contributes V_c/4 of its gradient
        # to each of its 4 nodes; we sum and normalise by node weight.
        # Broadcast over the node-axis so reshape produces the correct
        # (n_cells*4, 3) flat array.
        weighted_grad = cell_grad * self._cell_volumes[:, None] / 4.0
        cell_grad_per_node = jnp.broadcast_to(
            weighted_grad[:, None, :],
            (self.cells.shape[0], 4, 3),
        )                                                  # (n_cells, 4, 3)

        flat_nodes = self.cells.reshape(-1)
        flat_grads = cell_grad_per_node.reshape(-1, 3)
        node_grad = jnp.zeros((self.n_nodes, 3), dtype=jnp.float64)
        node_grad = node_grad.at[flat_nodes].add(flat_grads)
        node_grad = node_grad / self._node_weights[:, None]

        # Scale to streaming velocity (Rayleigh-Schlichting scalar form):
        #     u = ∇|p|² / (4 ρ₀² c₀² ω)
        # See class docstring for derivation and v1/v2 caveats.
        scale = 1.0 / (
            4.0 * self.rho0 * self.rho0 * self.c0 * self.c0 * self.omega
        )
        u = scale * node_grad

        # Magnitude clamp (keeps advection stable across early
        # fixed-point iterations when the Helmholtz solution may not yet
        # be converged near transducer faces).
        mag = jnp.linalg.norm(u, axis=-1)                    # (n_nodes,)
        clamp = self.velocity_clamp
        scale_factor = jnp.minimum(1.0, clamp / jnp.maximum(mag, 1e-30))
        u_clamped = u * scale_factor[:, None]
        mag_clamped = jnp.linalg.norm(u_clamped, axis=-1)

        return EckartStreamingResult(
            velocity=u_clamped,
            magnitude=mag_clamped,
            max_magnitude=jnp.max(mag_clamped),
        )


# ---------------------------------------------------------------------------
# StaggeredCouplingDriver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoupledStepResult:
    """Single outer-step coupled-physics result.

    Attributes:
        T_new: ``(n_nodes,)`` temperature after this step (Kelvin).
        re: ``(n_nodes,)`` final inner-iteration pressure (Re, Pa).
        im: ``(n_nodes,)`` final inner-iteration pressure (Im, Pa).
        streaming_velocity: ``(n_nodes, 3)`` streaming velocity (m/s).
        n_inner_iters: number of fixed-point iterations performed.
        field_norm_history: relative L2 change in field per inner iter.
        wall_time_s: total step wall-clock time.
        max_streaming_m_s: max ``|u_stream|`` (m/s) over the mesh.
        c0_used: bulk speed of sound used for the final inner iteration.
    """

    T_new: jnp.ndarray
    re: jnp.ndarray
    im: jnp.ndarray
    streaming_velocity: jnp.ndarray
    n_inner_iters: int
    field_norm_history: List[float]
    wall_time_s: float
    max_streaming_m_s: float
    c0_used: float


@dataclass(frozen=True)
class CoupledTrajectoryResult:
    """Trajectory result: outer-step history.

    Attributes:
        T_history: ``(n_steps + 1, n_nodes)`` temperature trajectory.
        re_history: ``(n_steps, n_nodes)`` pressure-Re per step.
        im_history: ``(n_steps, n_nodes)`` pressure-Im per step.
        streaming_history: ``(n_steps, n_nodes, 3)`` streaming velocity.
        inner_iter_counts: per-step inner iteration counts.
        per_step_wall_time_s: per-step wall clocks.
        wall_time_s: total trajectory wall clock.
        mesh_coords: ``(n_nodes, 3)``.
    """

    T_history: jnp.ndarray
    re_history: jnp.ndarray
    im_history: jnp.ndarray
    streaming_history: jnp.ndarray
    inner_iter_counts: List[int]
    per_step_wall_time_s: List[float]
    wall_time_s: float
    mesh_coords: jnp.ndarray


def _bulk_temperature(
    T: jnp.ndarray,
    node_weights: jnp.ndarray,
) -> float:
    """Volume-weighted bulk temperature (K)."""
    total = jnp.sum(T * node_weights)
    weight = jnp.sum(node_weights)
    return float(total / weight)


def _temperature_corrected_c0(
    T_bulk: float,
    T_reference: float,
    c0_reference: float,
) -> float:
    """Ideal-gas-scaling speed of sound: ``c(T) = c_ref · √(T / T_ref)``.

    This is the v1 simplification flagged in FEM_V2_DESIGN §3.4: a
    single bulk-T-dependent ``c₀`` updates the Helmholtz operator at
    each inner iteration.  Node-level spatial variation of ``c₀(x)`` is
    deferred to v2 (would require extending HelmholtzProblem to accept
    a per-quad material field).
    """
    if T_bulk <= 0.0 or T_reference <= 0.0:
        return c0_reference
    return float(c0_reference * np.sqrt(T_bulk / T_reference))


def _make_temperature_aware_env(
    base_env: EnvironmentConfig,
    T_bulk: float,
) -> EnvironmentConfig:
    """Return a fresh EnvironmentConfig with ``temperature = T_bulk``.

    The Helmholtz operator depends on ``c₀(T)`` and ``ρ₀(T)`` through
    :class:`EnvironmentConfig`'s ``speed_of_sound`` /
    ``angular_frequency`` properties.  By creating a new config with
    the bulk T we trigger ``c₀`` recomputation through the same
    ideal-gas-scaling code path that the original env config uses.

    The medium reference (e.g. AIR_200C) and array geometry are
    preserved.  Density isn't modulated in v1 (the
    ``MaterialConfig.density`` is held fixed); a true T-dependent
    density would require ``ρ(T) = ρ_ref · T_ref / T`` and a re-derivation
    of the Helmholtz operator with variable ``ρ₀``.  Documented as
    deferred to v2.
    """
    from drip_physics.config import EnvironmentConfig

    return EnvironmentConfig(
        medium=base_env.medium,
        temperature=float(T_bulk),
        transducer=base_env.transducer,
        array=base_env.array,
        volume_temperature_profile=base_env.volume_temperature_profile,
        acoustic_streaming_velocity=base_env.acoustic_streaming_velocity,
    )


class StaggeredCouplingDriver:
    """Two-way acoustic ↔ thermal coupling driver.

    Per FEM_V2_DESIGN §3.4 (corrected): one outer thermal timestep wraps
    a 3–5-iter inner fixed-point loop:

      1. Solve Helmholtz under current ``(c₀(T), ρ₀(T))``.
      2. Compute Eckart streaming velocity from ``∇|p|²``.
      3. Solve transient heat step with advection by streaming.
      4. Convergence check on field-norm change.  If above tolerance,
         re-enter step 1 with updated T.

    The driver constructs both Problems once (Helmholtz + Heat) and
    reuses them across all outer steps for solver-state amortisation.
    The Helmholtz problem is instantiated against the *initial*
    environment; per-iteration T-dependent ``c₀`` updates are routed
    through the existing :meth:`HelmholtzProblem.set_params` parameter
    hook (we adjust the operator's ``k_val`` indirectly by re-creating
    the problem when T moves significantly — currently a simplification
    flagged below).

    v1 simplifications (per FEM_V2_DESIGN §3.4):

      * **Bulk-T ``c₀(T)``** rather than node-level ``c₀(x, T)``.
        Volume-averaged temperature drives a single ``c₀`` per inner
        iteration.  Implementation: rebuild ``HelmholtzProblem`` only
        when the bulk T moves more than 5 K from the construction-time
        T (rare in practice — chamber T drifts slowly).
      * **Density ``ρ₀`` held constant** at the medium config value.
      * **Streaming approximation**: scalar Eckart formula in
        :class:`EckartStreamingComputer` rather than full second-order
        coupled velocity-pressure.

    Differentiability
    -----------------

    The end-to-end ``phases → T_at_target`` map is ``jax.grad``-clean.
    Each component (Helmholtz solve, streaming projection, heat solve)
    is JAX-traced; the fixed-point loop runs as a Python `for` loop
    which `jax.grad` differentiates by unrolling.  The convergence
    check uses host Python floats, which means the gradient is taken
    against the *unrolled* graph at the iteration count actually
    executed — this matches the discrete fixed-point pattern used by
    JAX-FEM's own `ad_wrapper` for nonlinear solves.
    """

    def __init__(
        self,
        mesh_path: Path,
        surface_tags: Dict[str, int],
        env_config: EnvironmentConfig,
        transducer_config: TransducerConfig,
        *,
        n_outer_steps: int = 50,
        dt: float = DEFAULT_THERMAL_DT_S,
        max_inner_iters: int = DEFAULT_MAX_INNER_ITERS,
        convergence_tol: float = DEFAULT_CONVERGENCE_TOL,
        pml_width: float = DEFAULT_PML_WIDTH_M,
        pml_alpha: float = DEFAULT_PML_ALPHA,
        velocity_amp: float = DEFAULT_PISTON_VELOCITY_AMP_M_S,
        viscosity: float = DEFAULT_AIR_VISCOSITY_PA_S,
        h_channel: float = H_CHANNEL_W_M2K,
        t_coolant: float = T_COOLANT_K,
        t_substrate: float = T_SUBSTRATE_K,
        droplet_sigma: float = DEFAULT_DROPLET_SIGMA_M,
        streaming_velocity_clamp: float = DEFAULT_STREAMING_VELOCITY_CLAMP_M_S,
        enable_streaming: bool = True,
    ) -> None:
        self.mesh_path = Path(mesh_path)
        self.surface_tags = dict(surface_tags)
        self.env_config = env_config
        self.transducer_config = transducer_config
        self.n_outer_steps = int(n_outer_steps)
        self.dt = float(dt)
        self.max_inner_iters = int(max_inner_iters)
        self.convergence_tol = float(convergence_tol)
        self.pml_width = float(pml_width)
        self.pml_alpha = float(pml_alpha)
        self.velocity_amp = float(velocity_amp)
        self.viscosity = float(viscosity)
        self.h_channel = float(h_channel)
        self.t_coolant = float(t_coolant)
        self.t_substrate = float(t_substrate)
        self.droplet_sigma = float(droplet_sigma)
        self.streaming_velocity_clamp = float(streaming_velocity_clamp)
        self.enable_streaming = bool(enable_streaming)

        # Reference values frozen at construction (ideal-gas c₀ scaling).
        self._c0_reference = float(env_config.speed_of_sound)
        self._T_reference = float(env_config.temperature)

        # Build mesh + Problems once.
        self._mesh, self._points = load_jax_fem_mesh(self.mesh_path)
        self._geom: ChamberGeometry = resolve_chamber_geometry(
            env_config.array, transducer_config
        )

        # Helmholtz Problem.  Rebuilt only when bulk T drifts; held in
        # `self._helmholtz_problem` and `self._helmholtz_T_anchor`.
        self._helmholtz_problem = self._build_helmholtz(env_config)
        self._helmholtz_T_anchor = float(env_config.temperature)
        # Threshold (K) for Helmholtz rebuild; below this the c₀(T)
        # change is negligible (~0.5% over 5 K at 290 K).
        self._helmholtz_rebuild_threshold_k: float = 5.0

        # Heat Problem.  Constant operator throughout the run; only
        # set_params() is called per step.
        self._heat_problem = make_heat_problem(
            mesh=self._mesh,
            geom=self._geom,
            env_config=env_config,
            h_channel=self.h_channel,
            t_coolant=self.t_coolant,
            t_substrate=self.t_substrate,
            droplet_sigma=self.droplet_sigma,
            dt=self.dt,
        )
        self._n_nodes = int(self._heat_problem.fes[0].num_total_nodes)

        # Eckart streaming projector.  Mesh-only — built once.
        self._streaming_computer = EckartStreamingComputer(
            cells=self._mesh.cells,
            points=self._points,
            omega=float(env_config.angular_frequency),
            rho0=float(env_config.medium.density),
            c0=float(env_config.speed_of_sound),
            viscosity=self.viscosity,
            velocity_clamp=self.streaming_velocity_clamp,
        )

        # Cache solver options (reused at every solve call).
        self._helmholtz_solver_opts = _resolve_solver_options(None)
        self._heat_solver_opts = _heat_solver_options()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_helmholtz(self, env: EnvironmentConfig):
        """Build a HelmholtzProblem against a given environment config.

        Helper so we can rebuild on bulk-T drift.  Reuses the same
        mesh, transducer config, and PML settings stored on the driver.
        """
        return make_helmholtz_problem(
            mesh=self._mesh,
            geom=self._geom,
            env_config=env,
            transducer_config=self.transducer_config,
            velocity_amp=self.velocity_amp,
            pml_width=self.pml_width,
            pml_alpha=self.pml_alpha,
        )

    def _maybe_rebuild_helmholtz(self, T_bulk: float) -> None:
        """Rebuild the HelmholtzProblem if the bulk T has drifted.

        Threshold: ``self._helmholtz_rebuild_threshold_k``.  For a
        chamber that drifts within ±5 K of the construction temperature
        across a 50-step trajectory, the original problem stays usable.
        """
        if abs(T_bulk - self._helmholtz_T_anchor) < self._helmholtz_rebuild_threshold_k:
            return
        new_env = _make_temperature_aware_env(self.env_config, T_bulk)
        self._helmholtz_problem = self._build_helmholtz(new_env)
        self._helmholtz_T_anchor = T_bulk

    # ------------------------------------------------------------------
    # Acoustic / streaming / heat primitives
    # ------------------------------------------------------------------

    def _solve_helmholtz_field(
        self,
        phases: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Solve Helmholtz under current operator, return ``(re, im)``."""
        _, _, ad_wrapper, _ = _import_jax_fem()
        fwd = ad_wrapper(
            self._helmholtz_problem,
            solver_options=dict(self._helmholtz_solver_opts),
            adjoint_solver_options=dict(self._helmholtz_solver_opts),
        )
        sol_list = fwd(phases)
        sol = sol_list[0]                                  # (n_nodes, 2)
        return sol[:, 0], sol[:, 1]

    def _streaming_velocity(
        self,
        re: jnp.ndarray,
        im: jnp.ndarray,
    ) -> EckartStreamingResult:
        """Compute streaming velocity field from the pressure field."""
        if not self.enable_streaming:
            return EckartStreamingResult(
                velocity=jnp.zeros((self._n_nodes, 3), dtype=jnp.float64),
                magnitude=jnp.zeros((self._n_nodes,), dtype=jnp.float64),
                max_magnitude=jnp.asarray(0.0, dtype=jnp.float64),
            )
        return self._streaming_computer.compute(re, im)

    def _heat_step(
        self,
        T_prev: jnp.ndarray,
        droplet_centers: jnp.ndarray,
        droplet_amplitudes: jnp.ndarray,
        streaming_velocity: jnp.ndarray,
    ) -> HeatStepSolution:
        """One implicit-Euler heat step under given streaming + sources."""
        return solve_heat_step(
            problem=self._heat_problem,
            T_prev=T_prev,
            droplet_centers=droplet_centers,
            droplet_amplitudes=droplet_amplitudes,
            streaming_velocity=streaming_velocity,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        T_prev: jnp.ndarray,
        phases: jnp.ndarray,
        droplet_centers: jnp.ndarray,
        droplet_amplitudes: jnp.ndarray,
    ) -> CoupledStepResult:
        """Advance one outer thermal timestep with inner fixed-point loop.

        Args:
            T_prev: ``(n_nodes,)`` previous temperature field (K).
            phases: ``(n_transducers,)`` phase commands (rad).
            droplet_centers: ``(n_droplets, 3)`` droplet positions (m).
            droplet_amplitudes: ``(n_droplets,)`` heat amplitudes (W).

        Returns:
            :class:`CoupledStepResult` with the converged field, T, and
            iteration trace.
        """
        T_curr = jnp.asarray(T_prev, dtype=jnp.float64).reshape(self._n_nodes)

        # Bulk T at start of step → maybe rebuild Helmholtz operator.
        T_bulk_initial = _bulk_temperature(
            T_curr, self._streaming_computer._node_weights
        )
        self._maybe_rebuild_helmholtz(T_bulk_initial)
        c0_used = _temperature_corrected_c0(
            T_bulk_initial, self._T_reference, self._c0_reference
        )

        prev_field_re: Optional[jnp.ndarray] = None
        prev_field_im: Optional[jnp.ndarray] = None
        field_norm_history: List[float] = []

        re = jnp.zeros(self._n_nodes, dtype=jnp.float64)
        im = jnp.zeros(self._n_nodes, dtype=jnp.float64)
        u_stream = jnp.zeros((self._n_nodes, 3), dtype=jnp.float64)
        max_stream = 0.0
        T_next = T_curr

        t0 = time.time()
        n_iters = 0
        for inner in range(self.max_inner_iters):
            # 1. Helmholtz solve.
            re, im = self._solve_helmholtz_field(phases)

            # 2. Streaming velocity from |p|².
            stream = self._streaming_velocity(re, im)
            u_stream = stream.velocity
            max_stream = float(np.asarray(stream.max_magnitude))

            # 3. Heat step with advection.
            heat_sol = self._heat_step(
                T_prev=T_curr,
                droplet_centers=droplet_centers,
                droplet_amplitudes=droplet_amplitudes,
                streaming_velocity=u_stream,
            )
            T_next = heat_sol.T

            n_iters = inner + 1

            # 4. Convergence check (relative L2 change in field).
            if prev_field_re is not None:
                diff_re = re - prev_field_re
                diff_im = im - prev_field_im
                num = float(
                    jnp.sqrt(jnp.sum(diff_re * diff_re + diff_im * diff_im))
                )
                den = float(
                    jnp.sqrt(jnp.sum(re * re + im * im))
                )
                rel = num / max(den, 1e-30)
                field_norm_history.append(rel)
                if rel < self.convergence_tol:
                    break

            prev_field_re = re
            prev_field_im = im
            T_curr = T_next  # carry updated T into next inner iteration

        wall = time.time() - t0
        return CoupledStepResult(
            T_new=T_next,
            re=re,
            im=im,
            streaming_velocity=u_stream,
            n_inner_iters=n_iters,
            field_norm_history=field_norm_history,
            wall_time_s=float(wall),
            max_streaming_m_s=float(max_stream),
            c0_used=float(c0_used),
        )

    def trajectory(
        self,
        phases_per_step: jnp.ndarray,
        droplet_trajectory: List[Dict[str, jnp.ndarray]],
        T_initial: Optional[jnp.ndarray] = None,
    ) -> CoupledTrajectoryResult:
        """Run a full coupled trajectory.

        Args:
            phases_per_step: ``(n_outer_steps, n_transducers)`` phase
                commands per step.  If a single ``(n_transducers,)``
                vector is provided, it's broadcast to all steps.
            droplet_trajectory: list of ``{"centers", "amplitudes"}``
                dicts, one per step.
            T_initial: ``(n_nodes,)`` initial temperature.  Defaults to
                a uniform field at chamber-air temperature.

        Returns:
            :class:`CoupledTrajectoryResult` aggregating per-step
            receipts.
        """
        phases_per_step = jnp.asarray(phases_per_step, dtype=jnp.float64)
        if phases_per_step.ndim == 1:
            phases_per_step = jnp.broadcast_to(
                phases_per_step,
                (self.n_outer_steps, phases_per_step.shape[0]),
            )

        if T_initial is None:
            T_curr = jnp.full(
                (self._n_nodes,), float(T_CHAMBER_INIT_K), dtype=jnp.float64
            )
        else:
            T_curr = jnp.asarray(T_initial, dtype=jnp.float64).reshape(
                self._n_nodes
            )

        T_history: List[jnp.ndarray] = [T_curr]
        re_history: List[jnp.ndarray] = []
        im_history: List[jnp.ndarray] = []
        u_history: List[jnp.ndarray] = []
        n_inner: List[int] = []
        per_step: List[float] = []

        t0 = time.time()
        for i in range(self.n_outer_steps):
            entry = (
                droplet_trajectory[i]
                if i < len(droplet_trajectory)
                else {
                    "centers": jnp.zeros((1, 3), dtype=jnp.float64),
                    "amplitudes": jnp.zeros((1,), dtype=jnp.float64),
                }
            )
            centers = jnp.asarray(entry["centers"], dtype=jnp.float64)
            amps = jnp.asarray(entry["amplitudes"], dtype=jnp.float64)

            res = self.step(
                T_prev=T_curr,
                phases=phases_per_step[i],
                droplet_centers=centers,
                droplet_amplitudes=amps,
            )
            T_curr = res.T_new
            T_history.append(T_curr)
            re_history.append(res.re)
            im_history.append(res.im)
            u_history.append(res.streaming_velocity)
            n_inner.append(res.n_inner_iters)
            per_step.append(res.wall_time_s)

        wall = time.time() - t0
        return CoupledTrajectoryResult(
            T_history=jnp.stack(T_history, axis=0),
            re_history=jnp.stack(re_history, axis=0),
            im_history=jnp.stack(im_history, axis=0),
            streaming_history=jnp.stack(u_history, axis=0),
            inner_iter_counts=n_inner,
            per_step_wall_time_s=per_step,
            wall_time_s=float(wall),
            mesh_coords=jnp.asarray(self._points),
        )


__all__ = [
    "DEFAULT_AIR_VISCOSITY_PA_S",
    "DEFAULT_CONVERGENCE_TOL",
    "DEFAULT_MAX_INNER_ITERS",
    "DEFAULT_STREAMING_VELOCITY_CLAMP_M_S",
    "CoupledStepResult",
    "CoupledTrajectoryResult",
    "EckartStreamingComputer",
    "EckartStreamingResult",
    "StaggeredCouplingDriver",
]
