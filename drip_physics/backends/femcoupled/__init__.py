"""FEM-coupled-physics backend root package.

This package implements the v1 FEM-coupled-physics forward model that
will sit behind the ``compute_pressure_from_phases`` interface, mirroring
``drip_physics.backends.jwave_backend.JWaveBackend``. Phase 1 ships only
the chamber-mesh generator; subsequent phases land:

  * Phase 2 — ``HelmholtzProblem`` (parametric Robin BC at transducer
    faces, Sommerfeld v0 PML).
  * Phase 3 — ``HeatTransientProblem`` (transient heat equation).
  * Phase 4 — ``StaggeredCouplingDriver`` orchestrating per-step
    thermal-then-acoustic solves.
  * Phase 5 — ``femcoupled_backend.py`` exposing the
    :class:`PressureBackend` protocol so the new solver is selectable
    via ``PressureConfig(backend="femcoupled", …)``.

See ``simulations/ml_inverse/research/FEM_V2_DESIGN.md`` for the design
contract and ``FEM_DISCOVERY_REPORT.md`` for Phase 0's GREEN receipts.

Deliverables landed
-------------------

* Phase 1 — :func:`build_l1_chamber_mesh` produces the L1 chamber
  ``.msh`` with the three named surface groups (``transducer_faces``,
  ``chamber_walls``, ``substrate``) plus per-transducer physical groups
  (``transducer_0..transducer_N-1``) so phase commands can be applied
  per-disk.
* Phase 2 — :func:`make_helmholtz_problem` / :func:`solve_helmholtz`
  build and solve the time-harmonic acoustic Helmholtz problem with
  parametric Robin (piston) BCs at transducer faces and Sommerfeld
  radiation at the outer boundaries.
* Phase 3 — :func:`make_heat_problem` / :func:`solve_heat_step` /
  :func:`solve_heat_trajectory` advance the transient heat equation
  with implicit Euler, parametric droplet heat sources, and the
  corrected BC split (Dirichlet substrate / Robin cooled walls /
  insulating transducer faces). Accepts an optional advection
  velocity field as the Phase 4 two-way coupling hook.

Example
-------

::

    from drip_physics.config import L1_ARRAY, MA40S4S_40KHZ
    from drip_physics.backends.femcoupled import build_l1_chamber_mesh

    path, info = build_l1_chamber_mesh(L1_ARRAY, MA40S4S_40KHZ)
    print(info["element_count"], info["transducer_face_count"])
"""

from .geometry import (
    ChamberGeometry,
    compute_geometry_hash,
    resolve_chamber_geometry,
    transducer_centers,
)
from .mesh import DEFAULT_MESH_CACHE, build_l1_chamber_mesh, load_jax_fem_mesh

# Phase 2 (Helmholtz) and Phase 3 (Heat) symbols are exposed via lazy
# attribute lookup so the main simulations env (without ``meshio`` /
# ``jax_fem``) still imports the package cleanly. The symbols become
# available the first time they are accessed.
_HELMHOLTZ_EXPORTS = {
    "DEFAULT_PISTON_VELOCITY_AMP_M_S",
    "DEFAULT_PML_ALPHA",
    "DEFAULT_PML_WIDTH_M",
    "DEFAULT_SOLVER_OPTIONS",
    "RECOMMENDED_PML_WIDTH_M",
    "HelmholtzSolution",
    "compute_field",
    "make_helmholtz_problem",
    "solve_helmholtz",
}

_HEAT_EXPORTS = {
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
}

_COUPLING_EXPORTS = {
    "DEFAULT_AIR_VISCOSITY_PA_S",
    "DEFAULT_CONVERGENCE_TOL",
    "DEFAULT_MAX_INNER_ITERS",
    "DEFAULT_STREAMING_VELOCITY_CLAMP_M_S",
    "CoupledStepResult",
    "CoupledTrajectoryResult",
    "EckartStreamingComputer",
    "EckartStreamingResult",
    "StaggeredCouplingDriver",
}


def __getattr__(name: str):  # noqa: D401  — module-level helper
    if name in _HELMHOLTZ_EXPORTS:
        from . import helmholtz as _helmholtz  # local import (lazy)

        return getattr(_helmholtz, name)
    if name in _HEAT_EXPORTS:
        from . import heat as _heat  # local import (lazy)

        return getattr(_heat, name)
    if name in _COUPLING_EXPORTS:
        from . import coupling as _coupling  # local import (lazy)

        return getattr(_coupling, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChamberGeometry",
    "CoupledStepResult",
    "CoupledTrajectoryResult",
    "DEFAULT_AIR_VISCOSITY_PA_S",
    "DEFAULT_CONVERGENCE_TOL",
    "DEFAULT_DROPLET_SIGMA_M",
    "DEFAULT_MAX_INNER_ITERS",
    "DEFAULT_MESH_CACHE",
    "DEFAULT_PISTON_VELOCITY_AMP_M_S",
    "DEFAULT_STREAMING_VELOCITY_CLAMP_M_S",
    "DEFAULT_THERMAL_DT_S",
    "EckartStreamingComputer",
    "EckartStreamingResult",
    "H_CHANNEL_W_M2K",
    "HeatStepSolution",
    "HeatTrajectorySolution",
    "HelmholtzSolution",
    "StaggeredCouplingDriver",
    "T_CHAMBER_INIT_K",
    "T_COOLANT_K",
    "T_CRUCIBLE_K",
    "T_SUBSTRATE_K",
    "build_l1_chamber_mesh",
    "compute_field",
    "compute_geometry_hash",
    "gaussian_heat_source",
    "load_jax_fem_mesh",
    "make_heat_problem",
    "make_helmholtz_problem",
    "resolve_chamber_geometry",
    "solve_heat_step",
    "solve_heat_trajectory",
    "solve_helmholtz",
    "transducer_centers",
]
