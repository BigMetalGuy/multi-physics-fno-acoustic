"""
Physics — minimal subset for the multi-physics FNO surrogate pipeline.

This is a slim cut of the broader drip_physics engine, containing only the
modules used by the ml_inverse training/inversion/distillation pipeline and
the femcoupled FEM backend. The full drip_physics engine (control blocks,
trajectory solvers, batch generators, disturbance models, etc.) is not
included here — those are simulation-runtime concerns orthogonal to the
forward surrogate training contribution.

Modules included:
    config            — material / array / transducer / environment configs
    core              — ArrayGeometry, PhaseArray, SimulationParams types
    pressure          — analytical acoustic forward (1/r superposition)
    pressure_backend  — backend dispatcher (analytical / jwave / femcoupled)
    geometry          — cylindrical/planar array generators
    pathtrack         — trajectory simulation for inverse-design eval
    api               — high-level run helpers (run_simulation, sweep_targets)
    backends.jwave_backend    — j-Wave spectral Helmholtz solver
    backends.femcoupled_backend — FEM-coupled forward (Helmholtz + heat + streaming)
    backends.femcoupled.*     — FEM building blocks (mesh, helmholtz, heat, coupling)

Import submodules explicitly:
    from drip_physics.config import PressureConfig, EnvironmentConfig
    from drip_physics.core import ArrayGeometry, PhaseArray, SimulationParams
    from drip_physics.backends.femcoupled_backend import compute_pressure_from_phases
"""

__version__ = "0.1.0-fno-public"
