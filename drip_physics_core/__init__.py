"""
Drip Physics Core
==================

Shared interface package between the simulations engine (drip_physics) and
the design-tool web backend (web/design-tool).

Holds the canonical type definitions and the parametric design module:

- ``drip_physics_core.config``: MaterialConfig, EnvironmentConfig, ArrayConfig,
  TransducerConfig, DropletConfig, ThermalConfig, PressureConfig, plus all
  built-in profiles (ALUMINUM_LIQUID, AIR_20C, L1_ARRAY, MA40S4S_40KHZ, ...).
- ``drip_physics_core.design``: Constraints, optimizer, sensitivity, sweeps,
  uncertainty, machine spec — the parametric design engine.

This package is intentionally slow-moving and treated as a stable API
contract. The full simulation engine in ``drip_physics`` imports from here.

Phase A of the simulations + design-tool repo split. Phase B will move
this package into its own repo and wire both sides via package install.
"""

# Public types (re-exported for convenience).
from drip_physics_core.config import (
    # Constants
    GRAVITY,
    # Material
    MaterialConfig,
    STYROFOAM,
    ALUMINUM_LIQUID,
    ALUMINUM_SOLID,
    AIR_20C,
    AIR_200C,
    aluminum_viscosity,
    make_al_mg_alloy,
    AL_MG_2_5,
    AL_MG_4_5,
    AL_MG_4_0,
    # Thermal
    ThermalConfig,
    DEFAULT_THERMAL,
    # Pressure
    PressureConfig,
    # Transducer
    TransducerConfig,
    MA40S4S_40KHZ,
    # Array
    ArrayConfig,
    L1_ARRAY,
    # Droplet
    DropletConfig,
    aluminum_droplet,
    styrofoam_bead,
    # Environment
    EnvironmentConfig,
    default_environment,
    styrofoam_test_environment,
    aluminum_print_environment,
    # Chamber gradient
    linear_chamber_gradient,
    DEFAULT_CHAMBER_GRADIENT,
)

__version__ = "0.1.0"

__all__ = [
    "GRAVITY",
    "MaterialConfig",
    "STYROFOAM",
    "ALUMINUM_LIQUID",
    "ALUMINUM_SOLID",
    "AIR_20C",
    "AIR_200C",
    "aluminum_viscosity",
    "make_al_mg_alloy",
    "AL_MG_2_5",
    "AL_MG_4_5",
    "AL_MG_4_0",
    "ThermalConfig",
    "DEFAULT_THERMAL",
    "PressureConfig",
    "TransducerConfig",
    "MA40S4S_40KHZ",
    "ArrayConfig",
    "L1_ARRAY",
    "DropletConfig",
    "aluminum_droplet",
    "styrofoam_bead",
    "EnvironmentConfig",
    "default_environment",
    "styrofoam_test_environment",
    "aluminum_print_environment",
    "linear_chamber_gradient",
    "DEFAULT_CHAMBER_GRADIENT",
]
