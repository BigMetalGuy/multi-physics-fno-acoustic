"""Backward-compat shim — types moved to ``drip_physics_core.config``.

This module is kept so existing ``from drip_physics.config import X`` calls
continue to work. New code should import from ``drip_physics_core.config``
directly.

This shim will be removed after Phase B of the repo split (when
drip_physics_core moves to its own repo) and all internal callers have
migrated.

See: research/L3_FULL_PAIRWISE_PLAN.md (Phase A: extract drip_physics_core).
"""
# ruff: noqa: F401, F403

# Wildcard re-export so any symbol the original module exposed continues
# to be reachable via ``drip_physics.config``.
from drip_physics_core.config import *  # noqa: F401,F403

# Explicit re-exports so static analyzers / IDEs still resolve names.
from drip_physics_core.config import (  # noqa: F401
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
