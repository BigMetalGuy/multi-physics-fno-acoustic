"""
Thermal model for droplet cooling during acoustic levitation flight (SW-16).

Stateless, pure-function module. Computes instantaneous heat loss rate and
phase state (liquid / mushy / solid) for a single droplet at a single timestep.

Physics:
    - Ranz-Marshall forced convection (Nu = 2 + 0.6 Re^0.5 Pr^1/3)
    - Stefan-Boltzmann radiative loss
    - Three-state phase model: liquid -> mushy (latent heat plateau) -> solid
    - Acoustic corrections: oscillatory particle velocity at trap, Eckart
      streaming, Yarin toroidal insulation factor

References:
    - Ranz & Marshall (1952): evaporation from drops, Chem. Eng. Prog. 48(3)
    - Yarin et al. (1999): evaporation of levitated droplets, J. Fluid Mech. 399
    - Derivation 5 (proofs/D05): thermal analysis for 300 um Al droplet

See ASSUMPTIONS.md ASMP-001 for the 8.9x discrepancy discussion.

All units SI throughout. Every parameter documented with units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from .config import (
    EnvironmentConfig,
    MaterialConfig,
    ThermalConfig,
    DEFAULT_THERMAL,
)

# =============================================================================
# Constants
# =============================================================================

# Stefan-Boltzmann constant [W / (m^2 K^4)]
STEFAN_BOLTZMANN: float = 5.670374419e-8

# Prandtl number for air — approximately constant across 200-1200 K range
_PR_AIR: float = 0.71


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class ThermalResult:
    """Result of a single-timestep thermal computation.

    Attributes:
        dT_dt: K/s — rate of temperature change (negative = cooling).
        dlf_dt: 1/s — rate of liquid fraction change (negative = solidifying).
        heat_flux_W: W — total heat loss rate (positive = heat leaving droplet).
        Nu_eff: Effective Nusselt number (dimensionless).
        phase: Current phase state: 'liquid', 'mushy', or 'solid'.
        Nu_components: Breakdown dict for debugging / visualization.
    """
    dT_dt: float
    dlf_dt: float
    heat_flux_W: float
    Nu_eff: float
    phase: str
    Nu_components: Dict[str, float]


# =============================================================================
# Air property correlations
# =============================================================================

def air_properties_at_temperature(T_K: float) -> Dict[str, float]:
    """Compute air thermophysical properties at temperature T.

    Uses ideal-gas density scaling and Sutherland-like power-law fits.
    Valid range approximately 200-1200 K.

    Args:
        T_K: Air temperature in Kelvin.

    Returns:
        dict with keys:
            density     — kg/m^3 (ideal gas, 1 atm)
            viscosity   — Pa s (Sutherland approximation, exponent 0.7)
            conductivity — W/(m K) (power-law fit, exponent 0.8)
            prandtl     — dimensionless (~0.71 for air)

    Raises:
        ValueError: If T_K <= 0.
    """
    if T_K <= 0.0:
        raise ValueError(f"Temperature must be positive, got {T_K} K")

    # Reference conditions: 293.15 K (20 C), 1 atm
    T_ref = 293.15

    # Ideal gas: rho proportional to 1/T at constant pressure
    rho = 1.204 * (T_ref / T_K)  # kg/m^3

    # Dynamic viscosity: Sutherland-like power law
    # mu ~ T^0.7 (simplified; full Sutherland adds ~1% accuracy)
    mu = 1.825e-5 * (T_K / T_ref) ** 0.7  # Pa s

    # Thermal conductivity of air: power-law fit
    k = 0.0257 * (T_K / T_ref) ** 0.8  # W/(m K)

    return {
        'density': rho,
        'viscosity': mu,
        'conductivity': k,
        'prandtl': _PR_AIR,
    }


# =============================================================================
# Biot number check
# =============================================================================

def compute_biot_number(
    droplet_material: MaterialConfig,
    droplet_radius: float,
    h: float,
) -> float:
    """Verify lumped-capacitance assumption is valid (Bi << 0.1).

    The Biot number Bi = h * R / k_droplet compares external convective
    resistance to internal conductive resistance. Lumped capacitance
    (uniform droplet temperature) requires Bi << 0.1.

    Args:
        droplet_material: Droplet MaterialConfig (needs thermal_conductivity).
        droplet_radius: Droplet radius in meters.
        h: Convective heat transfer coefficient in W/(m^2 K).

    Returns:
        Biot number (dimensionless).

    Raises:
        ValueError: If thermal_conductivity is zero or negative.
    """
    k_droplet = droplet_material.thermal_conductivity
    if k_droplet <= 0.0:
        raise ValueError(
            f"Material '{droplet_material.name}' has non-positive "
            f"thermal_conductivity={k_droplet} W/(m K). "
            f"Cannot compute Biot number."
        )
    return h * droplet_radius / k_droplet


# =============================================================================
# Core thermal computation
# =============================================================================

def compute_heat_loss(
    droplet_material: MaterialConfig,
    env: EnvironmentConfig,
    thermal_cfg: ThermalConfig,
    temperature: float,
    liquid_fraction: float,
    velocity: NDArray[np.float64],
    droplet_radius: float,
    droplet_mass: float,
    grad_p_magnitude: float,
    position: NDArray[np.float64] | None = None,
    dt: float = 1e-4,
) -> ThermalResult:
    """Compute instantaneous heat loss and phase state for a droplet.

    Pure function — no side effects, no mutation of inputs.

    Args:
        droplet_material: MaterialConfig for the droplet (provides cp, L,
            emissivity, solidus_temperature, thermal_conductivity).
        env: EnvironmentConfig (provides medium density, angular_frequency,
            temperature for air property scaling).
        thermal_cfg: ThermalConfig with correction factors (C_node, C_trap,
            streaming_velocity, ambient_temperature).
        temperature: Current droplet temperature in K.
        liquid_fraction: Current liquid fraction, 0.0 (fully solid) to
            1.0 (fully liquid).
        velocity: Droplet velocity vector [vx, vy, vz] in m/s.
        droplet_radius: Droplet radius in m.
        droplet_mass: Droplet mass in kg.
        grad_p_magnitude: Magnitude of acoustic pressure gradient |grad(p)|
            at droplet position, in Pa/m. Used to estimate oscillatory
            particle velocity at the trap.
        position: Optional droplet position [x, y, z] in m. When provided
            and env.volume_temperature_profile is not None, the local ambient
            temperature is read from the spatial profile instead of the uniform
            thermal_cfg.ambient_temperature.
        dt: Simulation timestep in seconds (default 1e-4). Used for numerical
            clamping to prevent temperature overshoot past phase boundaries
            in a single step.

    Returns:
        ThermalResult with dT_dt, dlf_dt, heat_flux_W, Nu_eff, phase,
        and Nu_components breakdown.

    Raises:
        ValueError: If liquid_fraction is outside [0, 1] or temperature <= 0
            or droplet_mass <= 0 or droplet_radius <= 0.
    """
    # --- Input validation ---------------------------------------------------
    if not (0.0 <= liquid_fraction <= 1.0):
        raise ValueError(
            f"liquid_fraction must be in [0, 1], got {liquid_fraction}"
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature} K")
    if droplet_mass <= 0.0:
        raise ValueError(f"droplet_mass must be positive, got {droplet_mass} kg")
    if droplet_radius <= 0.0:
        raise ValueError(
            f"droplet_radius must be positive, got {droplet_radius} m"
        )

    # --- Step 1: Effective Nusselt number -----------------------------------

    # Air properties at film temperature T_film = (T_drop + T_amb)/2
    # per Ranz-Marshall specification. [ASMP-058 RESOLVED]
    T_amb = thermal_cfg.ambient_temperature
    T_film = (temperature + T_amb) / 2.0
    rho_air = env.medium.density * (293.15 / T_film)  # kg/m^3

    # Oscillatory particle velocity at trap location
    # v_osc = |grad(p)| / (rho_air * omega)
    # At a pressure antinode (trap), particle velocity is minimized;
    # C_node accounts for being near (not exactly at) a velocity node.
    omega = env.angular_frequency  # rad/s
    if omega > 0.0 and rho_air > 0.0:
        v_osc = grad_p_magnitude / (rho_air * omega)  # m/s
    else:
        v_osc = 0.0

    # Effective velocity at droplet surface: vector sum of contributions
    # - v_fall: droplet fall velocity relative to air
    # - v_eckart: Eckart acoustic streaming (bulk flow from intensity gradient)
    # - v_osc * C_node: oscillatory velocity attenuated at pressure antinode
    v_fall = float(np.linalg.norm(velocity))  # m/s
    v_eckart = thermal_cfg.streaming_velocity  # [ASMP-044] m/s, order-of-magnitude estimate
    C_node = thermal_cfg.C_node  # [ASMP-042] dimensionless, weak basis for falling droplet
    v_eff = np.sqrt(
        v_fall**2 + v_eckart**2 + (v_osc * C_node)**2
    )  # m/s

    # Air viscosity at film temperature (Sutherland approximation)
    mu_air = 1.825e-5 * (T_film / 293.15) ** 0.7  # Pa s

    # Reynolds number based on effective velocity and droplet diameter
    droplet_diameter = 2.0 * droplet_radius  # m
    if mu_air > 0.0:
        Re_eff = rho_air * v_eff * droplet_diameter / mu_air
    else:
        Re_eff = 0.0

    # [ASMP-052] Ranz-Marshall correlation — well-established for uniform crossflow,
    # but application to acoustically-driven flow is non-standard. See ASSUMPTIONS.md.
    # Nu = 2 + 0.6 * Re^0.5 * Pr^(1/3)
    # Ranz & Marshall (1952), standard for forced convection on spheres
    Nu_forced = 2.0 + 0.6 * Re_eff**0.5 * _PR_AIR ** (1.0 / 3.0)

    # [ASMP-043] C_trap: Yarin toroidal flow correction — derived for stationary
    # levitated droplets, not falling droplets. Physical basis weak. See ASSUMPTIONS.md.
    # Yarin et al. (1999) report 30-70% reduction depending on geometry.
    C_trap = thermal_cfg.C_trap  # dimensionless, typically 0.4-0.8
    Nu_eff = max(2.0, Nu_forced * C_trap)  # floor at conduction limit (Nu=2)

    # --- Step 2: Heat loss computation --------------------------------------

    # Thermal conductivity of air at film temperature
    k_air = 0.0257 * (T_film / 293.15) ** 0.8  # W/(m K)

    # Convective heat transfer coefficient
    h = Nu_eff * k_air / droplet_diameter  # W/(m^2 K)

    # Droplet surface area (sphere)
    A = 4.0 * np.pi * droplet_radius**2  # m^2

    # Convective heat loss: Newton's law of cooling
    # Q_conv = h * A * (T_droplet - T_ambient)
    # Use spatial temperature profile if available, else fall back to uniform ambient.
    if position is not None and env.volume_temperature_profile is not None:
        T_amb = env.volume_temperature_profile(position)  # K, local ambient
    else:
        T_amb = thermal_cfg.ambient_temperature  # K, uniform fallback
    Q_conv = h * A * (temperature - T_amb)  # W

    # Radiative heat loss: Stefan-Boltzmann law
    # Q_rad = epsilon * sigma * A * (T^4 - T_amb^4)
    Q_rad = (
        droplet_material.emissivity
        * STEFAN_BOLTZMANN
        * A
        * (temperature**4 - T_amb**4)
    )  # W

    Q_total = Q_conv + Q_rad  # W (positive = heat leaving droplet)

    # --- Step 3: Phase logic (supports alloy solidus-liquidus range) --------
    #
    # Pure metals (solidus == liquidus): binary liquid/solid, latent heat
    # released isothermally at the melting point.
    #
    # Alloys (solidus < liquidus): mushy zone spans [T_solidus, T_liquidus].
    # Temperature drops continuously through the range while latent heat is
    # released proportionally. Uses lever-rule approximation for equilibrium
    # liquid fraction: f_L = (T - T_s) / (T_l - T_s).
    #
    # For rapid solidification (>100 K/s, which applies here at ~2000 K/s),
    # Scheil equation would be more accurate, but requires partition coefficient
    # data not yet available for all Al-Mg compositions. Lever rule overestimates
    # liquid fraction at a given temperature (slower solidification) — this is
    # conservative (predicts later solidification than reality).

    T_solidus = droplet_material.effective_solidus     # K
    T_liquidus = droplet_material.effective_liquidus    # K
    L = droplet_material.latent_heat                    # J/kg
    cp = droplet_material.specific_heat                 # J/(kg K)
    has_mushy_range = (T_liquidus - T_solidus) > 1.0    # >1K range = alloy

    if temperature > T_liquidus:
        # LIQUID: above liquidus, normal sensible cooling
        phase = 'liquid'
        if cp > 0.0:
            dT_dt = -Q_total / (droplet_mass * cp)  # K/s
        else:
            dT_dt = 0.0
        dlf_dt = 0.0  # 1/s

        # Clamp: don't overshoot liquidus in one step
        # dt_lookahead uses the caller's simulation dt to prevent temperature
        # overshoot into the mushy zone in a single step.
        dt_lookahead = dt  # [ASMP-057] Numerical stability, not physics
        if dT_dt < 0.0 and (temperature + dT_dt * dt_lookahead) < T_liquidus:
            dT_dt = -(temperature - T_liquidus) / dt_lookahead

    elif has_mushy_range and temperature > T_solidus:
        # MUSHY (alloy): temperature drops through solidus-liquidus range
        # while latent heat is released. Effective heat capacity includes
        # both sensible and latent contributions:
        #   cp_eff = cp + L × d(f_L)/dT = cp + L / (T_l - T_s)
        # This allows temperature to decrease continuously through the mushy
        # zone instead of locking at a single temperature.
        phase = 'mushy'
        dT_range = T_liquidus - T_solidus  # K
        cp_eff = cp + L / dT_range  # J/(kg·K), effective heat capacity in mushy zone
        if cp_eff > 0.0:
            dT_dt = -Q_total / (droplet_mass * cp_eff)  # K/s
        else:
            dT_dt = 0.0
        # Liquid fraction tracks temperature via lever rule
        dlf_dt = dT_dt / dT_range  # 1/s (negative when cooling)

        # Clamp: don't overshoot solidus
        # dt_lookahead uses the caller's simulation dt to prevent temperature
        # overshoot past solidus in a single step.
        dt_lookahead = dt  # [ASMP-057] Numerical stability, not physics
        if dT_dt < 0.0 and (temperature + dT_dt * dt_lookahead) < T_solidus:
            dT_dt = -(temperature - T_solidus) / dt_lookahead
            dlf_dt = dT_dt / dT_range

    elif not has_mushy_range and liquid_fraction > 0.0:
        # MUSHY (pure metal): temperature locked at solidus while latent heat
        # depletes the liquid fraction. This is the isothermal solidification
        # plateau — correct for pure metals with a single melting point.
        phase = 'mushy'
        dT_dt = 0.0  # K/s — temperature held constant

        if L > 0.0:
            # dlf_dt = -Q_total / (m * L): heat removes liquid fraction
            dlf_dt = -Q_total / (droplet_mass * L)  # 1/s
        else:
            # No latent heat: instantaneous solidification
            dlf_dt = -1e6  # 1/s (effectively instant)

    else:
        # SOLID: below solidus, cooling with solid specific heat
        # Caller should pass ALUMINUM_SOLID material for accurate cp
        phase = 'solid'
        if cp > 0.0:
            dT_dt = -Q_total / (droplet_mass * cp)  # K/s
        else:
            dT_dt = 0.0
        dlf_dt = 0.0  # 1/s

    # --- Step 4: Build result -----------------------------------------------

    return ThermalResult(
        dT_dt=dT_dt,
        dlf_dt=dlf_dt,
        heat_flux_W=Q_total,
        Nu_eff=Nu_eff,
        phase=phase,
        Nu_components={
            'convective': Q_conv,       # W
            'radiative': Q_rad,         # W
            'Re_eff': Re_eff,           # dimensionless
            'v_eff': v_eff,             # m/s
            'v_osc': v_osc,             # m/s
            'h': h,                     # W/(m^2 K)
        },
    )
