"""
Bonding feasibility models.

Two models:
  1. Orme & Huang (1997) thermal remelting — legacy, no acoustic field.
  2. Acoustic solid-state bonding — 20 kHz sonotrode, primary for Drip.

The acoustic model has uncertain parameters (ASMP-060 through ASMP-062)
that will be updated when TE-002 test results are available.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Orme & Huang (1997) thermal remelting model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrmeBondResult:
    """Result of Orme remelting feasibility check."""

    orme_ratio: float           # (T_arrival - T_m) / (T_m - T_sub)
    remelting_feasible: bool    # ratio > threshold
    remelt_margin: float        # ratio - threshold


def orme_remelt(
    T_arrival: float,
    T_melting: float,
    T_substrate: float,
    threshold: float = 1.6,
) -> OrmeBondResult:
    """Orme & Huang (1997) remelting criterion.

    Requires the arriving droplet to have enough thermal energy to
    remelt the previously solidified substrate through conduction alone.
    This model assumes NO acoustic field — purely thermal.

    The ratio (T_i - T_m) / (T_m - T_sub) must exceed ~1.6 for
    remelting to initiate.

    Args:
        T_arrival: Droplet temperature at landing [K]
        T_melting: Melting point of metal [K]
        T_substrate: Substrate temperature far from surface [K]
        threshold: Minimum ratio for remelting (default 1.6 per paper)
    """
    denom = T_melting - T_substrate
    if abs(denom) < 0.1:
        # Substrate at melting point — ratio diverges
        ratio = float("inf") if T_arrival > T_melting else float("-inf")
    else:
        ratio = (T_arrival - T_melting) / denom

    return OrmeBondResult(
        orme_ratio=ratio,
        remelting_feasible=ratio > threshold,
        remelt_margin=ratio - threshold,
    )


# ---------------------------------------------------------------------------
# Acoustic solid-state bonding model (20 kHz sonotrode)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AcousticBondResult:
    """Result of acoustic bonding feasibility check."""

    feasible: bool
    T_arrival_above_min: bool       # droplet hot enough for diffusion
    amplitude_sufficient: bool      # sonotrode delivers enough energy
    mechanism: str                  # "liquid", "semi_solid", or "solid_state"
    details: dict                   # margins and intermediate values


def acoustic_bond(
    T_arrival: float,
    T_melting: float,
    T_substrate: float,
    impact_velocity: float,
    droplet_diameter: float,
    sonotrode_amplitude_um: float = 15.0,
    sonotrode_frequency: float = 20_000.0,
    # --- Uncertain parameters — updated by TE-002 results ---
    # [ASMP-061] Minimum sonotrode amplitude for oxide disruption.
    # TE-002 sweeps 5-25 μm. Literature suggests 5-10 μm sufficient.
    min_amplitude_um: float = 5.0,
    # [ASMP-062] Solidus temperature — above this, liquid phase present.
    solidus_temperature: float = 0.0,       # K (0 = use T_melting)
) -> AcousticBondResult:
    """Acoustic bonding feasibility via 20 kHz sonotrode.

    The mechanism requires LIQUID PHASE at the droplet-substrate interface.
    Ultrasonic cavitation in the liquid aluminum generates micro-jets that
    fracture the Al₂O₃ oxide layer, enabling wetting and metallurgical
    bonding upon solidification.

    Cavitation cannot occur in a solid. Without liquid phase at the
    interface, the oxide barrier remains intact and bonding fails
    (no clamping pressure to enable solid-state welding like UAM).

    The droplet must arrive above solidus (liquid or semi-solid).
    L1 strategy: use larger droplets (600-800 μm) that retain enough
    heat to arrive liquid in air at 200°C chamber. Scale down to
    300-500 μm when argon atmosphere is available.

    Uncertain parameters are placeholders until TE-002 test results.
    """
    T_solidus = solidus_temperature if solidus_temperature > 0 else T_melting

    # Gate 1: Liquid phase required at interface.
    # Droplet must arrive above solidus to provide the liquid medium
    # in which cavitation disrupts the oxide layer.
    has_liquid = T_arrival >= T_solidus

    # Gate 2: Amplitude — sonotrode must deliver enough energy to
    # drive cavitation in the liquid phase.
    amplitude_ok = sonotrode_amplitude_um >= min_amplitude_um

    # Classify bonding mechanism
    if T_arrival >= T_melting:
        mechanism = "liquid"
    elif T_arrival >= T_solidus:
        mechanism = "semi_solid"
    else:
        mechanism = "solid_no_bond"

    feasible = has_liquid and amplitude_ok

    liquid_margin = T_arrival - T_solidus

    details = {
        "T_arrival_K": T_arrival,
        "T_solidus_K": T_solidus,
        "liquid_margin_K": liquid_margin,
        "has_liquid_phase": has_liquid,
        "sonotrode_amplitude_um": sonotrode_amplitude_um,
        "min_amplitude_um": min_amplitude_um,
        "amplitude_margin_um": sonotrode_amplitude_um - min_amplitude_um,
        "mechanism": mechanism,
        "sonotrode_frequency_Hz": sonotrode_frequency,
    }

    return AcousticBondResult(
        feasible=feasible,
        T_arrival_above_min=has_liquid,
        amplitude_sufficient=amplitude_ok,
        mechanism=mechanism,
        details=details,
    )
