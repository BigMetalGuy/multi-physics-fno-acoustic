"""
Physics evaluation engine with registry pattern.

Each physics domain is a callable that takes a DesignPoint and returns
a dict of computed values. The registry makes it trivial to add new
physics modules (D14 solidification, D15-17 Koopman, etc.) without
refactoring.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import replace
from types import MappingProxyType
from typing import Callable, Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp

from drip_physics_core.config import GRAVITY
from .bonding import AcousticBondResult, acoustic_bond, orme_remelt
from .constraints import ConstraintSet, l1_constraints
from .models import BondingMode, DesignPoint, DesignReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sim-reach provider hook
# ---------------------------------------------------------------------------
# `_compute_single_crucible_reach_sim` runs a closed-loop pathtrack simulation
# to derive the realistic single-crucible reach. The simulation engine
# (drip_physics) is the only thing that owns ArrayGeometry / SimulationParams /
# simulate_path_tracking; drip_physics_core does not. Rather than import
# drip_physics from here (which would invert the dependency graph and break
# the design-tool consumer that only installs drip_physics_core), we accept
# a registered callable and route through it.
#
# drip_physics auto-registers a provider on import. Consumers that don't have
# drip_physics installed get a clear error if they ask for sim-derived reach,
# and `_compute_single_crucible_reach_sim`'s exception handler falls back to
# the bang-bang upper bound.
_SimReachProvider = Callable[..., float]
_sim_reach_provider: Optional[_SimReachProvider] = None


def register_sim_reach_provider(fn: _SimReachProvider) -> None:
    """Register the simulator that powers sim-derived crucible reach.

    Called by drip_physics on import. If your consumer never installs
    drip_physics (e.g. the design tool backend), do not call this — the
    design engine will fall back to the bang-bang upper bound.

    The provider's signature must match the keyword-only signature of
    `_sim_reach_lru` (see below).
    """
    global _sim_reach_provider
    _sim_reach_provider = fn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STEFAN_BOLTZMANN = 5.670374e-8  # W/(m² K⁴)
PR_AIR = 0.71                  # Prandtl number of air

# Latent heat mushy-zone half-width [K].  Latent heat is spread over
# T_melt ± this value using an effective-cp approach.
_MUSHY_HALFWIDTH_K = 2.5


# ---------------------------------------------------------------------------
# Physics module type
# ---------------------------------------------------------------------------
PhysicsEval = Callable[[DesignPoint], dict]


# ---------------------------------------------------------------------------
# Individual physics evaluators
# ---------------------------------------------------------------------------
def _eval_acoustics(dp: DesignPoint) -> dict:
    """D1/D2: Gor'kov force and ka validity.

    F_max uses the correct Bruus time-averaged Gor'kov formulation:
      F_max = V * k * p² * |f1 + 3f2/2| / (4 * ρ_m * c²)
    matching force.py's M1/M2 coefficients.
    """
    c_m = dp.atmosphere_sound_speed
    k_wave = 2.0 * np.pi * dp.transducer.frequency / c_m
    ka = k_wave * dp.droplet_radius

    rho_m = dp.atmosphere_density
    rho_p = dp.material.density
    c_p = dp.material.sound_speed

    f1 = 1.0 - (rho_m * c_m ** 2) / (rho_p * c_p ** 2)
    f2 = 2.0 * (rho_p - rho_m) / (2.0 * rho_p + rho_m)

    # Coherent focused pressure: all N transducers in-phase at focal point
    p_single = dp.transducer.p_ref * dp.transducer.r_ref / dp.array.cylinder_radius
    p_focused = dp.n_transducers * p_single

    # Gor'kov peak force with time-averaging (matching force.py M1/M2):
    #   F = V * k * p² * |f1 + 3f2/2| / (4 * ρ_m * c²)
    # The factor of 4 = 2 (time-averaging of p²) × 2 (spatial max of sin(2kx))
    V = dp.droplet_volume
    gorkov_factor = abs(f1 + 1.5 * f2)
    F_max = V * k_wave * p_focused ** 2 * gorkov_factor / (4.0 * rho_m * c_m ** 2)

    return {
        "ka": ka,
        "ka_valid": ka < 0.5,
        "f1": f1,
        "f2": f2,
        "contrast_factor": f1 + 1.5 * f2,
        "F_max": F_max,
        "force_to_weight": F_max / dp.droplet_weight if dp.droplet_weight > 0 else 0.0,
        "focused_pressure": p_focused,
    }


def _eval_array(dp: DesignPoint) -> dict:
    """D3: Array geometry checks."""
    lam = dp.wavelength
    d_arc = 2.0 * np.pi * dp.array.cylinder_radius / dp.array.transducers_per_ring
    return {
        "n_transducers": dp.n_transducers,
        "arc_spacing": d_arc,
        "nyquist_spacing_ok": d_arc < lam / 2.0,
    }


def _eval_control(dp: DesignPoint) -> dict:
    """D4: Phase control / steering range."""
    d_arc = 2.0 * np.pi * dp.array.cylinder_radius / dp.array.transducers_per_ring
    alpha_max = np.pi / (4.0 * d_arc)

    # Geometric hard limit: can't steer beyond the array cylinder
    geom_limit = dp.array.cylinder_radius

    # Force-limited reach: model as critically damped PD controller
    # that must reach target AND stop by landing time.
    # For PD control: x_max ≈ F_max × t_fall² / (m × π²)
    # (quarter-period of damped oscillation — reach and stop)
    # This is computed in _eval_multi_droplet after acoustics + thermal run,
    # so we use a placeholder here. The real value is set post-registry.
    max_disp = geom_limit  # overridden by _eval_build_geometry

    return {
        "alpha_max": alpha_max,
        "max_displacement": max_disp,
        "displacement_ok": dp.target_displacement <= max_disp,
        "geometric_limit": geom_limit,
    }


def _compute_single_crucible_reach(F_max: float, droplet_mass: float,
                                    fall_time: float, array_radius: float) -> float:
    """Bang-bang kinematic upper bound on reach (no controller modeled).

    Time-optimal trajectory: accelerate at a_max for t/2, decelerate at
    a_max for the remaining t/2 so the droplet arrives with zero lateral
    velocity. Position at landing:

        x_accel  = 1/2 · a_max · (t/2)²        [first half]
        v_mid    = a_max · t/2
        x_decel  = v_mid · t/2 − 1/2 · a_max · (t/2)²
        x_max    = x_accel + x_decel = a_max · t² / 4

    This is the kinematic upper bound — no real controller delivers this.
    `_compute_single_crucible_reach_sim` calls the closed-loop pathtrack
    sim for the realistic number; use that one for headline reporting.
    Clamped to array radius (can't steer beyond the cylinder).
    """
    if droplet_mass <= 0 or fall_time <= 0:
        return 0.0
    a_max = F_max / droplet_mass
    reach = 0.25 * a_max * fall_time ** 2
    return min(reach, array_radius)


def _compute_single_crucible_reach_sim(
    dp: DesignPoint, ideal_target: float, fall_time: float
) -> float:
    """Closed-loop reach via pathtrack.simulate_path_tracking.

    Asks the controller to land the droplet at `ideal_target` (the
    bang-bang upper bound) and measures the actual landing position —
    that achieved displacement is the realistic single-crucible reach
    under the current controller, drag, and acoustic-force model.

    This calls into pathtrack/trajectory/inverse, so any improvement to
    those modules flows into the design engine automatically. Cost is
    ~100-300 ms per call; results are LRU-cached on quantized inputs so
    Monte Carlo sweeps that vary non-reach-relevant fields amortize.

    Falls back to the bang-bang upper bound on any sim failure (logged).
    """
    if ideal_target <= 0 or fall_time <= 0:
        return 0.0
    try:
        return _sim_reach_lru(
            droplet_diameter=_quantize(dp.droplet_diameter, 1e-6),
            cylinder_radius=_quantize(dp.array.cylinder_radius, 1e-3),
            ring_count=int(dp.array.ring_count),
            transducers_per_ring=int(dp.array.transducers_per_ring),
            ring_spacing=_quantize(dp.array.ring_spacing, 1e-3),
            chamber_temperature=_quantize(dp.chamber_temperature, 1.0),
            transducer_frequency=_quantize(dp.transducer.frequency, 100.0),
            material_density=_quantize(dp.material.density, 10.0),
            material_sound_speed=_quantize(dp.material.sound_speed, 10.0),
            ideal_target=_quantize(ideal_target, 0.001),
        )
    except Exception as e:
        logger.warning(
            "Sim-derived reach failed (%s); falling back to bang-bang upper bound", e
        )
        return ideal_target


def _quantize(x: float, step: float) -> float:
    """Round to nearest multiple of `step` for stable LRU keying."""
    return round(x / step) * step


@functools.lru_cache(maxsize=4096)
def _sim_reach_lru(
    *,
    droplet_diameter: float,
    cylinder_radius: float,
    ring_count: int,
    transducers_per_ring: int,
    ring_spacing: float,
    chamber_temperature: float,
    transducer_frequency: float,
    material_density: float,
    material_sound_speed: float,
    ideal_target: float,
) -> float:
    """LRU-cached delegate to the registered sim-reach provider.

    drip_physics_core does not depend on the simulation engine, so the
    actual closed-loop pathtrack call lives in drip_physics and is wired
    in via `register_sim_reach_provider`. Inputs are kept as hashable
    scalars so the cache key is deterministic across callers that pass
    equivalent fields.
    """
    if _sim_reach_provider is None:
        raise RuntimeError(
            "Sim-derived reach requires a registered provider. Install "
            "drip_physics (which auto-registers on import) or pass "
            "`use_sim_reach=False` to evaluate_design()."
        )
    return _sim_reach_provider(
        droplet_diameter=droplet_diameter,
        cylinder_radius=cylinder_radius,
        ring_count=ring_count,
        transducers_per_ring=transducers_per_ring,
        ring_spacing=ring_spacing,
        chamber_temperature=chamber_temperature,
        transducer_frequency=transducer_frequency,
        material_density=material_density,
        material_sound_speed=material_sound_speed,
        ideal_target=ideal_target,
    )


def _compute_build_area(n_crucibles: int, reach: float) -> tuple[float, str]:
    """Compute total build area from circle-packed crucible arrangement.

    Each crucible covers a circle of radius `reach`. Crucibles are arranged
    with mandatory overlap (no coverage gaps). Uses hex packing for N >= 7.

    Returns (area_m2, pattern_name).
    """
    import math

    r = reach
    if r <= 0:
        return 0.0, "none"

    single_area = math.pi * r * r

    if n_crucibles <= 0:
        return 0.0, "none"

    if n_crucibles == 1:
        return single_area, "single"

    if n_crucibles == 2:
        # Two circles, centers offset by r (significant overlap for gap-free)
        # Union area = 2πr² - overlap
        # Footprint is a stadium: length 3r, width 2r
        # Effective area ≈ 2πr² × 0.73
        return single_area * 2 * 0.73, "linear_pair"

    if n_crucibles == 3:
        # Equilateral triangle, centers at r√3 apart for full overlap
        # Spacing = r (not r√3) for gap-free coverage at center
        # Footprint: equilateral triangle with rounded vertices
        # Effective area ≈ 3πr² × 0.78
        return single_area * 3 * 0.78, "triangle"

    if n_crucibles == 4:
        # Square arrangement, spacing = r√2 for diagonal coverage
        # Effective area ≈ 4πr² × 0.75
        return single_area * 4 * 0.75, "square"

    if n_crucibles == 5:
        # 1 center + 4 ring at distance r from center
        # Very efficient — center fills all gaps
        # Effective area ≈ 5πr² × 0.82
        return single_area * 5 * 0.82, "center_plus_4"

    if n_crucibles == 6:
        # Hex ring, 6 circles at distance r from center (no center)
        # Small gap in middle — acceptable with overlap
        # Effective area ≈ 6πr² × 0.78
        return single_area * 6 * 0.78, "hex_ring"

    # N >= 7: 1 center + (N-1) hex ring
    # General hex packing: each crucible adds ~(3√3/2)r² effective area
    # with edge correction ~0.85
    hex_cell_area = (3.0 * math.sqrt(3.0) / 2.0) * r * r  # ≈ 2.598 r²
    effective_area = n_crucibles * hex_cell_area * 0.85
    return effective_area, "hex_packed"


def _eval_thermal(dp: DesignPoint) -> dict:
    """D5: In-flight cooling via coupled fall + Ranz-Marshall ODE."""
    thermal = _simulate_inflight_cooling(dp)
    Bi = (
        thermal["h_avg"] * dp.droplet_radius / dp.material.thermal_conductivity
        if dp.material.thermal_conductivity > 0
        else float("inf")
    )
    cooling_rate = (
        thermal["temp_drop"] / thermal["fall_time"]
        if thermal["fall_time"] > 0
        else 0.0
    )
    return {**thermal, "Bi": Bi, "cooling_rate": cooling_rate}


def _eval_machine(dp: DesignPoint) -> dict:
    """D9-11: Full machine power budget and BOM cost.

    Power includes: 40 kHz array, 20 kHz sonotrode, crucible bank,
    chamber heater, substrate heater, control electronics, PSU overhead.

    Cost includes: transducers, driver boards, array frame, sonotrode
    system, crucible bank, heaters, FPGA, PSU, frame/enclosure.
    """
    n = dp.n_transducers
    crucibles = dp.effective_crucibles

    # ---- Power breakdown ----
    array_power = dp.transducer_power_w * n
    bonding_power = dp.sonotrode_power_w
    crucible_power = sum(c.crucible_power_w for c in crucibles)

    # Substrate heater: Stefan-Boltzmann + natural convection estimate
    build_vol_radius = dp.array.cylinder_radius * 0.4  # 40% of array radius
    A_sub = np.pi * build_vol_radius ** 2
    T_sub = dp.substrate_temperature
    T_amb = dp.chamber_temperature
    h_conv_est = 20.0  # W/(m² K), natural convection
    Q_conv = h_conv_est * A_sub * (T_sub - T_amb)
    Q_rad = 0.3 * STEFAN_BOLTZMANN * A_sub * (T_sub ** 4 - T_amb ** 4)
    substrate_power = max(50.0, (Q_conv + Q_rad) * 1.5)  # 1.5× safety

    heating_power = dp.chamber_heater_power_w + substrate_power
    controls_power = dp.fpga_power_w + n * 1.0  # 1W per driver channel

    raw_power = array_power + bonding_power + crucible_power + heating_power + controls_power
    total_power = raw_power / dp.psu_efficiency

    # ---- Cost breakdown (full BOM) ----
    total_cost = (
        n * dp.transducer_cost                  # transducer elements
        + n * dp.driver_cost_per_element        # driver boards
        + dp.array_frame_cost                   # array structure
        + dp.sonotrode_cost                     # bonding subsystem
        + sum(c.crucible_cost for c in crucibles)  # crucible bank
        + dp.chamber_heater_cost
        + dp.substrate_heater_cost
        + dp.fpga_cost
        + dp.psu_cost
        + dp.frame_cost
    )

    return {
        "power_per_element": dp.transducer_power_w,
        "array_power_w": array_power,
        "bonding_power_w": bonding_power,
        "crucible_power_w": crucible_power,
        "heating_power_w": heating_power,
        "controls_power_w": controls_power,
        "total_power": total_power,
        "total_cost": total_cost,
    }


def _eval_multi_droplet(dp: DesignPoint, F_max: float,
                        build_area_override: float = 0.0) -> dict:
    """Multi-crucible throughput constraints.

    Acoustic power splits across simultaneous droplets:
      F_per_drop = F_total / N_simultaneous

    Two regimes:
      SEPARATION_REQUIRED: droplets need λ/2 radial separation (packing limit)
      FREE_SUPERPOSITION: no spatial constraint, only force budget
    """
    import math

    weight = dp.droplet_weight
    force_threshold = 1.0  # F/W must exceed 1.0 per droplet

    # Force budget limit
    force_limit = max(1, int(F_max / (force_threshold * weight))) if weight > 0 else 1

    # Use derived build area if available, else fall back to input
    build_area = build_area_override if build_area_override > 0 else dp.build_area_m2

    # Packing limit (regime-dependent)
    from .models import AcousticRegime
    if dp.acoustic_regime == AcousticRegime.SEPARATION_REQUIRED:
        sep = dp.min_droplet_separation  # λ/2
        exclusion_area = math.pi * sep ** 2
        packing_limit = max(1, int(build_area / exclusion_area)) if exclusion_area > 0 else 1
    else:
        packing_limit = force_limit  # no spatial constraint

    # Effective simultaneous = min(force, packing, crucibles)
    max_N = min(force_limit, packing_limit, dp.n_crucibles)

    F_per = F_max / max_N if max_N > 0 else F_max
    fw_per = F_per / weight if weight > 0 else 0.0

    # Nozzle validation
    nozzle_ok = True
    for c in dp.effective_crucibles:
        expected_d = 1.89 * c.nozzle_diameter
        if abs(expected_d - dp.droplet_diameter) / max(dp.droplet_diameter, 1e-9) > 0.2:
            nozzle_ok = False
            break

    # FGM compositions
    compositions = set(c.mg_weight_pct for c in dp.effective_crucibles)

    return {
        "n_crucibles": dp.n_crucibles,
        "max_simultaneous_drops": max_N,
        "force_per_droplet": F_per,
        "force_weight_per_droplet": fw_per,
        "packing_limit": packing_limit,
        "force_limit": force_limit,
        "nozzle_diameter_ok": nozzle_ok,
        "fgm_compositions_available": len(compositions),
        "atmosphere": dp.atmosphere,
        "sound_speed": dp.atmosphere_sound_speed,
        "wavelength_mm": dp.wavelength * 1e3,
    }


def _eval_throughput(dp: DesignPoint, max_simultaneous: int = 1,
                     fall_time_override: float = 0.0) -> dict:
    """Throughput with multi-crucible time-stepping.

    Layer thickness from D6 Weber-number splat analysis.
    Cycle time = fall + sonotrode dwell + nozzle recharge + settle × (N-1).
    Build rate scales with N_simultaneous crucibles.
    """
    layer_t = dp.droplet_diameter * dp.splat_spreading_ratio * 0.3
    splat_d = dp.droplet_diameter * dp.splat_spreading_ratio

    # Fall time: use override from thermal ODE if available, else free-fall approx
    H = dp.array_height
    if fall_time_override > 0:
        fall_time = fall_time_override
    else:
        fall_time = float(np.sqrt(2.0 * H / GRAVITY)) if H > 0 else 0.0

    # Single-droplet cycle time
    cycle_per_drop = fall_time + dp.sonotrode_dwell_time + dp.nozzle_recharge_time

    # Multi-crucible: N drops pipelined with settle_time between reconfigurations
    N = max(1, max_simultaneous)
    effective_cycle = cycle_per_drop + dp.acoustic_settle_time * (N - 1)
    dps = N / effective_cycle if effective_cycle > 0 else 0.0

    vol_rate = dp.droplet_volume * dps * dp.material_utilization  # m³/s
    build_rate = vol_rate * 3.6e9  # cm³/hr
    build_time_1cm3 = 1.0 / build_rate if build_rate > 0 else float("inf")

    return {
        "layer_thickness": layer_t,
        "splat_diameter": splat_d,
        "cycle_time": effective_cycle,
        "droplets_per_second": dps,
        "build_rate_cm3_hr": build_rate,
        "build_time_1cm_cube_hr": build_time_1cm3,
    }


def _eval_microstructure(dp: DesignPoint) -> dict:
    """D14: Solidification microstructure from cooling rate.

    SDAS = A * R^(-n) where A, n are material properties.
    Grain size ~ 4 * SDAS (empirical for equiaxed).
    Hardness from Hall-Petch: HV = HV0 + k/sqrt(d).
    """
    # Estimate cooling rate from lumped capacitance (Nu=2, conservative)
    d = dp.droplet_diameter
    T_film = (dp.crucible_temperature + dp.chamber_temperature) / 2.0
    k_air = dp.atmosphere_conductivity_at(T_film)
    h_est = 2.0 * k_air / d if d > 0 else 0.0
    A_surf = dp.droplet_surface_area
    m = dp.droplet_mass
    cp = dp.material.specific_heat
    dT = dp.crucible_temperature - dp.chamber_temperature

    cr_est = h_est * A_surf * dT / (m * cp) if (m * cp) > 0 else 0.0

    A_sdas = dp.material.sdas_coefficient
    n_sdas = dp.material.sdas_exponent

    sdas = A_sdas * cr_est ** (-n_sdas) if cr_est > 0 and A_sdas > 0 else 0.0
    grain = 4.0 * sdas

    hv0 = dp.material.hall_petch_hv0
    k_hp = dp.material.hall_petch_k
    hardness = hv0 + k_hp / np.sqrt(grain) if grain > 0 else hv0

    if cr_est > 500:
        regime = "equiaxed_fine"
    elif cr_est > 50:
        regime = "equiaxed_coarse"
    elif cr_est > 5:
        regime = "mixed"
    else:
        regime = "columnar"

    sdas_conv = A_sdas * 50.0 ** (-n_sdas) if A_sdas > 0 else 1.0
    ratio = sdas / sdas_conv if sdas_conv > 0 else 0.0

    return {
        "sdas_um": sdas,
        "grain_size_um": grain,
        "hardness_hv": hardness,
        "microstructure_regime": regime,
        "sdas_ratio_vs_cast": ratio,
    }


def _eval_fgm(dp: DesignPoint) -> dict:
    """D13: FGM composition gradient feasibility."""
    import math as _math

    mg_start = dp.fgm_mg_start_pct
    mg_end = dp.fgm_mg_end_pct
    max_step = dp.fgm_max_step_pct
    delta_mg = abs(mg_end - mg_start)

    # FGM requires distinct compositions across crucibles. If the user
    # asked for a gradient with multiple crucibles but didn't supply
    # crucible_configs, effective_crucibles auto-generates them at the
    # default mg_weight_pct=0 — silently producing a uniform-composition
    # bank that cannot deliver any gradient.
    if (
        delta_mg >= 1e-6
        and dp.n_crucibles > 1
        and not dp.crucible_configs
    ):
        logger.warning(
            "FGM gradient requested (%.2f%% → %.2f%% Mg) with n_crucibles=%d "
            "but crucible_configs is empty. Auto-generated crucibles all carry "
            "the default mg_weight_pct=0 — the gradient will not be realisable. "
            "Pass explicit crucible_configs with distinct mg_weight_pct values.",
            mg_start, mg_end, dp.n_crucibles,
        )

    if delta_mg < 1e-6 or max_step <= 0:
        return {
            "fgm_min_layers": 0,
            "fgm_gradient_height_mm": 0.0,
            "fgm_gradient_achievable": True,
        }

    n_layers = _math.ceil(delta_mg / max_step)
    layer_t = dp.droplet_diameter * dp.splat_spreading_ratio * 0.3
    gradient_height = n_layers * layer_t * 1000  # mm

    achievable = 0.0 <= mg_start <= 10.0 and 0.0 <= mg_end <= 10.0

    return {
        "fgm_min_layers": n_layers,
        "fgm_gradient_height_mm": gradient_height,
        "fgm_gradient_achievable": achievable,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PHYSICS_REGISTRY: Dict[str, PhysicsEval] = {
    "acoustics": _eval_acoustics,
    "array": _eval_array,
    "control": _eval_control,
    "thermal": _eval_thermal,
    "machine": _eval_machine,
    "microstructure": _eval_microstructure,
    "fgm": _eval_fgm,
    # NOTE: throughput and multi_droplet are run after the registry loop
    # because they depend on F_max (from acoustics) and fall_time (from thermal)
}


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------
def evaluate_design(
    dp: DesignPoint,
    constraint_set: Optional[ConstraintSet] = None,
    *,
    use_sim_reach: bool = True,
) -> DesignReport:
    """Evaluate all coupled physics for a DesignPoint.

    Uses the unified ConstraintSet for feasibility checking.
    If no constraint_set is provided, uses l1_constraints().

    `use_sim_reach=True` (default) routes single-crucible reach through a
    closed-loop pathtrack simulation so the design engine inherits any
    controller-model improvement. Set False for hot loops (Monte Carlo,
    sweeps) where the bang-bang upper bound is a sufficient proxy and
    the ~100-300 ms sim cost per sample is unacceptable.
    """
    # Run all registered physics modules. Each evaluator must succeed —
    # silent partial results would poison every downstream computation
    # (build geometry, multi-droplet, throughput, bonding).
    results: dict = {}
    for name, evaluator in PHYSICS_REGISTRY.items():
        try:
            results.update(evaluator(dp))
        except Exception as e:
            raise RuntimeError(
                f"Physics evaluator '{name}' failed: {e}"
            ) from e

    # Build geometry (depends on F_max, fall_time, n_crucibles).
    # These keys are mandatory outputs of acoustics/thermal — index access
    # surfaces a registry contract violation rather than a silent zero.
    F_max = results["F_max"]
    fall_time = results["fall_time"]

    bang_bang_reach = _compute_single_crucible_reach(
        F_max, dp.droplet_mass, fall_time, dp.array.cylinder_radius,
    )
    if use_sim_reach:
        single_reach = _compute_single_crucible_reach_sim(
            dp, ideal_target=bang_bang_reach, fall_time=fall_time,
        )
    else:
        single_reach = bang_bang_reach
    build_area_derived, build_pattern = _compute_build_area(dp.n_crucibles, single_reach)

    # Override max_displacement with physics-based reach
    results["max_displacement"] = single_reach
    results["displacement_ok"] = dp.target_displacement <= single_reach
    results["single_crucible_reach_mm"] = single_reach * 1e3
    results["build_area_cm2"] = build_area_derived * 1e4
    results["build_pattern"] = build_pattern

    # Chamber dimensions (derived from array geometry)
    wall_t = 0.005  # 5mm Al wall
    chamber_r = dp.array.cylinder_radius
    chamber_h = dp.array_height + 0.10  # 10cm clearance
    chamber_vol = np.pi * chamber_r ** 2 * chamber_h * 1e3  # litres
    bv_r = single_reach  # build volume radius = crucible reach
    bv_h = dp.array_height * 0.8
    results["chamber_inner_radius_mm"] = chamber_r * 1e3
    results["chamber_height_mm"] = chamber_h * 1e3
    results["chamber_wall_mm"] = wall_t * 1e3
    results["chamber_volume_L"] = chamber_vol
    results["build_volume_radius_mm"] = bv_r * 1e3
    results["build_volume_height_mm"] = bv_h * 1e3

    # Use derived build area for multi-droplet packing calculations
    multi = _eval_multi_droplet(dp, F_max, build_area_override=build_area_derived)
    results.update(multi)

    max_simul = multi["max_simultaneous_drops"]
    throughput = _eval_throughput(dp, max_simultaneous=max_simul, fall_time_override=fall_time)
    results.update(throughput)

    # Run bonding (needs thermal results)
    T_arr = results["T_arrival"]
    impact_v = results["impact_velocity"]

    orme = orme_remelt(T_arr, dp.material.effective_solidus, dp.substrate_temperature)

    acoustic_result: Optional[AcousticBondResult] = None
    if dp.bonding_mode == BondingMode.ACOUSTIC_BOND:
        acoustic_result = acoustic_bond(
            T_arrival=T_arr,
            T_melting=dp.material.effective_solidus,
            T_substrate=dp.substrate_temperature,
            impact_velocity=impact_v,
            droplet_diameter=dp.droplet_diameter,
            sonotrode_amplitude_um=dp.sonotrode_amplitude_um,
            sonotrode_frequency=dp.sonotrode_frequency,
            solidus_temperature=dp.material.effective_solidus,
        )

    # Build preliminary report (constraints filled in below)
    ka = results.get("ka", 0.0)
    Bi = results.get("Bi", 0.0)
    lf = results.get("liquid_fraction", 0.0)

    report = DesignReport(
        diameter_um=dp.droplet_diameter * 1e6,
        chamber_C=dp.chamber_temperature - 273.15,
        crucible_C=dp.crucible_temperature - 273.15,
        ka=ka,
        ka_valid=ka < 0.5,
        f1=results.get("f1", 0.0),
        f2=results.get("f2", 0.0),
        contrast_factor=results.get("contrast_factor", 0.0),
        F_max=results.get("F_max", 0.0),
        force_to_weight=results.get("force_to_weight", 0.0),
        n_transducers=results.get("n_transducers", 0),
        focused_pressure=results.get("focused_pressure", 0.0),
        nyquist_spacing_ok=results.get("nyquist_spacing_ok", False),
        arc_spacing=results.get("arc_spacing", 0.0),
        alpha_max=results.get("alpha_max", 0.0),
        max_displacement=results.get("max_displacement", 0.0),
        displacement_ok=results.get("displacement_ok", False),
        fall_time=results.get("fall_time", 0.0),
        impact_velocity=results.get("impact_velocity", 0.0),
        Re_avg=results.get("Re_avg", 0.0),
        Nu_avg=results.get("Nu_avg", 0.0),
        h_avg=results.get("h_avg", 0.0),
        tau_conv=results.get("tau_conv", 0.0),
        Bi=Bi,
        T_arrival=T_arr,
        T_arrival_C=T_arr - 273.15,
        temp_drop=results.get("temp_drop", 0.0),
        liquid_fraction_at_landing=lf,
        solidification_time=results.get("solidification_time"),
        orme_ratio=orme.orme_ratio,
        remelting_feasible=orme.remelting_feasible,
        remelt_margin=orme.remelt_margin,
        acoustic_bond_feasible=acoustic_result.feasible if acoustic_result else None,
        acoustic_bond_details=(
            MappingProxyType(acoustic_result.details) if acoustic_result else None
        ),
        cooling_rate=results.get("cooling_rate", 0.0),
        # Microstructure (D14)
        sdas_um=results.get("sdas_um", 0.0),
        grain_size_um=results.get("grain_size_um", 0.0),
        hardness_hv=results.get("hardness_hv", 0.0),
        microstructure_regime=results.get("microstructure_regime", ""),
        sdas_ratio_vs_cast=results.get("sdas_ratio_vs_cast", 0.0),
        # Throughput
        layer_thickness=results.get("layer_thickness", 0.0),
        splat_diameter=results.get("splat_diameter", 0.0),
        cycle_time=results.get("cycle_time", 0.0),
        droplets_per_second=results.get("droplets_per_second", 0.0),
        build_rate_cm3_hr=results.get("build_rate_cm3_hr", 0.0),
        build_time_1cm_cube_hr=results.get("build_time_1cm_cube_hr", 0.0),
        # FGM
        fgm_min_layers=results.get("fgm_min_layers", 0),
        fgm_gradient_height_mm=results.get("fgm_gradient_height_mm", 0.0),
        fgm_gradient_achievable=results.get("fgm_gradient_achievable", False),
        # Multi-crucible / acoustic
        n_crucibles=results.get("n_crucibles", 1),
        max_simultaneous_drops=results.get("max_simultaneous_drops", 1),
        force_per_droplet=results.get("force_per_droplet", 0.0),
        force_weight_per_droplet=results.get("force_weight_per_droplet", 0.0),
        packing_limit=results.get("packing_limit", 0),
        force_limit=results.get("force_limit", 0),
        nozzle_diameter_ok=results.get("nozzle_diameter_ok", True),
        fgm_compositions_available=results.get("fgm_compositions_available", 1),
        single_crucible_reach_mm=results.get("single_crucible_reach_mm", 0.0),
        build_area_cm2=results.get("build_area_cm2", 0.0),
        build_pattern=results.get("build_pattern", ""),
        chamber_inner_radius_mm=results.get("chamber_inner_radius_mm", 0.0),
        chamber_height_mm=results.get("chamber_height_mm", 0.0),
        chamber_wall_mm=results.get("chamber_wall_mm", 5.0),
        chamber_volume_L=results.get("chamber_volume_L", 0.0),
        build_volume_radius_mm=results.get("build_volume_radius_mm", 0.0),
        build_volume_height_mm=results.get("build_volume_height_mm", 0.0),
        # Atmosphere fields are read from the DesignPoint properties — they
        # are not produced by any registry evaluator. Earlier code routed
        # these through results.get(...) and silently returned defaults.
        atmosphere=dp.atmosphere,
        sound_speed=dp.atmosphere_sound_speed,
        wavelength_mm=dp.wavelength * 1e3,
        # Power/cost (full machine)
        power_per_element=results.get("power_per_element", 0.0),
        array_power_w=results.get("array_power_w", 0.0),
        bonding_power_w=results.get("bonding_power_w", 0.0),
        crucible_power_w=results.get("crucible_power_w", 0.0),
        heating_power_w=results.get("heating_power_w", 0.0),
        controls_power_w=results.get("controls_power_w", 0.0),
        total_power=results.get("total_power", 0.0),
        total_cost=results.get("total_cost", 0.0),
    )

    # Post-assembly: refine SDAS with exact cooling rate from thermal ODE
    exact_cr = results.get("cooling_rate", 0.0)
    if exact_cr > 0 and dp.material.sdas_coefficient > 0:
        exact_sdas = dp.material.sdas_coefficient * exact_cr ** (-dp.material.sdas_exponent)
        exact_grain = 4.0 * exact_sdas
        exact_hv = (
            dp.material.hall_petch_hv0
            + dp.material.hall_petch_k / np.sqrt(exact_grain)
            if exact_grain > 0
            else dp.material.hall_petch_hv0
        )
        report = replace(
            report,
            sdas_um=exact_sdas,
            grain_size_um=exact_grain,
            hardness_hv=exact_hv,
        )

    # Unified constraint checking
    cs = constraint_set if constraint_set is not None else l1_constraints()
    constraint_results = cs.check(report)
    feasible = all(cr.satisfied for cr in constraint_results.values())

    # Return final report with constraints populated (immutable mapping)
    return replace(
        report,
        feasible=feasible,
        constraints=MappingProxyType(constraint_results),
    )


# ---------------------------------------------------------------------------
# Thermal ODE solver
# ---------------------------------------------------------------------------
def _simulate_inflight_cooling(dp: DesignPoint) -> dict:
    """Coupled fall + cooling ODE with latent heat and optional thermal gradient.

    Models sensible heat (convection + radiation) AND latent heat of
    fusion via an effective-cp mushy-zone approach.  When T is within
    T_melt +/- _MUSHY_HALFWIDTH_K, cp_eff includes L/(2*dT_mushy).

    When use_thermal_gradient is True, the ambient temperature varies
    linearly with z: hot at the build plate (bottom, z=0), cool at
    the transducers (top, z=H). This models the real chamber where
    the heated bed creates a thermal gradient.
    """
    d = dp.droplet_diameter
    r = dp.droplet_radius
    m = dp.droplet_mass
    A = dp.droplet_surface_area
    cp = dp.material.specific_heat
    L_fus = dp.material.latent_heat
    eps = dp.material.emissivity
    T_init = dp.crucible_temperature
    T_amb_uniform = dp.chamber_temperature
    T_melt = dp.material.effective_solidus
    H = dp.array_height
    dT_mushy = _MUSHY_HALFWIDTH_K

    # Thermal gradient: T_amb(z) = T_bottom + (T_top - T_bottom) * z/H
    use_gradient = dp.use_thermal_gradient
    if use_gradient:
        T_top = dp.gradient_T_top_C + 273.15
        T_bottom = dp.gradient_T_bottom_C + 273.15
    else:
        T_top = T_amb_uniform
        T_bottom = T_amb_uniform

    def _T_amb_at(z: float) -> float:
        """Local ambient temperature at height z."""
        if not use_gradient or H <= 0:
            return T_amb_uniform
        frac = max(0.0, min(1.0, z / H))  # 0 at bottom, 1 at top
        return T_bottom + (T_top - T_bottom) * frac

    def rhs(t: float, y: np.ndarray) -> list[float]:
        T_drop, v, z = float(y[0]), float(y[1]), float(y[2])
        if z <= 0:
            return [0.0, 0.0, 0.0]

        # Local ambient temperature (constant or z-dependent)
        T_amb_local = _T_amb_at(z)

        # Air properties at film temperature
        T_film = (T_drop + T_amb_local) / 2.0
        mu = dp.atmosphere_viscosity_at(T_film)
        k_air = dp.atmosphere_conductivity_at(T_film)
        rho_air = dp.atmosphere_density_at(T_film)

        # Drag (Schiller-Naumann)
        Re = rho_air * abs(v) * d / mu if v > 0.01 else 0.0
        if Re < 1e-10:
            Cd = 0.0
        elif Re < 1.0:
            Cd = 24.0 / Re
        elif Re < 1000.0:
            Cd = (24.0 / Re) * (1.0 + 0.15 * Re ** 0.687)
        else:
            Cd = 0.44

        F_drag = 0.5 * Cd * rho_air * np.pi * r ** 2 * v ** 2
        a = GRAVITY - F_drag / m

        # Ranz-Marshall convection
        Nu = 2.0 + 0.6 * Re ** 0.5 * PR_AIR ** (1.0 / 3.0) if Re > 0.1 else 2.0
        h_conv = Nu * k_air / d

        Q_conv = h_conv * A * (T_drop - T_amb_local)
        Q_rad = eps * STEFAN_BOLTZMANN * A * (T_drop ** 4 - T_amb_local ** 4)
        Q_total = Q_conv + Q_rad

        # Effective cp with latent heat in mushy zone
        if L_fus > 0 and abs(T_drop - T_melt) < dT_mushy:
            cp_eff = cp + L_fus / (2.0 * dT_mushy)
        else:
            cp_eff = cp

        dTdt = -Q_total / (m * cp_eff)
        return [dTdt, a, -v]

    y0 = [T_init, 0.0, H]

    def hit_ground(t: float, y: np.ndarray) -> float:
        return float(y[2])

    hit_ground.terminal = True   # type: ignore[attr-defined]
    hit_ground.direction = -1    # type: ignore[attr-defined]

    # 5 s integration window covers any physically reasonable fall_time for L1
    # (gravity-only fall from 1 m takes ~0.45 s; drag-dominated micro-droplets
    # in heavy gas can extend this, but >5 s would be unphysical).
    sol = solve_ivp(
        rhs, [0.0, 5.0], y0, method="RK45",
        events=hit_ground, max_step=1e-4, rtol=1e-8,
    )

    if sol.status < 0:
        raise RuntimeError(
            f"Thermal ODE failed: {sol.message} "
            f"(droplet={d*1e6:.0f}μm, crucible={T_init-273.15:.0f}°C, "
            f"chamber={T_amb_uniform-273.15:.0f}°C)"
        )
    if not (sol.t_events and len(sol.t_events[0]) > 0):
        raise RuntimeError(
            "Thermal ODE: droplet did not reach ground within 5 s window — "
            f"check inputs (droplet={d*1e6:.0f}μm, array_height={H*1e3:.1f}mm)"
        )

    T_final = float(sol.y[0, -1])
    t_fall = float(sol.t[-1])
    v_final = float(sol.y[1, -1])

    # Solidification time
    t_solid: Optional[float] = None
    for i, T in enumerate(sol.y[0]):
        if T <= T_melt:
            t_solid = float(sol.t[i])
            break

    # Use equilibrium_liquid_fraction for alloy support (mushy zone)
    lf = dp.material.equilibrium_liquid_fraction(T_final)

    # Time-weighted averages of Re, Nu, h via trapezoidal integration.
    # Earlier code used linspace-by-step-index, which oversampled stiff regions
    # (rapid cooling near solidus, deceleration near landing) where RK45
    # naturally clusters its steps.
    T_traj = sol.y[0]
    v_traj = sol.y[1]
    z_traj = sol.y[2]
    Re_traj = np.zeros_like(sol.t)
    Nu_traj = np.zeros_like(sol.t)
    h_traj = np.zeros_like(sol.t)
    for i in range(len(sol.t)):
        T_drop = float(T_traj[i])
        v = float(v_traj[i])
        z_pos = float(z_traj[i])
        T_amb_local = _T_amb_at(z_pos)
        T_film = (T_drop + T_amb_local) / 2.0
        mu = dp.atmosphere_viscosity_at(T_film)
        k_air = dp.atmosphere_conductivity_at(T_film)
        rho_air = dp.atmosphere_density_at(T_film)
        Re_i = rho_air * abs(v) * d / mu if v > 0.01 else 0.0
        Nu_i = 2.0 + 0.6 * Re_i ** 0.5 * PR_AIR ** (1.0 / 3.0) if Re_i > 0.1 else 2.0
        Re_traj[i] = Re_i
        Nu_traj[i] = Nu_i
        h_traj[i] = Nu_i * k_air / d

    if t_fall > 0:
        Re_avg = float(np.trapezoid(Re_traj, sol.t) / t_fall)
        Nu_avg = float(np.trapezoid(Nu_traj, sol.t) / t_fall)
        h_avg = float(np.trapezoid(h_traj, sol.t) / t_fall)
    else:
        Re_avg = float(Re_traj[0]) if len(Re_traj) else 0.0
        Nu_avg = float(Nu_traj[0]) if len(Nu_traj) else 0.0
        h_avg = float(h_traj[0]) if len(h_traj) else 0.0
    tau_conv = (m * cp) / (h_avg * A) if h_avg > 0 else float("inf")

    return {
        "T_arrival": T_final,
        "temp_drop": T_init - T_final,
        "fall_time": t_fall,
        "impact_velocity": v_final,
        "Re_avg": Re_avg,
        "Nu_avg": Nu_avg,
        "h_avg": h_avg,
        "tau_conv": tau_conv,
        "solidification_time": t_solid,
        "liquid_fraction": lf,
    }
