"""
Core data models for the parametric design engine.

DesignPoint composes config objects from drip_physics_core.config —
no field duplication, single source of truth.

DesignPoint and CrucibleConfig are Pydantic v2 ``BaseModel`` so the
backend can derive ``/api/schema`` from ``DesignPoint.model_json_schema()``
without a hand-curated parameter list. Each input field carries
``json_schema_extra`` metadata (``unit``, ``group``, ``display_scale``,
``display_offset``, ``step``, ``hidden``) that the schema endpoint reads
to emit the same shape the frontend already consumes.

DesignReport stays a ``@dataclass(frozen=True)`` for now — output-side
schema derivation is deferred (output fields use ``MappingProxyType``
which Pydantic doesn't natively understand).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

from types import MappingProxyType

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from drip_physics_core.config import (
    ALUMINUM_LIQUID,
    ArrayConfig,
    L1_ARRAY,
    MA40S4S_40KHZ,
    MaterialConfig,
    TransducerConfig,
    GRAVITY,
)


class BondingMode(str, Enum):
    """Which bonding model to use as the primary feasibility gate."""

    ORME_REMELT = "orme_remelt"        # Thermal remelting (Orme & Huang 1997)
    ACOUSTIC_BOND = "acoustic_bond"    # 20 kHz sonotrode solid-state bonding


class AcousticRegime(str, Enum):
    """How multiple simultaneous droplets share the acoustic field."""

    SEPARATION_REQUIRED = "separation_required"
    """Coherent superposition can't create independent traps close together.
    Each droplet needs minimum λ/2 radial separation from others."""

    FREE_SUPERPOSITION = "free_superposition"
    """Holographic phase can create arbitrary independent traps anywhere.
    Only constraint is total acoustic power split across N droplets."""


# ---------------------------------------------------------------------------
# CrucibleConfig
# ---------------------------------------------------------------------------
class CrucibleConfig(BaseModel):
    """Single crucible/nozzle unit in the crucible bank.

    Each crucible has its own induction heater, nozzle, and feedstock.
    Composition can be set independently for real-time FGM alloying.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    nozzle_diameter: float = Field(
        default=318e-6, gt=0,
        description="m (Rayleigh: d_drop ≈ 1.89 × d_nozzle)",
    )
    mg_weight_pct: float = Field(
        default=0.0, ge=0, le=100,
        description="wt% Mg composition loaded",
    )
    crucible_power_w: float = Field(
        default=800.0, ge=0,
        description="W, induction heater per crucible",
    )
    crucible_cost: float = Field(
        default=1000.0, ge=0,
        description="$, heater ($800) + crucible+nozzle ($200)",
    )


# ---------------------------------------------------------------------------
# DesignPoint
# ---------------------------------------------------------------------------
#
# Field metadata convention (json_schema_extra):
#
#   unit            display unit string ("um", "C", "ms", "%", "W", "$", ...)
#   group           UI grouping bucket ("Geometry", "Thermal", "Process", ...)
#   display_scale   multiplier from SI to display (e.g. 1e6 for m→μm)
#   display_offset  additive offset to subtract before display (e.g. 273.15 for K→°C)
#   step            UI slider step in DISPLAY units
#   hidden          if True, schema endpoint suppresses this field from /api/schema
#
# The schema endpoint (`/api/schema`) walks ``DesignPoint.model_json_schema()``,
# extracts each scalar field along with its json_schema_extra metadata, and
# emits the same ``ParamSchema`` shape the frontend already consumes.
# ``hidden=True`` fields are skipped — they remain valid inputs to
# evaluate_design but don't appear in the UI's param list.
#
# Pydantic ``ge``/``le``/``gt``/``lt`` constraints are in SI. The schema
# endpoint converts to display-unit min/max via display_scale + display_offset.
# ---------------------------------------------------------------------------


class DesignPoint(BaseModel):
    """All design variables for a Drip machine configuration.

    Composes MaterialConfig, TransducerConfig, ArrayConfig from config.py.
    Every derived quantity flows from these inputs alone.

    Frozen Pydantic BaseModel — use ``model_copy(update={...})`` to derive
    a modified DesignPoint, not ``dataclasses.replace``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # -- Composed config objects --
    material: MaterialConfig = Field(
        description="Material profile of the droplet feedstock",
        json_schema_extra={"hidden": True, "group": "Material"},
    )
    transducer: TransducerConfig = Field(
        description="Transducer datasheet profile",
        json_schema_extra={"hidden": True, "group": "Acoustic"},
    )
    array: ArrayConfig = Field(
        description="Cylindrical array geometry",
        json_schema_extra={"group": "Array"},
    )

    # -- Scalars not naturally part of any config object --
    droplet_diameter: float = Field(
        ge=1e-4, le=1e-3,
        description="m (100-1000 μm)",
        title="Droplet diameter",
        json_schema_extra={
            "unit": "um", "group": "Geometry",
            "display_scale": 1e6, "step": 10,
        },
    )
    crucible_temperature: float = Field(
        ge=973.15, le=1373.15,
        description="K (700-1100 °C)",
        title="Crucible temperature",
        json_schema_extra={
            "unit": "C", "group": "Thermal",
            "display_scale": 1, "display_offset": -273.15, "step": 5,
        },
    )
    chamber_temperature: float = Field(
        ge=293.15, le=773.15,
        description="K (20-500 °C)",
        title="Chamber temperature",
        json_schema_extra={
            "unit": "C", "group": "Thermal",
            "display_scale": 1, "display_offset": -273.15, "step": 5,
        },
    )

    # -- Defaults --
    substrate_temperature: float = Field(
        default=523.15, ge=293.15, le=773.15,
        description="K (250 °C heated bed; range 20-500 °C)",
        title="Substrate temperature",
        json_schema_extra={
            "unit": "C", "group": "Thermal",
            "display_scale": 1, "display_offset": -273.15, "step": 5,
        },
    )
    target_displacement: float = Field(
        default=5.0e-3, ge=0,
        description="m",
        json_schema_extra={"hidden": True, "unit": "mm", "display_scale": 1e3},
    )
    bonding_mode: BondingMode = Field(
        default=BondingMode.ACOUSTIC_BOND,
        title="Bonding mode",
        json_schema_extra={"group": "Process"},
    )

    # Cost / power per element
    transducer_cost: float = Field(
        default=5.33, ge=0,
        description="$/element",
        json_schema_extra={"hidden": True, "unit": "$", "group": "Cost"},
    )
    transducer_power_w: float = Field(
        default=0.5, ge=0,
        description="W/element",
        json_schema_extra={"hidden": True, "unit": "W", "group": "Power"},
    )

    # Sonotrode parameters (20 kHz bonding subsystem)
    sonotrode_amplitude_um: float = Field(
        default=15.0, ge=2, le=30,
        description="μm peak-to-peak",
        title="Sonotrode amplitude",
        json_schema_extra={
            "unit": "um", "group": "Process",
            "display_scale": 1, "step": 0.5,
        },
    )
    sonotrode_frequency: float = Field(
        default=20_000.0, gt=0,
        description="Hz",
        json_schema_extra={"hidden": True, "unit": "Hz", "group": "Process"},
    )

    # Throughput parameters
    nozzle_recharge_time: float = Field(
        default=5e-3, ge=1e-3, le=20e-3,
        description="s (1-20 ms range; ~5 ms typical)",
        title="Nozzle recharge time",
        json_schema_extra={
            "unit": "ms", "group": "Throughput",
            "display_scale": 1e3, "step": 0.5,
        },
    )
    sonotrode_dwell_time: float = Field(
        default=10e-3, ge=0,
        description="s (1-50 ms, from TE-002)",
        json_schema_extra={"hidden": True, "unit": "ms", "display_scale": 1e3},
    )
    splat_spreading_ratio: float = Field(
        default=1.5, ge=1.0, le=3.0,
        description="dimensionless (1.2-2.0 typical, 1.0-3.0 hard bounds; D6)",
        title="Splat spreading ratio",
        json_schema_extra={"unit": "", "group": "Process", "step": 0.1},
    )
    build_area_m2: float = Field(
        default=1e-4, gt=0,
        description="m² (1 cm², L1 demo scale)",
        json_schema_extra={
            "hidden": True, "unit": "cm2",
            "group": "Geometry", "display_scale": 1e4,
        },
    )
    material_utilization: float = Field(
        default=0.95, ge=0.5, le=1.0,
        description="fraction landing on target (50-100%)",
        title="Material utilization",
        json_schema_extra={
            "unit": "%", "group": "Throughput",
            "display_scale": 100, "step": 1,
        },
    )

    # FGM composition gradient
    fgm_mg_start_pct: float = Field(
        default=0.0, ge=0, le=100,
        description="wt% Mg at bottom",
        json_schema_extra={"hidden": True, "unit": "%", "group": "FGM"},
    )
    fgm_mg_end_pct: float = Field(
        default=4.5, ge=0, le=100,
        description="wt% Mg at top",
        json_schema_extra={"hidden": True, "unit": "%", "group": "FGM"},
    )
    fgm_max_step_pct: float = Field(
        default=1.0, ge=0,
        description="max wt% change per layer (D13)",
        json_schema_extra={"hidden": True, "unit": "%", "group": "FGM"},
    )

    # Thermal gradient (z-dependent chamber temperature)
    use_thermal_gradient: bool = Field(
        default=False,
        description="if True, T_amb varies with z",
        json_schema_extra={"hidden": True, "group": "Thermal"},
    )
    gradient_T_top_C: float = Field(
        default=50.0,
        description="°C at top (transducer end)",
        json_schema_extra={"hidden": True, "unit": "C", "group": "Thermal"},
    )
    gradient_T_bottom_C: float = Field(
        default=300.0,
        description="°C at bottom (build plate)",
        json_schema_extra={"hidden": True, "unit": "C", "group": "Thermal"},
    )

    # Multi-crucible system
    n_crucibles: int = Field(
        default=1, ge=1, le=8,
        description="if empty, auto-generate",
        title="Number of crucibles",
        json_schema_extra={"unit": "", "group": "Crucible", "step": 1},
    )
    crucible_configs: tuple[CrucibleConfig, ...] = Field(
        default=(),
        description="if empty, auto-generated from droplet diameter",
        json_schema_extra={"hidden": True, "group": "Crucible"},
    )

    # Acoustic regime
    acoustic_regime: AcousticRegime = Field(
        default=AcousticRegime.SEPARATION_REQUIRED,
        title="Acoustic regime",
        json_schema_extra={"group": "Acoustic"},
    )
    acoustic_settle_time: float = Field(
        default=1e-3, ge=0,
        description="s, phase reconfiguration time",
        json_schema_extra={"hidden": True, "unit": "ms", "display_scale": 1e3},
    )

    # Atmosphere
    atmosphere: Literal["air", "argon"] = Field(
        default="air",
        title="Atmosphere",
        json_schema_extra={"group": "Chamber"},
    )

    # Machine subsystem power (independent inputs until designs finalized)
    sonotrode_power_w: float = Field(
        default=600.0, ge=100, le=2000,
        description="W, 20 kHz bonding delivered power",
        title="Sonotrode power",
        json_schema_extra={"unit": "W", "group": "Bonding", "step": 50},
    )
    chamber_heater_power_w: float = Field(
        default=150.0, ge=0, le=500,
        description="W, resistive chamber heater",
        title="Chamber heater",
        json_schema_extra={"unit": "W", "group": "Thermal", "step": 10},
    )
    fpga_power_w: float = Field(
        default=10.0, ge=0,
        description="W, control electronics",
        json_schema_extra={"hidden": True, "unit": "W", "group": "Power"},
    )
    psu_efficiency: float = Field(
        default=0.85, gt=0, le=1,
        description="fraction (15% overhead)",
        json_schema_extra={"hidden": True, "unit": "", "group": "Power"},
    )

    # Machine subsystem costs (all hidden — domain-review pending; cost-as-registry refactor will replace)
    driver_cost_per_element: float = Field(
        default=2.50, ge=0,
        description="$/element, phase driver board",
        json_schema_extra={"hidden": True, "unit": "$", "group": "Cost"},
    )
    array_frame_cost: float = Field(
        default=500.0, ge=0,
        description="$, machined Al frame",
        json_schema_extra={"hidden": True, "unit": "$", "group": "Cost"},
    )
    sonotrode_cost: float = Field(
        default=720.0, ge=0,
        description="$, transducer + generator + horn",
        json_schema_extra={"hidden": True, "unit": "$", "group": "Cost"},
    )
    chamber_heater_cost: float = Field(
        default=150.0, ge=0,
        description="$",
        json_schema_extra={"hidden": True, "unit": "$", "group": "Cost"},
    )
    substrate_heater_cost: float = Field(
        default=300.0, ge=0,
        description="$, heater + PID",
        json_schema_extra={"hidden": True, "unit": "$", "group": "Cost"},
    )
    fpga_cost: float = Field(
        default=200.0, ge=0,
        description="$",
        json_schema_extra={"hidden": True, "unit": "$", "group": "Cost"},
    )
    psu_cost: float = Field(
        default=100.0, ge=0,
        description="$",
        json_schema_extra={"hidden": True, "unit": "$", "group": "Cost"},
    )
    frame_cost: float = Field(
        default=400.0, ge=0,
        description="$, structure + enclosure",
        json_schema_extra={"hidden": True, "unit": "$", "group": "Cost"},
    )

    # Air property reference values (for temperature scaling in thermal ODE)
    air_density_ref: float = Field(
        default=1.204, gt=0,
        description="kg/m³ at 293.15 K",
        json_schema_extra={"hidden": True, "unit": "kg/m^3"},
    )
    air_viscosity_ref: float = Field(
        default=1.825e-5, gt=0,
        description="Pa s at 293.15 K",
        json_schema_extra={"hidden": True, "unit": "Pa.s"},
    )
    air_conductivity_ref: float = Field(
        default=0.0257, gt=0,
        description="W/(m K) at 293.15 K",
        json_schema_extra={"hidden": True, "unit": "W/(m.K)"},
    )
    air_temperature_ref: float = Field(
        default=293.15, gt=0,
        description="K",
        json_schema_extra={"hidden": True, "unit": "K"},
    )

    # ---- Derived geometry ----

    @property
    def droplet_radius(self) -> float:
        return self.droplet_diameter / 2.0

    @property
    def droplet_volume(self) -> float:
        return (4.0 / 3.0) * np.pi * self.droplet_radius ** 3

    @property
    def droplet_mass(self) -> float:
        return self.material.density * self.droplet_volume

    @property
    def droplet_weight(self) -> float:
        return self.droplet_mass * GRAVITY

    @property
    def droplet_surface_area(self) -> float:
        return 4.0 * np.pi * self.droplet_radius ** 2

    @property
    def n_transducers(self) -> int:
        return self.array.ring_count * self.array.transducers_per_ring

    @property
    def array_height(self) -> float:
        return (self.array.ring_count - 1) * self.array.ring_spacing

    @property
    def wavelength(self) -> float:
        return self.atmosphere_sound_speed / self.transducer.frequency

    @property
    def effective_crucibles(self) -> tuple[CrucibleConfig, ...]:
        """Crucible configs, auto-generated if not explicitly provided."""
        if self.crucible_configs:
            return self.crucible_configs
        default_nozzle = self.droplet_diameter / 1.89  # Rayleigh inverse
        return tuple(CrucibleConfig(nozzle_diameter=default_nozzle) for _ in range(self.n_crucibles))

    @property
    def min_droplet_separation(self) -> float:
        """Minimum radial separation between simultaneous droplets (λ/2)."""
        return self.wavelength / 2.0

    # ---- Atmosphere-aware properties ----

    @property
    def atmosphere_sound_speed(self) -> float:
        """Speed of sound in the chamber atmosphere at chamber temperature."""
        if self.atmosphere == "argon":
            # Argon: c = 323 m/s at 293K, monatomic ideal gas: c ∝ sqrt(T)
            return 323.0 * float(np.sqrt(self.chamber_temperature / 293.15))
        return self.speed_of_sound_at(self.chamber_temperature)

    @property
    def atmosphere_density(self) -> float:
        """Atmosphere density at chamber temperature."""
        if self.atmosphere == "argon":
            # Argon: ρ = 1.634 kg/m³ at 293K, ideal gas: ρ ∝ 1/T
            return 1.634 * (293.15 / self.chamber_temperature)
        return self.air_density_at(self.chamber_temperature)

    @property
    def atmosphere_viscosity(self) -> float:
        """Atmosphere dynamic viscosity at chamber temperature."""
        if self.atmosphere == "argon":
            # Argon: μ = 2.27e-5 Pa·s at 293K, scales ∝ T^0.72
            return 2.27e-5 * (self.chamber_temperature / 293.15) ** 0.72
        return self.air_viscosity_at(self.chamber_temperature)

    @property
    def atmosphere_conductivity(self) -> float:
        """Atmosphere thermal conductivity at chamber temperature."""
        if self.atmosphere == "argon":
            # Argon: k = 0.0177 W/(m·K) at 293K (~69% of air), scales ∝ T^0.72
            return 0.0177 * (self.chamber_temperature / 293.15) ** 0.72
        return self.air_conductivity_at(self.chamber_temperature)

    # ---- Air property scaling (legacy, used when atmosphere=="air") ----

    def speed_of_sound_at(self, T: float) -> float:
        """Speed of sound in air at temperature T [K]."""
        return 331.3 * float(np.sqrt(T / 273.15))

    def air_density_at(self, T: float) -> float:
        return self.air_density_ref * (self.air_temperature_ref / T)

    def air_viscosity_at(self, T: float) -> float:
        return self.air_viscosity_ref * (T / self.air_temperature_ref) ** 0.7

    def air_conductivity_at(self, T: float) -> float:
        return self.air_conductivity_ref * (T / self.air_temperature_ref) ** 0.8

    # ---- Atmosphere-aware temperature scaling (for thermal ODE) ----

    def atmosphere_density_at(self, T: float) -> float:
        if self.atmosphere == "argon":
            return 1.634 * (293.15 / T)
        return self.air_density_at(T)

    def atmosphere_viscosity_at(self, T: float) -> float:
        if self.atmosphere == "argon":
            return 2.27e-5 * (T / 293.15) ** 0.72
        return self.air_viscosity_at(T)

    def atmosphere_conductivity_at(self, T: float) -> float:
        if self.atmosphere == "argon":
            return 0.0177 * (T / 293.15) ** 0.72
        return self.air_conductivity_at(T)


# ---------------------------------------------------------------------------
# DesignReport (frozen dataclass — output side, schema derivation deferred)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DesignReport:
    """Complete parametric evaluation of a DesignPoint."""

    # Input echo
    diameter_um: float
    chamber_C: float
    crucible_C: float

    # D1/D2: Acoustics
    ka: float
    ka_valid: bool
    f1: float
    f2: float
    contrast_factor: float
    F_max: float                        # N
    force_to_weight: float

    # D3: Array
    n_transducers: int
    focused_pressure: float             # Pa
    nyquist_spacing_ok: bool
    arc_spacing: float                  # m

    # D4: Control
    alpha_max: float                    # rad/m
    max_displacement: float             # m
    displacement_ok: bool

    # D5: Thermal (in-flight)
    fall_time: float                    # s
    impact_velocity: float              # m/s
    Re_avg: float
    Nu_avg: float
    h_avg: float                        # W/(m² K)
    tau_conv: float                     # s
    Bi: float
    T_arrival: float                    # K
    T_arrival_C: float
    temp_drop: float                    # K
    liquid_fraction_at_landing: float
    solidification_time: Optional[float]  # s or None

    # Orme bonding (always computed, informational)
    orme_ratio: float
    remelting_feasible: bool
    remelt_margin: float

    # Acoustic bonding (primary when mode=ACOUSTIC_BOND)
    acoustic_bond_feasible: Optional[bool] = None
    acoustic_bond_details: Optional[MappingProxyType] = None

    # D14: Cooling rate + microstructure
    cooling_rate: float = 0.0           # K/s
    sdas_um: float = 0.0               # um, secondary dendrite arm spacing
    grain_size_um: float = 0.0         # um, estimated grain diameter
    hardness_hv: float = 0.0           # Vickers hardness
    microstructure_regime: str = ""    # equiaxed_fine / equiaxed_coarse / mixed / columnar
    sdas_ratio_vs_cast: float = 0.0   # lambda_drip / lambda_conventional

    # Throughput
    layer_thickness: float = 0.0       # m
    splat_diameter: float = 0.0        # m
    cycle_time: float = 0.0            # s per droplet
    droplets_per_second: float = 0.0   # Hz
    build_rate_cm3_hr: float = 0.0     # cm³/hr
    build_time_1cm_cube_hr: float = 0.0  # hours for 1 cm³

    # FGM composition
    fgm_min_layers: int = 0
    fgm_gradient_height_mm: float = 0.0
    fgm_gradient_achievable: bool = False

    # Multi-crucible / acoustic
    n_crucibles: int = 1
    max_simultaneous_drops: int = 1
    force_per_droplet: float = 0.0          # F_max / N_simultaneous [N]
    force_weight_per_droplet: float = 0.0   # force_per_droplet / weight
    packing_limit: int = 0                  # build_area / (π × (λ/2)²)
    force_limit: int = 0                    # floor(F_max / (threshold × weight))
    nozzle_diameter_ok: bool = True
    fgm_compositions_available: int = 1     # distinct compositions in crucible bank

    # Build geometry (derived from physics)
    single_crucible_reach_mm: float = 0.0   # max steering displacement per crucible
    build_area_cm2: float = 0.0             # total build area from packing
    build_pattern: str = ""                 # packing pattern name

    # Atmosphere
    atmosphere: str = "air"
    sound_speed: float = 0.0                # m/s at chamber temp
    wavelength_mm: float = 0.0              # mm

    # Chamber dimensions (derived from array geometry)
    chamber_inner_radius_mm: float = 0.0
    chamber_height_mm: float = 0.0
    chamber_wall_mm: float = 5.0
    chamber_volume_L: float = 0.0
    build_volume_radius_mm: float = 0.0
    build_volume_height_mm: float = 0.0

    # D9-11: Power / Cost (full machine)
    power_per_element: float = 0.0
    array_power_w: float = 0.0
    bonding_power_w: float = 0.0
    crucible_power_w: float = 0.0
    heating_power_w: float = 0.0
    controls_power_w: float = 0.0
    total_power: float = 0.0               # full machine, including PSU overhead
    total_cost: float = 0.0                # full BOM

    # Overall
    feasible: bool = False
    constraints: MappingProxyType = field(
        default_factory=lambda: MappingProxyType({})
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def l1_design_point(
    droplet_diameter: float = 600e-6,
    crucible_temperature_C: float = 850.0,
    chamber_temperature_C: float = 200.0,
    substrate_temperature_C: float = 250.0,
    bonding_mode: BondingMode = BondingMode.ACOUSTIC_BOND,
) -> DesignPoint:
    """Construct the L1 prototype DesignPoint with Al liquid defaults.

    L1 strategy: 600 μm droplets in air at 200°C chamber.
    Droplet arrives liquid (above solidus) for ultrasonic cavitation bonding.
    Scale down to 300-500 μm when argon atmosphere is available (L2).
    """
    return DesignPoint(
        material=ALUMINUM_LIQUID,
        transducer=MA40S4S_40KHZ,
        array=L1_ARRAY,
        droplet_diameter=droplet_diameter,
        crucible_temperature=crucible_temperature_C + 273.15,
        chamber_temperature=chamber_temperature_C + 273.15,
        substrate_temperature=substrate_temperature_C + 273.15,
        bonding_mode=bonding_mode,
    )
