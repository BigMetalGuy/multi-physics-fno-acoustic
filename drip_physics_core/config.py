"""
Config system for the physics engine (SW-16, Section 1).

Single source of truth for all physical constants and derived quantities.
No physical constant is set directly anywhere else in the codebase —
everything is derived from structured config at runtime.

Public types:
    MaterialConfig    — Pydantic v2 — density, bulk modulus, viscosity, surface tension
    TransducerConfig  — Pydantic v2 — frequency, reference pressure, aperture
    ArrayConfig       — Pydantic v2 — cylinder geometry (radius, rings, spacing)
    DropletConfig     — frozen dataclass — diameter, initial conditions, drag regime
    EnvironmentConfig — frozen dataclass — medium, temperature, derived quantities
    ThermalConfig     — frozen dataclass — heat-transfer corrections
    PressureConfig    — frozen dataclass — pressure backend selection

The first three (MaterialConfig, TransducerConfig, ArrayConfig) are
Pydantic v2 ``BaseModel`` because they appear inside ``DesignPoint`` and
need to participate in ``DesignPoint.model_json_schema()`` so the design
backend can derive its ``/api/schema`` endpoint without a hand-curated
list. Read access (``mc.density``, ``arr.ring_count``) is unchanged
versus the previous ``@dataclass(frozen=True)`` form. Construction by
keyword (``MaterialConfig(name=..., density=..., ...)``) is unchanged.

DropletConfig / EnvironmentConfig / ThermalConfig / PressureConfig stay
as frozen dataclasses because they live in the simulation engine's hot
path and never appear nested inside ``DesignPoint``.

Required profiles:
    STYROFOAM         — ~25 kg/m³, first hardware test material
    ALUMINUM_LIQUID   — 2385 kg/m³, c=4700 m/s (liquid, not solid)
    ALUMINUM_SOLID    — 2700 kg/m³, c=6420 m/s (reference only)
    MA40S4S_40KHZ     — Murata 40 kHz transducer
    AIR_20C           — Air at 20°C
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator


# =============================================================================
# Constants
# =============================================================================

GRAVITY = 9.81  # m/s²


# =============================================================================
# MaterialConfig (Pydantic)
# =============================================================================


class MaterialConfig(BaseModel):
    """Material properties for a droplet or medium.

    Sound speed is DERIVED from bulk_modulus and density: c = sqrt(K / rho).
    Never hardcode sound speed — change bulk_modulus instead.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Human-readable label")
    density: float = Field(gt=0, description="kg/m³")
    bulk_modulus: float = Field(gt=0, description="Pa (sound speed derived)")
    dynamic_viscosity: float = Field(ge=0, description="Pa·s")
    surface_tension: float = Field(ge=0, description="N/m")
    compressibility: float = Field(description="1/Pa  (β = 1/K, for Gor'kov f₁)")
    state: Literal["solid", "liquid", "gas"] = "solid"
    specific_heat: float = Field(default=0.0, ge=0, description="J/(kg·K)")
    latent_heat: float = Field(default=0.0, ge=0, description="J/kg")
    emissivity: float = Field(default=0.0, ge=0, le=1, description="dimensionless")
    solidus_temperature: float = Field(default=0.0, ge=0, description="K")
    liquidus_temperature: float = Field(
        default=0.0, ge=0,
        description="K (0 = same as solidus, i.e. pure metal)",
    )
    thermal_conductivity: float = Field(default=0.0, ge=0, description="W/(m·K)")
    mg_weight_fraction: float = Field(
        default=0.0, ge=0, le=1, description="Mg content, 0-1 (for Al-Mg alloys)",
    )
    # Microstructure (D14): SDAS = A * cooling_rate^(-n) [um]
    sdas_coefficient: float = Field(
        default=50.0, gt=0,
        description="A, um·(K/s)^n (Kurz & Fisher 1998)",
    )
    sdas_exponent: float = Field(
        default=0.33, gt=0,
        description="n, universal for dendritic solidification",
    )
    hall_petch_hv0: float = Field(default=25.0, ge=0, description="HV, base Vickers hardness")
    hall_petch_k: float = Field(
        default=40.0, ge=0,
        description="HV·um^0.5, Hall-Petch coefficient",
    )
    # Solidification model parameters (Scheil equation; default = no partitioning → lever rule)
    partition_coefficient: float = Field(
        default=1.0, gt=0,
        description="k, solute partition coefficient (1.0 = no partitioning)",
    )
    eutectic_temperature: float = Field(
        default=0.0, ge=0,
        description="K, eutectic temperature (0 = use solidus_temperature fallback)",
    )

    @property
    def effective_solidus(self) -> float:
        """Solidus temperature [K]. Falls back to solidus_temperature."""
        return self.solidus_temperature

    @property
    def effective_liquidus(self) -> float:
        """Liquidus temperature [K]. Falls back to solidus if liquidus not set."""
        if self.liquidus_temperature > 0.0:
            return self.liquidus_temperature
        return self.solidus_temperature

    @property
    def effective_eutectic(self) -> float:
        """Eutectic temperature [K] for Scheil model lower bound.

        Falls back to solidus_temperature if eutectic not set.
        For Al-Mg binary, eutectic is ~723 K (450 C).
        """
        if self.eutectic_temperature > 0.0:
            return self.eutectic_temperature
        return self.solidus_temperature

    def lever_rule_liquid_fraction(self, temperature: float) -> float:
        """Equilibrium liquid fraction at given temperature via lever rule.

        Assumes perfect mixing in both liquid and solid phases.
        Appropriate for slow cooling rates (< 100 K/s).

        For pure metals (solidus == liquidus): binary 0 or 1.
        For alloys (solidus < liquidus): linear interpolation.

        Returns:
            0.0 (fully solid) to 1.0 (fully liquid).
        """
        T_s = self.effective_solidus
        T_l = self.effective_liquidus
        if T_s <= 0.0:
            return 1.0  # no solidification data
        if T_l <= T_s:
            # Pure metal: binary transition
            return 1.0 if temperature >= T_s else 0.0
        # Alloy: linear lever rule between solidus and liquidus
        if temperature >= T_l:
            return 1.0
        if temperature <= T_s:
            return 0.0
        return (temperature - T_s) / (T_l - T_s)

    def scheil_liquid_fraction(self, temperature: float) -> float:
        """Scheil equation liquid fraction for rapid solidification.

        f_L = ((T - T_eutectic) / (T_l - T_eutectic))^(1/(1-k))

        More accurate than lever rule at cooling rates > 100 K/s.
        Assumes no diffusion in solid, complete mixing in liquid.

        For k = 1.0 (no partitioning), falls back to lever rule.

        Returns:
            0.0 (fully solid) to 1.0 (fully liquid).
        """
        k = self.partition_coefficient
        if k >= 1.0:
            return self.lever_rule_liquid_fraction(temperature)

        T_l = self.effective_liquidus
        T_e = self.effective_eutectic

        if T_l <= 0.0 or T_l <= T_e:
            return 1.0  # no solidification data

        if temperature >= T_l:
            return 1.0
        if temperature <= T_e:
            return 0.0

        # Scheil equation: f_L = ((T - T_e) / (T_l - T_e))^(1/(1-k))
        fraction = (temperature - T_e) / (T_l - T_e)
        exponent = 1.0 / (1.0 - k)
        return float(fraction ** exponent)

    def equilibrium_liquid_fraction(self, temperature: float) -> float:
        """Liquid fraction using the appropriate model for this material.

        Uses Scheil equation when partition_coefficient != 1.0 (alloys with
        known partitioning data, rapid solidification). Falls back to lever
        rule when partition_coefficient == 1.0 (pure metals or unknown k).

        Returns:
            0.0 (fully solid) to 1.0 (fully liquid).
        """
        if self.partition_coefficient < 1.0:
            return self.scheil_liquid_fraction(temperature)
        return self.lever_rule_liquid_fraction(temperature)

    @property
    def sound_speed(self) -> float:
        """Speed of sound in material: c = sqrt(K / ρ)  [m/s]."""
        return float(np.sqrt(self.bulk_modulus / self.density))

    def __repr__(self) -> str:
        return (
            f"MaterialConfig('{self.name}', ρ={self.density} kg/m³, "
            f"c={self.sound_speed:.1f} m/s, state={self.state})"
        )


# --- Built-in profiles -------------------------------------------------------

STYROFOAM = MaterialConfig(
    name="styrofoam",
    density=25.0,                          # kg/m³ (EPS, typical 20-30)
    bulk_modulus=300_000.0,                 # Pa (~300 kPa, EPS range 200-500 kPa)
    dynamic_viscosity=0.0,                 # N/A for solid
    surface_tension=0.0,                   # N/A for solid
    compressibility=1.0 / 300_000.0,       # 1/K
    state="solid",
)
# Verify: c = sqrt(300e3 / 25) ≈ 109.5 m/s  ✓

ALUMINUM_LIQUID = MaterialConfig(
    name="aluminum_liquid",
    density=2385.0,                        # kg/m³ (liquid Al near melting point)
    bulk_modulus=2385.0 * 4700.0**2,       # Pa — gives c = 4700 m/s
    dynamic_viscosity=1.3e-3,              # Pa·s (liquid Al at ~660°C)
    surface_tension=0.87,                  # N/m (liquid Al)
    compressibility=1.0 / (2385.0 * 4700.0**2),
    state="liquid",
    specific_heat=1080.0,                  # J/(kg·K)
    latent_heat=397_000.0,                 # J/kg
    emissivity=0.15,                       # [ASMP-036] dimensionless — D5 used 0.20, range 0.05-0.25
    solidus_temperature=933.15,            # K (660°C) — pure Al: solidus == liquidus
    liquidus_temperature=933.15,           # K (660°C) — pure Al: no mushy zone
    thermal_conductivity=91.0,             # W/(m·K)
    mg_weight_fraction=0.0,
)
# Verify: c = sqrt(K / ρ) = sqrt(2385 * 4700² / 2385) = 4700 m/s  ✓

ALUMINUM_SOLID = MaterialConfig(
    name="aluminum_solid",
    density=2700.0,                        # kg/m³
    bulk_modulus=2700.0 * 6420.0**2,       # Pa — gives c = 6420 m/s
    dynamic_viscosity=0.0,                 # N/A for solid
    surface_tension=0.0,                   # N/A for solid
    compressibility=1.0 / (2700.0 * 6420.0**2),
    state="solid",
    specific_heat=900.0,                   # J/(kg·K)
    latent_heat=0.0,                       # J/kg (already solidified)
    emissivity=0.09,                       # dimensionless (solid Al)
    solidus_temperature=933.15,            # K (660°C)
    liquidus_temperature=933.15,           # K (660°C)
    thermal_conductivity=237.0,            # W/(m·K)
    mg_weight_fraction=0.0,
)
# Verify: c = 6420 m/s  ✓


# --- Al-Mg alloy profiles (from internal alloy_standards.json reference) ------
#
# Al-Mg phase diagram (well-established binary):
#   Solidus: T_s(wt%Mg) ≈ 933 - 19.1 × Mg_wt% [K] (calibrated to portal 5083)
#   Liquidus: T_l(wt%Mg) ≈ 933 - 3.4 × Mg_wt% [K] (linear fit, 0-5 wt% Mg)
#   Source: ASM Binary Phase Diagrams, verified against portal solidus/liquidus
#   Note: These are equilibrium values; rapid solidification shifts both lower.
#
# Viscosity model from portal (physics.py):
#   η(T_C) = 0.85e-3 × exp(3000 / (T_C + 273.15))  [Pa·s]
#   At 660°C: η ≈ 1.3e-3 Pa·s (matches ALUMINUM_LIQUID above)
#   At 850°C: η ≈ 0.95e-3 Pa·s


def aluminum_viscosity(temperature_K: float) -> float:
    """Temperature-dependent Al dynamic viscosity [Pa·s].

    Arrhenius model: η = η₀ × exp(E_a / (R × T))
    Fitted to literature values (Iida & Guthrie, 1988):
        η(933 K) = 1.30 mPa·s  (at melting point)
        η(1123 K) = 0.90 mPa·s (at 850°C superheat)

    Note: the internal portal (physics.py) has a viscosity model
    (0.85e-3 × exp(3000/T)) but it gives ~21 mPa·s at 660°C — a 16×
    error vs literature. This function uses corrected Arrhenius parameters.

    Valid range: ~900-1500 K (liquid aluminum and Al-Mg alloys).
    Mg content has weak effect on viscosity (<10% at 5 wt% Mg).

    Args:
        temperature_K: Temperature in Kelvin.
    """
    _ETA0 = 1.479e-4   # Pa·s (pre-exponential factor)
    _EA_OVER_R = 2028.0  # K (activation energy / gas constant)
    return _ETA0 * np.exp(_EA_OVER_R / temperature_K)


def make_al_mg_alloy(
    mg_weight_percent: float,
    temperature_K: float = 1123.15,
    name: Optional[str] = None,
) -> MaterialConfig:
    """Create an Al-Mg alloy MaterialConfig at given composition and temperature.

    Uses Al-Mg binary phase diagram for solidus/liquidus and linear
    interpolation of thermal properties between pure Al and portal 5083
    data (4.5 wt% Mg endpoint).

    Args:
        mg_weight_percent: Mg content in weight percent (0-5 range).
        temperature_K: Initial droplet temperature [K] for viscosity calc.
        name: Optional name override. Defaults to "al_mg_{pct}".

    Returns:
        MaterialConfig for the alloy in liquid state.

    Source: internal alloy_standards.json (5052, 5083, 5086 data),
            ASM Binary Phase Diagrams for Al-Mg.
    """
    mg = mg_weight_percent
    if not (0.0 <= mg <= 10.0):
        raise ValueError(f"Mg content must be 0-10 wt%, got {mg}")

    # --- Phase diagram (ASM binary + portal validation) ---
    T_solidus_K = 933.15 - 19.1 * mg    # K
    T_liquidus_K = 933.15 - 6.7 * mg    # K

    # --- Density: linear interpolation ---
    rho_solid = 2700.0 - 8.0 * mg  # kg/m³ (solid, room temp, from portal trend)
    rho_liquid = rho_solid * 0.89   # ~11% volume expansion on melting

    # --- Thermal conductivity: drops with Mg content ---
    k_solid = 222.0 - 23.0 * mg    # W/(m·K), linear fit to portal data
    k_liquid = k_solid * 0.42       # liquid Al thermal conductivity ratio

    # --- Specific heat: slight increase with Mg ---
    cp_liquid = 1080.0 + 5.0 * mg   # J/(kg·K), weak Mg dependence

    # --- Latent heat: decreases slightly with Mg ---
    L = 397_000.0 - 5400.0 * mg     # J/kg

    # --- Viscosity at given temperature ---
    eta = aluminum_viscosity(temperature_K)

    # --- Sound speed in liquid: decreases with Mg ---
    c_liquid = 4700.0 - 60.0 * mg   # m/s
    K_bulk = rho_liquid * c_liquid**2  # Pa

    alloy_name = name or f"al_mg_{mg:.1f}"

    return MaterialConfig(
        name=alloy_name,
        density=rho_liquid,
        bulk_modulus=K_bulk,
        dynamic_viscosity=eta,
        surface_tension=0.87 - 0.01 * mg,   # N/m, slight decrease with Mg
        compressibility=1.0 / K_bulk,
        state="liquid",
        specific_heat=cp_liquid,
        latent_heat=L,
        emissivity=0.15,
        solidus_temperature=T_solidus_K,
        liquidus_temperature=T_liquidus_K,
        thermal_conductivity=k_liquid,
        mg_weight_fraction=mg / 100.0,
    )


# Pre-built alloy profiles matching portal 5xxx series
AL_MG_2_5 = make_al_mg_alloy(2.5, name="al_5052")   # 5052: Mg 2.2-2.8%
AL_MG_4_5 = make_al_mg_alloy(4.5, name="al_5083")   # 5083: Mg 4.0-4.9%
AL_MG_4_0 = make_al_mg_alloy(4.0, name="al_5086")   # 5086: Mg 3.5-4.5%

AIR_20C = MaterialConfig(
    name="air_20c",
    density=1.204,                         # kg/m³ at 20°C, 1 atm
    bulk_modulus=1.204 * 343.0**2,         # Pa — adiabatic: K = γP ≈ ρc² for ideal gas (0.3% at 20°C)
    dynamic_viscosity=1.825e-5,            # Pa·s (air at 20°C)
    surface_tension=0.0,                   # N/A for gas
    compressibility=1.0 / (1.204 * 343.0**2),
    state="gas",
    specific_heat=1005.0,                  # J/(kg·K) at 20°C, 1 atm
    thermal_conductivity=0.0257,           # W/(m·K) at 20°C
)

AIR_200C = MaterialConfig(
    name="air_200c",
    density=0.746,                         # kg/m³ at 200°C, 1 atm (ideal gas: ρ ∝ 1/T)
    bulk_modulus=0.746 * 436.0**2,         # Pa — c≈436 m/s at 473 K
    dynamic_viscosity=2.59e-5,             # Pa·s at 200°C
    surface_tension=0.0,                   # N/A for gas
    compressibility=1.0 / (0.746 * 436.0**2),
    state="gas",
    specific_heat=1023.0,                  # J/(kg·K) at 473 K
    thermal_conductivity=0.0383,           # W/(m·K) at 473 K
)


# =============================================================================
# ThermalConfig (frozen dataclass — sim engine internal, not in DesignPoint)
# =============================================================================

@dataclass(frozen=True)
class ThermalConfig:
    """Thermal correction parameters for acoustic levitation heat transfer.

    These are parameterized uncertainty values — not derived from first principles.
    See ASSUMPTIONS.md for classification.

    Attributes:
        C_node: velocity-node correction at pressure antinode
        C_trap: Yarin toroidal insulation correction
        streaming_velocity: m/s, Eckart streaming estimate
        ambient_temperature: K (chamber default)
        initial_droplet_temp: K (ejection temperature)
    """
    C_node: float = 0.2              # [ASMP-042] velocity-node correction — weak basis for falling droplet
    C_trap: float = 0.6              # [ASMP-043] Yarin toroidal insulation — levitation literature, not falling
    streaming_velocity: float = 2.0  # [ASMP-044] m/s, order-of-magnitude Eckart estimate (could be 0.5-5)
    ambient_temperature: float = 573.15  # [ASMP-045] K (300 C) — assumes uniform chamber, real gradient unmodeled
    initial_droplet_temp: float = 1123.15  # [ASMP-046] K (850 C) — D5 used 700 C (973 K)


DEFAULT_THERMAL = ThermalConfig()


# =============================================================================
# PressureConfig (frozen dataclass)
# =============================================================================

@dataclass(frozen=True)
class PressureConfig:
    """Configuration for the pluggable pressure computation backend.

    Attributes:
        backend: Which solver to use. One of "analytical", "jwave",
                 "femcoupled", "fenics", "kwave", "compare".
        grid_n: Cubic grid resolution per axis for PDE solvers. Used when
                ``grid_shape`` is ``None``. Kept for backward compatibility
                with existing callers; new code should prefer ``grid_shape``.
        grid_shape: Per-axis grid resolution ``(Nx, Ny, Nz)``. When set,
                    takes precedence over ``grid_n``. Use this for
                    non-cubic (anisotropic) grids, e.g. a tall narrow
                    chamber where x,y can be coarser than z.
        grid_dx: Grid spacing in meters (lambda/10 at 40 kHz by default).
        comparison_log_threshold: Only log discrepancies above this
                                  relative error (0.05 = 5%).
        comparison_primary: Which backend returns the value in compare mode.
        pml_width: PML layer width [m] for FEM backends. Default 4 mm
                   (Phase 2.1 / Phase 4 default).
        crucible_inlet_radius: Crucible-inlet disk radius [m] for the
                                FEM chamber mesh.  Default 2.5 mm.
        mesh_path: Optional explicit ``.msh`` path for the FEM backend.
                   When ``None`` the backend builds (or reuses cached)
                   mesh on first call.
    """
    backend: str = "analytical"
    grid_n: int = 128
    grid_shape: Optional[Tuple[int, int, int]] = None
    grid_dx: float = 0.858e-3   # lambda/10 at 40 kHz
    comparison_log_threshold: float = 0.05
    comparison_primary: str = "analytical"
    # FEM backend extras (ignored by analytical / jwave / kwave / fenics).
    pml_width: float = 4e-3
    crucible_inlet_radius: float = 2.5e-3
    mesh_path: Optional[str] = None

    @property
    def effective_grid_shape(self) -> Tuple[int, int, int]:
        """Per-axis grid shape used by solvers. Falls back to cubic ``grid_n``
        when ``grid_shape`` is ``None`` (backward-compatible default)."""
        if self.grid_shape is not None:
            return tuple(int(g) for g in self.grid_shape)
        return (int(self.grid_n), int(self.grid_n), int(self.grid_n))


# =============================================================================
# TransducerConfig (Pydantic)
# =============================================================================


class TransducerConfig(BaseModel):
    """Single transducer element specification.

    Reference pressure p_ref is measured at distance r_ref from the
    transducer face (from datasheet). Never extrapolate beyond datasheet.

    Directivity is DERIVED from aperture_radius via sin(ka·sinθ)/(ka·sinθ)
    (unnormalized sinc).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Human-readable label")
    frequency: float = Field(gt=0, description="Hz")
    p_ref: float = Field(gt=0, description="Pa at r_ref (from datasheet)")
    r_ref: float = Field(gt=0, description="m (reference distance for p_ref)")
    aperture_radius: float = Field(
        gt=0,
        description="m (half of element diameter)",
    )

    @property
    def wavelength_in_air(self) -> float:
        """Wavelength at 20°C in air (approximate convenience value).

        Uses hardcoded 343.0 m/s for speed of sound. Callers needing
        temperature-accurate values should use
        ``EnvironmentConfig.speed_of_sound`` instead.
        """
        return 343.0 / self.frequency

    def wavelength(self, sound_speed: float) -> float:
        """Wavelength in a medium with given sound speed [m]."""
        return sound_speed / self.frequency

    def wavenumber(self, sound_speed: float) -> float:
        """Wavenumber k = 2π/λ [rad/m]."""
        return 2.0 * np.pi * self.frequency / sound_speed

    def directivity(self, theta: float, sound_speed: float) -> float:
        """Directivity factor D(θ) = sin(ka·sinθ)/(ka·sinθ) (unnormalized sinc).

        Args:
            theta: Angle from normal in radians
            sound_speed: Speed of sound in medium [m/s]

        Returns:
            Directivity factor in [0, 1]
        """
        ka = self.wavenumber(sound_speed) * self.aperture_radius
        arg = ka * np.sin(theta)
        return float(np.sinc(arg / np.pi))  # np.sinc uses sinc(x) = sin(πx)/(πx)


# --- Built-in profile ---------------------------------------------------------

MA40S4S_40KHZ = TransducerConfig(
    name="MA40S4S_40KHZ",
    frequency=40_000.0,                    # Hz
    p_ref=10.0,                            # Pa at 0.3 m (Murata datasheet, ~114 dB SPL)
    r_ref=0.3,                             # m
    aperture_radius=2.5e-3,                # m (5 mm diameter element)
)


# =============================================================================
# ArrayConfig (Pydantic)
# =============================================================================


class ArrayConfig(BaseModel):
    """Transducer array geometry specification.

    Describes a cylindrical array of rings. Actual transducer positions
    are generated from this config by geometry.py.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cylinder_radius: float = Field(
        gt=0, description="m",
        json_schema_extra={"hidden": True, "unit": "mm", "display_scale": 1e3},
    )
    ring_count: int = Field(
        ge=1, le=20, description="number of horizontal rings",
        title="Array rings",
        json_schema_extra={
            "api_key": "array_rings",
            "unit": "", "group": "Array", "step": 1,
        },
    )
    transducers_per_ring: int = Field(
        ge=4, le=24, description="elements per ring",
        title="Emitters per ring",
        json_schema_extra={
            "api_key": "transducers_per_ring",
            "unit": "", "group": "Array", "step": 1,
        },
    )
    ring_spacing: float = Field(
        gt=0, description="m (vertical distance between rings)",
        json_schema_extra={"hidden": True, "unit": "mm", "display_scale": 1e3},
    )

    @property
    def total_transducers(self) -> int:
        return self.ring_count * self.transducers_per_ring

    @property
    def height(self) -> float:
        """Total array height [m]."""
        return (self.ring_count - 1) * self.ring_spacing


# --- Built-in profile ---------------------------------------------------------

L1_ARRAY = ArrayConfig(
    cylinder_radius=0.05,          # 5 cm
    ring_count=10,
    transducers_per_ring=12,
    ring_spacing=0.04,             # 4 cm between rings → 36 cm total height
)


# =============================================================================
# DropletConfig (frozen dataclass — sim engine internal, not in DesignPoint)
# =============================================================================

@dataclass(frozen=True)
class DropletConfig:
    """Droplet physical description and initial conditions.

    Re and drag regime are computed at runtime from diameter, velocity,
    and medium viscosity — never hardcoded.

    ka << 1 (Gor'kov validity) is checked automatically on construction.

    Attributes:
        material: MaterialConfig of droplet material
        diameter: m
        initial_position: [x, y, z] in m
        initial_velocity: [vx, vy, vz] in m/s
    """
    material: MaterialConfig
    diameter: float                                             # m
    initial_position: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.4])
    )
    initial_velocity: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )

    def __post_init__(self) -> None:
        """Validate droplet parameters and defensively copy NDArrays."""
        if self.diameter <= 0:
            raise ValueError(
                f"DropletConfig: diameter must be positive, got {self.diameter}"
            )
        if self.diameter < 10e-6 or self.diameter > 2e-3:
            warnings.warn(
                f"Droplet diameter {self.diameter*1e6:.0f} µm outside typical range [10, 2000]",
                stacklevel=2,
            )
        object.__setattr__(
            self, "initial_position",
            np.array(self.initial_position, dtype=np.float64, copy=True),
        )
        object.__setattr__(
            self, "initial_velocity",
            np.array(self.initial_velocity, dtype=np.float64, copy=True),
        )

    @property
    def radius(self) -> float:
        """Droplet radius [m]."""
        return self.diameter / 2.0

    @property
    def volume(self) -> float:
        """Droplet volume [m³]."""
        return (4.0 / 3.0) * np.pi * self.radius ** 3

    @property
    def mass(self) -> float:
        """Droplet mass [kg]."""
        return self.material.density * self.volume

    @property
    def weight(self) -> float:
        """Gravitational force on droplet [N]."""
        return self.mass * GRAVITY

    def reynolds_number(self, velocity_magnitude: float, medium: MaterialConfig) -> float:
        """Compute Reynolds number for current conditions.

        Re = ρ_medium · |v| · d / μ_medium
        """
        if medium.dynamic_viscosity <= 0.0:
            raise ValueError(
                f"Medium '{medium.name}' has non-positive viscosity "
                f"({medium.dynamic_viscosity}). Zero or negative viscosity is "
                f"an invalid configuration — Reynolds number is undefined."
            )
        return (
            medium.density * velocity_magnitude * self.diameter
            / medium.dynamic_viscosity
        )

    def drag_regime(self, velocity_magnitude: float, medium: MaterialConfig) -> str:
        """Select drag law from Re.

        Returns:
            "stokes" (Re < 1), "schiller_naumann" (Re < 1000), or "newton" (Re >= 1000)
        """
        re = self.reynolds_number(velocity_magnitude, medium)
        if re < 1.0:
            return "stokes"
        elif re < 1000.0:
            return "schiller_naumann"
        else:
            return "newton"

    def drag_coefficient(self, velocity_magnitude: float, medium: MaterialConfig) -> float:
        """Compute drag coefficient Cd for current Re.

        - Re < 1:    Stokes regime — Cd = 24/Re
        - Re < 1000: Schiller-Naumann — Cd = (24/Re)(1 + 0.15·Re^0.687)
        - Re >= 1000: Newton — Cd = 0.44

        Schiller-Naumann drag correlation (1935), valid for 1 < Re < 1000:
            Cd = (24/Re)(1 + 0.15 Re^0.687)
        """
        re = self.reynolds_number(velocity_magnitude, medium)
        if re < 1e-10:
            # Stationary droplet (Re ≈ 0): no drag force. See Schiller-Naumann correlation.
            return 0.0
        if re < 1.0:
            return 24.0 / re
        elif re < 1000.0:
            return (24.0 / re) * (1.0 + 0.15 * re ** 0.687)
        else:
            return 0.44

    def with_material(self, new_material: MaterialConfig) -> 'DropletConfig':
        """Return a new DropletConfig with updated material, preserving all other fields.

        Used when a droplet undergoes a phase transition (e.g., liquid -> solid
        mid-flight). DropletConfig is frozen, so we create a new instance.
        """
        return DropletConfig(
            material=new_material,
            diameter=self.diameter,
            initial_position=self.initial_position.copy(),
            initial_velocity=self.initial_velocity.copy(),
        )

    def check_gorkov_validity(
        self, transducer: TransducerConfig, medium: MaterialConfig
    ) -> bool:
        """Check ka << 1 condition for Gor'kov potential validity.

        Warns if ka > 0.5 (approaching breakdown of small-particle assumption).

        Returns:
            True if ka << 1 (valid), False if questionable.
        """
        k = transducer.wavenumber(medium.sound_speed)
        ka = k * self.radius
        if ka > 0.5:
            warnings.warn(
                f"Gor'kov validity: ka = {ka:.3f} > 0.5 for "
                f"'{self.material.name}' droplet (d={self.diameter*1e6:.0f} µm) "
                f"at f={transducer.frequency/1e3:.0f} kHz in '{medium.name}'. "
                f"Small-particle assumption may break down.",
                stacklevel=2,
            )
            return False
        return True


# --- Built-in droplet configs ------------------------------------------------

def aluminum_droplet(
    diameter: float = 300e-6,
    position: Optional[NDArray[np.float64]] = None,
    velocity: Optional[NDArray[np.float64]] = None,
) -> DropletConfig:
    """Create an aluminum liquid droplet config."""
    return DropletConfig(
        material=ALUMINUM_LIQUID,
        diameter=diameter,
        initial_position=position if position is not None else np.array([0.0, 0.0, 0.4]),
        initial_velocity=velocity if velocity is not None else np.array([0.0, 0.0, 0.0]),
    )


def styrofoam_bead(
    diameter: float = 1e-3,
    position: Optional[NDArray[np.float64]] = None,
    velocity: Optional[NDArray[np.float64]] = None,
) -> DropletConfig:
    """Create a styrofoam test bead config."""
    return DropletConfig(
        material=STYROFOAM,
        diameter=diameter,
        initial_position=position if position is not None else np.array([0.0, 0.0, 0.4]),
        initial_velocity=velocity if velocity is not None else np.array([0.0, 0.0, 0.0]),
    )


# =============================================================================
# EnvironmentConfig (frozen dataclass — sim engine internal, not in DesignPoint)
# =============================================================================

@dataclass(frozen=True)
class EnvironmentConfig:
    """Simulation environment: medium, temperature, derived physics.

    Speed of sound is derived from temperature:
        c(T) = 331.3 * sqrt(T_K / 273.15)  [m/s]

    alpha_max is derived from force balance:
        Maximum steering angle where acoustic force can overcome gravity
        on the configured droplet. NOT a geometric argument.

    Placeholder fields (volume_temperature_profile, acoustic_streaming_velocity)
    are Optional[None] and warn loudly if accessed.

    Attributes:
        medium: MaterialConfig of the surrounding fluid (default air)
        temperature: K (drives c(T))
        transducer: TransducerConfig
        array: ArrayConfig
        volume_temperature_profile: Optional spatial T(x,y,z) — PLACEHOLDER
        acoustic_streaming_velocity: Optional steady flow — PLACEHOLDER
    """
    medium: MaterialConfig = field(default_factory=lambda: AIR_20C)
    temperature: float = 293.15                  # K (20°C)
    transducer: TransducerConfig = field(default_factory=lambda: MA40S4S_40KHZ)
    array: ArrayConfig = field(default_factory=lambda: L1_ARRAY)
    volume_temperature_profile: Optional[Callable[[NDArray[np.float64]], float]] = None
    acoustic_streaming_velocity: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate environment parameters."""
        if self.temperature <= 0:
            raise ValueError(
                f"EnvironmentConfig: temperature must be positive (Kelvin), got {self.temperature}"
            )

    @property
    def speed_of_sound(self) -> float:
        """Speed of sound in medium derived from temperature.

        c(T) = 331.3 * sqrt(T_K / 273.15) [m/s]

        where T_K is absolute temperature in Kelvin.
        Equivalent to c = 331.3 * sqrt(1 + T_C/273.15) with T_C in °C.

        This is the authoritative value — do not hardcode 343 m/s anywhere.
        """
        return 331.3 * np.sqrt(self.temperature / 273.15)

    @property
    def wavelength(self) -> float:
        """Wavelength λ = c / f [m]."""
        return self.speed_of_sound / self.transducer.frequency

    @property
    def wavenumber(self) -> float:
        """Wavenumber k = 2π / λ [rad/m]."""
        return 2.0 * np.pi / self.wavelength

    @property
    def angular_frequency(self) -> float:
        """Angular frequency ω = 2πf [rad/s]."""
        return 2.0 * np.pi * self.transducer.frequency

    def alpha_max(self, droplet: DropletConfig) -> float:
        """Maximum steering gradient α_max.

        Returns the geometric aliasing limit: the largest phase gradient
        the array can produce before spatial aliasing degrades the focal
        pattern. This is α_max = π / (4·d_arc) where d_arc is the arc
        spacing between adjacent transducers on a ring.

        Also checks force sufficiency: whether the array can overcome
        droplet gravity at any alpha. Warns if it cannot.

        Returns α_max in rad/m.
        """
        rho_m = self.medium.density
        c = self.speed_of_sound
        k = self.wavenumber
        p_ref = self.transducer.p_ref
        r_ref = self.transducer.r_ref

        rho_p = droplet.material.density
        c_p = droplet.material.sound_speed
        f1 = 1.0 - (rho_m * c**2) / (rho_p * c_p**2)
        f2 = 2.0 * (rho_p - rho_m) / (2.0 * rho_p + rho_m)

        V = droplet.volume
        kappa_m = 1.0 / (rho_m * c**2)
        p_single = p_ref * r_ref / self.array.cylinder_radius
        n_trans = self.array.total_transducers
        p_focused = n_trans * p_single

        monopole_coeff = f1 * kappa_m                        # 1/Pa
        dipole_coeff = 3.0 * f2 / (4.0 * rho_m * c**2)      # 1/Pa
        force_per_alpha = (
            V * (abs(monopole_coeff) + abs(dipole_coeff))
            * k * p_focused**2
        )

        if force_per_alpha < 1e-30:
            warnings.warn(
                "alpha_max calculation: force_per_alpha ≈ 0 — "
                "acoustic force too weak for meaningful steering. "
                "Check transducer p_ref and array geometry."
            )
            return 0.0

        alpha_gravity = droplet.weight / force_per_alpha

        d_arc = (2.0 * np.pi * self.array.cylinder_radius
                 / self.array.transducers_per_ring)
        alpha_geometric = np.pi / (4.0 * d_arc)

        if alpha_gravity > alpha_geometric:
            warnings.warn(
                f"Force sufficiency: array needs α={alpha_gravity:.2f} rad/m "
                f"to overcome gravity on {droplet.material.name} droplet, but "
                f"geometric limit is α_max={alpha_geometric:.2f} rad/m. "
                f"Acoustic force may be insufficient for full steering."
            )

        return float(alpha_geometric)

    def get_temperature_at(self, position: NDArray[np.float64]) -> float:
        """Get temperature at a spatial position.

        PLACEHOLDER: volume_temperature_profile is not yet implemented.
        Warns loudly and returns the uniform environment temperature.
        """
        if self.volume_temperature_profile is not None:
            return self.volume_temperature_profile(position)
        warnings.warn(
            "volume_temperature_profile is None — spatial temperature variation "
            "not modeled. Returning uniform environment temperature. "
            "Oscar's thermal team must provide the math before this can be filled.",
            stacklevel=2,
        )
        return self.temperature

    def get_streaming_velocity(self) -> float:
        """Get acoustic streaming velocity.

        PLACEHOLDER: not yet modeled. Warns loudly and returns 0.0.
        """
        if self.acoustic_streaming_velocity is not None:
            return self.acoustic_streaming_velocity
        warnings.warn(
            "acoustic_streaming_velocity is None — acoustic streaming not modeled. "
            "At high acoustic intensities this is non-negligible but is out of scope "
            "for styrofoam testing. Returning 0.0.",
            stacklevel=2,
        )
        return 0.0

    def gorkov_f1(self, droplet_material: MaterialConfig) -> float:
        """Bruus f₁ coefficient (compressibility contrast).

        f₁ = 1 - (ρ_medium · c_medium²) / (ρ_particle · c_particle²)
        """
        rho_m = self.medium.density
        c_m = self.speed_of_sound
        rho_p = droplet_material.density
        c_p = droplet_material.sound_speed
        return 1.0 - (rho_m * c_m**2) / (rho_p * c_p**2)

    def gorkov_f2(self, droplet_material: MaterialConfig) -> float:
        """Bruus f₂ coefficient (density contrast).

        f₂ = 2(ρ_particle - ρ_medium) / (2·ρ_particle + ρ_medium)
        """
        rho_p = droplet_material.density
        rho_m = self.medium.density
        return 2.0 * (rho_p - rho_m) / (2.0 * rho_p + rho_m)

    def check_droplet(self, droplet: DropletConfig) -> None:
        """Run all validity checks for a droplet in this environment."""
        droplet.check_gorkov_validity(self.transducer, self.medium)

    def __repr__(self) -> str:
        return (
            f"EnvironmentConfig(T={self.temperature:.1f} K, "
            f"c={self.speed_of_sound:.1f} m/s, "
            f"medium='{self.medium.name}', "
            f"f={self.transducer.frequency/1e3:.0f} kHz)"
        )


# =============================================================================
# Convenience constructors
# =============================================================================

def default_environment() -> EnvironmentConfig:
    """Standard environment: air at 20°C, MA40S4S transducer, L1 array."""
    return EnvironmentConfig()


def styrofoam_test_environment() -> EnvironmentConfig:
    """Environment configured for styrofoam hardware testing."""
    return EnvironmentConfig(
        medium=AIR_20C,
        temperature=293.15,  # 20°C
        transducer=MA40S4S_40KHZ,
        array=L1_ARRAY,
    )


def aluminum_print_environment(temperature_k: float = 933.0) -> EnvironmentConfig:
    """Environment for aluminum printing.

    Default temperature 933 K (660°C, Al melting point).
    Speed of sound at 933 K ≈ 606 m/s (NOT 343).
    """
    return EnvironmentConfig(
        medium=AIR_20C,  # medium is still air (with adjusted c from temperature)
        temperature=temperature_k,
        transducer=MA40S4S_40KHZ,
        array=L1_ARRAY,
    )


# =============================================================================
# Chamber temperature gradient
# =============================================================================

def linear_chamber_gradient(
    T_top_K: float = 323.15,     # 50°C at z=height (transducer end)
    T_bottom_K: float = 673.15,  # 400°C at z=0 (build plate)
    height: float = 0.4,         # m, array height
) -> Callable[[NDArray[np.float64]], float]:
    """Linear temperature gradient from build plate (z=0) to transducers (z=height).

    Models the real chamber gradient: ~50°C near the transducers at the top
    and ~400°C near the heated build plate at the bottom.

    Args:
        T_top_K: Temperature at top of array [K]. Default 323.15 K (50°C).
        T_bottom_K: Temperature at bottom [K]. Default 673.15 K (400°C).
        height: Array height [m]. Default 0.4 m.

    Returns:
        Callable that takes position [x, y, z] and returns local temperature [K].
    """
    def profile(position: NDArray[np.float64]) -> float:
        z = float(position[2])
        frac = max(0.0, min(1.0, z / height))  # 0 at bottom, 1 at top
        return T_bottom_K + (T_top_K - T_bottom_K) * frac
    return profile


DEFAULT_CHAMBER_GRADIENT: Callable[[NDArray[np.float64]], float] = linear_chamber_gradient()
