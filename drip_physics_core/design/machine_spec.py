"""
Machine specification generator.

Takes an optimized DesignPoint + DesignReport and generates a structured
machine specification: BOM, dimensions, thermal design, control requirements,
and safety envelope.

Domain-review status: the code does what its docstrings claim and is
arithmetically correct, but the *values* it produces are estimates
that need cross-functional sign-off before being treated as a quote
or build spec:

- BOM unit costs come from the DesignPoint scalar defaults (`*_cost`
  fields). They are not pulled from a supplier registry. Cross-check
  with Molly (EE) and the Ohio Aluminum partnership before quoting.
- Chamber dimensions use a 5 mm wall thickness literal and a 40 % /
  80 % build-volume fraction relative to the array radius/height.
  Validate against CAD before fabricating.
- Thermal power uses Stefan-Boltzmann + a 20 W/m²·K natural-convection
  coefficient with a 1.5× safety factor. Oscar should review whether
  a forced-convection or FEM model is required for L1.
- Control requirements (8-bit phase resolution, 1 kHz update rate,
  600 W bonding power) are hardcoded — Ryota owns these.

The cost-as-registry refactor (deferred) replaces the BOM half of
this. Until then, treat outputs as planning estimates, not contracts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from drip_physics_core.config import GRAVITY
from .models import DesignPoint, DesignReport, l1_design_point
from .physics_eval import evaluate_design


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BOMItem:
    """Single item in the bill of materials."""

    category: str
    name: str
    quantity: int
    unit_cost_usd: float
    total_cost_usd: float
    notes: str = ""


@dataclass(frozen=True)
class ChamberDimensions:
    """Physical dimensions of the acoustic chamber."""

    inner_radius_m: float
    outer_radius_m: float       # inner + wall thickness
    height_m: float
    wall_thickness_m: float
    build_volume_radius_m: float
    build_volume_height_m: float


@dataclass(frozen=True)
class ThermalDesign:
    """Thermal management specification."""

    chamber_target_C: float
    substrate_target_C: float
    max_part_temp_C: float
    heater_power_w: float       # estimated substrate heater power
    cooling_required_w: float   # heat rejection from transducers
    insulation_type: str
    thermal_gradient: str       # description of gradient strategy


@dataclass(frozen=True)
class ControlRequirements:
    """Control system requirements."""

    guidance_frequency_hz: float
    guidance_n_channels: int
    phase_resolution_bits: int
    update_rate_hz: float
    bonding_frequency_hz: float
    bonding_power_w: float
    bonding_amplitude_um: float


@dataclass(frozen=True)
class SafetyEnvelope:
    """Operating limits and failure modes."""

    max_chamber_temp_C: float
    max_substrate_temp_C: float
    max_crucible_temp_C: float
    max_acoustic_pressure_Pa: float
    max_droplet_velocity_ms: float
    failure_modes: tuple[str, ...]


@dataclass(frozen=True)
class MachineSpec:
    """Complete machine specification."""

    name: str
    version: str
    design_point: DesignPoint
    report: DesignReport

    # Sub-specs
    bom: tuple[BOMItem, ...]
    chamber: ChamberDimensions
    thermal: ThermalDesign
    control: ControlRequirements
    safety: SafetyEnvelope

    # Summary
    total_cost_usd: float
    total_power_w: float

    def to_json(self) -> str:
        """Export as JSON for CAD/portal integration."""
        return json.dumps(self._to_dict(), indent=2)

    def _to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "droplet_diameter_um": self.report.diameter_um,
            "chamber_temperature_C": self.report.chamber_C,
            "crucible_temperature_C": self.report.crucible_C,
            "array": {
                "n_transducers": self.report.n_transducers,
                "cylinder_radius_m": self.design_point.array.cylinder_radius,
                "ring_count": self.design_point.array.ring_count,
                "transducers_per_ring": self.design_point.array.transducers_per_ring,
                "ring_spacing_m": self.design_point.array.ring_spacing,
                "height_m": self.design_point.array_height,
            },
            "chamber": {
                "inner_radius_m": self.chamber.inner_radius_m,
                "height_m": self.chamber.height_m,
                "build_volume_radius_m": self.chamber.build_volume_radius_m,
                "build_volume_height_m": self.chamber.build_volume_height_m,
            },
            "thermal": {
                "chamber_target_C": self.thermal.chamber_target_C,
                "substrate_target_C": self.thermal.substrate_target_C,
                "heater_power_w": self.thermal.heater_power_w,
                "cooling_required_w": self.thermal.cooling_required_w,
            },
            "control": {
                "guidance_frequency_hz": self.control.guidance_frequency_hz,
                "guidance_channels": self.control.guidance_n_channels,
                "bonding_frequency_hz": self.control.bonding_frequency_hz,
                "bonding_power_w": self.control.bonding_power_w,
            },
            "performance": {
                "ka": self.report.ka,
                "force_to_weight": self.report.force_to_weight,
                "T_arrival_C": self.report.T_arrival_C,
                "fall_time_ms": self.report.fall_time * 1000,
                "cooling_rate_Ks": self.report.cooling_rate,
                "feasible": self.report.feasible,
            },
            "cost": {
                "transducers_usd": sum(
                    i.total_cost_usd for i in self.bom
                    if i.category == "Acoustic Array"
                ),
                "total_usd": self.total_cost_usd,
            },
            "bom": [
                {
                    "category": i.category,
                    "name": i.name,
                    "qty": i.quantity,
                    "unit_cost": i.unit_cost_usd,
                    "total": i.total_cost_usd,
                    "notes": i.notes,
                }
                for i in self.bom
            ],
        }


# ---------------------------------------------------------------------------
# BOM generation
# ---------------------------------------------------------------------------
def _generate_bom(dp: DesignPoint, report: DesignReport) -> tuple[BOMItem, ...]:
    """Generate bill of materials from design point."""
    items: list[BOMItem] = []
    n = report.n_transducers

    # Acoustic array (40 kHz guidance)
    items.append(BOMItem(
        category="Acoustic Array",
        name=f"40 kHz transducer ({dp.transducer.name})",
        quantity=n,
        unit_cost_usd=dp.transducer_cost,
        total_cost_usd=n * dp.transducer_cost,
        notes="Murata MA40S4S or equivalent",
    ))
    items.append(BOMItem(
        category="Acoustic Array",
        name="Transducer driver board",
        quantity=n,
        unit_cost_usd=2.50,
        total_cost_usd=n * 2.50,
        notes="Per-element phase control, 8-bit resolution",
    ))
    items.append(BOMItem(
        category="Acoustic Array",
        name="Array frame (machined Al)",
        quantity=1,
        unit_cost_usd=500.0,
        total_cost_usd=500.0,
        notes=f"Cylindrical, R={dp.array.cylinder_radius*100:.0f}cm, "
              f"H={dp.array_height*100:.0f}cm",
    ))

    # Bonding subsystem (20 kHz sonotrode)
    items.append(BOMItem(
        category="Bonding",
        name="20 kHz Langevin transducer",
        quantity=1,
        unit_cost_usd=160.0,
        total_cost_usd=160.0,
        notes="2000W, substrate-coupled",
    ))
    items.append(BOMItem(
        category="Bonding",
        name="20 kHz ultrasonic generator",
        quantity=1,
        unit_cost_usd=310.0,
        total_cost_usd=310.0,
        notes="Matched to transducer, includes tariff",
    ))
    items.append(BOMItem(
        category="Bonding",
        name="Sonotrode stepped horn (Ti-6Al-4V)",
        quantity=1,
        unit_cost_usd=250.0,
        total_cost_usd=250.0,
        notes="Half-wave design, amplitude gain ~4x",
    ))

    # Crucible / droplet generation
    items.append(BOMItem(
        category="Crucible",
        name="Induction heating system",
        quantity=1,
        unit_cost_usd=800.0,
        total_cost_usd=800.0,
        notes=f"Target: {dp.crucible_temperature - 273.15:.0f}°C",
    ))
    items.append(BOMItem(
        category="Crucible",
        name="Graphite crucible + nozzle assembly",
        quantity=1,
        unit_cost_usd=200.0,
        total_cost_usd=200.0,
        notes=f"Orifice: {dp.droplet_diameter*1e6*0.53:.0f} um "
              f"(Rayleigh: d_drop ≈ 1.89 × d_orifice)",
    ))

    # Thermal management
    items.append(BOMItem(
        category="Thermal",
        name="Chamber heater (resistive)",
        quantity=1,
        unit_cost_usd=150.0,
        total_cost_usd=150.0,
        notes=f"Target: {dp.chamber_temperature - 273.15:.0f}°C",
    ))
    items.append(BOMItem(
        category="Thermal",
        name="Substrate heater + PID controller",
        quantity=1,
        unit_cost_usd=300.0,
        total_cost_usd=300.0,
        notes=f"Target: {dp.substrate_temperature - 273.15:.0f}°C",
    ))

    # Control electronics
    items.append(BOMItem(
        category="Control",
        name="FPGA / MCU control board",
        quantity=1,
        unit_cost_usd=200.0,
        total_cost_usd=200.0,
        notes="Phase array control + bonding timing",
    ))
    items.append(BOMItem(
        category="Control",
        name="Power supply (48V, 5A)",
        quantity=1,
        unit_cost_usd=100.0,
        total_cost_usd=100.0,
    ))

    # Structure
    items.append(BOMItem(
        category="Structure",
        name="Frame and enclosure",
        quantity=1,
        unit_cost_usd=400.0,
        total_cost_usd=400.0,
        notes="Aluminum extrusion + sheet metal",
    ))

    return tuple(items)


# ---------------------------------------------------------------------------
# Spec generation
# ---------------------------------------------------------------------------
def generate_machine_spec(
    dp: DesignPoint,
    report: Optional[DesignReport] = None,
    name: str = "NF0.3-L1",
    version: str = "1.0",
) -> MachineSpec:
    """Generate complete machine specification from a design point.

    Args:
        dp: Optimized design point.
        report: Pre-computed report (computed if None).
        name: Machine designation.
        version: Spec version string.
    """
    if report is None:
        report = evaluate_design(dp)

    bom = _generate_bom(dp, report)
    total_cost = sum(i.total_cost_usd for i in bom)

    # Chamber dimensions
    wall_thickness = 0.005  # 5mm Al wall
    chamber = ChamberDimensions(
        inner_radius_m=dp.array.cylinder_radius,
        outer_radius_m=dp.array.cylinder_radius + wall_thickness,
        height_m=dp.array_height + 0.10,  # 10cm clearance above/below
        wall_thickness_m=wall_thickness,
        build_volume_radius_m=dp.array.cylinder_radius * 0.4,  # 40% of array radius
        build_volume_height_m=dp.array_height * 0.8,
    )

    # Thermal design
    # Substrate heater: estimate power from Stefan-Boltzmann + convection losses
    A_substrate = np.pi * chamber.build_volume_radius_m ** 2
    T_sub = dp.substrate_temperature
    T_amb = dp.chamber_temperature
    h_conv_est = 20.0  # natural convection W/m²K
    Q_conv = h_conv_est * A_substrate * (T_sub - T_amb)
    Q_rad = 0.3 * 5.67e-8 * A_substrate * (T_sub ** 4 - T_amb ** 4)
    heater_power = max(50.0, (Q_conv + Q_rad) * 1.5)  # 1.5x safety factor

    # Cooling: transducer dissipation
    cooling = dp.transducer_power_w * report.n_transducers

    thermal = ThermalDesign(
        chamber_target_C=dp.chamber_temperature - 273.15,
        substrate_target_C=dp.substrate_temperature - 273.15,
        max_part_temp_C=250.0,  # microstructure preservation limit
        heater_power_w=heater_power,
        cooling_required_w=cooling,
        insulation_type="Ceramic fiber blanket, 25mm",
        thermal_gradient=(
            f"Top (transducers): <85°C. "
            f"Bottom (substrate): {dp.substrate_temperature - 273.15:.0f}°C. "
            f"Air column: {dp.chamber_temperature - 273.15:.0f}°C uniform."
        ),
    )

    # Control requirements
    control = ControlRequirements(
        guidance_frequency_hz=dp.transducer.frequency,
        guidance_n_channels=report.n_transducers,
        phase_resolution_bits=8,  # 256 steps = 1.4° resolution
        update_rate_hz=1000.0,    # phase update rate
        bonding_frequency_hz=dp.sonotrode_frequency,
        bonding_power_w=600.0,    # estimated delivered power
        bonding_amplitude_um=dp.sonotrode_amplitude_um,
    )

    # Safety envelope
    safety = SafetyEnvelope(
        max_chamber_temp_C=300.0,
        max_substrate_temp_C=700.0,
        max_crucible_temp_C=1100.0,
        max_acoustic_pressure_Pa=report.focused_pressure * 1.5,
        max_droplet_velocity_ms=report.impact_velocity * 2.0,
        failure_modes=(
            "Crucible leak: molten Al release. Mitigation: catch tray + thermal fuse.",
            "Transducer overheat: >85°C. Mitigation: thermal cutoff on array frame.",
            "Sonotrode decouple: loss of bonding. Mitigation: acoustic feedback sensor.",
            "Chamber overpressure: N/A (atmospheric operation).",
            "Nozzle clog: no droplet production. Mitigation: induction reheat cycle.",
        ),
    )

    total_power = (
        report.total_power          # 40 kHz array
        + 600.0                     # 20 kHz bonding
        + heater_power              # substrate heater
        + 800.0                     # induction crucible heater (estimated)
    )

    return MachineSpec(
        name=name,
        version=version,
        design_point=dp,
        report=report,
        bom=bom,
        chamber=chamber,
        thermal=thermal,
        control=control,
        safety=safety,
        total_cost_usd=total_cost,
        total_power_w=total_power,
    )


def format_machine_spec(spec: MachineSpec) -> str:
    """Human-readable machine spec report."""
    lines = [
        "=" * 70,
        f"  MACHINE SPECIFICATION: {spec.name} v{spec.version}",
        "=" * 70,
        "",
        "  DESIGN POINT",
        f"    Droplet:     {spec.report.diameter_um:.0f} um "
        f"({spec.design_point.material.name})",
        f"    Crucible:    {spec.report.crucible_C:.0f}°C",
        f"    Chamber:     {spec.report.chamber_C:.0f}°C",
        f"    Substrate:   {spec.design_point.substrate_temperature - 273.15:.0f}°C",
        f"    Feasible:    {'YES' if spec.report.feasible else 'NO'}",
        "",
        "  CHAMBER DIMENSIONS",
        f"    Inner radius:      {spec.chamber.inner_radius_m * 100:.1f} cm",
        f"    Height:            {spec.chamber.height_m * 100:.1f} cm",
        f"    Build volume:      {spec.chamber.build_volume_radius_m * 100:.1f} cm "
        f"radius x {spec.chamber.build_volume_height_m * 100:.1f} cm tall",
        "",
        "  BILL OF MATERIALS",
    ]

    # Group BOM by category
    categories: dict[str, list[BOMItem]] = {}
    for item in spec.bom:
        categories.setdefault(item.category, []).append(item)

    for cat, items in categories.items():
        cat_total = sum(i.total_cost_usd for i in items)
        lines.append(f"    {cat} (${cat_total:,.0f}):")
        for item in items:
            lines.append(
                f"      {item.quantity:>3}x {item.name:<40} "
                f"${item.total_cost_usd:>8,.0f}"
            )
            if item.notes:
                lines.append(f"           {item.notes}")

    lines += [
        f"    {'─' * 56}",
        f"    TOTAL: ${spec.total_cost_usd:>8,.0f}",
        "",
        "  THERMAL DESIGN",
        f"    Chamber target:    {spec.thermal.chamber_target_C:.0f}°C",
        f"    Substrate target:  {spec.thermal.substrate_target_C:.0f}°C",
        f"    Max part temp:     {spec.thermal.max_part_temp_C:.0f}°C",
        f"    Heater power:      {spec.thermal.heater_power_w:.0f} W",
        f"    Cooling required:  {spec.thermal.cooling_required_w:.0f} W",
        f"    Gradient:          {spec.thermal.thermal_gradient}",
        "",
        "  CONTROL",
        f"    Guidance:   {spec.control.guidance_frequency_hz/1000:.0f} kHz, "
        f"{spec.control.guidance_n_channels} channels, "
        f"{spec.control.phase_resolution_bits}-bit phase",
        f"    Bonding:    {spec.control.bonding_frequency_hz/1000:.0f} kHz, "
        f"{spec.control.bonding_power_w:.0f} W, "
        f"{spec.control.bonding_amplitude_um:.0f} um",
        "",
        "  POWER BUDGET",
        f"    40 kHz array:      {spec.report.total_power:.0f} W",
        f"    20 kHz bonding:    600 W",
        f"    Substrate heater:  {spec.thermal.heater_power_w:.0f} W",
        f"    Crucible heater:   800 W (est.)",
        f"    TOTAL:             {spec.total_power_w:.0f} W",
        "",
        "  SAFETY",
    ]
    for mode in spec.safety.failure_modes:
        lines.append(f"    - {mode}")

    lines += ["", "=" * 70]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-built convenience
# ---------------------------------------------------------------------------
def l1_machine_spec() -> MachineSpec:
    """Generate L1 machine spec with defaults."""
    dp = l1_design_point(chamber_temperature_C=200)
    return generate_machine_spec(dp)
