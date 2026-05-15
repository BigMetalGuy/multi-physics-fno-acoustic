"""Report formatting and plot generation."""

from __future__ import annotations

import logging
import os
import warnings
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

from .models import BondingMode, DesignPoint, DesignReport
from .sweeps import sweep_2d, sweep_diameter


def format_report(r: DesignReport) -> str:
    """Human-readable text report."""
    lines = [
        "=" * 70,
        f"  DRIP PARAMETRIC DESIGN REPORT",
        f"  Droplet: {r.diameter_um:.0f} um | "
        f"Chamber: {r.chamber_C:.0f} C | Crucible: {r.crucible_C:.0f} C",
        "=" * 70,
        "",
        "  ACOUSTICS (D1/D2)",
        f"    ka                  = {r.ka:.4f}  {'pass' if r.ka_valid else 'FAIL'}",
        f"    f1 (compressibility)= {r.f1:.4f}",
        f"    f2 (density)        = {r.f2:.4f}",
        f"    Phi (contrast)      = {r.contrast_factor:.4f}",
        f"    F_max (focal)       = {r.F_max * 1e9:.1f} nN",
        f"    F/W ratio           = {r.force_to_weight:.2f}",
        "",
        "  ARRAY (D3)",
        f"    Transducers         = {r.n_transducers}",
        f"    Focused pressure    = {r.focused_pressure:.0f} Pa",
        f"    Arc spacing         = {r.arc_spacing * 1e3:.1f} mm",
        "",
        "  CONTROL (D4)",
        f"    alpha_max           = {r.alpha_max:.1f} rad/m",
        f"    Max displacement    = {r.max_displacement * 1e3:.1f} mm",
        f"    Target OK           {'pass' if r.displacement_ok else 'FAIL'}",
        "",
        "  THERMAL (D5 — in-flight cooling)",
        f"    Fall time           = {r.fall_time * 1e3:.1f} ms",
        f"    Impact velocity     = {r.impact_velocity:.2f} m/s",
        f"    Re (avg)            = {r.Re_avg:.1f}",
        f"    Nu (avg)            = {r.Nu_avg:.2f}",
        f"    h (avg)             = {r.h_avg:.0f} W/m2K",
        f"    tau_conv            = {r.tau_conv * 1e3:.0f} ms",
        f"    Bi                  = {r.Bi:.5f}  {'pass' if r.Bi < 0.1 else 'FAIL'}",
        f"    T_arrival           = {r.T_arrival_C:.1f} C  ({r.T_arrival:.1f} K)",
        f"    Temp drop           = {r.temp_drop:.1f} K",
        f"    Liquid at landing   = {'YES' if r.liquid_fraction_at_landing > 0.5 else 'NO'}",
    ]
    if r.solidification_time is not None:
        pct = r.solidification_time / r.fall_time * 100 if r.fall_time > 0 else 0
        lines.append(
            f"    Solidifies at       = {r.solidification_time * 1e3:.1f} ms "
            f"({pct:.0f}% through fall)"
        )
    else:
        lines.append("    Solidification      = NONE (stays liquid)")

    lines += [
        "",
        "  BONDING — Orme & Huang 1997 (thermal remelt, informational)",
        f"    (T_arr-T_m)/(T_m-T_sub) = {r.orme_ratio:.2f}  "
        f"(need > 1.6)  {'PASS' if r.remelting_feasible else 'FAIL'}",
        f"    Margin              = {r.remelt_margin:+.2f}",
    ]

    if r.acoustic_bond_feasible is not None:
        details = r.acoustic_bond_details or {}
        lines += [
            "",
            "  BONDING — 20 kHz Sonotrode (cavitation in liquid, PRIMARY)",
            f"    Feasible            = {'PASS' if r.acoustic_bond_feasible else 'FAIL'}",
            f"    Mechanism           = {details.get('mechanism', 'unknown')}",
            f"    Liquid phase        = {'YES' if details.get('has_liquid_phase') else 'NO (need liquid for cavitation)'}",
            f"    Margin above solidus= {details.get('liquid_margin_K', 0):+.0f} K",
            f"    Amplitude margin    = {details.get('amplitude_margin_um', 0):+.1f} um",
        ]

    lines += [
        "",
        "  MICROSTRUCTURE (D14)",
        f"    SDAS                = {r.sdas_um:.1f} um",
        f"    Grain size          = {r.grain_size_um:.1f} um",
        f"    Hardness            = {r.hardness_hv:.0f} HV",
        f"    Regime              = {r.microstructure_regime}",
        f"    vs conventional     = {r.sdas_ratio_vs_cast:.2f}x (< 1 is better)",
        "",
        "  THROUGHPUT",
        f"    Layer thickness     = {r.layer_thickness * 1e6:.0f} um",
        f"    Splat diameter      = {r.splat_diameter * 1e6:.0f} um",
        f"    Cycle time          = {r.cycle_time * 1e3:.1f} ms/droplet",
        f"    Droplet rate        = {r.droplets_per_second:.1f} Hz",
        f"    Build rate          = {r.build_rate_cm3_hr:.4f} cm3/hr",
        f"    1 cm3 cube          = {r.build_time_1cm_cube_hr:.1f} hr",
        "",
        "  FGM COMPOSITION (D13)",
        f"    Gradient            = {0.0:.1f}% -> {4.5:.1f}% Mg",
        f"    Min layers          = {r.fgm_min_layers}",
        f"    Gradient height     = {r.fgm_gradient_height_mm:.2f} mm",
        f"    Achievable          = {'YES' if r.fgm_gradient_achievable else 'NO'}",
        "",
        f"  COOLING RATE          = {r.cooling_rate:.0f} K/s",
        "",
        "  POWER / COST (D9-11)",
        f"    Per element          = {r.power_per_element:.1f} W",
        f"    Total power          = {r.total_power:.0f} W",
        f"    Total transducer $   = ${r.total_cost:.0f}",
        "",
        "  FEASIBILITY",
    ]

    for name, entry in r.constraints.items():
        # Support both ConstraintResult objects and legacy (val, limit, ok) tuples
        if hasattr(entry, "satisfied"):
            status = "pass" if entry.satisfied else "FAIL"
            lines.append(
                f"    {status:>4} {name}: {entry.value:.4f} vs {entry.limit:.4f}"
                f"  (margin: {entry.margin:+.4f})"
            )
        else:
            val, limit, ok = entry
            status = "pass" if ok else "FAIL"
            lines.append(f"    {status:>4} {name}: {val:.4f} vs {limit:.4f}")

    lines += [
        "",
        f"  OVERALL: {'FEASIBLE' if r.feasible else 'NOT FEASIBLE'}",
        "=" * 70,
    ]
    return "\n".join(lines)


def generate_plots(
    base: DesignPoint,
    output_dir: str = "output",
) -> None:
    """Generate parametric sweep plots."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed — skipping plots")
        return

    # ---- 1. Diameter sweep ----
    diameters = np.linspace(50e-6, 800e-6, 60)
    reports = sweep_diameter(base, diameters)
    d_um = [r.diameter_um for r in reports]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Parametric Sweep: Droplet Diameter\n"
        f"Chamber={base.chamber_temperature - 273.15:.0f} C, "
        f"Crucible={base.crucible_temperature - 273.15:.0f} C, "
        f"Array={base.array.ring_count}x{base.array.transducers_per_ring}",
        fontsize=13, fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(d_um, [r.ka for r in reports], "b-", linewidth=2)
    ax.axhline(0.5, color="r", linestyle="--", label="ka = 0.5 limit")
    ax.set_xlabel("Droplet diameter (um)")
    ax.set_ylabel("ka")
    ax.set_title("Gor'kov Validity (D2)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(d_um, [r.force_to_weight for r in reports], "g-", linewidth=2)
    ax.axhline(1.0, color="r", linestyle="--", label="F = W")
    ax.set_xlabel("Droplet diameter (um)")
    ax.set_ylabel("F_acoustic / W")
    ax.set_title("Force-to-Weight Ratio (D1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(d_um, [r.T_arrival_C for r in reports], "r-", linewidth=2)
    ax.axhline(660, color="k", linestyle="--", label="T_melt (660 C)")
    ax.set_xlabel("Droplet diameter (um)")
    ax.set_ylabel("Arrival Temperature (C)")
    ax.set_title("In-Flight Cooling (D5)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(d_um, [r.tau_conv * 1e3 for r in reports], "b-", linewidth=2, label="tau_conv")
    ax.plot(d_um, [r.fall_time * 1e3 for r in reports], "k--", linewidth=1, label="t_fall")
    ax.set_xlabel("Droplet diameter (um)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Thermal Time Constant vs Fall Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(d_um, [r.orme_ratio for r in reports], "m-", linewidth=2, label="Orme")
    ax.axhline(1.6, color="r", linestyle="--", label="Remelt threshold")
    ax.axhline(0.0, color="k", linestyle=":", alpha=0.3)
    ax.set_xlabel("Droplet diameter (um)")
    ax.set_ylabel("(T_arr - T_m) / (T_m - T_sub)")
    ax.set_title("Orme Remelting (informational)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(d_um, [r.cooling_rate for r in reports], "c-", linewidth=2)
    ax.set_xlabel("Droplet diameter (um)")
    ax.set_ylabel("Cooling Rate (K/s)")
    ax.set_title("Average Cooling Rate")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parametric_diameter_sweep.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ---- 2. Diameter x Chamber contour ----
    d_range = np.linspace(100e-6, 600e-6, 40)
    T_range = np.linspace(273.15 + 200, 273.15 + 800, 40)
    grid = sweep_2d(base, d_range, T_range)

    d_um_grid = d_range * 1e6
    T_C_grid = T_range - 273.15

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Design Space: Diameter x Chamber Temperature\n"
        f"Crucible={base.crucible_temperature - 273.15:.0f} C, "
        f"Array={base.array.ring_count}x{base.array.transducers_per_ring}",
        fontsize=13, fontweight="bold",
    )

    ax = axes[0]
    Z = grid["T_arrival_C"]
    CS = ax.contourf(T_C_grid, d_um_grid, Z, levels=20, cmap="RdYlBu_r")
    ax.contour(T_C_grid, d_um_grid, Z, levels=[660], colors="black", linewidths=2)
    plt.colorbar(CS, ax=ax, label="Arrival Temp (C)")
    ax.set_xlabel("Chamber Temperature (C)")
    ax.set_ylabel("Droplet Diameter (um)")
    ax.set_title("Arrival Temperature\n(black = 660 C melting)")

    ax = axes[1]
    Z_ka = grid["ka"]
    CS2 = ax.contourf(T_C_grid, d_um_grid, Z_ka, levels=20, cmap="viridis")
    ax.contour(T_C_grid, d_um_grid, Z_ka, levels=[0.5], colors="red", linewidths=2)
    plt.colorbar(CS2, ax=ax, label="ka")
    ax.set_xlabel("Chamber Temperature (C)")
    ax.set_ylabel("Droplet Diameter (um)")
    ax.set_title("Gor'kov Validity (ka)\n(red = 0.5 limit)")

    ax = axes[2]
    Z_ab = grid["acoustic_bond_feasible"].astype(float)
    CS3 = ax.contourf(T_C_grid, d_um_grid, Z_ab, levels=[-0.5, 0.5, 1.5],
                      colors=["#ffcccc", "#ccffcc"], alpha=0.7)
    ax.contour(T_C_grid, d_um_grid, grid["T_arrival_C"], levels=[660],
               colors="blue", linewidths=2)
    ax.contour(T_C_grid, d_um_grid, grid["ka"], levels=[0.5],
               colors="red", linewidths=2, linestyles="--")
    plt.colorbar(CS3, ax=ax, label="Acoustic bond feasible")
    ax.set_xlabel("Chamber Temperature (C)")
    ax.set_ylabel("Droplet Diameter (um)")
    ax.set_title("Acoustic Bonding Feasibility\n(green = feasible)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parametric_design_space.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ---- 3. Feasibility overlay ----
    fig, ax = plt.subplots(figsize=(10, 8))
    feasible_mask = grid["feasible"].astype(float)
    ax.contourf(T_C_grid, d_um_grid, feasible_mask, levels=[-0.5, 0.5, 1.5],
                colors=["#ffcccc", "#ccffcc"], alpha=0.7)
    ax.contour(T_C_grid, d_um_grid, grid["ka"], levels=[0.5],
               colors="red", linewidths=2, linestyles="--")
    ax.contour(T_C_grid, d_um_grid, grid["T_arrival_C"], levels=[660],
               colors="blue", linewidths=2)

    ax.plot(base.chamber_temperature - 273.15, base.droplet_diameter * 1e6,
            "k*", markersize=15, zorder=10, label="Design Point")

    ax.set_xlabel("Chamber Temperature (C)", fontsize=12)
    ax.set_ylabel("Droplet Diameter (um)", fontsize=12)
    ax.set_title(
        "Feasibility Map: Acoustic + Thermal + Bonding\n"
        f"Crucible={base.crucible_temperature - 273.15:.0f} C, "
        f"Array={base.array.ring_count}x{base.array.transducers_per_ring}",
        fontsize=12, fontweight="bold",
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="red", linestyle="--", linewidth=2,
               label="ka = 0.5 (Gor'kov limit)"),
        Line2D([0], [0], color="blue", linewidth=2,
               label="T_arrival = 660 C (melting)"),
        Line2D([0], [0], marker="*", color="k", markersize=12,
               linestyle="None", label="Design Point"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, "parametric_feasibility_map.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Plots saved to %s/: diameter_sweep, design_space, feasibility_map", output_dir)
