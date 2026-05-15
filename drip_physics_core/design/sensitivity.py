"""
Sensitivity analysis via finite-difference Jacobian.

Computes partial derivatives of every output with respect to every input.
Answers: "which design variable matters most for each performance metric?"
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass

logger = logging.getLogger(__name__)
from typing import Callable, Dict, List, Sequence, Tuple

from .models import DesignPoint, DesignReport
from .physics_eval import evaluate_design

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUTS: tuple[str, ...] = (
    "droplet_diameter",
    "crucible_temperature",
    "chamber_temperature",
    "substrate_temperature",
    "sonotrode_amplitude_um",
)

DEFAULT_OUTPUTS: tuple[str, ...] = (
    "ka",
    "F_max",
    "force_to_weight",
    "T_arrival_C",
    "liquid_fraction_at_landing",
    "tau_conv",
    "cooling_rate",
    "orme_ratio",
    "total_cost",
    "Bi",
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SensitivityResult:
    """Full sensitivity analysis result."""

    jacobian: Dict[Tuple[str, str], float]      # (input, output) -> dOut/dIn
    elasticity: Dict[Tuple[str, str], float]     # normalized: (dOut/Out)/(dIn/In)
    rankings: Dict[str, List[str]]               # output -> [inputs by |elasticity|]
    base_inputs: Dict[str, float]
    base_outputs: Dict[str, float]


@dataclass(frozen=True)
class TornadoEntry:
    """One bar in a tornado chart."""

    input_name: str
    elasticity: float
    abs_elasticity: float
    delta_low: float        # output change when input decreases
    delta_high: float       # output change when input increases


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_field(report: DesignReport, name: str) -> float:
    """Extract a numeric field from DesignReport by name."""
    val = getattr(report, name)
    if val is None:
        return 0.0
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    return float(val)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_sensitivity(
    dp: DesignPoint,
    inputs: Sequence[str] = DEFAULT_INPUTS,
    outputs: Sequence[str] = DEFAULT_OUTPUTS,
    rel_delta: float = 0.01,
    evaluator: Callable[[DesignPoint], DesignReport] = evaluate_design,
) -> SensitivityResult:
    """Compute finite-difference Jacobian and elasticity at a design point.

    Uses central differences for O(h^2) accuracy.

    Args:
        dp: Base design point to analyze.
        inputs: DesignPoint field names to perturb.
        outputs: DesignReport field names to track.
        rel_delta: Relative perturbation size (default 1%).
        evaluator: Function to evaluate a DesignPoint (injectable for testing).
    """
    base_report = evaluator(dp)
    base_outputs = {name: _get_field(base_report, name) for name in outputs}
    base_inputs = {name: float(getattr(dp, name)) for name in inputs}

    jacobian: Dict[Tuple[str, str], float] = {}

    for inp in inputs:
        val = base_inputs[inp]
        delta = abs(val * rel_delta)
        if delta < 1e-15:
            continue

        dp_high = dp.model_copy(update={inp: val + delta})
        dp_low = dp.model_copy(update={inp: val - delta})
        r_high = evaluator(dp_high)
        r_low = evaluator(dp_low)

        for out in outputs:
            v_high = _get_field(r_high, out)
            v_low = _get_field(r_low, out)
            jacobian[(inp, out)] = (v_high - v_low) / (2.0 * delta)

    # Elasticity: (dOut/Out) / (dIn/In) = (dOut/dIn) * (In/Out)
    elasticity: Dict[Tuple[str, str], float] = {}
    for (inp, out), dydx in jacobian.items():
        x0 = base_inputs[inp]
        y0 = base_outputs[out]
        if abs(y0) > 1e-15 and abs(x0) > 1e-15:
            elasticity[(inp, out)] = (dydx * x0) / y0
        else:
            elasticity[(inp, out)] = 0.0

    # Rankings: for each output, sort inputs by |elasticity|
    rankings: Dict[str, List[str]] = {}
    for out in outputs:
        pairs = [(inp, abs(elasticity.get((inp, out), 0.0))) for inp in inputs]
        pairs.sort(key=lambda p: p[1], reverse=True)
        rankings[out] = [p[0] for p in pairs]

    return SensitivityResult(
        jacobian=jacobian,
        elasticity=elasticity,
        rankings=rankings,
        base_inputs=base_inputs,
        base_outputs=base_outputs,
    )


def tornado_data(
    result: SensitivityResult,
    output_name: str,
) -> List[TornadoEntry]:
    """Build tornado chart data for a single output variable."""
    if output_name not in result.rankings:
        return []

    entries = []
    for inp in result.rankings[output_name]:
        e = result.elasticity.get((inp, output_name), 0.0)
        j = result.jacobian.get((inp, output_name), 0.0)
        delta_in = result.base_inputs.get(inp, 0.0) * 0.01
        entries.append(TornadoEntry(
            input_name=inp,
            elasticity=e,
            abs_elasticity=abs(e),
            delta_low=-j * delta_in,
            delta_high=j * delta_in,
        ))
    return entries


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def generate_sensitivity_plots(
    result: SensitivityResult,
    output_dir: str = "output",
) -> None:
    """Generate tornado charts and coupling heatmap."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        warnings.warn("matplotlib not installed — skipping sensitivity plots")
        return

    inputs = list(result.base_inputs.keys())
    outputs = list(result.base_outputs.keys())

    # ---- Coupling heatmap ----
    matrix = np.zeros((len(inputs), len(outputs)))
    for i, inp in enumerate(inputs):
        for j, out in enumerate(outputs):
            matrix[i, j] = result.elasticity.get((inp, out), 0.0)

    fig, ax = plt.subplots(figsize=(max(10, len(outputs) * 1.2), max(5, len(inputs) * 0.8)))
    im = ax.imshow(np.abs(matrix), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(outputs)))
    ax.set_xticklabels(outputs, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(inputs)))
    ax.set_yticklabels(inputs, fontsize=9)
    ax.set_title("Sensitivity Heatmap: |elasticity|", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="|elasticity|")

    # Annotate cells
    for i in range(len(inputs)):
        for j in range(len(outputs)):
            val = matrix[i, j]
            color = "white" if abs(val) > np.max(np.abs(matrix)) * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sensitivity_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Tornado charts for key outputs ----
    key_outputs = ["T_arrival_C", "ka", "force_to_weight", "cooling_rate"]
    for out_name in key_outputs:
        if out_name not in result.rankings:
            continue

        entries = tornado_data(result, out_name)
        if not entries:
            continue

        fig, ax = plt.subplots(figsize=(8, max(3, len(entries) * 0.5)))
        y_pos = range(len(entries))
        colors = ["#e74c3c" if e.elasticity < 0 else "#2ecc71" for e in entries]
        bars = [e.elasticity for e in entries]
        labels = [e.input_name for e in entries]

        ax.barh(y_pos, bars, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Elasticity (% output change / % input change)")
        base_val = result.base_outputs.get(out_name, 0)
        ax.set_title(
            f"Sensitivity Tornado: {out_name}\n(base = {base_val:.4g})",
            fontsize=12, fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sensitivity_tornado_{out_name}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    logger.info("Sensitivity plots saved to %s/: heatmap + tornados", output_dir)
