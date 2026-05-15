"""
Test plan generator — value-of-information driven experiment ranking.

Takes uncertainty analysis results and sensitivity data, combines them
to produce a prioritized list of experiments with decision criteria
and expected outcomes. This is the "what should we measure next?" engine.

Domain-review status: the ranking math is correct (priority_score =
(VOI × 100 + sensitivity) / cost × 1000, then sorted), but the
catalog feeding it (`_EXPERIMENT_CATALOG`) holds placeholder values
that need lab-level validation before driving real spend:

- Per-experiment cost (e.g. $500 for nozzle characterization, $1000
  for the TE-002 sonotrode amplitude sweep) — Jamie + Oscar should
  confirm against actual lab quotes / equipment rates.
- Per-experiment time (1-5 days) — Molly should confirm against lab
  scheduling.
- Decision criteria strings ("acceptable variance < ±10 µm" etc.)
  are stand-ins; domain owners should specify the real go/no-go
  thresholds.
- Cumulative confidence assumes parameter independence
  (`min(1.0, baseline + sum(voi_scores))`); the real coupling
  between chamber and substrate temperature, for instance, is not
  modeled.

Treat the output as a triage shortlist, not a procurement plan,
until the catalog has had a lab sign-off pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .constraints import ConstraintSet, l1_constraints
from .models import DesignPoint, l1_design_point
from .sensitivity import SensitivityResult, compute_sensitivity
from .uncertainty import (
    MonteCarloResult,
    UncertainParameter,
    UncertaintyModel,
    VOIResult,
    monte_carlo,
    value_of_information,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Experiment:
    """A proposed experiment to reduce parameter uncertainty."""

    name: str
    parameter: str                  # UncertainParameter.name it resolves
    source_tag: str                 # ASMP tag or reference
    description: str
    estimated_cost_usd: float
    estimated_time_days: float
    voi_score: float                # confidence improvement if this param is known
    sensitivity_score: float        # max |elasticity| of this param across outputs
    priority_score: float           # combined ranking metric
    decision_criterion: str         # "if X > Y, proceed; else redesign Z"
    expected_outcome_range: str     # "expecting 250-400°C based on model"
    reduces_uncertainty_on: str     # which outputs are most affected


@dataclass(frozen=True)
class ExperimentPlan:
    """Prioritized list of experiments."""

    experiments: tuple[Experiment, ...]
    baseline_confidence: float
    target_confidence: float
    total_cost_usd: float
    total_time_days: float
    cumulative_confidence: tuple[tuple[str, float], ...]  # after each experiment


# ---------------------------------------------------------------------------
# Experiment catalog — domain knowledge encoded as data
# ---------------------------------------------------------------------------

# Map from UncertainParameter.name to experiment metadata.
# These are PLACEHOLDERS — domain experts should refine costs, timelines,
# and decision criteria based on actual lab capabilities.
_EXPERIMENT_CATALOG: dict[str, dict] = {
    "droplet_diameter": {
        "name": "Nozzle characterization — droplet size distribution",
        "description": (
            "Measure droplet diameter distribution from each nozzle orifice "
            "using high-speed camera + image analysis. Establishes actual "
            "mean and variance for the droplet generator."
        ),
        "cost_usd": 500.0,
        "time_days": 2.0,
        "decision": (
            "If std < 20 um: proceed with nominal diameter. "
            "If std > 50 um: redesign nozzle or add size-selection gate."
        ),
        "expected": "Expecting 280-320 um based on 300 um nozzle orifice",
        "reduces": "ka, T_arrival_C, tau_conv, cooling_rate",
    },
    "crucible_temperature": {
        "name": "Thermocouple calibration — crucible temperature accuracy",
        "description": (
            "Calibrate Type-K thermocouple in crucible against reference "
            "pyrometer. Establish actual melt temperature at ejection."
        ),
        "cost_usd": 200.0,
        "time_days": 1.0,
        "decision": (
            "If accuracy < +/-10K: use existing thermocouple. "
            "If accuracy > +/-25K: upgrade to Type-S or add redundant sensor."
        ),
        "expected": "Expecting 840-860°C based on crucible heater setpoint",
        "reduces": "T_arrival_C, cooling_rate",
    },
    "chamber_temperature": {
        "name": "Chamber thermal mapping — spatial temperature profile",
        "description": (
            "Map chamber air temperature at 5+ locations along the array "
            "height using thermocouples. Establish uniformity and gradient."
        ),
        "cost_usd": 300.0,
        "time_days": 1.0,
        "decision": (
            "If uniformity < +/-15K: use single chamber temp in model. "
            "If gradient > 50K: implement linear_chamber_gradient() in config."
        ),
        "expected": "Expecting 175-225°C based on heater power and insulation",
        "reduces": "T_arrival_C, orme_ratio, cooling_rate",
    },
    "substrate_temperature": {
        "name": "Substrate heater characterization",
        "description": (
            "Measure substrate surface temperature under operating conditions "
            "with embedded thermocouple. Validate heater PID setpoint accuracy."
        ),
        "cost_usd": 200.0,
        "time_days": 1.0,
        "decision": (
            "If accuracy < +/-10K: proceed. "
            "If > +/-25K: improve PID tuning or add surface pyrometer."
        ),
        "expected": "Expecting 190-210°C at 200°C setpoint",
        "reduces": "orme_ratio (informational at this operating point)",
    },
    "sonotrode_amplitude_um": {
        "name": "TE-002 — Sonotrode amplitude and bonding threshold",
        "description": (
            "Measure actual sonotrode tip displacement amplitude via "
            "accelerometer at 20 kHz. Sweep 5-25 um and correlate with "
            "bond quality. THIS IS THE CRITICAL PATH EXPERIMENT."
        ),
        "cost_usd": 1000.0,    # TE-002 budget
        "time_days": 5.0,
        "decision": (
            "If bonding at 5 um amplitude: large margin, reduce power. "
            "If bonding only at 20+ um: tight margin, verify horn gain. "
            "If no bonding at 25 um: rethink approach (see TE-002 FAIL path)."
        ),
        "expected": "Expecting bonding threshold at 5-15 um based on UAM literature",
        "reduces": "acoustic_bond_feasible (primary L1 gate)",
    },
}


# ---------------------------------------------------------------------------
# Test plan generation
# ---------------------------------------------------------------------------
def generate_test_plan(
    dp: DesignPoint,
    model: UncertaintyModel,
    constraint_set: ConstraintSet,
    sensitivity: Optional[SensitivityResult] = None,
    voi: Optional[VOIResult] = None,
    *,
    n_mc_samples: int = 200,
    target_confidence: float = 0.99,
) -> ExperimentPlan:
    """Generate a prioritized test plan from uncertainty + sensitivity analysis.

    Priority score = VOI × sensitivity × (1 / cost). Higher = more valuable
    per dollar spent.

    Args:
        dp: Nominal design point.
        model: Uncertain parameters to analyze.
        constraint_set: Constraints for feasibility checking.
        sensitivity: Pre-computed sensitivity (computed if None).
        voi: Pre-computed VOI (computed if None).
        n_mc_samples: Samples for MC/VOI if not pre-computed.
        target_confidence: Desired feasibility confidence.
    """
    # Compute if not provided
    if sensitivity is None:
        sensitivity = compute_sensitivity(dp)
    if voi is None:
        voi = value_of_information(dp, model, constraint_set,
                                   n_samples=n_mc_samples)

    # Build experiment list
    experiments: list[Experiment] = []
    for param in model.parameters:
        catalog_entry = _EXPERIMENT_CATALOG.get(param.name, {})
        if not catalog_entry:
            # No catalog entry — create a generic one
            catalog_entry = {
                "name": f"Measure {param.name}",
                "description": f"Reduce uncertainty on {param.name}",
                "cost_usd": 500.0,
                "time_days": 2.0,
                "decision": f"Update {param.name} in model with measured value",
                "expected": f"Current range: {param.low:.4g} to {param.high:.4g}",
                "reduces": "unknown",
            }

        # VOI score for this parameter
        voi_score = voi.parameter_voi.get(param.name, 0.0)

        # Max sensitivity elasticity for this parameter across all outputs
        sens_score = 0.0
        for (inp, _out), elast in sensitivity.elasticity.items():
            if inp == param.name:
                sens_score = max(sens_score, abs(elast))

        # Combined priority: VOI * sensitivity / cost
        cost = catalog_entry["cost_usd"]
        priority = (voi_score * 100 + sens_score) / max(cost, 1.0) * 1000

        experiments.append(Experiment(
            name=catalog_entry["name"],
            parameter=param.name,
            source_tag=param.source,
            description=catalog_entry["description"],
            estimated_cost_usd=cost,
            estimated_time_days=catalog_entry["time_days"],
            voi_score=voi_score,
            sensitivity_score=sens_score,
            priority_score=priority,
            decision_criterion=catalog_entry["decision"],
            expected_outcome_range=catalog_entry["expected"],
            reduces_uncertainty_on=catalog_entry["reduces"],
        ))

    # Sort by priority descending
    experiments.sort(key=lambda e: e.priority_score, reverse=True)

    # Compute cumulative confidence (approximate — assumes independence)
    cumulative: list[tuple[str, float]] = []
    running_confidence = voi.baseline_confidence
    for exp in experiments:
        running_confidence = min(1.0, running_confidence + exp.voi_score)
        cumulative.append((exp.name, running_confidence))

    total_cost = sum(e.estimated_cost_usd for e in experiments)
    total_time = max((e.estimated_time_days for e in experiments), default=0)

    return ExperimentPlan(
        experiments=tuple(experiments),
        baseline_confidence=voi.baseline_confidence,
        target_confidence=target_confidence,
        total_cost_usd=total_cost,
        total_time_days=total_time,
        cumulative_confidence=tuple(cumulative),
    )


def format_test_plan(plan: ExperimentPlan) -> str:
    """Human-readable test plan report."""
    lines = [
        "=" * 70,
        "  DRIP L1 TEST PLAN — Experiment Priority",
        "=" * 70,
        f"  Baseline confidence: {plan.baseline_confidence:.1%}",
        f"  Target confidence:   {plan.target_confidence:.1%}",
        f"  Total budget:        ${plan.total_cost_usd:,.0f}",
        f"  Critical path:       {plan.total_time_days:.0f} days",
        "",
    ]

    for i, exp in enumerate(plan.experiments):
        lines += [
            f"  {'─' * 64}",
            f"  #{i+1}  {exp.name}",
            f"  {'─' * 64}",
            f"    Parameter:     {exp.parameter}",
            f"    Source:         {exp.source_tag}",
            f"    Cost:           ${exp.estimated_cost_usd:,.0f}  |  "
            f"Time: {exp.estimated_time_days:.0f} day(s)",
            f"    VOI:            +{exp.voi_score:.1%} confidence",
            f"    Sensitivity:    {exp.sensitivity_score:.2f} (max |elasticity|)",
            f"    Priority:       {exp.priority_score:.1f}",
            f"",
            f"    Description:",
            f"      {exp.description}",
            f"",
            f"    Decision criterion:",
            f"      {exp.decision_criterion}",
            f"",
            f"    Expected outcome:",
            f"      {exp.expected_outcome_range}",
            f"",
            f"    Reduces uncertainty on:",
            f"      {exp.reduces_uncertainty_on}",
            "",
        ]

    lines += [
        f"  {'─' * 64}",
        "  CUMULATIVE CONFIDENCE AFTER EACH EXPERIMENT:",
    ]
    for name, conf in plan.cumulative_confidence:
        marker = " <-- target" if conf >= plan.target_confidence else ""
        lines.append(f"    {conf:.1%}  after: {name}{marker}")

    lines += ["", "=" * 70]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-built convenience
# ---------------------------------------------------------------------------
def l1_test_plan(n_mc_samples: int = 200) -> ExperimentPlan:
    """Generate test plan for L1 with defaults."""
    dp = l1_design_point(chamber_temperature_C=200)
    model = UncertaintyModel.l1_defaults()
    cs = l1_constraints()
    return generate_test_plan(dp, model, cs, n_mc_samples=n_mc_samples)
