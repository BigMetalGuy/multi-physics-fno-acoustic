"""
Uncertainty propagation via Monte Carlo simulation.

Defines uncertain parameters, propagates them through the design engine,
and computes value-of-information to rank experiments by impact.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .constraints import ConstraintSet, l1_constraints
from .models import DesignPoint, DesignReport, l1_design_point
from .physics_eval import evaluate_design
from .sensitivity import _get_field

logger = logging.getLogger(__name__)


def _wilson_interval(
    n_success: int, n_total: int, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (lower, upper) bounds. Defaults to 95% (z=1.96). For small
    n_total or proportions near 0/1, this is dramatically more honest
    than the Wald (raw fraction +/- z*sqrt(p(1-p)/n)) interval.
    """
    if n_total <= 0:
        return (0.0, 0.0)
    p = n_success / n_total
    z2 = z * z
    denom = 1.0 + z2 / n_total
    center = (p + z2 / (2.0 * n_total)) / denom
    half = (z / denom) * math.sqrt(
        p * (1.0 - p) / n_total + z2 / (4.0 * n_total * n_total)
    )
    return (max(0.0, center - half), min(1.0, center + half))


def _report_has_nan(report: DesignReport, fields: Sequence[str]) -> bool:
    """Return True if any of the named scalar fields on report is NaN/inf."""
    for name in fields:
        v = _get_field(report, name)
        if v is None:
            continue
        try:
            if not math.isfinite(float(v)):
                return True
        except (TypeError, ValueError):
            continue
    return False

# ---------------------------------------------------------------------------
# Default outputs to track
# ---------------------------------------------------------------------------
DEFAULT_MC_OUTPUTS: tuple[str, ...] = (
    "ka", "F_max", "force_to_weight", "T_arrival_C",
    "liquid_fraction_at_landing", "tau_conv", "cooling_rate",
    "orme_ratio", "total_cost", "Bi",
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UncertainParameter:
    """A design parameter with known uncertainty bounds."""

    name: str
    nominal: float
    low: float
    high: float
    distribution: Literal["uniform", "normal", "triangular"] = "uniform"
    source: str = ""

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw n samples from this parameter's distribution."""
        if self.distribution == "uniform":
            return rng.uniform(self.low, self.high, n)
        elif self.distribution == "normal":
            std = (self.high - self.low) / 4.0  # low/high ≈ 2-sigma
            samples = rng.normal(self.nominal, std, n)
            return np.clip(samples, self.low, self.high)
        elif self.distribution == "triangular":
            return rng.triangular(self.low, self.nominal, self.high, n)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


@dataclass(frozen=True)
class UncertaintyModel:
    """Collection of uncertain parameters for Monte Carlo analysis."""

    parameters: tuple[UncertainParameter, ...]

    def validate(self, dp: DesignPoint) -> None:
        """Raise ValueError if any parameter name doesn't match a DesignPoint field."""
        bad = []
        for p in self.parameters:
            if not hasattr(dp, p.name):
                bad.append(p.name)
            else:
                val = getattr(dp, p.name)
                if not isinstance(val, (int, float)):
                    bad.append(f"{p.name} (not numeric: {type(val).__name__})")
        if bad:
            raise ValueError(
                f"UncertaintyModel parameters not valid on DesignPoint: {bad}"
            )

    @staticmethod
    def l1_defaults() -> UncertaintyModel:
        """Pre-built uncertainty model for L1 analysis."""
        return UncertaintyModel(parameters=(
            UncertainParameter(
                "droplet_diameter", 600e-6, 550e-6, 650e-6,
                "normal", "Droplet generator variance +/- 50um (L1: 600um)",
            ),
            UncertainParameter(
                "crucible_temperature", 1123.15, 1098.15, 1148.15,
                "uniform", "Thermocouple accuracy +/- 25K",
            ),
            UncertainParameter(
                "chamber_temperature", 473.15, 448.15, 498.15,
                "uniform", "Chamber heating uniformity +/- 25K",
            ),
            UncertainParameter(
                "sonotrode_amplitude_um", 15.0, 10.0, 25.0,
                "triangular", "ASMP-060, sonotrode power calibration",
            ),
            UncertainParameter(
                "substrate_temperature", 473.15, 448.15, 498.15,
                "uniform", "Substrate heater accuracy +/- 25K",
            ),
        ))


@dataclass(frozen=True)
class MonteCarloResult:
    """Aggregated results from Monte Carlo uncertainty propagation.

    `confidence` is the point estimate n_feasible / n_evaluated.
    `confidence_lower_95` / `confidence_upper_95` are the Wilson 95%
    bounds — use these when reporting to a user, since the point
    estimate alone can be misleading at small n_samples.

    `n_evaluation_failures` tracks samples where evaluate_design raised
    or returned non-finite headline outputs. They are excluded from
    the feasibility denominator and from output_stats so that ODE
    failures don't masquerade as constraint violations.
    """

    n_samples: int
    n_feasible: int
    n_evaluation_failures: int
    confidence: float
    confidence_lower_95: float
    confidence_upper_95: float
    output_stats: dict          # output_name -> {mean, std, p5, p50, p95}
    binding_constraints: tuple  # ((name, violation_count), ...) sorted desc


@dataclass(frozen=True)
class VOIResult:
    """Value-of-information analysis result."""

    baseline_confidence: float
    parameter_voi: dict         # param_name -> delta confidence
    rankings: tuple             # param names sorted by VOI descending
    experiment_recommendations: tuple  # source tags for top-ranked


# ---------------------------------------------------------------------------
# Monte Carlo propagation
# ---------------------------------------------------------------------------
def monte_carlo(
    base: DesignPoint,
    model: UncertaintyModel,
    constraint_set: ConstraintSet,
    *,
    n_samples: int = 500,
    outputs: Sequence[str] = DEFAULT_MC_OUTPUTS,
    seed: int = 42,
) -> MonteCarloResult:
    """Propagate parameter uncertainties through the design engine.

    Samples N design points from the uncertainty distributions, evaluates
    each, and computes statistics on outputs and feasibility.

    Args:
        base: Nominal design point.
        model: Uncertain parameters to sample.
        constraint_set: Constraints to check feasibility.
        n_samples: Number of Monte Carlo samples (default 500, ~25s).
        outputs: DesignReport field names to track statistics.
        seed: Random seed for reproducibility.
    """
    model.validate(base)
    rng = np.random.default_rng(seed)

    # Pre-sample all parameters
    all_samples = {
        p.name: p.sample(rng, n_samples) for p in model.parameters
    }

    # Evaluate
    n_feasible = 0
    n_evaluation_failures = 0
    output_values: dict[str, list[float]] = {name: [] for name in outputs}
    constraint_violations: dict[str, int] = {}

    for i in range(n_samples):
        kwargs = {p.name: float(all_samples[p.name][i]) for p in model.parameters}
        dp_i = base.model_copy(update=kwargs)

        try:
            # Hot loop: bang-bang reach is sufficient. Headline reach
            # comes from the single sim-backed /api/evaluate call.
            report_i = evaluate_design(dp_i, use_sim_reach=False)
        except Exception as e:
            n_evaluation_failures += 1
            logger.debug("MC sample %d: evaluate_design raised %s", i, e)
            continue

        # Skip samples where headline outputs are non-finite — counting
        # these as constraint violations would mislabel ODE failures as
        # infeasibility. Treat them as evaluation failures instead.
        if _report_has_nan(report_i, outputs):
            n_evaluation_failures += 1
            logger.debug("MC sample %d: non-finite output, skipping", i)
            continue

        results_i = constraint_set.check(report_i)

        if all(cr.satisfied for cr in results_i.values()):
            n_feasible += 1
        else:
            for name, cr in results_i.items():
                if not cr.satisfied:
                    constraint_violations[name] = constraint_violations.get(name, 0) + 1

        for name in outputs:
            output_values[name].append(_get_field(report_i, name))

    n_evaluated = n_samples - n_evaluation_failures

    # Aggregate output statistics over evaluated samples only.
    output_stats: dict[str, dict[str, float]] = {}
    for name in outputs:
        vals = np.array(output_values[name], dtype=float)
        if vals.size == 0:
            output_stats[name] = {"mean": 0.0, "std": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}
            continue
        output_stats[name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p5": float(np.percentile(vals, 5)),
            "p50": float(np.percentile(vals, 50)),
            "p95": float(np.percentile(vals, 95)),
        }

    # Sort binding constraints by violation count
    binding = sorted(constraint_violations.items(), key=lambda x: -x[1])

    point = n_feasible / n_evaluated if n_evaluated > 0 else 0.0
    lo, hi = _wilson_interval(n_feasible, n_evaluated)

    if n_evaluation_failures > 0:
        logger.warning(
            "Monte Carlo: %d/%d samples failed evaluation (excluded from feasibility)",
            n_evaluation_failures, n_samples,
        )

    return MonteCarloResult(
        n_samples=n_samples,
        n_feasible=n_feasible,
        n_evaluation_failures=n_evaluation_failures,
        confidence=point,
        confidence_lower_95=lo,
        confidence_upper_95=hi,
        output_stats=output_stats,
        binding_constraints=tuple(binding),
    )


# ---------------------------------------------------------------------------
# Value-of-information analysis
# ---------------------------------------------------------------------------
def value_of_information(
    base: DesignPoint,
    model: UncertaintyModel,
    constraint_set: ConstraintSet,
    *,
    n_samples: int = 500,
    seed: int = 42,
) -> VOIResult:
    """Compute value-of-information for each uncertain parameter.

    For each parameter P:
      1. Run MC with all params uncertain -> baseline confidence
      2. Run MC with P fixed at nominal -> confidence_without_P
      3. VOI(P) = confidence_without_P - baseline_confidence

    Higher VOI = knowing P exactly improves confidence the most
    = this experiment is most valuable.

    Note: Total evaluations = n_samples * (1 + n_params).
    With defaults (500 samples, 5 params) = 3000 evals ~ 150s.
    Use n_samples=200 for quick estimates (~60s).
    """
    model.validate(base)

    # Baseline: all parameters uncertain
    baseline = monte_carlo(base, model, constraint_set,
                           n_samples=n_samples, seed=seed)

    # Leave-one-out: fix each parameter, run MC with the rest.
    # Each leave-one-out run uses an offset seed so the sample sets are
    # decorrelated — sharing the seed across runs would make the VOI
    # estimates statistically dependent (every variant draws the same
    # noise, just with one parameter pinned).
    param_voi: dict[str, float] = {}
    for i, p in enumerate(model.parameters):
        reduced_params = tuple(
            pp for pp in model.parameters if pp.name != p.name
        )
        reduced_model = UncertaintyModel(parameters=reduced_params)
        result_without = monte_carlo(base, reduced_model, constraint_set,
                                     n_samples=n_samples, seed=seed + i + 1)
        param_voi[p.name] = result_without.confidence - baseline.confidence

    # Rank by VOI descending
    ranked = sorted(param_voi.keys(), key=lambda k: -param_voi[k])

    # Extract source tags for top-ranked
    source_map = {p.name: p.source for p in model.parameters}
    recommendations = tuple(source_map.get(name, "") for name in ranked)

    return VOIResult(
        baseline_confidence=baseline.confidence,
        parameter_voi=param_voi,
        rankings=tuple(ranked),
        experiment_recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Pre-built convenience
# ---------------------------------------------------------------------------
def l1_uncertainty_analysis(
    n_samples: int = 200,
) -> tuple[MonteCarloResult, VOIResult]:
    """Run full uncertainty analysis with L1 defaults.

    Args:
        n_samples: Samples per MC run (default 200 for ~60s total).
    """
    dp = l1_design_point(chamber_temperature_C=200)
    model = UncertaintyModel.l1_defaults()
    cs = l1_constraints()

    mc = monte_carlo(dp, model, cs, n_samples=n_samples)
    voi = value_of_information(dp, model, cs, n_samples=n_samples)

    return mc, voi
