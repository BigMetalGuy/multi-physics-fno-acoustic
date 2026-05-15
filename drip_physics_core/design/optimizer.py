"""
Design space optimization.

Single-objective optimization, Pareto frontier generation, and
feasibility boundary mapping. Uses scipy.optimize for search
and grid + non-dominated sort for multi-objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats.qmc import LatinHypercube

from .constraints import ConstraintSet, l1_constraints
from .models import DesignPoint, DesignReport, l1_design_point
from .physics_eval import evaluate_design


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DesignVariable:
    """A DesignPoint field to optimize over."""

    name: str
    low: float
    high: float


@dataclass(frozen=True)
class OptimizationResult:
    """Result of single-objective optimization."""

    optimal_point: DesignPoint
    optimal_report: DesignReport
    objective_value: float
    n_evaluations: int
    feasible: bool
    converged: bool
    all_margins: dict


@dataclass(frozen=True)
class ParetoPoint:
    """One point on the Pareto frontier."""

    design_point: DesignPoint
    report: DesignReport
    objective_values: tuple


@dataclass(frozen=True)
class ParetoFrontier:
    """Set of non-dominated design points."""

    points: tuple
    objective_names: tuple
    n_samples_evaluated: int
    n_feasible: int


@dataclass(frozen=True)
class FeasibilityMap:
    """2D grid of feasibility and margin values."""

    x_values: np.ndarray
    y_values: np.ndarray
    feasible: np.ndarray        # bool, shape (len_x, len_y)
    min_margin: np.ndarray      # float, shape (len_x, len_y)
    x_name: str
    y_name: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_dp(
    base: DesignPoint,
    variables: Sequence[DesignVariable],
    x_vec: np.ndarray,
) -> DesignPoint:
    """Construct a DesignPoint from base + optimizer vector."""
    kwargs = {v.name: float(x_vec[i]) for i, v in enumerate(variables)}
    return base.model_copy(update=kwargs)


def _validate_variables(
    base: DesignPoint,
    variables: Sequence[DesignVariable],
) -> None:
    """Check that variable names are valid scalar fields on DesignPoint."""
    valid_names = set(base.__class__.model_fields.keys())
    for v in variables:
        if v.name not in valid_names:
            raise ValueError(
                f"DesignVariable '{v.name}' is not a field on DesignPoint. "
                f"Valid fields: {sorted(valid_names)}"
            )
        val = getattr(base, v.name)
        if not isinstance(val, (int, float)):
            raise ValueError(
                f"DesignVariable '{v.name}' must be numeric, got {type(val)}"
            )


# ---------------------------------------------------------------------------
# Single-objective optimization
# ---------------------------------------------------------------------------
def optimize_single(
    base: DesignPoint,
    objective: Callable[[DesignReport], float],
    constraint_set: ConstraintSet,
    variables: Sequence[DesignVariable],
    *,
    minimize: bool = True,
    max_evaluations: int = 500,
    seed: int = 42,
) -> OptimizationResult:
    """Find the optimal DesignPoint for a single objective.

    Uses scipy.optimize.differential_evolution (gradient-free, handles
    box bounds natively, robust with non-smooth objectives from ODE solves).

    Infeasible points receive a penalty of 1e12.
    """
    _validate_variables(base, variables)

    if not variables:
        report = evaluate_design(base)
        return OptimizationResult(
            optimal_point=base,
            optimal_report=report,
            objective_value=objective(report),
            n_evaluations=1,
            feasible=constraint_set.feasible(report),
            converged=True,
            all_margins=constraint_set.margins(report),
        )

    bounds = [(v.low, v.high) for v in variables]
    eval_count = [0]

    def cost(x: np.ndarray) -> float:
        eval_count[0] += 1
        dp = _make_dp(base, variables, x)
        # Optimizer inner loop: bang-bang reach is sufficient. The final
        # optimal_report below uses the sim-derived reach for headline.
        report = evaluate_design(dp, use_sim_reach=False)
        val = objective(report)
        if not minimize:
            val = -val
        if not constraint_set.feasible(report):
            val += 1e12
        return val

    result = differential_evolution(
        cost,
        bounds=bounds,
        maxiter=max_evaluations,
        seed=seed,
        tol=1e-6,
        polish=False,
    )

    optimal_dp = _make_dp(base, variables, result.x)
    optimal_report = evaluate_design(optimal_dp)
    obj_val = objective(optimal_report)

    return OptimizationResult(
        optimal_point=optimal_dp,
        optimal_report=optimal_report,
        objective_value=obj_val,
        n_evaluations=eval_count[0],
        feasible=constraint_set.feasible(optimal_report),
        converged=bool(result.success),
        all_margins=constraint_set.margins(optimal_report),
    )


# ---------------------------------------------------------------------------
# Pareto frontier (grid + non-dominated sort)
# ---------------------------------------------------------------------------
def _is_dominated(a: tuple, b: tuple) -> bool:
    """True if point b dominates point a (all objectives <=, at least one <)."""
    all_leq = all(bj <= aj for aj, bj in zip(a, b))
    any_lt = any(bj < aj for aj, bj in zip(a, b))
    return all_leq and any_lt


def pareto_frontier(
    base: DesignPoint,
    objectives: Sequence[tuple[str, Callable[[DesignReport], float]]],
    constraint_set: ConstraintSet,
    variables: Sequence[DesignVariable],
    *,
    n_samples: int = 2000,
    seed: int = 42,
) -> ParetoFrontier:
    """Find Pareto frontier via Latin Hypercube Sampling + non-dominated sort.

    All objectives are MINIMIZED. To maximize, negate the callable.
    """
    _validate_variables(base, variables)

    n_vars = len(variables)
    sampler = LatinHypercube(d=n_vars, seed=seed)
    samples = sampler.random(n=n_samples)

    # Scale samples to bounds
    lows = np.array([v.low for v in variables])
    highs = np.array([v.high for v in variables])
    scaled = samples * (highs - lows) + lows

    # Evaluate all samples
    feasible_points: list[ParetoPoint] = []
    for i in range(n_samples):
        dp = _make_dp(base, variables, scaled[i])
        report = evaluate_design(dp, use_sim_reach=False)
        if constraint_set.feasible(report):
            obj_vals = tuple(obj_fn(report) for _, obj_fn in objectives)
            feasible_points.append(ParetoPoint(
                design_point=dp,
                report=report,
                objective_values=obj_vals,
            ))

    # Non-dominated sort (O(n^2), fine for n < 2000)
    non_dominated: list[ParetoPoint] = []
    for i, pi in enumerate(feasible_points):
        dominated = False
        for j, pj in enumerate(feasible_points):
            if i != j and _is_dominated(pi.objective_values, pj.objective_values):
                dominated = True
                break
        if not dominated:
            non_dominated.append(pi)

    # Sort by first objective
    non_dominated.sort(key=lambda p: p.objective_values[0])

    return ParetoFrontier(
        points=tuple(non_dominated),
        objective_names=tuple(name for name, _ in objectives),
        n_samples_evaluated=n_samples,
        n_feasible=len(feasible_points),
    )


# ---------------------------------------------------------------------------
# Feasibility boundary map
# ---------------------------------------------------------------------------
def feasibility_boundary(
    base: DesignPoint,
    constraint_set: ConstraintSet,
    var_x: DesignVariable,
    var_y: DesignVariable,
    *,
    resolution: int = 50,
) -> FeasibilityMap:
    """Compute 2D feasibility map over two design variables.

    Note: resolution=50 means 2500 evaluations (~125s at 50ms each).
    Use resolution=25 for faster results (~31s).
    """
    _validate_variables(base, [var_x, var_y])

    x_vals = np.linspace(var_x.low, var_x.high, resolution)
    y_vals = np.linspace(var_y.low, var_y.high, resolution)

    feasible_grid = np.zeros((resolution, resolution), dtype=bool)
    margin_grid = np.zeros((resolution, resolution))

    for i, xi in enumerate(x_vals):
        for j, yj in enumerate(y_vals):
            dp = base.model_copy(update={var_x.name: float(xi), var_y.name: float(yj)})
            report = evaluate_design(dp, use_sim_reach=False)
            margins = constraint_set.margins(report)
            feasible_grid[i, j] = constraint_set.feasible(report)
            margin_grid[i, j] = min(margins.values()) if margins else 0.0

    return FeasibilityMap(
        x_values=x_vals,
        y_values=y_vals,
        feasible=feasible_grid,
        min_margin=margin_grid,
        x_name=var_x.name,
        y_name=var_y.name,
    )


# ---------------------------------------------------------------------------
# Pre-built convenience functions
# ---------------------------------------------------------------------------
def l1_optimize_diameter() -> OptimizationResult:
    """Find optimal droplet diameter for L1 (maximize arrival temperature)."""
    return optimize_single(
        base=l1_design_point(chamber_temperature_C=200),
        objective=lambda r: -r.T_arrival_C,  # maximize by negating
        constraint_set=l1_constraints(),
        variables=[
            DesignVariable("droplet_diameter", 100e-6, 800e-6),
        ],
        minimize=True,
        max_evaluations=200,
    )
