"""
Constraint system for the parametric design engine.

Converts user requirements (resolution, bonding, cost) into evaluatable
physics constraints. Computes feasibility, margins, and identifies
binding constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Sequence

from .models import BondingMode, DesignReport


class ConstraintCategory(Enum):
    PHYSICS = "physics"
    PERFORMANCE = "performance"
    COST = "cost"
    SAFETY = "safety"


@dataclass(frozen=True)
class RequirementSet:
    """What the user/customer needs from the machine."""

    resolution_um: Optional[float] = None
    bonding_mode: BondingMode = BondingMode.ACOUSTIC_BOND
    max_part_temp_C: Optional[float] = None
    max_cost_usd: Optional[float] = None
    min_throughput_cm3_hr: Optional[float] = None
    max_landing_error_mm: Optional[float] = None


@dataclass(frozen=True)
class ConstraintResult:
    """Result of evaluating a single constraint against a report."""

    name: str
    value: float
    limit: float
    margin: float           # positive = satisfied, negative = violated
    satisfied: bool
    category: ConstraintCategory


@dataclass(frozen=True)
class Constraint:
    """A named, evaluatable physics or performance constraint.

    Args:
        name: Human-readable constraint name.
        evaluate: Extracts the metric from a DesignReport.
        limit: Threshold value.
        category: Classification for reporting.
        upper_bound: If True, value < limit is good. If False, value > limit is good.
    """

    name: str
    evaluate: Callable[[DesignReport], float]
    limit: float
    category: ConstraintCategory
    upper_bound: bool = True

    def check(self, report: DesignReport) -> ConstraintResult:
        """Evaluate this constraint against a DesignReport."""
        value = self.evaluate(report)
        if self.upper_bound:
            margin = self.limit - value
            satisfied = value < self.limit
        else:
            margin = value - self.limit
            satisfied = value > self.limit
        return ConstraintResult(
            name=self.name,
            value=value,
            limit=self.limit,
            margin=margin,
            satisfied=satisfied,
            category=self.category,
        )


@dataclass(frozen=True)
class ConstraintSet:
    """Collection of constraints with batch evaluation."""

    constraints: tuple[Constraint, ...]

    def check(self, report: DesignReport) -> Dict[str, ConstraintResult]:
        """Evaluate all constraints. Returns dict keyed by constraint name."""
        return {c.name: c.check(report) for c in self.constraints}

    def feasible(self, report: DesignReport) -> bool:
        """True if all constraints are satisfied."""
        return all(c.check(report).satisfied for c in self.constraints)

    def margins(self, report: DesignReport) -> Dict[str, float]:
        """Margin for each constraint. Positive = satisfied."""
        return {c.name: c.check(report).margin for c in self.constraints}

    def binding(self, report: DesignReport, n: int = 3) -> list[str]:
        """Return the n constraints closest to violation (smallest margin)."""
        m = self.margins(report)
        return sorted(m, key=lambda k: m[k])[:n]

    @staticmethod
    def from_requirements(req: RequirementSet) -> ConstraintSet:
        """Build a constraint set from user requirements.

        Always includes physics-validity constraints (ka, Bi, force).
        Conditionally adds cost/performance constraints when requirements
        specify them.
        """
        constraints: list[Constraint] = []

        # -- Physics (always included) --
        constraints.append(Constraint(
            name="ka < 0.5 (Gor'kov valid)",
            evaluate=lambda r: r.ka,
            limit=0.5,
            category=ConstraintCategory.PHYSICS,
            upper_bound=True,
        ))
        constraints.append(Constraint(
            name="Bi << 0.1 (lumped capacitance)",
            evaluate=lambda r: r.Bi,
            limit=0.1,
            category=ConstraintCategory.PHYSICS,
            upper_bound=True,
        ))
        constraints.append(Constraint(
            name="Force > 0.1x weight",
            evaluate=lambda r: r.force_to_weight,
            limit=0.1,
            category=ConstraintCategory.PHYSICS,
            upper_bound=False,
        ))

        # -- Bonding --
        if req.bonding_mode == BondingMode.ACOUSTIC_BOND:
            constraints.append(Constraint(
                name="Acoustic bonding feasible",
                evaluate=lambda r: (
                    1.0 if r.acoustic_bond_feasible else 0.0
                ) if r.acoustic_bond_feasible is not None else 0.0,
                limit=0.5,
                category=ConstraintCategory.PERFORMANCE,
                upper_bound=False,
            ))
        else:
            constraints.append(Constraint(
                name="Orme remelt > 1.6",
                evaluate=lambda r: r.orme_ratio,
                limit=1.6,
                category=ConstraintCategory.PERFORMANCE,
                upper_bound=False,
            ))

        # -- Cost (conditional) --
        if req.max_cost_usd is not None:
            limit = req.max_cost_usd
            constraints.append(Constraint(
                name=f"Cost < ${limit:.0f}",
                evaluate=lambda r: r.total_cost,
                limit=limit,
                category=ConstraintCategory.COST,
                upper_bound=True,
            ))

        # -- Resolution (conditional) --
        if req.resolution_um is not None:
            limit = req.resolution_um
            constraints.append(Constraint(
                name=f"Resolution < {limit:.0f} um",
                evaluate=lambda r: r.diameter_um,
                limit=limit,
                category=ConstraintCategory.PERFORMANCE,
                upper_bound=True,
            ))

        # -- Steering range (always included) --
        constraints.append(Constraint(
            name="Steering range sufficient",
            evaluate=lambda r: 1.0 if r.displacement_ok else 0.0,
            limit=0.5,
            category=ConstraintCategory.PERFORMANCE,
            upper_bound=False,
        ))

        # -- Landing error (conditional) --
        if req.max_landing_error_mm is not None:
            raise NotImplementedError(
                "max_landing_error_mm is not wired to the design engine. "
                "Landing error requires a closed-loop controller simulation "
                "(pathtrack.py) — the design report does not currently expose "
                "landing-error statistics. Drop this requirement or implement."
            )

        # -- Part temperature (conditional) --
        if req.max_part_temp_C is not None:
            raise NotImplementedError(
                "max_part_temp_C is not wired to the design engine. "
                "Bulk part temperature requires a layer-stack thermal model "
                "that does not yet exist. Drop this requirement or implement."
            )

        return ConstraintSet(constraints=tuple(constraints))


# ---------------------------------------------------------------------------
# Pre-built sets
# ---------------------------------------------------------------------------

def l1_requirements() -> RequirementSet:
    """L1 demo requirements."""
    return RequirementSet(
        resolution_um=800.0,
        bonding_mode=BondingMode.ACOUSTIC_BOND,
        max_cost_usd=50_000.0,
    )


def l1_constraints() -> ConstraintSet:
    """L1 demo constraint set."""
    return ConstraintSet.from_requirements(l1_requirements())
