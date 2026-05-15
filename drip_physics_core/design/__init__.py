"""
Parametric system design engine for Drip 3D.

Connects derivation results (D1-D19 + Orme 1997) and the 20 kHz
sonotrode bonding model into a single coupled calculation. Every
number flows from a DesignPoint — zero hardcoding.

Usage:
    from drip_physics.design import evaluate_design, l1_design_point

    dp = l1_design_point()  # 600um, 200C chamber, 850C crucible
    report = evaluate_design(dp)
    print(format_report(report))
"""

from .bonding import (
    AcousticBondResult,
    OrmeBondResult,
    acoustic_bond,
    orme_remelt,
)
from .constraints import (
    Constraint,
    ConstraintCategory,
    ConstraintResult,
    ConstraintSet,
    RequirementSet,
    l1_constraints,
    l1_requirements,
)
from .models import BondingMode, DesignPoint, DesignReport, l1_design_point
from .physics_eval import PHYSICS_REGISTRY, evaluate_design
from .optimizer import (
    DesignVariable,
    FeasibilityMap,
    OptimizationResult,
    ParetoFrontier,
    ParetoPoint,
    feasibility_boundary,
    l1_optimize_diameter,
    optimize_single,
    pareto_frontier,
)
from .reports import format_report, generate_plots
from .sensitivity import (
    SensitivityResult,
    TornadoEntry,
    compute_sensitivity,
    generate_sensitivity_plots,
    tornado_data,
)
from .sweeps import sweep_2d, sweep_diameter
from .test_plan import (
    Experiment,
    ExperimentPlan,
    format_test_plan,
    generate_test_plan,
    l1_test_plan,
)
from .machine_spec import (
    BOMItem,
    ChamberDimensions,
    ControlRequirements,
    MachineSpec,
    SafetyEnvelope,
    ThermalDesign,
    format_machine_spec,
    generate_machine_spec,
    l1_machine_spec,
)
from .uncertainty import (
    MonteCarloResult,
    UncertainParameter,
    UncertaintyModel,
    VOIResult,
    monte_carlo,
    value_of_information,
)

__all__ = [
    "AcousticBondResult",
    "BondingMode",
    "Constraint",
    "ConstraintCategory",
    "ConstraintResult",
    "ConstraintSet",
    "DesignPoint",
    "DesignReport",
    "OrmeBondResult",
    "PHYSICS_REGISTRY",
    "RequirementSet",
    "SensitivityResult",
    "TornadoEntry",
    "acoustic_bond",
    "compute_sensitivity",
    "evaluate_design",
    "format_report",
    "generate_plots",
    "generate_sensitivity_plots",
    "l1_constraints",
    "l1_design_point",
    "l1_requirements",
    "orme_remelt",
    "sweep_2d",
    "sweep_diameter",
    "tornado_data",
    "DesignVariable",
    "FeasibilityMap",
    "OptimizationResult",
    "ParetoFrontier",
    "ParetoPoint",
    "feasibility_boundary",
    "l1_optimize_diameter",
    "optimize_single",
    "pareto_frontier",
    "MonteCarloResult",
    "UncertainParameter",
    "UncertaintyModel",
    "VOIResult",
    "monte_carlo",
    "value_of_information",
]
