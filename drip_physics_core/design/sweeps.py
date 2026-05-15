"""Parametric sweep utilities."""

from __future__ import annotations

from typing import List

import numpy as np

from .models import DesignPoint, DesignReport
from .physics_eval import evaluate_design


def sweep_diameter(
    base: DesignPoint,
    diameters: np.ndarray,
) -> List[DesignReport]:
    """Sweep droplet diameter, keeping all other parameters fixed.

    Uses the bang-bang reach approximation (use_sim_reach=False) — sweeps
    are hot loops where per-sample sim cost is unacceptable. The headline
    /api/evaluate path uses the sim-derived reach.
    """
    return [
        evaluate_design(
            base.model_copy(update={"droplet_diameter": float(d)}),
            use_sim_reach=False,
        )
        for d in diameters
    ]


def sweep_2d(
    base: DesignPoint,
    diameters: np.ndarray,
    chamber_temps: np.ndarray,
) -> np.ndarray:
    """Sweep diameter x chamber temperature.

    Returns structured array with key output fields.
    """
    dtype = np.dtype([
        ("diameter_um", float),
        ("chamber_C", float),
        ("T_arrival_C", float),
        ("liquid_fraction", float),
        ("ka", float),
        ("force_to_weight", float),
        ("orme_ratio", float),
        ("acoustic_bond_feasible", bool),
        ("feasible", bool),
    ])
    results = np.empty((len(diameters), len(chamber_temps)), dtype=dtype)

    for i, d in enumerate(diameters):
        for j, Tc in enumerate(chamber_temps):
            dp = base.model_copy(
                update={
                    "droplet_diameter": float(d),
                    "chamber_temperature": float(Tc),
                    "substrate_temperature": float(Tc),
                },
            )
            r = evaluate_design(dp, use_sim_reach=False)
            results[i, j] = (
                r.diameter_um,
                r.chamber_C,
                r.T_arrival_C,
                r.liquid_fraction_at_landing,
                r.ka,
                r.force_to_weight,
                r.orme_ratio,
                r.acoustic_bond_feasible if r.acoustic_bond_feasible is not None else False,
                r.feasible,
            )
    return results
