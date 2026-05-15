"""Emit ``comparison_v0_vs_v1.json`` from two ``summary.json`` receipts.

Reads ``_fem_three_way_v0/summary.json`` and ``_fem_three_way_v1/summary.json``
and writes a compact diff into v1's directory.  The diff is the empirical
proof that the j-Wave grid resolution fix did (or did not) break the v0
anomaly ``jWave-vs-FEM > analytical-vs-FEM`` in 5/5 configs.

Usage::

    PYTHONPATH=. python3 drip_physics/backends/femcoupled/compare_three_way_v0_v1.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


RESEARCH_DIR = Path(__file__).resolve().parents[3] / "ml_inverse" / "research"


def _load(p: Path) -> Dict:
    with open(p) as f:
        return json.load(f)


def _agg_delta(v0: Dict, v1: Dict, key: str) -> Dict[str, float]:
    """Diff a single ``rel_l2_aggregate[key]`` block (mean/median/p90/min/max)."""
    a = v0["rel_l2_aggregate"][key]
    b = v1["rel_l2_aggregate"][key]
    return {
        stat: {"v0": float(a[stat]), "v1": float(b[stat]), "delta": float(b[stat] - a[stat])}
        for stat in ("mean", "median", "p90", "min", "max")
    }


def _per_config_delta(v0: Dict, v1: Dict) -> list[Dict]:
    out = []
    for c0, c1 in zip(v0["per_config"], v1["per_config"]):
        assert c0["label"] == c1["label"], (
            f"label mismatch: {c0['label']} vs {c1['label']}"
        )
        rl0 = c0["rel_l2"]
        rl1 = c1["rel_l2"]
        out.append({
            "idx": c0["idx"],
            "label": c0["label"],
            "rel_l2_v0": rl0,
            "rel_l2_v1": rl1,
            "delta": {
                k: float(rl1[k] - rl0[k])
                for k in ("analytical_vs_jwave", "analytical_vs_femcoupled", "jwave_vs_femcoupled")
            },
            "anomaly_v0": rl0["jwave_vs_femcoupled"] > rl0["analytical_vs_femcoupled"],
            "anomaly_v1": rl1["jwave_vs_femcoupled"] > rl1["analytical_vs_femcoupled"],
        })
    return out


def main(
    v0_dir: Path | None = None,
    v1_dir: Path | None = None,
    out_path: Path | None = None,
) -> None:
    v0_dir = Path(v0_dir) if v0_dir is not None else RESEARCH_DIR / "_fem_three_way_v0"
    v1_dir = Path(v1_dir) if v1_dir is not None else RESEARCH_DIR / "_fem_three_way_v1"
    out_path = Path(out_path) if out_path is not None else v1_dir / "comparison_v0_vs_v1.json"

    v0 = _load(v0_dir / "summary.json")
    v1 = _load(v1_dir / "summary.json")

    diff = {
        "v0": {
            "timestamp": v0["timestamp"],
            "jwave_grid_n": v0["config"].get("jwave_grid_n"),
            "jwave_grid_dx_m": v0["config"].get("jwave_grid_dx_m"),
            "anomaly_count": v0["anomalies"]["fem_jw_exceeds_an_fc_count"],
        },
        "v1": {
            "timestamp": v1["timestamp"],
            "jwave_grid_n": v1["config"].get("jwave_grid_n"),
            "jwave_grid_dx_m": v1["config"].get("jwave_grid_dx_m"),
            "jwave_lambda_over_dx": v1["config"].get("jwave_lambda_over_dx"),
            "anomaly_count": v1["anomalies"]["fem_jw_exceeds_an_fc_count"],
        },
        "rel_l2_aggregate_delta": {
            "analytical_vs_jwave": _agg_delta(v0, v1, "analytical_vs_jwave"),
            "analytical_vs_femcoupled": _agg_delta(v0, v1, "analytical_vs_femcoupled"),
            "jwave_vs_femcoupled": _agg_delta(v0, v1, "jwave_vs_femcoupled"),
        },
        "per_config": _per_config_delta(v0, v1),
        "anomaly_resolved": (
            v0["anomalies"]["fem_jw_exceeds_an_fc_count"] == 5
            and v1["anomalies"]["fem_jw_exceeds_an_fc_count"] < v0["anomalies"]["fem_jw_exceeds_an_fc_count"]
        ),
        "verdict": (
            "Under-resolution diagnosis CONFIRMED — anomaly count dropped."
            if v1["anomalies"]["fem_jw_exceeds_an_fc_count"] < v0["anomalies"]["fem_jw_exceeds_an_fc_count"]
            else "Diagnosis INCOMPLETE — anomaly persists; investigate beyond grid resolution."
        ),
    }

    with open(out_path, "w") as f:
        json.dump(diff, f, indent=2)
    print(f"wrote {out_path}")
    print(f"v0 anomaly count: {diff['v0']['anomaly_count']} / 5")
    print(f"v1 anomaly count: {diff['v1']['anomaly_count']} / 5")
    print(f"verdict: {diff['verdict']}")


if __name__ == "__main__":
    main()
