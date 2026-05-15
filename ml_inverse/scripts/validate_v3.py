#!/usr/bin/env python3
"""Validate Phase 1 v3 dataset against the success criteria.

The v3 agent bailed before running validation; this script is the
falsifiable check that v3 actually fixed the v1/v2 pathologies.

Success criteria (from PHASE1 v3 spec):

1. **Im symmetry** — rollout field_imag should be roughly symmetric around
   zero with std comparable to Re std (not 10x narrower like v1/v2).
2. **Cross-rollout phase divergence** — circular phase variance across
   rollouts should be substantially > v1's ~0.005 mean.
3. **Quality gates** — 0 NaN, 0 inf, 0 all-zero fields, phases in [0, 2pi),
   train/val/test disjoint, every split contains both sources.

Compares v3 vs v2 stats side-by-side. Saves a markdown report.

Usage::

    cd simulations
    python ml_inverse/scripts/validate_v3.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

ML_INV = _REPO / "ml_inverse"
DATA = ML_INV / "data"
V1 = DATA / "inverse_dataset.h5"
V2 = DATA / "inverse_dataset_v2.h5"
V3 = DATA / "inverse_dataset_v3.h5"


def _stats(arr: np.ndarray) -> Dict[str, float]:
    a = np.asarray(arr).ravel()
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
    }


def _stratified_field_stats(h5_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute Re/Im stats stratified by source (uniform/rollout)."""
    import h5py

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    with h5py.File(h5_path, "r") as f:
        source = f["source"][:]
        for src_name, src_code in [("uniform", 0), ("rollout", 1)]:
            mask = source == src_code
            if not mask.any():
                out[src_name] = {}
                continue
            re = f["field_real"][mask]
            im = f["field_imag"][mask]
            out[src_name] = {
                "field_real": _stats(re),
                "field_imag": _stats(im),
            }
    return out


def _cross_rollout_phase_variance(h5_path: Path) -> Dict[str, float]:
    """Mean / max circular phase variance across rollouts at each (frame, transducer).

    Replicates Spike B from 02_rollout_diagnostics.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        source = f["source"][:]
        rollout_mask = source == 1
        traj_id = f["trajectory_id"][rollout_mask]
        frame_idx = f["frame_idx"][rollout_mask]
        phases = f["phases"][rollout_mask]  # (n_rollout_samples, n_transducers)

    unique_traj = np.unique(traj_id)
    if unique_traj.size < 2:
        return {"mean": float("nan"), "max": float("nan"), "n_rollouts": int(unique_traj.size)}

    # For each frame_idx, gather phase configs across trajectories.
    unique_frames = np.unique(frame_idx)

    variances = []
    for fi in unique_frames:
        frame_mask = frame_idx == fi
        if frame_mask.sum() < 2:
            continue
        phs = phases[frame_mask]  # (n_traj_with_this_frame, n_transducers)
        # circular variance per transducer = 1 - |mean(exp(i*phi))|
        z = np.exp(1j * phs.astype(np.float64))
        mean_z = z.mean(axis=0)  # (n_transducers,)
        circ_var = 1.0 - np.abs(mean_z)  # (n_transducers,)
        variances.append(circ_var.mean())  # mean over transducers

    if not variances:
        return {"mean": float("nan"), "max": float("nan"), "n_rollouts": int(unique_traj.size)}

    return {
        "mean": float(np.mean(variances)),
        "max": float(np.max(variances)),
        "n_rollouts": int(unique_traj.size),
        "n_frames_compared": len(variances),
    }


def _quality_gates(h5_path: Path) -> Dict[str, bool]:
    """Check NaN/inf/all-zero/phase-range/split-disjoint/both-sources."""
    import h5py

    sys.path.insert(0, str(ML_INV))
    from dataset import InverseSurrogateDataset

    gates: Dict[str, bool] = {}
    with h5py.File(h5_path, "r") as f:
        re = f["field_real"][:]
        im = f["field_imag"][:]
        phases = f["phases"][:]
        gates["nan_count_zero"] = bool(
            (~np.isfinite(re)).sum() == 0 and (~np.isfinite(im)).sum() == 0
        )
        gates["all_zero_field_count_zero"] = bool(
            (np.all(re == 0, axis=tuple(range(1, re.ndim)))
             & np.all(im == 0, axis=tuple(range(1, im.ndim)))).sum() == 0
        )
        gates["phases_in_range"] = bool(
            (phases >= 0).all() and (phases < 2 * np.pi + 1e-6).all()
        )

    # Splits: load via Dataset, three partitions, verify disjoint.
    train = InverseSurrogateDataset(str(h5_path), split="train")
    val = InverseSurrogateDataset(str(h5_path), split="val")
    test = InverseSurrogateDataset(str(h5_path), split="test")

    sizes = train.partition_sizes
    gates["splits_nonempty"] = bool(sizes["train"] > 0 and sizes["val"] > 0 and sizes["test"] > 0)
    # Trajectory disjointness is asserted internally; if we got here, it passed.
    gates["splits_disjoint_asserted"] = True

    # Both sources in every split.
    for split, ds in [("train", train), ("val", val), ("test", test)]:
        sources_present = set(int(s) for s in np.unique(ds.source_array))
        gates[f"split_{split}_has_both_sources"] = sources_present == {0, 1}

    return gates


def _empirical_z_range(h5_path: Path) -> Dict[str, Tuple[float, float]]:
    """Return slice_z_m range for uniform vs rollout, if column exists."""
    import h5py

    out: Dict[str, Tuple[float, float]] = {}
    with h5py.File(h5_path, "r") as f:
        if "slice_z_m" not in f:
            return out
        z = f["slice_z_m"][:]
        source = f["source"][:]
        for src_name, src_code in [("uniform", 0), ("rollout", 1)]:
            mask = source == src_code
            if mask.any():
                out[src_name] = (float(z[mask].min()), float(z[mask].max()))
    return out


def _format_md_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---:" if h != headers[0] else "---" for h in headers]) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> None:
    if not V3.exists():
        print(f"ERROR: {V3} does not exist. Run generate_dataset.py first.", file=sys.stderr)
        sys.exit(2)
    if not V2.exists():
        print(f"WARNING: {V2} not found; v3-vs-v2 comparison will be partial.", file=sys.stderr)

    print(f"Validating {V3}...")
    v3_stats = _stratified_field_stats(V3)
    v3_phase_var = _cross_rollout_phase_variance(V3)
    v3_gates = _quality_gates(V3)
    v3_z_range = _empirical_z_range(V3)

    v2_stats = _stratified_field_stats(V2) if V2.exists() else {}
    v1_phase_var = _cross_rollout_phase_variance(V1) if V1.exists() else {"mean": float("nan"), "max": float("nan")}

    # Report.
    out_lines = []
    out_lines.append("# Phase 1 v3 — Validation Report")
    out_lines.append("")
    out_lines.append("Auto-generated by `validate_v3.py`. Compares v3 against v2 stats and v1 phase variance baseline.")
    out_lines.append("")

    # Success criterion 1: Im symmetry.
    out_lines.append("## 1. Im symmetry (PRIMARY SUCCESS CRITERION)")
    out_lines.append("")
    out_lines.append("v3 rollout `field_imag` should be symmetric around 0 with std comparable to Re std (not 10x narrower).")
    out_lines.append("")
    re3 = v3_stats.get("rollout", {}).get("field_real", {})
    im3 = v3_stats.get("rollout", {}).get("field_imag", {})
    re2 = v2_stats.get("rollout", {}).get("field_real", {}) if v2_stats else {}
    im2 = v2_stats.get("rollout", {}).get("field_imag", {}) if v2_stats else {}

    rows = [
        ["v3 rollout Re", re3.get("min"), re3.get("max"), re3.get("mean"), re3.get("std")],
        ["v3 rollout Im", im3.get("min"), im3.get("max"), im3.get("mean"), im3.get("std")],
        ["v2 rollout Re", re2.get("min", "n/a"), re2.get("max", "n/a"), re2.get("mean", "n/a"), re2.get("std", "n/a")],
        ["v2 rollout Im", im2.get("min", "n/a"), im2.get("max", "n/a"), im2.get("mean", "n/a"), im2.get("std", "n/a")],
    ]
    out_lines.append(_format_md_table(rows, ["series", "min", "max", "mean", "std"]))
    out_lines.append("")

    if im3 and re3:
        ratio_v3 = im3["std"] / re3["std"] if re3["std"] > 0 else float("nan")
        # Symmetry check: |mean| / std small AND std comparable to Re std.
        sym = abs(im3["mean"]) / im3["std"] if im3["std"] > 0 else float("inf")
        verdict = "PASS" if (ratio_v3 > 0.5 and sym < 0.3) else "FAIL"
        out_lines.append(f"- v3 Im_std / Re_std = **{ratio_v3:.3f}** (target: > 0.5; v1/v2 was ~0.10)")
        out_lines.append(f"- v3 Im |mean|/std = **{sym:.3f}** (target: < 0.3 for roughly symmetric; v1/v2 was ~2.1)")
        out_lines.append(f"- **Verdict: {verdict}**")
        out_lines.append("")

    # Success criterion 2: cross-rollout phase variance.
    out_lines.append("## 2. Cross-rollout phase divergence")
    out_lines.append("")
    out_lines.append("v3 rollouts should produce more diverse phase configs than v1 (was ~0.005 mean circular variance).")
    out_lines.append("")
    rows = [
        ["v3", v3_phase_var.get("mean"), v3_phase_var.get("max"), v3_phase_var.get("n_rollouts"), v3_phase_var.get("n_frames_compared")],
        ["v1 (baseline)", v1_phase_var.get("mean"), v1_phase_var.get("max"), v1_phase_var.get("n_rollouts", "n/a"), v1_phase_var.get("n_frames_compared", "n/a")],
    ]
    out_lines.append(_format_md_table(rows, ["dataset", "mean circ_var", "max circ_var", "n_rollouts", "n_frames"]))
    out_lines.append("")
    if v3_phase_var.get("mean") is not None:
        improved = v3_phase_var["mean"] > 5 * (v1_phase_var.get("mean") or 0)
        verdict2 = "PASS" if improved else "WEAK"
        out_lines.append(f"- v3 mean circ_var = **{v3_phase_var['mean']:.4f}** (v1 was {v1_phase_var.get('mean', float('nan')):.4f})")
        out_lines.append(f"- **Verdict: {verdict2}** (target: > 5x v1 baseline)")
        out_lines.append("")

    # Empirical z-range.
    out_lines.append("## 3. Empirical slice_z_m range")
    out_lines.append("")
    rows = []
    for src in ["uniform", "rollout"]:
        rng = v3_z_range.get(src)
        rows.append([src, rng[0] if rng else "n/a", rng[1] if rng else "n/a"])
    out_lines.append(_format_md_table(rows, ["source", "z_min (m)", "z_max (m)"]))
    out_lines.append("")
    out_lines.append("Both ranges should sit within [0, 0.20] m (the array region) for the Im pathology to dissolve.")
    out_lines.append("")

    # Quality gates.
    out_lines.append("## 4. Quality gates")
    out_lines.append("")
    rows = []
    for k, v in v3_gates.items():
        rows.append([k, "PASS" if v else "FAIL"])
    out_lines.append(_format_md_table(rows, ["gate", "status"]))
    out_lines.append("")

    # Summary.
    out_lines.append("## Summary")
    out_lines.append("")
    primary_pass = (im3 and re3 and im3["std"] / re3["std"] > 0.5 and abs(im3["mean"]) / im3["std"] < 0.3) if im3 else False
    secondary_pass = v3_phase_var.get("mean", 0) > 5 * (v1_phase_var.get("mean") or 0.001)
    all_gates = all(v3_gates.values()) if v3_gates else False
    out_lines.append(f"- Primary criterion (Im symmetry): **{'PASS' if primary_pass else 'FAIL'}**")
    out_lines.append(f"- Secondary criterion (rollout diversity): **{'PASS' if secondary_pass else 'WEAK'}**")
    out_lines.append(f"- All quality gates: **{'PASS' if all_gates else 'FAIL'}**")

    report = "\n".join(out_lines)
    out_path = ML_INV / "notebooks" / "v3_validation_report.md"
    out_path.write_text(report)

    # Print compact stdout summary.
    print(report)
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
