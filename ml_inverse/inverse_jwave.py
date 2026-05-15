"""Differentiable inverse solver — gradient descent through the j-Wave PDE.

JAX/optax counterpart to ``inverse.py`` (which runs through the FNO v2.1
PyTorch surrogate). Same architectural pattern — given a target pressure
field, recover the per-transducer phase vector by Adam descent on a loss
``mean|p_pred − p_target|^2`` — but the forward model here is the full 3D
Helmholtz solve from ``drip_physics.backends.jwave_backend.JWaveBackend``,
backed by ``jax.value_and_grad`` rather than torch autograd.

The shared insight (Phase-3 architectural bet): **the inverse loop is
invariant to which differentiable forward model is plugged in.** Same
optimiser, same loss, same convergence pattern — only the physics changes.

CLI
---
::

    python -m ml_inverse.inverse_jwave \
        --dataset ml_inverse/data/inverse_dataset_jwave_3d.h5 \
        --target-idx 0 1 2 3 4 \
        --n-iters 50 \
        --n-restarts 1 \
        --lr 1e-2 \
        --output-dir ml_inverse/research/_jwave_inverse

Notes
-----
- Targets are loaded from a j-Wave-grounded dataset (must have
  ``attrs.physics_backend == 'jwave'``).
- ``--n-restarts`` controls how many random-init restarts are run per
  target. The "best" restart (lowest final loss) is reported as the
  primary result; per-restart phases are also saved for multi-modality
  diagnostics (pairwise phase L2 between restarts).
- Random init (no warm-start) so results are comparable across targets.
- Loss is computed on the raw complex 3D field (``mean |p_pred − p_target|^2``),
  matching the spike's setup.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow running as a script: drip_physics is the package one level up.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


# =============================================================================
# Result types
# =============================================================================


@dataclass(frozen=True)
class JWaveRestartResult:
    """Per-target inverse result through the j-Wave PDE."""

    target_idx: int
    seed: int
    n_iters: int
    initial_loss: float
    final_loss: float
    loss_curve: np.ndarray  # shape (n_iters,)
    grad_finite: np.ndarray  # bool, shape (n_iters,)
    iter_wall_s: np.ndarray  # shape (n_iters,)
    phases_init: np.ndarray  # shape (n_t,)
    phases_final: np.ndarray  # shape (n_t,)
    phases_true: np.ndarray  # shape (n_t,) — GT phases for the target
    final_field_error: float  # ||p_pred - p_target|| / ||p_target||
    phase_recovery_error: float  # mean-shift-removed L2 phase err
    wall_time_s: float

    def to_summary_dict(self) -> Dict[str, object]:
        """Compact JSON-serializable summary (drops large arrays)."""
        return {
            "target_idx": int(self.target_idx),
            "seed": int(self.seed),
            "n_iters": int(self.n_iters),
            "initial_loss": float(self.initial_loss),
            "final_loss": float(self.final_loss),
            "final_field_error": float(self.final_field_error),
            "phase_recovery_error": float(self.phase_recovery_error),
            "wall_time_s": float(self.wall_time_s),
            "iter_wall_s_mean": float(self.iter_wall_s.mean()),
            "iter_wall_s_median": float(np.median(self.iter_wall_s)),
            "all_grads_finite": bool(self.grad_finite.all()),
            "n_grads_finite": int(self.grad_finite.sum()),
            "loss_ratio_final_initial": float(
                self.final_loss / max(self.initial_loss, 1e-30)
            ),
        }


# =============================================================================
# Phase recovery error (mean-shift-removed L2)
# =============================================================================


def phase_recovery_error(
    phases_pred: np.ndarray, phases_true: np.ndarray
) -> float:
    """Wrapped, mean-shift-removed relative L2 phase error.

    Acoustic inverse problems are equivariant under a global phase
    constant. We wrap the per-transducer phase difference into
    ``[-pi, pi)`` to handle the 2-pi periodicity correctly, then
    subtract the mean difference (the global-shift component) before
    computing relative L2 against the mean-centred GT.

    Returns 0.0 for ``phases_pred == phases_true`` and for any global
    constant shift; ~1.0 for random phases.
    """
    diff = np.asarray(phases_pred, dtype=np.float64) - np.asarray(
        phases_true, dtype=np.float64
    )
    # Wrap into [-pi, pi).
    diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
    # Remove the global-shift mode — but only the mean *of the wrapped diff*.
    diff_centered = diff - diff.mean()
    # Reference scale: mean-centred GT phases (also wrapped to [-pi, pi)).
    true_centered = np.asarray(phases_true, dtype=np.float64)
    true_centered = true_centered - true_centered.mean()
    num = float(np.linalg.norm(diff_centered))
    den = float(np.linalg.norm(true_centered))
    return num / max(den, 1e-12)


# =============================================================================
# Solver
# =============================================================================


def solve_inverse_jwave(
    *,
    backend,
    array,
    env,
    target_field_jnp,
    phases_true: np.ndarray,
    target_idx: int,
    n_iters: int = 50,
    lr: float = 1e-2,
    seed: int = 42,
) -> JWaveRestartResult:
    """Run gradient-descent inverse on a single target through j-Wave.

    Parameters
    ----------
    backend
        ``JWaveBackend`` instance (already configured).
    array
        ``ArrayGeometry`` matching the dataset.
    env
        ``EnvironmentConfig`` matching the dataset.
    target_field_jnp
        Target field as a ``jnp.ndarray`` of shape ``(n, n, n)`` complex.
    phases_true
        Ground-truth phases used to make ``target_field_jnp``. Used only
        for the phase-recovery-error diagnostic — never as init.
    target_idx
        Index of this target (for logging / receipts).
    n_iters
        Adam iterations.
    lr
        Adam learning rate. Spike validated 1e-2.
    seed
        RNG seed for the random init. Restart 0 uses ``seed``.

    Returns
    -------
    JWaveRestartResult
    """
    import jax
    import jax.numpy as jnp
    import optax

    n_t = int(array.num_transducers)
    amps_jnp = jnp.ones(n_t, dtype=jnp.float32)

    # Stop-gradient on target (it is data, not a parameter).
    target_field_jnp = jax.lax.stop_gradient(target_field_jnp)

    def loss_fn(phi):
        field = backend._solve_field_jax(array, phi, amps_jnp, env)
        diff = field - target_field_jnp
        return jnp.mean(jnp.abs(diff) ** 2)

    value_and_grad_fn = jax.value_and_grad(loss_fn)

    # Random init (no warm-start) — keep targets comparable.
    rng = np.random.default_rng(int(seed))
    phases_init_np = rng.uniform(0.0, 2.0 * np.pi, size=n_t).astype(np.float32)
    phases_init_jnp = jnp.asarray(phases_init_np)

    optimizer = optax.adam(float(lr))
    opt_state = optimizer.init(phases_init_jnp)

    losses: List[float] = []
    grad_finite: List[bool] = []
    iter_walls: List[float] = []
    phases = phases_init_jnp

    # Initial loss (separate eval so the curve aligns to "loss BEFORE step i").
    initial_loss = float(loss_fn(phases))

    t_start = time.time()
    for i in range(int(n_iters)):
        t_iter0 = time.time()
        loss_val, g = value_and_grad_fn(phases)
        loss_val.block_until_ready()
        g.block_until_ready()
        finite = bool(jnp.all(jnp.isfinite(g)))
        updates, opt_state = optimizer.update(g, opt_state)
        phases = optax.apply_updates(phases, updates)
        # Wrap to [0, 2pi) — cosmetic; loss is 2pi-periodic.
        phases = phases % (2.0 * jnp.pi)
        t_iter = time.time() - t_iter0

        losses.append(float(loss_val))
        grad_finite.append(finite)
        iter_walls.append(float(t_iter))

        # Per-iter log only at start + every 10 + last iter.
        if i < 3 or i % 10 == 0 or i == n_iters - 1:
            gnorm = float(jnp.linalg.norm(g))
            logger.info(
                "  target=%d iter=%3d loss=%.4e ||grad||=%.3e finite=%s t=%.2fs",
                target_idx,
                i,
                float(loss_val),
                gnorm,
                finite,
                t_iter,
            )

    wall = time.time() - t_start
    final_loss = losses[-1] if losses else float("nan")

    phases_final_np = np.asarray(phases) % (2.0 * np.pi)
    phases_true_mod = (np.asarray(phases_true).astype(np.float32)) % (2.0 * np.pi)

    rec_err = phase_recovery_error(phases_final_np, phases_true_mod)

    # Final field error in raw space.
    with_final = backend._solve_field_jax(
        array,
        jnp.asarray(phases_final_np, dtype=jnp.float32),
        amps_jnp,
        env,
    )
    with_final.block_until_ready()
    diff_final = with_final - target_field_jnp
    field_err = float(
        jnp.linalg.norm(diff_final)
        / jnp.maximum(jnp.linalg.norm(target_field_jnp), 1e-30)
    )

    return JWaveRestartResult(
        target_idx=int(target_idx),
        seed=int(seed),
        n_iters=int(len(losses)),
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        loss_curve=np.asarray(losses, dtype=np.float64),
        grad_finite=np.asarray(grad_finite, dtype=bool),
        iter_wall_s=np.asarray(iter_walls, dtype=np.float64),
        phases_init=phases_init_np,
        phases_final=phases_final_np.astype(np.float32),
        phases_true=phases_true_mod.astype(np.float32),
        final_field_error=float(field_err),
        phase_recovery_error=float(rec_err),
        wall_time_s=float(wall),
    )


# =============================================================================
# Visualization
# =============================================================================


def _save_heatmap_png(
    target_field: np.ndarray,
    recovered_field: np.ndarray,
    out_path: Path,
    title: str,
) -> Path:
    """Save a heatmap PNG of |target| vs |recovered| at the central z-slice.

    ``target_field`` and ``recovered_field`` are 3D complex arrays of equal
    shape ``(n, n, n)``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = target_field.shape[0]
    z_center = n // 2
    target_slice = np.abs(target_field[:, :, z_center])
    recovered_slice = np.abs(recovered_field[:, :, z_center])
    diff_slice = np.abs(target_field[:, :, z_center] - recovered_field[:, :, z_center])

    vmax = float(max(target_slice.max(), recovered_slice.max(), 1e-9))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, arr, label, vm in (
        (axes[0], target_slice, "Target |p|", vmax),
        (axes[1], recovered_slice, "Recovered |p|", vmax),
        (axes[2], diff_slice, "|target - recovered|", vmax),
    ):
        im = ax.imshow(arr.T, origin="lower", cmap="viridis", vmin=0.0, vmax=vm)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _save_convergence_png(
    result: JWaveRestartResult, out_path: Path, title: str
) -> Path:
    """Save a convergence-trace PNG (loss vs iter) for one target."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(result.loss_curve) + 1), result.loss_curve, "o-")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (mean|p_pred - p_target|^2)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.text(
        0.95,
        0.95,
        f"final field err: {result.final_field_error:.3e}\n"
        f"phase rec err: {result.phase_recovery_error:.3f}\n"
        f"wall: {result.wall_time_s:.1f}s",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# =============================================================================
# Driver
# =============================================================================


def _load_target_from_dataset(
    dataset_path: Path, target_idx: int
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load (phases_true, target_field_complex_3d, dataset_attrs) for one row.

    Returns
    -------
    phases_true : np.ndarray, shape (n_t,) float32
    target_field : np.ndarray, shape (n, n, n) complex64
    attrs : dict — relevant dataset attributes (n_transducers, grid_dx_m, etc.)
    """
    import h5py

    with h5py.File(dataset_path, "r") as f:
        n_rows = f["phases"].shape[0]
        if target_idx < 0 or target_idx >= n_rows:
            raise IndexError(
                f"target_idx {target_idx} out of range "
                f"(dataset has {n_rows} rows)"
            )
        phases_true = np.asarray(f["phases"][target_idx], dtype=np.float32)
        fr = np.asarray(f["field_real"][target_idx], dtype=np.float32)
        fi = np.asarray(f["field_imag"][target_idx], dtype=np.float32)
        target_field = (fr + 1j * fi).astype(np.complex64)

        # Pull relevant attrs.
        attrs = {
            "physics_backend": str(f.attrs.get("physics_backend", "unknown")),
            "n_transducers": int(f.attrs["n_transducers"]),
            "grid_dx_m": float(f.attrs["grid_dx_m"]),
            "grid_resolution": tuple(int(x) for x in f.attrs["grid_resolution"]),
            "frequency_hz": float(f.attrs["frequency_hz"]),
            "sound_speed_m_s": float(f.attrs["sound_speed_m_s"]),
            "array_radius_m": float(f.attrs.get("array_radius_m", 0.02)),
            "array_height_m": float(f.attrs.get("array_height_m", 0.02)),
            "rings": None,
            "per_ring": None,
        }
        # transducer_positions for geometry round-trip.
        attrs["transducer_positions"] = np.asarray(
            f["transducer_positions"], dtype=np.float64
        )
    return phases_true, target_field, attrs


def _build_array_from_attrs(attrs: dict):
    """Build the same ArrayGeometry the dataset was generated with."""
    from drip_physics.core import ArrayGeometry, SimulationParams

    positions = np.asarray(attrs["transducer_positions"], dtype=np.float64)
    params = SimulationParams()
    return ArrayGeometry(positions=positions, params=params)


def _pairwise_phase_l2(phases_per_restart: np.ndarray) -> Dict[str, float]:
    """Compute pairwise mean-shift-removed L2 phase distances between restarts.

    Parameters
    ----------
    phases_per_restart
        Array of shape ``(n_restarts, n_t)`` — final recovered phases for
        each restart on the same target.

    Returns
    -------
    dict with keys:
      - ``pairs``: list of ``(i, j, dist)`` triples (one per pair)
      - ``mean``: mean pairwise distance
      - ``min``: minimum pairwise distance
      - ``max``: maximum pairwise distance
    """
    n_restarts = phases_per_restart.shape[0]
    pair_distances: List[Tuple[int, int, float]] = []
    for i in range(n_restarts):
        for j in range(i + 1, n_restarts):
            d = phase_recovery_error(
                phases_per_restart[i], phases_per_restart[j]
            )
            pair_distances.append((i, j, float(d)))
    if not pair_distances:
        return {"pairs": [], "mean": 0.0, "min": 0.0, "max": 0.0}
    dists = np.asarray([d for _, _, d in pair_distances], dtype=np.float64)
    return {
        "pairs": [(int(i), int(j), float(d)) for i, j, d in pair_distances],
        "mean": float(dists.mean()),
        "min": float(dists.min()),
        "max": float(dists.max()),
    }


def run_inverse_targets(
    *,
    dataset_path: Path,
    target_indices: Sequence[int],
    output_dir: Path,
    n_iters: int = 50,
    n_restarts: int = 1,
    lr: float = 1e-2,
    seed: int = 42,
) -> List[JWaveRestartResult]:
    """Run the j-Wave inverse on multiple targets, persist receipts.

    For each target, ``n_restarts`` independent random-init restarts are
    run. The restart with the lowest final loss is selected as the
    primary result. Per-restart phases are saved alongside for
    multi-modality diagnostics.

    Restart ``r`` for target ``t`` uses seed ``seed + 10000*t + r`` so:
      * restart 0 of every target with the same base seed reproduces
        across runs (matches v1 output when ``n_restarts == 1``);
      * different targets get different RNG streams (no collision);
      * different restarts within a target get different streams.

    Side effects
    ------------
    - Per-target heatmap PNG: ``output_dir/inverse_target{idx}_field.png``
    - Per-target convergence PNG: ``output_dir/inverse_target{idx}_convergence.png``
    - Per-target NPZ: ``output_dir/inverse_target{idx}_phases.npz`` (init/final/true phases, loss curve)
      For ``n_restarts > 1``, the NPZ also contains ``phases_per_restart`` and
      ``loss_curves_per_restart``.
    - Aggregate ``output_dir/summary.json`` (with multi-modality witness).

    Returns the list of best-per-target ``JWaveRestartResult``.
    """
    import jax
    import jax.numpy as jnp

    from drip_physics.backends.jwave_backend import JWaveBackend
    from drip_physics.config import EnvironmentConfig, PressureConfig

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset attributes from %s", dataset_path)
    # Peek at row 0 to grab attrs (geometry, dx).
    _, _, attrs = _load_target_from_dataset(dataset_path, 0)

    if attrs["physics_backend"] != "jwave":
        logger.warning(
            "Dataset physics_backend=%s — expected 'jwave'. The j-Wave "
            "inverse will still run, but the target was generated with a "
            "different forward and the recovery test is degraded.",
            attrs["physics_backend"],
        )

    env = EnvironmentConfig()
    array = _build_array_from_attrs(attrs)
    grid_n = attrs["grid_resolution"][0]
    grid_dx = attrs["grid_dx_m"]

    press_cfg = PressureConfig(
        backend="jwave",
        grid_n=grid_n,
        grid_dx=grid_dx,
    )
    backend = JWaveBackend(config=press_cfg, env=env)

    logger.info(
        "Backend: jwave (grid_n=%d, dx=%.4e m, side=%.2f mm)",
        grid_n,
        grid_dx,
        grid_n * grid_dx * 1e3,
    )
    logger.info("JAX devices: %s", jax.devices())

    results: List[JWaveRestartResult] = []
    summary_records: List[Dict[str, object]] = []

    t_total0 = time.time()

    n_restarts_int = max(1, int(n_restarts))

    for k, target_idx in enumerate(target_indices):
        logger.info(
            "[target %d/%d, idx=%d] loading + running inverse "
            "(%d restart%s)",
            k + 1,
            len(target_indices),
            target_idx,
            n_restarts_int,
            "s" if n_restarts_int != 1 else "",
        )
        phases_true, target_field_np, _attrs_unused = _load_target_from_dataset(
            dataset_path, int(target_idx)
        )
        target_field_jnp = jnp.asarray(target_field_np, dtype=jnp.complex64)

        # Run all restarts.
        per_restart: List[JWaveRestartResult] = []
        # When n_restarts == 1, restart 0 uses the bare ``seed`` so the
        # output reproduces the v1 inverse_jwave receipts exactly.
        for r in range(n_restarts_int):
            if n_restarts_int == 1:
                restart_seed = int(seed)
            else:
                restart_seed = int(seed) + 10000 * int(target_idx) + int(r)
            logger.info(
                "  [target=%d restart=%d/%d] seed=%d",
                target_idx,
                r + 1,
                n_restarts_int,
                restart_seed,
            )
            r_result = solve_inverse_jwave(
                backend=backend,
                array=array,
                env=env,
                target_field_jnp=target_field_jnp,
                phases_true=phases_true,
                target_idx=int(target_idx),
                n_iters=int(n_iters),
                lr=float(lr),
                seed=restart_seed,
            )
            per_restart.append(r_result)

        # Best-of: lowest final loss.
        best_idx = int(
            np.argmin(np.asarray([rr.final_loss for rr in per_restart]))
        )
        result = per_restart[best_idx]
        results.append(result)

        # Multi-modality witness (only meaningful for n_restarts >= 2).
        phases_per_restart = np.stack(
            [rr.phases_final for rr in per_restart], axis=0
        )
        if n_restarts_int >= 2:
            mm_witness = _pairwise_phase_l2(phases_per_restart)
        else:
            mm_witness = {"pairs": [], "mean": 0.0, "min": 0.0, "max": 0.0}

        # Recovered field (fresh forward at best-restart final phases).
        amps_jnp = jnp.ones(int(array.num_transducers), dtype=jnp.float32)
        with_final = backend._solve_field_jax(
            array,
            jnp.asarray(result.phases_final, dtype=jnp.float32),
            amps_jnp,
            env,
        )
        with_final.block_until_ready()
        recovered_field = np.asarray(with_final).astype(np.complex64)

        heatmap_path = output_dir / f"inverse_target{target_idx}_field.png"
        _save_heatmap_png(
            target_field_np,
            recovered_field,
            heatmap_path,
            title=(
                f"j-Wave inverse target_idx={target_idx} "
                f"(field_err={result.final_field_error:.3e}, "
                f"phase_err={result.phase_recovery_error:.3f}, "
                f"best of {n_restarts_int})"
            ),
        )

        conv_path = output_dir / f"inverse_target{target_idx}_convergence.png"
        _save_convergence_png(
            result,
            conv_path,
            title=(
                f"Convergence — target_idx={target_idx} "
                f"(best restart {best_idx} of {n_restarts_int})"
            ),
        )

        npz_path = output_dir / f"inverse_target{target_idx}_phases.npz"
        loss_curves_per_restart = np.stack(
            [rr.loss_curve for rr in per_restart], axis=0
        )
        final_losses_per_restart = np.asarray(
            [rr.final_loss for rr in per_restart], dtype=np.float64
        )
        field_errors_per_restart = np.asarray(
            [rr.final_field_error for rr in per_restart], dtype=np.float64
        )
        phase_errors_per_restart = np.asarray(
            [rr.phase_recovery_error for rr in per_restart], dtype=np.float64
        )
        np.savez(
            npz_path,
            # Best-restart fields (back-compat with v1 NPZ).
            phases_init=result.phases_init,
            phases_final=result.phases_final,
            phases_true=result.phases_true,
            loss_curve=result.loss_curve,
            iter_wall_s=result.iter_wall_s,
            grad_finite=result.grad_finite,
            # Multi-restart additions.
            phases_per_restart=phases_per_restart,
            loss_curves_per_restart=loss_curves_per_restart,
            final_losses_per_restart=final_losses_per_restart,
            field_errors_per_restart=field_errors_per_restart,
            phase_errors_per_restart=phase_errors_per_restart,
            best_restart_idx=np.int64(best_idx),
        )

        rec = result.to_summary_dict()
        rec["heatmap_png"] = str(heatmap_path)
        rec["convergence_png"] = str(conv_path)
        rec["phases_npz"] = str(npz_path)
        rec["n_restarts"] = int(n_restarts_int)
        rec["best_restart_idx"] = int(best_idx)
        rec["per_restart_final_loss"] = [float(x) for x in final_losses_per_restart]
        rec["per_restart_field_error"] = [float(x) for x in field_errors_per_restart]
        rec["per_restart_phase_error"] = [float(x) for x in phase_errors_per_restart]
        rec["per_restart_wall_time_s"] = [float(rr.wall_time_s) for rr in per_restart]
        rec["multimodality_pairwise_phase_l2"] = mm_witness
        summary_records.append(rec)

        logger.info(
            "  --> target=%d  best_restart=%d  final_loss=%.4e  "
            "field_err=%.3e  phase_err=%.3f  wall(best)=%.1fs  "
            "pairwise_phase_l2_mean=%.3f",
            target_idx,
            best_idx,
            result.final_loss,
            result.final_field_error,
            result.phase_recovery_error,
            result.wall_time_s,
            mm_witness["mean"],
        )

        # Incremental summary write — partial run still produces an artifact.
        partial_summary = {
            "dataset_path": str(dataset_path),
            "n_iters": int(n_iters),
            "n_restarts": int(n_restarts_int),
            "lr": float(lr),
            "seed": int(seed),
            "n_targets_done": int(k + 1),
            "n_targets_total": int(len(target_indices)),
            "target_indices": [int(t) for t in target_indices],
            "results": summary_records,
            "wall_time_s_so_far": float(time.time() - t_total0),
        }
        with open(output_dir / "summary.json", "w") as fh:
            json.dump(partial_summary, fh, indent=2)

    elapsed_total = time.time() - t_total0
    logger.info(
        "All %d targets done in %.1f s (%.1f min)",
        len(target_indices),
        elapsed_total,
        elapsed_total / 60.0,
    )

    # Final summary with completion flag.
    final_summary = {
        "dataset_path": str(dataset_path),
        "n_iters": int(n_iters),
        "n_restarts": int(n_restarts_int),
        "lr": float(lr),
        "seed": int(seed),
        "n_targets_done": int(len(results)),
        "n_targets_total": int(len(target_indices)),
        "target_indices": [int(t) for t in target_indices],
        "results": summary_records,
        "wall_time_s_total": float(elapsed_total),
        "wall_time_min_total": float(elapsed_total / 60.0),
    }
    with open(output_dir / "summary.json", "w") as fh:
        json.dump(final_summary, fh, indent=2)

    return results


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=str,
        default="ml_inverse/data/inverse_dataset_jwave_3d.h5",
        help="Path to the j-Wave HDF5 dataset.",
    )
    p.add_argument(
        "--target-idx",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="One or more target indices into the dataset.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="ml_inverse/research/_jwave_inverse",
        help="Output directory for receipts and PNGs.",
    )
    p.add_argument("--n-iters", type=int, default=50)
    p.add_argument(
        "--n-restarts",
        type=int,
        default=1,
        help=(
            "Number of independent random-init restarts per target. "
            "Best (lowest final loss) is reported. Default 1 reproduces "
            "the v1 receipts."
        ),
    )
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    repo_sim = Path(__file__).resolve().parent.parent
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (repo_sim / args.dataset).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_sim / args.output_dir).resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Run "
            f"generate_jwave_dataset first."
        )

    run_inverse_targets(
        dataset_path=dataset_path,
        target_indices=args.target_idx,
        output_dir=output_dir,
        n_iters=int(args.n_iters),
        n_restarts=int(args.n_restarts),
        lr=float(args.lr),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
