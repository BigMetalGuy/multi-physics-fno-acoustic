"""Differentiable inverse solver — gradient descent through the frozen FNO.

Phase 3 productionization of the FNO-inverse risk spike
(``research/_spike_fno_inverse.py``, verdict GREEN at 10.9 ms/iter MPS).

Given a target pressure field on the 2D 64x64 slice and the per-sample
``slice_z``, recover a 120-dim phase configuration that produces (an
approximation of) that field by running Adam on the phase vector while
the FNO weights are frozen. The forward model is treated as a black-box
differentiable map; the inverse loop is invariant to which forward model
is plugged in (the architectural bet behind Phase 3).

Multi-restart is mandatory by default — the acoustic-inverse landscape
is structurally multi-modal: many phase configurations produce the same
field. Random-init restarts characterize that multi-modality, and the
best restart is selected by **field error** (not phase error).

Public surface
--------------
* ``InverseSolver`` — the gradient-descent inverse, configurable
  iter / lr / restarts / device / seed. Defaults from the spike.
* ``InverseSolution`` — named tuple with the receipts trail for one
  ``solve()`` call (best phases, loss curve, all restart results).
* ``RestartResult`` — per-restart receipt.
* ``make_controller_from_inverse`` — Phase 4 stub: wraps an
  ``InverseSolver`` into a ``predict(input) -> output`` callable that
  ``block_compiler.register_ml_model`` accepts.

CLI
---
``python -m ml_inverse.inverse --target-source v3 --target-idx 42``
loads FNO v2.1, picks test sample idx=42, runs the inverse, and persists
receipts (per-restart loss curves, recovered field PNG, target/recovered
side-by-side PNG, convergence summary JSON).

Design notes
------------
1. **Forward model frozen.** All FNO weights are set to
   ``requires_grad_(False)`` at construction. An assertion confirms zero
   trainable parameters in the model itself.
2. **Field error is the convergence metric.** Phase error is reported
   for diagnostic value (multi-modality witness) but never used to
   select between restarts.
3. **Reproducibility.** ``base_seed + restart_idx`` per restart. Both
   numpy and torch RNGs seeded.
4. **Device-agnostic.** MPS default; CPU and CUDA supported via the
   ``device`` arg.
5. **Receipts.** Every solve produces a JSON-serializable convergence
   summary. CLI writes it to disk as part of the audit trail.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ml_inverse.adapter import ChannelNormalizer
from ml_inverse.model import PhaseConditionedFNO


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RestartResult:
    """Per-restart receipt.

    All tensors live on CPU as float32 numpy arrays for serializability.
    """

    restart_idx: int
    seed: int
    init_kind: str  # 'random' | 'warm_start'
    n_iters: int
    final_loss: float
    final_field_error: float
    phases: np.ndarray  # (n_transducers,)
    loss_curve: np.ndarray  # (n_iters,)
    converged: bool
    convergence_iter: Optional[int]
    wall_time_s: float

    def to_dict(self) -> dict:
        """JSON-serializable summary; drops the loss_curve and phases arrays."""
        d = {
            "restart_idx": int(self.restart_idx),
            "seed": int(self.seed),
            "init_kind": str(self.init_kind),
            "n_iters": int(self.n_iters),
            "final_loss": float(self.final_loss),
            "final_field_error": float(self.final_field_error),
            "converged": bool(self.converged),
            "convergence_iter": (
                int(self.convergence_iter)
                if self.convergence_iter is not None
                else None
            ),
            "wall_time_s": float(self.wall_time_s),
        }
        return d


class InverseSolution(NamedTuple):
    """Returned by ``InverseSolver.solve``.

    Attributes
    ----------
    phases : np.ndarray (n_transducers,)
        Best phase configuration across restarts (selected by field error).
    final_loss : float
        Best restart's final normalized MSE loss.
    final_field_error : float
        Best restart's ``||model(phases) - target|| / ||target||``.
    loss_curve : np.ndarray (n_iters,)
        Best restart's per-iter loss trace.
    convergence_iter : Optional[int]
        Iter at which the plateau test first fired for the best restart;
        ``None`` if the run hit ``n_iters`` without plateau.
    all_restart_results : List[RestartResult]
        Per-restart receipt — full receipts trail for the multi-modality
        diagnostic.
    """

    phases: np.ndarray
    final_loss: float
    final_field_error: float
    loss_curve: np.ndarray
    convergence_iter: Optional[int]
    all_restart_results: List[RestartResult]


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    """Resolve a string device label or torch.device, with MPS fallback warn."""
    if isinstance(device, torch.device):
        return device
    label = str(device).lower()
    if label == "mps":
        if not torch.backends.mps.is_available():
            logger.warning("device='mps' requested but MPS unavailable; falling back to cpu")
            return torch.device("cpu")
        return torch.device("mps")
    if label == "cuda":
        if not torch.cuda.is_available():
            logger.warning("device='cuda' requested but CUDA unavailable; falling back to cpu")
            return torch.device("cpu")
        return torch.device("cuda")
    if label == "cpu":
        return torch.device("cpu")
    return torch.device(label)


def _seed_all(seed: int) -> None:
    """Seed numpy + torch (CPU/MPS/CUDA) for reproducibility within a restart."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    # MPS reads from torch.manual_seed; no separate API.


class InverseSolver:
    """Gradient-descent inverse through a frozen forward model.

    The forward model must be a callable ``model(x) -> field`` where
    ``x`` is a packed ``(B, n_transducers + 1)`` tensor with the
    per-sample ``slice_z`` (m) in the last column (the v2.1 contract).
    All forward-model parameters are frozen at construction; only the
    phase vector is optimized.

    Parameters
    ----------
    forward_model
        A trained ``PhaseConditionedFNO`` (or any module with the same
        signature). Its parameters are frozen in-place.
    normalizer
        ``ChannelNormalizer`` used at training time. Required so the CLI
        can normalize raw GT fields when constructing inverse targets;
        the solver itself operates entirely in normalized space (the
        forward model returns normalized outputs). Provided for
        bookkeeping and so callers don't have to plumb it separately.
    n_iters
        Adam iterations per restart. Spike used 500-1000; default 500.
    lr
        Adam learning rate. Spike validated lr=1e-1 as the best of the
        sweep.
    n_restarts
        Number of independent random-init restarts. The best (by field
        error) is returned, but ALL restart receipts are persisted.
    device
        ``'mps'`` / ``'cpu'`` / ``'cuda'`` — resolved at construction.
    seed
        Base seed; restart ``i`` uses ``seed + i``.
    plateau_window
        Iters of trailing window used for the plateau-detection test.
    plateau_rel_tol
        Relative-drop threshold below which the plateau test fires and
        terminates the restart early.
    """

    def __init__(
        self,
        forward_model: torch.nn.Module,
        normalizer: ChannelNormalizer,
        *,
        n_iters: int = 500,
        lr: float = 1e-1,
        n_restarts: int = 3,
        device: Union[str, torch.device] = "mps",
        seed: int = 42,
        plateau_window: int = 25,
        plateau_rel_tol: float = 1e-5,
    ) -> None:
        if int(n_iters) < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")
        if int(n_restarts) < 1:
            raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")
        if float(lr) <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")

        self._device: torch.device = _resolve_device(device)
        self._model: torch.nn.Module = forward_model.to(self._device).eval()
        self._normalizer: ChannelNormalizer = normalizer
        self._n_iters: int = int(n_iters)
        self._lr: float = float(lr)
        self._n_restarts: int = int(n_restarts)
        self._seed: int = int(seed)
        self._plateau_window: int = int(plateau_window)
        self._plateau_rel_tol: float = float(plateau_rel_tol)

        # Freeze forward-model parameters in-place.
        for p in self._model.parameters():
            p.requires_grad_(False)
        n_trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        if n_trainable != 0:
            raise AssertionError(
                f"forward_model has {n_trainable} trainable parameters after "
                f"freeze; expected 0. Inverse solver requires a frozen forward model."
            )

        # Discover the model's transducer count via the publicly-exposed attr.
        n_tx = getattr(self._model, "n_transducers", None)
        if n_tx is None:
            raise AttributeError(
                "forward_model is missing 'n_transducers' attribute; "
                "cannot infer phase-vector dimensionality."
            )
        self._n_transducers: int = int(n_tx)

    # ----- properties --------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def n_transducers(self) -> int:
        return self._n_transducers

    @property
    def n_iters(self) -> int:
        return self._n_iters

    @property
    def lr(self) -> float:
        return self._lr

    @property
    def n_restarts(self) -> int:
        return self._n_restarts

    @property
    def normalizer(self) -> ChannelNormalizer:
        return self._normalizer

    # ----- core solve --------------------------------------------------------

    def _run_one_restart(
        self,
        target_field_norm: torch.Tensor,
        slice_z_val: float,
        init_phases: torch.Tensor,
        restart_idx: int,
        restart_seed: int,
        init_kind: str,
    ) -> RestartResult:
        """Single Adam run on a fixed init.

        ``target_field_norm`` shape: ``(1, 2, Nx, Ny)``, on device.
        ``init_phases`` shape: ``(n_transducers,)``, on device.
        """
        _seed_all(restart_seed)

        phases = init_phases.detach().clone().to(self._device)
        phases.requires_grad_(True)

        slice_z_const = torch.tensor(
            [slice_z_val], device=self._device, dtype=torch.float32
        )

        optimizer = torch.optim.Adam([phases], lr=self._lr)

        losses: List[float] = []
        convergence_iter: Optional[int] = None
        t0 = time.perf_counter()

        for it in range(self._n_iters):
            optimizer.zero_grad()
            x = torch.cat(
                [phases.unsqueeze(0), slice_z_const.unsqueeze(0)], dim=-1
            )  # (1, n_tx + 1)
            y_pred = self._model(x)
            loss = torch.nn.functional.mse_loss(y_pred, target_field_norm)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

            # Plateau test: relative drop over trailing window.
            if (
                convergence_iter is None
                and it >= 2 * self._plateau_window
            ):
                recent = np.asarray(losses[-self._plateau_window:])
                older = np.asarray(
                    losses[-2 * self._plateau_window:-self._plateau_window]
                )
                older_mean = float(older.mean())
                if older_mean > 0.0:
                    rel_drop = (older_mean - float(recent.mean())) / older_mean
                    if rel_drop < self._plateau_rel_tol:
                        convergence_iter = it + 1
                        # Don't break — let the run finish to a clean budget so
                        # all restarts spend the same compute. This matches the
                        # spike's "extended runs at lr=1e-1 1000 iters" behavior.
                        # If we wanted hard early-stop, swap `pass` for `break`.
                        # For now: log and continue.
                        pass

        if self._device.type == "mps":
            torch.mps.synchronize()
        elif self._device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        # Final field error in normalized space.
        with torch.no_grad():
            x_final = torch.cat(
                [phases.unsqueeze(0), slice_z_const.unsqueeze(0)], dim=-1
            )
            y_final = self._model(x_final)
            field_err = (
                (y_final - target_field_norm).norm()
                / target_field_norm.norm().clamp_min(1e-12)
            ).item()

        return RestartResult(
            restart_idx=int(restart_idx),
            seed=int(restart_seed),
            init_kind=str(init_kind),
            n_iters=len(losses),
            final_loss=float(losses[-1]) if losses else float("nan"),
            final_field_error=float(field_err),
            phases=phases.detach().cpu().numpy().astype(np.float32),
            loss_curve=np.asarray(losses, dtype=np.float32),
            converged=bool(convergence_iter is not None),
            convergence_iter=convergence_iter,
            wall_time_s=float(wall),
        )

    def solve(
        self,
        target_field: torch.Tensor,
        slice_z: float,
        init_phases: Optional[torch.Tensor] = None,
    ) -> InverseSolution:
        """Recover phases for a single target field.

        Parameters
        ----------
        target_field : torch.Tensor
            Target field in **normalized** space, shape ``(2, Nx, Ny)`` or
            ``(1, 2, Nx, Ny)``. Caller is responsible for normalization;
            use ``solver.normalizer.normalize(raw_pa)`` first if working
            from raw Pa GT. (We keep raw vs normalized explicit at the
            module boundary; the forward model and loss live in
            normalized space.)
        slice_z : float
            Per-sample slice z-plane in meters.
        init_phases : Optional[torch.Tensor]
            ``(n_transducers,)`` initial phases for the warm-start
            restart. If provided, restart 0 uses this; remaining
            restarts are random. If ``None``, all restarts are random.

        Returns
        -------
        InverseSolution
            See class docstring. ``phases`` is the best restart's final
            configuration (selected by field error).
        """
        # ---------- normalize target shape ----------
        if target_field.ndim == 3:
            target = target_field.unsqueeze(0)
        elif target_field.ndim == 4:
            target = target_field
        else:
            raise ValueError(
                f"target_field must be (2,Nx,Ny) or (1,2,Nx,Ny); "
                f"got shape {tuple(target_field.shape)}"
            )
        target = target.to(self._device, dtype=torch.float32)

        if init_phases is not None:
            if init_phases.ndim != 1 or init_phases.shape[0] != self._n_transducers:
                raise ValueError(
                    f"init_phases must be ({self._n_transducers},); "
                    f"got shape {tuple(init_phases.shape)}"
                )

        # ---------- run restarts ----------
        restart_results: List[RestartResult] = []
        for r in range(self._n_restarts):
            restart_seed = self._seed + r
            if r == 0 and init_phases is not None:
                init = init_phases.detach().clone().to(
                    self._device, dtype=torch.float32
                )
                init_kind = "warm_start"
            else:
                _seed_all(restart_seed)
                init = (
                    torch.rand(self._n_transducers, device=self._device)
                    * 2.0
                    * float(np.pi)
                ).to(torch.float32)
                init_kind = "random"

            result = self._run_one_restart(
                target_field_norm=target,
                slice_z_val=float(slice_z),
                init_phases=init,
                restart_idx=r,
                restart_seed=restart_seed,
                init_kind=init_kind,
            )
            restart_results.append(result)
            logger.info(
                "restart %d/%d (%s, seed=%d): final_loss=%.4e field_err=%.4e %s",
                r + 1,
                self._n_restarts,
                init_kind,
                restart_seed,
                result.final_loss,
                result.final_field_error,
                ("(plateau @ %d)" % result.convergence_iter)
                if result.converged
                else "",
            )

        # ---------- pick best restart by FIELD error ----------
        best_idx = int(
            np.argmin([r.final_field_error for r in restart_results])
        )
        best = restart_results[best_idx]

        return InverseSolution(
            phases=best.phases,
            final_loss=best.final_loss,
            final_field_error=best.final_field_error,
            loss_curve=best.loss_curve,
            convergence_iter=best.convergence_iter,
            all_restart_results=restart_results,
        )

    def solve_batch(
        self,
        target_fields: torch.Tensor,
        slice_zs: Sequence[float],
        init_phases: Optional[torch.Tensor] = None,
    ) -> List[InverseSolution]:
        """Sequential per-target solve. Returns a list of solutions.

        A truly batched implementation would share a single Adam state
        across the batch (parallel optimizer, ``n_targets * n_tx``
        trainable params). Sequential is correct and simple; the batched
        variant is an optimization for Phase 4 if controller throughput
        becomes a bottleneck.
        """
        if target_fields.ndim != 4:
            raise ValueError(
                f"target_fields must be (B, 2, Nx, Ny); got "
                f"{tuple(target_fields.shape)}"
            )
        if target_fields.shape[0] != len(slice_zs):
            raise ValueError(
                f"batch size {target_fields.shape[0]} != len(slice_zs) "
                f"{len(slice_zs)}"
            )
        if init_phases is not None and init_phases.shape[0] != target_fields.shape[0]:
            raise ValueError(
                f"init_phases batch size {init_phases.shape[0]} != "
                f"target_fields batch size {target_fields.shape[0]}"
            )

        out: List[InverseSolution] = []
        for i in range(target_fields.shape[0]):
            init_i = (
                init_phases[i] if init_phases is not None else None
            )
            out.append(
                self.solve(
                    target_fields[i],
                    slice_z=float(slice_zs[i]),
                    init_phases=init_i,
                )
            )
        return out


# ---------------------------------------------------------------------------
# Multi-modality diagnostic
# ---------------------------------------------------------------------------


def phase_recovery_error(
    phases_pred: np.ndarray, phases_true: np.ndarray
) -> float:
    """Mean-shift-removed L2 phase error.

    Phases are equivalent up to a global constant; subtract the mean
    difference before computing relative L2.
    """
    diff = phases_pred - phases_true
    diff_centered = diff - diff.mean()
    num = float(np.linalg.norm(diff_centered))
    den = float(np.linalg.norm(phases_true - phases_true.mean()))
    return num / max(den, 1e-12)


def multi_modality_witness(
    restart_results: Sequence[RestartResult],
    phases_true: Optional[np.ndarray] = None,
) -> dict:
    """Summarize the multi-modality property across restarts.

    For each pair (i, j), compute the mean-shift-removed L2 between
    their phase solutions. High pairwise phase variance + low field
    error variance is the witness signature: equivalent fields, distinct
    phase configs.

    If ``phases_true`` is given, also reports per-restart phase error
    against the GT phases.
    """
    n = len(restart_results)
    field_errs = np.array([r.final_field_error for r in restart_results])

    # Pairwise phase distance (mean-shift-removed).
    pairwise = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise.append(
                phase_recovery_error(
                    restart_results[i].phases, restart_results[j].phases
                )
            )
    pairwise_arr = np.asarray(pairwise) if pairwise else np.array([0.0])

    summary = {
        "n_restarts": int(n),
        "field_error_mean": float(field_errs.mean()),
        "field_error_std": float(field_errs.std()),
        "field_error_min": float(field_errs.min()),
        "field_error_max": float(field_errs.max()),
        "pairwise_phase_l2_mean": float(pairwise_arr.mean()),
        "pairwise_phase_l2_std": float(pairwise_arr.std()),
        "pairwise_phase_l2_max": float(pairwise_arr.max()),
    }
    if phases_true is not None:
        per_restart = [
            phase_recovery_error(r.phases, phases_true)
            for r in restart_results
        ]
        summary["phase_err_vs_true_mean"] = float(np.mean(per_restart))
        summary["phase_err_vs_true_std"] = float(np.std(per_restart))
    return summary


# ---------------------------------------------------------------------------
# Phase 4 stub: ControllerFn adapter
# ---------------------------------------------------------------------------


class InverseControllerAdapter:
    """Wrap an ``InverseSolver`` to satisfy ``register_ml_model``'s contract.

    ``block_compiler.register_ml_model`` requires ``predict(input) ->
    output`` with input shape ``(2,)`` and output shape ``(2,)``.

    For Phase 3 this adapter is a *stub* — the real Phase 4 wiring needs
    to map a 2D error signal (``[error_x, error_y]``) into a target
    pressure field (e.g. via an analytical focal model that places a
    focus at the desired offset), then run ``solver.solve`` to recover
    phases, then collapse those 120 phases down to the 2D ``(dx, dy)``
    correction the block diagram expects.

    This stub records the input, runs a dummy forward through the
    solver if a target-field hook is supplied, and returns the input
    unchanged. The signature is what matters here — it compiles and
    passes ``register_ml_model``'s smoke test.
    """

    def __init__(
        self,
        solver: InverseSolver,
        target_field_fn: Optional[
            Callable[[np.ndarray], Tuple[torch.Tensor, float]]
        ] = None,
        phases_to_correction_fn: Optional[
            Callable[[np.ndarray], np.ndarray]
        ] = None,
    ) -> None:
        self._solver = solver
        self._target_field_fn = target_field_fn
        self._phases_to_correction_fn = phases_to_correction_fn
        self._last_solution: Optional[InverseSolution] = None

    def predict(self, input_signal: np.ndarray) -> np.ndarray:
        """Map a 2D signal to a 2D correction.

        See class docstring for the Phase 4 plan. Stub returns the
        identity for now so the smoke test in ``register_ml_model``
        passes. Phase 4 will replace the body.
        """
        arr = np.asarray(input_signal, dtype=np.float64)
        if arr.shape != (2,):
            raise ValueError(
                f"InverseControllerAdapter.predict expects shape (2,), got {arr.shape}"
            )

        if self._target_field_fn is not None:
            target_field, slice_z = self._target_field_fn(arr)
            self._last_solution = self._solver.solve(
                target_field=target_field, slice_z=float(slice_z)
            )
            if self._phases_to_correction_fn is not None:
                out = self._phases_to_correction_fn(self._last_solution.phases)
                out_arr = np.asarray(out, dtype=np.float64).reshape(-1)
                if out_arr.shape != (2,):
                    raise ValueError(
                        f"phases_to_correction_fn must return shape (2,); "
                        f"got {out_arr.shape}"
                    )
                return out_arr
        return arr.copy()

    @property
    def last_solution(self) -> Optional[InverseSolution]:
        return self._last_solution


def make_controller_from_inverse(
    solver: InverseSolver,
    *,
    target_field_fn: Optional[
        Callable[[np.ndarray], Tuple[torch.Tensor, float]]
    ] = None,
    phases_to_correction_fn: Optional[
        Callable[[np.ndarray], np.ndarray]
    ] = None,
) -> InverseControllerAdapter:
    """Factory: wraps ``solver`` for ``block_compiler.register_ml_model``.

    Returns an object with a ``predict(input: np.ndarray) -> np.ndarray``
    method, satisfying the ML-block protocol.

    Phase 4 will provide concrete ``target_field_fn`` (error -> target
    pressure field) and ``phases_to_correction_fn`` (phases -> position
    offset) implementations.
    """
    return InverseControllerAdapter(
        solver=solver,
        target_field_fn=target_field_fn,
        phases_to_correction_fn=phases_to_correction_fn,
    )


# ---------------------------------------------------------------------------
# CLI helpers — checkpoint loading + visualization
# ---------------------------------------------------------------------------


def load_v21_forward_model(
    checkpoint_path: Union[str, Path],
    normalizer_path: Union[str, Path],
    device: Union[str, torch.device] = "mps",
) -> Tuple[PhaseConditionedFNO, ChannelNormalizer]:
    """Reconstruct ``PhaseConditionedFNO`` v2.1 from checkpoint + normalizer.

    Mirrors the spike's ``build_v21_model_from_checkpoint`` so the inverse
    module is self-contained for CLI use.
    """
    ckpt = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    cfg = ckpt["model_config"]
    norm = ChannelNormalizer.load(str(normalizer_path))

    kwargs = {
        "field_dims": cfg["field_dims"],
        "grid_shape": tuple(cfg["grid_shape"]),
        "n_transducers": cfg["n_transducers"],
        "n_modes": tuple(cfg["n_modes"]),
        "hidden_channels": cfg["hidden_channels"],
        "cond_channels": cfg["cond_channels"],
        "out_channels": cfg["out_channels"],
        "transducer_positions": np.asarray(
            cfg["transducer_positions"], dtype=np.float32
        ),
        "grid_coords": cfg["grid_coords"],
        "wavenumber": cfg["wavenumber"],
        "slice_z": cfg.get("slice_z", 0.09),
        "prior_version": cfg["prior_version"],
        "residual_prior": cfg["residual_prior"],
        "prior_norm_mean": np.asarray(cfg["prior_norm_mean"], dtype=np.float32),
        "prior_norm_std": np.asarray(cfg["prior_norm_std"], dtype=np.float32),
    }

    model = PhaseConditionedFNO(**kwargs)
    sd = {
        k: v
        for k, v in ckpt["model_state_dict"].items()
        if torch.is_tensor(v)
    }
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(
            "load_v21_forward_model: %d missing keys; sample=%s",
            len(missing),
            missing[:3],
        )
    if unexpected:
        logger.warning(
            "load_v21_forward_model: %d unexpected keys; sample=%s",
            len(unexpected),
            unexpected[:3],
        )
    return model.to(_resolve_device(device)).eval(), norm


def render_field_pair_png(
    target_field_norm: np.ndarray,
    recovered_field_norm: np.ndarray,
    out_path: Union[str, Path],
    title: str = "Target vs Recovered (normalized)",
) -> Path:
    """Side-by-side 2x2 panel: (target Re, target Im, recovered Re, recovered Im).

    Inputs shape ``(2, Nx, Ny)``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    pairs = [
        ("Target Re", target_field_norm[0]),
        ("Target Im", target_field_norm[1]),
        ("Recovered Re", recovered_field_norm[0]),
        ("Recovered Im", recovered_field_norm[1]),
    ]
    # Shared scale per channel (target sets the scale).
    re_scale = float(np.abs(target_field_norm[0]).max())
    im_scale = float(np.abs(target_field_norm[1]).max())
    scales = [re_scale, im_scale, re_scale, im_scale]

    for ax, (label, arr), v in zip(axes.flatten(), pairs, scales):
        v_safe = max(v, 1e-9)
        im = ax.imshow(
            arr.T, origin="lower", cmap="RdBu_r", vmin=-v_safe, vmax=v_safe
        )
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def render_recovered_field_png(
    recovered_field_norm: np.ndarray,
    out_path: Union[str, Path],
    title: str = "Recovered field (normalized)",
) -> Path:
    """Single-row Re/Im PNG of the recovered field. Shape ``(2, Nx, Ny)``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, ch_idx, label in zip(axes, (0, 1), ("Re", "Im")):
        v = max(float(np.abs(recovered_field_norm[ch_idx]).max()), 1e-9)
        im = ax.imshow(
            recovered_field_norm[ch_idx].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-v,
            vmax=v,
        )
        ax.set_title(f"{label} (normalized)")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def render_loss_curves_png(
    restart_results: Sequence[RestartResult],
    out_path: Union[str, Path],
    title: str = "Per-restart loss curves",
) -> Path:
    """Plot all restart loss curves together (log y)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in restart_results:
        label = (
            f"restart {r.restart_idx} ({r.init_kind}, "
            f"field_err={r.final_field_error:.3e})"
        )
        ax.plot(np.arange(1, len(r.loss_curve) + 1), r.loss_curve, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized MSE loss")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the differentiable inverse solver on a single target. "
            "Loads FNO v2.1, picks a target from the v3 dataset, runs "
            "multi-restart Adam, persists receipts."
        )
    )
    parser.add_argument(
        "--target-source",
        choices=("v3",),
        default="v3",
        help="Where to load the target from. Currently only v3 supported.",
    )
    parser.add_argument(
        "--target-idx",
        type=int,
        default=42,
        help="Index into the chosen test split.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Which split to draw the target from.",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=500,
        help="Adam iterations per restart.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-1,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=3,
        help="Number of independent restarts.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Torch device: mps / cpu / cuda.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed; restart i uses seed + i.",
    )
    parser.add_argument(
        "--output-path",
        default="ml_inverse/inverse_results",
        help="Output directory for receipts (relative to simulations/).",
    )
    parser.add_argument(
        "--checkpoint",
        default="ml_inverse/models/fno_surrogate_v2_1.pt",
        help="Path to FNO checkpoint.",
    )
    parser.add_argument(
        "--normalizer",
        default="ml_inverse/models/fno_surrogate_v2_1_norm.npz",
        help="Path to ChannelNormalizer NPZ.",
    )
    parser.add_argument(
        "--data",
        default="ml_inverse/data/inverse_dataset_v3.h5",
        help="Path to v3 HDF5 dataset.",
    )
    parser.add_argument(
        "--use-model-target",
        action="store_true",
        default=True,
        help=(
            "Define the inverse target as model(phases_true) rather than the "
            "raw GT field. Default True (matches the spike). Disable with "
            "--use-gt-target to treat the GT field as the inverse target."
        ),
    )
    parser.add_argument(
        "--use-gt-target",
        dest="use_model_target",
        action="store_false",
        help="Use the raw GT field (normalized) as the inverse target.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-restart info logging.",
    )
    return parser


def _cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve paths relative to simulations/ (the parent of ml_inverse).
    sim_dir = Path(__file__).resolve().parents[1]
    ckpt_path = (sim_dir / args.checkpoint).resolve()
    norm_path = (sim_dir / args.normalizer).resolve()
    data_path = (sim_dir / args.data).resolve()
    out_dir = (sim_dir / args.output_path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("device         : %s", args.device)
    logger.info("checkpoint     : %s", ckpt_path)
    logger.info("normalizer     : %s", norm_path)
    logger.info("data           : %s", data_path)
    logger.info("output         : %s", out_dir)
    logger.info("split / idx    : %s / %d", args.split, args.target_idx)
    logger.info(
        "iters / lr / restarts: %d / %g / %d",
        args.n_iters,
        args.lr,
        args.n_restarts,
    )

    # ---------- load model ----------
    model, norm = load_v21_forward_model(
        ckpt_path, norm_path, device=args.device
    )
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("model params: %d", n_total)

    # ---------- load target sample (lazy import to avoid h5py on import) ----------
    from ml_inverse.dataset import InverseSurrogateDataset

    ds = InverseSurrogateDataset(
        str(data_path), split=args.split, seed=args.seed
    )
    if args.target_idx < 0 or args.target_idx >= len(ds):
        raise IndexError(
            f"target_idx {args.target_idx} out of range for split "
            f"'{args.split}' (n={len(ds)})"
        )
    sample = ds[int(args.target_idx)]
    phases_true = sample["phases"].numpy().astype(np.float32)
    slice_z_val = float(sample["slice_z"].item())
    logger.info(
        "sample slice_z = %.2f mm (%.6f m)", slice_z_val * 1000.0, slice_z_val
    )

    # ---------- construct target field (normalized) ----------
    device = _resolve_device(args.device)
    if args.use_model_target:
        with torch.no_grad():
            x_true = torch.cat(
                [
                    torch.from_numpy(phases_true).unsqueeze(0).to(device),
                    torch.tensor([[slice_z_val]], device=device),
                ],
                dim=-1,
            )
            target_field_norm = model(x_true)[0].detach().cpu()
        target_kind = "model(phases_true)"
    else:
        gt_field = sample["field"]  # (2, Nx, Ny) raw Pa
        target_field_norm = norm.normalize(gt_field).cpu()
        target_kind = "normalize(GT)"

    logger.info(
        "target field [%s]: shape=%s mean=%+.3e std=%.3e max|.|=%.3e",
        target_kind,
        tuple(target_field_norm.shape),
        target_field_norm.mean().item(),
        target_field_norm.std().item(),
        target_field_norm.abs().max().item(),
    )

    # ---------- run inverse ----------
    solver = InverseSolver(
        forward_model=model,
        normalizer=norm,
        n_iters=int(args.n_iters),
        lr=float(args.lr),
        n_restarts=int(args.n_restarts),
        device=args.device,
        seed=int(args.seed),
    )
    t0 = time.perf_counter()
    solution = solver.solve(target_field=target_field_norm, slice_z=slice_z_val)
    wall = time.perf_counter() - t0

    # ---------- recovered field for visualization ----------
    with torch.no_grad():
        x_best = torch.cat(
            [
                torch.from_numpy(solution.phases).unsqueeze(0).to(device),
                torch.tensor([[slice_z_val]], device=device),
            ],
            dim=-1,
        )
        recovered_field_norm = model(x_best)[0].detach().cpu().numpy()

    # ---------- multi-modality witness ----------
    witness = multi_modality_witness(
        solution.all_restart_results, phases_true=phases_true
    )

    # ---------- persist receipts ----------
    tag = f"target{args.target_idx}"
    summary_path = out_dir / f"inverse_{tag}_summary.json"
    phases_path = out_dir / f"inverse_{tag}_phases.npz"
    recovered_png = out_dir / f"inverse_{tag}_recovered.png"
    pair_png = out_dir / f"inverse_{tag}_target_vs_recovered.png"
    curves_png = out_dir / f"inverse_{tag}_loss_curves.png"

    summary = {
        "target_idx": int(args.target_idx),
        "split": str(args.split),
        "target_kind": target_kind,
        "slice_z_m": float(slice_z_val),
        "config": {
            "n_iters": int(args.n_iters),
            "lr": float(args.lr),
            "n_restarts": int(args.n_restarts),
            "device": str(args.device),
            "seed": int(args.seed),
        },
        "best": {
            "restart_idx": int(
                np.argmin(
                    [
                        r.final_field_error
                        for r in solution.all_restart_results
                    ]
                )
            ),
            "final_loss": float(solution.final_loss),
            "final_field_error": float(solution.final_field_error),
            "convergence_iter": (
                int(solution.convergence_iter)
                if solution.convergence_iter is not None
                else None
            ),
        },
        "restarts": [r.to_dict() for r in solution.all_restart_results],
        "multi_modality_witness": witness,
        "wall_time_s": float(wall),
        "wall_time_per_restart_s": float(
            wall / max(int(args.n_restarts), 1)
        ),
    }

    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("wrote %s", summary_path)

    np.savez(
        phases_path,
        phases_best=solution.phases,
        phases_true=phases_true,
        loss_curve_best=solution.loss_curve,
        all_phases=np.stack(
            [r.phases for r in solution.all_restart_results]
        ),
        all_loss_curves=np.stack(
            [r.loss_curve for r in solution.all_restart_results]
        ),
    )
    logger.info("wrote %s", phases_path)

    render_recovered_field_png(
        recovered_field_norm,
        recovered_png,
        title=(
            f"Recovered field (target_idx={args.target_idx}, "
            f"field_err={solution.final_field_error:.3e})"
        ),
    )
    logger.info("wrote %s", recovered_png)

    render_field_pair_png(
        target_field_norm.numpy(),
        recovered_field_norm,
        pair_png,
        title=(
            f"Target vs Recovered (idx={args.target_idx}, "
            f"field_err={solution.final_field_error:.3e})"
        ),
    )
    logger.info("wrote %s", pair_png)

    render_loss_curves_png(
        solution.all_restart_results,
        curves_png,
        title=f"Per-restart loss curves (target_idx={args.target_idx})",
    )
    logger.info("wrote %s", curves_png)

    # ---------- console summary ----------
    print()
    print(f"=== Inverse summary — target_idx={args.target_idx} ===")
    print(f"target kind        : {target_kind}")
    print(f"slice_z            : {slice_z_val * 1000.0:.2f} mm")
    print(f"wall time          : {wall:.2f} s ({args.n_restarts} restarts)")
    print(f"best field error   : {solution.final_field_error:.4e}")
    print(f"best final loss    : {solution.final_loss:.4e}")
    print(
        f"convergence iter   : "
        f"{solution.convergence_iter if solution.convergence_iter is not None else 'n/a'}"
    )
    print()
    print("--- per-restart receipts ---")
    for r in solution.all_restart_results:
        print(
            f"  restart {r.restart_idx} ({r.init_kind:>10}, seed={r.seed}): "
            f"iters={r.n_iters}  loss={r.final_loss:.4e}  "
            f"field_err={r.final_field_error:.4e}  "
            f"converged={r.converged}  wall={r.wall_time_s:.2f}s"
        )
    print()
    print("--- multi-modality witness ---")
    print(json.dumps(witness, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """End-to-end smoke test on test_idx=0 with reduced budget.

    Run via ``python -m ml_inverse.inverse --quiet`` won't trigger this;
    it is only fired by direct script invocation when no CLI argv is
    given. Kept light: 50 iters, 2 restarts, prints OK or raises.
    """
    sim_dir = Path(__file__).resolve().parents[1]
    ckpt_path = sim_dir / "ml_inverse" / "models" / "fno_surrogate_v2_1.pt"
    norm_path = (
        sim_dir / "ml_inverse" / "models" / "fno_surrogate_v2_1_norm.npz"
    )
    data_path = sim_dir / "ml_inverse" / "data" / "inverse_dataset_v3.h5"

    if not ckpt_path.exists() or not data_path.exists():
        print(
            "[smoke] skipping — checkpoint or dataset not present at "
            f"{ckpt_path} / {data_path}"
        )
        return

    device = "cpu"  # CPU for portability; CLI uses MPS by default
    model, norm = load_v21_forward_model(ckpt_path, norm_path, device=device)
    from ml_inverse.dataset import InverseSurrogateDataset

    ds = InverseSurrogateDataset(str(data_path), split="test", seed=42)
    sample = ds[0]
    phases_true = sample["phases"].numpy().astype(np.float32)
    slice_z_val = float(sample["slice_z"].item())

    with torch.no_grad():
        x = torch.cat(
            [
                torch.from_numpy(phases_true).unsqueeze(0),
                torch.tensor([[slice_z_val]]),
            ],
            dim=-1,
        )
        target = model(x)[0].cpu()

    solver = InverseSolver(
        forward_model=model,
        normalizer=norm,
        n_iters=50,
        lr=1e-1,
        n_restarts=2,
        device=device,
        seed=0,
    )
    sol = solver.solve(target_field=target, slice_z=slice_z_val)
    assert sol.phases.shape == (model.n_transducers,)
    assert len(sol.all_restart_results) == 2
    assert sol.loss_curve.size == 50
    print(
        f"[smoke] OK — best field_err={sol.final_field_error:.4e} "
        f"(2 restarts, 50 iters, cpu)"
    )


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) > 1:
        raise SystemExit(_cli_main())
    _smoke_test()
