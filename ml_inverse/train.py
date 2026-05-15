"""Phase 2 — train the forward FNO surrogate.

Run from ``simulations/`` (so that ``import ml_inverse...`` resolves)::

    python -m ml_inverse.train --n-epochs 30

CLI flags
---------
    --n-epochs           default 30
    --batch-size         default 32
    --lr                 default 1e-3   (NOT 1e-2; H1Loss + Cosine on small
                                          datasets diverges at 1e-2)
    --device             cpu | cuda | mps   default auto-detect
    --seed               default 42
    --data-path          default ml_inverse/data/inverse_dataset_v3.h5
    --n-modes            optional override, e.g. ``--n-modes 16 16``
    --hidden-channels    default 64
    --norm-sample-cap    optional sub-sample for normalizer estimation
    --output-dir         default ml_inverse/models

DataLoader notes
----------------
``num_workers=0`` is intentional: deterministic, no fork/IPC, fine for a
63k-sample dataset where HDF5 random access is the bottleneck rather than
CPU. Documented per PHASE2_PROMPT.md reproducibility section so this is
not silently changed later. If you raise ``num_workers``, also set
``worker_init_fn=lambda wid: numpy.random.seed(seed + wid)``.

Source-stratified eval
----------------------
Three test loaders are constructed:

    1. ``test_all`` — unfiltered.
    2. ``test_uniform`` — ``filter_source='uniform'``.
    3. ``test_rollout`` — ``filter_source='rollout'``.

All use the SAME ``seed=42`` so the trajectory partition is identical.
The Trainer's ``test_loaders`` dict carries all three; final test loop
also computes overall vs. source-stratified losses manually for
reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Top-level neuralop imports per PHASE2_RESEARCH §2.
#
# We use ``torch.optim.AdamW`` rather than ``neuralop.training.AdamW``:
# the neuralop variant produces NaN parameters after the first
# optimizer step on Apple-Silicon MPS (post-step model output becomes
# all-NaN even when forward + grads were finite). ``torch.optim.AdamW``
# trains identically on the FNO and is portable to CPU/CUDA/MPS.
# Documented as a deviation from PHASE2_RESEARCH §7 in the report.
try:
    from neuralop import H1Loss, LpLoss
except ImportError as exc:  # pragma: no cover - import-time check
    print(
        "ERROR: neuraloperator not installed. Install with:\n"
        "    pip install neuraloperator\n"
        "(Note: torch is NOT auto-installed by neuraloperator; install "
        "torch>=2.0 separately if missing — issue #321.)",
        file=sys.stderr,
    )
    raise

from ml_inverse.adapter import (
    TrainerAdapterDataset,
    compute_channel_normalizer,
)
from ml_inverse.dataset import InverseSurrogateDataset
from ml_inverse.model import PhaseConditionedFNO


# ---------------------------------------------------------------------------
# Per-sample relative H1 loss (Phase 6.4 diagnostic)
# ---------------------------------------------------------------------------

class PerSampleRelativeH1Loss(torch.nn.Module):
    """Per-sample relative H1: mean over batch of ‖pred-target‖_H1 / ‖target‖_H1.

    The neuraloperator H1Loss with ``relative=True`` (default) pools all
    samples in a single global denominator before computing the relative
    norm — so a batch with one 1 Pa sample and one 300 Pa sample shares
    the same scaling. This means an FNO can satisfy the loss by getting
    the *average* magnitude right while ignoring per-sample structure
    (the failure mode observed in Phase 6.3 — "predict the mean"
    attractor at ratio_mean ≈ 1.0).

    This per-sample variant computes the H1 ratio for each sample
    independently and averages. Cannot be satisfied by mean-prediction:
    each sample's loss is normalised by its own magnitude, so
    structure errors are penalised equally regardless of scale.

    H1 = sqrt(L2(value)^2 + L2(gradient)^2) where the gradient is
    computed via central finite differences along each spatial axis.
    Channels are pooled into the L2 sum.
    """

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        if d not in (1, 2, 3):
            raise ValueError(f"PerSampleRelativeH1Loss: d must be 1/2/3, got {d}")
        self.d = int(d)
        self.eps = float(eps)

    def _h1_per_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample H1 norm. Input shape (B, C, *spatial); returns (B,)."""
        # L2 over all non-batch dims (channels + spatial)
        spatial_dims = tuple(range(1, x.ndim))
        l2_sq = x.pow(2).sum(dim=spatial_dims)
        # Gradient via finite diff along each spatial axis. Spatial axes
        # are 2..2+d (since 0=batch, 1=channel).
        grad_sq = torch.zeros_like(l2_sq)
        for ax in range(2, 2 + self.d):
            d_x = torch.diff(x, dim=ax)
            grad_sq = grad_sq + d_x.pow(2).sum(dim=spatial_dims)
        return torch.sqrt(l2_sq + grad_sq + self.eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff_h1 = self._h1_per_sample(pred - target)
        target_h1 = self._h1_per_sample(target)
        return (diff_h1 / (target_h1 + self.eps)).mean()


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> torch.Generator:
    """Seed numpy, torch, cuda, and return a torch.Generator for DataLoader.

    Order matters — call BEFORE constructing model / dataloaders.
    """
    np.random.seed(seed)
    # ``np.random.default_rng(seed)`` returns a Generator; this seeds the
    # global module RNG, used by helpers that don't accept a Generator.
    torch.manual_seed(seed)
    # Cheap on CPU; a no-op when cuda is unavailable but documented in
    # PHASE2_PROMPT to set unconditionally.
    torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def auto_device(arg: Optional[str]) -> torch.device:
    """Pick a sensible device when ``--device`` not provided.

    Preference: cuda > mps > cpu.
    """
    if arg is not None:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataloader construction
# ---------------------------------------------------------------------------


def build_loader(
    dataset: TrainerAdapterDataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: Optional[torch.Generator],
) -> DataLoader:
    """Construct a DataLoader. ``num_workers=0`` is intentional."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
        drop_last=False,
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# Per-epoch loss curve logger
# ---------------------------------------------------------------------------


class CurveLogger:
    """Writes a CSV row per epoch (train, val H1, val L2, test H1, test L2).

    Used to verify convergence trend, not just final loss.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._fh = open(path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow([
            "epoch",
            "train_h1",
            "val_h1",
            "val_l2",
            "test_all_h1",
            "test_all_l2",
            "elapsed_s",
        ])
        self._fh.flush()

    def log(
        self,
        epoch: int,
        *,
        train_h1: float,
        val_h1: Optional[float],
        val_l2: Optional[float],
        test_all_h1: Optional[float],
        test_all_l2: Optional[float],
        elapsed_s: float,
    ) -> None:
        def _fmt(x: Optional[float]) -> str:
            return f"{x:.6e}" if x is not None else ""

        self._writer.writerow([
            epoch,
            _fmt(train_h1),
            _fmt(val_h1),
            _fmt(val_l2),
            _fmt(test_all_h1),
            _fmt(test_all_l2),
            f"{elapsed_s:.3f}",
        ])
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Manual eval (source-stratified + sanity checks)
# ---------------------------------------------------------------------------


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def eval_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    losses: Dict[str, torch.nn.Module],
    device: torch.device,
) -> Dict[str, float]:
    """Compute mean of each named loss over a DataLoader.

    Sample-weighted: each batch's loss is summed with weight equal to
    batch size, divided by total samples at the end. Avoids the
    "last batch smaller" mean-bias in naive epoch averaging.
    """
    model.eval()
    totals = {name: 0.0 for name in losses}
    n_total = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        x = batch["x"]
        y = batch["y"]
        y_pred = model(x)
        bs = x.shape[0]
        for name, loss_fn in losses.items():
            val = loss_fn(y_pred, y)
            totals[name] += float(val.item()) * bs
        n_total += bs
    return {name: (totals[name] / n_total) if n_total > 0 else float("nan") for name in losses}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Phase 2 FNO surrogate.")
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)  # NOT 1e-2.
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--data-path",
        type=str,
        default="ml_inverse/data/inverse_dataset_v3.h5",
    )
    p.add_argument(
        "--n-modes",
        type=int,
        nargs="+",
        default=None,
        help="Optional override; default is grid // 4 per axis.",
    )
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument(
        "--norm-sample-cap",
        type=int,
        default=None,
        help="Sub-sample N samples for normalizer; default uses full train split.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="ml_inverse/models",
    )
    p.add_argument(
        "--checkpoint-name",
        type=str,
        default="fno_surrogate_v1",
    )
    p.add_argument(
        "--residual-prior",
        action="store_true",
        default=False,
        help=(
            "v2: model returns normalize(prior) + fno_output instead of "
            "fno_output. Requires the prior to be enabled (transducer "
            "positions, grid coords, wavenumber from HDF5)."
        ),
    )
    p.add_argument(
        "--correct-prior",
        action="store_true",
        default=False,
        help=(
            "v2.1: replace the broken v2 physical prior with the full "
            "formula matching compute_pressure_field "
            "(per-sample slice_z, +k·r, p_ref·r_ref scale, piston "
            "directivity). Implies --residual-prior. See "
            "research/V2_PRIOR_ATTRIBUTION.md."
        ),
    )
    p.add_argument(
        "--correct-prior-3d",
        action="store_true",
        default=False,
        help=(
            "v2.1_3d: 3D extension of --correct-prior. The four bug fixes "
            "(per-sample slice_z, +k·r sign, p_ref·r_ref·D(θ)/r amplitude, "
            "sinc directivity) are applied at every voxel of the 3D grid. "
            "No slice_z packing because z is a grid axis. Requires "
            "field_dims=3 dataset. Implies --residual-prior. See "
            "research/3D_RETRY.md."
        ),
    )
    p.add_argument(
        "--prior-scale-factor",
        type=float,
        default=1.0,
        help=(
            "Scale the residual prior by this factor before adding to the "
            "FNO output. Default 1.0 = original residual_prior behavior. "
            "Use ~0.0033 (1/300) when targets are in a much smaller "
            "magnitude regime than the analytical free-field prior, e.g. "
            "FEM-coupled chamber fields (~1 Pa) vs analytical prior "
            "(~hundreds of Pa). Phase 6.3 diagnostic — see "
            "research/FNO_F_TRAINING_RESULTS_PHASE62.md."
        ),
    )
    p.add_argument(
        "--per-sample-loss",
        action="store_true",
        default=False,
        help=(
            "Use per-sample relative H1 loss instead of the default "
            "neuraloperator H1Loss (which pools samples in a global "
            "denominator). The default-pooled loss can be satisfied by "
            "predicting the dataset mean — Phase 6.3 confirmed this "
            "failure mode at ratio_mean ≈ 1.0. Per-sample relative "
            "normalisation makes each sample's loss respect its own "
            "magnitude, removing the mean-prediction attractor. "
            "Phase 6.4 diagnostic."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---------- reproducibility (BEFORE anything stochastic) ----------
    generator = seed_everything(args.seed)

    device = auto_device(args.device)
    print(f"[train] device: {device}")
    print(f"[train] seed: {args.seed}  num_workers: 0 (intentional, see header)")

    data_path = str(Path(args.data_path).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / f"{args.checkpoint_name}.pt"
    norm_path = output_dir / f"{args.checkpoint_name}_norm.npz"
    curves_path = output_dir / f"{args.checkpoint_name}_curves.csv"
    metrics_path = output_dir / f"{args.checkpoint_name}_metrics.json"

    # ---------- datasets ----------
    print(f"[train] loading dataset: {data_path}")
    train_raw = InverseSurrogateDataset(data_path, split="train", seed=args.seed)
    val_raw = InverseSurrogateDataset(data_path, split="val", seed=args.seed)
    test_raw = InverseSurrogateDataset(data_path, split="test", seed=args.seed)
    test_uniform_raw = InverseSurrogateDataset(
        data_path, split="test", seed=args.seed, filter_source="uniform"
    )
    test_rollout_raw = InverseSurrogateDataset(
        data_path, split="test", seed=args.seed, filter_source="rollout"
    )

    print(
        f"[train] split sizes: train={len(train_raw)} val={len(val_raw)} "
        f"test={len(test_raw)} (test_uniform={len(test_uniform_raw)} "
        f"test_rollout={len(test_rollout_raw)})"
    )
    print(
        f"[train] grid_shape={train_raw.grid_shape} field_dims={train_raw.field_dims} "
        f"n_transducers={train_raw.n_transducers}"
    )

    # ---------- normalizer (train split only) ----------
    print("[train] computing per-channel normalizer (train split)...")
    norm = compute_channel_normalizer(
        train_raw, sample_cap=args.norm_sample_cap, seed=args.seed
    )
    print(f"[train] normalizer mean={norm.mean.tolist()}  std={norm.std.tolist()}")
    norm.save(norm_path)
    print(f"[train] saved normalizer -> {norm_path}")

    # Mutex: --correct-prior (2D) and --correct-prior-3d are alternatives;
    # both cannot be set at once. The 3D path requires a 3D dataset; the 2D
    # path requires a 2D dataset. Each implies --residual-prior because the
    # corrected prior is only useful as the model's default output (the FNO
    # learns the residual against it).
    if args.correct_prior and args.correct_prior_3d:
        raise SystemExit(
            "--correct-prior (2D) and --correct-prior-3d are mutually exclusive."
        )
    if args.correct_prior_3d and train_raw.field_dims != 3:
        raise SystemExit(
            f"--correct-prior-3d requires a 3D dataset (field_dims=3); "
            f"got field_dims={train_raw.field_dims} from {data_path}."
        )
    if args.correct_prior and train_raw.field_dims != 2:
        raise SystemExit(
            f"--correct-prior requires a 2D dataset (field_dims=2); "
            f"got field_dims={train_raw.field_dims} from {data_path}. "
            f"Use --correct-prior-3d for 3D."
        )

    # v2.1 wiring: when --correct-prior is set, the adapter packs
    # slice_z as the trailing element of x so model.forward can compute
    # the per-sample physical prior with the right z. v2.1_3d does NOT
    # pack slice_z (z is a grid axis), so pack_slice_z stays False there.
    pack_slice_z = bool(args.correct_prior)
    if (args.correct_prior or args.correct_prior_3d) and not args.residual_prior:
        flag = "--correct-prior" if args.correct_prior else "--correct-prior-3d"
        print(f"[train] {flag} implies --residual-prior; enabling.")
        args.residual_prior = True

    train_ds = TrainerAdapterDataset(train_raw, norm=norm, pack_slice_z=pack_slice_z)
    val_ds = TrainerAdapterDataset(val_raw, norm=norm, pack_slice_z=pack_slice_z)
    test_ds = TrainerAdapterDataset(test_raw, norm=norm, pack_slice_z=pack_slice_z)
    test_uniform_ds = TrainerAdapterDataset(
        test_uniform_raw, norm=norm, pack_slice_z=pack_slice_z
    )
    test_rollout_ds = TrainerAdapterDataset(
        test_rollout_raw, norm=norm, pack_slice_z=pack_slice_z
    )

    # ---------- dataloaders ----------
    train_loader = build_loader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    val_loader = build_loader(
        val_ds, batch_size=args.batch_size, shuffle=False, generator=None
    )
    test_loader = build_loader(
        test_ds, batch_size=args.batch_size, shuffle=False, generator=None
    )
    test_uniform_loader = build_loader(
        test_uniform_ds, batch_size=args.batch_size, shuffle=False, generator=None
    )
    test_rollout_loader = build_loader(
        test_rollout_ds, batch_size=args.batch_size, shuffle=False, generator=None
    )

    # ---------- model ----------
    # Read transducer positions, grid coords, frequency, and sound speed
    # from the HDF5 attrs/datasets so the model can compute the
    # analytical free-field physical-prior channels. Without this prior,
    # broadcast-only Pattern A stagnates at the zero-prediction baseline
    # on this dataset (verified empirically; see model.py docstring).
    import h5py
    with h5py.File(data_path, "r") as f:
        transducer_positions = f["transducer_positions"][...].astype(np.float32)
        grid_coords_axes = []
        for ax in ("x", "y", "z")[: train_raw.field_dims]:
            grid_coords_axes.append(f[f"grid_coords/{ax}"][...].astype(np.float32))
        frequency_hz = float(f.attrs["frequency_hz"])
        sound_speed_m_s = float(f.attrs["sound_speed_m_s"])
        # Use the median uniform z as the prior's z-plane for the 2D field
        # (the dataset spans z ∈ [0, 0.18] m, but the per-sample slice_z
        # variation is partially absorbed by the lifting MLP's residual).
        slice_z_default = 0.09
    wavenumber = 2.0 * np.pi * frequency_hz / sound_speed_m_s
    print(
        f"[train] physical prior: f={frequency_hz/1e3:.1f} kHz  c={sound_speed_m_s:.1f} m/s  "
        f"k={wavenumber:.2f} rad/m  slice_z={slice_z_default*1000:.1f} mm"
    )

    n_modes_arg: Optional[Tuple[int, ...]] = (
        tuple(int(m) for m in args.n_modes) if args.n_modes else None
    )
    # v2: pass per-channel (mean, std) so the prior is normalized at output
    # time, matching the loss's coordinate space. Reuses the adapter's stats.
    residual_kwargs = {}
    if args.residual_prior:
        residual_kwargs = dict(
            residual_prior=True,
            prior_norm_mean=norm.mean,
            prior_norm_std=norm.std,
            prior_scale_factor=float(args.prior_scale_factor),
        )
    if args.correct_prior:
        prior_version_arg = "v2.1"
    elif args.correct_prior_3d:
        prior_version_arg = "v2.1_3d"
    else:
        prior_version_arg = "v2"
    # ``slice_z`` is the legacy v2-prior fallback for 2D fields; v2.1_3d
    # ignores it entirely (z is a grid axis). Pass None in 3D so the
    # constructor doesn't try to broadcast a scalar into a 3D buffer.
    slice_z_kwarg = None if train_raw.field_dims == 3 else slice_z_default
    model = PhaseConditionedFNO(
        field_dims=train_raw.field_dims,
        grid_shape=train_raw.grid_shape,
        n_transducers=train_raw.n_transducers,
        n_modes=n_modes_arg,
        hidden_channels=args.hidden_channels,
        transducer_positions=transducer_positions,
        grid_coords=grid_coords_axes,
        wavenumber=wavenumber,
        slice_z=slice_z_kwarg,
        prior_version=prior_version_arg,
        **residual_kwargs,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[train] model n_modes={model.n_modes} hidden={model.hidden_channels} "
        f"params={n_params:,}  prior_enabled={model.prior_enabled}  "
        f"residual_prior={model.residual_prior}"
    )

    # ---------- optimizer / scheduler / losses ----------
    # torch.optim.AdamW (NOT neuralop.training.AdamW) — see import-time
    # comment for rationale. neuralop.AdamW NaNs on MPS post-step.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    if args.per_sample_loss:
        train_loss_fn: torch.nn.Module = PerSampleRelativeH1Loss(d=train_raw.field_dims)
        print("[train] loss: per-sample relative H1 (Phase 6.4 diagnostic)")
    else:
        train_loss_fn = H1Loss(d=train_raw.field_dims)
    # Eval losses stay on neuralop's pooled H1/L2 so eval metrics remain
    # comparable to prior phases. The training loss is the only thing
    # that changes when --per-sample-loss is on.
    eval_h1_fn = H1Loss(d=train_raw.field_dims)
    eval_l2_fn = LpLoss(d=train_raw.field_dims, p=2)
    eval_losses = {"h1": eval_h1_fn, "l2": eval_l2_fn}

    # ---------- training loop ----------
    # We hand-roll the training loop (instead of neuralop.Trainer) so that:
    #   1. We can write per-epoch curves to CSV cleanly.
    #   2. We can do source-stratified eval at the end without fighting the
    #      Trainer's eval-by-key contract.
    #   3. We retain the first-batch warning-as-error guard inside
    #      ``model.forward`` (Trainer's ``model(**sample)`` works fine with
    #      our ``forward(self, x)`` because Trainer only forwards 'x').
    # The Trainer was scaffolded but moving back to a hand loop is the
    # documented escape hatch in PHASE2_RESEARCH §7.
    curve_logger = CurveLogger(curves_path)
    best_val_h1 = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    train_started = time.time()

    n_batches_per_epoch = max(1, len(train_loader))
    log_interval = max(1, n_batches_per_epoch // 10)
    for epoch in range(1, args.n_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_total = 0.0
        train_count = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = _move_batch(batch, device)
            x = batch["x"]
            y = batch["y"]
            optimizer.zero_grad()
            y_pred = model(x)
            loss = train_loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            bs = x.shape[0]
            train_total += float(loss.item()) * bs
            train_count += bs
            if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
                pct = 100.0 * (batch_idx + 1) / n_batches_per_epoch
                avg = train_total / max(train_count, 1)
                print(
                    f"[train]   ep {epoch}/{args.n_epochs} batch {batch_idx+1}/{n_batches_per_epoch} "
                    f"({pct:5.1f}%)  running H1={avg:.4e}",
                    flush=True,
                )
        scheduler.step()
        train_h1 = train_total / max(train_count, 1)

        # Run eval every epoch on val; eval on test_all only every 5
        # epochs to keep the wallclock tight.
        val_metrics = eval_loader(model, val_loader, eval_losses, device)
        test_all_h1 = None
        test_all_l2 = None
        if epoch % 5 == 0 or epoch == args.n_epochs:
            test_all_metrics = eval_loader(model, test_loader, eval_losses, device)
            test_all_h1 = test_all_metrics["h1"]
            test_all_l2 = test_all_metrics["l2"]

        elapsed = time.time() - epoch_start
        print(
            f"[train] epoch {epoch:3d}/{args.n_epochs}  "
            f"train_h1={train_h1:.4e}  val_h1={val_metrics['h1']:.4e}  "
            f"val_l2={val_metrics['l2']:.4e}  "
            + (f"test_h1={test_all_h1:.4e}  test_l2={test_all_l2:.4e}  "
               if test_all_h1 is not None else "")
            + f"({elapsed:.1f}s)"
        )

        curve_logger.log(
            epoch,
            train_h1=train_h1,
            val_h1=val_metrics["h1"],
            val_l2=val_metrics["l2"],
            test_all_h1=test_all_h1,
            test_all_l2=test_all_l2,
            elapsed_s=elapsed,
        )

        if val_metrics["h1"] < best_val_h1:
            best_val_h1 = val_metrics["h1"]
            best_state = {
    k: v.detach().cpu().clone()
    for k, v in model.state_dict().items()
    if torch.is_tensor(v)
}
            # Mid-training checkpoint: save the best-so-far state to disk
            # every time it improves. Lets the run resume / recover if
            # the wall-clock budget runs out before all epochs finish.
            mid_ckpt_path = output_dir / f"{args.checkpoint_name}_best.pt"
            torch.save({
                "model_state_dict": best_state,
                "epoch": epoch,
                "best_val_h1": float(best_val_h1),
            }, mid_ckpt_path)

    train_elapsed = time.time() - train_started
    curve_logger.close()
    print(f"[train] training done in {train_elapsed:.1f}s; best val H1 {best_val_h1:.4e}")

    # Restore best weights for final eval / checkpoint.
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------- final source-stratified eval ----------
    print("[eval] running source-stratified test eval...")
    test_all_metrics = eval_loader(model, test_loader, eval_losses, device)
    test_uniform_metrics = eval_loader(model, test_uniform_loader, eval_losses, device)
    test_rollout_metrics = eval_loader(model, test_rollout_loader, eval_losses, device)

    print(
        "[eval] test loss summary (normalized space):\n"
        f"  ALL      : H1={test_all_metrics['h1']:.4e}  L2={test_all_metrics['l2']:.4e}\n"
        f"  UNIFORM  : H1={test_uniform_metrics['h1']:.4e}  L2={test_uniform_metrics['l2']:.4e}\n"
        f"  ROLLOUT  : H1={test_rollout_metrics['h1']:.4e}  L2={test_rollout_metrics['l2']:.4e}"
    )

    # Flag a > 3x gap explicitly.
    gap_l2 = test_rollout_metrics["l2"] / max(test_uniform_metrics["l2"], 1e-12)
    if gap_l2 > 3.0:
        print(
            f"[eval] WARNING: rollout/uniform L2 gap = {gap_l2:.2f}x  (>3x threshold). "
            f"Surrogate may not transfer from training to operating distribution."
        )

    # ---------- save checkpoint ----------
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "field_dims": int(train_raw.field_dims),
            "grid_shape": list(train_raw.grid_shape),
            "n_transducers": int(train_raw.n_transducers),
            "n_modes": list(model.n_modes),
            "hidden_channels": int(model.hidden_channels),
            "cond_channels": int(model.cond_channels),
            "out_channels": int(model.out_channels),
            "prior_enabled": bool(model.prior_enabled),
            "residual_prior": bool(model.residual_prior),
            "prior_version": str(model.prior_version),
            "wavenumber": float(wavenumber),
            "slice_z": float(slice_z_default),
            # Save the geometry too — the buffers go via state_dict but
            # the constructor needs these to rebuild the architecture.
            "transducer_positions": transducer_positions.tolist(),
            "grid_coords": [g.tolist() for g in grid_coords_axes],
            # v2: persist the prior-normalization stats alongside the
            # model so eval can rebuild it without consulting the adapter.
            "prior_norm_mean": norm.mean.tolist() if model.residual_prior else None,
            "prior_norm_std": norm.std.tolist() if model.residual_prior else None,
        },
        "training_config": {
            "n_epochs": int(args.n_epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "data_path": data_path,
            "device": str(device),
        },
        "metrics": {
            "best_val_h1": float(best_val_h1),
            "test_all": {k: float(v) for k, v in test_all_metrics.items()},
            "test_uniform": {k: float(v) for k, v in test_uniform_metrics.items()},
            "test_rollout": {k: float(v) for k, v in test_rollout_metrics.items()},
        },
    }
    torch.save(ckpt, ckpt_path)
    print(f"[train] saved checkpoint -> {ckpt_path}")

    # Mirror metrics to a JSON for quick external read.
    with open(metrics_path, "w") as fh:
        json.dump(ckpt["metrics"], fh, indent=2)
    print(f"[train] saved metrics    -> {metrics_path}")
    print(f"[train] curves           -> {curves_path}")


if __name__ == "__main__":
    main()
