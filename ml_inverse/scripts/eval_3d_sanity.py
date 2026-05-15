"""3D mean-prediction sanity check + source-stratified loss summary.

Lightweight evaluation script for the v2.1_3d FNO surrogate. Mirrors the
mean-prediction sanity check from ``eval_phase2.py`` (the full driver does
matplotlib plots and FFT-band analysis that aren't needed for the receipts
in ``research/3D_RETRY.md``).

Run from ``simulations/``::

    python -m ml_inverse.scripts.eval_3d_sanity \
        --checkpoint ml_inverse/models/fno_surrogate_3d_v2_corrected.pt \
        --norm ml_inverse/models/fno_surrogate_3d_v2_corrected_norm.npz \
        --data-path ml_inverse/data/inverse_dataset_3d_v1.h5

Outputs JSON to stdout with mean-pred ratio, test H1/L2, and a
source-stratified breakdown (uniform-only when the dataset has no rollouts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running as a module from simulations/.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from neuralop import H1Loss, LpLoss

from ml_inverse.adapter import ChannelNormalizer, TrainerAdapterDataset
from ml_inverse.dataset import InverseSurrogateDataset
from ml_inverse.model import PhaseConditionedFNO


def _auto_device(arg: Optional[str]) -> torch.device:
    if arg is not None:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model_from_checkpoint(
    ckpt: dict,
    *,
    device: torch.device,
) -> PhaseConditionedFNO:
    """Reconstruct ``PhaseConditionedFNO`` from a saved training checkpoint.

    The checkpoint stores both the state dict and the geometric / hyper-
    parameter config needed to rebuild the constructor; we trust those
    rather than re-reading the HDF5 here.
    """
    cfg = ckpt["model_config"]
    transducer_positions = np.asarray(
        cfg["transducer_positions"], dtype=np.float32
    )
    grid_coords = [np.asarray(g, dtype=np.float32) for g in cfg["grid_coords"]]
    residual_kwargs: Dict[str, object] = {}
    if cfg["residual_prior"]:
        residual_kwargs["residual_prior"] = True
        residual_kwargs["prior_norm_mean"] = np.asarray(
            cfg["prior_norm_mean"], dtype=np.float32
        )
        residual_kwargs["prior_norm_std"] = np.asarray(
            cfg["prior_norm_std"], dtype=np.float32
        )
    slice_z_kwarg = (
        None if int(cfg["field_dims"]) == 3 else float(cfg["slice_z"])
    )
    model = PhaseConditionedFNO(
        field_dims=int(cfg["field_dims"]),
        grid_shape=tuple(int(s) for s in cfg["grid_shape"]),
        n_transducers=int(cfg["n_transducers"]),
        n_modes=tuple(int(m) for m in cfg["n_modes"]),
        hidden_channels=int(cfg["hidden_channels"]),
        cond_channels=int(cfg["cond_channels"]),
        out_channels=int(cfg["out_channels"]),
        transducer_positions=transducer_positions,
        grid_coords=grid_coords,
        wavenumber=float(cfg["wavenumber"]),
        slice_z=slice_z_kwarg,
        prior_version=str(cfg["prior_version"]),
        **residual_kwargs,
    )
    # The training checkpoint stores the full state dict — including a
    # ``_metadata`` key that ``torch.nn.Module.load_state_dict`` rejects in
    # strict mode. Drop it (purely a hint for old PyTorch state-dict
    # serialization) before loading.
    state_dict = {
        k: v for k, v in ckpt["model_state_dict"].items() if k != "_metadata"
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def _eval_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    losses: Dict[str, torch.nn.Module],
    device: torch.device,
) -> Dict[str, float]:
    totals = {name: 0.0 for name in losses}
    n_total = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        y_pred = model(x)
        bs = x.shape[0]
        for name, loss_fn in losses.items():
            totals[name] += float(loss_fn(y_pred, y).item()) * bs
        n_total += bs
    return {
        name: (totals[name] / n_total) if n_total > 0 else float("nan")
        for name in losses
    }


@torch.no_grad()
def _mean_prediction_ratio(
    model: torch.nn.Module,
    train_ds: TrainerAdapterDataset,
    test_loader: DataLoader,
    device: torch.device,
    *,
    seed: int = 123,
    n_train_for_mean: int = 500,
) -> Dict[str, float]:
    """Mean-prediction sanity ratio: ``||pred - gt|| / ||train_mean - gt||``.

    Threshold: < 0.5 => PASS. The training mean in normalized space is
    approximately zero, so this rejects the trivial "always predict mean"
    failure mode (which v1 collapsed to).
    """
    rng = np.random.default_rng(seed)
    n_train = len(train_ds)
    sub = rng.choice(n_train, size=min(n_train_for_mean, n_train), replace=False)
    ys = [train_ds[int(i)]["y"].numpy() for i in sub]
    y_train_mean = np.mean(np.stack(ys, axis=0), axis=0)
    y_train_mean_t = torch.from_numpy(y_train_mean).to(device)

    # Use the first test batch (matches eval_phase2 behavior).
    batch = next(iter(test_loader))
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    y_pred = model(x)
    diff_model = (y_pred - y).reshape(y.shape[0], -1).norm(dim=1)
    diff_mean = (
        (y_train_mean_t.unsqueeze(0).expand_as(y) - y)
        .reshape(y.shape[0], -1)
        .norm(dim=1)
    )
    ratios = (diff_model / diff_mean).cpu().numpy()
    ratio_mean = float(ratios.mean())
    return {
        "ratio_mean": ratio_mean,
        "ratio_median": float(np.median(ratios)),
        "verdict": "PASS" if ratio_mean < 0.5 else "FAIL",
        "n_test_samples": int(ratios.size),
        "n_train_for_mean": int(sub.size),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--norm", type=str, required=True)
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _auto_device(args.device)
    print(f"[eval-3d] device: {device}", file=sys.stderr)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = _build_model_from_checkpoint(ckpt, device=device)
    norm = ChannelNormalizer.load(args.norm)

    field_dims = int(ckpt["model_config"]["field_dims"])
    pack_slice_z = ckpt["model_config"]["prior_version"] == "v2.1"

    train_raw = InverseSurrogateDataset(
        args.data_path, split="train", seed=args.seed
    )
    test_raw = InverseSurrogateDataset(
        args.data_path, split="test", seed=args.seed
    )
    test_uniform_raw = InverseSurrogateDataset(
        args.data_path, split="test", seed=args.seed, filter_source="uniform"
    )
    test_rollout_raw = InverseSurrogateDataset(
        args.data_path, split="test", seed=args.seed, filter_source="rollout"
    )

    train_ds = TrainerAdapterDataset(train_raw, norm=norm, pack_slice_z=pack_slice_z)
    test_ds = TrainerAdapterDataset(test_raw, norm=norm, pack_slice_z=pack_slice_z)
    test_uniform_ds = TrainerAdapterDataset(
        test_uniform_raw, norm=norm, pack_slice_z=pack_slice_z
    )
    test_rollout_ds = TrainerAdapterDataset(
        test_rollout_raw, norm=norm, pack_slice_z=pack_slice_z
    )

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    test_uniform_loader = DataLoader(
        test_uniform_ds, batch_size=args.batch_size, shuffle=False
    )
    test_rollout_loader = DataLoader(
        test_rollout_ds, batch_size=args.batch_size, shuffle=False
    )

    losses = {
        "h1": H1Loss(d=field_dims),
        "l2": LpLoss(d=field_dims, p=2),
    }

    # Use the first test batch for mean-prediction ratio (matches eval_phase2).
    sanity = _mean_prediction_ratio(model, train_ds, test_loader, device, seed=args.seed)

    test_all = _eval_loader(model, test_loader, losses, device)
    test_uniform = _eval_loader(model, test_uniform_loader, losses, device)
    test_rollout = _eval_loader(model, test_rollout_loader, losses, device)

    summary = {
        "checkpoint": str(args.checkpoint),
        "data_path": str(args.data_path),
        "device": str(device),
        "field_dims": field_dims,
        "prior_version": str(ckpt["model_config"]["prior_version"]),
        "n_train": len(train_raw),
        "n_test": len(test_raw),
        "n_test_uniform": len(test_uniform_raw),
        "n_test_rollout": len(test_rollout_raw),
        "test_all": test_all,
        "test_uniform": test_uniform,
        "test_rollout": test_rollout,
        "mean_pred_check": sanity,
    }
    out = json.dumps(summary, indent=2)
    print(out)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as fh:
            fh.write(out)


if __name__ == "__main__":
    main()
