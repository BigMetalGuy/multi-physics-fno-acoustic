"""Mean-prediction sanity check for ``train.py``-format FNO checkpoints.

When to run
===========

After training any FNO surrogate via ``ml_inverse.train``, run this script
to verify the model has learned something meaningful — i.e. its
predictions are notably better than predicting the dataset's training
mean field for every input.

Pass criterion: ``ratio_mean = median(model_err) / median(mean_err) < 0.5``.
A FAIL means the model collapsed to the train mean (the classic Pattern A
failure mode that motivated the v2.1 prior fix; see
``research/V2_PRIOR_ATTRIBUTION.md``).

This sibling to ``eval_phase2.py`` exists because:

* ``eval_phase2.py`` expects an older checkpoint format with embedded
  ``model_config`` and uses ``ChannelNormalizer.compute_from_loader``;
* ``train.py``-format checkpoints save only ``model_state_dict`` to the
  ``_best.pt`` artifact (the full ckpt with ``model_config`` may be
  absent if training aborted early), and the normalizer is the persisted
  ``_norm.npz`` sibling.

The dataset-fallback config-reconstruction path mirrors what
``disagreement_analysis.py`` does for the same checkpoint family — see
``_resolve_model_config`` there.

Example
=======

::

    PYTHONPATH=. python3 -m ml_inverse.scripts.mean_pred_sanity \\
        --data-path ml_inverse/data/inverse_dataset_jwave_3d_full.h5 \\
        --ckpt-best ml_inverse/models/fno_surrogate_jwave_v1_3d_best.pt \\
        --norm ml_inverse/models/fno_surrogate_jwave_v1_3d_norm.npz \\
        --device mps \\
        --out ml_inverse/research/_jwave_eval/mean_pred.json

Receipts land at the path passed via ``--out``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import Subset

# Allow running both as ``python -m`` and as a plain script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_inverse.adapter import ChannelNormalizer, TrainerAdapterDataset  # noqa: E402
from ml_inverse.dataset import InverseSurrogateDataset  # noqa: E402
from ml_inverse.disagreement_analysis import _resolve_model_config  # noqa: E402
from ml_inverse.model import PhaseConditionedFNO  # noqa: E402

logger = logging.getLogger("ml_inverse.mean_pred_sanity")


def _build_model_from_config(cfg: Dict[str, Any], device: torch.device) -> PhaseConditionedFNO:
    """Construct a ``PhaseConditionedFNO`` from a ``train.py``-format config."""
    kwargs: Dict[str, Any] = {
        "field_dims": cfg["field_dims"],
        "grid_shape": tuple(cfg["grid_shape"]),
        "n_transducers": cfg["n_transducers"],
        "n_modes": tuple(cfg["n_modes"]),
        "hidden_channels": cfg["hidden_channels"],
        "cond_channels": cfg.get("cond_channels"),
        "out_channels": cfg.get("out_channels", 2),
    }
    if cfg.get("prior_enabled"):
        kwargs.update(
            transducer_positions=np.asarray(cfg["transducer_positions"], dtype=np.float32),
            grid_coords=[np.asarray(g, dtype=np.float32) for g in cfg["grid_coords"]],
            wavenumber=float(cfg["wavenumber"]),
            slice_z=cfg.get("slice_z"),
        )
    if cfg.get("residual_prior"):
        kwargs.update(
            residual_prior=True,
            prior_norm_mean=np.asarray(cfg["prior_norm_mean"], dtype=np.float32),
            prior_norm_std=np.asarray(cfg["prior_norm_std"], dtype=np.float32),
        )
    kwargs["prior_version"] = cfg.get("prior_version", "v2")

    model = PhaseConditionedFNO(**kwargs).to(device)
    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--data-path",
        required=True,
        help="HDF5 dataset used at training time (provides geometry + split logic).",
    )
    p.add_argument(
        "--ckpt-best",
        required=True,
        help="Path to the train.py ``_best.pt`` checkpoint to evaluate.",
    )
    p.add_argument(
        "--norm",
        required=True,
        help="Path to the persisted normalizer ``.npz`` (sibling of the ckpt).",
    )
    p.add_argument(
        "--config-source",
        default=None,
        help="Optional sibling full ckpt with embedded ``model_config``. If "
        "absent, config is reconstructed from --data-path + state_dict shapes.",
    )
    p.add_argument("--device", default="mps", help="torch device string")
    p.add_argument(
        "--n-train-mean",
        type=int,
        default=80,
        help="Number of train samples to average for the mean-field baseline.",
    )
    p.add_argument(
        "--n-test",
        type=int,
        default=30,
        help="Number of test samples to evaluate (capped at split size).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        default="/tmp/mean_pred_sanity.json",
        help="Where to write the JSON summary.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="PASS threshold on ratio_mean (default 0.5).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s %(message)s",
    )

    device = torch.device(args.device)
    logger.info("device=%s", device)

    ckpt = torch.load(args.ckpt_best, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_config")
    if model_cfg is None:
        model_cfg = _resolve_model_config(
            ckpt=ckpt,
            config_source=args.config_source,
            dataset_h5=args.data_path,
            device=device,
        )
    logger.info(
        "config: field_dims=%d grid=%s prior_version=%s",
        model_cfg["field_dims"],
        list(model_cfg["grid_shape"]),
        model_cfg.get("prior_version"),
    )

    model = _build_model_from_config(model_cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("loaded best weights; best_val_h1=%s", ckpt.get("best_val_h1", "n/a"))

    norm = ChannelNormalizer.load(args.norm)
    raw = InverseSurrogateDataset(args.data_path)
    n = len(raw)

    rng = np.random.default_rng(args.seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    train_n = int(0.7 * n)
    val_n = int(0.15 * n)
    train_idx = indices[:train_n].tolist()
    test_idx = indices[train_n + val_n :].tolist()

    adapter = TrainerAdapterDataset(raw, norm=norm)
    train_split = Subset(adapter, train_idx)
    test_split = Subset(adapter, test_idx)

    n_mean = min(args.n_train_mean, len(train_split))
    ys = [train_split[i]["y"].numpy() for i in range(n_mean)]
    y_train_mean = np.mean(np.stack(ys, axis=0), axis=0)
    y_train_mean_t = torch.from_numpy(y_train_mean).to(device)

    n_test = min(args.n_test, len(test_split))
    diffs_model: list[float] = []
    diffs_mean: list[float] = []
    with torch.no_grad():
        for i in range(n_test):
            item = test_split[i]
            x = item["x"].unsqueeze(0).to(device)
            y = item["y"].unsqueeze(0).to(device)
            y_pred = model(x)
            d_model = (y_pred - y).reshape(1, -1).norm(dim=1).item()
            d_mean = (y_train_mean_t.unsqueeze(0) - y).reshape(1, -1).norm(dim=1).item()
            diffs_model.append(d_model)
            diffs_mean.append(d_mean)

    dm = np.array(diffs_model)
    de = np.array(diffs_mean)
    ratios = dm / np.maximum(de, 1e-12)
    ratio_mean = float(ratios.mean())
    ratio_median = float(np.median(ratios))
    verdict = "PASS" if ratio_mean < args.threshold else "FAIL"

    logger.info("ratio_mean   = %.6f", ratio_mean)
    logger.info("ratio_median = %.6f", ratio_median)
    logger.info("verdict      = %s (threshold ratio < %.2f)", verdict, args.threshold)
    logger.info("median model err = %.4f", float(np.median(dm)))
    logger.info("median mean err  = %.4f", float(np.median(de)))

    out = {
        "ckpt_best": args.ckpt_best,
        "data_path": args.data_path,
        "n_train_mean_samples": int(n_mean),
        "n_test_samples": int(n_test),
        "ratio_mean": ratio_mean,
        "ratio_median": ratio_median,
        "verdict": verdict,
        "threshold": float(args.threshold),
        "median_model_err": float(np.median(dm)),
        "median_mean_err": float(np.median(de)),
        "best_val_h1": float(ckpt.get("best_val_h1", float("nan"))),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    logger.info("saved -> %s", out_path)

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
