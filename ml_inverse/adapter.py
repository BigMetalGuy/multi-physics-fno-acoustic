"""Trainer adapter — bridges Phase 1 Dataset keys to ``neuralop.Trainer``.

LOAD-BEARING CONTRACT
=====================

``neuralop.Trainer`` calls ``model(**sample)`` on each batch, where
``sample`` is the dict returned by ``Dataset.__getitem__`` after the
DataLoader collates. The Trainer requires the keys ``'x'`` (model
input) and ``'y'`` (loss target). Our Phase 1 Dataset uses descriptive
keys ``{'phases', 'field', 'source', 'slice_z'}`` instead, intentionally
(see ``ORCHESTRATION.md``).

This module wraps ``InverseSurrogateDataset`` so ``__getitem__`` yields
``{'x': phases, 'y': field, 'source': source, 'slice_z': slice_z}``.

DO NOT:
    * Rename the input arg from ``x`` → ``phases``. Trainer hard-codes ``x``.
    * Add ``forward_kwargs`` indirection. The neuralop API does not honor
      it; misnamed kwargs to FNO trigger a ``UserWarning`` and are silently
      discarded (see ``PHASE2_RESEARCH.md`` §8 #1).

PER-CHANNEL NORMALIZATION (PHASE2_RESEARCH §8 #9)
=================================================

Re and Im channels of the Phase 1 v3 dataset have very different scales
(Re std ~373 Pa, Im std ~229 Pa). Without normalization the H1Loss
gradient is dominated by Re and the model converges towards the
zero-channel mean for Im. ``ChannelNormalizer`` is computed on the train
split only (uniform + rollout combined) and saved alongside the
checkpoint as ``models/fno_surrogate_v1_norm.npz``. Eval de-normalizes
back to original Pa for plotting / downstream physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ml_inverse.dataset import InverseSurrogateDataset


@dataclass(frozen=True)
class ChannelNormalizer:
    """Per-channel (mean, std) for the Re/Im field tensor.

    Shapes are ``(C,)`` where ``C`` matches the field channel count
    (typically 2). ``std`` is clamped at ``eps`` to avoid divide-by-zero.
    """

    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    def __post_init__(self) -> None:
        # ``__post_init__`` on a frozen dataclass cannot rebind. Use
        # ``object.__setattr__`` only for sanity checks.
        if self.mean.shape != self.std.shape:
            raise ValueError(
                f"mean/std shape mismatch: {self.mean.shape} vs {self.std.shape}"
            )
        if self.mean.ndim != 1:
            raise ValueError(
                f"normalizer expects 1D per-channel stats; got {self.mean.ndim}D"
            )

    @property
    def n_channels(self) -> int:
        return int(self.mean.shape[0])

    def normalize(self, field: torch.Tensor) -> torch.Tensor:
        """Apply ``(field - mean) / max(std, eps)`` channel-wise.

        ``field`` shape: ``(C, *grid)``. Broadcasts along spatial dims.
        """
        if field.shape[0] != self.n_channels:
            raise ValueError(
                f"field has {field.shape[0]} channels, normalizer expects "
                f"{self.n_channels}"
            )
        view = (slice(None),) + (None,) * (field.ndim - 1)
        m = torch.from_numpy(self.mean.astype(np.float32))[view].to(field.device)
        s = torch.from_numpy(self.std.astype(np.float32))[view].to(field.device)
        s = torch.clamp(s, min=self.eps)
        return (field - m) / s

    def denormalize(self, field: torch.Tensor) -> torch.Tensor:
        """Reverse: ``field * std + mean``."""
        if field.shape[0] != self.n_channels:
            raise ValueError(
                f"field has {field.shape[0]} channels, normalizer expects "
                f"{self.n_channels}"
            )
        view = (slice(None),) + (None,) * (field.ndim - 1)
        m = torch.from_numpy(self.mean.astype(np.float32))[view].to(field.device)
        s = torch.from_numpy(self.std.astype(np.float32))[view].to(field.device)
        s = torch.clamp(s, min=self.eps)
        return field * s + m

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=np.asarray(self.mean, dtype=np.float32),
            std=np.asarray(self.std, dtype=np.float32),
            eps=np.float32(self.eps),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ChannelNormalizer":
        with np.load(str(path)) as data:
            return cls(
                mean=np.asarray(data["mean"], dtype=np.float32),
                std=np.asarray(data["std"], dtype=np.float32),
                eps=float(data["eps"]) if "eps" in data.files else 1e-6,
            )


def compute_channel_normalizer(
    train_dataset: InverseSurrogateDataset,
    *,
    sample_cap: Optional[int] = None,
    seed: int = 42,
) -> ChannelNormalizer:
    """Compute per-channel (mean, std) over the train split fields.

    Streams samples to avoid loading the full field tensor in memory.
    Combines uniform and rollout — the normalizer is global to the train
    split.

    Parameters
    ----------
    train_dataset
        ``InverseSurrogateDataset(split='train', filter_source=None)``.
    sample_cap
        If set, sub-samples this many rows (deterministic via ``seed``)
        instead of streaming the full split. Useful for very large
        datasets; for v3's 50k-train it's fine to use the whole split.
    seed
        RNG seed for sub-sampling order.
    """
    n = len(train_dataset)
    if n == 0:
        raise ValueError("train_dataset is empty; cannot compute normalizer")

    if sample_cap is not None and sample_cap < n:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n, size=sample_cap, replace=False)
    else:
        indices = np.arange(n)

    # Online (Welford-style) accumulators per channel.
    sample0 = train_dataset[int(indices[0])]
    field0 = sample0["field"].numpy()  # (C, *grid)
    n_channels = field0.shape[0]

    count = np.zeros(n_channels, dtype=np.float64)
    mean = np.zeros(n_channels, dtype=np.float64)
    m2 = np.zeros(n_channels, dtype=np.float64)

    def _update(field_arr: np.ndarray) -> None:
        # Each per-channel slice has ``prod(grid)`` pixels.
        for c in range(n_channels):
            flat = field_arr[c].ravel().astype(np.float64)
            n_new = flat.size
            mean_new = flat.mean()
            var_new = flat.var()
            n_old = count[c]
            n_total = n_old + n_new
            delta = mean_new - mean[c]
            new_mean = mean[c] + delta * (n_new / n_total)
            # Combined sum-of-squared-deviations (Welford parallel update):
            new_m2 = m2[c] + var_new * n_new + delta**2 * n_old * n_new / n_total
            mean[c] = new_mean
            m2[c] = new_m2
            count[c] = n_total

    _update(field0)
    for idx in indices[1:]:
        sample = train_dataset[int(idx)]
        _update(sample["field"].numpy())

    var = m2 / np.maximum(count, 1.0)
    std = np.sqrt(var)
    return ChannelNormalizer(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )


class TrainerAdapterDataset(Dataset):
    """Renames Phase 1 keys to neuralop Trainer keys; applies normalization.

    ``__getitem__`` returns::

        {
            'x':       phases  (n_transducers,) float32,    # model input
            'y':       (field - mean) / std    (C, *grid)    # loss target
            'source':  uint8 scalar (0=uniform, 1=rollout),  # passthrough
            'slice_z': float32 scalar (m),                   # passthrough
        }

    Trainer ignores extra keys; downstream eval uses 'source' and
    'slice_z' for stratification / plotting.

    The wrapped Dataset's ``filter_source`` is honored — pass it through
    on construction (don't filter here).

    v2.1 prior wiring
    -----------------
    When ``pack_slice_z=True`` (used by the ``--correct-prior`` v2.1
    training path), ``x`` becomes ``cat([phases, slice_z], dim=-1)`` of
    width ``n_transducers + 1``. ``model.forward`` unpacks the last
    column as the per-sample 2D slice z-plane to compute the corrected
    physical prior. NaN ``slice_z`` (v1 files lacking the column) is
    rejected loudly so we never silently train v2.1 on bogus z values.
    """

    def __init__(
        self,
        wrapped: InverseSurrogateDataset,
        *,
        norm: ChannelNormalizer,
        pack_slice_z: bool = False,
    ) -> None:
        self._wrapped = wrapped
        self._norm = norm
        self._pack_slice_z = bool(pack_slice_z)

    def __len__(self) -> int:
        return len(self._wrapped)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._wrapped[idx]
        field = sample["field"]                       # (C, *grid) float32
        if field.shape[0] != self._norm.n_channels:
            raise ValueError(
                f"field channel count {field.shape[0]} != normalizer "
                f"channel count {self._norm.n_channels}"
            )
        y = self._norm.normalize(field)

        x = sample["phases"]
        if self._pack_slice_z:
            sz = sample["slice_z"]
            if not torch.isfinite(sz).item():
                raise ValueError(
                    "pack_slice_z=True requires finite slice_z; got NaN. "
                    "v2.1 prior cannot be computed without per-sample z. "
                    "Use a v2+ HDF5 file containing the slice_z_m column."
                )
            x = torch.cat([x, sz.reshape(1).to(x.dtype)], dim=0)

        return {
            "x": x,
            "y": y,
            "source": sample["source"],
            "slice_z": sample["slice_z"],
        }

    @property
    def normalizer(self) -> ChannelNormalizer:
        return self._norm

    @property
    def wrapped(self) -> InverseSurrogateDataset:
        return self._wrapped
