"""PyTorch Dataset for the FNO surrogate (phases -> pressure field).

The HDF5 format is documented in ``simulations/ml_inverse/SCHEMA.md``.

Key invariants enforced here
----------------------------
* **Trajectory-level split.** ``unique(trajectory_id)`` is partitioned into
  train/val/test by deterministic hash; frames within one rollout
  trajectory cannot leak across splits. The split sets are asserted disjoint
  in ``__init__``.
* **Dimension-agnostic.** ``field_dims`` is read from HDF5 attrs and used
  everywhere - no hardcoded ``2`` or ``3`` for shape logic.
* **Source filter.** ``filter_source`` selects ``None`` / ``'uniform'`` /
  ``'rollout'`` for downstream eval that needs to compute loss per source.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


_SOURCE_TO_CODE = {"uniform": 0, "rollout": 1}


def _hash_trajectory_id(traj_id: int, seed: int) -> int:
    """Deterministic hash of (trajectory_id, seed) -> non-negative int.

    Used for assigning trajectories to splits without depending on Python's
    salted hash (``hash()`` is randomized per interpreter run).
    """
    payload = f"{int(traj_id)}::{int(seed)}".encode("utf-8")
    digest = hashlib.sha1(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _split_trajectory_ids(
    traj_ids: np.ndarray,
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Partition unique trajectory_ids into train/val/test arrays.

    The partition is deterministic in ``(traj_ids, seed)``: trajectories are
    hashed, sorted by hash, then sliced.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            "train_frac + val_frac must be < 1 to leave room for test, "
            f"got {train_frac} + {val_frac} = {train_frac + val_frac}"
        )

    unique_ids = np.unique(traj_ids)
    hashes = np.asarray(
        [_hash_trajectory_id(int(tid), seed) for tid in unique_ids],
        dtype=np.uint64,
    )
    order = np.argsort(hashes)
    ordered = unique_ids[order]

    n = ordered.size
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    # Guarantee at least one trajectory per split if possible.
    n_train = max(n_train, 1) if n >= 3 else n_train
    n_val = max(n_val, 1) if n >= 3 else n_val
    n_test = n - n_train - n_val
    if n_test < 1 and n >= 3:
        n_val = max(0, n_val - 1)
        n_test = n - n_train - n_val

    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }


class InverseSurrogateDataset(Dataset):
    """Yields ``(phases, field, source, slice_z)`` for the FNO surrogate.

    ``__getitem__`` returns a dict::

        {
            "phases":  torch.float32, shape (n_transducers,),
            "field":   torch.float32, shape (2, *grid_shape),  # ch0=Re, ch1=Im
            "source":  torch.uint8 scalar (0=uniform, 1=rollout),
            "slice_z": torch.float32 scalar (meters; NaN if v1 file w/o column),
        }

    The ``slice_z`` scalar is the field's slice z-plane (2D) or grid-center z
    (3D, where defined). Phase 1 v2 added this column; v1 files lack it and
    yield NaN (callers should treat as "unknown z" — the v1 generator used a
    fixed z=0.10m for uniform samples and the droplet's current z for rollout
    samples but did not store the latter).

    Parameters
    ----------
    h5_path
        Path to the HDF5 produced by ``generate_dataset.py``.
    split
        One of ``'train'``, ``'val'``, ``'test'``.
    train_frac, val_frac
        Fractions of unique trajectories assigned to train and val. ``test``
        gets the remainder.
    seed
        Hash seed for the deterministic split. Must match across train/val/
        test instantiations to guarantee disjointness.
    filter_source
        ``None`` to include all samples, ``'uniform'`` or ``'rollout'`` to
        keep only that subset within the chosen split.

    Attributes
    ----------
    field_dims : int
        Spatial dimension count (2 or 3) read from HDF5 attrs.
    n_transducers : int
        Read from HDF5 attrs.
    grid_shape : tuple of int
        ``(Nx, Ny)`` or ``(Nx, Ny, Nz)``.
    source_array : np.ndarray of uint8
        Per-sample source codes for the current split (post-filter), exposed
        for downstream stratified eval without needing to iterate the
        Dataset.
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        seed: int = 42,
        filter_source: Optional[str] = None,
    ) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py required to load the inverse dataset. "
                "Install: pip install h5py"
            ) from exc

        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        if filter_source is not None and filter_source not in _SOURCE_TO_CODE:
            raise ValueError(
                f"filter_source must be None / 'uniform' / 'rollout', "
                f"got {filter_source!r}"
            )

        self.h5_path = str(Path(h5_path).resolve())
        self.split = split
        self.seed = int(seed)
        self.filter_source = filter_source

        with h5py.File(self.h5_path, "r") as f:
            self._validate_h5(f)
            self.field_dims: int = int(f.attrs["field_dims"])
            self.n_transducers: int = int(f.attrs["n_transducers"])
            grid_resolution = np.asarray(f.attrs["grid_resolution"]).reshape(-1)
            if grid_resolution.size != self.field_dims:
                raise ValueError(
                    f"grid_resolution length {grid_resolution.size} != "
                    f"field_dims {self.field_dims} in {self.h5_path}"
                )
            self.grid_shape: Tuple[int, ...] = tuple(int(s) for s in grid_resolution)

            traj_all = f["trajectory_id"][...]
            source_all = f["source"][...]
            n_total = traj_all.shape[0]

            # Compute split trajectory_id sets.
            partitions = _split_trajectory_ids(
                traj_all,
                train_frac=train_frac,
                val_frac=val_frac,
                seed=self.seed,
            )
            train_set = set(int(t) for t in partitions["train"])
            val_set = set(int(t) for t in partitions["val"])
            test_set = set(int(t) for t in partitions["test"])

            # Hard guarantee no leakage: the three sets must be disjoint.
            assert train_set.isdisjoint(
                val_set
            ), "leaky split: train and val share trajectory_ids"
            assert train_set.isdisjoint(
                test_set
            ), "leaky split: train and test share trajectory_ids"
            assert val_set.isdisjoint(
                test_set
            ), "leaky split: val and test share trajectory_ids"

            chosen = {"train": train_set, "val": val_set, "test": test_set}[split]
            chosen_arr = (
                np.fromiter(chosen, dtype=np.int64, count=len(chosen))
                if chosen
                else np.empty(0, dtype=np.int64)
            )

            mask = np.isin(traj_all, chosen_arr)
            if filter_source is not None:
                code = _SOURCE_TO_CODE[filter_source]
                mask &= source_all == code

            self._row_indices = np.flatnonzero(mask).astype(np.int64)

            # Cache the post-filter source column so downstream code can
            # stratify without iterating __getitem__.
            self.source_array: np.ndarray = source_all[self._row_indices].astype(
                np.uint8,
                copy=True,
            )

            self._n_total_in_file = n_total
            self._partition_sizes = {
                "train": len(train_set),
                "val": len(val_set),
                "test": len(test_set),
            }
            # Backward-compat: v1 files lack the slice_z_m column. Cache a
            # boolean so __getitem__ skips the lookup on every call.
            self._has_slice_z = "slice_z_m" in f

        # Lazy-open the file in __getitem__ for multi-worker DataLoader safety.
        self._h5_file: Optional[Any] = None

    # ----- introspection helpers ---------------------------------------------

    @property
    def n_samples(self) -> int:
        return int(self._row_indices.size)

    @property
    def partition_sizes(self) -> Dict[str, int]:
        """Number of unique trajectory_ids per split."""
        return dict(self._partition_sizes)

    def __len__(self) -> int:
        return self.n_samples

    # ----- file handle (lazy, multi-worker safe) -----------------------------

    def _ensure_open(self) -> Any:
        import h5py

        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __getstate__(self) -> Dict[str, Any]:
        # Drop the open file handle on pickle (DataLoader workers re-open).
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __del__(self) -> None:  # pragma: no cover - destructor side effects
        try:
            if self._h5_file is not None:
                self._h5_file.close()
        except Exception:
            pass

    # ----- item access -------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(idx)
        row = int(self._row_indices[idx])
        f = self._ensure_open()

        phases = np.asarray(f["phases"][row], dtype=np.float32)
        # field shape: (*grid_shape,). Stack Re/Im into channel-first tensor.
        re = np.asarray(f["field_real"][row], dtype=np.float32)
        im = np.asarray(f["field_imag"][row], dtype=np.float32)
        # Validate shape against grid_shape so misaligned files fail loudly.
        if re.shape != self.grid_shape:
            raise ValueError(
                f"field_real row {row} has shape {re.shape}, "
                f"expected {self.grid_shape}"
            )
        field = np.stack([re, im], axis=0)  # (2, *grid_shape)
        source = int(f["source"][row])
        # Backward-compat: v1 files lack the slice_z_m column.
        if self._has_slice_z:
            slice_z_val = float(f["slice_z_m"][row])
        else:
            slice_z_val = float("nan")

        return {
            "phases": torch.from_numpy(phases),
            "field": torch.from_numpy(field),
            "source": torch.tensor(source, dtype=torch.uint8),
            "slice_z": torch.tensor(slice_z_val, dtype=torch.float32),
        }

    # ----- internal validation -----------------------------------------------

    @staticmethod
    def _validate_h5(f: Any) -> None:
        required_datasets = (
            "phases",
            "field_real",
            "field_imag",
            "trajectory_id",
            "source",
            "frame_idx",
            "transducer_positions",
        )
        for name in required_datasets:
            if name not in f:
                raise ValueError(
                    f"HDF5 missing required dataset '{name}' "
                    f"(see simulations/ml_inverse/SCHEMA.md)"
                )
        required_attrs = (
            "field_dims",
            "grid_resolution",
            "n_transducers",
            "frequency_hz",
            "sound_speed_m_s",
            "n_trajectories",
            "seed",
            "n_skipped_nonfinite",
            "drip_physics_version",
        )
        for name in required_attrs:
            if name not in f.attrs:
                raise ValueError(
                    f"HDF5 missing required attr '{name}' "
                    f"(see simulations/ml_inverse/SCHEMA.md)"
                )
