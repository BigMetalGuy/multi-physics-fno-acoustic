"""Phase-conditioned FNO surrogate.

Maps transducer phase commands ``(B, n_transducers)`` to the complex
acoustic pressure field on a regular grid, represented as two real
channels ``(B, 2, *grid_shape)`` (Re, Im).

Conditioning pattern
====================

We use **Pattern A** from PHASE2_RESEARCH §5 (small MLP → broadcast →
FNO), augmented with a **physical-prior channel** when transducer
geometry is provided. The FNO's input channels are:

1. ``cond_channels`` from the lifting MLP, broadcast spatially. Carries
   the phase configuration as a uniform feature vector.
2. (optional) **2 physical-prior channels** from the analytical
   free-field acoustic pressure: ``Re(sum_i (1/r_i) e^{j(phi_i - k r_i(x,y))})``
   and ``Im(...)``. These give the FNO a strong spatially-varying signal
   tied to the phase input — the FNO then learns the residual between
   free-field and the simulation's full physics. Without this prior,
   the broadcast-only Pattern A path gives the FNO no spatial structure
   tied to the input (only the default grid positional embedding), and
   training stagnates near the zero-prediction baseline on this dataset.

The 2 physical channels are computed analytically inside ``forward``;
they require ``transducer_positions``, ``grid_coords``, and
``wavenumber`` to be passed at construction. If any are missing,
the model falls back to plain Pattern A.

Residual connection (v2)
========================

When ``residual_prior=True`` (v2 default behavior, opt-in flag), the
model returns ``normalize(prior) + fno_residual`` instead of
``fno_residual`` alone. Rationale: v1 collapsed to predicting the
training mean (~zero in normalized space) despite having the analytical
prior available as input channels — Pattern A's uniform-broadcast
conditioning was structurally insufficient for the FNO to extract
per-pixel phase information. The residual framing makes the analytical
solution the *default* output; the FNO only needs to learn a small
correction. If the FNO collapses to outputting zero (its current
attractor), the prediction equals the analytical solution — at least
non-degenerate.

The prior is computed in physical units (Pa) but the loss is computed
in normalized space (the adapter pre-divides ground truth by per-channel
std). For the residual addition to live in the same coordinate space as
the loss, the prior must be normalized first — using the SAME
``(mean, std)`` the adapter applies to the ground truth. This is what
``prior_norm_mean`` and ``prior_norm_std`` are for: pass them at
construction so they get registered as buffers, and ``forward`` will
apply ``(prior - mean) / std`` channel-wise before adding the residual.

If ``residual_prior=False`` (v1 behavior), the prior is only used as an
input channel to the FNO and the model output is whatever the FNO
produces directly.

Forward signature contract (load-bearing)
-----------------------------------------
``forward(self, x)`` — single positional arg named ``x``. This matches
``neuralop.Trainer``'s call ``model(**sample)`` with ``sample = {'x': ...,
'y': ...}``. We deliberately do NOT include ``**kwargs`` in the signature
so any future caller passing the wrong key raises a ``TypeError``
instead of silently dropping data.

First-batch warning guard
-------------------------
On the first ``forward`` call we wrap the inner ``self.fno(...)`` with
``warnings.simplefilter('error', UserWarning)``. The Pattern A path does
not pass kwargs to ``FNO.forward``, but ``FNO.forward(self, x,
output_shape=None, **kwargs)`` will silently warn-and-discard unrecognized
kwargs if anyone later rewires this wrapper. PHASE2_RESEARCH §8 #1
documents this as the highest-priority silent-failure mode.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from neuralop.models import FNO


def _default_modes(grid_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Default mode count: ``g // 4`` per axis (16 for a 64-cell axis).

    Matches PHASE2_RESEARCH §8 #3 floor (~10–12 modes for acoustic
    interference at λ/2 ≈ 4.3 mm on a 50 mm × 50 mm 64×64 grid).
    """
    return tuple(max(4, int(g) // 4) for g in grid_shape)


def _build_grid_coords(
    grid_coords: Sequence[Sequence[float]] | None,
    grid_shape: Tuple[int, ...],
) -> Optional[torch.Tensor]:
    """Stack per-axis grid coords into a ``(*grid_shape, n_dim)`` tensor.

    Returns ``None`` if ``grid_coords`` is None.
    """
    if grid_coords is None:
        return None
    if len(grid_coords) != len(grid_shape):
        raise ValueError(
            f"grid_coords has {len(grid_coords)} axes; "
            f"grid_shape has {len(grid_shape)} axes."
        )
    arrays = []
    for axis, axis_coords in enumerate(grid_coords):
        arr = np.asarray(axis_coords, dtype=np.float32).reshape(-1)
        if arr.size != grid_shape[axis]:
            raise ValueError(
                f"grid_coords[{axis}] length {arr.size} != "
                f"grid_shape[{axis}]={grid_shape[axis]}"
            )
        arrays.append(arr)
    # meshgrid with indexing='ij' so axis i has shape grid_shape[i].
    mesh = np.meshgrid(*arrays, indexing="ij")
    stacked = np.stack(mesh, axis=-1)  # (*grid_shape, n_dim)
    return torch.from_numpy(stacked.astype(np.float32))


class PhaseConditionedFNO(nn.Module):
    """Forward surrogate: ``phases (B, n_transducers) -> field (B, 2, *grid)``.

    Parameters
    ----------
    field_dims
        Spatial dimension count (2 or 3). Must equal ``len(grid_shape)``.
    grid_shape
        Per-axis resolution, e.g. ``(64, 64)`` for the 2D Phase 1 v3 dataset.
    n_transducers
        Phase-vector length. 120 for the L1 array (10 rings × 12).
    n_modes
        Fourier modes retained per axis. Defaults to ``g // 4`` per axis.
        Each entry must satisfy Nyquist: ``n_modes[i] < grid_shape[i] // 2``.
    hidden_channels
        Internal FNO width. 64 by default.
    cond_channels
        Output width of the phase-lifting MLP; broadcasts to FNO input.
    out_channels
        Output channel count. 2 for Re/Im split (do NOT change without also
        revisiting ``complex_data`` and the Phase 1 schema).
    transducer_positions
        Optional ``(n_transducers, 3)`` array of physical transducer
        positions in meters. If provided along with ``grid_coords`` and
        ``wavenumber``, the model adds 2 physical-prior channels (Re/Im
        of the analytical free-field pressure) to the FNO input.
    grid_coords
        Optional sequence of per-axis coordinate arrays (m). Length =
        ``field_dims``. For 2D: ``[x_coords, y_coords]``. For 3D:
        ``[x, y, z]``. Used together with ``transducer_positions``.
    wavenumber
        Optional acoustic wavenumber ``k = 2π f / c`` (rad / m). 40 kHz
        in air at 343 m/s ≈ 732.6 rad/m.
    slice_z
        Optional fixed z-plane (m) at which the 2D field is computed.
        For 3D grids, set ``None`` and pass ``z`` as the third
        ``grid_coords`` axis. For Phase 1 v3 the z varies per-sample so
        this physical-prior augmentation uses the median z (0.09 m for
        v3) — model learns z-residual from the lifted phase signal.
    residual_prior
        v2 flag. When True (and the prior is enabled), forward returns
        ``normalize(prior) + fno_output`` instead of ``fno_output``.
        Requires ``prior_norm_mean`` and ``prior_norm_std`` to live in
        the same coordinate space as the training loss (which is
        normalized by the adapter).
    prior_norm_mean
        Per-channel mean used to normalize the prior at output time. Must
        be shape ``(out_channels,)``. Required when ``residual_prior=True``.
    prior_norm_std
        Per-channel std used to normalize the prior at output time. Same
        shape as ``prior_norm_mean``. Floored at ``1e-6`` to avoid div-by-zero.
    prior_version
        ``'v2'`` (default; legacy v2 broken prior — fixed slice_z, sign
        ``-k·r``, amplitude ``1/r``, no directivity), ``'v2.1'`` (the
        full physical formula matching ``compute_pressure_field``: per-
        sample ``slice_z`` from x[..., -1], sign ``+k·r``, amplitude
        ``p_ref·r_ref·D(θ)/r``, piston-source directivity; 2D only), or
        ``'v2.1_3d'`` (the same four corrections applied at every voxel
        of a 3D ``(Nx, Ny, Nz)`` grid; no slice_z packing because the
        field is computed everywhere from the grid's own z-axis — see
        ``research/3D_RETRY.md``). Selects which prior is computed in
        ``_physical_prior``. v2.1 expects the input ``x`` to have shape
        ``(B, n_transducers + 1)`` with the per-sample ``slice_z`` packed
        as the last column; v2.1_3d expects ``(B, n_transducers)`` only.
        Both preserve the ``forward(x)`` single-arg contract for
        ``neuralop.Trainer``.
    p_ref, r_ref, aperture_radius
        Physical transducer parameters (Pa, m, m). Required only for the
        v2.1 / v2.1_3d priors. Defaults match ``MA40S4S_40KHZ`` config
        used by ``generate_dataset.py``: ``p_ref=10.0``, ``r_ref=0.3``,
        ``aperture_radius=2.5e-3``. The v2 prior ignores these.

    Notes
    -----
    The physical-prior channels are computed at construction time
    using ``transducer_positions``, ``grid_coords``, and
    ``wavenumber``; they cache a per-pixel, per-transducer "k·r" tensor
    on a registered buffer and only the per-sample phase φ_i mutates
    each forward.
    """

    def __init__(
        self,
        *,
        field_dims: int,
        grid_shape: Tuple[int, ...],
        n_transducers: int,
        n_modes: Optional[Tuple[int, ...]] = None,
        hidden_channels: int = 64,
        cond_channels: int = 32,
        out_channels: int = 2,
        n_layers: int = 4,
        transducer_positions: Optional[np.ndarray] = None,
        grid_coords: Optional[Sequence[Sequence[float]]] = None,
        wavenumber: Optional[float] = None,
        slice_z: Optional[float] = None,
        residual_prior: bool = False,
        prior_norm_mean: Optional[np.ndarray] = None,
        prior_norm_std: Optional[np.ndarray] = None,
        prior_version: str = "v2",
        p_ref: float = 10.0,
        r_ref: float = 0.3,
        aperture_radius: float = 2.5e-3,
        prior_scale_factor: float = 1.0,
    ) -> None:
        super().__init__()

        if int(field_dims) != len(grid_shape):
            raise ValueError(
                f"field_dims ({field_dims}) must equal len(grid_shape) "
                f"({len(grid_shape)})"
            )
        if any(int(g) <= 0 for g in grid_shape):
            raise ValueError(f"grid_shape must be positive: {grid_shape}")
        if int(n_transducers) <= 0:
            raise ValueError(f"n_transducers must be positive: {n_transducers}")

        modes = _default_modes(grid_shape) if n_modes is None else tuple(int(m) for m in n_modes)
        if len(modes) != len(grid_shape):
            raise ValueError(
                f"n_modes length {len(modes)} != grid_shape length "
                f"{len(grid_shape)}"
            )
        # Nyquist guard — silent garbage if violated.
        for axis_idx, (m, g) in enumerate(zip(modes, grid_shape)):
            if m >= g // 2:
                raise ValueError(
                    f"n_modes[{axis_idx}]={m} violates Nyquist for "
                    f"grid_shape[{axis_idx}]={g} (must be < {g // 2})"
                )
            if m < 1:
                raise ValueError(
                    f"n_modes[{axis_idx}]={m} must be >= 1"
                )

        if int(n_layers) < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.field_dims: int = int(field_dims)
        self.grid_shape: Tuple[int, ...] = tuple(int(g) for g in grid_shape)
        self.n_transducers: int = int(n_transducers)
        self.n_modes: Tuple[int, ...] = modes
        self.hidden_channels: int = int(hidden_channels)
        self.cond_channels: int = int(cond_channels)
        self.out_channels: int = int(out_channels)
        self.n_layers: int = int(n_layers)

        # Decide whether physical prior is enabled.
        prior_enabled = (
            transducer_positions is not None
            and grid_coords is not None
            and wavenumber is not None
        )
        self._prior_enabled: bool = bool(prior_enabled)

        # ``prior_version`` selects which analytical prior is computed.
        #   - 'v2'  : the legacy broken prior. Fixed slice_z (default 0.09 m),
        #             sign -k·r, amplitude 1/r, no directivity, no p_ref·r_ref.
        #             See research/V2_PRIOR_ATTRIBUTION.md for the audit.
        #   - 'v2.1': the full formula matching compute_pressure_field. Uses
        #             per-sample slice_z packed as x[..., -1], sign +k·r,
        #             amplitude p_ref·r_ref·D(θ)/r, piston-source directivity.
        if prior_version not in ("v2", "v2.1", "v2.1_3d"):
            raise ValueError(
                f"prior_version must be 'v2', 'v2.1' or 'v2.1_3d', "
                f"got {prior_version!r}"
            )
        self._prior_version: str = prior_version

        if prior_enabled:
            tx = np.asarray(transducer_positions, dtype=np.float32)
            if tx.ndim != 2 or tx.shape[1] != 3:
                raise ValueError(
                    f"transducer_positions must be (n_transducers, 3), got {tx.shape}"
                )
            if tx.shape[0] != self.n_transducers:
                raise ValueError(
                    f"transducer_positions has {tx.shape[0]} rows; expected "
                    f"{self.n_transducers}"
                )
            grid_arrays = [np.asarray(c, dtype=np.float32).reshape(-1) for c in grid_coords]
            mesh = np.meshgrid(*grid_arrays, indexing="ij")
            grid_xy = np.stack(mesh, axis=-1)  # (*grid_shape, field_dims)

            if self._prior_version == "v2":
                # Legacy path: precompute kr and inv_r at a fixed slice_z.
                if self.field_dims == 2:
                    if slice_z is None:
                        slice_z_val = 0.09  # v3 median; safe default
                    else:
                        slice_z_val = float(slice_z)
                    z_field = np.full(
                        grid_xy.shape[:-1] + (1,), slice_z_val, dtype=np.float32
                    )
                    grid_xyz = np.concatenate([grid_xy, z_field], axis=-1)
                elif self.field_dims == 3:
                    grid_xyz = grid_xy
                else:
                    raise ValueError(f"unsupported field_dims for prior: {self.field_dims}")

                diff = grid_xyz[..., None, :] - tx[None, ..., :]  # (*grid, n_tx, 3)
                r = np.linalg.norm(diff, axis=-1)  # (*grid, n_tx)
                kr = (float(wavenumber) * r).astype(np.float32)
                kr_t = torch.from_numpy(np.moveaxis(kr, -1, 0).copy())  # (n_tx, *grid)
                self.register_buffer("kr", kr_t)
                weight_t = torch.from_numpy(
                    (1.0 / np.clip(r, 1e-3, None)).astype(np.float32)
                )
                weight_t = weight_t.movedim(-1, 0)  # (n_tx, *grid)
                self.register_buffer("inv_r", weight_t)
            elif self._prior_version == "v2.1":
                # v2.1 path: precompute the per-pixel xy grid and per-transducer
                # geometry; the per-sample slice_z arrives via x[..., -1] and
                # the full prior is computed in ``_physical_prior_v21`` each
                # forward (still vectorized; cost ~O(B·n_tx·grid)).
                if self.field_dims != 2:
                    raise ValueError(
                        "prior_version='v2.1' currently supports field_dims=2 only "
                        f"(got {self.field_dims}). For 3D use 'v2.1_3d' instead."
                    )
                # grid_xy shape: (Nx, Ny, 2). Register as buffer so .to(device) moves it.
                grid_xy_t = torch.from_numpy(grid_xy.astype(np.float32))
                self.register_buffer("_grid_xy", grid_xy_t)
                # Transducer positions: (n_tx, 3) — keep as buffer for device moves.
                self.register_buffer(
                    "_tx_positions", torch.from_numpy(tx.astype(np.float32))
                )
                # Cylindrical-array normals: pointing inward toward the z-axis.
                # Matches drip_physics/pressure.py:173 ``compute_transducer_normals``.
                xy_dist = np.sqrt(tx[:, 0] ** 2 + tx[:, 1] ** 2)
                xy_dist = np.maximum(xy_dist, 1e-10)
                normals = np.zeros_like(tx)
                normals[:, 0] = -tx[:, 0] / xy_dist
                normals[:, 1] = -tx[:, 1] / xy_dist
                normals[:, 2] = 0.0
                self.register_buffer(
                    "_tx_normals", torch.from_numpy(normals.astype(np.float32))
                )
                # Scalars stored as 0-d float buffers so they move with .to(device).
                self.register_buffer(
                    "_wavenumber", torch.tensor(float(wavenumber), dtype=torch.float32)
                )
                self.register_buffer(
                    "_p_ref", torch.tensor(float(p_ref), dtype=torch.float32)
                )
                self.register_buffer(
                    "_r_ref", torch.tensor(float(r_ref), dtype=torch.float32)
                )
                self.register_buffer(
                    "_aperture", torch.tensor(float(aperture_radius), dtype=torch.float32)
                )
            else:
                # v2.1_3d path: 3D extension of the corrected v2.1 prior.
                # The four bug fixes from V2_PRIOR_ATTRIBUTION (per-sample slice_z,
                # +k·r sign, p_ref·r_ref·D(θ)/r amplitude, sinc directivity) are
                # all per-voxel scalar/elementwise ops, so they generalize directly
                # to the (Nx, Ny, Nz) grid. The slice_z packing is unnecessary in
                # 3D because z is one of the grid axes — every voxel already has
                # its own (x, y, z). Cost ~O(B·n_tx·Nx·Ny·Nz); precompute the
                # static (per-pixel, per-tx) geometry once and only the per-sample
                # phase phi_i mutates each forward (mirrors the v2 caching trick).
                if self.field_dims != 3:
                    raise ValueError(
                        "prior_version='v2.1_3d' requires field_dims=3 "
                        f"(got {self.field_dims}). For 2D use 'v2.1' instead."
                    )
                # grid_xyz shape: (Nx, Ny, Nz, 3). The 3D meshgrid stack already
                # contains the full coordinate at every voxel.
                grid_xyz_t = torch.from_numpy(grid_xy.astype(np.float32))
                # diff: (Nx, Ny, Nz, n_tx, 3) — one vector per (voxel, transducer).
                diff_np = (
                    grid_xy[..., None, :].astype(np.float32) - tx[None, None, None, :, :]
                )
                r_np = np.linalg.norm(diff_np, axis=-1).astype(np.float32)
                r_np = np.maximum(r_np, 1e-6)
                # Cylindrical inward normals (z-axis ring), same as v2.1.
                xy_dist = np.sqrt(tx[:, 0] ** 2 + tx[:, 1] ** 2)
                xy_dist = np.maximum(xy_dist, 1e-10)
                normals_np = np.zeros_like(tx)
                normals_np[:, 0] = -tx[:, 0] / xy_dist
                normals_np[:, 1] = -tx[:, 1] / xy_dist
                normals_np[:, 2] = 0.0
                # Directivity D(θ) = |sinc(a k sin θ / 2)|; theta from inward
                # normal and the line to the field point. cos θ via dot product.
                cos_theta = np.sum(
                    (diff_np / r_np[..., None]) * normals_np[None, None, None, :, :],
                    axis=-1,
                )
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                sin_theta = np.sqrt(np.clip(1.0 - cos_theta ** 2, a_min=0.0, a_max=None))
                directivity_arg = (
                    float(aperture_radius) * float(wavenumber) * sin_theta / 2.0
                )
                small = np.abs(directivity_arg) < 1e-10
                safe_arg = np.where(small, 1.0, directivity_arg)
                directivity_np = np.where(
                    small, 1.0, np.sin(safe_arg) / safe_arg,
                ).astype(np.float32)
                directivity_np = np.abs(directivity_np)
                # Combined per-voxel-per-tx coefficients used in forward.
                # amplitude = p_ref · r_ref · D(θ) / r          (scalar magnitude)
                # kr        = k · r                              (used in cos/sin)
                # Both depend only on geometry — phases enter only via phi_i.
                amplitude_np = (
                    float(p_ref) * float(r_ref) * directivity_np / r_np
                ).astype(np.float32)
                kr_np = (float(wavenumber) * r_np).astype(np.float32)
                # Move tx axis to position 0 so broadcast against (B, n_tx, ...) is cheap.
                amplitude_t = torch.from_numpy(
                    np.moveaxis(amplitude_np, -1, 0).copy()
                )  # (n_tx, Nx, Ny, Nz)
                kr_t = torch.from_numpy(np.moveaxis(kr_np, -1, 0).copy())
                self.register_buffer("_amplitude_3d", amplitude_t)
                self.register_buffer("_kr_3d", kr_t)
                # Keep the wavenumber + raw geometry around for downstream
                # introspection / receipts (no autograd dependency).
                self.register_buffer(
                    "_wavenumber", torch.tensor(float(wavenumber), dtype=torch.float32)
                )
                self.register_buffer(
                    "_p_ref", torch.tensor(float(p_ref), dtype=torch.float32)
                )
                self.register_buffer(
                    "_r_ref", torch.tensor(float(r_ref), dtype=torch.float32)
                )
                self.register_buffer(
                    "_aperture", torch.tensor(float(aperture_radius), dtype=torch.float32)
                )
                self.register_buffer(
                    "_tx_positions", torch.from_numpy(tx.astype(np.float32))
                )
                self.register_buffer(
                    "_tx_normals", torch.from_numpy(normals_np.astype(np.float32))
                )

        # ---------- v2 residual-prior wiring ----------
        # The residual mode requires the prior; raise loudly if mismatched.
        if residual_prior and not prior_enabled:
            raise ValueError(
                "residual_prior=True requires the analytical prior "
                "(transducer_positions, grid_coords, wavenumber)."
            )
        self._residual_prior: bool = bool(residual_prior)
        if self._residual_prior:
            if prior_norm_mean is None or prior_norm_std is None:
                raise ValueError(
                    "residual_prior=True requires prior_norm_mean and "
                    "prior_norm_std (so the prior lives in the same "
                    "normalized coordinate space as the training loss)."
                )
            mean_arr = np.asarray(prior_norm_mean, dtype=np.float32).reshape(-1)
            std_arr = np.asarray(prior_norm_std, dtype=np.float32).reshape(-1)
            if mean_arr.shape != (int(out_channels),):
                raise ValueError(
                    f"prior_norm_mean must have shape ({int(out_channels)},), "
                    f"got {mean_arr.shape}"
                )
            if std_arr.shape != (int(out_channels),):
                raise ValueError(
                    f"prior_norm_std must have shape ({int(out_channels)},), "
                    f"got {std_arr.shape}"
                )
            std_arr = np.maximum(std_arr, 1e-6).astype(np.float32)
            self.register_buffer("_prior_mean", torch.from_numpy(mean_arr))
            self.register_buffer("_prior_std", torch.from_numpy(std_arr))

        # Scaling factor for the residual prior. Default 1.0 preserves the
        # original residual_prior=True behavior. Lower values down-weight
        # the prior so it doesn't dominate when the target is in a much
        # smaller magnitude regime (e.g. FEM-coupled chamber fields at
        # ~1 Pa with a free-field analytical prior at ~hundreds of Pa).
        self.register_buffer(
            "_prior_scale_factor",
            torch.tensor(float(prior_scale_factor), dtype=torch.float32),
        )

        # Pattern A: small MLP lifts phases to cond_channels.
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.n_transducers, 128),
            nn.GELU(),
            nn.Linear(128, self.cond_channels),
        )

        # Inner FNO. complex_data=False; we represent complex fields as
        # 2 real channels per orchestrator decision.
        # If physical prior is enabled, FNO sees cond_channels + 2 channels.
        in_channels_total = self.cond_channels + (2 if self._prior_enabled else 0)
        self.fno = FNO(
            n_modes=self.n_modes,
            in_channels=in_channels_total,
            out_channels=self.out_channels,
            hidden_channels=self.hidden_channels,
            n_layers=self.n_layers,
            complex_data=False,
        )

        # One-shot guard flag. After the first forward we stop wrapping
        # in ``warnings.catch_warnings`` (cheap, but unnecessary
        # per-call overhead during training).
        self._warned_once: bool = False

    @property
    def prior_enabled(self) -> bool:
        return self._prior_enabled

    @property
    def residual_prior(self) -> bool:
        return self._residual_prior

    @property
    def prior_version(self) -> str:
        return self._prior_version

    @property
    def expected_input_dim(self) -> int:
        """Trailing dim of ``x`` expected by ``forward``.

        v2: ``n_transducers``. v2.1: ``n_transducers + 1`` (the last
        column carries per-sample ``slice_z`` in meters). v2.1_3d:
        ``n_transducers`` (no slice_z packing — z is a grid axis, every
        voxel already has its own coordinate). The adapter must produce
        ``x`` matching this width.
        """
        if self._prior_version == "v2.1":
            return self.n_transducers + 1
        return self.n_transducers

    def _normalize_prior(self, prior: torch.Tensor) -> torch.Tensor:
        """Apply ``(prior - mean) / std`` per channel.

        Buffers are shape ``(out_channels,)``; broadcast to
        ``(1, C, *grid_shape)`` so the spatial dims are size-1 broadcast.
        """
        view = (None, slice(None)) + (None,) * len(self.grid_shape)
        m = self._prior_mean[view]  # type: ignore[index]
        s = self._prior_std[view]  # type: ignore[index]
        return (prior - m) / s

    def _broadcast_cond(self, z: torch.Tensor) -> torch.Tensor:
        """Broadcast ``(B, C)`` lifted vector to ``(B, C, *grid_shape)``.

        Dimension-agnostic.
        """
        view_index = (slice(None), slice(None)) + (None,) * len(self.grid_shape)
        target_shape = (z.shape[0], z.shape[1], *self.grid_shape)
        return z[view_index].expand(*target_shape).contiguous()

    def _physical_prior(self, phases: torch.Tensor) -> torch.Tensor:
        """Legacy v2 free-field prior — BROKEN (kept for reproducibility).

        Returns ``(B, 2, *grid_shape)`` with Re and Im of
        ``sum_i (1/r_i) e^{j (phi_i - k r_i)}``.

        See ``research/V2_PRIOR_ATTRIBUTION.md`` for the four defects:
            1. ``slice_z`` hardcoded to 0.09 m at __init__ time.
            2. Sign on ``k·r`` flipped (``phi - k·r`` vs ``+k·r``).
            3. Amplitude is ``1/r`` instead of ``p_ref·r_ref·D(θ)/r``.
            4. Piston-source directivity ``D(θ)`` missing.

        For the corrected version see ``_physical_prior_v21``.
        """
        # phases: (B, n_tx). kr/inv_r buffer: (n_tx, *grid).
        view = (slice(None), slice(None)) + (None,) * len(self.grid_shape)
        phi = phases[view]  # (B, n_tx, 1, ...)
        kr = self.kr.unsqueeze(0)  # (1, n_tx, *grid)
        inv_r = self.inv_r.unsqueeze(0)  # (1, n_tx, *grid)
        arg = phi - kr  # (B, n_tx, *grid)
        re = (inv_r * torch.cos(arg)).sum(dim=1)
        im = (inv_r * torch.sin(arg)).sum(dim=1)
        return torch.stack([re, im], dim=1)

    def _physical_prior_v21(
        self,
        phases: torch.Tensor,
        slice_z: torch.Tensor,
    ) -> torch.Tensor:
        """Corrected free-field prior matching ``compute_pressure_field``.

        Implements the full formula from ``drip_physics/pressure.py:457-473``:

            P(r) = sum_i [p_ref · r_ref · D(θ_i) / r_i] · exp(j (k·r_i + φ_i))

        with per-sample ``slice_z`` (audit fix #1), sign ``+k·r`` (#2),
        scale ``p_ref · r_ref`` (#3), and piston directivity (#4). Pure
        PyTorch ops so the prior is autograd-friendly through ``phases``
        for future Phase 3 inverse use; ``slice_z`` is treated as a static
        per-sample scalar.

        Parameters
        ----------
        phases : (B, n_tx) float32
            Per-transducer phase commands (rad).
        slice_z : (B,) float32
            Per-sample 2D slice z-coordinate (m).

        Returns
        -------
        (B, 2, Nx, Ny) float32 — Re and Im of the complex pressure field.

        Notes
        -----
        Cost: O(B · n_tx · Nx · Ny). For B=32, n_tx=120, 64×64 grid this
        is ~16M ops/forward — cheap relative to the FNO. Acoustic
        attenuation (Bass et al. 1995, ~0.12 Np/m at 40 kHz, 20°C) is
        omitted: ~1.2% effect over 0.1 m, well inside the audit's
        residual ratio of 0.0036. Add later if precision matters.
        """
        # ---------- v2.1 FIX (gap 1): per-sample slice_z ----------
        # Build (B, Nx, Ny, 3) field point cloud from the cached xy grid
        # plus the per-sample slice_z. Each sample gets its own z-plane.
        B = phases.shape[0]
        Nx, Ny = int(self.grid_shape[0]), int(self.grid_shape[1])
        # _grid_xy: (Nx, Ny, 2). Broadcast across batch.
        grid_xy = self._grid_xy.unsqueeze(0).expand(B, Nx, Ny, 2)  # (B, Nx, Ny, 2)
        z = slice_z.view(B, 1, 1, 1).expand(B, Nx, Ny, 1)  # (B, Nx, Ny, 1)
        field_pts = torch.cat([grid_xy, z], dim=-1)  # (B, Nx, Ny, 3)

        # Transducer positions: (n_tx, 3) -> broadcast to (1, 1, 1, n_tx, 3).
        tx = self._tx_positions.view(1, 1, 1, self.n_transducers, 3)
        # field_pts: (B, Nx, Ny, 3) -> (B, Nx, Ny, 1, 3)
        diff = field_pts.unsqueeze(-2) - tx  # (B, Nx, Ny, n_tx, 3)
        r = diff.norm(dim=-1)  # (B, Nx, Ny, n_tx)
        r = torch.clamp(r, min=1e-6)  # avoid divide-by-zero

        # ---------- v2.1 FIX (gap 4): piston-source directivity ----------
        # cos(θ) = normal · (diff / r); θ is angle between transducer normal
        # and the line to the field point. Then D(θ) = sinc_unnormalized
        # (a · k · sin(θ) / 2) per ``transducer_directivity`` in pressure.py.
        normals = self._tx_normals.view(1, 1, 1, self.n_transducers, 3)
        cos_theta = (diff * normals).sum(dim=-1) / r  # (B, Nx, Ny, n_tx)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        # sin(θ) = sqrt(1 - cos²θ); avoids arccos for differentiability.
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta * cos_theta, min=0.0))
        # arg = a * k * sin(θ) / 2  per pressure.py:160.
        directivity_arg = self._aperture * self._wavenumber * sin_theta / 2.0
        # sinc_unnormalized: sin(x)/x with the x→0 limit handled.
        small = directivity_arg.abs() < 1e-10
        safe_arg = torch.where(small, torch.ones_like(directivity_arg), directivity_arg)
        directivity = torch.where(
            small,
            torch.ones_like(directivity_arg),
            torch.sin(safe_arg) / safe_arg,
        )
        directivity = directivity.abs()  # match pressure.py:170

        # ---------- v2.1 FIX (gap 3): p_ref · r_ref amplitude ----------
        # Full physical scale matches ``compute_pressure_field`` line 461-466.
        amplitude = self._p_ref * self._r_ref * directivity / r  # (B, Nx, Ny, n_tx)

        # ---------- v2.1 FIX (gap 2): sign +k·r (was -k·r) ----------
        # phases: (B, n_tx) -> (B, 1, 1, n_tx).
        phi = phases.view(B, 1, 1, self.n_transducers)
        phase_arg = self._wavenumber * r + phi  # +k·r matches pressure.py:457

        # Sum complex phasors over the n_tx axis.
        re = (amplitude * torch.cos(phase_arg)).sum(dim=-1)  # (B, Nx, Ny)
        im = (amplitude * torch.sin(phase_arg)).sum(dim=-1)
        return torch.stack([re, im], dim=1)  # (B, 2, Nx, Ny)

    def _physical_prior_v21_3d(self, phases: torch.Tensor) -> torch.Tensor:
        """3D extension of the corrected v2.1 prior — full free-field formula
        evaluated at every voxel of the ``(Nx, Ny, Nz)`` grid.

        Implements the same formula as ``_physical_prior_v21`` but with z as
        a grid axis instead of a per-sample scalar. The four bug fixes from
        ``research/V2_PRIOR_ATTRIBUTION.md`` (per-sample slice_z, +k·r sign,
        p_ref·r_ref·D(θ)/r amplitude, sinc directivity) all generalize
        directly: they are scalar/per-pixel operations and don't depend on
        the field's dimensionality.

        Implementation note (slice_z packing):
            The 2D v2.1 path packs per-sample slice_z as ``x[:, -1]`` because
            the 2D field is *one* z-plane that varies per sample. In 3D,
            every voxel has its own (x, y, z) directly from the grid axes,
            so there is no per-sample z to thread through. ``x`` is plain
            ``(B, n_transducers)`` and ``slice_z`` does not appear.

        Performance: the per-pixel-per-tx geometry (``amplitude``, ``kr``)
        is precomputed at construction and registered as a static buffer.
        Only the per-sample phase φ_i mutates each forward, mirroring the
        v2 caching strategy. Forward cost is dominated by two trig
        evaluations and one sum over the n_tx axis at every voxel.

        Parameters
        ----------
        phases : (B, n_tx) float32
            Per-transducer phase commands (rad).

        Returns
        -------
        (B, 2, Nx, Ny, Nz) float32 — Re and Im of the complex pressure.
        """
        B = phases.shape[0]
        # Precomputed buffers: (n_tx, Nx, Ny, Nz). Add the leading batch axis.
        amplitude = self._amplitude_3d.unsqueeze(0)  # (1, n_tx, Nx, Ny, Nz)
        kr = self._kr_3d.unsqueeze(0)                # (1, n_tx, Nx, Ny, Nz)
        # Phases broadcast over the spatial dims:
        # phases (B, n_tx) -> (B, n_tx, 1, 1, 1).
        view = (slice(None), slice(None)) + (None,) * len(self.grid_shape)
        phi = phases[view]
        phase_arg = kr + phi  # +k·r matches pressure.py (gap 2 fix)
        re = (amplitude * torch.cos(phase_arg)).sum(dim=1)  # (B, Nx, Ny, Nz)
        im = (amplitude * torch.sin(phase_arg)).sum(dim=1)
        return torch.stack([re, im], dim=1)  # (B, 2, Nx, Ny, Nz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the complex pressure field for a batch of phase configs.

        Parameters
        ----------
        x
            Phase configurations, shape ``(B, n_transducers)`` for v2 or
            ``(B, n_transducers + 1)`` for v2.1 with the per-sample
            ``slice_z`` (m) packed as the last column. Named ``x`` to
            satisfy the ``neuralop.Trainer`` ``model(**sample)`` contract —
            do NOT rename.

        Returns
        -------
        torch.Tensor
            Predicted field, shape ``(B, out_channels, *grid_shape)``,
            float32. Channel 0 = Re, channel 1 = Im.
        """
        if x.ndim != 2:
            raise ValueError(
                f"forward expects (B, expected_input_dim); got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.expected_input_dim:
            raise ValueError(
                f"forward expects {self.expected_input_dim} input features "
                f"(prior_version={self._prior_version!r}, "
                f"n_transducers={self.n_transducers}), got {x.shape[1]}"
            )

        # Split phases / slice_z. v2.1 packs slice_z as x[:, -1]; v2 and
        # v2.1_3d use plain (B, n_transducers) input.
        if self._prior_version == "v2.1":
            phases = x[:, : self.n_transducers]
            slice_z_batch = x[:, -1]  # (B,)
        else:
            phases = x
            slice_z_batch = None

        z = self.cond_mlp(phases)                 # (B, cond_channels)
        spatial = self._broadcast_cond(z)         # (B, cond_channels, *grid)

        prior: Optional[torch.Tensor] = None
        if self._prior_enabled:
            if self._prior_version == "v2.1":
                assert slice_z_batch is not None
                prior = self._physical_prior_v21(phases, slice_z_batch)
            elif self._prior_version == "v2.1_3d":
                prior = self._physical_prior_v21_3d(phases)
            else:
                prior = self._physical_prior(phases)
            spatial = torch.cat([spatial, prior], dim=1)

        if not self._warned_once:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                y = self.fno(spatial)
            self._warned_once = True
        else:
            y = self.fno(spatial)

        # v2 residual path: add the *normalized* prior to the FNO output so
        # the model defaults to the analytical free-field solution and only
        # needs to learn a small correction. The loss is computed in
        # normalized space by the adapter, so the prior must be normalized
        # too — using the same per-channel (mean, std) the adapter uses.
        if self._residual_prior:
            assert prior is not None  # enforced by __init__ guards
            # `prior_scale_factor` lets us down-weight the prior when its
            # magnitude is in a different regime than the target (e.g.
            # FEM-coupled targets at ~1 Pa vs free-field analytical prior
            # at ~hundreds of Pa). Default 1.0 = original behavior.
            y = self._prior_scale_factor * self._normalize_prior(prior) + y

        return y
