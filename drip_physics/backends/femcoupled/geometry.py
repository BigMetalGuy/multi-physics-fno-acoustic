"""Geometry helpers for the FEM-coupled-physics chamber mesh.

This module defines:

  * :class:`ChamberGeometry` — a frozen dataclass that resolves the L1
    chamber dimensions out of an :class:`ArrayConfig` /
    :class:`TransducerConfig` pair into concrete cylinder and transducer
    placement parameters.
  * :func:`compute_geometry_hash` — deterministic short-hex digest used
    to key the persisted ``.msh`` cache (FEM_V2_DESIGN §4 step 1, Q5
    resolution in FEM_DISCOVERY_REPORT §6).

No GMSH calls live here — see :mod:`mesh`. This separation lets unit
tests verify geometry math without bringing up the GMSH state machine.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from drip_physics.config import ArrayConfig, TransducerConfig

# Vertical clearance above the top transducer ring and below the bottom
# ring. ~λ/4 at 40 kHz in air (8.575 mm wavelength → ~2 mm), giving the
# Helmholtz solver enough room for the radiating boundary terms before
# the lid / substrate are encountered.
TRANSDUCER_VERTICAL_PADDING_M: float = 0.005  # 5 mm

# Maximum allowed transducer face radius relative to ring spacing. Faces
# would otherwise overlap vertically; we error rather than silently
# truncate.
MAX_FACE_DIAMETER_RATIO: float = 0.95


@dataclass(frozen=True)
class ChamberGeometry:
    """Resolved chamber geometry derived from config objects.

    All distances are SI (metres). Coordinates use the same convention
    as ``drip_physics.geometry``: chamber axis is z, substrate at
    ``z=0``, lid at ``z=height``.
    """

    cylinder_radius: float                      # m
    height: float                                # m
    n_rings: int
    n_per_ring: int
    ring_spacing: float                          # m
    bottom_ring_z: float                         # m, lowest ring centre z
    transducer_face_radius: float                # m (aperture half-width)

    @property
    def total_transducers(self) -> int:
        return self.n_rings * self.n_per_ring

    @property
    def transducer_face_diameter(self) -> float:
        return 2.0 * self.transducer_face_radius


def resolve_chamber_geometry(
    array: ArrayConfig,
    transducer: TransducerConfig,
    vertical_padding_m: float = TRANSDUCER_VERTICAL_PADDING_M,
) -> ChamberGeometry:
    """Resolve the L1 chamber geometry from ``ArrayConfig`` + ``TransducerConfig``.

    Height is the ring stack height plus padding on top and bottom; the
    bottom ring sits ``vertical_padding_m`` above the substrate so the
    transducer disks have headroom for finite-size piston BCs.
    """

    if vertical_padding_m <= 0:
        raise ValueError(
            f"vertical_padding_m must be positive, got {vertical_padding_m}"
        )
    face_diameter = 2.0 * transducer.aperture_radius
    if face_diameter > MAX_FACE_DIAMETER_RATIO * array.ring_spacing:
        raise ValueError(
            f"Transducer face diameter ({face_diameter*1e3:.2f} mm) exceeds "
            f"{MAX_FACE_DIAMETER_RATIO*100:.0f}% of ring spacing "
            f"({array.ring_spacing*1e3:.2f} mm) — faces would overlap."
        )

    stack_height = (array.ring_count - 1) * array.ring_spacing
    height = stack_height + 2.0 * vertical_padding_m

    return ChamberGeometry(
        cylinder_radius=float(array.cylinder_radius),
        height=float(height),
        n_rings=int(array.ring_count),
        n_per_ring=int(array.transducers_per_ring),
        ring_spacing=float(array.ring_spacing),
        bottom_ring_z=float(vertical_padding_m),
        transducer_face_radius=float(transducer.aperture_radius),
    )


def transducer_centers(geom: ChamberGeometry) -> np.ndarray:
    """Return the (N, 3) array of transducer face centre coordinates.

    Index ordering is row-major over (ring, slot): index 0 is ring 0
    slot 0, index 1 is ring 0 slot 1, …, index ``n_per_ring-1`` is ring
    0's last slot, index ``n_per_ring`` is ring 1 slot 0. The same
    ordering keys the per-face physical groups in :mod:`mesh` so phase
    commands map directly to the array index.
    """
    centers = np.empty((geom.total_transducers, 3), dtype=np.float64)
    idx = 0
    # Bottom ring at z = bottom_ring_z; subsequent rings every ring_spacing.
    for ring in range(geom.n_rings):
        z = geom.bottom_ring_z + ring * geom.ring_spacing
        for slot in range(geom.n_per_ring):
            theta = 2.0 * np.pi * slot / geom.n_per_ring
            x = geom.cylinder_radius * np.cos(theta)
            y = geom.cylinder_radius * np.sin(theta)
            centers[idx, 0] = x
            centers[idx, 1] = y
            centers[idx, 2] = z
            idx += 1
    return centers


def compute_geometry_hash(
    geom: ChamberGeometry,
    focus_h_mm: float,
    bulk_h_mm: float,
    *,
    crucible_inlet_radius_m: float = 0.0,
    n_chars: int = 8,
) -> str:
    """Deterministic short-hex digest keying the persisted mesh cache.

    Inputs feed into a SHA-256 with fixed-width float formatting (so
    floating-point drift in unrelated pipelines doesn't perturb the
    hash). Output is ``n_chars`` of lowercase hex.

    The ``crucible_inlet_radius_m`` parameter participates in the hash
    so that a mesh built with the new top-face split (Phase 3 follow-up,
    FEM_V2_DESIGN §3.7) gets a fresh cache key — meshes from earlier
    builds (where the top face was lumped into ``chamber_walls``)
    remain reachable under the legacy hash with ``crucible_inlet_radius_m=0``.
    """
    if n_chars <= 0 or n_chars > 64:
        raise ValueError(f"n_chars must be in [1, 64], got {n_chars}")
    base = (
        f"r={geom.cylinder_radius:.9e}|"
        f"h={geom.height:.9e}|"
        f"rs={geom.ring_spacing:.9e}|"
        f"nr={geom.n_rings}|"
        f"npr={geom.n_per_ring}|"
        f"a={geom.transducer_face_radius:.9e}|"
        f"focus={focus_h_mm:.6f}|"
        f"bulk={bulk_h_mm:.6f}"
    )
    # Only append the inlet field when non-zero so legacy mesh builds
    # (no top-face split) preserve their existing cache hash.
    if crucible_inlet_radius_m > 0.0:
        base = f"{base}|inlet={crucible_inlet_radius_m:.9e}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:n_chars]


__all__ = [
    "ChamberGeometry",
    "MAX_FACE_DIAMETER_RATIO",
    "TRANSDUCER_VERTICAL_PADDING_M",
    "compute_geometry_hash",
    "resolve_chamber_geometry",
    "transducer_centers",
]
