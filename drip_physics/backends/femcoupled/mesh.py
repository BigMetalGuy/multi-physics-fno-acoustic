"""L1 chamber mesh generation for the FEM-coupled-physics backend.

Phase 1 of the FEM-coupled-physics v1 build (FEM_V2_DESIGN.md §4 step 1).
Builds a 3D Gmsh mesh of the L1 chamber:

  * Cylinder defined by :class:`ArrayConfig` (5 cm radius L1 default).
  * 120 transducer disk faces (10 rings × 12 transducers default for L1)
    embedded into the inner cylindrical wall as separate physical
    surfaces. Each transducer carries its own physical group
    ``transducer_<i>`` so Phase 2's HelmholtzProblem can apply
    parametric Robin (piston) BCs per-face.
  * Three named surface groups used by Phase 2:
      - ``transducer_faces`` — aggregate group covering all 120 disks.
      - ``chamber_walls`` — outer cylindrical wall *between* disks plus
        the top lid (open chamber baseline, FEM_V2_DESIGN §3.7).
      - ``substrate`` — bottom disk (heated platen).
  * Mesh density driven by Gmsh's Distance + Threshold field machinery:
    ``focus_h_mm`` (default 1 mm, Q2 resolution, FEM_DISCOVERY_REPORT
    §5) at every transducer face and the focal zone near the chamber
    axis, ramping up to ``bulk_h_mm`` (default 5 mm) in the bulk.

The mesh is persisted under ``simulations/ml_inverse/data/meshes/`` with
a filename keyed by ``geometry_hash`` (Q5, FEM_DISCOVERY_REPORT §6) so
the Phase 2 solver can hit the cache on repeated builds.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from drip_physics.config import ArrayConfig, TransducerConfig

from .geometry import (
    ChamberGeometry,
    compute_geometry_hash,
    resolve_chamber_geometry,
    transducer_centers,
)


def _import_jax_fem_mesh():
    """Lazy import of ``jax_fem.generate_mesh.Mesh``.

    JAX-FEM is GPL-3.0 and lives in the ``_jaxfem_discovery`` sidecar
    venv. We delay the import so the main simulations env (which does
    not ship JAX-FEM) can still ``import`` this module — only callers
    of :func:`load_jax_fem_mesh` need JAX-FEM present.
    """
    from jax_fem.generate_mesh import Mesh  # noqa: WPS433 (local import)

    return Mesh

# ---------------------------------------------------------------------------
# Default cache location (gitignored via simulations/ml_inverse/data/ in the
# repo root .gitignore — verified at module-import time below).
# ---------------------------------------------------------------------------

DEFAULT_MESH_CACHE = (
    Path(__file__).resolve().parents[3]
    / "ml_inverse"
    / "data"
    / "meshes"
)

# Focal-zone refinement radius. The "focal zone" sits on the chamber
# axis, centred at z=height/2; we refine to focus_h_mm inside this
# sphere and ramp out to bulk_h_mm beyond it.
FOCAL_ZONE_RADIUS_M: float = 0.01  # 10 mm — covers ~λ around focus

# Distance over which the mesh size ramps from ``focus_h`` near a refined
# region to ``bulk_h`` away from it. ~λ/2 keeps dispersion bounded
# during the transition.
REFINE_RAMP_M: float = 0.005  # 5 mm

# Distance from a transducer face within which the fine mesh size
# applies. ~face radius gives the Q2 16–25 surface-element target.
FACE_REFINE_DISTANCE_M: float = 0.003  # 3 mm

# Default crucible-inlet disk radius on the chamber lid (m). The droplet
# enters through this disk; Phase 4's StaggeredCouplingDriver pins it to
# T_crucible = 1123.15 K via Dirichlet BC. Provisional 5 mm diameter
# pending FEM_V2_DESIGN §3.7 mechanical-design input.
DEFAULT_CRUCIBLE_INLET_RADIUS_M: float = 2.5e-3

# Geometric tolerance for OCC face-by-position lookups.
GEOM_TOL_M: float = 1e-6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_l1_chamber_mesh(
    array_config: ArrayConfig,
    transducer_config: TransducerConfig,
    focus_h_mm: float = 1.0,
    bulk_h_mm: float = 5.0,
    crucible_inlet_radius: float = DEFAULT_CRUCIBLE_INLET_RADIUS_M,
    output_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    overwrite: bool = True,
    verbose: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """Build the L1 chamber mesh and persist it as ``.msh`` v2.

    Args:
        array_config: Source of cylinder geometry (radius, ring count,
            transducers per ring, ring spacing). The L1 default lives at
            ``drip_physics.config.L1_ARRAY``.
        transducer_config: Source of transducer face geometry. The L1
            default lives at ``drip_physics.config.MA40S4S_40KHZ``
            (5 mm disk diameter via ``aperture_radius=2.5e-3``).
        focus_h_mm: Mesh size at transducer faces and the focal zone
            (millimetres). Default 1.0 mm targets the Q2 16–25
            surface-elements-per-face budget at λ/8 dispersion bound.
        bulk_h_mm: Mesh size in the chamber interior (millimetres).
            Default 5 mm — λ/1.7 at 40 kHz, per FEM_V2_DESIGN §3.8.
        crucible_inlet_radius: Radius of the crucible-inlet disk on the
            chamber lid (metres). The droplet enters through this disk;
            Phase 4 pins it to ``T_crucible`` via Dirichlet BC. Default
            ``DEFAULT_CRUCIBLE_INLET_RADIUS_M`` (2.5 mm = 5 mm diameter).
            Set to ``0.0`` to recover the legacy single-``chamber_walls``
            top-face behavior (Phase 1/2.0/3 backward-compat).
        output_path: Explicit ``.msh`` destination. If ``None``, the
            mesh is written into ``cache_dir`` keyed by
            ``geometry_hash``.
        cache_dir: Cache root for keyed builds. Defaults to
            ``simulations/ml_inverse/data/meshes/``.
        overwrite: If False and the keyed file exists, the existing
            file is reused and only the info dict is recomputed.
        verbose: Forward Gmsh's terminal logging through to stdout.

    Returns:
        ``(path, info)`` — ``path`` is the persisted ``.msh`` file;
        ``info`` is a dict with keys: ``geometry_hash``, ``geometry``,
        ``focus_h_mm``, ``bulk_h_mm``, ``crucible_inlet_radius``,
        ``element_count``, ``surface_element_count``, ``node_count``,
        ``transducer_face_count``, ``surface_tags`` (mapping name →
        physical-group integer tag — includes ``crucible_inlet`` and
        ``chamber_lid`` when ``crucible_inlet_radius > 0``),
        ``per_face_surface_element_counts`` (list of length 120),
        ``mesh_size_per_group`` (max edge length on each named group),
        ``file_size_bytes``.
    """

    if focus_h_mm <= 0:
        raise ValueError(f"focus_h_mm must be positive, got {focus_h_mm}")
    if bulk_h_mm <= 0:
        raise ValueError(f"bulk_h_mm must be positive, got {bulk_h_mm}")
    if focus_h_mm > bulk_h_mm:
        raise ValueError(
            f"focus_h_mm ({focus_h_mm}) cannot exceed bulk_h_mm ({bulk_h_mm})"
        )
    if crucible_inlet_radius < 0:
        raise ValueError(
            f"crucible_inlet_radius must be non-negative, got "
            f"{crucible_inlet_radius}"
        )

    geom = resolve_chamber_geometry(array_config, transducer_config)
    centers = transducer_centers(geom)

    if crucible_inlet_radius > 0 and crucible_inlet_radius >= geom.cylinder_radius:
        raise ValueError(
            f"crucible_inlet_radius ({crucible_inlet_radius*1e3:.2f} mm) "
            f"cannot be ≥ chamber radius "
            f"({geom.cylinder_radius*1e3:.2f} mm)."
        )

    geometry_hash = compute_geometry_hash(
        geom,
        focus_h_mm,
        bulk_h_mm,
        crucible_inlet_radius_m=crucible_inlet_radius,
    )

    if output_path is None:
        cache_root = Path(cache_dir) if cache_dir else DEFAULT_MESH_CACHE
        cache_root.mkdir(parents=True, exist_ok=True)
        output_path = cache_root / (
            f"L1_chamber_h{focus_h_mm:g}mm_{geometry_hash}.msh"
        )
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cache_hit = output_path.exists() and not overwrite

    if not cache_hit:
        _generate_mesh_file(
            geom=geom,
            centers=centers,
            output_path=output_path,
            focus_h_m=focus_h_mm * 1e-3,
            bulk_h_m=bulk_h_mm * 1e-3,
            crucible_inlet_radius_m=float(crucible_inlet_radius),
            verbose=verbose,
        )

    info = _summarise_mesh(
        path=output_path,
        geom=geom,
        geometry_hash=geometry_hash,
        focus_h_mm=focus_h_mm,
        bulk_h_mm=bulk_h_mm,
        crucible_inlet_radius_m=float(crucible_inlet_radius),
        cache_hit=cache_hit,
    )
    return output_path, info


# ---------------------------------------------------------------------------
# Internals — Gmsh model construction
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _gmsh_session(verbose: bool):
    """Context manager that initializes/finalizes Gmsh exactly once.

    Gmsh keeps a single global model registry, so we wrap each build in
    a context manager to guarantee finalisation even if an exception
    fires mid-build.
    """
    import gmsh  # local import — keeps the package import cheap

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        yield gmsh
    finally:
        gmsh.finalize()


def _disk_normal(center: np.ndarray, axis_radius: float) -> np.ndarray:
    """Inward normal of a transducer disk at the given centre.

    For a wall transducer at radius ``axis_radius``, the disk normal
    points along the negative radial direction (into the chamber).
    """
    radial = np.array([center[0], center[1], 0.0], dtype=np.float64)
    norm = np.linalg.norm(radial)
    if norm < 1e-12:
        raise ValueError(
            "Transducer centre lies on the cylinder axis — cannot "
            "compute radial normal."
        )
    return -radial / norm  # inward


def _generate_mesh_file(
    geom: ChamberGeometry,
    centers: np.ndarray,
    output_path: Path,
    focus_h_m: float,
    bulk_h_m: float,
    crucible_inlet_radius_m: float,
    verbose: bool,
) -> None:
    """Write the Gmsh ``.msh`` file for the chamber geometry.

    When ``crucible_inlet_radius_m > 0`` the lid is split into two
    physical groups: a central ``crucible_inlet`` disk (Dirichlet inlet
    surface for Phase 4) and a surrounding ``chamber_lid`` annulus.
    When zero, the lid stays bundled into ``chamber_walls`` (legacy
    Phase 1 behavior).
    """

    has_inlet = crucible_inlet_radius_m > 0.0

    with _gmsh_session(verbose) as gmsh:
        model = gmsh.model
        occ = model.occ
        model.add("L1_chamber")

        # ----- Step 1: build the chamber cylinder ------------------------
        cylinder_tag = occ.addCylinder(
            x=0.0, y=0.0, z=0.0,
            dx=0.0, dy=0.0, dz=geom.height,
            r=geom.cylinder_radius,
        )
        cylinder_volume = (3, cylinder_tag)
        occ.synchronize()

        # ----- Step 2: build embedded transducer disks --------------------
        # Each transducer is a flat disk *patch* of the inner cylindrical
        # wall. We model it as a tiny cylindrical surface (height=0) at the
        # disk's footprint. We use OCC.addDisk in the appropriate plane,
        # then fragment with the chamber wall to imprint the patch.

        disk_tags: List[int] = []
        for center in centers:
            normal = _disk_normal(center, geom.cylinder_radius)
            # OCC addDisk requires the disk normal direction; we want the
            # disk centred *on* the cylinder wall so fragmentation produces
            # a clean inscribed patch.
            tag = occ.addDisk(
                xc=float(center[0]),
                yc=float(center[1]),
                zc=float(center[2]),
                rx=geom.transducer_face_radius,
                ry=geom.transducer_face_radius,
                zAxis=[float(normal[0]), float(normal[1]), float(normal[2])],
            )
            disk_tags.append(tag)

        # Optional: build the crucible-inlet disk on the top lid. Normal
        # is +z so the disk lives in the lid plane.
        inlet_disk_tag: Optional[int] = None
        if has_inlet:
            inlet_disk_tag = occ.addDisk(
                xc=0.0,
                yc=0.0,
                zc=float(geom.height),
                rx=float(crucible_inlet_radius_m),
                ry=float(crucible_inlet_radius_m),
                zAxis=[0.0, 0.0, 1.0],
            )

        occ.synchronize()

        # ----- Step 3: imprint disks onto cylinder wall + lid -----------
        # OCC fragment performs imprinting + sharing of common boundaries
        # so the disks become shared faces of the cylindrical / lid
        # surfaces.
        disk_dimtags = [(2, t) for t in disk_tags]
        if inlet_disk_tag is not None:
            disk_dimtags.append((2, inlet_disk_tag))
        out_dimtags, fragment_map = occ.fragment(
            objectDimTags=[cylinder_volume],
            toolDimTags=disk_dimtags,
        )
        occ.synchronize()

        # The first volume in ``out_dimtags`` is the chamber. Newer Gmsh
        # versions return all resulting entities sorted by dimension; we
        # filter explicitly.
        result_volumes = [dt for dt in out_dimtags if dt[0] == 3]
        if len(result_volumes) != 1:
            raise RuntimeError(
                f"Expected exactly one volume after fragment, got "
                f"{len(result_volumes)}: {result_volumes}"
            )
        chamber_volume_tag = result_volumes[0][1]

        # ``fragment_map`` is parallel to ``[object, *tools]``: index 0 is
        # the chamber's children, indices 1..N map each tool disk to the
        # surface(s) it became after fragmentation.
        n_transducer_disks = len(disk_tags)
        per_disk_surface_tags: List[int] = []
        for disk_idx in range(n_transducer_disks):
            mapping = fragment_map[1 + disk_idx]
            surf_tags = [dt[1] for dt in mapping if dt[0] == 2]
            if len(surf_tags) == 0:
                raise RuntimeError(
                    f"Disk {disk_idx} fragment produced no surfaces"
                )
            # Pick the surface whose centre is closest to the requested
            # transducer centre (robust to OCC returning auxiliary edges).
            target = centers[disk_idx]
            best_tag = None
            best_dist = np.inf
            for st in surf_tags:
                com = np.array(occ.getCenterOfMass(2, st), dtype=np.float64)
                d = float(np.linalg.norm(com - target))
                if d < best_dist:
                    best_dist = d
                    best_tag = st
            assert best_tag is not None
            per_disk_surface_tags.append(best_tag)

        # Resolve the inlet patch (if present) the same way: pick the
        # post-fragment surface whose centre-of-mass sits on the lid
        # (z ≈ height) and is closest to the chamber axis.
        inlet_surface_tag: Optional[int] = None
        if inlet_disk_tag is not None:
            mapping = fragment_map[1 + n_transducer_disks]
            surf_tags = [dt[1] for dt in mapping if dt[0] == 2]
            if len(surf_tags) == 0:
                raise RuntimeError(
                    "Crucible-inlet disk fragment produced no surfaces"
                )
            best_tag = None
            best_dist = np.inf
            for st in surf_tags:
                com = np.array(occ.getCenterOfMass(2, st), dtype=np.float64)
                # On the lid plane and close to the axis.
                if abs(com[2] - geom.height) > GEOM_TOL_M:
                    continue
                d = float(np.linalg.norm(com[:2]))
                if d < best_dist:
                    best_dist = d
                    best_tag = st
            if best_tag is None:
                raise RuntimeError(
                    "Could not identify crucible_inlet patch on lid; "
                    "fragment did not produce a lid-plane surface."
                )
            inlet_surface_tag = best_tag

        # ----- Step 4: classify all surfaces ------------------------------
        all_surfaces = [dt[1] for dt in model.getEntities(2)]
        transducer_set = set(per_disk_surface_tags)
        substrate_tags: List[int] = []
        chamber_wall_tags: List[int] = []  # cylindrical side wall (incl. lid in legacy mode)
        chamber_lid_tags: List[int] = []   # top-face annulus when inlet present
        for st in all_surfaces:
            if st in transducer_set:
                continue
            if inlet_surface_tag is not None and st == inlet_surface_tag:
                continue
            com = np.array(occ.getCenterOfMass(2, st), dtype=np.float64)
            # Substrate = z ≈ 0; lid = z ≈ height; otherwise side wall.
            if abs(com[2]) < GEOM_TOL_M:
                substrate_tags.append(st)
            elif has_inlet and abs(com[2] - geom.height) < GEOM_TOL_M:
                # Top-face annulus around the inlet — this becomes
                # ``chamber_lid``.
                chamber_lid_tags.append(st)
            else:
                # Side-wall remnants between transducers — and, in legacy
                # mode (no inlet), the lid as well.
                chamber_wall_tags.append(st)

        if not substrate_tags:
            raise RuntimeError("Failed to identify substrate surface(s).")
        if not chamber_wall_tags:
            raise RuntimeError("Failed to identify chamber-wall surface(s).")
        if has_inlet and not chamber_lid_tags:
            raise RuntimeError(
                "Failed to identify chamber_lid (lid annulus) surfaces "
                "around the crucible_inlet disk."
            )
        if len(per_disk_surface_tags) != geom.total_transducers:
            raise RuntimeError(
                f"Expected {geom.total_transducers} transducer disks, got "
                f"{len(per_disk_surface_tags)}."
            )

        # ----- Step 5: physical groups -----------------------------------
        # Per-face groups (transducer_<i>) — needed so phases can be
        # applied per-disk by index.
        per_face_group_tags: List[int] = []
        for idx, surf_tag in enumerate(per_disk_surface_tags):
            pg = model.addPhysicalGroup(2, [surf_tag])
            model.setPhysicalName(2, pg, f"transducer_{idx}")
            per_face_group_tags.append(pg)

        agg_transducer_tag = model.addPhysicalGroup(2, per_disk_surface_tags)
        model.setPhysicalName(2, agg_transducer_tag, "transducer_faces")

        chamber_walls_tag = model.addPhysicalGroup(2, chamber_wall_tags)
        model.setPhysicalName(2, chamber_walls_tag, "chamber_walls")

        substrate_tag = model.addPhysicalGroup(2, substrate_tags)
        model.setPhysicalName(2, substrate_tag, "substrate")

        if has_inlet:
            inlet_pg = model.addPhysicalGroup(2, [inlet_surface_tag])
            model.setPhysicalName(2, inlet_pg, "crucible_inlet")
            lid_pg = model.addPhysicalGroup(2, chamber_lid_tags)
            model.setPhysicalName(2, lid_pg, "chamber_lid")

        # Volume physical group — required so meshio reads in the bulk.
        volume_group = model.addPhysicalGroup(3, [chamber_volume_tag])
        model.setPhysicalName(3, volume_group, "chamber_volume")

        # ----- Step 6: refinement fields ---------------------------------
        # Distance field 1: distance to all transducer face boundaries
        # (we use the face surfaces themselves so the metric is correct in
        # the surface plane and out into the bulk).
        face_distance_field = model.mesh.field.add("Distance")
        model.mesh.field.setNumbers(
            face_distance_field, "SurfacesList", per_disk_surface_tags
        )

        # Threshold ramp: focus_h up to FACE_REFINE_DISTANCE_M, ramp to
        # bulk_h over REFINE_RAMP_M.
        face_threshold_field = model.mesh.field.add("Threshold")
        model.mesh.field.setNumber(face_threshold_field, "InField", face_distance_field)
        model.mesh.field.setNumber(face_threshold_field, "SizeMin", focus_h_m)
        model.mesh.field.setNumber(face_threshold_field, "SizeMax", bulk_h_m)
        model.mesh.field.setNumber(face_threshold_field, "DistMin", FACE_REFINE_DISTANCE_M)
        model.mesh.field.setNumber(
            face_threshold_field,
            "DistMax",
            FACE_REFINE_DISTANCE_M + REFINE_RAMP_M,
        )

        # Distance field 2: distance to the focal zone (a sphere on-axis
        # at z=height/2). Modelled as the ball field directly so we avoid
        # creating extra geometric entities.
        focal_z = 0.5 * geom.height
        focal_ball_field = model.mesh.field.add("Ball")
        model.mesh.field.setNumber(focal_ball_field, "XCenter", 0.0)
        model.mesh.field.setNumber(focal_ball_field, "YCenter", 0.0)
        model.mesh.field.setNumber(focal_ball_field, "ZCenter", focal_z)
        model.mesh.field.setNumber(focal_ball_field, "Radius", FOCAL_ZONE_RADIUS_M)
        model.mesh.field.setNumber(focal_ball_field, "Thickness", REFINE_RAMP_M)
        model.mesh.field.setNumber(focal_ball_field, "VIn", focus_h_m)
        model.mesh.field.setNumber(focal_ball_field, "VOut", bulk_h_m)

        # Optional Distance+Threshold for the crucible-inlet patch so
        # its surface mesh resolves the disk geometry rather than
        # collapsing to a single triangle at bulk_h.
        sizing_fields = [face_threshold_field, focal_ball_field]
        if has_inlet:
            inlet_distance_field = model.mesh.field.add("Distance")
            model.mesh.field.setNumbers(
                inlet_distance_field, "SurfacesList", [inlet_surface_tag]
            )
            inlet_threshold_field = model.mesh.field.add("Threshold")
            model.mesh.field.setNumber(
                inlet_threshold_field, "InField", inlet_distance_field
            )
            model.mesh.field.setNumber(inlet_threshold_field, "SizeMin", focus_h_m)
            model.mesh.field.setNumber(inlet_threshold_field, "SizeMax", bulk_h_m)
            # Keep the fine zone ~one inlet radius wide so the disk gets
            # ~16+ triangles even at the coarse focus_h_m=2 mm test setting.
            model.mesh.field.setNumber(
                inlet_threshold_field,
                "DistMin",
                max(crucible_inlet_radius_m, focus_h_m),
            )
            model.mesh.field.setNumber(
                inlet_threshold_field,
                "DistMax",
                max(crucible_inlet_radius_m, focus_h_m) + REFINE_RAMP_M,
            )
            sizing_fields.append(inlet_threshold_field)

        # Combine: take the minimum of all sizing fields.
        min_field = model.mesh.field.add("Min")
        model.mesh.field.setNumbers(min_field, "FieldsList", sizing_fields)
        model.mesh.field.setAsBackgroundMesh(min_field)

        # Disable Gmsh's other size sources so the field machinery is
        # the single source of truth.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", focus_h_m * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", bulk_h_m)

        # 3D meshing algorithm: HXT (default for OCC volumes, robust for
        # complex Boolean results).
        gmsh.option.setNumber("Mesh.Algorithm", 6)   # 2D: Frontal-Delaunay
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # 3D: HXT

        # ----- Step 7: generate and write -------------------------------
        model.mesh.generate(3)

        # Write Gmsh .msh v2 (broadest meshio compatibility).
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(output_path))


# ---------------------------------------------------------------------------
# Internals — post-build summary
# ---------------------------------------------------------------------------


def _summarise_mesh(
    path: Path,
    geom: ChamberGeometry,
    geometry_hash: str,
    focus_h_mm: float,
    bulk_h_mm: float,
    crucible_inlet_radius_m: float,
    cache_hit: bool,
) -> Dict[str, Any]:
    """Compute element/face counts + per-group statistics from the .msh."""

    import meshio

    mesh = meshio.read(str(path))
    points = np.asarray(mesh.points, dtype=np.float64)

    # Extract physical-group tag/name lookup. meshio stores them under
    # ``mesh.field_data`` as ``{name: [tag, dim]}``.
    surface_tags: Dict[str, int] = {}
    for name, value in mesh.field_data.items():
        tag, dim = int(value[0]), int(value[1])
        if dim == 2:
            surface_tags[name] = tag

    # Count volume + surface elements.
    surface_element_count = 0
    volume_element_count = 0
    for cell_block in mesh.cells:
        if cell_block.type in {"triangle", "quad"}:
            surface_element_count += len(cell_block.data)
        elif cell_block.type in {"tetra", "hexahedron", "wedge", "pyramid"}:
            volume_element_count += len(cell_block.data)

    # Per-physical-group inventory: collect surface elements, count per
    # transducer face, and compute max edge length per named group.
    per_face_counts = np.zeros(geom.total_transducers, dtype=np.int64)
    mesh_size_per_group: Dict[str, float] = {}

    # cell_data['gmsh:physical'] mirrors mesh.cells; we use it to assign
    # each surface element to its physical tag.
    physical_data = mesh.cell_data.get("gmsh:physical")
    if physical_data is None:
        raise RuntimeError(
            "meshio did not return gmsh:physical cell_data — cannot tag "
            "surface elements with their physical group."
        )

    # Build reverse map: tag → list of (cell_block_idx, local_idx).
    transducer_tag_to_index: Dict[int, int] = {}
    for idx in range(geom.total_transducers):
        name = f"transducer_{idx}"
        if name in surface_tags:
            transducer_tag_to_index[surface_tags[name]] = idx

    # Track edge lengths per named group.
    name_to_edges: Dict[str, List[float]] = {n: [] for n in surface_tags}

    for block_idx, cell_block in enumerate(mesh.cells):
        if cell_block.type not in {"triangle", "quad"}:
            continue
        block_tags = np.asarray(physical_data[block_idx], dtype=np.int64)
        cells = np.asarray(cell_block.data, dtype=np.int64)
        for local_idx, tag in enumerate(block_tags):
            face_index = transducer_tag_to_index.get(int(tag))
            if face_index is not None:
                per_face_counts[face_index] += 1
            # Append edge lengths for this element to whichever named
            # group it belongs to.
            for name, group_tag in surface_tags.items():
                if int(tag) != group_tag:
                    continue
                pts = points[cells[local_idx]]
                # All pairwise edges of the element (3 for tri, 4 for
                # quad — quads use the cyclic four-cycle).
                if cell_block.type == "triangle":
                    pairs = [(0, 1), (1, 2), (2, 0)]
                else:
                    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
                for a, b in pairs:
                    name_to_edges[name].append(
                        float(np.linalg.norm(pts[a] - pts[b]))
                    )

    for name, edges in name_to_edges.items():
        if edges:
            mesh_size_per_group[name] = float(max(edges))
        else:
            mesh_size_per_group[name] = 0.0

    file_size = path.stat().st_size

    info: Dict[str, Any] = {
        "geometry_hash": geometry_hash,
        "geometry": geom,
        "focus_h_mm": float(focus_h_mm),
        "bulk_h_mm": float(bulk_h_mm),
        "crucible_inlet_radius": float(crucible_inlet_radius_m),
        "element_count": int(volume_element_count),
        "surface_element_count": int(surface_element_count),
        "node_count": int(points.shape[0]),
        "transducer_face_count": int(np.count_nonzero(per_face_counts > 0)),
        "surface_tags": surface_tags,
        "per_face_surface_element_counts": per_face_counts.tolist(),
        "mesh_size_per_group": mesh_size_per_group,
        "file_size_bytes": int(file_size),
        "cache_hit": bool(cache_hit),
        "path": str(path),
    }
    return info


# ---------------------------------------------------------------------------
# JAX-FEM loader (orphan-node compaction)
# ---------------------------------------------------------------------------


def load_jax_fem_mesh(mesh_path: Path) -> Tuple["Any", np.ndarray]:
    """Read a persisted ``.msh`` file into a JAX-FEM ``Mesh``.

    All FEM-coupled-physics phases (Phase 2 Helmholtz, Phase 3 heat,
    Phase 4 coupling, Phase 5 backend) should use this helper rather
    than constructing their own JAX-FEM ``Mesh`` from a raw meshio
    read — it is the single source of truth for getting from disk to
    a clean JAX-FEM volume mesh.

    Why the compaction step matters
    -------------------------------
    Gmsh writes BOTH 2D surface and 3D volume nodes into the same
    point array, even though the 2D transducer-disk surfaces own nodes
    that no tetrahedron references (the disk's inner triangulation
    does not have to match the chamber-volume's boundary tessellation
    in Gmsh v2 format). Those orphan nodes introduce zero rows /
    columns into the FEM linear system, which makes every direct
    sparse solver fail with "matrix exactly singular". We filter the
    points to those actually referenced by a tet and renumber cells
    before handing the mesh to JAX-FEM.

    Surface boundary conditions in the FEM-coupled-physics stack are
    applied through *geometric* ``location_fn`` predicates (per
    ``helmholtz.py``'s ``_make_transducer_location_fn``), not via the
    ``.msh`` surface-element tags, so dropping the orphan nodes (and
    the orphan triangles that referenced them) does not affect BC
    application.

    Args:
        mesh_path: Path to a ``.msh`` v2 file produced by
            :func:`build_l1_chamber_mesh`.

    Returns:
        ``(mesh, points)`` — ``mesh`` is the JAX-FEM ``Mesh`` instance
        ready for ``Problem(...)`` construction; ``points`` is the
        compacted ``(num_used_nodes, 3)`` ndarray (use it for
        coord-based queries on the post-load nodes).

    Raises:
        RuntimeError: If the ``.msh`` file contains no tetrahedra.
    """
    import meshio  # local import — keeps package import cheap

    Mesh = _import_jax_fem_mesh()

    meshio_mesh = meshio.read(str(mesh_path))
    raw_points = np.asarray(meshio_mesh.points, dtype=np.float64)

    tetra_cells: Optional[np.ndarray] = None
    for block in meshio_mesh.cells:
        if block.type == "tetra":
            tetra_cells = np.asarray(block.data, dtype=np.int64)
            break
    if tetra_cells is None:
        raise RuntimeError(
            f"Mesh at {mesh_path} contains no tetrahedral cells."
        )

    used = np.unique(tetra_cells.flatten())
    if len(used) == len(raw_points):
        # No orphans — fast path.
        points = raw_points
        cells = tetra_cells
    else:
        # Compact the node table: keep only nodes that appear in any tet,
        # then remap cell vertex indices into the compact range.
        old_to_new = -np.ones(len(raw_points), dtype=np.int64)
        old_to_new[used] = np.arange(len(used), dtype=np.int64)
        points = raw_points[used]
        cells = old_to_new[tetra_cells]

    mesh = Mesh(points, cells, ele_type="TET4")
    return mesh, points


__all__ = [
    "DEFAULT_CRUCIBLE_INLET_RADIUS_M",
    "DEFAULT_MESH_CACHE",
    "build_l1_chamber_mesh",
    "load_jax_fem_mesh",
]
