"""
Transducer array geometry generators.

This module provides functions to create common transducer array
configurations for acoustic levitation experiments.

Coordinate system:
    - X, Y: horizontal plane
    - Z: vertical (up is positive)
    - Origin: center of array

Standard configurations:
    - L1 Demo: 10 rings × 12 transducers = 120 total, R=5cm, H=40cm
    - Full system: 390 transducers (details TBD)
"""

import numpy as np
from numpy.typing import NDArray


def create_cylindrical_array(
    radius: float = 0.05,
    height: float = 0.4,
    rings: int = 10,
    per_ring: int = 12,
) -> NDArray[np.float64]:
    """
    Create a cylindrical transducer array.

    Transducers are arranged in horizontal rings stacked vertically.
    Each transducer points inward toward the central axis.

    Args:
        radius: Cylinder radius in meters (default 5 cm)
        height: Total height in meters (default 40 cm)
        rings: Number of horizontal rings (default 10)
        per_ring: Number of transducers per ring (default 12)

    Returns:
        Array of positions, shape (N, 3) where N = rings × per_ring

    Geometry:
        - Rings are evenly spaced from -height/2 to +height/2
        - Transducers in each ring are evenly spaced in angle
        - If rings=1, single ring at z=0
        - If per_ring=1, single transducer per ring (line array)

    Example:
        # Standard L1 array
        positions = create_cylindrical_array(
            radius=0.05, height=0.4, rings=10, per_ring=12)
        # Returns 120 positions
    """
    positions = []

    # Vertical positions of rings
    if rings == 1:
        z_positions = [0.0]
    else:
        z_positions = np.linspace(-height / 2, height / 2, rings)

    # Angular positions within each ring
    if per_ring == 1:
        angles = [0.0]
    else:
        angles = np.linspace(0, 2 * np.pi, per_ring, endpoint=False)

    for z in z_positions:
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append([x, y, z])

    return np.array(positions, dtype=np.float64)


def create_planar_array(
    extent: float = 0.1,
    spacing: float = 0.01,
    gap: float = 0.2,
) -> NDArray[np.float64]:
    """
    Create a planar transducer array (two opposing plates).

    Used for acoustic levitation experiments and testing.
    Creates a grid of transducers on two parallel planes facing each other.

    Args:
        extent: Half-width of each plate in meters (plate spans ±extent)
        spacing: Distance between adjacent transducers in meters
        gap: Distance between the two plates in meters

    Returns:
        Array of positions, shape (N, 3)

    Geometry:
        - Top plate at z = +gap/2 (transducers face down)
        - Bottom plate at z = -gap/2 (transducers face up)
        - Grid spans x,y ∈ [-extent, +extent]

    Example:
        # 10cm × 10cm plates, 1cm spacing, 20cm gap
        positions = create_planar_array(extent=0.05, spacing=0.01, gap=0.2)
    """
    positions = []

    # Grid of x,y positions
    n_per_side = int(2 * extent / spacing) + 1
    coords = np.linspace(-extent, extent, n_per_side)

    z_top = gap / 2
    z_bottom = -gap / 2

    for x in coords:
        for y in coords:
            # Top plate
            positions.append([x, y, z_top])
            # Bottom plate
            positions.append([x, y, z_bottom])

    return np.array(positions, dtype=np.float64)


def create_ring_array(
    radius: float = 0.05,
    n_transducers: int = 12,
    z_position: float = 0.0,
) -> NDArray[np.float64]:
    """
    Create a single ring of transducers.

    Useful for testing or building custom array geometries.

    Args:
        radius: Ring radius in meters
        n_transducers: Number of transducers in the ring
        z_position: Z position of the ring in meters

    Returns:
        Array of positions, shape (N, 3)

    Example:
        # Single ring at z=0
        ring = create_ring_array(radius=0.05, n_transducers=12)
    """
    angles = np.linspace(0, 2 * np.pi, n_transducers, endpoint=False)

    positions = np.zeros((n_transducers, 3))
    positions[:, 0] = radius * np.cos(angles)
    positions[:, 1] = radius * np.sin(angles)
    positions[:, 2] = z_position

    return positions


def create_custom_array(
    ring_specs: list,
) -> NDArray[np.float64]:
    """
    Create a custom array from ring specifications.

    Allows non-uniform ring configurations (different radii,
    different transducer counts per ring).

    Args:
        ring_specs: List of (z, radius, n_transducers) tuples

    Returns:
        Array of positions, shape (N, 3)

    Example:
        # Tapered cylinder: wider at top
        specs = [
            (-0.2, 0.04, 8),   # Bottom ring: small
            (-0.1, 0.05, 10),
            (0.0, 0.06, 12),   # Middle: medium
            (0.1, 0.07, 14),
            (0.2, 0.08, 16),   # Top: large
        ]
        positions = create_custom_array(specs)
    """
    all_positions = []

    for z, radius, n in ring_specs:
        ring = create_ring_array(radius, n, z)
        all_positions.append(ring)

    return np.vstack(all_positions)


def array_info(positions: NDArray[np.float64]) -> dict:
    """
    Compute useful information about an array geometry.

    Args:
        positions: Array positions, shape (N, 3)

    Returns:
        Dictionary with array statistics

    Example:
        info = array_info(positions)
        print(f"Array has {info['n_transducers']} transducers")
        print(f"Radius: {info['radius_xy']*1e3:.1f} mm")
    """
    n = positions.shape[0]

    # XY extent (radius)
    xy_distances = np.linalg.norm(positions[:, :2], axis=1)
    radius_xy = np.max(xy_distances)

    # Z extent
    z_min = np.min(positions[:, 2])
    z_max = np.max(positions[:, 2])
    height = z_max - z_min

    # Estimate transducer spacing (nearest neighbor average)
    if n > 1:
        from scipy.spatial import distance_matrix
        dists = distance_matrix(positions, positions)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        avg_spacing = np.mean(min_dists)
    else:
        avg_spacing = 0.0

    return {
        'n_transducers': n,
        'radius_xy': radius_xy,
        'z_min': z_min,
        'z_max': z_max,
        'height': height,
        'avg_spacing': avg_spacing,
    }
