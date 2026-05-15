"""
Inverse solver for acoustic droplet guidance.

Given a desired lateral deflection, compute the transducer phases
that will push a droplet in that direction.

Physics (from Derivation 4):
    The phase gradient method creates lateral forces by adding a
    linear phase ramp across the array:

        phi_i = -k|r_i - r_droplet| + alpha_x * x_i + alpha_y * y_i

    Where:
        - First term (-k|r|): focuses waves at droplet position
        - Second term (alpha_x * x_i): creates pressure gradient in x
        - Third term (alpha_y * y_i): creates pressure gradient in y

    The phase gradient alpha controls force direction and magnitude:
        F_x proportional to alpha_x
        F_y proportional to alpha_y

    Maximum gradient (alpha_max) is derived from force balance via
    EnvironmentConfig, NOT from a geometric argument.

Ported from: simulations/drip-sim/scripts/Scene.js (setupInverseDeflection)
"""

import warnings
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .core import ArrayGeometry, PhaseArray

if TYPE_CHECKING:
    from .core import RingArray, RingArray
    from .config import DropletConfig, EnvironmentConfig

_ALPHA_MAX_SAFETY_MARGIN = 0.8  # Leave 20% margin to avoid edge-of-array artifacts


# ---------------------------------------------------------------------------
# Core phase computation (shared by all solvers)
# ---------------------------------------------------------------------------

def _compute_base_phases(
    positions: NDArray[np.float64],
    droplet_pos: NDArray[np.float64],
    alpha_x: float,
    alpha_y: float,
    k: float,
) -> NDArray[np.float64]:
    """
    Compute raw phases for a set of transducer positions.

    This is the shared core used by both solve_inverse() and
    solve_inverse_with_rings(). No ring offsets are applied here.

    Args:
        positions: (N, 3) transducer positions in meters
        droplet_pos: [x, y, z] droplet position in meters
        alpha_x: Phase gradient in x (rad/m)
        alpha_y: Phase gradient in y (rad/m)
        k: Wavenumber (rad/m)

    Returns:
        (N,) array of raw phases (focusing + gradient, no wrapping)
    """
    # Vectorized distance computation
    displacements = droplet_pos - positions  # (N, 3)
    distances = np.linalg.norm(displacements, axis=1)  # (N,)

    # Focusing phase: constructive interference at droplet
    focusing_phases = -k * distances

    # Gradient phase: linear ramp for lateral force
    gradient_phases = alpha_x * positions[:, 0] + alpha_y * positions[:, 1]

    return focusing_phases + gradient_phases


# ---------------------------------------------------------------------------
# alpha_max and gradient scale
# ---------------------------------------------------------------------------

def compute_alpha_max(
    array: ArrayGeometry,
    env_config: Optional["EnvironmentConfig"] = None,
    droplet_config: Optional["DropletConfig"] = None,
) -> float:
    """
    Compute maximum phase gradient from force balance.

    When env_config and droplet_config are provided, alpha_max is derived
    from EnvironmentConfig.alpha_max(droplet) which performs a proper
    force-balance calculation (acoustic force must overcome gravity).

    When called without config objects (backward compatibility), falls
    back to the geometric estimate but emits a deprecation warning.

    Args:
        array: Transducer array geometry
        env_config: Environment configuration (preferred)
        droplet_config: Droplet configuration (preferred)

    Returns:
        Maximum gradient in rad/m
    """
    if env_config is not None and droplet_config is not None:
        return env_config.alpha_max(droplet_config)

    # Legacy fallback with deprecation warning
    warnings.warn(
        "compute_alpha_max called without EnvironmentConfig/DropletConfig. "
        "Using geometric estimate pi/(4d) which does NOT account for "
        "force balance. Pass env_config and droplet_config for correct results.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _geometric_alpha_max_fallback(array)


def _geometric_alpha_max_fallback(array: ArrayGeometry) -> float:
    """
    Legacy geometric alpha_max estimate: pi/(4d).

    [ASMP-007] This is a focal-quality limit from Derivation 4, not a
    force-balance limit. See ASSUMPTIONS.md. Both limits should coexist.
    """
    xy_positions = array.positions[:, :2]
    extent = np.max(np.linalg.norm(xy_positions, axis=1))

    if extent < 0.01:
        return 10.0

    n_around = max(4, int(np.sqrt(array.num_transducers)))
    spacing = 2 * np.pi * extent / n_around
    return float(np.pi / (4 * spacing))


def _compute_gradient_scale(
    array: ArrayGeometry,
    env_config: Optional["EnvironmentConfig"] = None,
    droplet_config: Optional["DropletConfig"] = None,
) -> float:
    """
    Compute gradient scale: maps target offset (m) to phase gradient (rad/m).

    When config objects are provided:
        - alpha_max from force balance (EnvironmentConfig.alpha_max)
        - max_offset from ArrayConfig.cylinder_radius (not hardcoded 50mm)

    When called without config (backward compat):
        - Falls back to geometric estimate with deprecation warning

    Args:
        array: Transducer array geometry
        env_config: Environment configuration (preferred)
        droplet_config: Droplet configuration (preferred)

    Returns:
        Gradient scale in rad/m per meter of offset
    """
    if env_config is not None and droplet_config is not None:
        alpha_max = env_config.alpha_max(droplet_config)
        max_offset = env_config.array.cylinder_radius
        if max_offset < 1e-6:
            warnings.warn(
                "ArrayConfig.cylinder_radius is near zero; "
                "cannot derive max_offset for gradient scale.",
                stacklevel=2,
            )
            return 100.0
        # Scale so that max_offset gives ~_ALPHA_MAX_SAFETY_MARGIN * alpha_max (leave margin)
        return _ALPHA_MAX_SAFETY_MARGIN * alpha_max / max_offset

    # Legacy fallback
    return _gradient_scale_legacy_fallback(array)


def _gradient_scale_legacy_fallback(array: ArrayGeometry) -> float:
    """Legacy gradient scale using geometric alpha_max and hardcoded max_offset."""
    xy_positions = array.positions[:, :2]
    extent = np.max(np.linalg.norm(xy_positions, axis=1))

    if extent < 0.01:
        return 100.0

    n_around = max(4, int(np.sqrt(array.num_transducers)))
    spacing = 2 * np.pi * extent / n_around
    alpha_max = np.pi / (4 * spacing)

    # max_offset derived from array extent (cylinder_radius)
    max_offset = extent
    return float(0.8 * alpha_max / max_offset)


# ---------------------------------------------------------------------------
# Clamping and validation
# ---------------------------------------------------------------------------

def _clamp_alpha(
    alpha_x: float,
    alpha_y: float,
    alpha_max: float,
) -> Tuple[float, float]:
    """
    Clamp gradient vector magnitude to alpha_max.

    If the requested gradient exceeds alpha_max, warn explicitly
    and scale back to alpha_max while preserving direction.

    Returns:
        Clamped (alpha_x, alpha_y)
    """
    alpha_mag = np.sqrt(alpha_x**2 + alpha_y**2)
    if alpha_mag > alpha_max and alpha_max > 0:
        warnings.warn(
            f"Requested phase gradient |alpha|={alpha_mag:.4f} rad/m "
            f"exceeds alpha_max={alpha_max:.4f} rad/m (force balance limit). "
            f"Clamping to alpha_max. Target offset may not be achievable.",
            stacklevel=3,
        )
        scale = alpha_max / alpha_mag
        return alpha_x * scale, alpha_y * scale
    return alpha_x, alpha_y


def validate_offset(
    array: ArrayGeometry,
    target_offset: Tuple[float, float],
    gradient_scale: Optional[float] = None,
    env_config: Optional["EnvironmentConfig"] = None,
    droplet_config: Optional["DropletConfig"] = None,
) -> Tuple[bool, float, float]:
    """
    Check if target offset is achievable without exceeding alpha_max.

    Args:
        array: Transducer array geometry
        target_offset: (x, y) in meters
        gradient_scale: Gradient scale (computed if None)
        env_config: Environment configuration (preferred)
        droplet_config: Droplet configuration (preferred)

    Returns:
        Tuple of (is_valid, alpha_requested, alpha_max)
    """
    if gradient_scale is None:
        gradient_scale = _compute_gradient_scale(array, env_config, droplet_config)

    target_x, target_y = target_offset
    alpha_x = target_x * gradient_scale
    alpha_y = target_y * gradient_scale

    alpha_requested = float(np.sqrt(alpha_x**2 + alpha_y**2))
    a_max = compute_alpha_max(array, env_config, droplet_config)

    is_valid = alpha_requested <= a_max

    return is_valid, alpha_requested, a_max


# ---------------------------------------------------------------------------
# Public solvers
# ---------------------------------------------------------------------------

def solve_inverse(
    array: ArrayGeometry,
    target_offset: Tuple[float, float],
    droplet_z: float = 0.0,
    droplet_xy: Optional[Tuple[float, float]] = None,
    amplitude: float = 1.0,
    gradient_scale: Optional[float] = None,
    env_config: Optional["EnvironmentConfig"] = None,
    droplet_config: Optional["DropletConfig"] = None,
) -> PhaseArray:
    """
    Compute phases to deflect a droplet toward a target offset.

    This is the main inverse solver - given where you want the droplet
    to land, compute the phases that will push it there.

    Args:
        array: Transducer array geometry
        target_offset: (x, y) lateral offset in meters - direction to push
        droplet_z: Current z position of droplet in meters
        droplet_xy: Current (x, y) position of droplet. If None, assumes (0, 0).
                   IMPORTANT: Must track droplet position for off-axis guidance!
        amplitude: Transducer amplitude (uniform for all)
        gradient_scale: Phase gradient scaling factor (rad/m per m of offset)
                       If None, computed from array geometry / config.
        env_config: EnvironmentConfig for force-balance alpha_max
        droplet_config: DropletConfig for mass/weight in alpha_max

    Returns:
        PhaseArray with phases configured for deflection

    Physics:
        The phase for each transducer is:
            phi_i = phi_focus + phi_gradient

        Where:
            phi_focus = -k * distance_to_droplet
            phi_gradient = alpha_x * x_i + alpha_y * y_i

    Example:
        # Legacy (backward compatible):
        phases = solve_inverse(array, target_offset=(5e-3, 0), droplet_z=0.1)

        # With config (preferred):
        from drip_physics.config import default_environment, aluminum_droplet
        env = default_environment()
        drop = aluminum_droplet()
        phases = solve_inverse(array, target_offset=(5e-3, 0), droplet_z=0.1,
                               env_config=env, droplet_config=drop)
    """
    target_x, target_y = target_offset

    # Droplet position
    if droplet_xy is not None:
        droplet_pos = np.array([droplet_xy[0], droplet_xy[1], droplet_z])
    else:
        droplet_pos = np.array([0.0, 0.0, droplet_z])

    # Gradient scale
    if gradient_scale is None:
        gradient_scale = _compute_gradient_scale(array, env_config, droplet_config)

    # Gradient coefficients (rad/m)
    # SIGN: Negative gradient creates force toward target
    alpha_x = -target_x * gradient_scale
    alpha_y = -target_y * gradient_scale

    # Clamp to alpha_max with explicit warning
    a_max = compute_alpha_max(array, env_config, droplet_config)
    alpha_x, alpha_y = _clamp_alpha(alpha_x, alpha_y, a_max)

    # Compute phases using shared core
    k = array.params.wavenumber
    phases = _compute_base_phases(array.positions, droplet_pos, alpha_x, alpha_y, k)

    amplitudes = np.full(array.num_transducers, amplitude)
    return PhaseArray(phases, amplitudes)


def solve_inverse_gradient(
    array: ArrayGeometry,
    alpha: Tuple[float, float],
    droplet_z: float = 0.0,
    amplitude: float = 1.0,
    env_config: Optional["EnvironmentConfig"] = None,
    droplet_config: Optional["DropletConfig"] = None,
) -> PhaseArray:
    """
    Compute phases given explicit phase gradient coefficients.

    Lower-level interface - use this if you want direct control
    over the gradient rather than specifying target offset.

    Args:
        array: Transducer array geometry
        alpha: (alpha_x, alpha_y) phase gradient coefficients in rad/m
        droplet_z: Current z position of droplet
        amplitude: Transducer amplitude
        env_config: EnvironmentConfig for alpha_max clamping
        droplet_config: DropletConfig for alpha_max clamping

    Returns:
        PhaseArray configured with given gradient
    """
    alpha_x, alpha_y = alpha
    droplet_pos = np.array([0.0, 0.0, droplet_z])

    # Clamp if config available
    a_max = compute_alpha_max(array, env_config, droplet_config)
    alpha_x, alpha_y = _clamp_alpha(alpha_x, alpha_y, a_max)

    k = array.params.wavenumber
    phases = _compute_base_phases(array.positions, droplet_pos, alpha_x, alpha_y, k)

    amplitudes = np.full(array.num_transducers, amplitude)
    return PhaseArray(phases, amplitudes)


def solve_focus(
    array: ArrayGeometry,
    focus_position: NDArray[np.float64],
    amplitude: float = 1.0,
) -> PhaseArray:
    """
    Compute phases to focus at a specific point (no gradient).

    This is the basic phased array focusing - all transducers
    configured so their waves arrive in-phase at the focus point.

    Args:
        array: Transducer array geometry
        focus_position: [x, y, z] focus point in meters
        amplitude: Transducer amplitude

    Returns:
        PhaseArray configured for focusing

    Note:
        This creates maximum pressure at the focus point but does
        NOT create a lateral force (no gradient). For guidance,
        use solve_inverse() instead.
    """
    focus_position = np.asarray(focus_position)
    k = array.params.wavenumber

    # Focus is just the base phase computation with zero gradient
    phases = _compute_base_phases(
        array.positions, focus_position,
        alpha_x=0.0, alpha_y=0.0, k=k,
    )

    amplitudes = np.full(array.num_transducers, amplitude)
    return PhaseArray(phases, amplitudes)


def solve_inverse_with_rings(
    ring_array: "RingArray",
    target_offset: Tuple[float, float],
    droplet_z: float = 0.0,
    droplet_xy: Optional[Tuple[float, float]] = None,
    amplitude: float = 1.0,
    gradient_scale: Optional[float] = None,
    active_rings_only: bool = True,
    env_config: Optional["EnvironmentConfig"] = None,
    droplet_config: Optional["DropletConfig"] = None,
) -> PhaseArray:
    """
    Compute phases respecting ring enabled states, phase offsets,
    and phase continuity at ring boundaries.

    This is the ring-aware wrapper around the shared core logic.
    It adds per-ring phase offsets and enforces phase continuity
    at ring handoff boundaries so there are no discontinuous jumps.

    Phase continuity constraint:
        After computing base phases + ring offsets, the phase at the
        end of ring N is adjusted to match the phase at the start of
        ring N+1. This is done by computing a continuity correction
        per ring that minimizes the phase jump at boundaries.

    Args:
        ring_array: RingArray with ring configuration
        target_offset: (x, y) lateral offset in meters
        droplet_z: Current z position of droplet in meters
        droplet_xy: Current (x, y) position of droplet
        amplitude: Base transducer amplitude
        gradient_scale: Phase gradient scaling factor
        active_rings_only: If True, disabled rings have zero amplitude
        env_config: EnvironmentConfig for force-balance alpha_max
        droplet_config: DropletConfig for mass/weight in alpha_max

    Returns:
        PhaseArray with ring-aware phases and amplitudes
    """
    array = ring_array.geometry
    target_x, target_y = target_offset

    # Droplet position
    if droplet_xy is not None:
        droplet_pos = np.array([droplet_xy[0], droplet_xy[1], droplet_z])
    else:
        droplet_pos = np.array([0.0, 0.0, droplet_z])

    # Gradient scale
    if gradient_scale is None:
        gradient_scale = _compute_gradient_scale(array, env_config, droplet_config)

    # Gradient coefficients
    alpha_x = -target_x * gradient_scale
    alpha_y = -target_y * gradient_scale

    # Clamp to alpha_max
    a_max = compute_alpha_max(array, env_config, droplet_config)
    alpha_x, alpha_y = _clamp_alpha(alpha_x, alpha_y, a_max)

    # Compute base phases using shared core
    k = array.params.wavenumber
    base_phases = _compute_base_phases(
        array.positions, droplet_pos, alpha_x, alpha_y, k,
    )

    # Get ring phase offsets and amplitudes
    ring_phase_offsets = ring_array.get_phase_offsets()

    if active_rings_only:
        ring_amplitudes = ring_array.get_effective_amplitudes()
    else:
        ring_amplitudes = np.ones(array.num_transducers)
        for ring in ring_array.rings:
            ring_amplitudes[ring.transducer_indices] = ring.amplitude

    # Apply ring offsets
    phases = base_phases + ring_phase_offsets

    # ---------------------------------------------------------------
    # Phase continuity constraint at ring boundaries
    # ---------------------------------------------------------------
    # For each adjacent pair of rings, compute the mean phase at the
    # boundary and apply a correction so that the average phase at the
    # end of ring N matches the average phase at the start of ring N+1.
    #
    # We accumulate a running correction so that adjustments propagate
    # smoothly through the array (ring 0 is the reference).
    # ---------------------------------------------------------------
    rings = ring_array.rings
    if len(rings) > 1:
        # Ring 0 is the reference (no correction applied).
        # For each subsequent ring, compute the phase mismatch between
        # the (already-corrected) previous ring and the current ring,
        # then shift the current ring to close the gap.
        for r_idx in range(1, len(rings)):
            prev_ring = rings[r_idx - 1]
            curr_ring = rings[r_idx]

            prev_indices = prev_ring.transducer_indices
            curr_indices = curr_ring.transducer_indices

            if len(prev_indices) == 0 or len(curr_indices) == 0:
                continue

            # Previous ring phases are already corrected (or ring 0 baseline)
            mean_prev = _circular_mean(phases[prev_indices])
            mean_curr = _circular_mean(phases[curr_indices])

            # Phase difference, wrapped to [-pi, pi]
            delta = _wrap_to_pi(mean_prev - mean_curr)

            # Shift current ring to match the previous ring's mean phase
            phases[curr_indices] += delta

    # Wrap to [0, 2*pi]
    phases = phases % (2 * np.pi)

    # Amplitudes: base * ring
    amplitudes = amplitude * ring_amplitudes

    return PhaseArray(phases, amplitudes)


# ---------------------------------------------------------------------------
# Phase continuity helpers
# ---------------------------------------------------------------------------

def _circular_mean(angles: NDArray[np.float64]) -> float:
    """
    Compute circular (angular) mean of a set of phase values.

    This avoids the wraparound problem with simple arithmetic mean
    on phase values (e.g., mean of 0.1 and 6.18 should be near 0, not 3.14).

    Args:
        angles: Array of phase values in radians

    Returns:
        Circular mean in radians
    """
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def _wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi] range."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


# ---------------------------------------------------------------------------
# Multi-droplet inverse solver
# ---------------------------------------------------------------------------

def _assign_droplets_to_zones(
    ring_array: "RingArray",
    droplet_positions: List[NDArray[np.float64]],
) -> Optional[Dict[int, List[int]]]:
    """Assign each droplet to a ring zone based on z-position.

    A zone is defined by the ring closest to each droplet's z-coordinate.
    Returns None if any two droplets map to the same ring (overlap).

    Args:
        ring_array: RingArray with ring configuration.
        droplet_positions: List of [x, y, z] arrays.

    Returns:
        Dict mapping ring_id to list of droplet indices, or None if overlap.
    """
    ring_zs = np.array([r.z_position for r in ring_array.rings])
    assignments: Dict[int, List[int]] = {}

    for i, pos in enumerate(droplet_positions):
        closest_ring = int(np.argmin(np.abs(ring_zs - pos[2])))
        if closest_ring not in assignments:
            assignments[closest_ring] = []
        assignments[closest_ring].append(i)

    # Check for overlap: any ring with more than one droplet
    for ring_id, droplets in assignments.items():
        if len(droplets) > 1:
            return None

    return assignments


def solve_inverse_multi(
    array: ArrayGeometry,
    droplet_states: List[Tuple[NDArray[np.float64], Tuple[float, float]]],
    env_config: Optional["EnvironmentConfig"] = None,
    droplet_configs: Optional[List["DropletConfig"]] = None,
    weights: Optional[NDArray[np.float64]] = None,
) -> PhaseArray:
    """Compute phases that simultaneously steer multiple droplets.

    Uses coherent multi-focal superposition: for each transducer, compute
    a weighted complex sum of the focusing + gradient phasors for all droplets.
    The resulting phase is the angle of the sum.

    When droplets are in non-overlapping ring zones, delegates to per-zone
    solve_inverse() calls (exact solution). In the overlap case, the coherent
    sum provides an analytically principled approximation.

    Args:
        array: Transducer array geometry.
        droplet_states: List of (position_3d, target_offset_xy) tuples.
            position_3d: [x, y, z] current droplet position in meters.
            target_offset_xy: (dx, dy) lateral offset FROM CURRENT POSITION
                to desired landing target, in meters.
                Computed as: (target_x - pos_x, target_y - pos_y).
        env_config: Environment configuration (for wavenumber, alpha_max).
        droplet_configs: Per-droplet configs (for force validation). Optional.
        weights: Priority weights per droplet. Default: equal weights.

    Returns:
        PhaseArray with computed phases for the entire array.
    """
    n_droplets = len(droplet_states)
    if n_droplets == 0:
        raise ValueError("droplet_states must be non-empty")

    # Default equal weights
    if weights is None:
        weights = np.ones(n_droplets)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (n_droplets,):
            raise ValueError(
                f"weights shape {weights.shape} doesn't match "
                f"number of droplets ({n_droplets})"
            )

    # Single-droplet: delegate directly to solve_inverse
    if n_droplets == 1:
        pos, offset = droplet_states[0]
        pos = np.asarray(pos, dtype=np.float64)
        dc = droplet_configs[0] if droplet_configs else None
        return solve_inverse(
            array,
            target_offset=offset,
            droplet_z=pos[2],
            droplet_xy=(pos[0], pos[1]),
            env_config=env_config,
            droplet_config=dc,
        )

    # Try zone-based decoupled solving via RingArray
    from .core import RingArray

    ring_array = RingArray(array)
    positions = [np.asarray(s[0], dtype=np.float64) for s in droplet_states]
    zone_map = _assign_droplets_to_zones(ring_array, positions)

    if zone_map is not None and len(zone_map) == n_droplets:
        # Non-overlapping zones: solve per-zone independently
        return _solve_decoupled(
            array, ring_array, droplet_states, zone_map,
            env_config, droplet_configs,
        )

    # Overlapping zones: coherent multi-focal superposition
    return _solve_coherent(
        array, droplet_states, env_config, droplet_configs, weights,
    )


def _solve_decoupled(
    array: ArrayGeometry,
    ring_array: "RingArray",
    droplet_states: List[Tuple[NDArray[np.float64], Tuple[float, float]]],
    zone_map: Dict[int, List[int]],
    env_config: Optional["EnvironmentConfig"],
    droplet_configs: Optional[List["DropletConfig"]],
) -> PhaseArray:
    """Solve per-zone independently when droplets don't overlap.

    Each droplet's closest ring(s) are solved independently, and the
    resulting phases are stitched together.
    """
    N = array.num_transducers
    phases = np.zeros(N)
    amplitudes = np.zeros(N)  # Only assigned rings radiate

    for ring_id, droplet_indices in zone_map.items():
        # Should be exactly one droplet per ring when decoupled
        drop_idx = droplet_indices[0]
        pos, offset = droplet_states[drop_idx]
        pos = np.asarray(pos, dtype=np.float64)
        dc = droplet_configs[drop_idx] if droplet_configs else None

        ring = ring_array.rings[ring_id]
        indices = ring.transducer_indices

        if len(indices) == 0:
            continue

        # Solve for this droplet on this ring's transducers
        # Use _compute_base_phases directly for the ring's transducers
        gradient_scale = _compute_gradient_scale(array, env_config, dc)
        target_x, target_y = offset
        alpha_x = -target_x * gradient_scale
        alpha_y = -target_y * gradient_scale
        a_max = compute_alpha_max(array, env_config, dc)
        alpha_x, alpha_y = _clamp_alpha(alpha_x, alpha_y, a_max)

        k = array.params.wavenumber
        ring_phases = _compute_base_phases(
            array.positions[indices], pos, alpha_x, alpha_y, k,
        )
        phases[indices] = ring_phases % (2 * np.pi)
        amplitudes[indices] = 1.0

    return PhaseArray(phases, amplitudes)


def _solve_coherent(
    array: ArrayGeometry,
    droplet_states: List[Tuple[NDArray[np.float64], Tuple[float, float]]],
    env_config: Optional["EnvironmentConfig"],
    droplet_configs: Optional[List["DropletConfig"]],
    weights: NDArray[np.float64],
) -> PhaseArray:
    """Coherent multi-focal superposition for overlapping zones.

    For each transducer, compute a weighted complex sum of the
    focusing + gradient phasors for all droplets.
    """
    N = array.num_transducers
    k = array.params.wavenumber
    trans_pos = array.positions  # (N, 3)

    complex_phases = np.zeros(N, dtype=complex)

    for i, ((pos, offset), w) in enumerate(zip(droplet_states, weights)):
        pos = np.asarray(pos, dtype=np.float64)
        dc = droplet_configs[i] if droplet_configs else None

        # Compute gradient for this droplet
        gradient_scale = _compute_gradient_scale(array, env_config, dc)
        target_x, target_y = offset
        alpha_x = -target_x * gradient_scale
        alpha_y = -target_y * gradient_scale
        a_max = compute_alpha_max(array, env_config, dc)
        alpha_x, alpha_y = _clamp_alpha(alpha_x, alpha_y, a_max)

        # Focusing: exp(-j * k * |r_i - r_drop|)
        distances = np.linalg.norm(trans_pos - pos, axis=1)
        focusing = np.exp(-1j * k * distances)

        # Gradient: exp(j * (alpha_x * x_i + alpha_y * y_i))
        gradient = np.exp(1j * (alpha_x * trans_pos[:, 0] + alpha_y * trans_pos[:, 1]))

        complex_phases += w * focusing * gradient

    phases = np.angle(complex_phases) % (2 * np.pi)
    amplitudes = np.ones(N)

    # Validate force directions
    _warn_on_bad_forces(array, PhaseArray(phases, amplitudes),
                        droplet_states, env_config, droplet_configs)

    return PhaseArray(phases, amplitudes)


def _warn_on_bad_forces(
    array: ArrayGeometry,
    phases: PhaseArray,
    droplet_states: List[Tuple[NDArray[np.float64], Tuple[float, float]]],
    env_config: Optional["EnvironmentConfig"],
    droplet_configs: Optional[List["DropletConfig"]],
) -> None:
    """Emit warning if any droplet's force is misaligned with target."""
    from .force import compute_force

    for i, (pos, offset) in enumerate(droplet_states):
        pos = np.asarray(pos, dtype=np.float64)
        target_x, target_y = offset
        target_mag = np.sqrt(target_x**2 + target_y**2)
        if target_mag < 1e-10:
            continue

        dc = droplet_configs[i] if droplet_configs else None
        force = compute_force(
            array, phases, pos,
            environment=env_config,
            droplet_config=dc,
        )

        # Check lateral alignment
        force_xy = force[:2]
        target_dir = np.array([target_x, target_y]) / target_mag
        force_mag = np.linalg.norm(force_xy)
        if force_mag < 1e-30:
            warnings.warn(
                f"solve_inverse_multi: droplet {i} has near-zero lateral force. "
                f"Target offset ({target_x*1e3:.2f}, {target_y*1e3:.2f}) mm may not "
                f"be achievable in multi-droplet configuration.",
                stacklevel=4,
            )
            continue

        alignment = float(np.dot(force_xy / force_mag, target_dir))
        if alignment < 0:
            warnings.warn(
                f"solve_inverse_multi: droplet {i} force is OPPOSITE to target "
                f"(alignment={alignment:.2f}). Multi-droplet interference is too "
                f"strong for this configuration.",
                stacklevel=4,
            )


def validate_multi_solution(
    array: ArrayGeometry,
    phases: PhaseArray,
    droplet_states: List[Tuple[NDArray[np.float64], Tuple[float, float]]],
    env_config: Optional["EnvironmentConfig"] = None,
    droplet_configs: Optional[List["DropletConfig"]] = None,
) -> List[Dict[str, Any]]:
    """Validate that computed phases produce correct force directions for all droplets.

    For each droplet, computes the acoustic force and checks:
    - Whether the lateral force is in the correct direction (alignment > 0)
    - The force magnitude
    - The alignment score (dot product of normalized force and target direction)

    Args:
        array: Transducer array geometry.
        phases: Computed PhaseArray to validate.
        droplet_states: List of (position_3d, target_offset_xy) tuples.
        env_config: Environment configuration.
        droplet_configs: Per-droplet configs.

    Returns:
        List of dicts, one per droplet, with keys:
            'droplet_index': int
            'force': ndarray (3,) — force vector in N
            'force_lateral': ndarray (2,) — lateral force component
            'force_magnitude': float — |F_lateral|
            'target_direction': ndarray (2,) — normalized target direction
            'alignment': float — dot(force_dir, target_dir), 1.0 = perfect
            'is_correct': bool — alignment > 0
    """
    from .force import compute_force

    results: List[Dict[str, Any]] = []

    for i, (pos, offset) in enumerate(droplet_states):
        pos = np.asarray(pos, dtype=np.float64)
        target_x, target_y = offset
        dc = droplet_configs[i] if droplet_configs else None

        force = compute_force(
            array, phases, pos,
            environment=env_config,
            droplet_config=dc,
        )

        force_lateral = force[:2]
        force_mag = float(np.linalg.norm(force_lateral))

        target_mag = np.sqrt(target_x**2 + target_y**2)
        if target_mag < 1e-10:
            # No target offset — any force direction is acceptable
            results.append({
                'droplet_index': i,
                'force': force.copy(),
                'force_lateral': force_lateral.copy(),
                'force_magnitude': force_mag,
                'target_direction': np.array([0.0, 0.0]),
                'alignment': 1.0,
                'is_correct': True,
            })
            continue

        target_dir = np.array([target_x, target_y]) / target_mag

        if force_mag < 1e-30:
            alignment = 0.0
        else:
            alignment = float(np.dot(force_lateral / force_mag, target_dir))

        results.append({
            'droplet_index': i,
            'force': force.copy(),
            'force_lateral': force_lateral.copy(),
            'force_magnitude': force_mag,
            'target_direction': target_dir.copy(),
            'alignment': alignment,
            'is_correct': alignment > 0,
        })

    return results
