"""
Pressure field computation for acoustic transducer arrays.

This module computes the complex pressure phasor at any point in space given
a transducer array configuration and phase settings. Returns complex values
to support BEM superposition and interference analysis.

Physics (from Derivation 1-3 and Ultraino validation):
    Each transducer emits a spherical wave with directivity:
        p_i(r, t) = (A_i x D(theta) / |r - r_i|) x sin(k|r - r_i| - wt + phi_i)

    Where D(theta) is the directivity pattern (sinc function for piston source).

    The total pressure is the superposition of all transducers:
        P(r) = sum_i (A_i x D(theta_i) / |r - r_i|) x e^(i(k|r - r_i| + phi_i))

    The pressure squared <p^2> = |P|^2 determines the Gor'kov potential.

Physical scaling:
    Reference pressure P0 and r_ref from TransducerConfig (datasheet values).
    Speed of sound from EnvironmentConfig.speed_of_sound (temperature-derived).
    Module-level TRANSDUCER_P0/REFERENCE_DISTANCE retained for backward
    compatibility when no config objects are passed.

Ported from: simulations/drip-sim/shaders/voxel/simulation.wgsl
Validated against: github.com/asiermarzo/Ultraino
"""

import warnings

import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, Optional, TYPE_CHECKING

from .core import ArrayGeometry, PhaseArray, SimulationParams

if TYPE_CHECKING:
    from .core import RingArray
    from .boundary import BoundaryReflectionModel
    from .config import EnvironmentConfig, TransducerConfig
    from .pressure_backend import PressureBackend


# =============================================================================
# Backward-compat defaults (prefer config objects)
# =============================================================================

# Reference pressure from a single transducer at reference distance
# Murata MA40S4S: ~115 dB SPL at 30cm = ~11 Pa
# At 1cm reference: ~200 Pa (due to 1/r spreading)
TRANSDUCER_P0 = 200.0  # [ASMP-002] assumed: 1/r extrapolation from datasheet. See ASSUMPTIONS.md.
                       # Config has p_ref=10 Pa at 0.3m (inconsistent). Force scales as p^2.
REFERENCE_DISTANCE = 0.01  # 1 cm

# Transducer aperture (piston radius for directivity calculation)
# Murata MA40S4S: ~10mm diameter = 5mm radius
TRANSDUCER_APERTURE = 0.005  # [ASMP-016] assumed: 5mm RADIUS (10mm diam). Config has 2.5mm radius (5mm diam). 2x discrepancy.


# [ASMP-061] Acoustic attenuation in air — Bass et al. (1995)
def air_attenuation_coefficient(frequency_hz: float, temperature_k: float) -> float:
    """Compute acoustic attenuation in air [Np/m].

    Classical absorption (viscous + thermal) dominates at 40 kHz.
    Bass et al. (1995), JASA 97(1), 680-683.

    Simplified formula valid for 20-80 kHz, 273-1000 K, 1 atm:
        alpha = 1.84e-11 * f^2 * sqrt(T/293.15) / (T/293.15)

    This gives:
        alpha(40kHz, 293K) ~ 0.12 Np/m  (negligible over 5cm)
        alpha(40kHz, 573K) ~ 0.19 Np/m  (still small)
        alpha(40kHz, 873K) ~ 0.23 Np/m  (2.3% loss over 10cm)
    """
    T_ratio = temperature_k / 293.15
    return 1.84e-11 * frequency_hz**2 * np.sqrt(T_ratio) / T_ratio


def _resolve_attenuation(
    env: Optional["EnvironmentConfig"],
) -> float:
    """Resolve attenuation coefficient alpha [Np/m] from env, or 0 for legacy path."""
    if env is not None:
        return air_attenuation_coefficient(env.transducer.frequency, env.temperature)
    return 0.0


def _resolve_pressure_params(
    env: Optional["EnvironmentConfig"],
    transducer: Optional["TransducerConfig"],
) -> Tuple[float, float, float]:
    """
    Resolve p_ref, r_ref, and aperture from config objects or module defaults.

    Returns:
        (p_ref, r_ref, aperture)
    """
    if transducer is not None:
        return transducer.p_ref, transducer.r_ref, transducer.aperture_radius
    if env is not None:
        tc = env.transducer
        return tc.p_ref, tc.r_ref, tc.aperture_radius
    # [ASMP-024] Legacy fallback: uses module-level constants, NOT config values.
    import warnings
    warnings.warn(
        "compute_pressure: no EnvironmentConfig or TransducerConfig provided. "
        "Falling back to legacy TRANSDUCER_P0=200 Pa at 0.01 m (differs from "
        "config MA40S4S_40KHZ p_ref=10 Pa at 0.3 m — ~2.25× force discrepancy). "
        "Pass env=EnvironmentConfig() for correct values.",
        DeprecationWarning,
        stacklevel=3,
    )
    return TRANSDUCER_P0, REFERENCE_DISTANCE, TRANSDUCER_APERTURE


def _resolve_wavenumber(
    array: ArrayGeometry,
    env: Optional["EnvironmentConfig"],
) -> float:
    """
    Resolve wavenumber k from EnvironmentConfig or fall back to array.params.

    When env is provided, k = 2*pi*f / c where c comes from temperature.
    Otherwise uses the legacy array.params.wavenumber (hardcoded 343 m/s).
    """
    if env is not None:
        return env.wavenumber
    return array.params.wavenumber


def transducer_directivity(
    angle: NDArray[np.float64],
    wavenumber: float,
    aperture: float = TRANSDUCER_APERTURE,
) -> NDArray[np.float64]:
    """
    Compute transducer directivity pattern.

    Uses the piston source model (sinc function), which is a good
    approximation for circular transducers like the Murata MA40S4S.

    From Ultraino (CalcField.java:49-50):
        dum = aperture * 0.5 * k * sin(angle)
        directivity = sinc(dum)

    Args:
        angle: Angle from transducer normal (radians), shape (N,) or (M, N)
        wavenumber: k = 2pi/lambda
        aperture: Transducer radius (default 5mm for MA40S4S)

    Returns:
        Directivity factor 0-1, same shape as angle

    Physics:
        D(theta) = sinc(ka sin(theta) / 2) where sinc(x) = sin(pi*x)/(pi*x)
        For theta=0 (on-axis): D = 1.0
        For theta=90 deg (perpendicular): D ~ 0.2 for ka~0.3
    """
    # ka * sin(theta) / 2 - argument to sinc
    arg = aperture * wavenumber * np.sin(angle) / 2

    # Physics formula uses sin(x)/x (unnormalized sinc)
    with np.errstate(divide='ignore', invalid='ignore'):
        directivity = np.where(
            np.abs(arg) < 1e-10,
            1.0,
            np.sin(arg) / arg
        )

    return np.abs(directivity)  # Ensure positive


def compute_transducer_normals(array: ArrayGeometry) -> NDArray[np.float64]:
    """
    Compute normal vectors for each transducer (pointing direction).

    For cylindrical arrays, transducers point inward toward the central axis.
    For planar arrays, they point along the z-axis.

    Args:
        array: Transducer array geometry

    Returns:
        Normal vectors, shape (N, 3), unit vectors
    """
    positions = array.positions
    N = array.num_transducers

    # For cylindrical array: normal points toward z-axis (inward)
    # Normal = -[x, y, 0] / |[x, y, 0]|
    xy_dist = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    xy_dist = np.maximum(xy_dist, 1e-10)  # Avoid div by zero

    normals = np.zeros((N, 3))
    normals[:, 0] = -positions[:, 0] / xy_dist  # Point inward in x
    normals[:, 1] = -positions[:, 1] / xy_dist  # Point inward in y
    normals[:, 2] = 0  # No z component (horizontal transducers)

    return normals


def compute_pressure(
    array: ArrayGeometry,
    phases: PhaseArray,
    point: NDArray[np.float64],
    ring_array: Optional["RingArray"] = None,
    use_directivity: bool = True,
    use_physical_units: bool = True,
    boundary_model: Optional["BoundaryReflectionModel"] = None,
    env: Optional["EnvironmentConfig"] = None,
    *,
    backend: Optional["PressureBackend"] = None,
) -> complex:
    """
    Compute complex pressure phasor at a single point.

    Returns the full complex phasor from all transducers superimposed
    at the given point. Use abs() for magnitude, np.angle() for phase.

    Backward compatible: callers that used float(result) or compared
    magnitudes will get equivalent behavior via abs(result).

    Args:
        array: Transducer array geometry
        phases: Phase configuration for all transducers
        point: Query point [x, y, z] in meters
        ring_array: Optional ring configuration for per-ring amplitude/phase modulation.
                   When provided, ring amplitudes and phase offsets are applied.
                   Disabled rings contribute zero amplitude.
        use_directivity: Include transducer directivity pattern (default True)
        use_physical_units: Scale to physical pressure in Pascals (default True)
        boundary_model: Optional BEM model for including boundary reflections.
                       When provided, computes total field = direct + scattered.
        env: Optional EnvironmentConfig for speed of sound and transducer params.
             When None, falls back to array.params.wavenumber and module-level
             TRANSDUCER_P0 / REFERENCE_DISTANCE constants.

    Returns:
        Complex pressure phasor (Pa if use_physical_units=True).
        Magnitude = abs(result), phase = np.angle(result).

    Physics (validated against Ultraino):
        P(r) = sum_i (P0 x A_i x D(theta_i) x r_ref / r_i) x e^(i(k x r_i + phi_i))

        Where:
        - P0 = TransducerConfig.p_ref (or TRANSDUCER_P0 fallback)
        - r_ref = TransducerConfig.r_ref (or REFERENCE_DISTANCE fallback)
        - D(theta) = sinc(ka sin(theta)/2) directivity pattern
        - 1/r spherical spreading
        - k = 2*pi*f / c, with c from EnvironmentConfig.speed_of_sound

        With boundary_model, adds scattered field from cylindrical boundary:
        P_total = P_direct + P_scattered (via Kirchhoff-Helmholtz integral)
    """
    if backend is not None:
        return backend.compute_pressure(
            array, phases, point,
            ring_array=ring_array,
            use_directivity=use_directivity,
            use_physical_units=use_physical_units,
            boundary_model=boundary_model,
            env=env,
        )

    point = np.asarray(point, dtype=np.float64)

    # Resolve config-based params
    p_ref, r_ref, aperture = _resolve_pressure_params(env, None)
    k = _resolve_wavenumber(array, env)
    # [ASMP-061] Acoustic attenuation in air — Bass et al. (1995)
    alpha = _resolve_attenuation(env)

    # Apply ring modulation if provided
    if ring_array is not None:
        effective_amplitudes = ring_array.get_effective_amplitudes() * phases.amplitudes
        effective_phases = (phases.phases + ring_array.get_phase_offsets()) % (2 * np.pi)
    else:
        effective_amplitudes = phases.amplitudes
        effective_phases = phases.phases

    # Vector from each transducer to the point
    # Shape: (N, 3)
    displacements = point - array.positions

    # Distance from each transducer to point
    # Shape: (N,)
    distances = np.linalg.norm(displacements, axis=1)

    # Avoid division by zero (point exactly at transducer)
    distances = np.maximum(distances, 1e-6)

    # Compute directivity if enabled
    if use_directivity:
        # Get transducer normals (pointing direction)
        normals = compute_transducer_normals(array)

        # Normalize displacement vectors
        directions = displacements / distances[:, np.newaxis]

        # Angle between transducer normal and direction to point
        # cos(theta) = normal . direction
        cos_angles = np.sum(normals * directions, axis=1)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles)

        # Compute directivity factor
        directivity = transducer_directivity(angles, k, aperture)
    else:
        directivity = 1.0

    # Absorption decay: exp(-alpha * r) per transducer path length
    attenuation = np.exp(-alpha * distances)

    # Phase contribution from each transducer: k x distance + transducer phase
    phase_contributions = k * distances + effective_phases

    # Amplitude contribution with directivity, spherical spreading, and absorption
    # A_i x D(theta) x exp(-alpha*r) x (r_ref / r) for physical scaling
    if use_physical_units:
        # Physical pressure: P0 at reference distance, 1/r spreading
        amplitude_contributions = (
            p_ref * effective_amplitudes * directivity *
            attenuation * r_ref / distances
        )
    else:
        # Normalized: amplitude / distance
        amplitude_contributions = effective_amplitudes * directivity * attenuation / distances

    # Sum complex phasors: sum A_i x e^(i x phase_i)
    real_sum = np.sum(amplitude_contributions * np.cos(phase_contributions))
    imag_sum = np.sum(amplitude_contributions * np.sin(phase_contributions))

    # Direct pressure (complex phasor)
    P_direct_complex = complex(real_sum, imag_sum)

    # Add scattered field from boundary if model provided
    if boundary_model is not None and boundary_model.config.enabled:
        from .boundary import is_inside_boundary
        if is_inside_boundary(point, boundary_model.config):
            # Get total pressure including reflections
            pressure = boundary_model.compute_total_pressure(
                array, phases, point, ring_array
            )
            return complex(pressure)
        else:
            # Outside boundary — warn instead of silent zero
            warnings.warn(
                f"Evaluation point {point} is outside the boundary model region. "
                f"Returning direct field only (no BEM contribution).",
                stacklevel=2,
            )
            return P_direct_complex

    return P_direct_complex


def compute_pressure_field(
    array: ArrayGeometry,
    phases: PhaseArray,
    points: NDArray[np.float64],
    ring_array: Optional["RingArray"] = None,
    use_directivity: bool = True,
    use_physical_units: bool = True,
    env: Optional["EnvironmentConfig"] = None,
    *,
    backend: Optional["PressureBackend"] = None,
) -> NDArray[np.complexfloating]:
    """
    Compute complex pressure phasor at multiple points (vectorized).

    This is the workhorse function for generating pressure field
    visualizations and force computations.

    Args:
        array: Transducer array geometry
        phases: Phase configuration for all transducers
        points: Query points, shape (M, 3) in meters
        ring_array: Optional ring configuration for per-ring amplitude/phase modulation.
                   When provided, ring amplitudes and phase offsets are applied.
                   Disabled rings contribute zero amplitude.
        use_directivity: Include transducer directivity pattern (default True)
        use_physical_units: Scale to physical pressure in Pascals (default True)
        env: Optional EnvironmentConfig for speed of sound and transducer params.

    Returns:
        Complex pressure phasors, shape (M,). Use np.abs() for magnitudes.

    Performance:
        Vectorized over query points. For M points and N transducers,
        this does O(M x N) operations. For a 100^3 grid (1M points)
        with 390 transducers, this is ~390M operations.

        On modern CPU with NumPy, expect ~100ms for a 100^3 grid.
        For real-time (1kHz), use sparse sampling or GPU compute.

        Ring modulation adds only O(N) preprocessing overhead.
    """
    if backend is not None:
        return backend.compute_pressure_field(
            array, phases, points,
            ring_array=ring_array,
            use_directivity=use_directivity,
            use_physical_units=use_physical_units,
            env=env,
        )

    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 3)

    M = points.shape[0]  # Number of query points
    N = array.num_transducers

    # Resolve config-based params
    p_ref, r_ref, aperture = _resolve_pressure_params(env, None)
    k = _resolve_wavenumber(array, env)
    # [ASMP-061] Acoustic attenuation in air — Bass et al. (1995)
    alpha = _resolve_attenuation(env)

    # Apply ring modulation if provided
    if ring_array is not None:
        effective_amplitudes = ring_array.get_effective_amplitudes() * phases.amplitudes
        effective_phases = (phases.phases + ring_array.get_phase_offsets()) % (2 * np.pi)
    else:
        effective_amplitudes = phases.amplitudes
        effective_phases = phases.phases

    # Broadcast: points (M, 1, 3) - positions (1, N, 3) -> displacements (M, N, 3)
    displacements = points[:, np.newaxis, :] - array.positions[np.newaxis, :, :]

    # Distances: (M, N)
    distances = np.linalg.norm(displacements, axis=2)
    distances = np.maximum(distances, 1e-6)  # Avoid div by zero

    # Compute directivity if enabled
    if use_directivity:
        # Get transducer normals (pointing direction)
        normals = compute_transducer_normals(array)  # (N, 3)

        # Normalize displacement vectors: (M, N, 3)
        directions = displacements / distances[:, :, np.newaxis]

        # Angle between transducer normal and direction to point
        # cos(theta) = normal . direction, broadcast: (M, N)
        cos_angles = np.sum(directions * normals[np.newaxis, :, :], axis=2)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles)  # (M, N)

        # Compute directivity factor
        directivity = transducer_directivity(angles, k, aperture)  # (M, N)
    else:
        directivity = 1.0

    # Absorption decay: exp(-alpha * r) per transducer path length
    attenuation = np.exp(-alpha * distances)  # (M, N)

    # Phase contributions: k x distance + transducer phase
    phase_contributions = k * distances + effective_phases[np.newaxis, :]

    # Amplitude contributions with directivity, spherical spreading, and absorption
    if use_physical_units:
        # Physical pressure: P0 at reference distance, 1/r spreading
        amplitude_contributions = (
            p_ref * effective_amplitudes[np.newaxis, :] * directivity *
            attenuation * r_ref / distances
        )
    else:
        # Normalized: amplitude x directivity / distance
        amplitude_contributions = effective_amplitudes[np.newaxis, :] * directivity * attenuation / distances

    # Sum complex phasors across transducers (axis=1)
    real_sum = np.sum(amplitude_contributions * np.cos(phase_contributions), axis=1)
    imag_sum = np.sum(amplitude_contributions * np.sin(phase_contributions), axis=1)

    # Return complex phasor array
    return real_sum + 1j * imag_sum


def compute_pressure_grid(
    array: ArrayGeometry,
    phases: PhaseArray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    resolution: Union[int, Tuple[int, int, int]] = 50,
    ring_array: Optional["RingArray"] = None,
    env: Optional["EnvironmentConfig"] = None,
    *,
    backend: Optional["PressureBackend"] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.complexfloating]]:
    """
    Compute pressure field on a 3D grid.

    Convenience function for generating volumetric pressure data.

    Args:
        array: Transducer array geometry
        phases: Phase configuration
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max)) in meters
        resolution: Grid points per axis (int or tuple)
        ring_array: Optional ring configuration for per-ring modulation
        env: Optional EnvironmentConfig for speed of sound

    Returns:
        Tuple of (X, Y, Z, P) where:
            X, Y, Z: Coordinate arrays, shape (nx, ny, nz)
            P: Complex pressure phasor array, shape (nx, ny, nz)

    Example:
        X, Y, Z, P = compute_pressure_grid(array, phases,
            bounds=((-0.1, 0.1), (-0.1, 0.1), (-0.2, 0.2)),
            resolution=100)

        # Plot a Z slice (magnitude)
        plt.pcolormesh(X[:, :, 50], Y[:, :, 50], np.abs(P[:, :, 50]))
    """
    if isinstance(resolution, int):
        resolution = (resolution, resolution, resolution)

    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    nx, ny, nz = resolution

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Flatten for vectorized computation
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # Compute pressure
    P_flat = compute_pressure_field(array, phases, points, ring_array=ring_array, env=env, backend=backend)

    # Reshape back to grid
    P = P_flat.reshape(X.shape)

    return X, Y, Z, P


def compute_pressure_slice(
    array: ArrayGeometry,
    phases: PhaseArray,
    slice_axis: str,
    slice_position: float,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    resolution: Union[int, Tuple[int, int]] = 100,
    ring_array: Optional["RingArray"] = None,
    env: Optional["EnvironmentConfig"] = None,
    *,
    backend: Optional["PressureBackend"] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complexfloating]]:
    """
    Compute pressure field on a 2D slice.

    This is the primary visualization function - generates heatmap data
    for a cross-section of the pressure field.

    Args:
        array: Transducer array geometry
        phases: Phase configuration
        slice_axis: Which axis to slice ('x', 'y', or 'z')
        slice_position: Position along slice axis in meters
        bounds: ((axis1_min, axis1_max), (axis2_min, axis2_max))
        resolution: Grid points per axis
        ring_array: Optional ring configuration for per-ring modulation
        env: Optional EnvironmentConfig for speed of sound

    Returns:
        Tuple of (A1, A2, P) where:
            A1, A2: Coordinate arrays for the two non-slice axes
            P: Complex pressure phasor array

    Example:
        # X-Y slice at Z = 0.1m (top-down view at droplet height)
        X, Y, P = compute_pressure_slice(array, phases,
            slice_axis='z', slice_position=0.1,
            bounds=((-0.1, 0.1), (-0.1, 0.1)))

        plt.pcolormesh(X, Y, np.abs(P), cmap='viridis')
    """
    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    (a1_min, a1_max), (a2_min, a2_max) = bounds
    n1, n2 = resolution

    a1 = np.linspace(a1_min, a1_max, n1)
    a2 = np.linspace(a2_min, a2_max, n2)
    A1, A2 = np.meshgrid(a1, a2, indexing='ij')

    # Build 3D points based on slice axis
    slice_axis = slice_axis.lower()
    if slice_axis == 'z':
        # X-Y slice at fixed Z (top-down view)
        points = np.stack([A1.ravel(), A2.ravel(),
                          np.full(A1.size, slice_position)], axis=1)
    elif slice_axis == 'y':
        # X-Z slice at fixed Y (side view)
        points = np.stack([A1.ravel(),
                          np.full(A1.size, slice_position),
                          A2.ravel()], axis=1)
    elif slice_axis == 'x':
        # Y-Z slice at fixed X
        points = np.stack([np.full(A1.size, slice_position),
                          A1.ravel(), A2.ravel()], axis=1)
    else:
        raise ValueError(f"slice_axis must be 'x', 'y', or 'z', got {slice_axis}")

    # Compute pressure
    P_flat = compute_pressure_field(array, phases, points, ring_array=ring_array, env=env, backend=backend)
    P = P_flat.reshape(A1.shape)

    return A1, A2, P
