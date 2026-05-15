"""
Gor'kov force computation for acoustic droplet manipulation.

This module computes the radiation force on a small particle in an
acoustic field using the FULL Gor'kov potential formulation.

Physics (from Derivation 1 and Ultraino validation):
    The time-averaged radiation force on a small sphere is:
        F = -grad(U)

    Where U is the full Gor'kov potential:
        U = V * [f1 * <p^2>/(2*rho*c^2) - f2 * (3*rho/4) * <v^2>]

    The particle velocity <v^2> relates to pressure gradient:
        <v^2> = |grad(p)|^2 / (rho*omega)^2

    Full force (validated against Ultraino CalcField.java):
        F = -grad(U) = K1 * p*grad(p) - K2 * (grad(p) . grad^2(p))

    Where (from Ultraino):
        K1 = (4*pi/3)*a^3 * f1 * kappa  (pressure/monopole term)
        K2 = (4*pi/3)*a^3 * f2 * (3*rho/4) * (1/(rho*omega))^2  (velocity/dipole term)

    Bruus f1/f2 coefficients (config-derived, NOT hardcoded):
        f1 = 1 - (rho_medium * c_medium^2) / (rho_particle * c_particle^2)
        f2 = 2*(rho_particle - rho_medium) / (2*rho_particle + rho_medium)

Key insight:
    The phase gradient method creates BOTH a pressure gradient AND
    a velocity field. Both contribute to the Gor'kov force.

THIS IS THE CRITICAL MODULE for validating the inverse solver.
"""

import warnings

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, TYPE_CHECKING

from .core import (
    ArrayGeometry, PhaseArray, DropletState,
    AIR_DENSITY, SPEED_OF_SOUND, DEFAULT_FREQUENCY,
    acoustic_contrast_factor, AL_DENSITY
)
from .config import (
    EnvironmentConfig,
    DropletConfig,
    MaterialConfig,
    ALUMINUM_LIQUID,
)
from .pressure import compute_pressure, compute_pressure_field
from .trace import sim_trace

if TYPE_CHECKING:
    from .core import RingArray
    from .boundary import BoundaryConfig
    from .pressure_backend import PressureBackend


# Default step size for numerical gradient (in meters)
# Should be << wavelength (~8.6mm) but >> numerical precision
DEFAULT_GRADIENT_STEP = 0.5e-3  # 0.5 mm


def _derive_f1_f2(
    particle_material: MaterialConfig,
    medium: MaterialConfig,
    speed_of_sound_medium: float,
    omega: float = 0.0,
    particle_radius: float = 0.0,
    apply_viscous_correction: bool = True,
) -> Tuple[float, float]:
    """
    Compute Bruus f1/f2 coefficients from material configs.

    f1 = 1 - (rho_medium * c_medium^2) / (rho_particle * c_particle^2)
    f2 = 2*(rho_particle - rho_medium) / (2*rho_particle + rho_medium)

    When ``apply_viscous_correction`` is True and the medium has non-zero
    dynamic viscosity, f2 receives the first-order Settnes-Bruus (2012)
    viscous boundary layer correction:

        delta_v = sqrt(2 * nu / omega)       # viscous penetration depth
        delta_tilde = delta_v / a             # dimensionless ratio
        f2_corrected = f2_inviscid * (1 + 3/2 * delta_tilde)

    For 300 um Al droplets in air at 40 kHz, delta_tilde ~ 0.07, giving
    a ~10% increase in f2 (stronger dipole contribution).

    [ASMP-060] Settnes-Bruus viscous f2 correction (2012)

    Args:
        particle_material: MaterialConfig for the particle.
        medium: MaterialConfig for the surrounding medium.
        speed_of_sound_medium: Speed of sound in the medium [m/s].
            Passed explicitly because EnvironmentConfig derives this from
            temperature, whereas MaterialConfig.sound_speed is derived from
            bulk_modulus/density (they may differ slightly for air).
        omega: Angular frequency [rad/s]. Required for viscous correction.
        particle_radius: Particle radius [m]. Required for viscous correction.
        apply_viscous_correction: If True (default), apply Settnes-Bruus
            first-order viscous correction to f2 when medium viscosity > 0.

    Returns:
        (f1, f2)
    """
    rho_m = medium.density
    c_m = speed_of_sound_medium
    rho_p = particle_material.density
    c_p = particle_material.sound_speed  # derived from bulk_modulus / density

    f1 = 1.0 - (rho_m * c_m ** 2) / (rho_p * c_p ** 2)
    f2 = 2.0 * (rho_p - rho_m) / (2.0 * rho_p + rho_m)

    # [ASMP-060] Settnes-Bruus viscous f2 correction (2012)
    # First-order correction for viscous boundary layer around the particle.
    if (
        apply_viscous_correction
        and medium.dynamic_viscosity > 0.0
        and omega > 0.0
        and particle_radius > 0.0
    ):
        nu = medium.dynamic_viscosity / rho_m  # kinematic viscosity [m^2/s]
        delta_v = np.sqrt(2.0 * nu / omega)    # viscous penetration depth [m]
        delta_tilde = delta_v / particle_radius  # dimensionless ratio

        f2_inviscid = f2
        f2 = f2_inviscid * (1.0 + 1.5 * delta_tilde)

        sim_trace.log(
            "viscous_f2_correction",
            delta_tilde=float(delta_tilde),
            f2_inviscid=float(f2_inviscid),
            f2_corrected=float(f2),
            correction_pct=float(1.5 * delta_tilde * 100.0),
            medium=medium.name,
        )

    return f1, f2


def compute_gorkov_coefficients(
    particle_radius: float,
    particle_density: float = AL_DENSITY,
    medium_density: float = AIR_DENSITY,
    speed_of_sound: float = SPEED_OF_SOUND,
    frequency: float = DEFAULT_FREQUENCY,
    particle_speed: Optional[float] = None,
    *,
    environment: Optional[EnvironmentConfig] = None,
    droplet_config: Optional[DropletConfig] = None,
) -> Tuple[float, float]:
    """
    Compute Gor'kov force coefficients M1 and M2.

    NEW (config-first): Pass ``environment`` and ``droplet_config`` to
    derive f1/f2 from MaterialConfig objects. This ensures the correct
    sound speed is used for each material (e.g. 4700 m/s for liquid Al,
    ~110 m/s for styrofoam).

    BACKWARD COMPAT: When config objects are not provided, the function
    falls back to the scalar arguments. If ``particle_speed`` is also
    omitted, it defaults to ALUMINUM_LIQUID.sound_speed (4700 m/s).

    Args:
        particle_radius: Particle radius in meters
        particle_density: Particle density (kg/m^3) [legacy]
        medium_density: Medium (air) density (kg/m^3) [legacy]
        speed_of_sound: Sound speed in medium (m/s) [legacy]
        frequency: Acoustic frequency (Hz) [legacy]
        particle_speed: Sound speed in particle (m/s) [legacy, deprecated]
        environment: EnvironmentConfig (preferred)
        droplet_config: DropletConfig (preferred)

    Returns:
        (M1, M2) - coefficients for pressure and velocity terms
    """
    # --- Determine f1, f2, and scalar parameters from config or legacy args ---
    if environment is not None and droplet_config is not None:
        # Config-first path: derive everything from config objects
        particle_mat = droplet_config.material
        medium_mat = environment.medium
        c_medium = environment.speed_of_sound
        rho_m = medium_mat.density
        rho_p = particle_mat.density
        freq = environment.transducer.frequency
        radius = droplet_config.radius

        omega_for_f2 = 2.0 * np.pi * freq
        f1, f2 = _derive_f1_f2(
            particle_mat, medium_mat, c_medium,
            omega=omega_for_f2,
            particle_radius=radius,
        )
    else:
        # Legacy scalar path
        if particle_speed is None:
            # Default to liquid aluminum sound speed (was 6420 solid -- WRONG)
            particle_speed = ALUMINUM_LIQUID.sound_speed  # 4700 m/s
            warnings.warn(
                "compute_gorkov_coefficients: particle_speed not provided and "
                "no config objects given. Defaulting to ALUMINUM_LIQUID "
                f"sound speed ({particle_speed:.0f} m/s). Pass environment= "
                "and droplet_config= for correct material-specific coefficients.",
                stacklevel=2,
            )
        rho_m = medium_density
        rho_p = particle_density
        c_medium = speed_of_sound
        freq = frequency
        radius = particle_radius

        # Compute f1/f2 from scalar values
        kappa_m = 1.0 / (rho_m * c_medium ** 2)
        kappa_p = 1.0 / (rho_p * particle_speed ** 2)
        f1 = 1.0 - kappa_p / kappa_m
        f2 = (2.0 * (rho_p / rho_m - 1.0)) / (2.0 * rho_p / rho_m + 1.0)

    # [ASMP-059] Re-check ka validity when material-aware coefficients are computed
    if environment is not None and droplet_config is not None:
        ka = environment.transducer.wavenumber(environment.speed_of_sound) * droplet_config.radius
        if ka > 0.5:
            warnings.warn(
                f"[ASMP-059] Gor'kov ka = {ka:.3f} > 0.5 for "
                f"'{droplet_config.material.name}' — force accuracy degrades. "
                f"Consider larger wavelength or smaller droplet.",
                stacklevel=2,
            )

    # --- Compute M1, M2 from f1, f2 ---
    V = (4.0 / 3.0) * np.pi * radius ** 3
    kappa = 1.0 / (rho_m * c_medium ** 2)
    omega = 2.0 * np.pi * freq
    pressure_to_velocity = 1.0 / (rho_m * omega)

    # [ASMP-010] Gor'kov coefficients (matching Ultraino CalcField.java:129-134)
    # Extra 0.5 factors are time-averaging: compute_pressure returns peak |P|, <p^2>=|P|^2/2
    M1 = V * f1 * 0.5 * kappa * 0.5
    M2 = V * f2 * (3.0 / 4.0) * rho_m * 0.5 * pressure_to_velocity ** 2

    return M1, M2


def _check_stencil_inside_cylinder(
    position: NDArray[np.float64],
    gradient_step: float,
    cylinder_radius: float,
) -> bool:
    """
    Check whether all 6 gradient stencil sample points lie inside the
    cylindrical array boundary.

    The cylinder boundary is defined as x^2 + y^2 < cylinder_radius^2.
    Z-bounds are not checked here (array height is variable and transducer
    positions are the authority for Z extent).

    Args:
        position: Centre of the stencil [x, y, z].
        gradient_step: The step size h used for central differences.
        cylinder_radius: Radius of the transducer array cylinder [m].

    Returns:
        True if ALL 6 stencil points are inside the cylinder, False otherwise.
        Emits a warning when a stencil point falls outside.
    """
    h = gradient_step
    r_cyl_sq = cylinder_radius ** 2
    all_inside = True

    offsets = [
        (h, 0.0), (-h, 0.0),   # +x, -x
        (0.0, h), (0.0, -h),   # +y, -y
    ]
    # +z and -z do not change the XY radius, so they are always inside
    # if the centre is inside.

    for dx, dy in offsets:
        sx = position[0] + dx
        sy = position[1] + dy
        r_sq = sx ** 2 + sy ** 2
        if r_sq >= r_cyl_sq:
            all_inside = False
            break  # one warning is enough

    if not all_inside:
        r_center = float(np.sqrt(position[0] ** 2 + position[1] ** 2))
        warnings.warn(
            f"Gradient stencil point outside cylinder boundary "
            f"(r_center={r_center*1e3:.1f} mm, step={h*1e3:.2f} mm, "
            f"cylinder_radius={cylinder_radius*1e3:.1f} mm). "
            f"Force computation may produce boundary artifacts.",
            stacklevel=3,
        )

    return all_inside


def compute_force(
    array: ArrayGeometry,
    phases: PhaseArray,
    position: NDArray[np.float64],
    droplet: Optional[DropletState] = None,
    gradient_step: float = DEFAULT_GRADIENT_STEP,
    return_magnitude: bool = False,
    ring_array: Optional["RingArray"] = None,
    use_full_gorkov: bool = False,
    boundary_config: Optional["BoundaryConfig"] = None,
    *,
    environment: Optional[EnvironmentConfig] = None,
    droplet_config: Optional[DropletConfig] = None,
    field_output: Optional[dict] = None,
    backend: Optional["PressureBackend"] = None,
) -> NDArray[np.float64]:
    """
    Compute FULL Gor'kov radiation force on a droplet at given position.

    This is the KEY function for validating the inverse solver.
    Given phases computed to push a droplet toward a target, this
    function should return a force vector pointing toward that target.

    Config-first usage (preferred):
        Pass ``environment`` and ``droplet_config`` keyword arguments so
        that f1/f2 coefficients are derived from MaterialConfig objects.

    Legacy usage (backward-compatible):
        Pass ``droplet`` (DropletState) for physical properties. The old
        scalar-based coefficient path is used with a deprecation warning
        when config objects are absent.

    Args:
        array: Transducer array geometry
        phases: Phase configuration
        position: Droplet position [x, y, z] in meters
        droplet: Optional droplet state (for physical properties) [legacy]
        gradient_step: Step size for numerical gradient (meters)
        return_magnitude: If True, also return force magnitude
        ring_array: Optional ring configuration for per-ring modulation
        use_full_gorkov: If True, include velocity term (default True)
        boundary_config: Optional boundary configuration for masking
        environment: EnvironmentConfig (preferred for f1/f2 derivation)
        droplet_config: DropletConfig (preferred for f1/f2 derivation)
        field_output: If provided (empty dict), filled in-place with
            'p_center' (float), 'grad_p' (NDArray shape (3,)),
            and 'grad_p_magnitude' (float). Allows the thermal model
            to access pressure gradient data without re-computing it.

    Returns:
        Force vector [Fx, Fy, Fz] in Newtons (if droplet or droplet_config provided)
        or normalized direction (if neither is provided)
    """
    position = np.asarray(position, dtype=np.float64)
    h = gradient_step

    # --- Boundary check: warn explicitly instead of silent zeros ---
    if boundary_config is not None:
        from .boundary import is_inside_boundary
        if not is_inside_boundary(position, boundary_config):
            warnings.warn(
                f"compute_force: position {position} is outside the array "
                f"boundary. Returning zero force.",
                stacklevel=2,
            )
            zero_force = np.zeros(3)
            if return_magnitude:
                return zero_force, 0.0
            return zero_force

    # --- Gradient stencil boundary check ---
    # Infer cylinder radius from the array geometry (max radial distance
    # of any transducer from the z-axis).
    if array.positions.shape[0] > 0:
        r_transducers = np.sqrt(
            array.positions[:, 0] ** 2 + array.positions[:, 1] ** 2
        )
        cylinder_radius = float(np.max(r_transducers))
        _check_stencil_inside_cylinder(position, h, cylinder_radius)

    # Sample pressure at 7 points: center + 6 neighbors for gradient.
    # compute_pressure returns complex phasors. We keep both complex values
    # and magnitudes:
    #   - Monopole term needs grad(|p|^2) — uses magnitudes
    #   - Dipole/velocity term needs |grad(p_complex)|^2 — uses complex phasors
    #
    # [D1-H1] The velocity term requires the gradient of the COMPLEX pressure
    # field, not the gradient of the magnitude. Near pressure nodes |p| -> 0,
    # the magnitude gradient vanishes but the complex gradient can be large
    # (rapid phase variation). Using |grad(p_complex)|^2 correctly captures
    # the velocity field contribution at nodes.
    #
    # [ASMP-031] FIXED: env forwarded to all compute_pressure calls below.
    p_center_complex = compute_pressure(array, phases, position, ring_array=ring_array, env=environment, backend=backend)
    p_center = abs(p_center_complex)

    # Compute pressure at +/- h in each direction — keep complex phasors
    p_complex_samples = {}
    for axis, name in enumerate(['x', 'y', 'z']):
        pos_plus = position.copy()
        pos_plus[axis] += h
        pos_minus = position.copy()
        pos_minus[axis] -= h
        p_complex_samples[f'{name}+'] = compute_pressure(array, phases, pos_plus, ring_array=ring_array, env=environment, backend=backend)
        p_complex_samples[f'{name}-'] = compute_pressure(array, phases, pos_minus, ring_array=ring_array, env=environment, backend=backend)

    # Compute gradient of |p| (magnitude first derivatives) for field_output
    grad_p = np.array([
        (abs(p_complex_samples['x+']) - abs(p_complex_samples['x-'])) / (2 * h),
        (abs(p_complex_samples['y+']) - abs(p_complex_samples['y-'])) / (2 * h),
        (abs(p_complex_samples['z+']) - abs(p_complex_samples['z-'])) / (2 * h),
    ])

    # Expose pressure field data for thermal model (no extra computation)
    if field_output is not None:
        field_output['p_center'] = p_center
        field_output['grad_p'] = grad_p.copy()
        field_output['grad_p_magnitude'] = float(np.linalg.norm(grad_p))

    # Monopole term: gradient of |p|^2 = gradient of (p * conj(p))
    grad_p2 = np.array([
        (abs(p_complex_samples['x+'])**2 - abs(p_complex_samples['x-'])**2) / (2 * h),
        (abs(p_complex_samples['y+'])**2 - abs(p_complex_samples['y-'])**2) / (2 * h),
        (abs(p_complex_samples['z+'])**2 - abs(p_complex_samples['z-'])**2) / (2 * h),
    ])

    # Velocity/dipole term: need gradient of |grad(p_complex)|^2
    # where grad(p_complex) is the gradient of the complex pressure phasor.
    # |grad(p_complex)|^2 = |dp/dx|^2 + |dp/dy|^2 + |dp/dz|^2
    if use_full_gorkov:
        grad_p_squared_samples = {}
        for axis, name in enumerate(['x', 'y', 'z']):
            for sign, suffix in [(+h, '+'), (-h, '-')]:
                pos_offset = position.copy()
                pos_offset[axis] += sign

                # Compute complex gradient at each offset point
                grad_complex_at_offset = np.zeros(3, dtype=complex)
                for axis2, name2 in enumerate(['x', 'y', 'z']):
                    pos_pp = pos_offset.copy()
                    pos_pp[axis2] += h
                    pos_pm = pos_offset.copy()
                    pos_pm[axis2] -= h
                    p_pp = compute_pressure(array, phases, pos_pp, ring_array=ring_array, env=environment, backend=backend)
                    p_pm = compute_pressure(array, phases, pos_pm, ring_array=ring_array, env=environment, backend=backend)
                    grad_complex_at_offset[axis2] = (p_pp - p_pm) / (2 * h)

                # |grad(p_complex)|^2 = sum of |dp/dx_i|^2
                grad_p_squared_samples[f'{name}{suffix}'] = float(
                    np.sum(np.abs(grad_complex_at_offset)**2)
                )

        grad_grad_p_squared = np.array([
            (grad_p_squared_samples['x+'] - grad_p_squared_samples['x-']) / (2 * h),
            (grad_p_squared_samples['y+'] - grad_p_squared_samples['y-']) / (2 * h),
            (grad_p_squared_samples['z+'] - grad_p_squared_samples['z-']) / (2 * h),
        ])
    else:
        grad_grad_p_squared = np.zeros(3)

    # --- Compute Gor'kov coefficients ---
    has_config = environment is not None and droplet_config is not None
    has_legacy = droplet is not None

    if has_config:
        M1, M2 = compute_gorkov_coefficients(
            particle_radius=droplet_config.radius,
            environment=environment,
            droplet_config=droplet_config,
        )
        force = -M1 * grad_p2 + M2 * grad_grad_p_squared

        force_mag = float(np.linalg.norm(force))
        if force_mag < 1e-30:
            warnings.warn(
                "compute_force: resulting force is effectively zero "
                f"(|F| = {force_mag:.2e} N). Check phase configuration "
                "and droplet position.",
                stacklevel=2,
            )

        if return_magnitude:
            return force, force_mag
        return force

    if has_legacy:
        M1, M2 = compute_gorkov_coefficients(
            particle_radius=droplet.radius,
            particle_density=droplet.density,
            frequency=array.params.frequency,
        )
        force = -M1 * grad_p2 + M2 * grad_grad_p_squared

        force_mag = float(np.linalg.norm(force))
        if force_mag < 1e-30:
            warnings.warn(
                "compute_force: resulting force is effectively zero "
                f"(|F| = {force_mag:.2e} N). Check phase configuration "
                "and droplet position.",
                stacklevel=2,
            )

        if return_magnitude:
            return force, force_mag
        return force

    # No droplet info at all -- return normalized direction from pressure term
    force_direction = -grad_p2
    magnitude = np.linalg.norm(force_direction)
    if magnitude < 1e-10:
        warnings.warn(
            "compute_force: pressure gradient is effectively zero at the "
            "requested position. Force direction is indeterminate.",
            stacklevel=2,
        )
    else:
        force_direction = force_direction / magnitude

    if return_magnitude:
        return force_direction, magnitude
    return force_direction


def compute_force_field(
    array: ArrayGeometry,
    phases: PhaseArray,
    points: NDArray[np.float64],
    gradient_step: float = DEFAULT_GRADIENT_STEP,
    ring_array: Optional["RingArray"] = None,
    *,
    environment: Optional[EnvironmentConfig] = None,
    droplet_config: Optional[DropletConfig] = None,
    backend: Optional["PressureBackend"] = None,
) -> NDArray[np.float64]:
    """
    Compute force field at multiple points (for quiver plots).

    Args:
        array: Transducer array geometry
        phases: Phase configuration
        points: Query points, shape (M, 3)
        gradient_step: Step size for numerical gradient
        ring_array: Optional ring configuration for per-ring modulation
        environment: Optional EnvironmentConfig for config-derived coefficients
        droplet_config: Optional DropletConfig for config-derived coefficients

    Returns:
        Force directions, shape (M, 3) - normalized vectors
    """
    points = np.asarray(points, dtype=np.float64)
    M = points.shape[0]
    forces = np.zeros((M, 3))
    h = gradient_step

    # Infer cylinder radius for stencil boundary check
    cylinder_radius = None
    if array.positions.shape[0] > 0:
        r_transducers = np.sqrt(
            array.positions[:, 0] ** 2 + array.positions[:, 1] ** 2
        )
        cylinder_radius = float(np.max(r_transducers))

    for i, pos in enumerate(points):
        # Check stencil boundary
        if cylinder_radius is not None:
            _check_stencil_inside_cylinder(pos, h, cylinder_radius)

        # 6-point stencil for gradient
        offsets = np.array([
            [h, 0, 0], [-h, 0, 0],
            [0, h, 0], [0, -h, 0],
            [0, 0, h], [0, 0, -h]
        ])

        sample_points = pos + offsets
        pressures_raw = compute_pressure_field(array, phases, sample_points, ring_array=ring_array, env=environment, backend=backend)
        pressures = np.abs(pressures_raw)
        p2 = pressures ** 2

        # Central difference gradient
        grad_p2 = np.array([
            (p2[0] - p2[1]) / (2 * h),
            (p2[2] - p2[3]) / (2 * h),
            (p2[4] - p2[5]) / (2 * h),
        ])

        # Force direction (negative gradient)
        force = -grad_p2
        magnitude = np.linalg.norm(force)
        if magnitude > 1e-10:
            force = force / magnitude

        forces[i] = force

    return forces


def compute_force_slice(
    array: ArrayGeometry,
    phases: PhaseArray,
    slice_axis: str,
    slice_position: float,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    resolution: int = 20,
    gradient_step: float = DEFAULT_GRADIENT_STEP,
    ring_array: Optional["RingArray"] = None,
    *,
    backend: Optional["PressureBackend"] = None,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Compute force field on a 2D slice (for quiver plots).

    Returns force vectors at a grid of points on the specified slice.

    Args:
        array: Transducer array geometry
        phases: Phase configuration
        slice_axis: Which axis to slice ('x', 'y', or 'z')
        slice_position: Position along slice axis in meters
        bounds: ((axis1_min, axis1_max), (axis2_min, axis2_max))
        resolution: Grid points per axis (keep low for quiver readability)
        gradient_step: Step size for numerical gradient
        ring_array: Optional ring configuration for per-ring modulation

    Returns:
        Tuple of (A1, A2, F1, F2) where:
            A1, A2: Coordinate grids for the slice
            F1, F2: Force components in the slice plane
    """
    (a1_min, a1_max), (a2_min, a2_max) = bounds

    a1 = np.linspace(a1_min, a1_max, resolution)
    a2 = np.linspace(a2_min, a2_max, resolution)
    A1, A2 = np.meshgrid(a1, a2, indexing='ij')

    # Build 3D points based on slice axis
    slice_axis = slice_axis.lower()
    if slice_axis == 'z':
        points = np.stack([A1.ravel(), A2.ravel(),
                          np.full(A1.size, slice_position)], axis=1)
        axis_indices = (0, 1)
    elif slice_axis == 'y':
        points = np.stack([A1.ravel(),
                          np.full(A1.size, slice_position),
                          A2.ravel()], axis=1)
        axis_indices = (0, 2)
    elif slice_axis == 'x':
        points = np.stack([np.full(A1.size, slice_position),
                          A1.ravel(), A2.ravel()], axis=1)
        axis_indices = (1, 2)
    else:
        raise ValueError(f"slice_axis must be 'x', 'y', or 'z', got {slice_axis}")

    forces = compute_force_field(array, phases, points, gradient_step, ring_array=ring_array, backend=backend)

    F1 = forces[:, axis_indices[0]].reshape(A1.shape)
    F2 = forces[:, axis_indices[1]].reshape(A1.shape)

    return A1, A2, F1, F2


def force_toward_target(
    array: ArrayGeometry,
    phases: PhaseArray,
    droplet_position: NDArray[np.float64],
    target_position: NDArray[np.float64],
    ring_array: Optional["RingArray"] = None,
) -> Tuple[float, NDArray[np.float64]]:
    """
    Check if computed force points toward target.

    Validation function: returns the alignment between the force direction
    and the direction to target. Perfect alignment = 1.0, opposite = -1.0.

    Args:
        array: Transducer array geometry
        phases: Phase configuration
        droplet_position: Current droplet position
        target_position: Desired landing position
        ring_array: Optional ring configuration for per-ring modulation

    Returns:
        Tuple of (alignment, force_direction) where:
            alignment: Dot product of force direction and target direction
                       1.0 = perfect, 0.0 = perpendicular, -1.0 = opposite
            force_direction: Normalized force vector
    """
    droplet_position = np.asarray(droplet_position)
    target_position = np.asarray(target_position)

    # Direction from droplet to target
    to_target = target_position - droplet_position
    distance = np.linalg.norm(to_target)
    if distance < 1e-10:
        return 1.0, np.zeros(3)  # Already at target

    to_target = to_target / distance

    # Compute force direction
    force_dir = compute_force(array, phases, droplet_position, ring_array=ring_array)
    force_magnitude = np.linalg.norm(force_dir)
    if force_magnitude < 1e-10:
        warnings.warn(
            "force_toward_target: computed force magnitude is effectively "
            "zero. Alignment is indeterminate.",
            stacklevel=2,
        )
        return 0.0, force_dir

    force_dir = force_dir / force_magnitude

    alignment = float(np.dot(force_dir, to_target))
    return alignment, force_dir
