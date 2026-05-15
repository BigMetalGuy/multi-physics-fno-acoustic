"""
Core data types for Drip physics engine.

This module defines the fundamental data structures used throughout
the physics simulation. All types are designed to be:
- Immutable where possible (use dataclasses with frozen=True)
- NumPy-friendly (positions as arrays)
- Serializable (for export/caching)

Physics constants from Derivation 4:
- Frequency: 40 kHz (standard ultrasonic transducer)
- Wavelength: 8.575 mm (at 343 m/s sound speed)
- Typical droplet: 300 µm Al, 34 ng mass

Config system (SW-16):
    All physical constants are defined in config.py. The module-level
    constants below are retained for backward compatibility but should
    be migrated to config objects in downstream code.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
from numpy.typing import NDArray

# Import the config system — canonical source of truth
from .config import (
    # Constants
    GRAVITY,
    # Material profiles
    MaterialConfig,
    STYROFOAM,
    ALUMINUM_LIQUID,
    ALUMINUM_SOLID,
    AIR_20C,
    # Transducer
    TransducerConfig,
    MA40S4S_40KHZ,
    # Array
    ArrayConfig,
    L1_ARRAY,
    # Droplet
    DropletConfig,
    aluminum_droplet,
    styrofoam_bead,
    # Thermal
    ThermalConfig,
    DEFAULT_THERMAL,
    # Environment
    EnvironmentConfig,
    default_environment,
    styrofoam_test_environment,
    aluminum_print_environment,
)

# =============================================================================
# Physical Constants (backward-compat — prefer config objects)
# =============================================================================

SPEED_OF_SOUND = 343.0  # m/s in air at ~20°C — use EnvironmentConfig.speed_of_sound
DEFAULT_FREQUENCY = 40_000.0  # Hz — use TransducerConfig.frequency
DEFAULT_WAVELENGTH = SPEED_OF_SOUND / DEFAULT_FREQUENCY  # ~8.575 mm

# Aluminum droplet properties (from Derivation 4) — use MaterialConfig profiles
AL_DENSITY = 2385.0  # kg/m³ (liquid aluminum) — use ALUMINUM_LIQUID.density
AL_DEFAULT_DIAMETER = 300e-6  # m (300 µm) — use DropletConfig.diameter
AIR_DENSITY = 1.2  # kg/m³ — use AIR_20C.density


# =============================================================================
# Core Data Types
# =============================================================================

@dataclass
class SimulationParams:
    """
    Global simulation parameters.

    These are the physics constants that apply to the entire simulation.
    Changing these affects wavelength and thus all phase calculations.

    Attributes:
        frequency: Transducer drive frequency in Hz (default 40 kHz)
        speed_of_sound: Sound speed in medium in m/s (default 343, air)

    Derived:
        wavelength: λ = c / f
        wavenumber: k = 2π / λ
        angular_freq: ω = 2πf
    """
    frequency: float = DEFAULT_FREQUENCY
    speed_of_sound: float = SPEED_OF_SOUND

    @property
    def wavelength(self) -> float:
        """Wavelength in meters."""
        return self.speed_of_sound / self.frequency

    @property
    def wavenumber(self) -> float:
        """Wavenumber k = 2π/λ in rad/m."""
        return 2 * np.pi / self.wavelength

    @property
    def angular_frequency(self) -> float:
        """Angular frequency ω = 2πf in rad/s."""
        return 2 * np.pi * self.frequency


@dataclass
class ArrayGeometry:
    """
    Transducer array configuration.

    Stores the positions of all transducers in the array. Positions are
    in meters, in a right-handed coordinate system where Z is up (vertical).

    For Drip's cylindrical array:
    - Transducers arranged in horizontal rings
    - Each ring at a different Z height
    - Transducers point inward toward the central axis

    Attributes:
        positions: (N, 3) array of transducer positions [x, y, z] in meters
        num_transducers: Number of transducers (computed from positions)
        params: Simulation parameters (frequency, sound speed)

    Class Methods:
        cylinder(): Create a cylindrical array
        planar(): Create a planar array (two opposing plates)
    """
    positions: NDArray[np.float64]  # Shape (N, 3)
    params: SimulationParams = field(default_factory=SimulationParams)

    def __post_init__(self):
        """Validate array geometry."""
        self.positions = np.asarray(self.positions, dtype=np.float64)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"positions must be (N, 3), got {self.positions.shape}")

    @property
    def num_transducers(self) -> int:
        """Number of transducers in array."""
        return self.positions.shape[0]

    @property
    def n(self) -> int:
        """Alias for num_transducers."""
        return self.num_transducers

    @classmethod
    def cylinder(
        cls,
        radius: float = 0.05,
        height: float = 0.4,
        rings: int = 10,
        per_ring: int = 12,
        params: Optional[SimulationParams] = None,
    ) -> "ArrayGeometry":
        """
        Create a cylindrical transducer array.

        Standard Drip L1 configuration: 10 rings × 12 transducers = 120 total
        Full Drip configuration: 390 transducers

        Args:
            radius: Cylinder radius in meters (default 5 cm)
            height: Cylinder height in meters (default 40 cm)
            rings: Number of horizontal rings (default 10)
            per_ring: Transducers per ring (default 12)
            params: Simulation parameters

        Returns:
            ArrayGeometry with transducers arranged cylindrically
        """
        from .geometry import create_cylindrical_array
        positions = create_cylindrical_array(radius, height, rings, per_ring)
        return cls(positions, params or SimulationParams())

    @classmethod
    def planar(
        cls,
        extent: float = 0.1,
        spacing: float = 0.01,
        gap: float = 0.2,
        params: Optional[SimulationParams] = None,
    ) -> "ArrayGeometry":
        """
        Create a planar transducer array (two opposing plates).

        Used for acoustic levitation experiments and testing.

        Args:
            extent: Half-width of each plate in meters
            spacing: Transducer spacing in meters
            gap: Distance between plates in meters
            params: Simulation parameters

        Returns:
            ArrayGeometry with two opposing planar arrays
        """
        from .geometry import create_planar_array
        positions = create_planar_array(extent, spacing, gap)
        return cls(positions, params or SimulationParams())


@dataclass
class PhaseArray:
    """
    Phase configuration for all transducers.

    Stores the phase offset (0 to 2π) and amplitude for each transducer.
    This is the control input to the acoustic system.

    Attributes:
        phases: (N,) array of phase offsets in radians [0, 2π]
        amplitudes: (N,) array of amplitude multipliers (default all 1.0)
        timestamp: Optional timestamp for sequenced playback (seconds)

    The phase determines when each transducer's wave reaches a point in space.
    By coordinating phases, we create constructive interference (focus) or
    pressure gradients (for pushing droplets).

    From Derivation 4, the phase gradient method uses:
        φᵢ = -k|rᵢ - r_focus| + αₓ·xᵢ + αᵧ·yᵢ

    Where:
        - First term: focuses waves at r_focus
        - Second/third terms: create lateral force via pressure gradient
    """
    phases: NDArray[np.float64]  # Shape (N,), values in [0, 2π]
    amplitudes: Optional[NDArray[np.float64]] = None  # Shape (N,), default 1.0
    timestamp: Optional[float] = None  # For sequenced playback

    def __post_init__(self):
        """Validate and normalize phases."""
        self.phases = np.asarray(self.phases, dtype=np.float64)
        if self.phases.ndim != 1:
            raise ValueError(f"phases must be 1D, got shape {self.phases.shape}")

        # Normalize phases to [0, 2π]
        self.phases = self.phases % (2 * np.pi)

        # Default amplitudes to 1.0
        if self.amplitudes is None:
            self.amplitudes = np.ones_like(self.phases)
        else:
            self.amplitudes = np.asarray(self.amplitudes, dtype=np.float64)
            if self.amplitudes.shape != self.phases.shape:
                raise ValueError("amplitudes must match phases shape")

    @property
    def n(self) -> int:
        """Number of transducers."""
        return len(self.phases)

    def to_uint8(self) -> NDArray[np.uint8]:
        """Convert phases to uint8 (0-255 maps to 0-2π) for hardware."""
        return np.round(self.phases * 255 / (2 * np.pi)).astype(np.uint8)

    def to_uint16(self) -> NDArray[np.uint16]:
        """Convert phases to uint16 (0-65535 maps to 0-2π) for hardware."""
        return np.round(self.phases * 65535 / (2 * np.pi)).astype(np.uint16)

    @classmethod
    def from_uint8(cls, data: NDArray[np.uint8]) -> "PhaseArray":
        """Create PhaseArray from uint8 hardware format."""
        phases = data.astype(np.float64) * (2 * np.pi) / 255
        return cls(phases)

    @classmethod
    def zeros(cls, n: int) -> "PhaseArray":
        """Create a PhaseArray with all phases at 0."""
        return cls(np.zeros(n))


@dataclass
class DropletState:
    """
    State of a single droplet in the simulation.

    Tracks position, velocity, and physical properties of a droplet
    as it falls through the acoustic field.

    Attributes:
        position: [x, y, z] position in meters
        velocity: [vx, vy, vz] velocity in m/s
        diameter: Droplet diameter in meters (default 300 µm)
        density: Droplet material density in kg/m³ (default Al liquid)
        target: Optional [x, y, z] target landing position

    Derived:
        mass: Computed from diameter and density
        weight: Mass × gravity

    From Derivation 4:
        - 300 µm Al droplet has mass ~34 ng
        - Weight ~0.33 µN
        - Acoustic forces at 10-40% of weight are sufficient for guidance
    """
    position: NDArray[np.float64]  # Shape (3,) [x, y, z]
    velocity: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    diameter: float = AL_DEFAULT_DIAMETER
    density: float = AL_DENSITY
    target: Optional[NDArray[np.float64]] = None  # Target landing position
    temperature: float = 1123.15        # K (850°C default, molten Al ejection)
    liquid_fraction: float = 1.0        # 1.0 = fully liquid, 0.0 = fully solid

    def __post_init__(self):
        """Ensure arrays are proper numpy arrays."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        if self.target is not None:
            self.target = np.asarray(self.target, dtype=np.float64)

    @property
    def radius(self) -> float:
        """Droplet radius in meters."""
        return self.diameter / 2

    @property
    def volume(self) -> float:
        """Droplet volume in m³."""
        return (4/3) * np.pi * self.radius**3

    @property
    def mass(self) -> float:
        """Droplet mass in kg."""
        return self.density * self.volume

    @property
    def weight(self) -> float:
        """Droplet weight (gravity force) in Newtons."""
        return self.mass * GRAVITY

    @property
    def z(self) -> float:
        """Z position (height) in meters."""
        return self.position[2]

    def copy(self) -> "DropletState":
        """Create a copy of this state."""
        return DropletState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            diameter=self.diameter,
            density=self.density,
            target=self.target.copy() if self.target is not None else None,
            temperature=self.temperature,
            liquid_fraction=self.liquid_fraction,
        )


# =============================================================================
# Ring Control Types (for multi-zone trapping and ring sequencing)
# =============================================================================

@dataclass
class RingConfig:
    """
    Configuration for a single transducer ring.

    Each ring can have independent amplitude, phase offset, and enabled state.
    This enables multi-zone trapping where N rings = N independent trap zones.

    Attributes:
        ring_id: Zero-indexed ring identifier (0 = top ring)
        z_position: Z-coordinate of this ring in meters
        amplitude: Amplitude multiplier [0.0, 1.0] (default 1.0)
        phase_offset: Additional phase offset in radians [0, 2π] (default 0)
        enabled: Whether this ring is active (default True)
        transducer_indices: Array of indices into ArrayGeometry.positions

    The phase_offset is added to all transducers in this ring, enabling
    interference pattern control between rings.
    """
    ring_id: int
    z_position: float
    amplitude: float = 1.0
    phase_offset: float = 0.0
    enabled: bool = True
    transducer_indices: NDArray[np.int32] = field(default_factory=lambda: np.array([], dtype=np.int32))

    @property
    def z_mm(self) -> float:
        """Z position in millimeters."""
        return self.z_position * 1000

    @property
    def num_transducers(self) -> int:
        """Number of transducers in this ring."""
        return len(self.transducer_indices)


@dataclass
class RingArray:
    """
    Ring-aware wrapper around ArrayGeometry.

    Provides explicit ring membership and per-ring configuration for
    multi-zone trapping and ring sequencing operations.

    Key insight: N rings = N independent trap zones = N simultaneous droplets.
    By controlling which rings are active and their relative phases, we can
    create multiple independent trapping zones at different Z-heights.

    Attributes:
        geometry: The underlying transducer array geometry
        rings: List of RingConfig objects (sorted top to bottom by Z)

    Usage:
        array = ArrayGeometry.cylinder(rings=10, per_ring=12)
        ring_array = RingArray(array)  # Auto-infers rings from Z-coords

        # Disable upper half
        for ring in ring_array.rings[:5]:
            ring.enabled = False

        # Get effective amplitudes for pressure computation
        amps = ring_array.get_effective_amplitudes()
    """
    geometry: "ArrayGeometry"
    rings: List[RingConfig] = field(default_factory=list)

    def __post_init__(self):
        """Build ring membership from geometry if not provided."""
        if not self.rings:
            self.rings = self._infer_rings()

    def _infer_rings(self) -> List[RingConfig]:
        """
        Infer ring membership from Z-coordinates.

        Groups transducers by unique Z values (within 1µm tolerance).
        Returns rings sorted from top (highest Z) to bottom (lowest Z).
        """
        z_coords = self.geometry.positions[:, 2]
        # Round to 6 decimal places (1µm precision) for grouping
        unique_z = np.unique(z_coords.round(decimals=6))

        rings = []
        for ring_id, z in enumerate(sorted(unique_z, reverse=True)):
            # Find transducers at this Z level (with tolerance)
            mask = np.abs(z_coords - z) < 1e-5
            indices = np.where(mask)[0].astype(np.int32)

            rings.append(RingConfig(
                ring_id=ring_id,
                z_position=float(z),
                transducer_indices=indices
            ))

        return rings

    @property
    def num_rings(self) -> int:
        """Number of rings in the array."""
        return len(self.rings)

    @property
    def num_transducers(self) -> int:
        """Total number of transducers."""
        return self.geometry.num_transducers

    def get_ring_at_z(self, z: float, tolerance: float = 0.005) -> Optional[RingConfig]:
        """
        Find the ring closest to a given Z coordinate.

        Args:
            z: Z-coordinate in meters
            tolerance: Maximum distance in meters (default 5mm)

        Returns:
            RingConfig if found within tolerance, None otherwise
        """
        for ring in self.rings:
            if abs(ring.z_position - z) < tolerance:
                return ring
        return None

    def get_active_transducer_mask(self) -> NDArray[np.bool_]:
        """
        Return boolean mask for enabled transducers.

        Returns:
            (N,) boolean array where True = transducer is active
        """
        mask = np.zeros(self.geometry.num_transducers, dtype=bool)
        for ring in self.rings:
            if ring.enabled:
                mask[ring.transducer_indices] = True
        return mask

    def get_effective_amplitudes(self) -> NDArray[np.float64]:
        """
        Return amplitude array with per-ring scaling applied.

        Disabled rings get amplitude 0. Enabled rings get their amplitude value.

        Returns:
            (N,) array of amplitude multipliers
        """
        amplitudes = np.zeros(self.geometry.num_transducers)
        for ring in self.rings:
            if ring.enabled:
                amplitudes[ring.transducer_indices] = ring.amplitude
        return amplitudes

    def get_phase_offsets(self) -> NDArray[np.float64]:
        """
        Return phase offset array for all transducers.

        Each transducer gets its ring's phase_offset added to its computed phase.

        Returns:
            (N,) array of phase offsets in radians
        """
        offsets = np.zeros(self.geometry.num_transducers)
        for ring in self.rings:
            offsets[ring.transducer_indices] = ring.phase_offset
        return offsets

    def enable_all(self, amplitude: float = 1.0):
        """Enable all rings with given amplitude."""
        for ring in self.rings:
            ring.enabled = True
            ring.amplitude = amplitude

    def disable_all(self):
        """Disable all rings."""
        for ring in self.rings:
            ring.enabled = False

    def enable_rings_in_range(self, z_min: float, z_max: float, amplitude: float = 1.0):
        """
        Enable only rings within Z range.

        Args:
            z_min: Minimum Z in meters
            z_max: Maximum Z in meters
            amplitude: Amplitude for enabled rings (default 1.0)
        """
        for ring in self.rings:
            if z_min <= ring.z_position <= z_max:
                ring.enabled = True
                ring.amplitude = amplitude
            else:
                ring.enabled = False

    def reset(self):
        """Reset all rings to default state (enabled, amplitude=1, phase_offset=0)."""
        for ring in self.rings:
            ring.enabled = True
            ring.amplitude = 1.0
            ring.phase_offset = 0.0


# =============================================================================
# Acoustic contrast factor (for Gor'kov force)
# =============================================================================

def acoustic_contrast_factor(
    rho_particle: float,
    rho_medium: float,
    c_particle: float = 6420.0,  # [ASMP-003] default is SOLID Al. Use 4700 for liquid, ~110 for styrofoam.
    c_medium: float = SPEED_OF_SOUND,
) -> float:
    """
    Compute the acoustic contrast factor Φ for Gor'kov force.

    From Derivation 1, the Gor'kov potential is:
        U = V × [f₁ × ⟨p²⟩ - f₂ × ⟨v²⟩]

    Where f₁ and f₂ depend on density and compressibility ratios.

    For a rigid sphere (good approximation for liquid metal in air):
        Φ ≈ 5ρ_p / (2ρ_p + ρ_m) - κ_p / κ_m

    For Al in air, Φ ≈ 2.5 (from Derivation 1).

    Args:
        rho_particle: Particle density (kg/m³)
        rho_medium: Medium density (kg/m³)
        c_particle: Sound speed in particle (m/s)
        c_medium: Sound speed in medium (m/s)

    Returns:
        Acoustic contrast factor Φ (dimensionless)
    """
    # Density ratio
    rho_ratio = rho_particle / rho_medium

    # Compressibility ratio (κ = 1/(ρc²))
    kappa_p = 1 / (rho_particle * c_particle**2)
    kappa_m = 1 / (rho_medium * c_medium**2)
    kappa_ratio = kappa_p / kappa_m

    # Monopole coefficient (compressibility contrast)
    f1 = 1 - kappa_ratio

    # Dipole coefficient (density contrast)
    f2 = (2 * (rho_ratio - 1)) / (2 * rho_ratio + 1)

    # Combined contrast factor
    # Note: Different papers define Φ differently; this follows Derivation 1
    phi = f1 + (3/2) * f2

    return phi


# Default contrast factor for Al droplet in air
AL_CONTRAST_FACTOR = acoustic_contrast_factor(AL_DENSITY, AIR_DENSITY)
