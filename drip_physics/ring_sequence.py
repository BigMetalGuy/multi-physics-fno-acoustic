"""
Ring Sequencing and Multi-Zone Control for DRIP.

This module provides the infrastructure for:
1. Multi-zone trapping: N rings = N independent trap zones = N simultaneous droplets
2. Dynamic handoff: Smooth transition as droplets cross zone boundaries
3. Ring sequencing: Time-varying ring configurations for complex manipulation

Key classes:
    Zone: A group of rings that control a region of Z-space
    MultiZoneController: Manages multiple zones for multi-droplet scenarios
    HandoffController: Handles smooth transitions between zones
    RingSequence: Time-based sequence of ring configurations

Architecture:
    RingArray (from core.py)
        - geometry: ArrayGeometry
        - rings: List[RingConfig]
                |
    MultiZoneController
        - zones: List[Zone]
        - droplet_zone_map: {droplet_id: zone_index}
                |
    HandoffController
        - smooth amplitude transitions at boundaries
        - anticipation-based handoff timing

Note: This module contains no hardcoded physical constants. All physics-
dependent values (forces, material properties) live in config.py and are
passed through by the calling code (pathtrack.py).

TODO [ring-handoff-latency]: Zone handoff currently assumes instantaneous
    ring switching. Real hardware has 10-100 us settling time (piezo + amplifier
    slew). Molly's PCB specs will determine actual latency. Until measured,
    transitions are modeled as instant amplitude blending. This may cause
    transient acoustic artifacts during real hardware handoff that are not
    captured in simulation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .core import RingArray, RingConfig


# =============================================================================
# Zone Management
# =============================================================================

@dataclass
class Zone:
    """
    A control zone consisting of one or more rings.

    Zones enable N-droplet control where each zone manages one droplet.
    The rings in a zone work together to create a trapping/guiding field
    for droplets within the zone's Z-range.

    Attributes:
        ring_ids: List of ring indices (into RingArray.rings)
        ring_array: Reference to the parent RingArray
        amplitude: Zone amplitude multiplier (applied to all rings)
        phase_offset: Zone phase offset (applied to all rings)

    Properties:
        z_min, z_max, z_center: Z-bounds of the zone
        active: Whether the zone is currently active
    """
    ring_ids: List[int]
    ring_array: "RingArray"
    amplitude: float = 1.0
    phase_offset: float = 0.0
    _active: bool = True

    @property
    def z_min(self) -> float:
        """Minimum Z of zone (bottom)."""
        return min(self.ring_array.rings[i].z_position for i in self.ring_ids)

    @property
    def z_max(self) -> float:
        """Maximum Z of zone (top)."""
        return max(self.ring_array.rings[i].z_position for i in self.ring_ids)

    @property
    def z_center(self) -> float:
        """Center Z of zone."""
        return (self.z_min + self.z_max) / 2

    @property
    def z_span(self) -> float:
        """Height of zone in meters."""
        return self.z_max - self.z_min

    @property
    def active(self) -> bool:
        """Whether zone is active."""
        return self._active

    @property
    def num_rings(self) -> int:
        """Number of rings in this zone."""
        return len(self.ring_ids)

    def activate(self, amplitude: float = 1.0):
        """
        Activate this zone with given amplitude.

        Sets all rings in the zone to enabled with the specified amplitude.
        """
        self._active = True
        self.amplitude = amplitude
        for ring_id in self.ring_ids:
            ring = self.ring_array.rings[ring_id]
            ring.enabled = True
            ring.amplitude = amplitude
            ring.phase_offset = self.phase_offset

    def deactivate(self):
        """
        Deactivate this zone.

        Sets all rings in the zone to disabled.
        """
        self._active = False
        for ring_id in self.ring_ids:
            self.ring_array.rings[ring_id].enabled = False

    def set_amplitude(self, amplitude: float):
        """Set amplitude for all rings in zone. Clamped to [0.0, 1.0]."""
        amplitude = float(np.clip(amplitude, 0.0, 1.0))
        self.amplitude = amplitude
        if self._active:
            for ring_id in self.ring_ids:
                self.ring_array.rings[ring_id].amplitude = amplitude

    def set_phase_offset(self, phase_offset: float):
        """Set phase offset for all rings in zone."""
        self.phase_offset = phase_offset
        for ring_id in self.ring_ids:
            self.ring_array.rings[ring_id].phase_offset = phase_offset

    def contains_z(self, z: float, margin: float = 0.005) -> bool:
        """
        Check if Z position is within this zone.

        Args:
            z: Z-coordinate in meters
            margin: Extra margin beyond zone bounds (default 5mm)

        Returns:
            True if z is within zone (plus margin)
        """
        return (self.z_min - margin) <= z <= (self.z_max + margin)


# =============================================================================
# Multi-Zone Controller
# =============================================================================

class MultiZoneController:
    """
    Controller for multi-droplet scenarios with zone-based ring control.

    Key insight: N rings = N independent trap zones = N simultaneous droplets.

    Each zone is a group of rings that work together to control droplets
    in that zone's Z-region. Droplets are assigned to zones based on their
    position, and control is localized to the assigned zone.

    Usage:
        ring_array = RingArray(ArrayGeometry.cylinder(rings=10, per_ring=12))
        controller = MultiZoneController(ring_array)

        # Create 2 equal zones (rings 0-4 and rings 5-9)
        zones = controller.create_equal_zones(2)

        # Assign droplets to zones
        controller.assign_droplet_to_zone(droplet_id=0, zone_index=0)
        controller.assign_droplet_to_zone(droplet_id=1, zone_index=1)

        # During simulation
        zone = controller.get_zone_for_droplet(droplet_id=0)
        if zone:
            # Compute phases only for this zone's rings
            ...
    """

    def __init__(self, ring_array: "RingArray"):
        self.ring_array = ring_array
        self.zones: List[Zone] = []
        self.droplet_zone_map: Dict[int, int] = {}  # droplet_id -> zone_index

    @property
    def num_zones(self) -> int:
        """Number of zones."""
        return len(self.zones)

    def create_equal_zones(self, num_zones: int) -> List[Zone]:
        """
        Divide rings equally into zones.

        For 10 rings and 2 zones: Zone 0 = [0,1,2,3,4], Zone 1 = [5,6,7,8,9]
        Zones are ordered from top (high Z) to bottom (low Z).

        Args:
            num_zones: Number of zones to create

        Returns:
            List of created Zone objects
        """
        if num_zones <= 0:
            raise ValueError("num_zones must be positive")

        num_rings = self.ring_array.num_rings
        if num_zones > num_rings:
            raise ValueError(f"Cannot create {num_zones} zones from {num_rings} rings")

        rings_per_zone = num_rings // num_zones
        remainder = num_rings % num_zones

        self.zones = []
        start = 0

        for i in range(num_zones):
            # Distribute remainder rings across first zones
            count = rings_per_zone + (1 if i < remainder else 0)
            end = start + count

            zone = Zone(
                ring_ids=list(range(start, end)),
                ring_array=self.ring_array,
            )
            self.zones.append(zone)
            start = end

        return self.zones

    def create_custom_zones(self, zone_specs: List[List[int]]) -> List[Zone]:
        """
        Create zones with custom ring assignments.

        Args:
            zone_specs: List of ring ID lists, e.g. [[0,1,2], [3,4], [5,6,7,8,9]]

        Returns:
            List of created Zone objects
        """
        self.zones = []
        for ring_ids in zone_specs:
            zone = Zone(ring_ids=ring_ids, ring_array=self.ring_array)
            self.zones.append(zone)
        return self.zones

    def assign_droplet_to_zone(self, droplet_id: int, zone_index: int):
        """
        Assign a droplet to be controlled by a specific zone.

        The zone is activated when a droplet is assigned to it.

        Args:
            droplet_id: Unique identifier for the droplet
            zone_index: Index into self.zones
        """
        if zone_index < 0 or zone_index >= len(self.zones):
            raise ValueError(f"Invalid zone_index {zone_index}")

        self.droplet_zone_map[droplet_id] = zone_index
        self.zones[zone_index].activate()

    def auto_assign_droplet(self, droplet_id: int, droplet_z: float) -> Optional[int]:
        """
        Automatically assign droplet to zone containing its Z position.

        Args:
            droplet_id: Unique identifier for the droplet
            droplet_z: Current Z position of droplet

        Returns:
            Zone index if assigned, None if no zone contains the position
        """
        for i, zone in enumerate(self.zones):
            if zone.contains_z(droplet_z):
                self.assign_droplet_to_zone(droplet_id, i)
                return i
        return None

    def update_zone_for_droplet(self, droplet_id: int, droplet_z: float) -> Optional[int]:
        """
        Update zone assignment based on droplet position (hard handoff).

        If droplet crosses into a new zone, immediately switch control.
        For smooth transitions, use HandoffController instead.

        Args:
            droplet_id: Unique identifier for the droplet
            droplet_z: Current Z position of droplet

        Returns:
            New zone index if handoff occurred, None otherwise
        """
        current_zone_idx = self.droplet_zone_map.get(droplet_id)

        for i, zone in enumerate(self.zones):
            if zone.contains_z(droplet_z):
                if i != current_zone_idx:
                    # Handoff to new zone — only deactivate old zone if no
                    # other droplet is still using it.
                    if current_zone_idx is not None:
                        other_users = [
                            d for d, z in self.droplet_zone_map.items()
                            if z == current_zone_idx and d != droplet_id
                        ]
                        if not other_users:
                            self.zones[current_zone_idx].deactivate()
                    zone.activate()
                    self.droplet_zone_map[droplet_id] = i
                    return i
                return None

        return None

    def get_zone_for_droplet(self, droplet_id: int) -> Optional[Zone]:
        """Get the zone controlling a droplet."""
        zone_idx = self.droplet_zone_map.get(droplet_id)
        if zone_idx is not None and zone_idx < len(self.zones):
            return self.zones[zone_idx]
        return None

    def get_zone_index_for_droplet(self, droplet_id: int) -> Optional[int]:
        """Get the zone index for a droplet."""
        return self.droplet_zone_map.get(droplet_id)

    def remove_droplet(self, droplet_id: int):
        """Remove droplet from zone management."""
        zone_idx = self.droplet_zone_map.pop(droplet_id, None)
        # Optionally deactivate zone if no other droplets are using it
        if zone_idx is not None:
            zone = self.zones[zone_idx]
            # Check if any other droplets use this zone
            other_droplets = [d for d, z in self.droplet_zone_map.items() if z == zone_idx]
            if not other_droplets:
                zone.deactivate()


# =============================================================================
# Handoff Controller (Smooth Zone Transitions)
# =============================================================================

@dataclass
class HandoffState:
    """State for an in-progress zone handoff."""
    from_zone_idx: int
    to_zone_idx: int
    progress: float = 0.0  # 0 = fully in from_zone, 1 = fully in to_zone
    start_z: float = 0.0
    end_z: float = 0.0


class HandoffController:
    """
    Manages smooth handoff between zones as droplets descend.

    The handoff process:
    1. Droplet in Zone A, approaching Zone B boundary
    2. Begin ramping Zone B amplitude up (anticipation)
    3. Simultaneously ramp Zone A amplitude down
    4. Complete handoff when droplet fully in Zone B

    This ensures continuous acoustic control during transitions,
    preventing droplet loss at zone boundaries.

    Usage:
        controller = MultiZoneController(ring_array)
        controller.create_equal_zones(2)
        controller.assign_droplet_to_zone(0, 0)

        handoff = HandoffController(controller)

        # During simulation loop:
        for t in timesteps:
            handoff.update(droplet_id=0, droplet_z=position[2], dt=dt)
            # Zones will smoothly transition
    """

    def __init__(
        self,
        multi_zone: MultiZoneController,
        anticipation_distance: float = 0.010,  # Start handoff 10mm before boundary
        min_overlap: float = 0.2,  # Minimum amplitude overlap (prevents dead zones)
    ):
        """
        Initialize handoff controller.

        Args:
            multi_zone: The MultiZoneController managing zones
            anticipation_distance: Distance before boundary to start handoff (meters)
            min_overlap: Minimum amplitude sum during transition (0-1)
        """
        self.multi_zone = multi_zone
        self.anticipation_distance = anticipation_distance
        self.min_overlap = min_overlap
        self._active_handoffs: Dict[int, HandoffState] = {}

    def update(self, droplet_id: int, droplet_z: float, dt: float) -> bool:
        """
        Update handoff state for a droplet.

        Call this each simulation step. Returns True if a handoff completed.

        Args:
            droplet_id: Droplet identifier
            droplet_z: Current Z position
            dt: Timestep (currently unused, but available for rate limiting)

        Returns:
            True if handoff completed this step, False otherwise
        """
        current_zone_idx = self.multi_zone.droplet_zone_map.get(droplet_id)
        if current_zone_idx is None:
            return False

        current_zone = self.multi_zone.zones[current_zone_idx]

        # Find next zone (the one below current, since droplets fall)
        next_zone_idx = self._find_next_zone_below(droplet_z, current_zone_idx)
        if next_zone_idx is None:
            # No zone below, check if we're exiting
            return False

        next_zone = self.multi_zone.zones[next_zone_idx]

        # Calculate transition region
        # Boundary is between current zone's bottom and next zone's top
        boundary_z = (current_zone.z_min + next_zone.z_max) / 2
        start_z = boundary_z + self.anticipation_distance
        end_z = boundary_z - self.anticipation_distance

        # Check if in transition region
        if start_z >= droplet_z >= end_z:
            # Calculate progress through transition (0 = just entered, 1 = completed)
            progress = (start_z - droplet_z) / (start_z - end_z)
            progress = np.clip(progress, 0, 1)

            # Apply smooth blending (cosine interpolation for smoother transition)
            blend = (1 - np.cos(progress * np.pi)) / 2

            # Calculate amplitudes with minimum overlap guarantee
            # At progress=0.5, both zones should have at least min_overlap/2
            current_amp = 1.0 - blend * (1.0 - self.min_overlap)
            next_amp = blend + self.min_overlap * (1 - blend)

            # Normalize total amplitude to never exceed 1.0.
            # Without this, outgoing + incoming can sum > 1.0 during
            # mid-transition, overdriving the transducers.
            total_amp = current_amp + next_amp
            if total_amp > 1.0:
                current_amp /= total_amp
                next_amp /= total_amp

            current_zone.set_amplitude(current_amp)
            next_zone.set_amplitude(next_amp)

            # Activate next zone if not already
            if not next_zone.active:
                next_zone.activate(amplitude=next_amp)

            # Store handoff state
            self._active_handoffs[droplet_id] = HandoffState(
                from_zone_idx=current_zone_idx,
                to_zone_idx=next_zone_idx,
                progress=progress,
                start_z=start_z,
                end_z=end_z,
            )

            # Complete handoff when progress reaches 1.0
            if progress >= 1.0:
                current_zone.deactivate()
                next_zone.set_amplitude(1.0)
                self.multi_zone.droplet_zone_map[droplet_id] = next_zone_idx
                del self._active_handoffs[droplet_id]
                return True

        return False

    def _find_next_zone_below(self, z: float, current_idx: int) -> Optional[int]:
        """Find the next zone below current position."""
        current_zone = self.multi_zone.zones[current_idx]

        best_idx = None
        best_z = float('-inf')

        for i, zone in enumerate(self.multi_zone.zones):
            if i != current_idx and zone.z_max < current_zone.z_min:
                # This zone is below current zone
                if zone.z_max > best_z:
                    best_z = zone.z_max
                    best_idx = i

        return best_idx

    def get_handoff_state(self, droplet_id: int) -> Optional[HandoffState]:
        """Get current handoff state for a droplet, if any."""
        return self._active_handoffs.get(droplet_id)

    def is_in_handoff(self, droplet_id: int) -> bool:
        """Check if droplet is currently in a handoff transition."""
        return droplet_id in self._active_handoffs


# =============================================================================
# Ring Sequence (Time-Based Configuration)
# =============================================================================

@dataclass
class RingSequenceStep:
    """
    A single step in a ring sequence.

    Defines ring states at a specific time point.

    Attributes:
        time: Timestamp for this step (seconds from start)
        ring_states: Dict mapping ring_id to config overrides
        transition_duration: Time to interpolate to this state (seconds)
    """
    time: float
    ring_states: Dict[int, Dict[str, Any]]  # {ring_id: {'amplitude': 0.5, 'enabled': True, ...}}
    transition_duration: float = 0.0  # 0 = instant switch


@dataclass
class RingSequence:
    """
    Time-based sequence of ring configurations.

    Use cases:
    1. Acoustic conveyor: Sequential ring activation to move droplet
    2. Pulsed trapping: Periodic amplitude modulation
    3. Complex patterns: Choreographed multi-ring sequences

    Usage:
        seq = RingSequence()
        seq.add_step(0.0, {0: {'amplitude': 1.0}, 1: {'amplitude': 0.0}})
        seq.add_step(0.1, {0: {'amplitude': 0.0}, 1: {'amplitude': 1.0}}, transition=0.02)

        # Get interpolated state at any time
        state = seq.get_state_at_time(0.05)
    """
    steps: List[RingSequenceStep] = field(default_factory=list)

    def add_step(
        self,
        time: float,
        ring_states: Dict[int, Dict[str, Any]],
        transition_duration: float = 0.0
    ):
        """
        Add a sequence step.

        Args:
            time: Time in seconds
            ring_states: Ring configurations {ring_id: {amplitude, phase_offset, enabled}}
            transition_duration: Interpolation time to reach this state
        """
        self.steps.append(RingSequenceStep(time, ring_states, transition_duration))
        self.steps.sort(key=lambda s: s.time)

    def get_state_at_time(self, t: float) -> Dict[int, Dict[str, Any]]:
        """
        Get interpolated ring states at time t.

        Returns:
            Dict of {ring_id: {amplitude, phase_offset, enabled}}
        """
        if not self.steps:
            return {}

        # Find surrounding steps
        prev_step = None
        next_step = None
        for step in self.steps:
            if step.time <= t:
                prev_step = step
            else:
                next_step = step
                break

        if prev_step is None:
            return self.steps[0].ring_states.copy()

        if next_step is None or next_step.transition_duration == 0:
            return prev_step.ring_states.copy()

        # Check if we're in transition period
        time_since_prev = t - prev_step.time
        if time_since_prev >= next_step.transition_duration:
            return next_step.ring_states.copy()

        # Interpolate between steps
        blend = time_since_prev / next_step.transition_duration
        return self._interpolate_states(prev_step.ring_states, next_step.ring_states, blend)

    def _interpolate_states(
        self,
        state1: Dict[int, Dict[str, Any]],
        state2: Dict[int, Dict[str, Any]],
        blend: float
    ) -> Dict[int, Dict[str, Any]]:
        """Linear interpolation of ring states."""
        result = {}
        all_rings = set(state1.keys()) | set(state2.keys())

        for ring_id in all_rings:
            s1 = state1.get(ring_id, {'amplitude': 0.0, 'phase_offset': 0.0, 'enabled': False})
            s2 = state2.get(ring_id, {'amplitude': 0.0, 'phase_offset': 0.0, 'enabled': False})

            result[ring_id] = {
                'amplitude': s1.get('amplitude', 0) * (1 - blend) + s2.get('amplitude', 0) * blend,
                'phase_offset': s1.get('phase_offset', 0) * (1 - blend) + s2.get('phase_offset', 0) * blend,
                'enabled': s2.get('enabled', True) if blend > 0.5 else s1.get('enabled', True),
            }

        return result

    @property
    def duration(self) -> float:
        """Total duration of sequence."""
        if not self.steps:
            return 0.0
        return self.steps[-1].time


class RingSequencer:
    """
    Applies a RingSequence to a RingArray over time.

    This is the runtime controller that updates ring states
    according to a predefined sequence.

    Usage:
        sequence = RingSequence()
        sequence.add_step(0.0, {0: {'amplitude': 1.0}, 1: {'amplitude': 0.5}})
        sequence.add_step(0.1, {0: {'amplitude': 0.5}, 1: {'amplitude': 1.0}})

        sequencer = RingSequencer(ring_array, sequence)

        # In simulation loop:
        sequencer.update(t)  # Updates ring_array.rings based on time
    """

    def __init__(self, ring_array: "RingArray", sequence: Optional[RingSequence] = None):
        self.ring_array = ring_array
        self.sequence = sequence or RingSequence()
        self._current_time = 0.0

    def update(self, t: float):
        """
        Update ring states to time t.

        Applies the sequence state at time t to the ring array.
        """
        self._current_time = t
        states = self.sequence.get_state_at_time(t)

        for ring_id, state in states.items():
            if ring_id < len(self.ring_array.rings):
                ring = self.ring_array.rings[ring_id]
                if 'amplitude' in state:
                    ring.amplitude = state['amplitude']
                if 'phase_offset' in state:
                    ring.phase_offset = state['phase_offset']
                if 'enabled' in state:
                    ring.enabled = state['enabled']

    def reset(self):
        """Reset sequencer to time 0."""
        self._current_time = 0.0
        self.update(0.0)

    @property
    def current_time(self) -> float:
        """Current sequence time."""
        return self._current_time
