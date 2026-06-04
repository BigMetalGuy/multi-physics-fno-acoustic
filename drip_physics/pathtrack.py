#!/usr/bin/env python3
"""
Path tracking simulation for droplet guidance.

This module contains the simulation functions that drive droplets through
the acoustic control chamber:
- simulate_path_tracking: single-droplet simulation with controller
- simulate_multi_droplet: multi-droplet TDM simulation
- compare_with_baseline: comparison helper
- main: demo entry point

Controller types and implementations live in drip_physics.controllers.
All public names from controllers are re-exported here for backward
compatibility.

Run with:
    python -m drip_physics.pathtrack
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

from .config import (
    EnvironmentConfig,
    DropletConfig,
    ThermalConfig,
    ALUMINUM_SOLID,
    GRAVITY,
)
from .thermal import compute_heat_loss, ThermalResult
from .core import ArrayGeometry, DropletState, PhaseArray, RingArray
from .inverse import solve_inverse, solve_inverse_with_rings
from .trajectory import compute_forces_at_state
from .force_plugin import ForceCompositor, ForceRegistry
from .ring_sequence import MultiZoneController, HandoffController
from .trace import sim_trace

# Re-export everything from controllers for backward compatibility.
# Any code doing `from drip_physics.pathtrack import ControllerState` etc.
# will continue to work.
from .controllers import (  # noqa: F401
    ControllerState,
    ControllerFn,
    IdealPath,
    compute_remaining_time,
    compute_legacy_control,
    compute_p_control,
    compute_pi_control,
    compute_pid_control,
    CONTROLLER_MODES,
    REMOVED_MODES,
    get_controller,
    _CONTROLLER_CLAMP_FRACTION,
    _DERIV_FILTER_ALPHA,
    _VEL_NORM_RATIO,
)

if TYPE_CHECKING:
    from .pressure_backend import PressureBackend


# =============================================================================
# Trajectory planning helper
# =============================================================================

def compute_optimal_trajectory(
    current_pos: np.ndarray,
    current_vel: np.ndarray,
    target: Tuple[float, float],
    z_floor: float,
) -> dict:
    """
    Compute optimal remaining trajectory from current state to target.

    This is the KEY function for receding horizon control:
    Given where we ARE and where we're GOING, compute what the
    optimal path looks like FROM HERE.

    For camera validation, this same function works:
    - current_pos comes from camera instead of simulation
    - Everything else is the same

    Args:
        current_pos: [x, y, z] current position
        current_vel: [vx, vy, vz] current velocity
        target: (target_x, target_y) landing target
        z_floor: landing height

    Returns:
        dict with trajectory plan
    """
    # Time remaining based on current fall velocity
    t_remaining = compute_remaining_time(current_pos[2], current_vel[2], z_floor)

    # Current lateral state
    pos_xy = current_pos[:2]
    vel_xy = current_vel[:2]
    target_xy = np.array([target[0], target[1]])

    # Where will we land with current velocity (no more force)?
    predicted_landing = pos_xy + vel_xy * t_remaining

    # Error if we coast from here
    landing_error = target_xy - predicted_landing

    # Required displacement from current position
    required_displacement = target_xy - pos_xy

    # What acceleration do we need to hit target?
    # x = x0 + v0*t + 0.5*a*t^2
    # a = 2*(target - pos - vel*t) / t^2
    if t_remaining > 0.01:
        optimal_accel = 2 * (required_displacement - vel_xy * t_remaining) / (t_remaining**2)
    else:
        optimal_accel = np.zeros(2)

    return {
        't_remaining': t_remaining,
        'required_displacement': required_displacement,
        'current_lateral_vel': vel_xy,
        'predicted_landing': predicted_landing,
        'landing_error': landing_error,
        'landing_error_mag': np.linalg.norm(landing_error),
        'optimal_accel': optimal_accel,
    }


# =============================================================================
# Trajectory data classes
# =============================================================================

@dataclass
class TrajectoryPoint:
    """Single point in trajectory with tracking info."""
    t: float
    position: np.ndarray
    velocity: np.ndarray
    ideal_position: np.ndarray       # For visualization (initial ideal path)
    pos_error: np.ndarray            # Error from initial ideal path
    predicted_landing: np.ndarray    # Where we'll land if we coast from here
    landing_error: np.ndarray        # Predicted error at landing
    t_remaining: float               # Time until landing
    force_acoustic: np.ndarray
    force_total: np.ndarray
    control_offset: Tuple[float, float]
    phases: Optional[PhaseArray] = None
    temperature: float = 1123.15      # K
    liquid_fraction: float = 1.0      # 0-1
    heat_flux: float = 0.0            # W (total)
    heat_flux_conv: float = 0.0       # W (convective component)
    heat_flux_rad: float = 0.0        # W (radiative component)
    nu_eff: float = 0.0               # Effective Nusselt number (dimensionless)
    phase: str = 'liquid'             # 'liquid', 'mushy', 'solid'
    material_name: str = ''           # Active material config name (e.g. 'aluminum_liquid', 'aluminum_solid')


@dataclass
class TrackedTrajectory:
    """Complete trajectory with path tracking data."""
    points: List[TrajectoryPoint]
    ideal_path: IdealPath
    target: Tuple[float, float]
    droplet: DropletState
    landing_position: Optional[np.ndarray] = None
    landing_error: Optional[float] = None

    # Tracking metrics
    max_deviation: float = 0.0      # Max deviation from ideal path
    mean_deviation: float = 0.0     # Mean deviation
    final_error: float = 0.0        # Error at landing

    def __len__(self):
        return len(self.points)

    @property
    def times(self) -> np.ndarray:
        return np.array([p.t for p in self.points])

    @property
    def positions(self) -> np.ndarray:
        return np.array([p.position for p in self.points])

    @property
    def ideal_positions(self) -> np.ndarray:
        return np.array([p.ideal_position for p in self.points])

    @property
    def errors(self) -> np.ndarray:
        return np.array([p.pos_error for p in self.points])

    @property
    def duration(self) -> float:
        return self.times[-1] - self.times[0]


# =============================================================================
# Main simulation function
# =============================================================================

def simulate_path_tracking(
    array: ArrayGeometry,
    target: Tuple[float, float],
    initial_position: np.ndarray,
    env_config: Optional[EnvironmentConfig] = None,
    droplet_config: Optional[DropletConfig] = None,
    initial_velocity: Optional[np.ndarray] = None,
    dt: float = 1e-4,
    max_duration: float = 0.5,
    z_floor: float = 0.0,
    control_mode: Any = 'legacy',  # str name or callable ControllerFn
    store_phases: bool = False,
    hover_fraction: float = 0.7,
    enabled_forces: Optional[list] = None,
    force_params: Optional[dict] = None,
    ring_config: Optional[dict] = None,
    thermal_config: Optional[ThermalConfig] = None,
    backend: Optional["PressureBackend"] = None,
    # Legacy kwargs — will be removed once all callers migrate to config objects
    droplet_diameter: float = 300e-6,
    droplet_density: float = 2385.0,
    kp: float = 2.0,
    kd: float = 0.15,
    ki: float = 0.0,
) -> TrackedTrajectory:
    """
    Simulate trajectory with selectable controller.

    Args:
        array: Transducer array geometry
        target: Target landing position (x, y) in meters
        initial_position: Starting position [x, y, z] in meters
        env_config: Environment configuration (REQUIRED)
        droplet_config: Droplet configuration (REQUIRED)
        initial_velocity: Starting velocity (default zeros)
        dt: Integration timestep (seconds)
        max_duration: Maximum simulation time (seconds)
        z_floor: Landing height (meters)
        control_mode: Controller mode name (see CONTROLLER_MODES)
        store_phases: Whether to store phase arrays per point
        hover_fraction: Fraction of drop to complete lateral motion
        enabled_forces: List of force names to enable (None = defaults)
        force_params: Per-force parameter overrides
        ring_config: Ring sequencing configuration dict

    Returns:
        TrackedTrajectory with full simulation and tracking data

    Control modes:
        - 'legacy':  PD on position error from ideal path (baseline for Ryota)
        - 'p_only':  Proportional-only (SW-16 C2)
        - 'pi':      Proportional + Integral with anti-windup (SW-16 C2)
        - 'pid':     Full PID with derivative filter (SW-16 C2)

        Removed in Layer 7 (archived in drip_physics/archive/deprecated_controllers.py):
        - 'mpc', 'velocity', 'bangbang', 'feedforward'

    TODO [ring-handoff-latency]: Ring switching latency (10-100 us) is
        currently unmodeled. See module docstring.
    """
    # --- Config resolution: REQUIRED, but legacy callers get a warning --------
    from .config import default_environment, aluminum_droplet as _aluminum_droplet

    if env_config is None:
        warnings.warn(
            "simulate_path_tracking: env_config not provided. "
            "Constructing default EnvironmentConfig. "
            "Pass env_config explicitly — this fallback will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        env_config = default_environment()

    if droplet_config is None:
        warnings.warn(
            "simulate_path_tracking: droplet_config not provided. "
            f"Constructing DropletConfig from legacy kwargs "
            f"(diameter={droplet_diameter}, density={droplet_density}). "
            "Pass droplet_config explicitly — this fallback will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        droplet_config = _aluminum_droplet(diameter=droplet_diameter)

    # Clear trace for this run
    sim_trace.clear()

    # Validate controller mode — accept callable or string name
    if callable(control_mode):
        controller_fn = control_mode
    else:
        controller_fn = get_controller(control_mode)

    # Trace simulation start
    _mode_name = control_mode if isinstance(control_mode, str) else getattr(control_mode, '__name__', 'callable')
    sim_trace.log(
        "sim_start",
        material=droplet_config.material.name,
        target=target,
        controller=_mode_name,
        dt=dt,
        max_duration=max_duration,
    )

    # Build ideal path
    ideal_path = IdealPath(
        z_start=initial_position[2],
        z_floor=z_floor,  # Melt pool surface
        target_x=target[0],
        target_y=target[1],
        hover_fraction=hover_fraction,
    )

    # Initialize state
    position = np.array(initial_position, dtype=float)
    velocity = np.array(initial_velocity if initial_velocity is not None else [0, 0, 0], dtype=float)

    mass = droplet_config.mass
    radius = droplet_config.radius

    # Thermal state
    if thermal_config is not None:
        temperature = thermal_config.initial_droplet_temp
        liquid_fraction = 1.0
        current_material = droplet_config.material  # starts as ALUMINUM_LIQUID
    else:
        temperature = (
            droplet_config.material.solidus_temperature
            if hasattr(droplet_config.material, 'solidus_temperature')
            and droplet_config.material.solidus_temperature > 0
            else 933.15
        )
        liquid_fraction = 1.0
        current_material = droplet_config.material

    droplet = DropletState(
        position=position.copy(),
        velocity=velocity.copy(),
        diameter=droplet_config.diameter,
        density=droplet_config.material.density,
        target=np.array([target[0], target[1], 0]),
        temperature=temperature,
        liquid_fraction=liquid_fraction,
    )

    # Set up force compositor
    compositor = ForceCompositor()
    if enabled_forces is not None:
        for force_name in enabled_forces:
            params = (force_params or {}).get(force_name, {})
            compositor.enable(force_name, params)
    else:
        for force in ForceRegistry.enabled_by_default():
            compositor.enable(force.name)

    # Set up ring array
    ring_array = None
    handoff_ctrl = None
    ring_mode = None

    if ring_config is not None:
        ring_mode = ring_config.get('mode', 'uniform')
        ring_array = RingArray(array)

        if ring_mode == 'uniform':
            pass
        elif ring_mode == 'manual':
            ring_states = ring_config.get('rings', [])
            for i, rs in enumerate(ring_states):
                if i < len(ring_array.rings):
                    ring_array.rings[i].enabled = rs.get('enabled', True)
                    ring_array.rings[i].amplitude = rs.get('amplitude', 1.0)
                    ring_array.rings[i].phase_offset = rs.get('phase_offset', 0.0)
        elif ring_mode == 'multi_zone':
            num_zones = ring_config.get('num_zones', 2)
            multi_ctrl = MultiZoneController(ring_array)
            zones = multi_ctrl.create_equal_zones(num_zones)
            droplet_z = initial_position[2]
            nearest_zone_idx = 0
            min_dist = float('inf')
            for i, zone in enumerate(zones):
                dist = abs(droplet_z - zone.z_center)
                if dist < min_dist:
                    min_dist = dist
                    nearest_zone_idx = i
            multi_ctrl.assign_droplet_to_zone(0, nearest_zone_idx)

            handoff_settings = ring_config.get('handoff', {})
            handoff_ctrl = HandoffController(
                multi_ctrl,
                anticipation_distance=handoff_settings.get('anticipation_distance', 0.010),
                min_overlap=handoff_settings.get('min_overlap', 0.2),
            )
        elif ring_mode == 'sequenced':
            pass  # Updated in simulation loop

    t_total = ideal_path.fall_time

    # Simulation loop
    points = []
    deviations = []
    t = 0.0

    # Carry-forward state for PI/PID controllers
    integral_accumulator = np.zeros(2)  # [ASMP-033] reset at ring handoffs
    prev_velocity = np.zeros(2)         # Previous lateral velocity for D filter
    block_states: Optional[Dict[str, Any]] = None  # Compiled block diagram states
    prev_ring_idx = -1                  # Track ring transitions for [ASMP-033]

    # Active droplet config tracks material transitions (ASMP-001).
    # When the droplet solidifies mid-flight, active_droplet_config is replaced
    # with a new DropletConfig using ALUMINUM_SOLID so that f1/f2 Gor'kov
    # coefficients use the correct sound speed (6420 m/s instead of 4700 m/s).
    active_droplet_config = droplet_config

    while t < max_duration and position[2] > z_floor:
        # 1. Compute trajectory plan
        plan = compute_optimal_trajectory(position, velocity, target, z_floor)

        # 2. Get ideal position for visualization
        errors = ideal_path.get_error(position, velocity)
        ideal_pos = np.array(errors['ideal_pos'])
        pos_error = errors['pos_error']
        deviations.append(errors['pos_error_mag'])

        # Detect ring transition for [ASMP-033] integral reset
        ring_changed = False
        if ring_array is not None and len(ring_array.rings) > 0:
            # Find the ring index whose z_position is closest to current droplet z
            ring_zs = np.array([r.z_position for r in ring_array.rings])
            current_ring_idx = int(np.argmin(np.abs(ring_zs - position[2])))
            if prev_ring_idx >= 0 and current_ring_idx != prev_ring_idx:
                # [ASMP-033] Ki accumulator resets at ring handoffs. New ring = new phase geometry.
                # Carrying accumulated error from previous ring drives wrong correction.
                # Anti-windup via output clamping. See ASSUMPTIONS.md.
                integral_accumulator = np.zeros(2)
                # Ring handoff: reset integral accumulators but preserve filter/TF state
                # PID blocks detect ring changes via state.ring_changed flag
                # Don't nuke the entire block_states dict
                ring_changed = True
                sim_trace.log(
                    "ring_handoff", t=t,
                    ring_from=prev_ring_idx, ring_to=current_ring_idx,
                )
            prev_ring_idx = current_ring_idx

        # 3. Compute control offset via unified interface
        ctrl_state = ControllerState(
            position=position.copy(),
            velocity=velocity.copy(),
            target=target,
            z_floor=z_floor,
            ideal_path=ideal_path,
            env_config=env_config,
            droplet_config=active_droplet_config,
            t_elapsed=t,
            t_total=t_total,
            kp=kp,
            kd=kd,
            ki=ki if ki else None,
            integral_accumulator=integral_accumulator,
            prev_velocity=prev_velocity.copy(),
            dt=dt,
            ring_changed=ring_changed,
            block_states=block_states,
        )
        control_offset = controller_fn(ctrl_state)

        # Carry forward integral accumulator, prev_velocity, and block_states for next step
        if ctrl_state.integral_accumulator is not None:
            integral_accumulator = ctrl_state.integral_accumulator.copy()
        if ctrl_state.prev_velocity is not None:
            prev_velocity = ctrl_state.prev_velocity.copy()
        if ctrl_state.block_states is not None:
            block_states = ctrl_state.block_states

        # 4. Compute phases (pass config to avoid legacy fallback)
        if ring_array is not None:
            if handoff_ctrl is not None:
                handoff_ctrl.update(droplet_id=0, droplet_z=position[2], dt=dt)

            phases = solve_inverse_with_rings(
                ring_array, target_offset=control_offset, droplet_z=position[2],
                droplet_xy=(position[0], position[1]),
                env_config=env_config, droplet_config=active_droplet_config,
            )
        else:
            phases = solve_inverse(
                array, target_offset=control_offset, droplet_z=position[2],
                droplet_xy=(position[0], position[1]),
                env_config=env_config, droplet_config=active_droplet_config,
            )

        # 5. Compute forces via ForceCompositor (config-driven)
        # [ASMP-031] FIXED: env forwarded via compute_forces_at_state -> compute_force -> compute_pressure.
        # F_acoustic is obtained via the shared helper; gravity + drag are handled
        # by the ForceCompositor plugins to avoid double-counting.
        field_data = {} if thermal_config is not None else None
        F_acoustic, _F_drag_unused, _F_gravity_unused = compute_forces_at_state(
            array, phases, position, velocity, droplet, mass, radius,
            env_config=env_config,
            droplet_config=active_droplet_config,
            include_drag=False,  # compositor's DragForce plugin handles drag
            ring_array=ring_array,
            field_output=field_data,
            backend=backend,
        )
        result = compositor.compute_total(
            position, velocity,
            env_config, active_droplet_config,
            acoustic_force=F_acoustic,
        )
        F_total = result.force

        # 8. Thermal update (if enabled)
        thermal_result = None
        if thermal_config is not None:
            grad_p_mag = field_data.get('grad_p_magnitude', 0.0) if field_data else 0.0
            thermal_result = compute_heat_loss(
                droplet_material=current_material,
                env=env_config,
                thermal_cfg=thermal_config,
                temperature=temperature,
                liquid_fraction=liquid_fraction,
                velocity=velocity,
                droplet_radius=radius,
                droplet_mass=mass,
                grad_p_magnitude=grad_p_mag,
                dt=dt,
            )

            # Update thermal state
            temperature += thermal_result.dT_dt * dt
            liquid_fraction = max(0.0, min(1.0, liquid_fraction + thermal_result.dlf_dt * dt))

            # Material transition: liquid -> solid (ASMP-001 territory)
            if liquid_fraction <= 0.0 and current_material != ALUMINUM_SOLID:
                if (hasattr(droplet_config.material, 'solidus_temperature')
                        and droplet_config.material.solidus_temperature > 0):
                    current_material = ALUMINUM_SOLID
                    active_droplet_config = droplet_config.with_material(ALUMINUM_SOLID)
                    warnings.warn(
                        f"[ASMP-001] Droplet solidified at t={t:.4f}s — "
                        f"switching to ALUMINUM_SOLID material "
                        f"(c_particle: {droplet_config.material.sound_speed:.0f} -> "
                        f"{ALUMINUM_SOLID.sound_speed:.0f} m/s)",
                        stacklevel=2,
                    )

            # Update droplet state
            droplet.temperature = temperature
            droplet.liquid_fraction = liquid_fraction

        # Store point
        point = TrajectoryPoint(
            t=t,
            position=position.copy(),
            velocity=velocity.copy(),
            ideal_position=ideal_pos,
            pos_error=pos_error.copy(),
            predicted_landing=plan['predicted_landing'].copy(),
            landing_error=plan['landing_error'].copy(),
            t_remaining=plan['t_remaining'],
            force_acoustic=F_acoustic.copy(),
            force_total=F_total.copy(),
            control_offset=control_offset,
            phases=phases if store_phases else None,
            temperature=temperature if thermal_config is not None else droplet.temperature,
            liquid_fraction=liquid_fraction if thermal_config is not None else droplet.liquid_fraction,
            heat_flux=thermal_result.heat_flux_W if thermal_result is not None else 0.0,
            heat_flux_conv=thermal_result.Nu_components.get('convective', 0.0) if thermal_result is not None else 0.0,
            heat_flux_rad=thermal_result.Nu_components.get('radiative', 0.0) if thermal_result is not None else 0.0,
            nu_eff=thermal_result.Nu_eff if thermal_result is not None else 0.0,
            phase=thermal_result.phase if thermal_result is not None else 'unknown',
            material_name=active_droplet_config.material.name,
        )
        points.append(point)

        # 6. Integration: semi-implicit Euler (symplectic). RK45 is used in
        # trajectory.py for higher accuracy, but pathtrack requires fixed-step
        # output for the controller loop. At dt=1e-4, Euler error is O(1e-8)
        # per step, acceptable for styrofoam at Re<10. For aluminum at Re~40,
        # consider sub-stepping with RK45 within each control interval.
        acceleration = F_total / mass
        velocity = velocity + acceleration * dt
        position = position + velocity * dt

        droplet.position = position.copy()
        droplet.velocity = velocity.copy()

        t += dt

    # Compute metrics
    landing_position = position.copy()
    landing_error = np.linalg.norm(landing_position[:2] - np.array([target[0], target[1]]))

    sim_trace.log(
        "sim_end", t=t,
        landing_error=float(landing_error),
        n_steps=len(points),
    )

    trajectory = TrackedTrajectory(
        points=points,
        ideal_path=ideal_path,
        target=target,
        droplet=droplet,
        landing_position=landing_position,
        landing_error=landing_error,
        max_deviation=max(deviations) if deviations else 0,
        mean_deviation=float(np.mean(deviations)) if deviations else 0,
        final_error=landing_error,
    )

    return trajectory


# =============================================================================
# Multi-droplet simulation
# =============================================================================

def simulate_multi_droplet(
    array: ArrayGeometry,
    droplet_configs: List[DropletConfig],
    targets: List[Tuple[float, float]],
    env_config: EnvironmentConfig,
    dt: float = 1e-4,
    max_duration: float = 0.5,
    weights: Optional[np.ndarray] = None,
    z_floor: float = 0.0,
    entry_delays: Optional[List[float]] = None,
    force_smoothing: Optional[float] = 0.3,
) -> List[TrackedTrajectory]:
    """Simulate multiple droplets with staggered entry times.

    At each timestep:
    1. Determine which droplets are active (entered and not yet landed)
    2. Compute joint phases via solve_inverse_multi() for active droplets
    3. Compute force on each active droplet from the shared phase field
    4. Integrate each active droplet's trajectory independently (Euler)
    5. Deactivate droplets that have landed

    No controller is applied — this uses pure feedforward from the
    joint inverse solver. For controlled multi-droplet trajectories,
    a future version should integrate the controller protocol.

    Note: Multi-droplet mode uses pure feedforward (no controller).
        control_offset in TrajectoryPoint is always (0, 0) because
        the steering is done directly by solve_inverse_multi, not
        through a controller -> offset -> inverse solver chain.

    Args:
        array: Transducer array geometry.
        droplet_configs: Per-droplet configuration (material, diameter, initial conditions).
        targets: Per-droplet landing target (x, y) in meters.
        env_config: Environment configuration (REQUIRED).
        dt: Integration timestep in seconds.
        max_duration: Maximum simulation time in seconds.
        weights: Priority weights per droplet. Default: equal weights.
        z_floor: Landing height in meters.
        entry_delays: Per-droplet entry time in seconds. Droplet i is inactive
            (no force computation, no integration) until t >= entry_delays[i].
            Default: all droplets enter at t=0.
        force_smoothing: Exponential moving average alpha for acoustic force.
            Range [0, 1]: 0 = pure previous force (no update), 1 = pure new
            force (no smoothing). Default 0.3 prevents oscillation from
            coherent superposition phase interference when multiple droplets
            are at similar positions. Set to None or 1.0 to disable.

    Returns:
        List of TrackedTrajectory objects, one per droplet.
    """
    from .inverse import solve_inverse_multi, validate_multi_solution
    from .force import compute_force
    from .force_plugin import ForceCompositor, ForceRegistry

    sim_trace.clear()

    n_droplets = len(droplet_configs)
    if n_droplets == 0:
        raise ValueError("droplet_configs must be non-empty")
    if len(targets) != n_droplets:
        raise ValueError(
            f"targets length ({len(targets)}) must match "
            f"droplet_configs length ({n_droplets})"
        )

    # Entry delays: default all at t=0
    _entry_delays = entry_delays if entry_delays is not None else [0.0] * n_droplets
    if any(d < 0 for d in _entry_delays):
        raise ValueError("entry_delays must be non-negative")
    if len(_entry_delays) != n_droplets:
        raise ValueError(
            f"entry_delays length ({len(_entry_delays)}) must match "
            f"droplet_configs length ({n_droplets})"
        )

    # Initialize per-droplet state
    positions = [dc.initial_position.copy() for dc in droplet_configs]
    velocities = [dc.initial_velocity.copy() for dc in droplet_configs]
    masses = [dc.mass for dc in droplet_configs]
    # Droplets start inactive until their entry time
    entered = [_entry_delays[i] <= 0.0 for i in range(n_droplets)]
    active = [entered[i] for i in range(n_droplets)]

    # Build ideal paths for each droplet (with starting XY offset)
    ideal_paths = [
        IdealPath(
            z_start=dc.initial_position[2],
            z_floor=z_floor,
            target_x=tgt[0],
            target_y=tgt[1],
            start_x=dc.initial_position[0],
            start_y=dc.initial_position[1],
        )
        for dc, tgt in zip(droplet_configs, targets)
    ]

    # Build DropletState objects for force computation
    droplet_states_obj = [
        DropletState(
            position=pos.copy(),
            velocity=vel.copy(),
            diameter=dc.diameter,
            density=dc.material.density,
            target=np.array([tgt[0], tgt[1], z_floor]),
        )
        for pos, vel, dc, tgt in zip(positions, velocities, droplet_configs, targets)
    ]

    # Set up per-droplet force compositors
    compositors = []
    for _ in range(n_droplets):
        comp = ForceCompositor()
        for force in ForceRegistry.enabled_by_default():
            comp.enable(force.name)
        compositors.append(comp)

    # Per-droplet trajectory points
    all_points: List[List[TrajectoryPoint]] = [[] for _ in range(n_droplets)]
    all_deviations: List[List[float]] = [[] for _ in range(n_droplets)]

    # Track which droplets have already emitted a force-misalignment warning
    # to avoid flooding with per-timestep warnings
    _warned_misalignment: set = set()

    # Force smoothing state: EMA to prevent oscillation from phase interference
    _smooth_enabled = force_smoothing is not None and force_smoothing < 1.0
    _smooth_alpha = force_smoothing if _smooth_enabled else 1.0
    _prev_forces: List[np.ndarray] = [np.zeros(3) for _ in range(n_droplets)]
    _first_active_frame: set = set()  # Track first active frame per droplet
    _tdm_turn: int = 0  # Which active-index slot gets focus this step (wraps within active set)
    _step_count: int = 0

    t = 0.0

    while t < max_duration and (any(active) or any(not e for e in entered)):
        # Check for newly entering droplets
        for i in range(n_droplets):
            if not entered[i] and t >= _entry_delays[i]:
                entered[i] = True
                active[i] = True
                # Re-initialize position and velocity from config at entry time
                positions[i] = droplet_configs[i].initial_position.copy()
                velocities[i] = droplet_configs[i].initial_velocity.copy()
                _prev_forces[i] = np.zeros(3)
                _first_active_frame.add(i)
                sim_trace.log("droplet_entry", t=t, droplet_idx=i)

        # Build active droplet states for the solver
        solver_states: List[Tuple[np.ndarray, Tuple[float, float]]] = []
        active_indices: List[int] = []
        active_dc: List[DropletConfig] = []

        for i in range(n_droplets):
            if active[i]:
                # Target offset = target - current position (lateral)
                target_offset = (
                    targets[i][0] - positions[i][0],
                    targets[i][1] - positions[i][1],
                )
                solver_states.append((positions[i].copy(), target_offset))
                active_indices.append(i)
                active_dc.append(droplet_configs[i])

        if not solver_states:
            break

        # 1. Compute phases
        # Strategy: if only 1 active droplet, use single solve_inverse (exact).
        # If multiple active, use round-robin time-division multiplexing:
        # each timestep, solve for ONE droplet (cycling through them).
        # This avoids coherent superposition oscillation when droplets
        # share ring zones. Each droplet gets full steering on its turn.
        if len(active_indices) == 1:
            pos, offset = solver_states[0]
            dc = active_dc[0]
            phases = solve_inverse(
                array, target_offset=offset,
                droplet_z=pos[2], droplet_xy=(pos[0], pos[1]),
                env_config=env_config, droplet_config=dc,
            )
        else:
            # Round-robin: _tdm_turn is an index into the full droplet array
            # (0..n_droplets-1). Find its position in the active list.
            _focus_idx = active_indices.index(_tdm_turn)
            pos, offset = solver_states[_focus_idx]
            dc = active_dc[_focus_idx]
            phases = solve_inverse(
                array, target_offset=offset,
                droplet_z=pos[2], droplet_xy=(pos[0], pos[1]),
                env_config=env_config, droplet_config=dc,
            )

        # Validate that computed phases produce forces in the correct direction.
        # Only warn once per droplet to avoid flooding with per-timestep warnings.
        # With TDM, only validate the focused droplet — off-turn droplets will
        # naturally have misaligned forces since phases weren't solved for them.
        if len(active_indices) > 1:
            focus_states = [solver_states[_focus_idx]]
            focus_dc = [active_dc[_focus_idx]]
            validation = validate_multi_solution(
                array, phases, focus_states, env_config, focus_dc,
            )
        else:
            validation = validate_multi_solution(
                array, phases, solver_states, env_config, active_dc,
            )
        for v in validation:
            if len(active_indices) > 1:
                drop_idx = active_indices[_focus_idx]
            else:
                drop_idx = active_indices[v['droplet_index']]
            if not v.get('is_correct', True) and drop_idx not in _warned_misalignment:
                _warned_misalignment.add(drop_idx)
                warnings.warn(
                    f"Multi-droplet: Droplet {drop_idx} force misaligned "
                    f"(alignment={v.get('alignment', 0):.2f}). "
                    f"Coherent superposition may be degraded.",
                    stacklevel=2,
                )

        # 2. For each active droplet: compute force, integrate
        for j, idx in enumerate(active_indices):
            pos = positions[idx]
            vel = velocities[idx]
            mass = masses[idx]
            dc = droplet_configs[idx]

            # Compute acoustic force from shared phases
            F_acoustic_raw = compute_force(
                array, phases, pos,
                environment=env_config,
                droplet_config=dc,
            )

            # Smooth force to prevent oscillation from phase interference
            # when multiple droplets are at similar positions
            if _smooth_enabled and t > 0 and idx not in _first_active_frame:
                F_acoustic = (
                    _smooth_alpha * F_acoustic_raw
                    + (1.0 - _smooth_alpha) * _prev_forces[idx]
                )
            else:
                F_acoustic = F_acoustic_raw
                _first_active_frame.discard(idx)
            _prev_forces[idx] = F_acoustic.copy()

            # Compute total force via compositor
            result = compositors[idx].compute_total(
                pos, vel, env_config, dc,
                acoustic_force=F_acoustic,
            )
            F_total = result.force

            # Ideal path error for visualization
            errors = ideal_paths[idx].get_error(pos, vel)
            ideal_pos = np.array(errors['ideal_pos'])
            pos_error = errors['pos_error']
            all_deviations[idx].append(errors['pos_error_mag'])

            # Predicted landing
            plan = compute_optimal_trajectory(pos, vel, targets[idx], z_floor)

            # Store point
            point = TrajectoryPoint(
                t=t,
                position=pos.copy(),
                velocity=vel.copy(),
                ideal_position=ideal_pos,
                pos_error=pos_error.copy(),
                predicted_landing=plan['predicted_landing'].copy(),
                landing_error=plan['landing_error'].copy(),
                t_remaining=plan['t_remaining'],
                force_acoustic=F_acoustic.copy(),
                force_total=F_total.copy(),
                control_offset=(0.0, 0.0),
                phases=None,
            )
            all_points[idx].append(point)

            # 3. Euler integration
            acceleration = F_total / mass
            new_vel = vel + acceleration * dt
            new_pos = pos + new_vel * dt

            velocities[idx] = new_vel
            positions[idx] = new_pos

            # Check landing
            if new_pos[2] <= z_floor:
                positions[idx][2] = z_floor  # Clamp to landing surface
                active[idx] = False
                sim_trace.log(
                    "droplet_landing", t=t, droplet_idx=idx,
                    position=(float(new_pos[0]), float(new_pos[1])),
                )

        # Advance TDM: step through all droplets, skip inactive ones
        if sum(active) > 0:
            _tdm_turn = (_tdm_turn + 1) % n_droplets
            # Skip landed droplets in round-robin
            attempts = 0
            while not active[_tdm_turn] and attempts < n_droplets:
                _tdm_turn = (_tdm_turn + 1) % n_droplets
                attempts += 1
            # Trace TDM focus every 100 steps to avoid spam
            if len(active_indices) > 1 and _step_count % 100 == 0:
                sim_trace.log("tdm_focus", t=t, focus_droplet=_tdm_turn)
        _step_count += 1
        t += dt

    # Build TrackedTrajectory objects
    trajectories: List[TrackedTrajectory] = []
    for i in range(n_droplets):
        landing_pos = positions[i].copy()
        landing_err = float(np.linalg.norm(
            landing_pos[:2] - np.array([targets[i][0], targets[i][1]])
        ))
        devs = all_deviations[i]

        traj = TrackedTrajectory(
            points=all_points[i],
            ideal_path=ideal_paths[i],
            target=targets[i],
            droplet=droplet_states_obj[i],
            landing_position=landing_pos,
            landing_error=landing_err,
            max_deviation=max(devs) if devs else 0.0,
            mean_deviation=float(np.mean(devs)) if devs else 0.0,
            final_error=landing_err,
        )
        trajectories.append(traj)

    return trajectories


# =============================================================================
# Comparison / demo (uses config objects)
# =============================================================================

def compare_with_baseline(
    array: ArrayGeometry,
    target: Tuple[float, float],
    initial_position: np.ndarray,
    env_config: EnvironmentConfig,
    droplet_config: DropletConfig,
    save_path: Optional[str] = None,
) -> dict:
    """
    Compare path tracking against baseline.

    Args:
        array: Transducer array geometry
        target: Target landing position (x, y)
        initial_position: Starting position
        env_config: Environment configuration (REQUIRED)
        droplet_config: Droplet configuration (REQUIRED)
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    from .trajectory import simulate_trajectory

    logger.info("Simulating trajectories...")

    # Baseline: fixed controller
    baseline = simulate_trajectory(
        array, target, initial_position,
        control_mode='fixed',
    )
    logger.info("  Fixed:        error = %.2f mm", baseline.landing_error*1000)

    # Path tracking (legacy PD)
    tracked = simulate_path_tracking(
        array, target, initial_position,
        env_config=env_config,
        droplet_config=droplet_config,
        control_mode='legacy',
    )
    logger.info("  Legacy PD:    error = %.2f mm", tracked.landing_error*1000)

    return {
        'baseline': baseline,
        'tracked': tracked,
    }


def main():
    """Demo path tracking controller with config objects."""
    from .config import default_environment, aluminum_droplet

    logger.info("=" * 60)
    logger.info("DRIP PATH TRACKING CONTROLLER")
    logger.info("=" * 60)
    logger.info("")

    # Create array
    array = ArrayGeometry.cylinder(radius=0.05, height=0.4, rings=10, per_ring=12)
    logger.info("Array: %s transducers", array.n)

    # Config objects (REQUIRED)
    env_config = default_environment()
    drop_config = aluminum_droplet()

    # Parameters
    target = (10e-3, 5e-3)
    initial_position = np.array([0, 0, 0.18])

    logger.info("Target: (%.1f, %.1f) mm", target[0]*1000, target[1]*1000)
    logger.info("Start:  %.0f mm height", initial_position[2]*1000)
    logger.info("Environment: %s", env_config)
    logger.info("Droplet: %s", drop_config)
    logger.info("")

    # Compare controllers
    results = compare_with_baseline(
        array, target, initial_position,
        env_config=env_config,
        droplet_config=drop_config,
        save_path='control_comparison.png',
    )

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS:")
    logger.info("-" * 60)
    logger.info("Fixed controller:  %.2f mm error", results['baseline'].landing_error*1000)
    logger.info("Legacy PD:         %.2f mm error", results['tracked'].landing_error*1000)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
