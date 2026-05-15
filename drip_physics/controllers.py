"""
Controller types and implementations for Drip droplet guidance.

This module contains:
- ControllerState: shared input dataclass for all controller modes
- ControllerFn: protocol for controller functions
- IdealPath: ideal arc trajectory reference
- compute_remaining_time: time-to-landing helper
- Controller functions: compute_legacy_control, compute_p_control,
  compute_pi_control, compute_pid_control
- CONTROLLER_MODES / REMOVED_MODES dispatch dicts
- get_controller: mode name -> controller function lookup

Controller Protocol:
    Every controller mode implements the same interface:
        compute_<mode>_control(state: ControllerState) -> Tuple[float, float]
    where ControllerState carries all needed info and the return
    is a (dx, dy) offset for the inverse solver.

Status of controller modes:
    - 'legacy':   PD on position error. Kept for Ryota as baseline reference.
    - 'p_only':   Proportional-only (SW-16 C2).
    - 'pi':       Proportional + Integral with anti-windup (SW-16 C2).
    - 'pid':      Full PID with derivative filter (SW-16 C2).

    Removed in Layer 7 (archived in drip_physics/archive/deprecated_controllers.py):
    - 'bangbang', 'feedforward', 'mpc', 'velocity' — heuristic modes with no
      path to real control theory. Raises ValueError if requested.

Config objects (EnvironmentConfig, DropletConfig) are REQUIRED arguments
at all public API boundaries. No hardcoded physical constants.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import (
    EnvironmentConfig,
    DropletConfig,
    GRAVITY,
)
from .trace import sim_trace


_CONTROLLER_CLAMP_FRACTION = 0.8  # Fraction of cylinder radius for max lateral offset

# Derivative filter coefficient (low-pass, single-pole IIR).
# alpha=0.3: reasonably aggressive noise rejection while remaining responsive.
# [ASMP-033] Not a physics constant — a signal-processing design choice.
_DERIV_FILTER_ALPHA = 0.3

# Velocity normalization ratio: vel_error is normalized by (norm_scale / _VEL_NORM_RATIO).
# This means velocity error is scaled 5x relative to position error, giving the D-term
# appropriate authority. Used consistently across legacy PD, PI, and PID controllers.
_VEL_NORM_RATIO = 0.2


# =============================================================================
# Controller State — shared input for all controller modes
# =============================================================================

@dataclass
class ControllerState:
    """
    Snapshot of everything a controller needs to compute its output.

    Note: integral_accumulator and prev_velocity are mutated in place by
    PI/PID controllers for carry-forward state. All other fields are read-only.

    Every controller mode receives a ControllerState and returns
    (offset_x, offset_y) for the inverse solver. This ensures a
    consistent interface across all modes.

    New fields for PI/PID controllers (SW-16, Group C2):
        ki:                   Integral gain (None = disabled)
        windup_limit:         Max accumulator magnitude in meters (anti-windup cap)
        deriv_filter:         Low-pass filter on D term to reduce noise amplification
        integral_accumulator: Mutable 2-element array carrying integral state across steps.
                              The simulation loop must carry this forward between timesteps.
        ring_changed:         Set True by simulation loop when ring index transitions.
                              PI/PID controllers reset the accumulator when this is True.
                              [ASMP-033] See ASSUMPTIONS.md for rationale.
        prev_velocity:        Previous velocity for derivative filter (2-element array).
    """
    position: NDArray[np.float64]       # [x, y, z]
    velocity: NDArray[np.float64]       # [vx, vy, vz]
    target: Tuple[float, float]         # landing target (x, y)
    z_floor: float                      # landing height
    ideal_path: "IdealPath"             # reference trajectory
    env_config: EnvironmentConfig       # REQUIRED
    droplet_config: DropletConfig        # REQUIRED
    t_elapsed: float = 0.0
    t_total: float = 0.5
    kp: Optional[float] = None          # PD proportional gain (used by legacy controller)
    kd: Optional[float] = None          # PD derivative gain (used by legacy controller)
    # PI/PID fields
    ki: Optional[float] = None                          # Integral gain
    windup_limit: Optional[float] = None                # Accumulator cap (meters)
    deriv_filter: bool = True                           # Low-pass on D term
    integral_accumulator: Optional[NDArray[np.float64]] = None  # Mutable carry-forward state
    ring_changed: bool = False                          # True when ring index transitions
    prev_velocity: Optional[NDArray[np.float64]] = None  # Previous lateral velocity for D filter
    dt: float = 1e-4                                     # Integration timestep (must match simulation loop dt)
    block_states: Optional[Dict[str, Any]] = None  # Carry-forward state for compiled block diagrams


class ControllerFn(Protocol):
    """Protocol for controller functions: state -> (offset_x, offset_y)."""
    def __call__(self, state: ControllerState) -> Tuple[float, float]: ...


# =============================================================================
# Ideal Path
# =============================================================================

@dataclass
class IdealPath:
    """
    Ideal ARC trajectory: lateral motion early, then drop straight down.

    The ideal path has THREE phases:
    1. PUSH (0-30%): Accelerate laterally toward target
    2. COAST (30-70%): Arrive directly above target, lateral velocity -> 0
    3. DROP (70-100%): Fall straight down with zero lateral velocity

    At z_hover (70% of drop), the droplet should be:
    - Position: (target_x, target_y, z_hover)
    - Velocity: (0, 0, vz) - pure vertical

    This ensures the droplet lands falling STRAIGHT DOWN.

    Config note: Uses GRAVITY from config module (not hardcoded).
    """
    z_start: float      # Starting height (m)
    z_floor: float      # Landing height (m)
    target_x: float     # Target X landing position (m)
    target_y: float     # Target Y landing position (m)
    start_x: float = 0.0   # Starting X position (m) — nozzle offset
    start_y: float = 0.0   # Starting Y position (m) — nozzle offset

    # Phase boundaries (fraction of total drop)
    # Setting to 1.0 = linear path (original behavior)
    # Setting to <1.0 = arc path (arrive above target, then drop straight)
    hover_fraction: float = 1.0  # Linear path for now

    # Precomputed values
    fall_time: float = field(init=False)
    z_hover: float = field(init=False)  # Height where we should be above target

    def __post_init__(self):
        """Compute fall time and hover height."""
        h = self.z_start - self.z_floor
        self.fall_time = np.sqrt(2 * h / GRAVITY) if h > 0 else 0.01

        # z_hover: height at which lateral motion should be complete
        # At 70% of drop, we should be directly above target
        self.z_hover = self.z_start - self.hover_fraction * (self.z_start - self.z_floor)

    def get_ideal_position(self, z: float) -> Tuple[float, float, float]:
        """
        Get ideal position at height z.

        Uses smooth S-curve for lateral motion that:
        - Starts at (start_x, start_y) at z_start
        - Reaches (target_x, target_y) at z_hover
        - Stays at (target_x, target_y) below z_hover
        """
        # Progress through total drop (0 = start, 1 = floor)
        total_progress = (self.z_start - z) / (self.z_start - self.z_floor)
        total_progress = np.clip(total_progress, 0, 1)

        # Progress through lateral phase (0 = start, 1 = above target)
        if total_progress <= self.hover_fraction:
            # Normalize to [0, 1] within lateral phase
            lateral_progress = total_progress / self.hover_fraction
            # S-curve: smooth acceleration then deceleration
            # Using smoothstep: 3t^2 - 2t^3
            smooth = 3 * lateral_progress**2 - 2 * lateral_progress**3
            # Interpolate from start position to target position
            x = self.start_x + (self.target_x - self.start_x) * smooth
            y = self.start_y + (self.target_y - self.start_y) * smooth
        else:
            # Below hover height: stay directly above target
            x = self.target_x
            y = self.target_y

        return (x, y, z)

    def get_ideal_velocity(self, z: float, vz: float) -> Tuple[float, float, float]:
        """
        Get ideal velocity at height z.

        Returns the velocity the droplet SHOULD have to follow ideal path.
        Below z_hover, lateral velocity should be zero.
        """
        total_progress = (self.z_start - z) / (self.z_start - self.z_floor)
        total_progress = np.clip(total_progress, 0, 1)

        if total_progress <= self.hover_fraction:
            # During lateral phase: compute velocity from S-curve derivative
            lateral_progress = total_progress / self.hover_fraction
            # d/dt[3t^2 - 2t^3] = 6t - 6t^2
            d_smooth = 6 * lateral_progress - 6 * lateral_progress**2

            # Chain rule: dx/dt = dx/dprogress * dprogress/dz * dz/dt
            # dprogress/dz = -1 / (z_start - z_floor)
            # dz/dt = vz
            dz = self.z_start - self.z_floor
            scale = -vz / (dz * self.hover_fraction) if dz > 0 else 0

            vx = (self.target_x - self.start_x) * d_smooth * scale
            vy = (self.target_y - self.start_y) * d_smooth * scale
        else:
            # Below hover: zero lateral velocity (dropping straight down)
            vx = 0.0
            vy = 0.0

        return (vx, vy, vz)

    def get_ideal_acceleration(self, z: float, vz: float, az: float = -GRAVITY) -> Tuple[float, float, float]:
        """
        Get ideal acceleration at height z.

        Returns the acceleration needed to follow the ideal S-curve path.
        This is used for feedforward control.
        """
        total_progress = (self.z_start - z) / (self.z_start - self.z_floor)
        total_progress = np.clip(total_progress, 0, 1)

        if total_progress <= self.hover_fraction:
            # During lateral phase: compute acceleration from S-curve second derivative
            lateral_progress = total_progress / self.hover_fraction
            # d^2/dt^2[3t^2 - 2t^3] with respect to lateral_progress: 6 - 12t
            d2_smooth = 6 - 12 * lateral_progress

            # Chain rule for acceleration (see velocity computation)
            dz = self.z_start - self.z_floor
            scale = -vz / (dz * self.hover_fraction) if dz > 0 else 0

            # Second derivative includes term from changing vz
            # a = d/dt[v] = d/dt[target * d_smooth * scale]
            # We need to account for both d_smooth changing and scale changing
            d_smooth = 6 * lateral_progress - 6 * lateral_progress**2

            # dscale/dt = d/dt[-vz / (dz * hover_frac)] = -az / (dz * hover_frac)
            dscale_dt = -az / (dz * self.hover_fraction) if dz > 0 else 0

            # dprogress/dt = -vz / (dz * hover_frac) = scale basically
            dprogress_dt = scale

            # a = (target - start) * (d2_smooth * dprogress_dt * scale + d_smooth * dscale_dt)
            ax = (self.target_x - self.start_x) * (d2_smooth * dprogress_dt * scale + d_smooth * dscale_dt)
            ay = (self.target_y - self.start_y) * (d2_smooth * dprogress_dt * scale + d_smooth * dscale_dt)
        else:
            # Below hover: zero lateral acceleration (free fall)
            ax = 0.0
            ay = 0.0

        return (ax, ay, az)

    def get_error(self, actual_pos: np.ndarray, actual_vel: np.ndarray) -> dict:
        """Compute tracking errors from ideal arc path."""
        ideal_pos = self.get_ideal_position(actual_pos[2])
        ideal_vel = self.get_ideal_velocity(actual_pos[2], actual_vel[2])

        pos_error = np.array(ideal_pos) - actual_pos
        vel_error = np.array(ideal_vel) - actual_vel

        return {
            'pos_error': pos_error,
            'vel_error': vel_error,
            'pos_error_mag': np.linalg.norm(pos_error[:2]),
            'vel_error_mag': np.linalg.norm(vel_error[:2]),
            'ideal_pos': ideal_pos,
            'ideal_vel': ideal_vel,
        }


# =============================================================================
# Trajectory planning helpers
# =============================================================================

def compute_remaining_time(z: float, vz: float, z_floor: float) -> float:
    """
    Compute time remaining until landing, accounting for current velocity.

    Solves: z + vz*t - 0.5*g*t^2 = z_floor
    Quadratic: -0.5*g*t^2 + vz*t + (z - z_floor) = 0

    Returns:
        Estimated time to landing (seconds)
    """
    h = z - z_floor
    if h <= 0:
        # Droplet has landed (at or below floor). Remaining time is zero.
        return 0.0

    # Quadratic formula: t = (vz + sqrt(vz^2 + 2gh)) / g
    discriminant = vz**2 + 2 * GRAVITY * h
    if discriminant < 0:
        warnings.warn(
            "compute_remaining_time: negative discriminant — droplet moving upward. "
            "Estimating time to apex + fall.",
            stacklevel=2,
        )
        # Time to reach apex: t_up = |vz| / g
        # Time to fall from apex: t_down = sqrt(2 * (z + vz^2/(2g) - z_floor) / g)
        t_up = abs(vz) / GRAVITY if GRAVITY > 0 else 0.1
        apex_z = z + vz**2 / (2 * GRAVITY) if GRAVITY > 0 else z
        fall_h = apex_z - z_floor
        t_down = np.sqrt(2 * max(fall_h, 0) / GRAVITY) if GRAVITY > 0 else 0.1
        return t_up + t_down

    t_remaining = (vz + np.sqrt(discriminant)) / GRAVITY

    if t_remaining <= 0:
        raise ValueError(
            f"compute_remaining_time: non-positive t_remaining={t_remaining:.6f}s "
            f"(z={z:.4f}, vz={vz:.4f}, z_floor={z_floor:.4f}). "
            "This should not occur for a droplet still above z_floor — check inputs."
        )

    return t_remaining


# =============================================================================
# Controller: legacy PD (BASELINE — kept for Ryota)
# =============================================================================

def compute_legacy_control(state: ControllerState) -> Tuple[float, float]:
    """
    CONTINUOUS PD CONTROLLER — always applies proportional force.

    Simple principle: The force is ALWAYS proportional to how far off we are.
    - Position error -> proportional force (kp)
    - Velocity error -> derivative force (kd)

    NORMALIZATION: Errors are normalized by target distance so gains work
    for any target size.

    This is the baseline reference controller. Kept for Ryota to compare
    against real control theory implementations in Section 5.
    """
    pos_xy = state.position[:2]
    vel_xy = state.velocity[:2]
    z = state.position[2]
    vz = state.velocity[2]

    target_xy = np.array([state.ideal_path.target_x, state.ideal_path.target_y])
    norm_scale = max(np.linalg.norm(target_xy), 0.005)

    # Get ideal position/velocity at current z
    ideal_pos = np.array(state.ideal_path.get_ideal_position(z)[:2])
    ideal_vel = np.array(state.ideal_path.get_ideal_velocity(z, vz)[:2])

    # Errors (target - actual)
    pos_error = ideal_pos - pos_xy
    vel_error = ideal_vel - vel_xy

    # Normalize errors
    pos_error_norm = pos_error / norm_scale
    vel_error_norm = vel_error / (norm_scale / _VEL_NORM_RATIO)

    # PD control -- use gains from ControllerState if provided, else defaults
    kp = state.kp if state.kp is not None else 1.0
    kd = state.kd if state.kd is not None else 0.5
    control = kp * pos_error_norm + kd * vel_error_norm

    # Scale to physical offset (meters). Clamp to config-derived limit.
    # [ASMP-032] Previously hardcoded to 0.02 (20mm), tuned for 135mm array + aluminum.
    # Now derived from array geometry: cylinder_radius is the physical max offset.
    max_control = state.env_config.array.cylinder_radius * _CONTROLLER_CLAMP_FRACTION  # 80% of cylinder radius
    control = control * max_control

    # Limit magnitude
    magnitude = np.linalg.norm(control)
    if magnitude > max_control:
        sim_trace.log(
            "force_clamp", t=state.t_elapsed,
            raw=float(magnitude), max=float(max_control), controller="legacy",
        )
        control = control / magnitude * max_control

    return (control[0], control[1])


# =============================================================================
# P controller — proportional only (SW-16 C2 baseline)
# =============================================================================

def compute_p_control(state: ControllerState) -> Tuple[float, float]:
    """
    PURE PROPORTIONAL CONTROLLER — position error only, no derivative term.

    Inline P-only logic (no delegation to legacy PD). This avoids mutating
    ControllerState, which violates the immutability contract.

    Uses the same normalization as compute_legacy_control so Kp values are
    directly comparable between P, PI, PID, and legacy modes.
    """
    pos_xy = state.position[:2]
    z = state.position[2]

    target_xy = np.array([state.ideal_path.target_x, state.ideal_path.target_y])
    norm_scale = max(np.linalg.norm(target_xy), 0.005)

    ideal_pos = np.array(state.ideal_path.get_ideal_position(z)[:2])
    pos_error = ideal_pos - pos_xy
    pos_error_norm = pos_error / norm_scale

    kp = state.kp if state.kp is not None else 1.0
    control = kp * pos_error_norm

    max_control = state.env_config.array.cylinder_radius * _CONTROLLER_CLAMP_FRACTION
    control = control * max_control

    magnitude = np.linalg.norm(control)
    if magnitude > max_control:
        sim_trace.log(
            "force_clamp", t=state.t_elapsed,
            raw=float(magnitude), max=float(max_control), controller="p_only",
        )
        control = control / magnitude * max_control

    return (float(control[0]), float(control[1]))


# =============================================================================
# PI controller — proportional + integral (SW-16 C2)
# =============================================================================

def compute_pi_control(state: ControllerState) -> Tuple[float, float]:
    """
    PROPORTIONAL + INTEGRAL CONTROLLER (SW-16 C2).

    P term: Kp * position_error  (same normalization as legacy PD)
    I term: Ki * integral_accumulator  (accumulated position error over time)

    Anti-windup: If output saturates at max_control, the integral accumulator
    is clamped to windup_limit to prevent unbounded growth. Default windup_limit
    is 0.01 m (10 mm) — the UI 'ctrl-windup-limit' field.

    Ring handoff reset [ASMP-033]: When state.ring_changed is True, the
    integral accumulator is reset to zero before computing the output.
    Each ring has an independent phase geometry; carrying accumulated error
    from ring N into ring N+1 drives correction in the wrong direction.

    The integral_accumulator field on ControllerState is updated IN PLACE.
    The simulation loop must carry state.integral_accumulator forward across
    timesteps (pass the same array to the next step's state construction).

    Args:
        state: Controller state with integral_accumulator for carry-forward.
               Must have ki set. windup_limit defaults to 0.01 m if None.

    Returns:
        (offset_x, offset_y) in meters, clamped to max_control.
    """
    pos_xy = state.position[:2]
    vel_xy = state.velocity[:2]
    z = state.position[2]
    vz = state.velocity[2]

    target_xy = np.array([state.ideal_path.target_x, state.ideal_path.target_y])
    norm_scale = max(np.linalg.norm(target_xy), 0.005)

    ideal_pos = np.array(state.ideal_path.get_ideal_position(z)[:2])
    pos_error = ideal_pos - pos_xy
    pos_error_norm = pos_error / norm_scale

    # Initialize or retrieve accumulator
    if state.integral_accumulator is None:
        state.integral_accumulator = np.zeros(2)
    # Keep prev_velocity current even though PI doesn't use D-term.
    # Prevents a transient spike if controller mode switches to PID mid-flight.
    if state.prev_velocity is None:
        state.prev_velocity = vel_xy.copy()
    else:
        state.prev_velocity = vel_xy.copy()

    # [ASMP-033] Ring handoff reset: new ring = new phase geometry.
    # Carrying accumulated error from previous ring drives wrong correction.
    # Anti-windup via output clamping. See ASSUMPTIONS.md.
    # On the handoff frame: reset accumulator to zero and skip accumulation for
    # this step — the old error is not meaningful in the new ring geometry.
    if state.ring_changed:
        state.integral_accumulator = np.zeros(2)
        # Return P-only output for the handoff frame (no integral contribution)
        kp = state.kp if state.kp is not None else 1.0
        control = kp * pos_error_norm
        max_control = state.env_config.array.cylinder_radius * _CONTROLLER_CLAMP_FRACTION
        control = control * max_control
        magnitude = np.linalg.norm(control)
        if magnitude > max_control:
            sim_trace.log(
                "force_clamp", t=state.t_elapsed,
                raw=float(magnitude), max=float(max_control), controller="pi",
            )
            control = control / magnitude * max_control
        return (float(control[0]), float(control[1]))

    # Windup limit (default 10 mm — matches UI ctrl-windup-limit field)
    windup_limit = state.windup_limit if state.windup_limit is not None else 0.01

    # Accumulate integral: simple rectangular (Euler) integration.
    # dt is approximated from t_total / expected steps; using 1ms if t_total unavailable.
    # The correct dt comes from the simulation loop dt — controllers receive t_elapsed
    # but not explicit dt. We use a fixed dt=1e-3 here as the step size implied by
    # the simulation loop default dt=1e-4 is too small to require anything fancier.
    # Integration timestep read from ControllerState (set by simulation loop).
    _dt = state.dt

    # Update accumulator (normalized error, same units as P term)
    state.integral_accumulator = state.integral_accumulator + pos_error_norm * _dt

    # Clamp accumulator magnitude to windup_limit
    accum_mag = np.linalg.norm(state.integral_accumulator)
    if accum_mag > windup_limit:
        state.integral_accumulator = state.integral_accumulator / accum_mag * windup_limit

    # Gains
    kp = state.kp if state.kp is not None else 1.0
    ki = state.ki if state.ki is not None else 0.0

    # P + I
    control = kp * pos_error_norm + ki * state.integral_accumulator

    # Scale to physical offset and clamp
    max_control = state.env_config.array.cylinder_radius * _CONTROLLER_CLAMP_FRACTION
    control = control * max_control

    magnitude = np.linalg.norm(control)
    if magnitude > max_control:
        sim_trace.log(
            "force_clamp", t=state.t_elapsed,
            raw=float(magnitude), max=float(max_control), controller="pi",
        )
        control = control / magnitude * max_control

    return (float(control[0]), float(control[1]))


# =============================================================================
# PID controller — proportional + integral + derivative (SW-16 C2)
# =============================================================================

def compute_pid_control(state: ControllerState) -> Tuple[float, float]:
    """
    FULL PID CONTROLLER (SW-16 C2).

    P term: Kp * position_error  (same normalization as legacy PD)
    I term: Ki * integral_accumulator  (same as PI controller, including ring reset)
    D term: Kd * velocity_error  (same as legacy PD's derivative term)

    Optional derivative filter: single-pole IIR low-pass on the D term.
    When deriv_filter=True, the D term is filtered using the previous velocity
    stored in state.prev_velocity. This reduces noise amplification in the
    derivative channel. Filter coefficient: alpha=0.3 (aggressive rejection).

    Ring handoff reset and anti-windup are identical to compute_pi_control.

    Args:
        state: Controller state with integral_accumulator and prev_velocity
               for carry-forward across timesteps.

    Returns:
        (offset_x, offset_y) in meters, clamped to max_control.
    """
    pos_xy = state.position[:2]
    vel_xy = state.velocity[:2]
    z = state.position[2]
    vz = state.velocity[2]

    target_xy = np.array([state.ideal_path.target_x, state.ideal_path.target_y])
    norm_scale = max(np.linalg.norm(target_xy), 0.005)

    ideal_pos = np.array(state.ideal_path.get_ideal_position(z)[:2])
    ideal_vel = np.array(state.ideal_path.get_ideal_velocity(z, vz)[:2])

    pos_error = ideal_pos - pos_xy
    vel_error = ideal_vel - vel_xy

    pos_error_norm = pos_error / norm_scale
    vel_error_norm = vel_error / (norm_scale / _VEL_NORM_RATIO)

    # Initialize accumulator and prev_velocity if needed
    if state.integral_accumulator is None:
        state.integral_accumulator = np.zeros(2)
    if state.prev_velocity is None:
        state.prev_velocity = vel_xy.copy()

    # [ASMP-033] Ring handoff reset: new ring = new phase geometry.
    # Carrying accumulated error from previous ring drives wrong correction.
    # Anti-windup via output clamping. See ASSUMPTIONS.md.
    # On the handoff frame: reset accumulator to zero and skip integral accumulation.
    if state.ring_changed:
        state.integral_accumulator = np.zeros(2)
        # Fall through to compute D term and P-only + D output for this frame
        # (no I contribution on handoff frame)
        windup_limit = state.windup_limit if state.windup_limit is not None else 0.01
    else:
        windup_limit = state.windup_limit if state.windup_limit is not None else 0.01
        _dt = state.dt

        # Integral update with windup clamping
        state.integral_accumulator = state.integral_accumulator + pos_error_norm * _dt
        accum_mag = np.linalg.norm(state.integral_accumulator)
        if accum_mag > windup_limit:
            state.integral_accumulator = state.integral_accumulator / accum_mag * windup_limit

    # Derivative term: optional low-pass filter
    if state.deriv_filter:
        # Single-pole IIR: filtered_vel = alpha * vel_xy + (1-alpha) * prev_vel
        filtered_vel = _DERIV_FILTER_ALPHA * vel_xy + (1.0 - _DERIV_FILTER_ALPHA) * state.prev_velocity
        # Update prev_velocity for next call (store filtered value)
        state.prev_velocity = filtered_vel.copy()
        # Recompute vel_error using filtered velocity
        vel_error_filtered = ideal_vel - filtered_vel
        d_term = vel_error_filtered / (norm_scale / _VEL_NORM_RATIO)
    else:
        d_term = vel_error_norm
        # Update prev_velocity for next call (store raw value)
        state.prev_velocity = vel_xy.copy()

    # Gains
    kp = state.kp if state.kp is not None else 1.0
    ki = state.ki if state.ki is not None else 0.0
    kd = state.kd if state.kd is not None else 0.5

    # P + I + D
    control = kp * pos_error_norm + ki * state.integral_accumulator + kd * d_term

    # Scale to physical offset and clamp
    max_control = state.env_config.array.cylinder_radius * _CONTROLLER_CLAMP_FRACTION
    control = control * max_control

    magnitude = np.linalg.norm(control)
    if magnitude > max_control:
        sim_trace.log(
            "force_clamp", t=state.t_elapsed,
            raw=float(magnitude), max=float(max_control), controller="pid",
        )
        control = control / magnitude * max_control

    return (float(control[0]), float(control[1]))


# =============================================================================
# Controller dispatch
# =============================================================================

CONTROLLER_MODES = {
    "legacy": compute_legacy_control,
    # SW-16 C2 — real control theory controllers
    "p_only": compute_p_control,
    "pi": compute_pi_control,
    "pid": compute_pid_control,
}

# Modes removed in Layer 7A. Archived in drip_physics/archive/deprecated_controllers.py.
# Layer 4 will replace these with proper G(s)-derived controllers (Ryota).
REMOVED_MODES = {"bangbang", "feedforward", "mpc", "velocity"}


def get_controller(mode: str) -> ControllerFn:
    """
    Get a controller function by mode name.

    Args:
        mode: One of 'legacy', 'p_only', 'pi', 'pid'

    Returns:
        Controller function with signature (ControllerState) -> (float, float)

    Raises:
        ValueError: If mode is unknown or has been removed
    """
    if mode in REMOVED_MODES:
        raise ValueError(
            f"Controller mode '{mode}' was removed in Layer 7. "
            f"Use 'legacy', 'p_only', 'pi', or 'pid' instead. "
            f"Archived code is in drip_physics/archive/deprecated_controllers.py."
        )
    if mode not in CONTROLLER_MODES:
        raise ValueError(
            f"Unknown control mode '{mode}'. "
            f"Available: {sorted(CONTROLLER_MODES.keys())}."
        )
    return CONTROLLER_MODES[mode]
