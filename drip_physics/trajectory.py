#!/usr/bin/env python3
"""
Trajectory simulation for Drip droplet guidance.

This module integrates the actual physics:
1. Gravity pulls droplet down
2. Acoustic force pushes droplet laterally
3. Air drag slows the droplet (Re-regime-selected drag law)
4. F=ma determines acceleration
5. RK45 adaptive integration gives velocity and position over time

The result is a predicted trajectory showing where the droplet
actually lands given a target and control strategy.

Drag law selection (from DropletConfig):
- Re < 1:    Stokes       Cd = 24/Re
- Re < 1000: Schiller-Naumann  Cd = (24/Re)(1 + 0.15*Re^0.687)
- Re >= 1000: Newton       Cd = 0.44

Integration: scipy.integrate.solve_ivp with RK45 (adaptive timestep
driven by error tolerance, not arbitrary dt).

Control modes:
- 'fixed': Always push toward target from origin (naive, overshoots)
- 'relative': Push toward (target - current_pos), tracks better
- 'pd': Relative + velocity damping, reduces overshoot
- 'predictive': Accounts for fall time, aims for landing spot

Run with:
    python -m drip_physics.trajectory
"""

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal, TYPE_CHECKING

from scipy.integrate import solve_ivp

from .config import (
    GRAVITY,
    DropletConfig,
    EnvironmentConfig,
    AIR_20C,
    aluminum_droplet,
    default_environment,
)
from .core import ArrayGeometry, DropletState, PhaseArray
from .inverse import solve_inverse
from .force import compute_force

if TYPE_CHECKING:
    from .pressure_backend import PressureBackend


# Control mode type
ControlMode = Literal['fixed', 'relative', 'pd', 'predictive']


def compute_control_offset(
    target: Tuple[float, float],
    position: np.ndarray,
    velocity: np.ndarray,
    z_floor: float,
    mode: ControlMode = 'fixed',
    kp: float = 1.0,
    kd: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute the effective target offset for the inverse solver.

    This is the CONTROL LOOP - it determines what direction to push
    based on the current state and desired target.

    Args:
        target: Final target position (x, y) in meters
        position: Current droplet position [x, y, z]
        velocity: Current droplet velocity [vx, vy, vz]
        z_floor: Landing Z position
        mode: Control strategy
        kp: Proportional gain (for pd mode)
        kd: Derivative gain (for pd mode)

    Returns:
        Effective target offset (x, y) to pass to inverse solver
    """
    target_xy = np.array([target[0], target[1]])
    pos_xy = position[:2]
    vel_xy = velocity[:2]

    if mode == 'fixed':
        # Original naive behavior: always push toward target from origin
        return target

    elif mode == 'relative':
        # Push toward target from CURRENT position
        # If at target, offset is zero (no push)
        error = target_xy - pos_xy
        return (error[0], error[1])

    elif mode == 'pd':
        # PD control: proportional + derivative (velocity damping)
        # error = where we want to be - where we are
        # control = kp * error - kd * velocity
        error = target_xy - pos_xy
        control = kp * error - kd * vel_xy
        return (control[0], control[1])

    elif mode == 'predictive':
        # Predict where droplet will land if we stop pushing now
        # Then push to correct the predicted error

        # Estimate remaining fall time (simple: sqrt(2h/g))
        z_remaining = position[2] - z_floor
        if z_remaining > 0:
            # More accurate: account for current vz
            # z = z0 + vz*t - 0.5*g*t^2 = z_floor
            # Solve for t using quadratic formula
            vz = velocity[2]
            discriminant = vz**2 + 2 * GRAVITY * z_remaining
            if discriminant > 0:
                t_fall = (vz + np.sqrt(discriminant)) / GRAVITY
            else:
                t_fall = 0.1  # Fallback
        else:
            t_fall = 0.01  # Almost landed

        # Predict landing position if no more lateral force
        # (simplified: assume constant lateral velocity)
        predicted_landing = pos_xy + vel_xy * t_fall

        # Error is difference between target and predicted landing
        error = target_xy - predicted_landing

        # Scale by remaining time (push harder when more time to correct)
        scale = min(t_fall / 0.1, 1.0)  # Cap at 1.0

        return (error[0] * scale, error[1] * scale)

    else:
        raise ValueError(f"Unknown control mode: {mode}")


@dataclass
class TrajectoryPoint:
    """Single point in trajectory."""
    t: float                    # Time (s)
    position: np.ndarray        # [x, y, z] (m)
    velocity: np.ndarray        # [vx, vy, vz] (m/s)
    force_acoustic: np.ndarray  # Acoustic force [Fx, Fy, Fz] (N)
    force_total: np.ndarray     # Total force including gravity/drag (N)
    phases: Optional[PhaseArray] = None  # Phase array at this instant


@dataclass
class Trajectory:
    """Complete trajectory from simulation."""
    points: List[TrajectoryPoint]
    target: Tuple[float, float]
    droplet: DropletState
    landing_position: Optional[np.ndarray] = None
    landing_error: Optional[float] = None

    def __len__(self):
        return len(self.points)

    @property
    def times(self) -> np.ndarray:
        return np.array([p.t for p in self.points])

    @property
    def positions(self) -> np.ndarray:
        return np.array([p.position for p in self.points])

    @property
    def velocities(self) -> np.ndarray:
        return np.array([p.velocity for p in self.points])

    @property
    def x(self) -> np.ndarray:
        return self.positions[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.positions[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.positions[:, 2]

    @property
    def duration(self) -> float:
        return self.times[-1] - self.times[0]


# =============================================================================
# Drag Force Computation
# =============================================================================

def compute_drag_force(
    velocity: np.ndarray,
    radius: float,
    droplet_config: Optional[DropletConfig] = None,
    env_config: Optional[EnvironmentConfig] = None,
) -> np.ndarray:
    """
    Compute drag force on a sphere using Re-based drag law selection.

    When config objects are provided, selects the appropriate drag regime:
    - Re < 1:    Stokes       F = -6*pi*mu*r*v  (Cd = 24/Re)
    - Re < 1000: Schiller-Naumann  Cd = (24/Re)(1 + 0.15*Re^0.687)
    - Re >= 1000: Newton       Cd = 0.44

    General drag: F = -0.5 * rho * |v| * A * Cd * v_hat

    Falls back to Stokes drag with air viscosity at 20C when config
    objects are not provided (backward compatibility).

    Args:
        velocity: Velocity vector [vx, vy, vz] (m/s)
        radius: Droplet radius (m)
        droplet_config: Optional config for Re-based drag selection
        env_config: Optional environment config for medium properties

    Returns:
        Drag force vector (N)
    """
    speed = float(np.linalg.norm(velocity))

    if droplet_config is None or env_config is None:
        # [ASMP-005] Backward-compatible Stokes drag -- invalid at Re~40. See ASSUMPTIONS.md.
        # Pass droplet_config and env_config for correct Schiller-Naumann regime.
        mu = AIR_20C.dynamic_viscosity
        return -6.0 * np.pi * mu * radius * velocity

    medium = env_config.medium

    if speed < 1e-12:
        return np.zeros(3)

    cd = droplet_config.drag_coefficient(speed, medium)

    # F_drag = -0.5 * rho_medium * |v|^2 * A_cross * Cd * v_hat
    # A_cross = pi * r^2  (projected area of sphere)
    rho_medium = medium.density
    area = np.pi * radius ** 2
    drag_magnitude = 0.5 * rho_medium * speed ** 2 * area * cd
    v_hat = velocity / speed

    return -drag_magnitude * v_hat


def _compute_drag_force_stokes_legacy(
    velocity: np.ndarray,
    radius: float,
    viscosity: float,
) -> np.ndarray:
    """
    Pure Stokes drag for convergence testing (Euler reference).

    F_drag = -6*pi*mu*r*v

    This is kept separate so the convergence test compares the same
    drag model between Euler and RK45, isolating the integrator change.
    """
    return -6.0 * np.pi * viscosity * radius * velocity


# =============================================================================
# Shared Force Computation Helper
# =============================================================================

def compute_forces_at_state(
    array: ArrayGeometry,
    phases: PhaseArray,
    position: np.ndarray,
    velocity: np.ndarray,
    droplet: DropletState,
    mass: float,
    radius: float,
    *,
    env_config: Optional[EnvironmentConfig] = None,
    droplet_config: Optional[DropletConfig] = None,
    include_drag: bool = True,
    ring_array=None,
    field_output: Optional[dict] = None,
    backend: Optional["PressureBackend"] = None,
) -> tuple:
    """Compute acoustic, drag, and gravity forces at a given droplet state.

    This is the shared force-computation core used by both simulate_trajectory()
    and (for F_acoustic only) simulate_path_tracking(). It consolidates the
    three-step force assembly that was previously inlined in each loop.

    trajectory.py uses all three returned forces.
    pathtrack.py uses only F_acoustic and lets ForceCompositor handle
    gravity + drag (avoids double-counting with the compositor plugins).

    Args:
        array: Transducer array geometry.
        phases: Pre-computed phase array from inverse solver.
        position: Current droplet position [x, y, z] (m).
        velocity: Current droplet velocity [vx, vy, vz] (m/s).
        droplet: DropletState for force computation.
        mass: Droplet mass (kg).
        radius: Droplet radius (m).
        env_config: Optional environment configuration.
        droplet_config: Optional droplet configuration.
        include_drag: Whether to compute drag force (default True).
        ring_array: Optional RingArray for multi-zone arrays.
        field_output: Optional dict to capture field metadata (e.g. grad_p_magnitude).

    Returns:
        Tuple of (F_acoustic, F_drag, F_gravity) as np.ndarray, each shape (3,).
    """
    F_acoustic = compute_force(
        array, phases, position, droplet,
        ring_array=ring_array,
        environment=env_config,
        droplet_config=droplet_config,
        field_output=field_output,
        backend=backend,
    )

    if include_drag:
        F_drag = compute_drag_force(velocity, radius, droplet_config, env_config)
    else:
        F_drag = np.zeros(3)

    F_gravity = np.array([0.0, 0.0, -mass * GRAVITY])

    return F_acoustic, F_drag, F_gravity


# =============================================================================
# Free-Fall ODE for RK45 (no acoustic force)
# =============================================================================

def _free_fall_ode(
    t: float,
    state: np.ndarray,
    mass: float,
    radius: float,
    include_drag: bool,
    droplet_config: Optional[DropletConfig],
    env_config: Optional[EnvironmentConfig],
) -> np.ndarray:
    """
    ODE right-hand side for free-fall (no acoustic force).

    State vector: [x, y, z, vx, vy, vz]
    Returns: [vx, vy, vz, ax, ay, az]
    """
    velocity = state[3:6]

    # Gravity
    F_gravity = np.array([0.0, 0.0, -mass * GRAVITY])

    # Drag
    if include_drag:
        F_drag = compute_drag_force(velocity, radius, droplet_config, env_config)
    else:
        F_drag = np.zeros(3)

    F_total = F_gravity + F_drag
    acceleration = F_total / mass

    return np.concatenate([velocity, acceleration])


# =============================================================================
# Euler Integrator (for convergence verification only)
# =============================================================================

def _simulate_free_fall_euler(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    mass: float,
    radius: float,
    dt: float,
    max_duration: float,
    z_floor: float,
    include_drag: bool,
    droplet_config: Optional[DropletConfig] = None,
    env_config: Optional[EnvironmentConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Semi-implicit Euler integration for convergence testing.

    Returns:
        (final_position, final_velocity) arrays
    """
    position = np.array(initial_position, dtype=float)
    velocity = np.array(initial_velocity, dtype=float)

    t = 0.0
    while t < max_duration and position[2] > z_floor:
        F_gravity = np.array([0.0, 0.0, -mass * GRAVITY])

        if include_drag:
            F_drag = compute_drag_force(velocity, radius, droplet_config, env_config)
        else:
            F_drag = np.zeros(3)

        F_total = F_gravity + F_drag
        acceleration = F_total / mass
        velocity = velocity + acceleration * dt
        position = position + velocity * dt
        t += dt

    return position, velocity


# =============================================================================
# RK45 Convergence Verification
# =============================================================================

def verify_rk45_convergence(
    droplet_config: Optional[DropletConfig] = None,
    env_config: Optional[EnvironmentConfig] = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    euler_dt: float = 1e-6,
    tolerance_pct: float = 1.0,
) -> bool:
    """
    Verify RK45 converges to the same result as fine-step Euler.

    Runs both integrators on a standard 300um Al droplet free-fall
    trajectory and checks that final positions agree within tolerance_pct%.

    Args:
        droplet_config: Droplet config (default: 300um Al)
        env_config: Environment config (default: air at 20C)
        rtol: RK45 relative tolerance
        atol: RK45 absolute tolerance
        euler_dt: Fine Euler timestep for reference
        tolerance_pct: Maximum allowed position difference in percent

    Returns:
        True if convergence verified within tolerance
    """
    if droplet_config is None:
        droplet_config = aluminum_droplet()
    if env_config is None:
        env_config = default_environment()

    radius = droplet_config.radius
    mass = droplet_config.mass

    initial_position = np.array(droplet_config.initial_position, dtype=float)
    initial_velocity = np.array(droplet_config.initial_velocity, dtype=float)

    z_floor = 0.0
    max_duration = 0.3  # Enough for a meaningful trajectory

    # --- Euler reference at fine dt ---
    euler_pos, euler_vel = _simulate_free_fall_euler(
        initial_position, initial_velocity,
        mass, radius, euler_dt, max_duration, z_floor,
        include_drag=True,
        droplet_config=droplet_config,
        env_config=env_config,
    )

    # --- RK45 ---
    state0 = np.concatenate([initial_position, initial_velocity])

    def z_floor_event(t, state, *args):
        return state[2] - z_floor
    z_floor_event.terminal = True
    z_floor_event.direction = -1

    sol = solve_ivp(
        _free_fall_ode,
        [0.0, max_duration],
        state0,
        method='RK45',
        rtol=rtol,
        atol=atol,
        events=z_floor_event,
        args=(mass, radius, True, droplet_config, env_config),
        dense_output=True,
    )

    if not sol.success:
        warnings.warn(
            f"RK45 convergence test: solver failed — {sol.message}",
            stacklevel=2,
        )
        return False

    rk45_pos = sol.y[:3, -1]

    # Compare final positions
    euler_displacement = np.linalg.norm(euler_pos - initial_position)
    if euler_displacement < 1e-12:
        warnings.warn(
            "RK45 convergence test: Euler displacement is ~0; "
            "cannot compute relative error.",
            stacklevel=2,
        )
        return False

    position_error = np.linalg.norm(rk45_pos - euler_pos)
    relative_error_pct = (position_error / euler_displacement) * 100.0

    if relative_error_pct > tolerance_pct:
        warnings.warn(
            f"RK45 convergence test FAILED: "
            f"position error = {relative_error_pct:.4f}% "
            f"(tolerance = {tolerance_pct}%). "
            f"Euler final = {euler_pos}, RK45 final = {rk45_pos}",
            stacklevel=2,
        )
        return False

    return True


# =============================================================================
# Main Simulation Functions
# =============================================================================

def simulate_trajectory(
    array: ArrayGeometry,
    target: Tuple[float, float],
    initial_position: np.ndarray,
    initial_velocity: Optional[np.ndarray] = None,
    droplet_diameter: float = 300e-6,
    droplet_density: float = 2385.0,  # Liquid aluminum
    dt: float = 1e-4,  # 0.1ms timestep (used for output sampling, not integration)
    max_duration: float = 0.5,  # Max simulation time
    z_floor: float = 0.0,  # Melt pool surface (landing target)
    include_drag: bool = True,
    store_phases: bool = False,
    control_mode: ControlMode = 'fixed',
    kp: float = 1.0,
    kd: float = 0.05,
    droplet_config: Optional[DropletConfig] = None,
    env_config: Optional[EnvironmentConfig] = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> Trajectory:
    """
    Simulate droplet trajectory through acoustic array using RK45.

    The trajectory is integrated in segments: at each output sample
    (spaced by dt), the control offset and acoustic force are recomputed,
    then RK45 integrates the equations of motion for that segment with
    adaptive internal timesteps driven by error tolerance.

    Args:
        array: Transducer array geometry
        target: Target landing offset (x, y) in meters
        initial_position: Starting [x, y, z] in meters
        initial_velocity: Starting velocity (default: zero)
        droplet_diameter: Droplet diameter in meters (ignored if droplet_config provided)
        droplet_density: Droplet density in kg/m3 (ignored if droplet_config provided)
        dt: Output sample interval in seconds (control update rate)
        max_duration: Maximum simulation time
        z_floor: Stop when droplet reaches this Z
        include_drag: Whether to include air drag
        store_phases: Whether to store phase arrays (memory intensive)
        control_mode: Control strategy ('fixed', 'relative', 'pd', 'predictive')
        kp: Proportional gain for PD control
        kd: Derivative gain for PD control
        droplet_config: Optional DropletConfig for Re-based drag
        env_config: Optional EnvironmentConfig for medium properties
        rtol: RK45 relative error tolerance
        atol: RK45 absolute error tolerance

    Returns:
        Trajectory object with full simulation results
    """
    if env_config is None or droplet_config is None:
        warnings.warn(
            "simulate_trajectory() called without env_config/droplet_config — "
            "falling back to legacy defaults. Pass config objects for correct physics.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Initialize droplet state
    position = np.array(initial_position, dtype=float)
    velocity = np.array(
        initial_velocity if initial_velocity is not None else [0, 0, 0],
        dtype=float,
    )

    # Derive mass from config or legacy parameters
    if droplet_config is not None:
        radius = droplet_config.radius
        mass = droplet_config.mass
    else:
        radius = droplet_diameter / 2
        volume = (4.0 / 3.0) * np.pi * radius ** 3
        mass = droplet_density * volume

    # Create DropletState for force computation (legacy interface)
    droplet = DropletState(
        position=position.copy(),
        velocity=velocity.copy(),
        diameter=droplet_config.diameter if droplet_config else droplet_diameter,
        density=droplet_config.material.density if droplet_config else droplet_density,
        target=np.array([target[0], target[1], 0]),
    )

    # Simulation loop: control updates at dt intervals, RK45 within each segment
    points = []
    t = 0.0

    while t < max_duration and position[2] > z_floor:
        # 1. Compute control offset based on control strategy
        control_offset = compute_control_offset(
            target, position, velocity, z_floor,
            mode=control_mode, kp=kp, kd=kd,
        )

        # 2. Compute phases for this control offset
        phases = solve_inverse(
            array, target_offset=control_offset, droplet_z=position[2],
            env_config=env_config, droplet_config=droplet_config,
        )

        # 3. Compute acoustic, drag, and gravity forces at current state
        F_acoustic, F_drag, F_gravity = compute_forces_at_state(
            array, phases, position, velocity, droplet, mass, radius,
            env_config=env_config,
            droplet_config=droplet_config,
            include_drag=include_drag,
        )

        # 4. Total force
        F_total = F_acoustic + F_drag + F_gravity

        # Store trajectory point
        point = TrajectoryPoint(
            t=t,
            position=position.copy(),
            velocity=velocity.copy(),
            force_acoustic=F_acoustic.copy(),
            force_total=F_total.copy(),
            phases=phases if store_phases else None,
        )
        points.append(point)

        # 6. RK45 integration over one control interval [t, t+dt]
        #    The acoustic force is held constant during this interval
        #    (piecewise-constant control), so the ODE is:
        #    dv/dt = (F_acoustic + F_gravity + F_drag(v)) / mass
        #    dx/dt = v

        def segment_ode(t_local, state):
            vel = state[3:6]
            if include_drag:
                F_d = compute_drag_force(
                    vel, radius, droplet_config, env_config,
                )
            else:
                F_d = np.zeros(3)
            F_tot = F_acoustic + F_gravity + F_d
            acc = F_tot / mass
            return np.concatenate([vel, acc])

        state0 = np.concatenate([position, velocity])
        t_end = min(t + dt, max_duration)

        # Event: stop if z < z_floor
        def z_event(t_local, state):
            return state[2] - z_floor
        z_event.terminal = True
        z_event.direction = -1

        sol = solve_ivp(
            segment_ode,
            [0.0, t_end - t],
            state0,
            method='RK45',
            rtol=rtol,
            atol=atol,
            events=z_event,
        )

        if not sol.success:
            warnings.warn(
                f"RK45 segment integration failed at t={t:.6f}s: {sol.message}",
                stacklevel=2,
            )
            break

        # Extract final state from this segment
        position = sol.y[:3, -1].copy()
        velocity = sol.y[3:6, -1].copy()

        # Update droplet state for force computation
        droplet.position = position.copy()
        droplet.velocity = velocity.copy()

        t += (sol.t[-1] if len(sol.t) > 1 else dt)

        # If the z_event fired, we've landed
        if sol.t_events and len(sol.t_events[0]) > 0:
            break

    # Compute landing position and error
    landing_position = position.copy()
    target_3d = np.array([target[0], target[1], 0])
    landing_error = float(np.linalg.norm(landing_position[:2] - target_3d[:2]))

    trajectory = Trajectory(
        points=points,
        target=target,
        droplet=droplet,
        landing_position=landing_position,
        landing_error=landing_error,
    )

    return trajectory


def simulate_free_fall(
    initial_position: np.ndarray,
    initial_velocity: Optional[np.ndarray] = None,
    droplet_diameter: float = 300e-6,
    droplet_density: float = 2385.0,
    dt: float = 1e-4,
    max_duration: float = 0.5,
    z_floor: float = 0.0,
    include_drag: bool = True,
    droplet_config: Optional[DropletConfig] = None,
    env_config: Optional[EnvironmentConfig] = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> Trajectory:
    """
    Simulate free-fall trajectory (no acoustic force) using RK45.

    Useful as a baseline comparison - where does droplet land
    without any guidance?

    Args:
        initial_position: Starting [x, y, z] in meters
        initial_velocity: Starting velocity (default: zero)
        droplet_diameter: Droplet diameter in meters (ignored if droplet_config provided)
        droplet_density: Droplet density in kg/m3 (ignored if droplet_config provided)
        dt: Output sample interval (for trajectory points)
        max_duration: Maximum simulation time
        z_floor: Stop when droplet reaches this Z
        include_drag: Whether to include air drag
        droplet_config: Optional DropletConfig for Re-based drag
        env_config: Optional EnvironmentConfig for medium properties
        rtol: RK45 relative error tolerance
        atol: RK45 absolute error tolerance

    Returns:
        Trajectory object
    """
    position = np.array(initial_position, dtype=float)
    velocity = np.array(
        initial_velocity if initial_velocity is not None else [0, 0, 0],
        dtype=float,
    )

    # Derive mass from config or legacy parameters
    if droplet_config is not None:
        radius = droplet_config.radius
        mass = droplet_config.mass
    else:
        radius = droplet_diameter / 2
        volume = (4.0 / 3.0) * np.pi * radius ** 3
        mass = droplet_density * volume

    # RK45 integration of the full free-fall trajectory
    state0 = np.concatenate([position, velocity])

    def z_event(t, state, *args):
        return state[2] - z_floor
    z_event.terminal = True
    z_event.direction = -1

    sol = solve_ivp(
        _free_fall_ode,
        [0.0, max_duration],
        state0,
        method='RK45',
        rtol=rtol,
        atol=atol,
        events=z_event,
        args=(mass, radius, include_drag, droplet_config, env_config),
        dense_output=True,
    )

    if not sol.success:
        warnings.warn(
            f"Free-fall RK45 integration failed: {sol.message}",
            stacklevel=2,
        )

    # Sample trajectory at regular dt intervals for output
    if sol.success and len(sol.t) > 1:
        t_final = sol.t[-1]
        t_samples = np.arange(0.0, t_final, dt)
        if len(t_samples) == 0:
            t_samples = np.array([0.0])
        # Also include the final time
        if t_samples[-1] < t_final:
            t_samples = np.append(t_samples, t_final)

        states = sol.sol(t_samples)  # shape (6, N)
    else:
        t_samples = np.array([0.0])
        states = state0.reshape(6, 1)

    points = []
    for i in range(len(t_samples)):
        pos_i = states[:3, i].copy()
        vel_i = states[3:6, i].copy()

        F_gravity = np.array([0.0, 0.0, -mass * GRAVITY])
        if include_drag:
            F_drag = compute_drag_force(vel_i, radius, droplet_config, env_config)
        else:
            F_drag = np.zeros(3)

        F_total = F_gravity + F_drag

        point = TrajectoryPoint(
            t=float(t_samples[i]),
            position=pos_i,
            velocity=vel_i,
            force_acoustic=np.zeros(3),
            force_total=F_total.copy(),
        )
        points.append(point)

    final_position = states[:3, -1].copy()

    trajectory = Trajectory(
        points=points,
        target=(0, 0),
        droplet=DropletState(position=final_position),
        landing_position=final_position.copy(),
        landing_error=float(np.linalg.norm(final_position[:2])),
    )

    return trajectory


def plot_trajectory(
    trajectory: Trajectory,
    freefall: Optional[Trajectory] = None,
    array: Optional[ArrayGeometry] = None,
    save_path: Optional[str] = None,
) -> 'plt.Figure':
    """
    Plot trajectory in 3D and 2D projections.

    Args:
        trajectory: Guided trajectory
        freefall: Optional free-fall comparison trajectory
        array: Optional array geometry to show
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 10))

    # 3D trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(trajectory.x * 1000, trajectory.y * 1000, trajectory.z * 1000,
             'b-', linewidth=2, label='Guided')
    if freefall:
        ax1.plot(freefall.x * 1000, freefall.y * 1000, freefall.z * 1000,
                 'gray', linestyle='--', linewidth=1, label='Free-fall')

    # Mark start, end, target
    ax1.scatter([0], [0], [trajectory.z[0]*1000], c='green', s=100, marker='o', label='Start')
    ax1.scatter([trajectory.landing_position[0]*1000],
                [trajectory.landing_position[1]*1000],
                [trajectory.landing_position[2]*1000],
                c='blue', s=100, marker='x', label='Landing')
    ax1.scatter([trajectory.target[0]*1000],
                [trajectory.target[1]*1000],
                [0], c='red', s=100, marker='*', label='Target')

    # Array outline
    if array is not None:
        r = array.positions[:, :2].max() * 1000
        theta = np.linspace(0, 2*np.pi, 50)
        z_min = array.positions[:, 2].min() * 1000
        z_max = array.positions[:, 2].max() * 1000
        ax1.plot(r*np.cos(theta), r*np.sin(theta), [z_min]*50, 'k--', alpha=0.3)
        ax1.plot(r*np.cos(theta), r*np.sin(theta), [z_max]*50, 'k--', alpha=0.3)

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Trajectory')
    ax1.legend()

    # XZ projection (side view)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(trajectory.x * 1000, trajectory.z * 1000, 'b-', linewidth=2, label='Guided')
    if freefall:
        ax2.plot(freefall.x * 1000, freefall.z * 1000, 'gray', linestyle='--', label='Free-fall')
    ax2.scatter([0], [trajectory.z[0]*1000], c='green', s=100, marker='o')
    ax2.scatter([trajectory.landing_position[0]*1000], [trajectory.landing_position[2]*1000],
                c='blue', s=100, marker='x')
    ax2.axvline(trajectory.target[0]*1000, color='red', linestyle=':', label='Target X')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Z (mm)')
    ax2.set_title('XZ Projection (Side View)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # XY projection (top view)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(trajectory.x * 1000, trajectory.y * 1000, 'b-', linewidth=2, label='Guided')
    if freefall:
        ax3.plot(freefall.x * 1000, freefall.y * 1000, 'gray', linestyle='--', label='Free-fall')
    ax3.scatter([0], [0], c='green', s=100, marker='o', label='Start')
    ax3.scatter([trajectory.landing_position[0]*1000], [trajectory.landing_position[1]*1000],
                c='blue', s=100, marker='x', label='Landing')
    ax3.scatter([trajectory.target[0]*1000], [trajectory.target[1]*1000],
                c='red', s=100, marker='*', label='Target')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_title('XY Projection (Top View)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # Force over time
    ax4 = fig.add_subplot(2, 2, 4)
    times = trajectory.times * 1000  # ms
    F_acoustic = np.array([p.force_acoustic for p in trajectory.points])
    ax4.plot(times, F_acoustic[:, 0] * 1e9, 'r-', label='Fx')
    ax4.plot(times, F_acoustic[:, 1] * 1e9, 'g-', label='Fy')
    ax4.plot(times, F_acoustic[:, 2] * 1e9, 'b-', label='Fz')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Acoustic Force (nN)')
    ax4.set_title('Acoustic Force vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved: %s", save_path)

    return fig


def compare_control_modes(
    array: ArrayGeometry,
    target: Tuple[float, float],
    initial_position: np.ndarray,
    save_path: Optional[str] = None,
) -> dict:
    """
    Compare all control modes and visualize results.

    Returns:
        Dictionary of {mode: Trajectory}
    """
    import matplotlib.pyplot as plt

    modes: List[ControlMode] = ['fixed', 'relative', 'pd', 'predictive']
    colors = {'fixed': 'red', 'relative': 'blue', 'pd': 'green', 'predictive': 'purple'}

    trajectories = {}

    logger.info("Simulating control modes...")
    for mode in modes:
        traj = simulate_trajectory(
            array, target, initial_position,
            control_mode=mode,
            kp=1.0,
            kd=0.08,  # Tuned damping
        )
        trajectories[mode] = traj
        logger.info("  %12s: landed at (%6.2f, %6.2f) mm, error = %.2f mm",
                     mode, traj.landing_position[0]*1000, traj.landing_position[1]*1000, traj.landing_error*1000)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # XY trajectories (top view)
    ax1 = axes[0, 0]
    for mode, traj in trajectories.items():
        ax1.plot(traj.x * 1000, traj.y * 1000, color=colors[mode], linewidth=2, label=mode)
        ax1.scatter([traj.landing_position[0]*1000], [traj.landing_position[1]*1000],
                    color=colors[mode], s=100, marker='x')
    ax1.scatter([0], [0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter([target[0]*1000], [target[1]*1000], c='black', s=150, marker='*', label='Target', zorder=5)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title('XY Trajectory Comparison (Top View)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # XZ trajectories (side view)
    ax2 = axes[0, 1]
    for mode, traj in trajectories.items():
        ax2.plot(traj.x * 1000, traj.z * 1000, color=colors[mode], linewidth=2, label=mode)
    ax2.axvline(target[0]*1000, color='black', linestyle=':', label='Target X')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Z (mm)')
    ax2.set_title('XZ Trajectory (Side View)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Landing error bar chart
    ax3 = axes[1, 0]
    errors = [trajectories[m].landing_error * 1000 for m in modes]
    bars = ax3.bar(modes, errors, color=[colors[m] for m in modes])
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_ylabel('Landing Error (mm)')
    ax3.set_title('Landing Error by Control Mode')
    # Add value labels
    for bar, err in zip(bars, errors):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{err:.1f}', ha='center', fontsize=10)

    # Force over time comparison
    ax4 = axes[1, 1]
    for mode, traj in trajectories.items():
        times = traj.times * 1000
        F_acoustic = np.array([p.force_acoustic for p in traj.points])
        F_mag = np.sqrt(F_acoustic[:, 0]**2 + F_acoustic[:, 1]**2)
        ax4.plot(times, F_mag * 1e9, color=colors[mode], linewidth=1.5, label=mode)
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Lateral Force |F_xy| (nN)')
    ax4.set_title('Lateral Force Magnitude vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Control Mode Comparison | Target: ({target[0]*1000:.0f}, {target[1]*1000:.0f}) mm', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved: %s", save_path)

    return trajectories


def main():
    """Demo trajectory simulation with control mode comparison."""
    import matplotlib.pyplot as plt

    logger.info("=" * 60)
    logger.info("DRIP TRAJECTORY SIMULATION - RK45 + Re-BASED DRAG")
    logger.info("=" * 60)
    logger.info("")

    # Verify RK45 convergence before running
    logger.info("Verifying RK45 convergence vs Euler at dt=1e-6...")
    converged = verify_rk45_convergence()
    if converged:
        logger.info("  PASS: RK45 converges within 1%% of fine-step Euler")
    else:
        logger.warning("  WARN: RK45 convergence check failed — see warnings")
    logger.info("")

    # Create array
    array = ArrayGeometry.cylinder(radius=0.05, height=0.4, rings=10, per_ring=12)
    logger.info("Array: %s transducers", array.n)
    logger.info("Z range: [%.0f, %.0f] mm", array.positions[:,2].min()*1000, array.positions[:,2].max()*1000)
    logger.info("")

    # Simulation parameters
    target = (10e-3, 5e-3)  # 10mm X, 5mm Y
    initial_position = np.array([0, 0, 0.18])  # Start near top of array

    logger.info("Target: (%.1f, %.1f) mm", target[0]*1000, target[1]*1000)
    logger.info("Start:  (%.1f, %.1f, %.0f) mm", initial_position[0]*1000, initial_position[1]*1000, initial_position[2]*1000)
    logger.info("")

    # Compare control modes
    trajectories = compare_control_modes(array, target, initial_position,
                                         save_path='control_comparison.png')

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY:")
    logger.info("-" * 60)
    logger.info("%-12s | %10s | %10s | %8s", "Mode", "Landing X", "Landing Y", "Error")
    logger.info("-" * 60)
    for mode, traj in trajectories.items():
        lx, ly = traj.landing_position[0]*1000, traj.landing_position[1]*1000
        err = traj.landing_error * 1000
        logger.info("%-12s | %10.2f | %10.2f | %8.2f", mode, lx, ly, err)
    logger.info("-" * 60)
    logger.info("%-12s | %10.2f | %10.2f | %8s", "Target", target[0]*1000, target[1]*1000, "0.00")
    logger.info("=" * 60)

    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
