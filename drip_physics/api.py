"""
High-level simulation API for ML researchers.

Quick start:
    from drip_physics.api import run_simulation, run_multi_droplet, sweep_targets

    result = run_simulation(target_mm=(5, 0))
    print(f"Landing error: {result.landing_error_mm:.2f} mm")

All functions use sensible defaults (styrofoam, L1 array, PID controller).
Override any parameter to customize.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .config import (
    DropletConfig,
    EnvironmentConfig,
    aluminum_droplet,
    default_environment,
    styrofoam_bead,
)
from .core import ArrayGeometry
from .pathtrack import TrackedTrajectory, simulate_multi_droplet, simulate_path_tracking
from .trace import sim_trace

logger = logging.getLogger(__name__)

# Default array and geometry matching L1 prototype
_DEFAULT_RADIUS = 0.05       # 5 cm cylinder radius
_DEFAULT_HEIGHT = 0.4        # 40 cm cylinder height
_DEFAULT_RINGS = 10
_DEFAULT_PER_RING = 12
_DEFAULT_Z_START = 0.4       # Top of L1 chamber (matches config defaults)
_DEFAULT_Z_FLOOR = 0.0       # Build plate


@dataclass(frozen=True)
class SimResult:
    """Simplified simulation result.

    Attributes:
        landing_error_mm: Distance from target at landing, in millimeters.
        flight_time_ms: Total flight duration in milliseconds.
        n_points: Number of trajectory sample points.
        trajectory: Full TrackedTrajectory for advanced use.
        trace_summary: Event counts from sim_trace (ring handoffs, clamps, etc.).
    """

    landing_error_mm: float
    flight_time_ms: float
    n_points: int
    trajectory: TrackedTrajectory
    trace_summary: Dict[str, int]


def _make_array() -> ArrayGeometry:
    """Build the default L1 cylindrical array."""
    return ArrayGeometry.cylinder(
        radius=_DEFAULT_RADIUS,
        height=_DEFAULT_HEIGHT,
        rings=_DEFAULT_RINGS,
        per_ring=_DEFAULT_PER_RING,
    )


def _make_droplet(
    material: str,
    diameter_mm: Optional[float] = None,
    position: Optional[np.ndarray] = None,
) -> DropletConfig:
    """Build a DropletConfig from material name and optional overrides."""
    pos = position if position is not None else np.array([0.0, 0.0, _DEFAULT_Z_START])

    if material == "aluminum":
        factory = aluminum_droplet
    elif material == "styrofoam":
        factory = styrofoam_bead
    else:
        raise ValueError(
            f"Unknown material '{material}'. Choose 'styrofoam' or 'aluminum'."
        )

    kwargs: dict = {"position": pos}
    if diameter_mm is not None:
        kwargs["diameter"] = diameter_mm * 1e-3

    return factory(**kwargs)


def _to_sim_result(traj: TrackedTrajectory) -> SimResult:
    """Convert a TrackedTrajectory into a SimResult."""
    error_mm = (traj.landing_error * 1000) if traj.landing_error is not None else float("nan")
    return SimResult(
        landing_error_mm=error_mm,
        flight_time_ms=traj.duration * 1000,
        n_points=len(traj.points),
        trajectory=traj,
        trace_summary=sim_trace.summary(),
    )


def run_simulation(
    target_mm: Tuple[float, float] = (5.0, 0.0),
    material: str = "styrofoam",
    controller: Union[str, object] = "pid",
    kp: float = 3.0,
    kd: float = 1.0,
    ki: float = 0.0,
    diameter_mm: Optional[float] = None,
    max_duration_s: float = 0.5,
    env: Optional[EnvironmentConfig] = None,
    droplet: Optional[DropletConfig] = None,
) -> SimResult:
    """Run a single droplet simulation with sensible defaults.

    Args:
        target_mm: Landing target (x, y) in millimeters.
        material: 'styrofoam' or 'aluminum'.
        controller: Controller mode name ('legacy', 'p_only', 'pi', 'pid')
            or a CompiledController instance.
        kp: Proportional gain.
        kd: Derivative gain.
        ki: Integral gain (only used by 'pi' and 'pid').
        diameter_mm: Droplet diameter in mm (None uses material default).
        max_duration_s: Maximum simulation time in seconds.
        env: Custom EnvironmentConfig (overrides defaults).
        droplet: Custom DropletConfig (overrides material/diameter).

    Returns:
        SimResult with landing_error_mm, flight_time_ms, and full trajectory.
    """
    if env is None:
        env = default_environment()

    if droplet is None:
        droplet = _make_droplet(material, diameter_mm)

    array = _make_array()
    target_m = (target_mm[0] * 1e-3, target_mm[1] * 1e-3)

    sim_trace.clear()

    traj = simulate_path_tracking(
        array=array,
        target=target_m,
        initial_position=droplet.initial_position.copy(),
        env_config=env,
        droplet_config=droplet,
        control_mode=controller,
        kp=kp,
        kd=kd,
        ki=ki,
        max_duration=max_duration_s,
        z_floor=_DEFAULT_Z_FLOOR,
    )

    return _to_sim_result(traj)


def run_multi_droplet(
    targets_mm: List[Tuple[float, float]],
    entry_delays_ms: Optional[List[float]] = None,
    material: str = "styrofoam",
    diameter_mm: Optional[float] = None,
    max_duration_s: float = 2.0,
    force_smoothing: float = 0.3,
) -> List[SimResult]:
    """Run a multi-droplet simulation with staggered entry.

    Uses pure feedforward from the joint inverse solver (no per-droplet
    controller). For controlled multi-droplet runs, use the lower-level
    ``simulate_multi_droplet`` directly.

    Args:
        targets_mm: List of (x, y) targets in mm, one per droplet.
        entry_delays_ms: Delay before each droplet enters (ms).
            Default: 50 ms spacing between droplets.
        material: 'styrofoam' or 'aluminum'.
        diameter_mm: Droplet diameter in mm (None uses material default).
        max_duration_s: Maximum simulation time in seconds.
        force_smoothing: EMA alpha for acoustic force (0-1). Default 0.3.

    Returns:
        List of SimResult, one per droplet.
    """
    env = default_environment()
    n = len(targets_mm)
    if n == 0:
        raise ValueError("targets_mm must be non-empty")

    array = _make_array()
    start_pos = np.array([0.0, 0.0, _DEFAULT_Z_START])

    configs = [_make_droplet(material, diameter_mm, position=start_pos) for _ in range(n)]
    targets_m = [(t[0] * 1e-3, t[1] * 1e-3) for t in targets_mm]

    if entry_delays_ms is None:
        entry_delays_ms = [i * 50.0 for i in range(n)]
    delays_s = [d * 1e-3 for d in entry_delays_ms]

    sim_trace.clear()

    trajectories = simulate_multi_droplet(
        array=array,
        droplet_configs=configs,
        targets=targets_m,
        env_config=env,
        dt=1e-4,
        max_duration=max_duration_s,
        z_floor=_DEFAULT_Z_FLOOR,
        entry_delays=delays_s,
        force_smoothing=force_smoothing,
    )

    return [_to_sim_result(traj) for traj in trajectories]


def sweep_targets(
    targets_mm: List[Tuple[float, float]],
    **kwargs: object,
) -> List[SimResult]:
    """Run ``run_simulation`` for each target in a list (parameter sweep).

    Args:
        targets_mm: List of (x, y) targets to sweep over.
        **kwargs: Forwarded to ``run_simulation()``.

    Returns:
        List of SimResult, one per target.
    """
    return [run_simulation(target_mm=t, **kwargs) for t in targets_mm]
