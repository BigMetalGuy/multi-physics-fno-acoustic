"""
Force Plugin System for DRIP Physics.

This module defines the interface for forces in the simulation.
Forces can be toggled on/off, have editable parameters, and link to derivations.

The Physics Configuration Page uses this interface to:
1. Display available forces with their equations
2. Allow users to edit parameters
3. Generate/update the simulation code

Architecture:
    ForcePlugin (abstract base)
        -> GravityForce
        -> DragForce (regime-selected via DropletConfig)
        -> AcousticForce (Gor'kov)
        -> BuoyancyForce
        -> CustomForce (user-defined)

Each force provides:
    - name, description, equation (LaTeX)
    - parameters with units and bounds
    - compute(state, env_config, droplet_config) -> force vector
    - link to derivation/proof document

Config objects (EnvironmentConfig, DropletConfig) are REQUIRED arguments
at all public API boundaries. No hardcoded physical constants.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import numpy as np
from numpy.typing import NDArray

from .config import (
    EnvironmentConfig,
    DropletConfig,
    GRAVITY,
)


# =============================================================================
# Parameter System
# =============================================================================

class ParameterType(Enum):
    """Types of editable parameters."""
    SCALAR = "scalar"       # Single number with units
    VECTOR = "vector"       # 3D vector
    CURVE = "curve"         # Interactive curve (list of control points)
    CHOICE = "choice"       # Dropdown selection
    BOOLEAN = "boolean"     # Toggle


@dataclass
class Parameter:
    """
    An editable parameter for a force.

    Attributes:
        name: Parameter identifier (e.g., "gravity_magnitude")
        display_name: Human-readable name (e.g., "Gravity (g)")
        param_type: Type of parameter
        default: Default value
        units: Physical units (e.g., "m/s^2")
        bounds: (min, max) for scalar/vector, None for unbounded
        choices: List of options for CHOICE type
        description: Help text
    """
    name: str
    display_name: str
    param_type: ParameterType
    default: Any
    units: str = ""
    bounds: Optional[tuple] = None
    choices: Optional[List[str]] = None
    description: str = ""


@dataclass
class CurvePoint:
    """A control point for interactive curve editing."""
    x: float
    y: float
    handle_left: Optional[tuple] = None   # Bezier handle
    handle_right: Optional[tuple] = None


@dataclass
class Curve:
    """Interactive curve defined by control points."""
    points: List[CurvePoint]
    x_label: str = "x"
    y_label: str = "y"
    x_units: str = ""
    y_units: str = ""

    def evaluate(self, x: float) -> float:
        """Interpolate curve value at x."""
        # Simple linear interpolation for now
        # TODO: Implement Bezier/spline interpolation
        if not self.points:
            return 0.0

        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return np.interp(x, xs, ys)


# =============================================================================
# Force Plugin Interface
# =============================================================================

@dataclass
class ForceResult:
    """Result of a force computation."""
    force: NDArray[np.float64]  # Force vector [Fx, Fy, Fz] in Newtons
    metadata: Dict[str, Any] = field(default_factory=dict)  # Debug info


class ForcePlugin(ABC):
    """
    Abstract base class for all forces in the simulation.

    To create a new force:
        1. Subclass ForcePlugin
        2. Implement required properties and compute()
        3. Register with ForceRegistry

    The Physics Page UI will automatically:
        - Display the force with its equation
        - Show editable parameters
        - Link to the derivation document

    CRITICAL: compute() takes EnvironmentConfig and DropletConfig as
    REQUIRED arguments. No hardcoded physical constants.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this force."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name."""
        pass

    @property
    @abstractmethod
    def equation_latex(self) -> str:
        """LaTeX representation of the force equation."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> List[Parameter]:
        """List of editable parameters."""
        pass

    @property
    def derivation_link(self) -> Optional[str]:
        """Path to derivation/proof document (relative to /proofs/)."""
        return None

    @property
    def description(self) -> str:
        """Detailed description of the force."""
        return ""

    @property
    def enabled_by_default(self) -> bool:
        """Whether this force is enabled by default."""
        return True

    @abstractmethod
    def compute(
        self,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        env_config: EnvironmentConfig,
        droplet_config: DropletConfig,
        params: Dict[str, Any],
        **kwargs,
    ) -> ForceResult:
        """
        Compute the force at given state.

        Args:
            position: [x, y, z] in meters
            velocity: [vx, vy, vz] in m/s
            env_config: Environment configuration (REQUIRED)
            droplet_config: Droplet configuration (REQUIRED)
            params: Current UI parameter overrides
            **kwargs: Additional context (acoustic field, etc.)

        Returns:
            ForceResult with force vector
        """
        pass

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """
        Validate parameter values.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        for param in self.parameters:
            if param.name not in params:
                continue
            value = params[param.name]
            if param.bounds and param.param_type == ParameterType.SCALAR:
                lo, hi = param.bounds
                if value < lo or value > hi:
                    errors.append(f"{param.display_name} must be between {lo} and {hi}")
        return errors


# =============================================================================
# Built-in Forces
# =============================================================================

class GravityForce(ForcePlugin):
    """Gravitational force: F = m * g. Uses GRAVITY from config."""

    @property
    def name(self) -> str:
        return "gravity"

    @property
    def display_name(self) -> str:
        return "Gravity"

    @property
    def equation_latex(self) -> str:
        return r"\vec{F} = m \cdot \vec{g}"

    @property
    def parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="g_direction",
                display_name="Direction",
                param_type=ParameterType.VECTOR,
                default=[0, 0, -1],
                description="Unit vector for gravity direction"
            ),
        ]

    @property
    def derivation_link(self) -> str:
        return "derivation-03-gravity.md"

    def compute(
        self,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        env_config: EnvironmentConfig,
        droplet_config: DropletConfig,
        params: Dict[str, Any],
        **kwargs,
    ) -> ForceResult:
        g_dir = np.array(params.get("g_direction", [0, 0, -1]), dtype=float)
        norm = np.linalg.norm(g_dir)
        if norm > 0:
            g_dir = g_dir / norm

        force = droplet_config.mass * GRAVITY * g_dir
        return ForceResult(force=force, metadata={"g": GRAVITY})


class DragForce(ForcePlugin):
    """
    Drag force with regime-selected law via DropletConfig.

    Uses DropletConfig.drag_coefficient() which selects:
      - Re < 1: Stokes
      - Re < 1000: Schiller-Naumann
      - Re >= 1000: Newton

    No hardcoded viscosity or density.
    """

    @property
    def name(self) -> str:
        return "drag"

    @property
    def display_name(self) -> str:
        return "Drag (Regime-Selected)"

    @property
    def equation_latex(self) -> str:
        return r"\vec{F} = -\frac{1}{2} C_d \rho A |\vec{v}| \vec{v}"

    @property
    def parameters(self) -> List[Parameter]:
        return []  # All params come from config objects

    @property
    def derivation_link(self) -> str:
        return "derivation-05-drag.md"

    @property
    def description(self) -> str:
        return (
            "Drag force with regime selection from Reynolds number. "
            "Stokes (Re<1), Schiller-Naumann (Re<1000), Newton (Re>=1000). "
            "All properties from EnvironmentConfig and DropletConfig."
        )

    def compute(
        self,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        env_config: EnvironmentConfig,
        droplet_config: DropletConfig,
        params: Dict[str, Any],
        **kwargs,
    ) -> ForceResult:
        speed = float(np.linalg.norm(velocity))
        if speed < 1e-12:
            return ForceResult(
                force=np.zeros(3),
                metadata={"reynolds": 0.0, "drag_regime": "stationary", "Cd": 0.0},
            )

        medium = env_config.medium
        re = droplet_config.reynolds_number(speed, medium)
        cd = droplet_config.drag_coefficient(speed, medium)
        regime = droplet_config.drag_regime(speed, medium)

        # F_drag = -0.5 * Cd * rho * A * |v| * v
        area = np.pi * droplet_config.radius ** 2
        force = -0.5 * cd * medium.density * area * speed * velocity

        return ForceResult(
            force=force,
            metadata={"reynolds": re, "drag_regime": regime, "Cd": cd},
        )


class AcousticForce(ForcePlugin):
    """Gor'kov acoustic radiation force: F = -grad(U)"""

    @property
    def name(self) -> str:
        return "acoustic"

    @property
    def display_name(self) -> str:
        return "Acoustic Radiation"

    @property
    def equation_latex(self) -> str:
        return r"\vec{F} = -\nabla U = -\nabla\left(\frac{4\pi}{3}a^3 \Phi \frac{\langle p^2 \rangle}{\rho c^2}\right)"

    @property
    def parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="gradient_step",
                display_name="Gradient step size",
                param_type=ParameterType.SCALAR,
                default=0.5e-3,
                units="m",
                bounds=(1e-4, 5e-3),
                description="Step size for numerical gradient computation"
            ),
        ]

    @property
    def derivation_link(self) -> str:
        return "derivation-01-gorkov.md"

    def compute(
        self,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        env_config: EnvironmentConfig,
        droplet_config: DropletConfig,
        params: Dict[str, Any],
        **kwargs,
    ) -> ForceResult:
        """Acoustic radiation force (passthrough from inverse solver).

        Unlike gravity and drag, acoustic force depends on the inverse solver
        output (phase array), which is computed upstream in the control loop.
        The force is pre-computed by compute_force() in force.py and passed
        here via kwargs['acoustic_force'] for force composition.

        This is NOT a stub — it's architecturally correct: the inverse solver
        + force.py compute the acoustic force; this plugin integrates it into
        the ForceCompositor's total force sum.
        """
        if 'acoustic_force' not in kwargs:
            warnings.warn(
                "AcousticForce.compute() called without 'acoustic_force' in kwargs — "
                "returning zero force. The caller must pre-compute acoustic force via "
                "compute_force() and pass it as acoustic_force=...",
                stacklevel=2,
            )
        acoustic_force = kwargs.get("acoustic_force", np.zeros(3))

        return ForceResult(
            force=np.array(acoustic_force),
            metadata={"source": "inverse_solver"}
        )


class BuoyancyForce(ForcePlugin):
    """Buoyancy force: F = rho_medium * V * g. All values from config."""

    @property
    def name(self) -> str:
        return "buoyancy"

    @property
    def display_name(self) -> str:
        return "Buoyancy"

    @property
    def equation_latex(self) -> str:
        return r"\vec{F} = \rho_{medium} V \vec{g}"

    @property
    def parameters(self) -> List[Parameter]:
        return []  # All params from config

    @property
    def enabled_by_default(self) -> bool:
        return False  # Usually negligible for dense droplets

    def compute(
        self,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        env_config: EnvironmentConfig,
        droplet_config: DropletConfig,
        params: Dict[str, Any],
        **kwargs,
    ) -> ForceResult:
        rho_medium = env_config.medium.density
        volume = droplet_config.volume
        # Buoyancy acts opposite to gravity (upward)
        force = rho_medium * volume * GRAVITY * np.array([0.0, 0.0, 1.0])

        return ForceResult(force=force, metadata={"rho_medium": rho_medium})


class AcousticStreamingForce(ForcePlugin):
    """Steady drag from acoustic streaming background flow.  [ASMP-062]

    Acoustic streaming (Eckart streaming) creates a bulk airflow driven by the
    acoustic intensity gradient.  For a focused cylindrical array the streaming
    pattern is complex, but as a first-order approximation we model it as a
    uniform upward flow (opposing gravity) with magnitude equal to the
    configured ``EnvironmentConfig.acoustic_streaming_velocity``.

    The force applied to the droplet is the Stokes drag from this background
    flow::

        F_streaming = 6 * pi * mu * a * v_streaming   (upward, +z)

    This is separate from the Gor'kov radiation force (AcousticForce) and from
    the droplet-motion drag (DragForce).  It is opt-in (``enabled_by_default =
    False``) because the streaming velocity estimate carries large uncertainty
    ([ASMP-044]).

    The streaming velocity is read from ``EnvironmentConfig.get_streaming_velocity()``.
    If ``acoustic_streaming_velocity`` is ``None`` on the config, the method
    warns and returns 0.0, so this force gracefully becomes zero.
    """

    @property
    def name(self) -> str:
        return "acoustic_streaming"

    @property
    def display_name(self) -> str:
        return "Acoustic Streaming Drag"

    @property
    def equation_latex(self) -> str:
        return r"\vec{F}_{stream} = 6\pi\mu a\, v_{stream}\,\hat{z}"

    @property
    def parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="direction",
                display_name="Streaming direction",
                param_type=ParameterType.VECTOR,
                default=[0.0, 0.0, 1.0],
                description=(
                    "Unit vector for streaming-induced drag.  Default +z (upward, "
                    "opposing gravity) as a first approximation of focused inward flow."
                ),
            ),
        ]

    @property
    def enabled_by_default(self) -> bool:
        return False  # opt-in: large uncertainty on streaming estimate

    @property
    def derivation_link(self) -> str:
        return "derivation-10-thermal.ipynb"  # streaming velocity discussed in D10

    @property
    def description(self) -> str:
        return (
            "Viscous drag on the droplet from the Eckart acoustic streaming "
            "background flow.  Modeled as Stokes drag (F = 6*pi*mu*a*v_stream) "
            "in a configurable direction (default upward).  Streaming velocity "
            "is read from EnvironmentConfig.acoustic_streaming_velocity."
        )

    def compute(
        self,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        env_config: EnvironmentConfig,
        droplet_config: DropletConfig,
        params: Dict[str, Any],
        **kwargs,
    ) -> ForceResult:
        v_stream = env_config.get_streaming_velocity()  # m/s, warns if None
        if v_stream <= 0.0:
            return ForceResult(
                force=np.zeros(3),
                metadata={"v_streaming": 0.0, "regime": "inactive"},
            )

        direction = np.array(
            params.get("direction", [0.0, 0.0, 1.0]), dtype=float
        )
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            return ForceResult(
                force=np.zeros(3),
                metadata={"v_streaming": v_stream, "regime": "zero_direction"},
            )
        direction = direction / norm

        mu = env_config.medium.dynamic_viscosity
        radius = droplet_config.radius

        # Stokes drag from the streaming flow
        f_mag = 6.0 * np.pi * mu * radius * v_stream
        force = f_mag * direction

        return ForceResult(
            force=force,
            metadata={
                "v_streaming": v_stream,
                "F_magnitude_nN": f_mag * 1e9,
                "regime": "stokes",
            },
        )


# =============================================================================
# Backward-Compatibility Alias
# =============================================================================

# Old name had a typo ("StokessDragForce"). Keep alias for any external code
# that referenced it, but the canonical class is now DragForce.
StokessDragForce = DragForce


# =============================================================================
# Force Registry
# =============================================================================

class ForceRegistry:
    """
    Registry of available forces.

    The Physics Page queries this to show available forces.
    Forces can be registered/unregistered dynamically.
    """

    # Mutable class-level dict — intentional for classmethod-only singleton pattern.
    # Use reset() to restore clean state in tests.
    _forces: Dict[str, ForcePlugin] = {}

    @classmethod
    def register(cls, force: ForcePlugin):
        """Register a force plugin. Silently skips if already registered."""
        if force.name in cls._forces:
            return  # already registered, skip silently
        cls._forces[force.name] = force

    @classmethod
    def unregister(cls, name: str):
        """Unregister a force by name."""
        cls._forces.pop(name, None)

    @classmethod
    def get(cls, name: str) -> Optional[ForcePlugin]:
        """Get a force by name."""
        return cls._forces.get(name)

    @classmethod
    def all(cls) -> List[ForcePlugin]:
        """Get all registered forces."""
        return list(cls._forces.values())

    @classmethod
    def enabled_by_default(cls) -> List[ForcePlugin]:
        """Get forces that are enabled by default."""
        return [f for f in cls._forces.values() if f.enabled_by_default]

    @classmethod
    def reset(cls) -> None:
        """Clear all registered forces and re-register built-ins.

        Useful for test isolation — ensures a clean registry state.
        """
        cls._forces = {}
        cls._register_builtins()

    @classmethod
    def _register_builtins(cls) -> None:
        """Register the standard built-in forces."""
        cls.register(GravityForce())
        cls.register(DragForce())
        cls.register(AcousticForce())
        cls.register(BuoyancyForce())
        cls.register(AcousticStreamingForce())


# Register built-in forces
ForceRegistry.register(GravityForce())
ForceRegistry.register(DragForce())
ForceRegistry.register(AcousticForce())
ForceRegistry.register(BuoyancyForce())
ForceRegistry.register(AcousticStreamingForce())

class _LegacyStokesDrag(DragForce):
    """Backward-compatible registration under the old 'drag_stokes' name."""
    @property
    def name(self) -> str:
        return "drag_stokes"

# Also register under legacy name for backward compatibility
ForceRegistry.register(_LegacyStokesDrag())


# =============================================================================
# Force Compositor
# =============================================================================

class ForceCompositor:
    """
    Combines multiple forces into a total force.

    This is what the simulation loop uses to compute net force.
    The Physics Page controls which forces are active.

    CRITICAL: compute_total() requires EnvironmentConfig and DropletConfig.
    """

    def __init__(self):
        self.active_forces: Dict[str, ForcePlugin] = {}
        self.params: Dict[str, Dict[str, Any]] = {}

    def enable(self, force_name: str, params: Optional[Dict[str, Any]] = None):
        """Enable a force with optional custom parameters."""
        force = ForceRegistry.get(force_name)
        if force:
            self.active_forces[force_name] = force
            if params:
                self.params[force_name] = params
            else:
                # Use defaults
                self.params[force_name] = {
                    p.name: p.default for p in force.parameters
                }

    def disable(self, force_name: str):
        """Disable a force."""
        self.active_forces.pop(force_name, None)
        self.params.pop(force_name, None)

    def set_param(self, force_name: str, param_name: str, value: Any):
        """Update a parameter value."""
        if force_name in self.params:
            self.params[force_name][param_name] = value

    def compute_total(
        self,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        env_config: EnvironmentConfig,
        droplet_config: DropletConfig,
        **kwargs,
    ) -> ForceResult:
        """
        Compute total force from all active forces.

        Args:
            position: [x, y, z] in meters
            velocity: [vx, vy, vz] in m/s
            env_config: Environment configuration (REQUIRED)
            droplet_config: Droplet configuration (REQUIRED)
            **kwargs: Passed through to each force plugin (e.g. acoustic_force)

        Returns:
            ForceResult with combined force vector and per-force metadata
        """
        total = np.zeros(3)
        metadata = {}

        for name, force in self.active_forces.items():
            params = self.params.get(name, {})
            result = force.compute(
                position, velocity,
                env_config, droplet_config,
                params, **kwargs,
            )
            total += result.force
            metadata[name] = {
                "force": result.force.tolist(),
                "metadata": result.metadata,
            }

        return ForceResult(force=total, metadata=metadata)

    def to_dict(self) -> Dict:
        """Serialize configuration for storage/UI."""
        return {
            "active_forces": list(self.active_forces.keys()),
            "params": self.params
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ForceCompositor":
        """Load configuration from dict."""
        compositor = cls()
        for force_name in data.get("active_forces", []):
            params = data.get("params", {}).get(force_name)
            compositor.enable(force_name, params)
        return compositor


# =============================================================================
# Code Generator (for Physics Page -> Python)
# =============================================================================

def generate_force_code(compositor: ForceCompositor) -> str:
    """
    Generate Python code for the current force configuration.

    This is called when user clicks "Apply" in the Physics Page.
    The generated code can be saved to a file or executed directly.
    """
    lines = [
        "# Auto-generated force computation",
        "# Generated by Physics Configuration Page",
        "",
        "import numpy as np",
        "from drip_physics.config import EnvironmentConfig, DropletConfig, GRAVITY",
        "",
        "def compute_total_force(position, velocity, env_config: EnvironmentConfig, "
        "droplet_config: DropletConfig, **kwargs):",
        "    '''Compute total force on droplet. Config objects are REQUIRED.'''",
        "    total = np.zeros(3)",
        "",
    ]

    for name, force in compositor.active_forces.items():
        params = compositor.params.get(name, {})
        lines.append(f"    # {force.display_name}: {force.equation_latex}")

        if name == "gravity":
            g_dir = params.get("g_direction", [0, 0, -1])
            lines.append(f"    g_dir = np.array({g_dir}) / np.linalg.norm({g_dir})")
            lines.append(f"    total += droplet_config.mass * GRAVITY * g_dir")

        elif name in ("drag", "drag_stokes"):
            lines.append("    speed = np.linalg.norm(velocity)")
            lines.append("    if speed > 1e-12:")
            lines.append("        cd = droplet_config.drag_coefficient(speed, env_config.medium)")
            lines.append("        area = np.pi * droplet_config.radius ** 2")
            lines.append("        total += -0.5 * cd * env_config.medium.density * area * speed * velocity")

        elif name == "acoustic":
            lines.append("    total += kwargs.get('acoustic_force', np.zeros(3))")

        elif name == "buoyancy":
            lines.append("    total += env_config.medium.density * droplet_config.volume * GRAVITY * np.array([0, 0, 1])")

        elif name == "acoustic_streaming":
            direction = params.get("direction", [0, 0, 1])
            lines.append(f"    stream_dir = np.array({direction}, dtype=float)")
            lines.append("    stream_dir = stream_dir / (np.linalg.norm(stream_dir) + 1e-30)")
            lines.append("    v_stream = env_config.get_streaming_velocity()")
            lines.append("    if v_stream > 0:")
            lines.append("        mu = env_config.medium.dynamic_viscosity")
            lines.append("        total += 6 * np.pi * mu * droplet_config.radius * v_stream * stream_dir")

        lines.append("")

    lines.append("    return total")

    return "\n".join(lines)
