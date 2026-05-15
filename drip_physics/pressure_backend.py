"""
Pressure backend abstraction for pluggable pressure solvers.

Provides a Protocol-based backend system so the simulation can swap between
analytical (current), PDE-based (jwave, fenics, kwave), or comparison
backends without changing caller code.

Usage:
    from drip_physics.config import PressureConfig
    from drip_physics.pressure_backend import create_backend

    config = PressureConfig(backend="analytical")
    backend = create_backend(config)
    p = backend.compute_pressure(array, phases, point)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from . import pressure as _pressure_module
from .config import PressureConfig

if TYPE_CHECKING:
    from .boundary import BoundaryReflectionModel
    from .config import EnvironmentConfig
    from .core import ArrayGeometry, PhaseArray, RingArray

logger = logging.getLogger(__name__)

AVAILABLE_BACKENDS = (
    "analytical", "jwave", "femcoupled", "fenics", "kwave", "compare",
)


# =============================================================================
# Protocol
# =============================================================================


@runtime_checkable
class PressureBackend(Protocol):
    """Protocol for pluggable pressure computation backends."""

    def compute_pressure(
        self,
        array: ArrayGeometry,
        phases: PhaseArray,
        point: NDArray[np.float64],
        *,
        ring_array: Optional[RingArray] = None,
        use_directivity: bool = True,
        use_physical_units: bool = True,
        boundary_model: Optional[BoundaryReflectionModel] = None,
        env: Optional[EnvironmentConfig] = None,
    ) -> complex: ...

    def compute_pressure_field(
        self,
        array: ArrayGeometry,
        phases: PhaseArray,
        points: NDArray[np.float64],
        *,
        ring_array: Optional[RingArray] = None,
        use_directivity: bool = True,
        use_physical_units: bool = True,
        env: Optional[EnvironmentConfig] = None,
    ) -> NDArray[np.complexfloating]: ...

    def invalidate_cache(self) -> None: ...


# =============================================================================
# AnalyticalBackend
# =============================================================================


class AnalyticalBackend:
    """Delegates to the existing analytical pressure solver in pressure.py.

    Stateless wrapper — no caching, no mutable state.
    """

    def compute_pressure(
        self,
        array: "ArrayGeometry",
        phases: "PhaseArray",
        point: NDArray[np.float64],
        *,
        ring_array: Optional["RingArray"] = None,
        use_directivity: bool = True,
        use_physical_units: bool = True,
        boundary_model: Optional["BoundaryReflectionModel"] = None,
        env: Optional["EnvironmentConfig"] = None,
    ) -> complex:
        return _pressure_module.compute_pressure(
            array,
            phases,
            point,
            ring_array=ring_array,
            use_directivity=use_directivity,
            use_physical_units=use_physical_units,
            boundary_model=boundary_model,
            env=env,
        )

    def compute_pressure_field(
        self,
        array: "ArrayGeometry",
        phases: "PhaseArray",
        points: NDArray[np.float64],
        *,
        ring_array: Optional["RingArray"] = None,
        use_directivity: bool = True,
        use_physical_units: bool = True,
        env: Optional["EnvironmentConfig"] = None,
    ) -> NDArray[np.complexfloating]:
        return _pressure_module.compute_pressure_field(
            array,
            phases,
            points,
            ring_array=ring_array,
            use_directivity=use_directivity,
            use_physical_units=use_physical_units,
            env=env,
        )

    def invalidate_cache(self) -> None:
        """No-op — analytical backend is stateless."""


# =============================================================================
# ComparisonBackend
# =============================================================================


@dataclass
class _ComparisonEntry:
    """Single comparison log entry (internal)."""

    relative_error: float
    abs_error: float
    position: Optional[tuple] = None


class ComparisonBackend:
    """Runs two backends and logs discrepancies.

    Always returns the primary backend's result. Discrepancies exceeding
    ``config.comparison_log_threshold`` are recorded for later analysis.
    """

    def __init__(
        self,
        primary: PressureBackend,
        secondary: PressureBackend,
        config: PressureConfig,
    ) -> None:
        self._primary = primary
        self._secondary = secondary
        self._config = config
        self._log: List[dict] = []

    def compute_pressure(
        self,
        array: "ArrayGeometry",
        phases: "PhaseArray",
        point: NDArray[np.float64],
        *,
        ring_array: Optional["RingArray"] = None,
        use_directivity: bool = True,
        use_physical_units: bool = True,
        boundary_model: Optional["BoundaryReflectionModel"] = None,
        env: Optional["EnvironmentConfig"] = None,
    ) -> complex:
        kwargs = dict(
            ring_array=ring_array,
            use_directivity=use_directivity,
            use_physical_units=use_physical_units,
            boundary_model=boundary_model,
            env=env,
        )
        result_primary = self._primary.compute_pressure(array, phases, point, **kwargs)
        result_secondary = self._secondary.compute_pressure(
            array, phases, point, **kwargs
        )
        self._log_comparison(
            result_primary,
            result_secondary,
            position=tuple(point) if point is not None else None,
        )
        return result_primary

    def compute_pressure_field(
        self,
        array: "ArrayGeometry",
        phases: "PhaseArray",
        points: NDArray[np.float64],
        *,
        ring_array: Optional["RingArray"] = None,
        use_directivity: bool = True,
        use_physical_units: bool = True,
        env: Optional["EnvironmentConfig"] = None,
    ) -> NDArray[np.complexfloating]:
        kwargs = dict(
            ring_array=ring_array,
            use_directivity=use_directivity,
            use_physical_units=use_physical_units,
            env=env,
        )
        result_primary = self._primary.compute_pressure_field(
            array, phases, points, **kwargs
        )
        result_secondary = self._secondary.compute_pressure_field(
            array, phases, points, **kwargs
        )
        self._log_comparison(result_primary, result_secondary, position=None)
        return result_primary

    def invalidate_cache(self) -> None:
        self._primary.invalidate_cache()
        self._secondary.invalidate_cache()

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------

    def _log_comparison(
        self,
        primary_result: complex | NDArray,
        secondary_result: complex | NDArray,
        position: Optional[tuple] = None,
    ) -> None:
        """Record discrepancy if it exceeds the configured threshold."""
        mag_primary = np.abs(primary_result)
        mag_secondary = np.abs(secondary_result)
        abs_err = float(np.max(np.abs(mag_primary - mag_secondary)))

        # Avoid division by zero — use max magnitude as denominator
        denom = float(np.maximum(np.max(mag_primary), 1e-30))
        rel_err = abs_err / denom

        if rel_err > self._config.comparison_log_threshold:
            entry = {
                "relative_error": rel_err,
                "abs_error": abs_err,
                "position": position,
            }
            self._log.append(entry)
            logger.info(
                "Pressure backend discrepancy: rel=%.4f abs=%.4e pos=%s",
                rel_err,
                abs_err,
                position,
            )

    def get_comparison_log(self) -> List[dict]:
        """Return accumulated comparison entries."""
        return list(self._log)

    def clear_log(self) -> None:
        """Reset the comparison log."""
        self._log = []


# =============================================================================
# Factory
# =============================================================================


def create_backend(
    config: PressureConfig,
    env: Optional["EnvironmentConfig"] = None,
) -> PressureBackend:
    """Create a pressure backend from configuration.

    Args:
        config: PressureConfig specifying which backend to use.
        env: Optional EnvironmentConfig (reserved for PDE backends that
             need medium properties at construction time).

    Returns:
        A PressureBackend instance.

    Raises:
        ValueError: If backend name is unknown.
        ImportError: If a PDE backend's dependencies are not installed.
    """
    backend_name = config.backend.lower()

    if backend_name == "analytical":
        return AnalyticalBackend()

    if backend_name == "jwave":
        try:
            from .backends.jwave_backend import JWaveBackend  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "jwave backend requires the 'jwave' package. "
                "Install it with: pip install jwave"
            ) from exc
        return JWaveBackend(config=config, env=env)

    if backend_name == "femcoupled":
        # Phase 5: FEM-coupled-physics drop-in.  Mirrors j-Wave signature;
        # selectable via PressureConfig(backend="femcoupled", ...).
        try:
            from .backends.femcoupled_backend import (  # type: ignore[import-not-found]
                FemcoupledBackend,
            )
        except ImportError as exc:
            raise ImportError(
                "femcoupled backend requires 'jax_fem' + 'meshio' + "
                "'gmsh' (see simulations/ml_inverse/_jaxfem_discovery)."
            ) from exc
        return FemcoupledBackend(config=config, env=env)

    if backend_name == "fenics":
        try:
            from .backends.fenics_backend import FenicsBackend  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "fenics backend requires the 'fenics' package. "
                "Install it with: pip install fenicsproject"
            ) from exc
        return FenicsBackend(config=config, env=env)

    if backend_name == "kwave":
        try:
            from .backends.kwave_backend import KWaveBackend  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "kwave backend requires the 'kwave-python' package. "
                "Install it with: pip install kwave-python"
            ) from exc
        return KWaveBackend(config=config, env=env)

    if backend_name == "compare":
        primary = create_backend(
            PressureConfig(backend=config.comparison_primary),
            env=env,
        )
        # Default secondary: if primary is analytical, secondary is jwave
        secondary_name = (
            "jwave" if config.comparison_primary == "analytical" else "analytical"
        )
        secondary = create_backend(
            PressureConfig(backend=secondary_name),
            env=env,
        )
        return ComparisonBackend(primary=primary, secondary=secondary, config=config)

    raise ValueError(
        f"Unknown pressure backend: {backend_name!r}. "
        f"Available backends: {', '.join(AVAILABLE_BACKENDS)}"
    )
