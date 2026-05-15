"""Pressure computation backends.

Available backends are discovered at import time based on installed packages.
Each backend implements the PressureBackend protocol from pressure_backend.py.
"""

AVAILABLE_BACKENDS: dict[str, str] = {
    "analytical": "drip_physics.pressure_backend.AnalyticalBackend",
}

try:
    from .jwave_backend import JWaveBackend

    AVAILABLE_BACKENDS["jwave"] = (
        "drip_physics.backends.jwave_backend.JWaveBackend"
    )
except ImportError:
    pass

# Future: fenics_backend, kwave_backend follow same pattern
