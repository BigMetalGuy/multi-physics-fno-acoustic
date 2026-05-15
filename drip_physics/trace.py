"""Simulation event tracing for physics decision audit trail.

Records simulation decisions (mode changes, ring handoffs, force clamps,
droplet entries/exits) for post-run debugging. Does NOT record per-frame
data — only decision points.

Usage:
    from drip_physics.trace import sim_trace

    sim_trace.log("ring_handoff", ring_from=3, ring_to=4, t=0.123)
    sim_trace.log("controller_switch", mode="pid", reason="block_diagram")
    sim_trace.log("force_clamp", t=0.045, raw_force=1.2e-6, clamped_to=5.4e-7)

    # Get all events
    events = sim_trace.events

    # Filter by category
    handoffs = sim_trace.events_by_category("ring_handoff")

    # Summary counts
    print(sim_trace.summary())  # {"ring_handoff": 3, "force_clamp": 12, ...}

    # Clear for next run
    sim_trace.clear()
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger("drip_physics.trace")


@dataclass
class TraceEvent:
    """Single traced simulation decision."""

    category: str  # e.g., "ring_handoff", "force_clamp", "sim_start"
    t: float = 0.0  # simulation time (seconds)
    details: Dict[str, Any] = field(default_factory=dict)


class SimulationTrace:
    """Accumulates simulation events for post-run analysis."""

    def __init__(self) -> None:
        self._events: List[TraceEvent] = []
        self._enabled: bool = True

    def log(self, category: str, t: float = 0.0, **details: Any) -> None:
        """Record a simulation decision event."""
        if not self._enabled:
            return
        event = TraceEvent(category=category, t=t, details=details)
        self._events.append(event)
        logger.debug("[%.4fs] %s: %s", t, category, details)

    @property
    def events(self) -> List[TraceEvent]:
        """Return a shallow copy of all recorded events."""
        return list(self._events)

    def events_by_category(self, category: str) -> List[TraceEvent]:
        """Return events matching the given category."""
        return [e for e in self._events if e.category == category]

    def clear(self) -> None:
        """Remove all recorded events (call at start of each simulation run)."""
        self._events.clear()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def summary(self) -> Dict[str, int]:
        """Return {category: count} for all recorded events."""
        counts: Dict[str, int] = {}
        for e in self._events:
            counts[e.category] = counts.get(e.category, 0) + 1
        return counts


# Module-level singleton
sim_trace = SimulationTrace()
