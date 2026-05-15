# drip_physics_core

Shared interface package between the simulations engine (`drip_physics/`) and
the design-tool web backend (`web/design-tool/`).

## Status

Phase A of the simulations + design-tool repo split.

This package was extracted out of `drip_physics/` so the design tool no longer
needs to depend on the heavy simulation engine just to consume types.

## What lives here

- `config.py` — Canonical type definitions: `MaterialConfig`,
  `EnvironmentConfig`, `ArrayConfig`, `TransducerConfig`, `DropletConfig`,
  `ThermalConfig`, `PressureConfig`, plus all built-in profiles
  (`ALUMINUM_LIQUID`, `AIR_20C`, `L1_ARRAY`, `MA40S4S_40KHZ`, ...).
- `design/` — The parametric design engine: constraints, optimizer,
  sensitivity, sweeps, uncertainty, machine spec, test plan, bonding,
  reports.

## What does NOT live here

The full physics simulation library (pressure, force, trajectory, thermal,
inverse, block diagrams, backends) stays in `drip_physics/` because it
mutates rapidly and depends on PDE solvers, optimizers, plotting, etc.

## Backward compatibility

Existing `from drip_physics.config import X` and
`from drip_physics.design import Y` calls continue to work via shim modules
in the old paths. Those shims will be removed in Phase B (when this package
moves to its own repo).

## Design rule

`drip_physics_core` may NOT depend on `drip_physics`. The one exception is
`design/physics_eval.py::_sim_reach_lru`, which does a lazy import inside
the function body for an optional sim-derived reach calculation. Phase B
will replace this with an injected callable / Protocol.
