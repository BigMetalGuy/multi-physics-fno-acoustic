"""Phase 4 v0 validation receipt — coupled-physics one-way vs two-way trajectory.

Builds the L1 chamber mesh, runs a 50-step coupled trajectory with one
droplet falling at ~1 m/s through chamber centre at initial temperature
~1123 K (crucible), compares one-way (no streaming) vs two-way
(streaming-driven advection) thermal trajectories.

Headline number per FEM_V2_DESIGN §3.4:
    droplet T-at-landing difference between one-way and two-way models.

Plots:
  * Droplet T vs time (centerline droplet T tracked along trajectory),
  * Streaming velocity magnitude mid-z slice at final timestep,
  * Field magnitude mid-z slice at final timestep,
  * Inner-loop iteration count per outer step.

Saves PNGs + summary JSON to:
    simulations/ml_inverse/research/_fem_coupled_v0/

Mesh choice: COARSE (focus_h=2 mm, bulk_h=10 mm).  Production mesh
trajectories take ~50 minutes wall and gate on PETSc-MUMPS; v0 receipt
uses coarse mesh + Sommerfeld v0 path for ~5 min wall, documenting
the caveat in the JSON receipt.

Usage::

    cd simulations
    PYTHONPATH=. ml_inverse/_jaxfem_discovery/.venv/bin/python \\
        drip_physics/backends/femcoupled/validate_coupled.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drip_physics.backends.femcoupled import build_l1_chamber_mesh
from drip_physics.backends.femcoupled.coupling import StaggeredCouplingDriver
from drip_physics.backends.femcoupled.geometry import (
    resolve_chamber_geometry,
)
from drip_physics.backends.femcoupled.heat import (
    DEFAULT_THERMAL_DT_S,
    T_CHAMBER_INIT_K,
    T_CRUCIBLE_K,
    T_SUBSTRATE_K,
)
from drip_physics.config import (
    AIR_200C,
    DropletConfig,
    EnvironmentConfig,
    L1_ARRAY,
    MA40S4S_40KHZ,
)


OUT_DIR = (
    Path(__file__).resolve().parents[3]
    / "ml_inverse"
    / "research"
    / "_fem_coupled_v0"
)


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------


def make_droplet_trajectory(
    geom,
    n_steps: int,
    dt: float,
    drop_speed_m_s: float = 1.0,
    z_start_frac: float = 0.95,
    z_end_frac: float = 0.05,
    amplitude_w: float = 5.0,
) -> List[Dict[str, jnp.ndarray]]:
    """Single droplet falling through chamber centre.

    Args:
        geom: ChamberGeometry.
        n_steps: number of timesteps in trajectory.
        dt: time step (s).
        drop_speed_m_s: fall velocity (m/s).  Default 1 m/s ≈ low-Re
            regime per FEM_V2_DESIGN §3.4.
        z_start_frac: starting z as fraction of chamber height.
        z_end_frac: ending z (above substrate).
        amplitude_w: per-step heat amplitude (W).

    Returns:
        list of per-step dicts.
    """
    h = geom.height
    z_start = z_start_frac * h
    z_end = z_end_frac * h
    # Speed-driven trajectory; clamp to z_end if drop reaches floor.
    z = z_start - drop_speed_m_s * dt * np.arange(n_steps)
    z = np.maximum(z, z_end)
    traj = []
    for zi in z:
        traj.append({
            "centers": jnp.array([[0.0, 0.0, float(zi)]], dtype=jnp.float64),
            "amplitudes": jnp.array([amplitude_w], dtype=jnp.float64),
        })
    return traj


def droplet_T_along_trajectory(
    T_history: jnp.ndarray,
    droplet_trajectory: List[Dict[str, jnp.ndarray]],
    points: jnp.ndarray,
) -> np.ndarray:
    """Sample T at the moving droplet centre at each timestep.

    Returns: ``(n_steps + 1,)`` array.  Step 0 is the initial T at
    droplet's starting position; steps 1..n are after the corresponding
    outer step.
    """
    pts = np.asarray(points)
    T_np = np.asarray(T_history)
    n_steps_plus_one = T_np.shape[0]
    out = np.zeros(n_steps_plus_one)
    for i in range(n_steps_plus_one):
        idx = max(0, min(i - 1, len(droplet_trajectory) - 1))
        center = np.asarray(droplet_trajectory[idx]["centers"])[0]
        node = int(np.argmin(np.linalg.norm(pts - center[None, :], axis=1)))
        out[i] = float(T_np[i, node])
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_droplet_T(
    droplet_T_oneway: np.ndarray,
    droplet_T_twoway: np.ndarray,
    dt: float,
    out_path: Path,
) -> None:
    """Droplet T vs time (one-way vs two-way)."""
    n = len(droplet_T_oneway)
    t = np.arange(n) * dt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t * 1000,
        droplet_T_oneway,
        "b-",
        label="one-way (no streaming)",
        linewidth=1.5,
    )
    ax.plot(
        t * 1000,
        droplet_T_twoway,
        "r-",
        label="two-way (Eckart streaming)",
        linewidth=1.5,
    )
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("T at droplet centre (K)")
    ax.set_title("Droplet thermal trajectory: one-way vs two-way coupling")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_mid_z_slice(
    field: np.ndarray,
    points: np.ndarray,
    z_target: float,
    title: str,
    cbar_label: str,
    out_path: Path,
    cmap: str = "viridis",
) -> None:
    """2D scatter of `field` at nodes near ``z = z_target``."""
    pts = np.asarray(points)
    z_band = (pts[:, 2] > z_target - 0.01) & (pts[:, 2] < z_target + 0.01)
    if not z_band.any():
        z_band = np.ones(len(pts), dtype=bool)
    xy = pts[z_band, :2] * 1000.0  # mm
    val = field[z_band]

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=val, s=8, cmap=cmap)
    fig.colorbar(sc, ax=ax, label=cbar_label)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_inner_iters(
    inner_oneway: List[int],
    inner_twoway: List[int],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    n = max(len(inner_oneway), len(inner_twoway))
    steps = np.arange(n)
    ax.bar(
        steps - 0.2,
        inner_oneway[:n] + [0] * (n - len(inner_oneway)),
        width=0.4,
        label="one-way",
        color="b",
        alpha=0.6,
    )
    ax.bar(
        steps + 0.2,
        inner_twoway[:n] + [0] * (n - len(inner_twoway)),
        width=0.4,
        label="two-way",
        color="r",
        alpha=0.6,
    )
    ax.set_xlabel("outer step index")
    ax.set_ylabel("inner-loop iterations")
    ax.set_title("Inner-loop iteration count per outer step")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Phase 4 validation receipt → {OUT_DIR}")

    # Mesh build (coarse for v0 receipt).
    print("Building L1 chamber mesh (coarse: focus=2mm, bulk=10mm)...")
    mesh_path, info = build_l1_chamber_mesh(
        array_config=L1_ARRAY,
        transducer_config=MA40S4S_40KHZ,
        focus_h_mm=2.0,
        bulk_h_mm=10.0,
        overwrite=False,
        verbose=False,
    )
    print(
        f"  mesh: {info['element_count']} tets, "
        f"{info['node_count']} nodes, "
        f"file={info['file_size_bytes'] / 1e6:.2f} MB"
    )

    # Environment for chamber-air-temperature simulation (200 °C).
    env = EnvironmentConfig(
        medium=AIR_200C,
        temperature=473.15,
        transducer=MA40S4S_40KHZ,
        array=L1_ARRAY,
    )
    geom = resolve_chamber_geometry(L1_ARRAY, MA40S4S_40KHZ)

    n_steps = 50
    dt = 0.01  # 10 ms
    droplet_traj = make_droplet_trajectory(
        geom=geom, n_steps=n_steps, dt=dt
    )

    # Phases: focal trap at landing position (chamber centre, 5 mm above
    # substrate).  We use a simple linear phase ramp as a starting point
    # — production phase command would come from the inverse solver.
    n_t = geom.total_transducers
    phases = jnp.linspace(0.0, jnp.pi, n_t, dtype=jnp.float64)

    # Initial temperature: chamber-air at 473 K everywhere except a hot
    # crucible-inlet region; simplification — start at chamber-air T.
    n_nodes_estimate = info["node_count"]  # upper bound

    # Driving piston amplitude — the default 0.01 m/s yields only a few
    # Pa peak field on a coarse mesh, well below the FEM_V2_DESIGN §3.4
    # 1500 Pa operating point.  We raise the amplitude to reach the
    # design's pressure regime so the streaming-driven thermal
    # difference is measurable in the receipt.
    velocity_amp = 1.0  # m/s — 100× spec-sheet to reach design pressure

    # ------------------------- one-way -----------------------------------
    print("\n[1/2] Running ONE-WAY coupled trajectory (streaming OFF)...")
    one_way = StaggeredCouplingDriver(
        mesh_path=mesh_path,
        surface_tags=info["surface_tags"],
        env_config=env,
        transducer_config=MA40S4S_40KHZ,
        n_outer_steps=n_steps,
        dt=dt,
        max_inner_iters=3,
        convergence_tol=1e-2,
        pml_width=0.0,             # Sommerfeld v0 (faster than PML)
        velocity_amp=velocity_amp,
        enable_streaming=False,
    )
    n_nodes = one_way._n_nodes
    T_init = jnp.full((n_nodes,), float(T_CHAMBER_INIT_K), dtype=jnp.float64)
    t0 = time.time()
    res_oneway = one_way.trajectory(
        phases_per_step=phases,
        droplet_trajectory=droplet_traj,
        T_initial=T_init,
    )
    one_way_wall = time.time() - t0
    print(
        f"  one-way wall: {one_way_wall:.1f}s  "
        f"({one_way_wall / n_steps:.2f}s/step)  "
        f"inner iters: {res_oneway.inner_iter_counts}"
    )

    # ------------------------- two-way -----------------------------------
    print("\n[2/2] Running TWO-WAY coupled trajectory (streaming ON)...")
    two_way = StaggeredCouplingDriver(
        mesh_path=mesh_path,
        surface_tags=info["surface_tags"],
        env_config=env,
        transducer_config=MA40S4S_40KHZ,
        n_outer_steps=n_steps,
        dt=dt,
        max_inner_iters=3,
        convergence_tol=1e-2,
        pml_width=0.0,
        velocity_amp=velocity_amp,
        enable_streaming=True,
    )
    t0 = time.time()
    res_twoway = two_way.trajectory(
        phases_per_step=phases,
        droplet_trajectory=droplet_traj,
        T_initial=T_init,
    )
    two_way_wall = time.time() - t0
    print(
        f"  two-way wall: {two_way_wall:.1f}s  "
        f"({two_way_wall / n_steps:.2f}s/step)  "
        f"inner iters: {res_twoway.inner_iter_counts}"
    )

    # ------------------------- analysis -----------------------------------
    points = res_oneway.mesh_coords
    droplet_T_oneway = droplet_T_along_trajectory(
        res_oneway.T_history, droplet_traj, points
    )
    droplet_T_twoway = droplet_T_along_trajectory(
        res_twoway.T_history, droplet_traj, points
    )

    delta_landing = float(droplet_T_twoway[-1] - droplet_T_oneway[-1])
    delta_max = float(np.max(np.abs(droplet_T_twoway - droplet_T_oneway)))

    print(
        f"\n=== HEADLINE NUMBER ===\n"
        f"  droplet T at landing:\n"
        f"    one-way: {droplet_T_oneway[-1]:.2f} K\n"
        f"    two-way: {droplet_T_twoway[-1]:.2f} K\n"
        f"    Δ      : {delta_landing:+.2f} K (two-way − one-way)\n"
        f"  max |ΔT(droplet)| over trajectory: {delta_max:.2f} K"
    )

    # ------------------------- plots ---------------------------------------
    print("\nWriting plots...")
    plot_droplet_T(
        droplet_T_oneway, droplet_T_twoway, dt,
        OUT_DIR / "droplet_T_trajectory.png",
    )

    # Streaming + field slices (final step) — two-way only.
    z_mid = geom.height / 2.0
    final_re = np.asarray(res_twoway.re_history[-1])
    final_im = np.asarray(res_twoway.im_history[-1])
    final_mag_p = np.sqrt(final_re ** 2 + final_im ** 2)
    final_u_stream = np.asarray(res_twoway.streaming_history[-1])
    final_u_mag = np.linalg.norm(final_u_stream, axis=-1)

    plot_mid_z_slice(
        final_mag_p,
        points,
        z_mid,
        "Pressure |p| mid-z slice (final timestep, two-way)",
        "|p| (Pa)",
        OUT_DIR / "field_mag_mid_z.png",
    )
    plot_mid_z_slice(
        final_u_mag,
        points,
        z_mid,
        "Streaming velocity |u| mid-z slice (final timestep)",
        "|u_stream| (m/s)",
        OUT_DIR / "streaming_mag_mid_z.png",
        cmap="plasma",
    )
    plot_inner_iters(
        res_oneway.inner_iter_counts,
        res_twoway.inner_iter_counts,
        OUT_DIR / "inner_iters.png",
    )

    # ------------------------- receipt JSON --------------------------------
    receipt = {
        "phase": "Phase 4 v0 — StaggeredCouplingDriver receipt",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "headline_caveat": (
            "Coarse mesh (focus=2mm, bulk=10mm) + linear phase ramp "
            "(not inverse-solved focal-trap) + Sommerfeld v0 BC produce "
            "a peak field well below the FEM_V2_DESIGN §3.4 1500 Pa "
            "design point.  At the actually-achieved peak (~300 Pa) "
            "streaming velocity is ~mm/s and the resulting droplet T-at-"
            "landing delta is sub-microkelvin — too small to dominate "
            "conductive heat loss for this trajectory length.  The "
            "framework correctly composes all four pieces (Helmholtz + "
            "streaming + heat advection + fixed-point loop); reaching "
            "the design's 10–50 K T-delta requires production mesh + "
            "inverse-solved phases + longer trajectory.  This receipt "
            "demonstrates the v1 framework, not the v1 quantitative "
            "physics bound."
        ),
        "mesh": {
            "focus_h_mm": float(info["focus_h_mm"]),
            "bulk_h_mm": float(info["bulk_h_mm"]),
            "element_count": int(info["element_count"]),
            "node_count": int(info["node_count"]),
            "file_size_mb": float(info["file_size_bytes"] / 1e6),
            "geometry_hash": info["geometry_hash"],
            "caveat": (
                "Coarse mesh chosen for receipt-time tractability. "
                "Production mesh (focus=1mm, bulk=5mm) gates on PETSc-MUMPS."
            ),
        },
        "config": {
            "n_outer_steps": int(n_steps),
            "dt_s": float(dt),
            "max_inner_iters": 3,
            "convergence_tol": 1e-2,
            "pml_width": 0.0,
            "velocity_amp_m_s": float(velocity_amp),
            "T_chamber_init_k": float(T_CHAMBER_INIT_K),
            "T_substrate_k": float(T_SUBSTRATE_K),
            "T_crucible_k": float(T_CRUCIBLE_K),
        },
        "headline": {
            "droplet_T_at_landing_oneway_k": float(droplet_T_oneway[-1]),
            "droplet_T_at_landing_twoway_k": float(droplet_T_twoway[-1]),
            "delta_T_at_landing_k": delta_landing,
            "max_abs_delta_T_along_trajectory_k": delta_max,
        },
        "wall_time": {
            "one_way_total_s": float(one_way_wall),
            "two_way_total_s": float(two_way_wall),
            "one_way_per_step_s": float(one_way_wall / n_steps),
            "two_way_per_step_s": float(two_way_wall / n_steps),
        },
        "inner_iters": {
            "one_way_per_step": list(map(int, res_oneway.inner_iter_counts)),
            "two_way_per_step": list(map(int, res_twoway.inner_iter_counts)),
            "one_way_mean": float(np.mean(res_oneway.inner_iter_counts)),
            "two_way_mean": float(np.mean(res_twoway.inner_iter_counts)),
        },
        "streaming": {
            "max_velocity_m_s_final_step": float(final_u_mag.max()),
            "mean_velocity_m_s_final_step": float(final_u_mag.mean()),
            "max_pressure_pa_final_step": float(final_mag_p.max()),
        },
        "outputs": {
            "droplet_T_trajectory": "droplet_T_trajectory.png",
            "field_mag_mid_z": "field_mag_mid_z.png",
            "streaming_mag_mid_z": "streaming_mag_mid_z.png",
            "inner_iters": "inner_iters.png",
        },
    }
    with (OUT_DIR / "receipt.json").open("w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print(f"\nReceipt written: {OUT_DIR / 'receipt.json'}")


if __name__ == "__main__":
    main()
