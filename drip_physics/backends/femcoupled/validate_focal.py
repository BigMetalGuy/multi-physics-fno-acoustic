"""Phase 2 v0 validation receipt — single-trap focal pressure FEM vs analytical.

Builds the L1 chamber mesh, solves the Helmholtz problem with phases set
to focus at the chamber centre, plots a mid-z slice of ``|p|``, compares
against the analytical free-field formula at the same target, and writes
both a PNG and a JSON summary into
``simulations/ml_inverse/research/_fem_helmholtz_v0/``.

This is the FEM analogue of ``ULTRAINO_VALIDATION.md`` for j-Wave, and
gives Phase 2.1 (Bermúdez PML) a quantitative baseline to beat.

Usage::

    cd simulations
    PYTHONPATH=. ml_inverse/_jaxfem_discovery/.venv/bin/python \
        drip_physics/backends/femcoupled/validate_focal.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from drip_physics.backends.femcoupled import build_l1_chamber_mesh
from drip_physics.backends.femcoupled.geometry import (
    resolve_chamber_geometry,
    transducer_centers,
)
from drip_physics.backends.femcoupled.helmholtz import (
    DEFAULT_PISTON_VELOCITY_AMP_M_S,
    solve_helmholtz,
)
from drip_physics.config import (
    L1_ARRAY,
    MA40S4S_40KHZ,
    default_environment,
)


# ---------------------------------------------------------------------------
# Output destination
# ---------------------------------------------------------------------------

OUT_DIR = (
    Path(__file__).resolve().parents[3]
    / "ml_inverse"
    / "research"
    / "_fem_helmholtz_v0"
)


def _free_field_pressure(
    centers: np.ndarray,
    phases: np.ndarray,
    target: np.ndarray,
    k: float,
    rho0: float,
    omega: float,
    v_amp: float,
) -> complex:
    """Analytical free-field complex pressure at ``target``.

    Spherical-source superposition (Marzo 2018 piston model with
    monopole approximation good for ka << 1):

        p(r) = Σᵢ (i·ω·ρ₀·v_amp / (2π·||r - rᵢ||))
                · exp(i·(φᵢ - k·||r - rᵢ||))

    The 1/(2π·d) prefactor is the Rayleigh-distance approximation of a
    rigid-piston aperture in a baffle (Marzo 2018 eq. 4 with sinc=1
    in the on-axis far-field limit).
    """
    diffs = target[None, :] - centers
    distances = np.linalg.norm(diffs, axis=1)
    amplitude = (1j * omega * rho0 * v_amp) / (2 * np.pi * distances)
    phase = phases - k * distances
    p = np.sum(amplitude * np.exp(1j * phase))
    return p


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 1) Build mesh
    # -----------------------------------------------------------------
    print(f"[{datetime.now():%H:%M:%S}] building L1 chamber mesh ...")
    t0 = time.time()
    mesh_path, info = build_l1_chamber_mesh(
        array_config=L1_ARRAY,
        transducer_config=MA40S4S_40KHZ,
        focus_h_mm=1.0,
        bulk_h_mm=5.0,
        cache_dir=OUT_DIR / "_mesh_cache",
        overwrite=False,
        verbose=False,
    )
    t_mesh = time.time() - t0
    print(
        f"  mesh: {info['element_count']} tets, "
        f"{info['node_count']} nodes  ({t_mesh:.2f} s)"
    )

    # -----------------------------------------------------------------
    # 2) Set up focusing phases
    # -----------------------------------------------------------------
    env = default_environment()
    geom = resolve_chamber_geometry(L1_ARRAY, MA40S4S_40KHZ)
    centers = transducer_centers(geom)
    focus = np.array([0.0, 0.0, geom.height / 2.0], dtype=np.float64)
    k = float(env.wavenumber)
    omega = float(env.angular_frequency)
    rho0 = float(env.medium.density)
    v_amp = DEFAULT_PISTON_VELOCITY_AMP_M_S

    distances = np.linalg.norm(centers - focus[None, :], axis=1)
    phases = jnp.asarray(-k * distances, dtype=jnp.float64)

    # -----------------------------------------------------------------
    # 3) FEM solve
    # -----------------------------------------------------------------
    print(f"[{datetime.now():%H:%M:%S}] FEM Helmholtz solve ...")
    t0 = time.time()
    sol = solve_helmholtz(
        mesh_path=mesh_path,
        surface_tags=info["surface_tags"],
        env_config=env,
        transducer_config=MA40S4S_40KHZ,
        phases=phases,
        velocity_amp=v_amp,
    )
    t_solve = time.time() - t0
    coords = np.asarray(sol.mesh_coords)
    re = np.asarray(sol.re)
    im = np.asarray(sol.im)
    mag = np.sqrt(re ** 2 + im ** 2)
    print(
        f"  solve wall: {t_solve:.2f} s  "
        f"|p|_max={mag.max():.4e} Pa  |p|_mean={mag.mean():.4e}"
    )

    # -----------------------------------------------------------------
    # 4) Locate focal peak in the central box
    # -----------------------------------------------------------------
    in_box = (
        (np.abs(coords[:, 0]) < 0.025)
        & (np.abs(coords[:, 1]) < 0.025)
        & (np.abs(coords[:, 2] - focus[2]) < 0.025)
    )
    sub = mag.copy()
    sub[~in_box] = -np.inf
    peak_idx = int(np.argmax(sub))
    peak_xyz = coords[peak_idx]
    err_mm = float(np.linalg.norm(peak_xyz - focus) * 1000.0)

    # -----------------------------------------------------------------
    # 5) Analytical free-field at the same target
    # -----------------------------------------------------------------
    p_analytical = _free_field_pressure(
        centers=centers,
        phases=np.asarray(phases),
        target=focus,
        k=k,
        rho0=rho0,
        omega=omega,
        v_amp=v_amp,
    )
    p_an_mag = abs(p_analytical)
    p_fem_at_focus = mag[peak_idx]

    # -----------------------------------------------------------------
    # 6) Mid-z slice plot
    # -----------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Pick nodes within ~1 mm of the focal z-plane.
        slice_mask = np.abs(coords[:, 2] - focus[2]) < 1e-3
        x_s = coords[slice_mask, 0] * 1000  # mm
        y_s = coords[slice_mask, 1] * 1000
        m_s = mag[slice_mask]

        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.tricontourf(x_s, y_s, m_s, levels=40, cmap="viridis")
        ax.scatter([0], [0], marker="x", color="white", s=80, label="target")
        ax.scatter(
            [peak_xyz[0] * 1000],
            [peak_xyz[1] * 1000],
            marker="+",
            color="red",
            s=120,
            label=f"FEM peak ({err_mm:.2f} mm err)",
        )
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
        ax.set_title(
            f"FEM v0 Helmholtz |p| @ z={focus[2] * 1000:.1f} mm\n"
            f"focus@(0,0,{focus[2] * 1000:.1f}mm); err={err_mm:.2f} mm; "
            f"|p|_FEM={p_fem_at_focus:.2e} Pa  |p|_an={p_an_mag:.2e} Pa"
        )
        plt.colorbar(sc, ax=ax, label="|p| (Pa)")
        ax.legend(loc="upper right")
        png_path = OUT_DIR / "focal_slice_z_mid.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  PNG written: {png_path}")
    except ImportError:
        print("  matplotlib not available; skipping plot.")
        png_path = None

    # -----------------------------------------------------------------
    # 7) JSON summary
    # -----------------------------------------------------------------
    summary: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "mesh": {
            "path": str(mesh_path),
            "element_count": int(info["element_count"]),
            "node_count": int(info["node_count"]),
            "focus_h_mm": float(info["focus_h_mm"]),
            "bulk_h_mm": float(info["bulk_h_mm"]),
            "build_wall_time_s": float(t_mesh),
        },
        "solve": {
            "wall_time_s": float(t_solve),
            "n_dofs": int(sol.n_dofs),
            "p_mag_max": float(mag.max()),
            "p_mag_mean": float(mag.mean()),
        },
        "focal_alignment": {
            "target_xyz_m": focus.tolist(),
            "peak_xyz_m": peak_xyz.tolist(),
            "peak_error_mm": err_mm,
            "p_fem_at_peak_pa": float(p_fem_at_focus),
        },
        "analytical_comparison": {
            "p_analytical_re_pa": float(p_analytical.real),
            "p_analytical_im_pa": float(p_analytical.imag),
            "p_analytical_mag_pa": float(p_an_mag),
            "magnitude_ratio_fem_over_analytical": float(
                p_fem_at_focus / p_an_mag
            ) if p_an_mag > 0 else None,
        },
        "physics": {
            "frequency_hz": float(env.transducer.frequency),
            "speed_of_sound_m_s": float(env.speed_of_sound),
            "wavelength_m": float(env.wavelength),
            "wavenumber_per_m": float(k),
            "rho0_kg_m3": float(rho0),
            "v_amp_m_s": float(v_amp),
        },
        "png": str(png_path) if png_path else None,
    }
    json_path = OUT_DIR / "validation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON written: {json_path}")

    print("\n=== Validation summary ===")
    print(f"  focal error:        {err_mm:.2f} mm")
    print(f"  |p|_FEM at peak:    {p_fem_at_focus:.4e} Pa")
    print(f"  |p|_analytical:     {p_an_mag:.4e} Pa")
    if p_an_mag > 0:
        print(
            f"  ratio FEM/an:       "
            f"{p_fem_at_focus / p_an_mag:.3f} "
            f"(1.0 = perfect free-field agreement)"
        )


if __name__ == "__main__":
    main()
