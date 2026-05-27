"""Millennium-problem exploration: Navier-Stokes existence & smoothness
via the 1D Burgers equation shock-formation transition.

Burgers' equation
    u_t + u * u_x = nu * u_xx
is the simplest fluid analog of Navier-Stokes (NS). For inviscid Burgers
(nu = 0) with periodic IC u(x, 0) = -sin(x), the classical result is
that a SHOCK forms at finite time t_shock = 1. At that instant,
|u_x|_max diverges, and the smooth-solution branch terminates. For
viscous Burgers (nu > 0), |u_x|_max grows rapidly then saturates.

The Clay Millennium problem on Navier-Stokes asks essentially the
same question in 3D: do smooth solutions exist for all time, or do
they develop singularities? Burgers' is the 1D toy proxy.

This script:
  1. Simulates 1D viscous Burgers' equation at a range of viscosities.
  2. Tracks |u_x|_max(t) on a time grid.
  3. Applies the Systrophe catcher (value + derivative variants) to
     the |u_x|_max(t) series to detect the shock-formation transition.

If the catcher catches the shock-formation moment, that's a tool
demonstration of NS-adjacent singularity-formation detection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systrophe.catchers.derivative_catcher import catch_smooth_transition
from systrophe.catchers.novelty_catcher import scan_novelty


def burgers_step(u: np.ndarray, dx: float, dt: float, nu: float) -> np.ndarray:
    """One explicit FTCS/upwind step of Burgers' equation."""
    # Conservative flux for upwind on the convective term
    u_pos = np.maximum(u, 0)
    u_neg = np.minimum(u, 0)
    du_left = u - np.roll(u, 1)
    du_right = np.roll(u, -1) - u
    convective = u_pos * du_left / dx + u_neg * du_right / dx
    laplacian = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx ** 2
    return u - dt * convective + dt * nu * laplacian


def simulate_burgers(
    nu: float, n_x: int = 200, t_max: float = 2.0,
    n_t_record: int = 100,
) -> dict:
    """Solve viscous Burgers' on [0, 2*pi] with u(x,0) = -sin(x).

    Returns the |u_x|_max(t) series at evenly-spaced recording times.
    """
    x = np.linspace(0, 2 * np.pi, n_x, endpoint=False)
    dx = x[1] - x[0]
    u = -np.sin(x)
    # CFL: dt < dx^2 / (2*nu)  AND  dt < dx / max(|u|)
    dt_cfl_diff = 0.4 * dx ** 2 / max(nu, 1e-9)
    dt_cfl_conv = 0.4 * dx / max(np.max(np.abs(u)), 1e-6)
    dt = min(dt_cfl_diff, dt_cfl_conv, 1e-4)
    n_steps = int(t_max / dt)
    record_interval = max(1, n_steps // n_t_record)

    ts = []
    ux_max_series = []
    for step in range(n_steps + 1):
        t = step * dt
        if step % record_interval == 0:
            ux = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
            ts.append(float(t))
            ux_max_series.append(float(np.max(np.abs(ux))))
        u = burgers_step(u, dx, dt, nu)
        if not np.all(np.isfinite(u)):
            break

    return {
        "nu": float(nu),
        "t_grid": ts,
        "ux_max_series": ux_max_series,
        "n_x": n_x,
        "t_max": t_max,
        "dx": float(dx),
        "dt": float(dt),
    }


def detect_shock_transition(t_grid: list, ux_max: list) -> dict:
    """Apply scan_novelty + derivative catcher to |u_x|_max(t).

    Run THREE catcher passes:
      A. value-level catcher on raw |u_x|_max series
      B. derivative catcher on log(|u_x|_max + 1) — log compresses the
         dynamic range so the rise+peak+fall shows up as a sharper feature.
      C. peak-finder via the time when d log|u_x|_max / dt is maximised
         (sharpest growth -> shock formation moment).
    """
    ts = np.array(t_grid)
    arr = np.array(ux_max)
    log_arr = np.log(arr + 1.0)

    def fn_scalar_raw(t_val):
        idx = int(np.argmin(np.abs(ts - t_val)))
        return float(arr[idx])

    def fn_scalar_log(t_val):
        idx = int(np.argmin(np.abs(ts - t_val)))
        return float(log_arr[idx])

    def fn_array_raw(t_val):
        return np.array([fn_scalar_raw(t_val)])

    n_pts = min(60, len(ts))
    idx_sub = np.linspace(0, len(ts) - 1, n_pts).astype(int)
    ts_sub = ts[idx_sub]

    # A. value scan on raw
    scan = scan_novelty(ts_sub, fn_array_raw, n_bits=32)

    # B. derivative catcher on log
    deriv = catch_smooth_transition(ts_sub, fn_scalar_log, n_bits=32)

    # C. analytic peak finder: time where d log|u_x|_max / dt is max
    dt = float(np.median(np.diff(ts)))
    log_arr_full = np.log(arr + 1.0)
    dlog_dt = np.gradient(log_arr_full, dt)
    t_max_dlog = float(ts[int(np.argmax(dlog_dt))])
    max_dlog = float(np.max(dlog_dt))

    return {
        "scan_verdict": scan.verdict,
        "scan_n_sharp": len(scan.sharp_features),
        "scan_sharp_t_values": [
            float(sf["parameter_value"]) for sf in scan.sharp_features
        ],
        "derivative_log_kind": deriv["kind"],
        "derivative_log_centre": deriv["estimated_transition_centre"],
        "max_ux_overall": float(np.max(arr)),
        "t_of_max_ux": float(ts[int(np.argmax(arr))]),
        "t_of_max_dlog_uxdt": t_max_dlog,
        "max_dlog_uxdt": max_dlog,
    }


def main() -> None:
    print("=" * 70)
    print("Burgers' equation shock-formation catcher (NS Millennium analog)")
    print("=" * 70)
    print()

    # Classical inviscid prediction: shock at t = 1.0
    # Viscous corrections: shock smeared but |u_x|_max still peaks near t=1
    nu_grid = [0.005, 0.01, 0.02, 0.05, 0.1]

    all_results = {}
    for nu in nu_grid:
        print(f"--- Viscosity nu = {nu} ---")
        sim = simulate_burgers(nu=nu, n_x=200, t_max=2.0, n_t_record=120)
        det = detect_shock_transition(sim["t_grid"], sim["ux_max_series"])
        print(f"  max |u_x|         = {det['max_ux_overall']:.2f}")
        print(f"  t of max |u_x|    = {det['t_of_max_ux']:.3f}")
        print(f"  scan_novelty (raw): {det['scan_verdict']}, n_sharp={det['scan_n_sharp']}")
        if det["scan_sharp_t_values"]:
            print(f"    sharp at t = {det['scan_sharp_t_values']}")
        print(f"  derivative catcher (log): kind={det['derivative_log_kind']}, "
              f"centre={det['derivative_log_centre']}")
        print(f"  t of max d log|u_x|/dt = {det['t_of_max_dlog_uxdt']:.3f} "
              f"(max rate = {det['max_dlog_uxdt']:.2f})")
        all_results[f"nu_{nu}"] = {
            "simulation": {k: v for k, v in sim.items() if k != "ux_max_series"
                            and k != "t_grid"},
            "detection": det,
        }
        print()

    out_path = Path(__file__).parent / "millennium_burgers_shock_catcher_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Wrote {out_path}")
    print()
    print("Interpretation")
    print("==============")
    print("  Inviscid Burgers' classical result: shock at t_shock = 1.0 from")
    print("  IC u(x,0) = -sin(x). At finite viscosity nu, |u_x|_max peaks near")
    print("  t = 1 then decays as viscous diffusion smooths the gradient.")
    print("  Catcher should flag the t = t_peak ~ 1 location as the transition.")


if __name__ == "__main__":
    main()
