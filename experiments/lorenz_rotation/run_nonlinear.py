"""Orchestrator: gauge-fixed, constraint-damped NONLINEAR cylindrical evolution.

The nonlinear capstone of the lorenz_rotation study. Evolves the full nonlinear
cylindrical Einstein vacuum (Jordan-Ehlers-Kundt / Einstein-Rosen wave-map) in the
areal + conformal gauge with Z4-style constraint damping, validates it, and drives it
with the chaotic Lorenz rotation a(t) -- confirming the full nonlinear spacetime is
stable (closing the arc adiabatic -> exact-linear -> coupled-linear -> nonlinear).

Writes lorenz_nonlinear_results.json + the mandatory novelty-catcher verdict.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from systrophe.catchers.novelty_catcher import scan_novelty, summarize_novelty_for_report

from nonlinear_cylindrical import NonlinearCylindrical, levi_civita_static
from rotating_dust_lorenz import RotatingDustLorenz, rotation_parameter_timeseries


def v_static():
    print("\n=== V1: static (Levi-Civita) fixed point ===")
    ev = NonlinearCylindrical(r_in=1.0, r_out=21.0, n=1500, kappa=1.0)
    psi0, om0 = levi_civita_static(ev.r, sigma=0.2)
    res = ev.evolve(t_max=5.0, psi0=psi0, om0=om0)
    interior = (ev.r > 3) & (ev.r < 15)
    drift = float(np.max(np.abs(res["state"][0] - psi0)[interior]))
    print(f"  interior drift over t=5: {drift:.2e}")
    return {"interior_drift": drift,
            "constraint_max": float(np.max(res["constraint_violation"]))}


def v_speed():
    print("\n=== V2: causal propagation at speed 1 (nonlinear twist pulse) ===")
    t0, w = 3.0, 0.6
    drive = lambda t: 0.05 * np.exp(-((t - t0) / w) ** 2)
    ev = NonlinearCylindrical(r_in=1.0, r_out=21.0, n=2000, omega_drive=drive)
    radii = [4.0, 9.0, 14.0]
    res = ev.evolve(t_max=17.0, record_radii=radii)
    t = res["t"]; rows = []
    for j, rr in enumerate(radii):
        k = int(np.argmax(np.abs(res["rec"][:, j, 1])))
        rows.append({"r": rr, "arrival": float(t[k]), "predicted": t0 + rr - 1.0})
        print(f"  r={rr:4.1f} arrival t={t[k]:5.2f}  predicted={t0+rr-1.0:5.2f}")
    return {"rows": rows}


def v_energy_convergence():
    print("\n=== V3: C-energy conservation + 2nd-order convergence ===")
    ev = NonlinearCylindrical(r_in=1.0, r_out=41.0, n=3000)
    r = ev.r
    psi0 = 0.03 * np.exp(-((r - 20) / 1.0) ** 2)
    om0 = 0.03 * np.exp(-((r - 20) / 1.0) ** 2)
    res = ev.evolve(t_max=8.0, psi0=psi0, om0=om0)
    E = res["c_energy"]
    e_drift = float(np.max(np.abs(E - E[0])) / E[0])

    def run(n):
        drv = lambda t: 0.1 * np.exp(-((t - 3.0) / 0.6) ** 2)
        e = NonlinearCylindrical(r_in=1.0, r_out=21.0, n=n, omega_drive=drv)
        rr = e.evolve(t_max=12.0, record_radii=[8.0])
        return rr["t"], rr["rec"][:, 0, 1]
    tc = np.linspace(0, 12, 200)
    s = {n: np.interp(tc, *run(n)) for n in (1000, 2000, 4000)}
    e_lo = np.max(np.abs(s[1000] - s[2000])); e_hi = np.max(np.abs(s[2000] - s[4000]))
    ratio = float(e_lo / e_hi)
    print(f"  C-energy rel drift = {e_drift:.2e}")
    print(f"  self-convergence ratio = {ratio:.2f}  (2nd order -> 4)")
    return {"c_energy_drift": e_drift, "convergence_ratio": ratio,
            "constraint_max": float(np.max(res["constraint_violation"]))}


def v_nonlinear_coupling():
    print("\n=== V4: nonlinear polarization coupling ===")
    strong = lambda t: 0.5 * np.exp(-((t - 3.0) / 0.6) ** 2)
    ev = NonlinearCylindrical(r_in=1.0, r_out=21.0, n=2000, omega_drive=strong)
    res = ev.evolve(t_max=12.0, record_radii=[6.0])
    psi6 = float(np.max(np.abs(res["rec"][:, 0, 0])))
    print(f"  strong twist sources psi at r=6: max|psi|={psi6:.2e} (linear theory: 0)")
    return {"psi_sourced_by_twist": psi6}


def v_constraint_damping():
    print("\n=== V5: constraint damping (with vs without) ===")
    def cfin(kappa):
        e = NonlinearCylindrical(r_in=1.0, r_out=41.0, n=2500, kappa=kappa)
        r = e.r
        psi0 = 0.05 * np.exp(-((r - 20) / 1.0) ** 2)
        om0 = 0.05 * np.exp(-((r - 20) / 1.0) ** 2)
        res = e.evolve(t_max=10.0, psi0=psi0, om0=om0)
        return float(res["constraint_violation"][-1])
    c0 = cfin(0.0); c2 = cfin(2.0)
    print(f"  final constraint: kappa=0 -> {c0:.2e}; kappa=2 -> {c2:.2e} "
          f"({c0/c2:.1f}x suppression)")
    return {"constraint_undamped": c0, "constraint_damped": c2}


def chaotic_drive():
    print("\n=== CHAOS: nonlinear evolution driven by the Lorenz rotation a(t) ===")
    m = RotatingDustLorenz(sigma=10.0, r=28.0, b=8.0 / 3.0)
    traj = m.integrate(np.array([1.0, 1.0, 1.0]), t_max=60.0, dt=0.01, t_transient=20.0)
    rot = rotation_parameter_timeseries(traj, a0=1.5, eps=0.2, clip_min=0.55)
    t_l = rot["t"] - rot["t"][0]
    twist = 0.05 * (rot["a"] - 1.5)            # map chaotic rotation -> twist amplitude
    drive = lambda t: float(np.interp(t, t_l, twist))
    ev = NonlinearCylindrical(r_in=1.0, r_out=30.0, n=2500, omega_drive=drive, kappa=1.0)
    res = ev.evolve(t_max=35.0, record_radii=[3.0, 6.0])
    psi, Pps, om, Pom, gam = res["state"]
    bounded = bool(np.all(np.isfinite(om)) and np.max(np.abs(om)) < 100)
    out = {
        "twist_drive_amplitude": float(np.max(np.abs(twist))),
        "max_omega": float(np.max(np.abs(res["rec"][:, :, 1]))),
        "max_psi_sourced": float(np.max(np.abs(res["rec"][:, :, 0]))),
        "c_energy_final": float(res["c_energy"][-1]),
        "constraint_max": float(np.max(res["constraint_violation"])),
        "bounded_stable": bounded,
    }
    print(f"  chaotic twist drive |amp|={out['twist_drive_amplitude']:.3f}")
    print(f"  response: max|omega|={out['max_omega']:.3e}  max|psi sourced|={out['max_psi_sourced']:.3e}")
    print(f"  constraint max={out['constraint_max']:.2e}  bounded/stable={bounded}")
    return out, ev, res


def main():
    t0 = time.time()
    results = {}
    results["V1_static"] = v_static()
    results["V2_speed"] = v_speed()
    results["V3_energy_convergence"] = v_energy_convergence()
    results["V4_nonlinear_coupling"] = v_nonlinear_coupling()
    results["V5_constraint_damping"] = v_constraint_damping()
    chaos, ev, res = chaotic_drive()
    results["CHAOS_driven"] = chaos

    # mandatory novelty catcher: scan observation radius of the chaotic-driven twist
    radii = np.linspace(2.5, 12.0, 14)
    full = ev.evolve(t_max=35.0, record_radii=list(radii))

    def fp(rr):
        j = int(np.argmin(np.abs(radii - rr)))
        col = full["rec"][:, j, 1]
        return np.quantile(col, np.linspace(0, 1, 16))

    scan = scan_novelty(radii, fp, n_bits=32, parameter_label="r_obs")
    results["catcher"] = {"verdict": scan.verdict, "sharp": scan.sharp_features}
    print("\n=== Novelty catcher (mandatory) ===")
    print(summarize_novelty_for_report(scan))

    results["runtime_seconds"] = round(time.time() - t0, 1)
    Path(__file__).with_name("lorenz_nonlinear_results.json").write_text(
        json.dumps(results, indent=2, default=float))
    print(f"\nWrote lorenz_nonlinear_results.json ({results['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
