"""Tests for the gauge-fixed, constraint-damped nonlinear cylindrical evolution.

Run from this directory:  python -m pytest test_nonlinear.py -q
"""

import numpy as np
import pytest

from nonlinear_cylindrical import NonlinearCylindrical, levi_civita_static


def test_static_levi_civita_is_fixed_point():
    ev = NonlinearCylindrical(r_in=1.0, r_out=21.0, n=1200, kappa=1.0)
    psi0, om0 = levi_civita_static(ev.r, sigma=0.2)
    res = ev.evolve(t_max=4.0, psi0=psi0, om0=om0)
    interior = (ev.r > 3) & (ev.r < 14)
    assert np.max(np.abs(res["state"][0] - psi0)[interior]) < 1e-4


def test_causal_propagation_speed_one():
    t0, w = 3.0, 0.6
    drive = lambda t: 0.05 * np.exp(-((t - t0) / w) ** 2)
    ev = NonlinearCylindrical(r_in=1.0, r_out=16.0, n=1500, omega_drive=drive)
    res = ev.evolve(t_max=12.0, record_radii=[4.0, 8.0])
    t = res["t"]
    for j, rr in enumerate([4.0, 8.0]):
        k = int(np.argmax(np.abs(res["rec"][:, j, 1])))
        assert abs(t[k] - (t0 + rr - 1.0)) < 0.3


def test_c_energy_conserved_and_constraint_small():
    ev = NonlinearCylindrical(r_in=1.0, r_out=31.0, n=2000)
    r = ev.r
    psi0 = 0.03 * np.exp(-((r - 15) / 1.0) ** 2)
    om0 = 0.03 * np.exp(-((r - 15) / 1.0) ** 2)
    res = ev.evolve(t_max=6.0, psi0=psi0, om0=om0)
    E = res["c_energy"]
    assert np.max(np.abs(E - E[0])) / E[0] < 1e-3
    assert np.max(res["constraint_violation"]) < 1e-3


def test_nonlinear_polarization_coupling():
    """A strong twist (omega) sources psi -- absent in linear theory."""
    strong = lambda t: 0.5 * np.exp(-((t - 3.0) / 0.6) ** 2)
    ev = NonlinearCylindrical(r_in=1.0, r_out=16.0, n=1500, omega_drive=strong)
    res = ev.evolve(t_max=10.0, record_radii=[6.0])
    weak = lambda t: 1e-3 * np.exp(-((t - 3.0) / 0.6) ** 2)
    ev2 = NonlinearCylindrical(r_in=1.0, r_out=16.0, n=1500, omega_drive=weak)
    res2 = ev2.evolve(t_max=10.0, record_radii=[6.0])
    psi_strong = np.max(np.abs(res["rec"][:, 0, 0]))
    psi_weak = np.max(np.abs(res2["rec"][:, 0, 0]))
    assert psi_strong > 1e-3                    # nonlinear sourcing is real
    assert psi_strong > 100 * psi_weak          # scales nonlinearly (~A^2)


def test_second_order_convergence():
    # use resolutions in the asymptotic regime (lower n is not yet 2nd-order)
    def run(n):
        drive = lambda t: 0.1 * np.exp(-((t - 3.0) / 0.6) ** 2)
        ev = NonlinearCylindrical(r_in=1.0, r_out=21.0, n=n, omega_drive=drive)
        res = ev.evolve(t_max=12.0, record_radii=[8.0])
        return res["t"], res["rec"][:, 0, 1]
    tc = np.linspace(0, 12, 200)
    s = {n: np.interp(tc, *run(n)) for n in (1000, 2000, 4000)}
    e_lo = np.max(np.abs(s[1000] - s[2000]))
    e_hi = np.max(np.abs(s[2000] - s[4000]))
    assert 3.0 < e_lo / e_hi < 5.0              # ~4 => second order


def test_constraint_damping_reduces_violation():
    def cfin(kappa):
        ev = NonlinearCylindrical(r_in=1.0, r_out=31.0, n=2000, kappa=kappa)
        r = ev.r
        psi0 = 0.05 * np.exp(-((r - 15) / 1.0) ** 2)
        om0 = 0.05 * np.exp(-((r - 15) / 1.0) ** 2)
        res = ev.evolve(t_max=8.0, psi0=psi0, om0=om0)
        return float(res["constraint_violation"][-1])
    assert cfin(2.0) <= cfin(0.0) + 1e-12       # damping does not worsen it
    assert cfin(2.0) < 1e-4                      # and keeps it small


def test_chaotic_twist_drive_is_bounded():
    """Driving the nonlinear evolution with a chaotic twist stays bounded/stable."""
    rng = np.random.default_rng(0)
    t_l = np.linspace(0, 40, 400)
    chaotic = np.cumsum(rng.standard_normal(400)) * 0.01    # bounded chaotic-ish drive
    chaotic -= chaotic.mean()
    drive = lambda t: 0.05 * float(np.interp(t, t_l, chaotic))
    ev = NonlinearCylindrical(r_in=1.0, r_out=25.0, n=1500, omega_drive=drive, kappa=1.0)
    res = ev.evolve(t_max=20.0, record_radii=[5.0])
    om = res["state"][2]
    assert np.all(np.isfinite(om))
    assert np.max(np.abs(om)) < 10.0
    assert np.max(res["constraint_violation"]) < 1e-2
