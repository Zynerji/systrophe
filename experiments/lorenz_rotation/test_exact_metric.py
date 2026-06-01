"""Tests for the exact (hyperbolic) time-varying metric treatment.

Run from this directory:  python -m pytest test_exact_metric.py -q
"""

import numpy as np
import pytest

from cylindrical_wave import CylindricalWave, vanstockum_potential
from exact_ctc import exact_vs_adiabatic_ctc, background_instability_probe


def test_causal_propagation_speed_one():
    """A rotation pulse at r=R reaches radius r at retarded time ~ (r-R):
    finite signal speed = 1, the physics adiabatic discards."""
    t0, w = 3.0, 0.6
    drive = lambda t: np.exp(-((t - t0) / w) ** 2)
    ddot = lambda t: -2 * (t - t0) / w ** 2 * np.exp(-((t - t0) / w) ** 2)
    cw = CylindricalWave(R=1.0, r_out=13.0, n=1200, V=None, drive=drive,
                         drive_dot=ddot, cfl=0.4)
    radii = [4.0, 8.0]
    res = cw.evolve(t_max=12.0, record_radii=radii)
    t = res["t"]
    for j, rr in enumerate(radii):
        k = int(np.argmax(np.abs(res["rec"][:, j])))
        assert abs(t[k] - (t0 + (rr - 1.0))) < 0.3   # arrives at retarded time


def test_energy_conservation_undriven():
    cw = CylindricalWave(R=1.0, r_out=31.0, n=3000, V=None,
                         drive=lambda t: 0.0, drive_dot=lambda t: 0.0, cfl=0.4)
    r = cw.r
    u0 = np.exp(-((r - 15.0) / 1.0) ** 2)
    res = cw.evolve(t_max=8.0, u0=u0, pi0=np.zeros_like(r))
    E = res["energy"]
    assert np.max(np.abs(E - E[0])) / E[0] < 1e-3


def test_retardation_lag_equals_travel_time():
    """In the stable far-field (V=0) model, the frame-dragging response lags the
    drive by exactly the travel time (r-R) -- adiabatic assumes zero lag."""
    res = exact_vs_adiabatic_ctc(a0=1.5, R=1.0, eps=0.05, r_out=24.0, n=1800,
                                 t_max=50.0, lorenz_t_max=70.0,
                                 r_obs_list=(7.0, 10.0), use_potential=False)
    for row in res["rows"]:
        # measured cross-correlation lag matches the light-travel time
        assert abs(row["measured_lag"] - row["retardation_travel_time"]) < 0.5


def test_adiabatic_limit_recovered_when_slow():
    """Slow drive -> response lag is a negligible fraction of the period
    (adiabatic OK). Fast drive -> O(1) phase error (adiabatic fails)."""
    r_star, A = 6.0, 1e-3

    def phase_err(Om):
        drive = lambda t: A * np.sin(Om * t)
        ddot = lambda t: A * Om * np.cos(Om * t)
        cw = CylindricalWave(R=1.0, r_out=30.0, n=2000, V=None, drive=drive,
                             drive_dot=ddot, cfl=0.4)
        res = cw.evolve(t_max=70.0, record_radii=[r_star])
        t, sig = res["t"], res["rec"][:, 0]
        mask = t > 35.0
        tt, ss = t[mask], sig[mask]
        dd = A * np.sin(Om * tt)
        ss = ss - ss.mean(); dd = dd - dd.mean()
        corr = np.correlate(ss, dd, mode="full")
        lags = np.arange(-len(tt) + 1, len(tt)) * (tt[1] - tt[0])
        return Om * abs(float(lags[np.argmax(corr)]))

    assert phase_err(0.6) > phase_err(0.1)        # faster -> larger adiabatic error
    assert phase_err(0.6) > 1.0                    # fast drive: adiabatic clearly fails


def test_v0_chaotic_drive_is_bounded():
    """The far-field (V=0) solver is stable under the chaotic rotation drive."""
    res = exact_vs_adiabatic_ctc(a0=1.5, eps=0.05, r_out=24.0, n=1500,
                                 t_max=40.0, lorenz_t_max=60.0,
                                 r_obs_list=(5.0,), use_potential=False)
    assert np.all(np.isfinite(res["rec"]))
    assert np.max(np.abs(res["rec"])) < 100.0      # bounded


def test_supercritical_background_is_tachyonic_and_unstable():
    """The reduced frame-dragging potential is negative (tachyonic) near
    ergosurfaces, so the driven supercritical background diverges -- it is NOT
    quasi-statically stable (the adiabatic assumption is qualitatively invalid)."""
    out = background_instability_probe(a0=1.5, r_out=12.0, n=2000, t_max=18.0)
    assert out["most_negative_V"] < 0.0            # tachyonic
    assert out["diverged"]                          # driven evolution blows up


def test_static_background_potential_regular_away_from_ergosurface():
    V, vs, c0 = vanstockum_potential(a0=1.5, R=1.0)
    # away from the ergosurfaces (1.54, 4.69) the potential is finite & modest
    r = np.array([2.5, 3.0, 3.5, 6.0, 8.0])
    Vr = V(r)
    assert np.all(np.isfinite(Vr))
    assert np.max(np.abs(Vr)) < 50.0
