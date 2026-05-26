"""Tests for Unruh-modulated AB fringes."""

import math

import numpy as np
import pytest

from systrophe.unruh_modulated_ab import (
    bose_einstein,
    observer_unruh_temperature,
    sweep_observer_radii,
    thermal_decoherence,
    total_visibility,
    unruh_ab_intensity,
)
from systrophe.vanstockum import VanStockumInterior
from systrophe.which_way_ancilla import path_coherence


# ----- Bose-Einstein ------------------------------------------------------


def test_bose_einstein_zero_T_is_zero():
    assert bose_einstein(1.0, 0.0) == 0.0


def test_bose_einstein_diverges_at_high_T():
    n_lo = bose_einstein(1.0, 1.0)
    n_hi = bose_einstein(1.0, 1000.0)
    assert n_hi > n_lo > 0.0


def test_bose_einstein_known_value():
    # nbar(omega=1, T=1) = 1 / (e - 1)
    assert bose_einstein(1.0, 1.0) == pytest.approx(
        1.0 / (math.e - 1.0), rel=1e-12,
    )


@pytest.mark.parametrize("bad", [-0.5, 0.0])
def test_bose_einstein_rejects_nonpositive_omega(bad):
    with pytest.raises(ValueError):
        bose_einstein(bad, 1.0)


def test_bose_einstein_rejects_negative_T():
    with pytest.raises(ValueError):
        bose_einstein(1.0, -0.1)


# ----- thermal_decoherence ------------------------------------------------


def test_thermal_decoherence_at_T_zero_is_one():
    assert thermal_decoherence(0.0) == pytest.approx(1.0, abs=1e-12)


def test_thermal_decoherence_monotone_in_T():
    Ts = [0.0, 0.5, 1.0, 2.0, 10.0, 100.0]
    etas = [thermal_decoherence(T) for T in Ts]
    diffs = np.diff(etas)
    assert np.all(diffs <= 1e-12)


def test_thermal_decoherence_goes_to_zero_at_large_T():
    assert thermal_decoherence(1e6, omega=1.0, gamma_tau=1.0) < 1e-6


def test_thermal_decoherence_infinite_T_handled():
    assert thermal_decoherence(float("inf")) == 0.0


def test_thermal_decoherence_rejects_negative_gamma_tau():
    with pytest.raises(ValueError):
        thermal_decoherence(1.0, gamma_tau=-0.1)


# ----- total_visibility ---------------------------------------------------


def test_inertial_observer_recovers_pure_englert():
    # T_U = 0: V_total should equal V_pure(kappa).
    for kappa in (0.0, 0.25, 0.5, 0.75, 1.0):
        V = total_visibility(kappa, T_U=0.0)
        assert V == pytest.approx(path_coherence(kappa), abs=1e-12)


def test_no_which_way_coupling_pure_thermal():
    # kappa = 0: V_total = eta_thermal(T_U).
    for T in (0.0, 0.5, 1.0, 5.0):
        V = total_visibility(0.0, T_U=T)
        assert V == pytest.approx(thermal_decoherence(T), abs=1e-12)


def test_total_visibility_factorizes():
    kappa, T = 0.4, 2.0
    V = total_visibility(kappa, T_U=T)
    assert V == pytest.approx(
        path_coherence(kappa) * thermal_decoherence(T), abs=1e-12,
    )


def test_total_visibility_in_unit_interval():
    rng = np.random.default_rng(42)
    for _ in range(20):
        kappa = float(rng.uniform(0, 1))
        T = float(rng.uniform(0, 50))
        V = total_visibility(kappa, T_U=T)
        assert 0.0 <= V <= 1.0


# ----- observer geometry --------------------------------------------------


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_observer_temperature_increases_with_radius(vs):
    # On the supercritical exterior, centripetal a = Omega^2 r / sqrt(F)
    # grows with r until F -> 0 at the next CH, so T_U does too.
    r_grid = np.linspace(1.05, 1.4, 6)
    Ts = [observer_unruh_temperature(vs, float(r)) for r in r_grid]
    diffs = np.diff(Ts)
    assert np.all(diffs >= -1e-12)


def test_observer_temperature_returns_inf_past_horizon(vs):
    # Inside an ergosurface (F <= 0) centripetal_acceleration_at_orbit
    # returns inf, so the observer Unruh temperature is also inf.
    # For omega=R=1 the exterior F changes sign between r=1.8 and r=1.9.
    T = observer_unruh_temperature(vs, 1.95)
    assert math.isinf(T)


def test_sweep_arrays_match_lengths(vs):
    r_grid = np.linspace(1.05, 1.5, 8)
    out = sweep_observer_radii(vs, r_grid, coupling=0.0, gamma_tau=0.5)
    assert out["r_observer"].shape == (8,)
    assert out["T_U"].shape == (8,)
    assert out["V_total"].shape == (8,)
    assert np.all(out["V_total"] >= 0.0)
    assert np.all(out["V_total"] <= 1.0)


# ----- intensity ----------------------------------------------------------


def test_intensity_at_zero_T_matches_pure_AB_fringes():
    y = np.linspace(-0.3, 0.3, 51)
    k, d, D_arm = 60.0, 0.3, 5.0
    phi_ab = 0.7
    I_unruh = unruh_ab_intensity(
        y, k, d, D_arm, phi_ab=phi_ab, coupling=0.0, T_U=0.0,
    )
    # Reproduces 2 + 2 cos(k dL + phi_ab) exactly.
    L_U = np.sqrt(D_arm * D_arm + d * d) + np.sqrt(D_arm * D_arm + (y - d) ** 2)
    L_L = np.sqrt(D_arm * D_arm + d * d) + np.sqrt(D_arm * D_arm + (y + d) ** 2)
    I_ref = 2.0 + 2.0 * np.cos(k * (L_U - L_L) + phi_ab)
    assert np.allclose(I_unruh, I_ref, atol=1e-12)


def test_intensity_at_high_T_is_flat():
    y = np.linspace(-0.5, 0.5, 51)
    I = unruh_ab_intensity(
        y, k=60.0, d=0.3, D_arm=5.0, phi_ab=0.7,
        coupling=0.0, T_U=1e6, gamma_tau=1.0,
    )
    assert np.allclose(I, 2.0, atol=1e-4)


def test_intensity_at_full_coupling_is_flat():
    y = np.linspace(-0.5, 0.5, 51)
    I = unruh_ab_intensity(
        y, k=60.0, d=0.3, D_arm=5.0, phi_ab=0.7,
        coupling=1.0, T_U=0.0,
    )
    assert np.allclose(I, 2.0, atol=1e-12)


def test_two_observers_see_different_visibility(vs):
    # Two observers at different orbit radii -> different T_U -> different V.
    T1 = observer_unruh_temperature(vs, r_observer=1.10)
    T2 = observer_unruh_temperature(vs, r_observer=1.40)
    assert T2 > T1
    V1 = total_visibility(0.0, T_U=T1, gamma_tau=0.5)
    V2 = total_visibility(0.0, T_U=T2, gamma_tau=0.5)
    assert V1 > V2
