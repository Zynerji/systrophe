"""Energy-condition diagnostics tests."""

import numpy as np
import pytest

from systrophe import VanStockumInterior
from systrophe.qftcs.energy_conditions import (
    energy_condition_report,
    proper_energy_density,
    total_energy_per_unit_length,
)


def test_density_positive_everywhere():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r = np.linspace(0.0, vs.R, 200)
    rho = proper_energy_density(vs.omega, r)
    assert np.all(rho > 0.0)


def test_density_axis_value():
    """rho(0) = omega^2 / (2 pi)."""
    omega = 0.7
    rho_axis = float(proper_energy_density(omega, 0.0))
    assert rho_axis == pytest.approx(omega ** 2 / (2.0 * np.pi), rel=1e-12)


def test_density_grows_exponentially_with_r():
    """rho(r) / rho(0) = exp(omega^2 r^2)."""
    omega = 0.5
    rho0 = float(proper_energy_density(omega, 0.0))
    rho_R = float(proper_energy_density(omega, 1.0))
    expected_ratio = np.exp(omega ** 2)
    assert rho_R / rho0 == pytest.approx(expected_ratio, rel=1e-12)


def test_total_energy_finite_closed_form():
    """Total energy per unit z-length = 1/2 (exp(a^2) - 1)."""
    omega, R = 1.0, 1.0
    energy = total_energy_per_unit_length(omega, R)
    assert energy == pytest.approx(0.5 * (np.e - 1.0), rel=1e-12)


def test_total_energy_positive_for_any_omega_R():
    for omega, R in [(0.1, 1.0), (1.0, 1.0), (2.0, 0.5)]:
        assert total_energy_per_unit_length(omega, R) > 0.0


def test_van_stockum_satisfies_all_four_energy_conditions():
    """The headline result: NEC, WEC, SEC, DEC all hold for dust at any (omega, R)."""
    for (omega, R) in [(0.1, 1.0), (0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (2.5, 0.4)]:
        vs = VanStockumInterior(omega=omega, R=R)
        rep = energy_condition_report(vs)
        assert rep.nec_holds, f"NEC fails for omega={omega}, R={R}"
        assert rep.wec_holds, f"WEC fails for omega={omega}, R={R}"
        assert rep.sec_holds, f"SEC fails for omega={omega}, R={R}"
        assert rep.dec_holds, f"DEC fails for omega={omega}, R={R}"


def test_report_contains_radial_density_array():
    """Report exposes the rho(r) profile for plotting / inspection."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    rep = energy_condition_report(vs)
    assert rep.rho.shape == rep.r_grid.shape
    # rho(0) should equal omega^2 / (2 pi)
    assert rep.rho[0] == pytest.approx(1.0 / (2.0 * np.pi), rel=1e-12)


def test_sec_residual_at_least_one_half_rho():
    """SEC residual rho * ((u.t)^2 - 1/2) >= rho/2 at the worst case t = u."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    rep = energy_condition_report(vs)
    rho_min = float(rep.rho.min())
    assert rep.sec_min_residual >= 0.5 * rho_min - 1e-12


def test_supercritical_a_gt_1_still_satisfies_energy_conditions():
    """Even in the interior CTC regime (a > 1), the dust source is healthy."""
    vs = VanStockumInterior(omega=2.0, R=1.0)  # a = 2; interior CTC shell
    assert vs.interior_regime == "supercritical"
    rep = energy_condition_report(vs)
    assert rep.nec_holds and rep.wec_holds and rep.sec_holds and rep.dec_holds
