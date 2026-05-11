"""Tests for vacuum polarization module."""

import math

import pytest

from systrophe.vacuum_polarization import (
    effective_fine_structure_running,
    heisenberg_euler_shift,
    lp_modified_E_critical,
    novelty_scan,
    pair_production_threshold,
    vacuum_polarization_at_r,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_heisenberg_euler_shift_non_negative():
    delta = heisenberg_euler_shift(electric_field=1.0)
    assert delta >= 0


def test_heisenberg_euler_quartic_scaling():
    delta1 = heisenberg_euler_shift(electric_field=1.0)
    delta2 = heisenberg_euler_shift(electric_field=2.0)
    assert delta2 == pytest.approx(16 * delta1, rel=1e-12)


def test_fine_structure_running_increases():
    alpha_low = effective_fine_structure_running(energy_scale=5e5)
    alpha_high = effective_fine_structure_running(energy_scale=1e10)
    assert alpha_high > alpha_low


def test_pair_production_below_threshold_exp_suppressed():
    res = pair_production_threshold(electric_field=1e-3 * 5.11e5 ** 2)
    assert res["regime"] == "below_threshold"


def test_pair_production_above_threshold():
    res = pair_production_threshold(electric_field=10 * 5.11e5 ** 2)
    assert res["regime"] == "above_threshold"


def test_lp_modified_E_critical_finite(vs):
    Ec = lp_modified_E_critical(vs, r=1.5)
    assert math.isfinite(Ec)
    assert Ec > 0


def test_vacuum_polarization_at_r_returns_dict(vs):
    res = vacuum_polarization_at_r(vs, r=1.5)
    assert "pi" in res
    assert "regime" in res


def test_vacuum_polarization_finite_in_exterior(vs):
    res = vacuum_polarization_at_r(vs, r=1.5)
    assert math.isfinite(res["pi"])


def test_novelty_scan_returns_verdict():
    res = novelty_scan(n_radii=10)
    assert "verdict" in res
