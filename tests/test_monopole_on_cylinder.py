"""Tests for monopole on cylinder module."""

import math

import pytest

from systrophe.monopole_on_cylinder import (
    NUT_charge_analog,
    dirac_quantization_condition,
    monopole_cylinder_interaction_energy,
    monopole_field_at_r,
    novelty_scan,
    witten_effect,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_dirac_quantization_returns_list(vs):
    charges = dirac_quantization_condition(vs, n_max=4)
    assert len(charges) == 4
    assert all(c > 0 for c in charges)


def test_dirac_charges_are_integer_multiples_of_first(vs):
    """g_n / g_1 = n."""
    charges = dirac_quantization_condition(vs, n_max=4)
    g1 = charges[0]
    for n, gn in enumerate(charges, start=1):
        assert gn == pytest.approx(n * g1, rel=1e-12)


def test_monopole_field_decreases_with_r():
    B1 = monopole_field_at_r(1.0)
    B10 = monopole_field_at_r(10.0)
    assert B10 < B1


def test_monopole_field_inverse_square():
    """B(r) ~ 1/r^2."""
    B1 = monopole_field_at_r(1.0)
    B2 = monopole_field_at_r(2.0)
    assert B2 == pytest.approx(B1 / 4.0, rel=1e-12)


def test_interaction_energy_finite(vs):
    U = monopole_cylinder_interaction_energy(vs)
    assert math.isfinite(U)


def test_NUT_charge_returns_value(vs):
    Q = NUT_charge_analog(vs)
    assert math.isfinite(Q)


def test_witten_effect_returns_dict(vs):
    res = witten_effect(vs, theta_angle=math.pi)
    assert "effective_angular_momentum" in res


def test_witten_effect_zero_theta_zero(vs):
    res = witten_effect(vs, theta_angle=0.0)
    assert res["effective_angular_momentum"] == 0.0


def test_witten_effect_proportional_to_theta(vs):
    r1 = witten_effect(vs, theta_angle=1.0)
    r2 = witten_effect(vs, theta_angle=2.0)
    assert r2["effective_angular_momentum"] == pytest.approx(
        2 * r1["effective_angular_momentum"], rel=1e-12
    )


def test_novelty_scan_returns_verdict():
    res = novelty_scan(n_a_values=15)
    assert "verdict" in res
