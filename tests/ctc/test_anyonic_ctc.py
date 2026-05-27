"""Tests for anyonic CTC module."""

import math

import pytest

from systrophe.ctc.anyonic_ctc import (
    AnyonicBand,
    all_band_anyon_data,
    braid_phase_at_band,
    fibonacci_anyon_dimension,
    fusion_rules,
    novelty_scan,
    pair_extinction_topological_order,
    quantum_dimension_for_band,
    topological_entanglement_entropy,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_braid_phase_proportional_to_band():
    theta1 = braid_phase_at_band(1, alpha=1.732)
    theta2 = braid_phase_at_band(2, alpha=1.732)
    assert theta2 == pytest.approx(2 * theta1, rel=1e-12)


def test_quantum_dimension_abelian_is_one():
    d = quantum_dimension_for_band(1, alpha=1.732, fibonacci=False)
    assert d == 1.0


def test_quantum_dimension_fibonacci_is_phi():
    d = quantum_dimension_for_band(1, alpha=1.732, fibonacci=True)
    phi = (1 + math.sqrt(5)) / 2
    assert d == pytest.approx(phi, rel=1e-12)


def test_topological_entropy_abelian_log_sqrt_n():
    """For n abelian Z anyons, D = sqrt(n), S = (1/2) log n."""
    S = topological_entanglement_entropy([1, 2, 3], alpha=1.732)
    expected = 0.5 * math.log(3)
    assert S == pytest.approx(expected, rel=1e-12)


def test_topological_entropy_fibonacci_uses_phi():
    """With phi quantum dim, S should be larger than abelian."""
    S_abel = topological_entanglement_entropy([1, 2, 3], alpha=1.732)
    S_fib = topological_entanglement_entropy([1, 2, 3], alpha=1.732, fibonacci=True)
    assert S_fib > S_abel


def test_fibonacci_dimension_golden():
    d = fibonacci_anyon_dimension()
    phi = (1 + math.sqrt(5)) / 2
    assert d == pytest.approx(phi, rel=1e-12)


def test_fusion_bosonic_at_zero():
    res = fusion_rules(0.0, 0.0)
    assert res["is_bosonic"] is True


def test_fusion_fermionic_at_pi():
    res = fusion_rules(math.pi / 2, math.pi / 2)
    assert res["is_fermionic"] is True


def test_pair_extinction_at_pi_zero_phase():
    res = pair_extinction_topological_order(1, alpha=1.732, delta=math.pi)
    assert res["theta_pair"] == 0.0


def test_pair_extinction_order_destroyed_flag():
    res = pair_extinction_topological_order(1, alpha=1.732, delta=math.pi)
    assert res["order_destroyed_at_pi"] is True


def test_all_band_anyon_data_supercritical(vs_super):
    bands = all_band_anyon_data(vs_super, n_bands=3)
    assert len(bands) == 3
    assert all(isinstance(b, AnyonicBand) for b in bands)


def test_all_band_anyon_data_subcritical_empty(vs_sub):
    bands = all_band_anyon_data(vs_sub)
    assert bands == []


def test_novelty_scan_returns_verdict():
    res = novelty_scan(n_band_max=8)
    assert "verdict" in res
