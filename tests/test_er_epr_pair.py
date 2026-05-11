"""Tests for ER=EPR pair module."""

import math

import pytest

from systrophe.er_epr_pair import (
    ERPairData,
    bell_pair_fidelity_at_delta,
    bridge_strength,
    entanglement_entropy_pair,
    er_epr_consistency_check,
    mutual_information_proxy,
    pair_extinction_decouples_systems,
    predict_bell_state_amplitudes,
    schmidt_decomposition_consistency,
    wormhole_throat_area,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_bridge_strength_at_zero_equals_one():
    assert bridge_strength(0.0) == 1.0


def test_bridge_strength_at_pi_equals_zero():
    assert bridge_strength(math.pi) == pytest.approx(0.0, abs=1e-12)


def test_bridge_strength_at_half_pi_equals_half():
    assert bridge_strength(math.pi / 2) == pytest.approx(0.5, rel=1e-12)


def test_bell_fidelity_at_zero():
    F = bell_pair_fidelity_at_delta(0.0)
    assert F == 1.0


def test_bell_fidelity_at_pi():
    F = bell_pair_fidelity_at_delta(math.pi)
    assert F == 0.5  # Pure product state has 0.5 fidelity with Bell pair


def test_mutual_information_zero_at_pi():
    """No entanglement at extinction -> I = 0."""
    I = mutual_information_proxy(math.pi)
    assert I == pytest.approx(0.0, abs=1e-12)


def test_mutual_information_zero_at_zero():
    """At delta=0, bs=1 (degenerate Schmidt rank=1), MI = 0 (per our convention)."""
    I = mutual_information_proxy(0.0)
    assert I == pytest.approx(0.0, abs=1e-12)


def test_mutual_information_peaks_at_half_pi():
    """Maximal mixing happens at bs = 1/2 -> delta = pi/2."""
    I_pi2 = mutual_information_proxy(math.pi / 2)
    I_pi4 = mutual_information_proxy(math.pi / 4)
    assert I_pi2 > I_pi4


def test_entanglement_entropy_zero_at_pi():
    S = entanglement_entropy_pair(math.pi)
    assert S == pytest.approx(0.0, abs=1e-12)


def test_entanglement_entropy_max_at_half_pi():
    """Schmidt rank 2 with equal coefficients -> S = 1 bit."""
    S = entanglement_entropy_pair(math.pi / 2)
    assert S == pytest.approx(1.0, rel=1e-9)


def test_wormhole_throat_at_pi_zero():
    A = wormhole_throat_area(math.pi)
    assert A == pytest.approx(0.0, abs=1e-12)


def test_wormhole_throat_at_zero_full():
    A = wormhole_throat_area(0.0, R_cylinder=1.0)
    assert A == pytest.approx(math.pi, rel=1e-12)


def test_er_epr_consistency_returns_list(vs):
    res = er_epr_consistency_check(vs)
    assert isinstance(res, list)
    assert all(isinstance(r, ERPairData) for r in res)


def test_er_epr_consistency_default_5_samples(vs):
    res = er_epr_consistency_check(vs)
    assert len(res) == 5  # default: 5 delta values


def test_pair_extinction_decouples_at_pi():
    assert pair_extinction_decouples_systems(math.pi) is True


def test_pair_extinction_does_not_decouple_at_zero():
    assert pair_extinction_decouples_systems(0.0) is False


def test_predict_bell_amplitudes_normalized():
    """|a_00|^2 + |a_01|^2 + |a_10|^2 + |a_11|^2 = 1 for all delta."""
    for d in [0.0, math.pi / 4, math.pi / 2, math.pi]:
        a00, a01, a10, a11 = predict_bell_state_amplitudes(d)
        norm = abs(a00)**2 + abs(a01)**2 + abs(a10)**2 + abs(a11)**2
        assert norm == pytest.approx(1.0, rel=1e-9)


def test_schmidt_rank_one_at_pi():
    res = schmidt_decomposition_consistency(math.pi)
    assert res["schmidt_rank"] == 1
    assert res["is_separable"] is True


def test_schmidt_max_entangled_at_pi_half():
    res = schmidt_decomposition_consistency(math.pi / 2)
    assert res["is_maximally_entangled"] is True


def test_novelty_scan_returns_dict():
    from systrophe.er_epr_pair import novelty_scan
    res = novelty_scan(n_deltas=15)
    assert "verdict" in res
    assert "sharp_features" in res
    assert res["verdict"] in ("uniform", "smooth", "novel_structure")


def test_novelty_scan_finds_no_uniformity():
    """The delta scan IS varying (bridge_strength changes), so verdict
    should not be 'uniform'."""
    from systrophe.er_epr_pair import novelty_scan
    res = novelty_scan(n_deltas=10)
    assert res["verdict"] != "uniform"
