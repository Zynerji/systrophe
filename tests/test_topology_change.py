"""Tests for topology change module."""

import math

import pytest

from systrophe.topology_change import (
    TopologyTransition,
    all_topology_transitions,
    ctc_band_merger_action,
    ctc_band_pinch_off_action,
    pair_extinction_topology_change,
    preferred_topology_change,
    topology_change_probability,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_pinch_off_action_returns_dict(vs_super):
    res = ctc_band_pinch_off_action(vs_super, band_index=0)
    assert res["transition_type"] == "pinch_off"
    assert "euclidean_action" in res


def test_pinch_off_unavailable_for_subcritical(vs_sub):
    res = ctc_band_pinch_off_action(vs_sub, band_index=0)
    assert res["available"] is False


def test_pinch_off_action_finite(vs_super):
    res = ctc_band_pinch_off_action(vs_super, band_index=0)
    assert math.isfinite(res["euclidean_action"])


def test_merger_action_returns_dict(vs_super):
    res = ctc_band_merger_action(vs_super, band_index=0)
    assert res["transition_type"] == "merger"


def test_merger_available_for_supercritical(vs_super):
    res = ctc_band_merger_action(vs_super, band_index=0)
    assert res["available"] is True


def test_topology_change_probability_in_unit():
    P = topology_change_probability(1.0)
    assert 0 <= P <= 1


def test_topology_change_probability_zero_for_infinite_action():
    P = topology_change_probability(float("inf"))
    assert P == 0.0


def test_topology_change_probability_unity_for_zero_action():
    P = topology_change_probability(0.0)
    assert P == 1.0


def test_preferred_returns_TopologyTransition(vs_super):
    t = preferred_topology_change(vs_super, n_bands=2)
    assert isinstance(t, TopologyTransition)


def test_preferred_subcritical_none(vs_sub):
    t = preferred_topology_change(vs_sub)
    assert t.transition_type == "none"


def test_preferred_has_min_action(vs_super):
    """preferred should match the min action in all_topology_transitions."""
    t = preferred_topology_change(vs_super, n_bands=3)
    all_t = all_topology_transitions(vs_super, n_bands=3)
    if all_t:
        assert t.euclidean_action == pytest.approx(all_t[0].euclidean_action, rel=1e-12)


def test_pair_extinction_at_pi_zero_action(vs_super):
    res = pair_extinction_topology_change(vs_super, delta=math.pi, band_index=0)
    if res.get("available", True):
        assert res["S_pair"] == pytest.approx(0.0, abs=1e-12)
        assert res["P_pair"] == 1.0


def test_pair_extinction_at_zero_no_change(vs_super):
    res = pair_extinction_topology_change(vs_super, delta=0.0, band_index=0)
    if res.get("available", True):
        assert res["S_pair"] == pytest.approx(res["S_single"], rel=1e-12)


def test_all_transitions_sorted(vs_super):
    ts = all_topology_transitions(vs_super, n_bands=3)
    for a, b in zip(ts[:-1], ts[1:]):
        assert a.euclidean_action <= b.euclidean_action + 1e-12


def test_all_transitions_subcritical_empty(vs_sub):
    ts = all_topology_transitions(vs_sub)
    assert ts == []
