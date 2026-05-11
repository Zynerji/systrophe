"""Tests for modified dispersion module."""

import math

import pytest

from systrophe.modified_dispersion import (
    arrival_time_delay,
    birefringence_test,
    LP_threshold_modification,
    LV_constraint_from_GRB,
    lp_path_length_through_ctc_bands,
    lp_specific_birefringence_signature,
    modified_speed_of_light,
    vacuum_birefringence_in_supercritical_band,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_modified_speed_at_zero_energy_unity():
    v = modified_speed_of_light(0.0)
    assert v == 1.0


def test_modified_speed_decreases_with_xi_positive():
    """Positive xi (sub-luminal) means v < 1."""
    v = modified_speed_of_light(1e15, xi=1.0)  # GeV
    assert v < 1.0


def test_modified_speed_negligible_for_low_energy():
    """Low-energy photons: v ~ 1 to very high precision."""
    v = modified_speed_of_light(1.0)  # 1 GeV photons
    assert abs(v - 1.0) < 1e-15


def test_arrival_time_delay_positive_for_xi_positive():
    """Sub-luminal high-energy photons arrive later."""
    delay = arrival_time_delay(1.0, 1e15, path_length=1.0, xi=1.0)
    assert delay > 0


def test_arrival_time_zero_at_E_QG_infinity():
    """In limit E_QG -> infinity, no LV correction."""
    delay = arrival_time_delay(1.0, 1e15, path_length=1.0, E_QG=1e50)
    assert abs(delay) < 1e-30


def test_lp_path_length_positive(vs_super):
    L = lp_path_length_through_ctc_bands(vs_super, r_min=1.5, r_max=5.0)
    assert L > 0


def test_LP_threshold_modification_returns_dict():
    res = LP_threshold_modification(reaction_threshold=1e20)  # eV (UHECR scale)
    assert "threshold_LV" in res
    assert "relative_shift" in res


def test_birefringence_returns_dict():
    res = birefringence_test(energy=1e15)
    assert "v_left" in res
    assert "v_right" in res
    assert "birefringence_amount" in res


def test_birefringence_opposite_xi_gives_opposite_speed():
    res = birefringence_test(energy=1e15, xi_left=1.0, xi_right=-1.0)
    # v_left < 1, v_right > 1
    assert res["v_left"] < 1.0
    assert res["v_right"] > 1.0


def test_vacuum_birefringence_supercritical_returns_results(vs_super):
    res = vacuum_birefringence_in_supercritical_band(vs_super)
    assert res["regime"] == "supercritical"
    assert len(res["results"]) > 0


def test_vacuum_birefringence_subcritical_empty(vs_sub):
    res = vacuum_birefringence_in_supercritical_band(vs_sub)
    assert res["regime"] == "subcritical"


def test_LV_constraint_GRB_returns_xi_bound():
    res = LV_constraint_from_GRB(observed_delay_seconds=0.1,
                                  energy_high_eV=1e9, distance_Mpc=1000)
    assert "xi_upper_bound" in res
    assert res["xi_upper_bound"] > 0


def test_LP_signature_supercritical_finite(vs_super):
    res = lp_specific_birefringence_signature(vs_super)
    assert "max_birefringence" in res


def test_LP_signature_subcritical_zero(vs_sub):
    res = lp_specific_birefringence_signature(vs_sub)
    assert res["max_birefringence"] == 0.0
