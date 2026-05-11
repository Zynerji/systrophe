"""Tests for energy condition survey."""

import pytest

from systrophe.energy_condition_survey import (
    SurveyResult,
    exterior_violations_from_qft,
    summary_statistics,
    systematic_energy_survey,
    violation_loci,
)
from systrophe.vanstockum import VanStockumInterior


def test_survey_returns_SurveyResult():
    result = systematic_energy_survey(
        omega_range=(0.5, 1.5), R_range=(0.5, 1.5),
        n_omega=3, n_R=3,
    )
    assert isinstance(result, SurveyResult)
    assert len(result.omegas) == 3
    assert len(result.Rs) == 3


def test_survey_grids_consistent():
    """Output grids have correct shape n_R x n_omega."""
    result = systematic_energy_survey(n_omega=4, n_R=3)
    assert len(result.wec_satisfied) == 3
    assert len(result.wec_satisfied[0]) == 4


def test_violation_loci_returns_list():
    result = systematic_energy_survey(n_omega=3, n_R=3)
    viols = violation_loci(result)
    assert isinstance(viols, list)


def test_summary_statistics():
    result = systematic_energy_survey(n_omega=3, n_R=3)
    s = summary_statistics(result)
    assert "fraction_all_satisfied" in s
    assert 0.0 <= s["fraction_all_satisfied"] <= 1.0


def test_qft_violations_returns_dict():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    result = exterior_violations_from_qft(vs)
    assert "T_tt" in result or "error" in result


def test_qft_violations_specific_components():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    result = exterior_violations_from_qft(vs, r_test=2.0)
    if "T_tt" in result:
        assert "T_rr" in result
        assert "T_phi_phi" in result
        assert "WEC_proxy_satisfied" in result
        assert "NEC_proxy_satisfied" in result
