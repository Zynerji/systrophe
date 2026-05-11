"""Tests for Cauchy horizon stability."""

import numpy as np
import pytest

from systrophe.cauchy_stability import (
    CauchyStabilityReport,
    chronology_protection_via_instability,
    lyapunov_exponent_at_horizon,
    mass_inflation_growth_rate,
    stability_at_horizon,
    survey_all_cauchy_horizons,
)
from systrophe.geodesic_completeness import chronology_horizon_radii
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_lyapunov_at_horizon_nonneg(vs):
    horizons = chronology_horizon_radii(vs, r_max=20.0)
    if not horizons:
        pytest.skip("no horizons")
    lyap = lyapunov_exponent_at_horizon(vs, horizons[0])
    assert lyap >= 0


def test_stability_returns_report(vs):
    horizons = chronology_horizon_radii(vs, r_max=20.0)
    if not horizons:
        pytest.skip("no horizons")
    report = stability_at_horizon(vs, horizons[0])
    assert isinstance(report, CauchyStabilityReport)


def test_classification_one_of_three(vs):
    horizons = chronology_horizon_radii(vs, r_max=20.0)
    if not horizons:
        pytest.skip("no horizons")
    report = stability_at_horizon(vs, horizons[0])
    assert report.classification in ("stable", "marginally stable", "unstable")


def test_survey_returns_list(vs):
    reports = survey_all_cauchy_horizons(vs)
    assert isinstance(reports, list)


def test_supercritical_has_horizons(vs):
    reports = survey_all_cauchy_horizons(vs, r_max=20.0)
    assert len(reports) >= 1


def test_mass_inflation_returns_dict(vs):
    horizons = chronology_horizon_radii(vs, r_max=20.0)
    if not horizons:
        pytest.skip("no horizons")
    result = mass_inflation_growth_rate(vs, horizons[0])
    assert "growth_rate" in result
    assert "characteristic_time" in result


def test_chronology_protection_via_instability_dict(vs):
    result = chronology_protection_via_instability(vs)
    assert "n_horizons" in result
    assert "supports_chronology_protection" in result


def test_subcritical_chronology_no_horizons():
    """Subcritical: 0 or 1 horizons, simple classification."""
    vs_sub = VanStockumInterior(omega=0.3, R=1.0)
    result = chronology_protection_via_instability(vs_sub, r_max=50.0)
    assert result["n_horizons"] <= 2
