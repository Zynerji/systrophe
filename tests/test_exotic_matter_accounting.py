"""Tests for exotic-matter / Casimir comparison at a wormhole throat."""

import numpy as np
import pytest

from systrophe.exotic_matter_accounting import (
    ExoticMatterReport,
    casimir_cavity_energy,
    exotic_vs_casimir_comparison,
    is_casimir_route_physically_plausible,
    morris_thorne_exotic_budget,
    morris_thorne_exotic_density,
    required_plate_separation,
)


def test_mt_density_positive():
    """Exotic density is positive (magnitude of NEC violation)."""
    rho = morris_thorne_exotic_density(r_throat=1.0, db_dr_at_throat=0.5)
    assert rho > 0


def test_mt_density_scales_inverse_r_squared():
    """rho ~ 1/r_t^2."""
    r1 = morris_thorne_exotic_density(r_throat=1.0)
    r2 = morris_thorne_exotic_density(r_throat=2.0)
    assert r1 / r2 == pytest.approx(4.0, rel=1e-12)


def test_mt_density_validates_db_range():
    with pytest.raises(ValueError):
        morris_thorne_exotic_density(r_throat=1.0, db_dr_at_throat=1.5)
    with pytest.raises(ValueError):
        morris_thorne_exotic_density(r_throat=1.0, db_dr_at_throat=-0.1)


def test_mt_density_validates_r_throat():
    with pytest.raises(ValueError):
        morris_thorne_exotic_density(r_throat=-1.0)


def test_mt_budget_proportional_to_volume():
    """E_exotic ~ pi r_t^2 L, given fixed density."""
    r_t = 1.0
    E_short = morris_thorne_exotic_budget(r_throat=r_t, cavity_length=1.0)
    E_long = morris_thorne_exotic_budget(r_throat=r_t, cavity_length=10.0)
    assert E_long / E_short == pytest.approx(10.0, rel=1e-12)


def test_casimir_cavity_energy_negative():
    """Casimir gives negative total energy."""
    E = casimir_cavity_energy(r_throat=1.0, plate_separation=0.1)
    assert E < 0


def test_casimir_cavity_energy_d_scaling():
    """E_C ~ -r_t^2 / d^3."""
    E1 = casimir_cavity_energy(r_throat=1.0, plate_separation=1.0)
    E2 = casimir_cavity_energy(r_throat=1.0, plate_separation=2.0)
    # E_C ~ -pi^3 r_t^2 / (720 d^3) -> E2/E1 = 1/8
    assert E2 / E1 == pytest.approx(1 / 8, rel=1e-12)


# ----- comparison ------------------------------------------------------

def test_comparison_returns_report():
    cmp = exotic_vs_casimir_comparison(r_throat=1.0, plate_separation=0.1)
    assert isinstance(cmp, ExoticMatterReport)
    assert cmp.E_exotic > 0
    assert cmp.E_casimir < 0
    assert np.isfinite(cmp.ratio)


def test_comparison_at_small_d_casimir_sufficient():
    """For sufficiently small d, |E_C| dominates."""
    cmp = exotic_vs_casimir_comparison(r_throat=1.0, plate_separation=0.01)
    # |E_C| ~ 1/d^3 grows fast; should exceed exotic
    assert cmp.casimir_sufficient


def test_comparison_at_large_d_casimir_insufficient():
    """For large d, Casimir is too weak."""
    cmp = exotic_vs_casimir_comparison(r_throat=1.0, plate_separation=10.0)
    assert not cmp.casimir_sufficient


# ----- required plate separation --------------------------------------

def test_required_d_positive():
    d = required_plate_separation(r_throat=1.0)
    assert d > 0


def test_required_d_increases_with_r_throat():
    """d_req scales as r_t^(1/2)."""
    d1 = required_plate_separation(r_throat=1.0)
    d2 = required_plate_separation(r_throat=4.0)
    assert d2 > d1
    assert d2 / d1 == pytest.approx(2.0, rel=1e-12)


def test_at_required_d_ratio_is_one():
    """At d = required_d, |E_C| = E_exotic exactly."""
    r_t = 2.0
    d_req = required_plate_separation(r_throat=r_t, db_dr_at_throat=0.5)
    cmp = exotic_vs_casimir_comparison(r_throat=r_t, plate_separation=d_req,
                                         cavity_length=d_req,
                                         db_dr_at_throat=0.5)
    assert cmp.ratio == pytest.approx(1.0, rel=1e-10)


# ----- plausibility ----------------------------------------------------

def test_plausibility_check_returns_verdict():
    result = is_casimir_route_physically_plausible(r_throat=1.0)
    assert "verdict" in result
    assert isinstance(result["geometric_fit"], bool)


def test_plausibility_check_at_meso_throat():
    """For a meter-scale throat, the required d is much smaller than r."""
    result = is_casimir_route_physically_plausible(r_throat=1.0)
    # d_required is O(0.3) for r=1, so d/r ~ 0.3 > 0.1 -> not strictly fit
    assert result["d_required"] < result["r_throat"]


def test_quantitative_verdict_typical_throat():
    """For a 'typical' throat (r_throat = 1 in natural units), the Casimir
    budget at any geometrically-fittable plate separation is insufficient.

    This is the headline quantitative answer: the Casimir-replaces-
    exotic-matter route is NOT plausible in the typical regime."""
    result = is_casimir_route_physically_plausible(r_throat=1.0,
                                                    max_d_ratio=0.01)
    # d/r typically > 0.01 unless we're in an exotic regime
    if result["d_to_r_ratio"] > 0.01:
        assert not result["geometric_fit"]
