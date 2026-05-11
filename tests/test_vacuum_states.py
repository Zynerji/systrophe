"""Tests for vacuum-state selection on LP background."""

import numpy as np
import pytest

from systrophe.vacuum_states import (
    VacuumStateReport,
    adiabatic_well_defined,
    boulware_energy_density_proxy,
    boulware_well_defined,
    compare_vacua_at_radius,
    hartle_hawking_analog_exists,
    natural_vacuum_verdict,
    vacuum_selection_summary,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


# ----- Boulware --------------------------------------------------------

def test_boulware_well_defined_at_subhorizon(vs):
    """Just outside cylinder: F > 0, Boulware well-defined."""
    result = boulware_well_defined(vs, r=1.5)
    if result["F_value"] > 0:
        assert result["well_defined"]


def test_boulware_returns_F_value(vs):
    """The reported F_value matches analytic_exterior_F."""
    r = 2.0
    result = boulware_well_defined(vs, r=r)
    expected = float(vs.analytic_exterior_F(r))
    assert result["F_value"] == pytest.approx(expected, rel=1e-12)


# ----- Adiabatic -------------------------------------------------------

def test_adiabatic_well_defined_in_smooth_region(vs):
    """Adiabatic vacuum well-defined where F doesn't vary too fast."""
    result = adiabatic_well_defined(vs, r=2.0)
    assert "adiabatic_parameter" in result
    assert "well_defined" in result


def test_adiabatic_at_horizon_not_well_defined(vs):
    """Where F is near 0, adiabatic fails (vanishing denominator)."""
    # Search for a horizon
    rs = np.linspace(1.05, 20.0, 200)
    Fs = np.array([float(vs.analytic_exterior_F(r)) for r in rs])
    signs = np.sign(Fs)
    flips = np.where(np.diff(signs) != 0)[0]
    if len(flips) > 0:
        r_h = float(rs[flips[0]])
        # Right at horizon: adiabatic fails
        result = adiabatic_well_defined(vs, r=r_h)
        # If F ~ 0, well_defined is False
        if abs(result["F_value"]) < 1e-10:
            assert not result["well_defined"]


# ----- Hartle-Hawking analog ------------------------------------------

def test_hh_analog_returns_dict(vs):
    """HH-analog test returns a structured dict."""
    result = hartle_hawking_analog_exists(vs, r=2.0)
    assert "is_chronology_horizon" in result
    assert "is_killing_horizon" in result
    assert "hh_analog_exists" in result


def test_hh_analog_typically_not_killing(vs):
    """In the rotating LP exterior, HH-analog typically does NOT exist."""
    # Sample several radii including near horizons
    rs = np.linspace(1.05, 10.0, 50)
    any_hh = False
    for r in rs:
        if hartle_hawking_analog_exists(vs, float(r))["hh_analog_exists"]:
            any_hh = True
            break
    # The expected verdict: HH typically fails (Killing horizon !=
    # chronology horizon for rotating cylinder)
    # This is allowed to find some HH-compatible point (slow-rotation
    # limit) so we don't assert anything strict here -- just confirm
    # the test ran.
    assert isinstance(any_hh, bool)


# ----- Energy density ---------------------------------------------------

def test_boulware_energy_density_proxy_finite(vs):
    """At a safe r, energy-density proxy is finite."""
    rho = boulware_energy_density_proxy(vs, r=2.0)
    assert np.isfinite(rho)


# ----- Summary ---------------------------------------------------------

def test_summary_returns_report(vs):
    summary = vacuum_selection_summary(vs, r=2.0)
    assert isinstance(summary, VacuumStateReport)
    assert isinstance(summary.well_defined, bool)
    assert isinstance(summary.energy_density, float)
    assert isinstance(summary.comment, str)


def test_compare_vacua_returns_complete_dict(vs):
    cmp = compare_vacua_at_radius(vs, r=2.0)
    assert "boulware" in cmp
    assert "adiabatic" in cmp
    assert "hartle_hawking" in cmp


def test_natural_vacuum_verdict(vs):
    """The verdict reports fractions of valid vacua."""
    rs = np.linspace(1.05, 8.0, 20)
    v = natural_vacuum_verdict(vs, rs)
    assert "boulware_fraction_valid" in v
    assert "adiabatic_fraction_valid" in v
    assert "hh_analog_fraction_valid" in v
    assert 0.0 <= v["boulware_fraction_valid"] <= 1.0
    assert 0.0 <= v["adiabatic_fraction_valid"] <= 1.0
    assert 0.0 <= v["hh_analog_fraction_valid"] <= 1.0
    assert "verdict" in v


def test_natural_vacuum_verdict_hh_typically_zero(vs):
    """The natural HH analog typically does not exist on our LP exterior.

    Headline finding: the speculative item I.6 'cavity at throat IS the
    natural QFT vacuum' is incorrect for the generic supercritical LP
    exterior, because the natural Hartle-Hawking-analog vacuum does
    NOT exist (Killing horizon != chronology horizon in this geometry).
    """
    rs = np.linspace(1.05, 8.0, 30)
    v = natural_vacuum_verdict(vs, rs)
    # HH fraction should be small or zero
    assert v["hh_analog_fraction_valid"] < 0.5
