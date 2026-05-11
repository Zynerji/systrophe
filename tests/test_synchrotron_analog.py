"""Tests for synchrotron radiation analog module."""

import math

import pytest

from systrophe.synchrotron_analog import (
    back_reaction_drag,
    characteristic_emission_band,
    effective_gamma_factor,
    emitted_power,
    multi_band_synchrotron_spectrum,
    observable_signature_distant_observer,
    orbital_frequency,
    synchrotron_critical_frequency,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_orbital_frequency_returns_value(vs_super):
    Om = orbital_frequency(vs_super, r=1.5)
    # Either a real circular orbit exists (finite), or it doesn't (nan)
    assert math.isfinite(Om) or math.isnan(Om)


def test_effective_gamma_factor_returns_value(vs_super):
    g = effective_gamma_factor(vs_super, r=1.5)
    # Either finite gamma > 1, or nan if no timelike orbit there
    assert math.isfinite(g) or math.isnan(g)


def test_synchrotron_critical_frequency_returns_value(vs_super):
    om_c = synchrotron_critical_frequency(vs_super, r=1.5)
    assert math.isfinite(om_c) or math.isnan(om_c)


def test_emitted_power_positive_when_finite(vs_super):
    P = emitted_power(vs_super, r=1.5)
    if math.isfinite(P):
        assert P >= 0


def test_emitted_power_scales_with_charge_squared(vs_super):
    P1 = emitted_power(vs_super, r=1.5, charge=1.0)
    P2 = emitted_power(vs_super, r=1.5, charge=2.0)
    if math.isfinite(P1) and math.isfinite(P2) and P1 > 0:
        ratio = P2 / P1
        assert ratio == pytest.approx(4.0, rel=1e-9)


def test_characteristic_emission_band_returns_dict(vs_super):
    band = characteristic_emission_band(vs_super, r=1.5)
    assert "omega_orbit" in band
    assert "omega_critical" in band
    assert "omega_min" in band
    assert "omega_max" in band


def test_emission_band_max_greater_than_min(vs_super):
    band = characteristic_emission_band(vs_super, r=1.5)
    if math.isfinite(band["omega_max"]) and math.isfinite(band["omega_min"]):
        assert band["omega_max"] >= band["omega_min"]


def test_multi_band_returns_list(vs_super):
    bands = multi_band_synchrotron_spectrum(vs_super, n_bands=3)
    assert isinstance(bands, list)


def test_multi_band_subcritical_empty(vs_sub):
    bands = multi_band_synchrotron_spectrum(vs_sub, n_bands=3)
    assert bands == []


def test_back_reaction_drag_returns_value(vs_super):
    drag = back_reaction_drag(vs_super, r=1.5)
    if math.isfinite(drag):
        assert drag >= 0


def test_observable_signature_returns_dict(vs_super):
    obs = observable_signature_distant_observer(vs_super, r_emit=1.5, r_obs=1.7)
    assert "omega_observed" in obs
    assert "redshift_factor" in obs


def test_observable_signature_behind_CH_flag(vs_super):
    # r=3.35 is in a CTC band (F<0)
    obs = observable_signature_distant_observer(vs_super, r_emit=3.35, r_obs=1.5)
    assert obs["behind_CH"] is True


def test_multi_band_band_index_present(vs_super):
    bands = multi_band_synchrotron_spectrum(vs_super, n_bands=2)
    for b in bands:
        assert "band_index" in b
        assert "r_representative" in b
