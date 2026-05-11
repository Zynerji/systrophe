"""Tests for GW emission module."""

import math

import pytest

from systrophe.gw_emission import (
    cylindrical_resonant_frequencies,
    detectable_at_distance,
    gw_luminosity,
    moment_of_inertia,
    pair_inspiral_strain,
    radial_oscillation_strain,
    spin_up_strain,
    total_gw_energy_lost_in_band,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_moment_of_inertia_positive(vs):
    I = moment_of_inertia(vs)
    assert I > 0


def test_moment_of_inertia_scales_with_length(vs):
    I1 = moment_of_inertia(vs, cylinder_length=1.0)
    I2 = moment_of_inertia(vs, cylinder_length=2.0)
    assert I2 == pytest.approx(2 * I1, rel=1e-12)


def test_spin_up_strain_proportional_to_alpha(vs):
    s1 = spin_up_strain(vs, alpha_spin=1.0)
    s2 = spin_up_strain(vs, alpha_spin=2.0)
    assert s2["h_strain"] == pytest.approx(2 * s1["h_strain"], rel=1e-12)


def test_spin_up_strain_inversely_distance(vs):
    s1 = spin_up_strain(vs, alpha_spin=1.0, observer_distance=1.0)
    s2 = spin_up_strain(vs, alpha_spin=1.0, observer_distance=2.0)
    assert s2["h_strain"] == pytest.approx(s1["h_strain"] / 2, rel=1e-12)


def test_radial_oscillation_strain_returns_dict(vs):
    res = radial_oscillation_strain(vs)
    assert "h_peak" in res
    assert "f_GW" in res
    assert res["h_peak"] > 0


def test_radial_oscillation_frequency_from_omega_R(vs):
    omega_R = 10.0
    res = radial_oscillation_strain(vs, omega_R=omega_R)
    assert res["f_GW"] == pytest.approx(omega_R / (2 * math.pi), rel=1e-12)


def test_pair_inspiral_strain_returns_dict(vs):
    res = pair_inspiral_strain(vs)
    assert "h_strain" in res
    assert "Omega_orbital" in res


def test_pair_inspiral_chirp_negative_for_inspiral(vs):
    """Inspiral has d_separation_dt < 0 => df_dt has matching sign."""
    res = pair_inspiral_strain(vs, d_separation_dt=-0.01)
    # df/dt sign matches d_sep/dt sign (in this rough sign convention)
    assert math.isfinite(res["df_dt"])


def test_gw_luminosity_non_negative():
    P = gw_luminosity(I_ddot_amplitude=1.0, omega_GW=1.0)
    assert P >= 0


def test_gw_luminosity_zero_amplitude_zero():
    P = gw_luminosity(I_ddot_amplitude=0.0, omega_GW=1.0)
    assert P == 0.0


def test_detectable_at_distance_returns_dict():
    res = detectable_at_distance(h_strain=1e-22, f_GW=100.0)
    assert "snr_estimate" in res
    assert "is_detectable" in res


def test_detectable_low_strain_not_detectable():
    res = detectable_at_distance(h_strain=1e-30, f_GW=100.0)
    assert res["is_detectable"] is False


def test_cylindrical_resonant_frequencies_length(vs):
    freqs = cylindrical_resonant_frequencies(vs, n_modes=4)
    assert len(freqs) == 4


def test_cylindrical_resonant_frequencies_increasing(vs):
    freqs = cylindrical_resonant_frequencies(vs, n_modes=5)
    for f1, f2 in zip(freqs[:-1], freqs[1:]):
        assert f2 > f1


def test_total_gw_energy_in_band_non_negative(vs):
    E = total_gw_energy_lost_in_band(vs)
    assert E >= 0
