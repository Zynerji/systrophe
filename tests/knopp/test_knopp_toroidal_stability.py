"""Tests for the Toroidal Knopp binary stability + GW signature module."""

import math

import pytest

from systrophe.knopp.knopp_toroidal import EffectiveToroidalKerrBinary
from systrophe.knopp.knopp_toroidal_stability import (
    LIGO_BAND_HZ,
    LISA_BAND_HZ,
    PTA_BAND_HZ,
    ToroidalStabilityReport,
    corrected_merger_time,
    detector_classification,
    gw_frequency,
    gw_frequency_in_hz,
    gw_luminosity_quadrupole,
    gw_strain,
    in_detector_band,
    orbital_energy,
    orbital_frequency,
    reduced_mass,
    spin_spin_correction_fraction,
    spin_spin_energy,
    stability_report,
    summarise_stability,
    time_to_merger,
    time_to_merger_in_seconds,
    total_mass,
)


@pytest.fixture
def binary_tight():
    return EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.0)


@pytest.fixture
def binary_wide():
    return EffectiveToroidalKerrBinary(M=1.0, d=100.0, chi=1.0)


# ----- equal-mass formulas ------------------------------------------------


def test_reduced_mass_equal_mass(binary_tight):
    assert reduced_mass(binary_tight) == pytest.approx(0.5, rel=1e-12)


def test_total_mass_equal_mass(binary_tight):
    assert total_mass(binary_tight) == pytest.approx(2.0, rel=1e-12)


def test_orbital_frequency_kepler(binary_tight):
    # M_tot = 2, r = 2 -> Omega = sqrt(2 / 8) = 1/2.
    assert orbital_frequency(binary_tight) == pytest.approx(0.5, rel=1e-12)


def test_orbital_energy_newtonian(binary_tight):
    # E = -M^2 / (2d) = -1 / 4
    assert orbital_energy(binary_tight) == pytest.approx(-0.25, rel=1e-12)


# ----- GW emission --------------------------------------------------------


def test_gw_frequency_is_omega_over_pi(binary_tight):
    assert gw_frequency(binary_tight) == pytest.approx(
        orbital_frequency(binary_tight) / math.pi, rel=1e-12,
    )


def test_gw_luminosity_peters_equal_mass(binary_tight):
    # (64/5) M^5 / r^5 with M = 1, r = 2 -> 64/(5*32) = 0.4
    assert gw_luminosity_quadrupole(binary_tight) == pytest.approx(
        0.4, rel=1e-12,
    )


def test_gw_luminosity_drops_with_separation():
    b_tight = EffectiveToroidalKerrBinary(M=1.0, d=2.0)
    b_wide = EffectiveToroidalKerrBinary(M=1.0, d=20.0)
    L_tight = gw_luminosity_quadrupole(b_tight)
    L_wide = gw_luminosity_quadrupole(b_wide)
    # L ~ 1/r^5; r 10x larger -> luminosity ~ 1e-5 of tight
    assert L_wide / L_tight == pytest.approx(1e-5, rel=1e-6)


def test_time_to_merger_peters_equal_mass(binary_tight):
    # (5/256) d^4 / M^3 with M = 1, d = 2 -> (5*16)/256 = 0.3125
    assert time_to_merger(binary_tight) == pytest.approx(0.3125, rel=1e-12)


def test_time_to_merger_scales_as_d_to_the_fourth(binary_tight):
    b_2d = EffectiveToroidalKerrBinary(M=1.0, d=4.0, chi=1.0)
    t1 = time_to_merger(binary_tight)
    t2 = time_to_merger(b_2d)
    # d 2x -> t_merger 16x
    assert t2 / t1 == pytest.approx(16.0, rel=1e-12)


# ----- spin-spin correction ----------------------------------------------


def test_spin_spin_energy_attractive_for_antiparallel(binary_tight):
    # Antiparallel maximal: -2 chi^2 M^4 / r^3 = -2*1*1/8 = -0.25.
    assert spin_spin_energy(binary_tight) == pytest.approx(-0.25, rel=1e-12)


def test_spin_spin_fraction_nonperturbative_at_d_two(binary_tight):
    # At d = 2M tight binary, SS coupling is O(orbital).
    frac = spin_spin_correction_fraction(binary_tight)
    assert frac > 0.5


def test_spin_spin_fraction_perturbative_at_wide_separation(binary_wide):
    # At d = 100M, SS coupling is sub-percent.
    frac = spin_spin_correction_fraction(binary_wide)
    assert frac < 0.01


def test_corrected_merger_time_shorter_than_pure_peters(binary_tight):
    """The attractive SS coupling speeds up coalescence."""
    t_quad = time_to_merger(binary_tight)
    t_corr = corrected_merger_time(binary_tight)
    assert t_corr < t_quad


# ----- physical units -----------------------------------------------------


def test_gw_frequency_hz_scales_inverse_with_M_solar(binary_tight):
    # f_Hz = f_geom / (M_solar * GM_sun/c^3) -> linear inverse M_solar.
    f1 = gw_frequency_in_hz(binary_tight, M_solar=1.0)
    f10 = gw_frequency_in_hz(binary_tight, M_solar=10.0)
    assert f10 == pytest.approx(f1 / 10.0, rel=1e-12)


def test_gw_strain_inverse_distance(binary_tight):
    h_near = gw_strain(binary_tight, distance_m=1.0, M_solar=1.0)
    h_far = gw_strain(binary_tight, distance_m=100.0, M_solar=1.0)
    assert h_far == pytest.approx(h_near / 100.0, rel=1e-12)


def test_gw_strain_rejects_bad_distance(binary_tight):
    with pytest.raises(ValueError):
        gw_strain(binary_tight, distance_m=-1.0)


# ----- detector classification --------------------------------------------


def test_ligo_band_for_stellar_mass(binary_tight):
    # M_solar = 10 -> f_GW ~ 3.2 kHz -> in LIGO band
    assert in_detector_band(binary_tight, "LIGO", M_solar=10.0) is True


def test_lisa_band_for_supermassive(binary_tight):
    # M_solar = 1e6 -> f_GW ~ 3.2e-2 Hz -> in LISA band
    assert in_detector_band(binary_tight, "LISA", M_solar=1e6) is True


def test_pta_band_for_ultramassive(binary_tight):
    # M_solar = 1e11 -> f_GW ~ 3.2e-7 Hz -> in PTA band (1e-9, 1e-6)
    assert in_detector_band(binary_tight, "PTA", M_solar=1e11) is True


def test_out_of_band_for_solar_mass(binary_tight):
    # M_solar = 1 -> f_GW ~ 32 kHz, above LIGO
    assert detector_classification(binary_tight, M_solar=1.0) == "out-of-band"


def test_detector_band_constants():
    assert LIGO_BAND_HZ == (10.0, 5000.0)
    assert LISA_BAND_HZ == (1e-4, 1e-1)
    assert PTA_BAND_HZ == (1e-9, 1e-6)


# ----- combined report ----------------------------------------------------


def test_stability_report_dataclass(binary_tight):
    r = stability_report(binary_tight, M_solar=10.0)
    assert isinstance(r, ToroidalStabilityReport)
    assert r.has_band is True
    assert r.detector_band in ("LIGO", "LISA", "PTA", "out-of-band")


def test_stability_report_n_orbits_catastrophic_for_tight_binary(binary_tight):
    """The 'working' configuration M=1, d=2M merges in << 1 orbit --
    the dynamical falsification of the toroidal Knopp framework at this
    parameter point."""
    r = stability_report(binary_tight)
    assert r.band_lifetime_vs_ctc_window < 0.1


def test_stability_report_n_orbits_many_for_wide_binary(binary_wide):
    """Wide binaries (no CTC band) survive many orbits.

    At d=100M, equal-mass: n_orbits = t_merge / T_orb
    = (5/256 d^4 / M^3) / (2 pi d^(3/2) / sqrt(2M))
    = (5/256)(d^(5/2)/M^(5/2)) / (2 pi / sqrt(2))
    ~ 440 orbits  -- much more than the catastrophic tight binary.
    """
    r = stability_report(binary_wide)
    assert r.band_lifetime_vs_ctc_window > 100.0


def test_summary_string_well_formed(binary_tight):
    r = stability_report(binary_tight, M_solar=10.0)
    s = summarise_stability(r)
    assert "Orbital:" in s
    assert "GW signature:" in s
    assert "Lifetime:" in s
    assert "detector band" in s
