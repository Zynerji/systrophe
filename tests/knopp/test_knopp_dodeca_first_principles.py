"""Tests for the first-principles audit (knopp_dodeca_first_principles)."""

import math

import numpy as np
import pytest

from systrophe.knopp.knopp_dodeca_first_principles import (
    collapse_dipole,
    coupling_spectrum,
    cusp_gap_profile,
    derive_all,
    face_point_crossover_gap,
    gap_for_target_enhancement,
    modes_above_threshold,
    pentagon_azimuthal_spectrum,
    pfa_face_energy,
    pfa_point_energy,
    pfa_point_energy_numeric,
    piston_coupling_closed_form,
    spectral_metrics,
    spiral_exponent,
    summarise_first_principles,
    sweep_storage_condition,
    swept_spiral_coupling,
    twisted_cusp_projection,
)


# ----- D1: PFA point vs face --------------------------------------------------


def test_pfa_point_closed_form_matches_quadrature():
    for d0, r_v in ((0.02, 0.05), (0.1, 0.02), (0.05, 0.1)):
        analytic = pfa_point_energy(d0, r_v)
        numeric = pfa_point_energy_numeric(d0, r_v)
        assert math.isclose(analytic, numeric, rel_tol=1e-4)


def test_face_dominates_point_below_crossover():
    area, r_v = 0.30, 0.05
    d_star = face_point_crossover_gap(area, r_v)
    assert pfa_face_energy(d_star / 10, area) > pfa_point_energy(d_star / 10, r_v)
    assert pfa_face_energy(d_star * 10, area) < pfa_point_energy(d_star * 10, r_v)


def test_pfa_validates_input():
    with pytest.raises(ValueError):
        pfa_point_energy(0.0, 1.0)
    with pytest.raises(ValueError):
        pfa_face_energy(0.1, -1.0)


# ----- D2: Fermat spiral + C5 selection ---------------------------------------


def test_twisted_cusp_projects_as_fermat_spiral():
    assert math.isclose(spiral_exponent(), 2.0, abs_tol=1e-9)
    y = np.linspace(0.1, 1.0, 50)
    rho, phi = twisted_cusp_projection(y, tau=0.8)
    assert np.allclose(rho, (phi / 0.8) ** 2 / (2 * 0.66))


def test_pentagon_selects_m_multiples_of_5():
    spec = pentagon_azimuthal_spectrum()
    assert int(np.argmax(spec[1:]) + 1) == 5
    # all power at m = 0 mod 5; off-pentagonal m suppressed
    off = [spec[m] for m in range(1, 25) if m % 5 != 0]
    assert max(off) < 1e-6


# ----- D3: enhancement is calibration, not derivation --------------------------


def test_cusp_gap_closes_at_contact_circle():
    R, h = 0.66, 0.20
    rho_star = R - math.sqrt(R ** 2 - h ** 2)
    assert cusp_gap_profile(np.array([rho_star]), h, R)[0] < 1e-12


def test_gap_for_x7_is_physical():
    g_c = gap_for_target_enhancement(7.0, h=0.20, area=0.30)
    assert 0.001 < g_c < 0.2   # a real, sub-face-size conformal gap exists


# ----- D4: coupling spectra ----------------------------------------------------


def test_piston_has_exact_nulls_every_fifth_mode():
    c = coupling_spectrum("piston")
    for n in (5, 10, 15, 20):
        assert c[n - 1] < 1e-3 * np.max(c)
        assert abs(piston_coupling_closed_form(n)) < 1e-12


def test_static_spiral_fills_piston_nulls():
    c_pi = coupling_spectrum("piston")
    c_sp = coupling_spectrum("spiral")
    assert spectral_metrics(c_sp)["min_coupling"] > \
        100 * spectral_metrics(c_pi)["min_coupling"]
    assert spectral_metrics(c_sp)["participation_ratio"] > \
        spectral_metrics(c_pi)["participation_ratio"]


def test_point_source_is_uniformly_weak():
    c_pt = coupling_spectrum("point")
    assert modes_above_threshold(c_pt, drive=1.0) == 0


def test_full_spectrum_needs_face_spiral_and_sweep():
    drive = 0.98
    assert modes_above_threshold(coupling_spectrum("piston"), drive) < 24
    assert modes_above_threshold(coupling_spectrum("spiral"), drive) < 24
    assert modes_above_threshold(swept_spiral_coupling(), drive) == 24


def test_sweep_storage_condition():
    assert sweep_storage_condition(t_sweep=2.0, Q=60.0)        # sqrt(60)/1.5 = 5.2
    assert not sweep_storage_condition(t_sweep=10.0, Q=60.0)
    with pytest.raises(ValueError):
        sweep_storage_condition(t_sweep=0.0, Q=60.0)


def test_coupling_spectrum_rejects_unknown_source():
    with pytest.raises(ValueError):
        coupling_spectrum("cube")


# ----- D5: collapse dipole ------------------------------------------------------


def test_pure_c5_field_has_no_dipole():
    assert abs(collapse_dipole(sat=1.0, eps=0.0)) < 1e-10
    assert abs(collapse_dipole(sat=1.0, eps=0.0, beta5=0.9)) < 1e-10


def test_dipole_equals_sat_times_eps():
    for sat, eps in ((0.9, 0.22), (0.5, 0.1), (1.0, 0.5)):
        assert math.isclose(collapse_dipole(sat, eps), sat * eps, rel_tol=1e-6)


# ----- report --------------------------------------------------------------------


def test_derive_all_report():
    r = derive_all()
    assert r.pfa_closed_form_error < 1e-4
    assert math.isclose(r.spiral_exponent, 2.0, abs_tol=1e-9)
    assert r.dominant_azimuthal_mode == 5
    assert r.piston_null_modes == 4
    assert r.point_modes_ringing == 0
    assert r.spiral_swept_modes_ringing == 24
    assert r.full_spectrum_needs_all_three
    assert r.storage_outlives_sweep
    assert abs(r.dipole_at_eps0) < 1e-10
    assert r.catcher_verdict in ("novel_structure", "smooth", "uniform")
    text = summarise_first_principles(r)
    for tag in ("D1 DERIVED", "D2 DERIVED", "D3 ASSUMED",
                "D4 DERIVED", "D5 CORRECTED"):
        assert tag in text
