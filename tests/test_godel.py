"""Goedel rotating-dust universe tests."""

import numpy as np
import pytest

from systrophe.spacetimes import GodelUniverse, godel_ctc_radius


def test_ctc_radius_is_arcsinh_1():
    """The CTC threshold is r = arcsinh(1) = ln(1 + sqrt(2))."""
    expected = float(np.arcsinh(1.0))
    assert godel_ctc_radius() == pytest.approx(expected, rel=1e-12)
    assert godel_ctc_radius() == pytest.approx(np.log(1.0 + np.sqrt(2.0)), rel=1e-12)


def test_construction_requires_positive_a():
    GodelUniverse(a=1.0)
    with pytest.raises(ValueError):
        GodelUniverse(a=0.0)
    with pytest.raises(ValueError):
        GodelUniverse(a=-1.0)


def test_metric_components_at_r_zero():
    """At the rotation axis r = 0 the metric reduces to Minkowski-like (a^2 prefactor)."""
    g = GodelUniverse(a=1.5)
    assert g.gtt(0.0) == pytest.approx(-1.5 ** 2)
    assert g.gtphi(0.0) == pytest.approx(0.0)
    assert g.gphiphi(0.0) == pytest.approx(0.0)
    assert g.grr(0.0) == pytest.approx(1.5 ** 2)
    assert g.gzz(0.0) == pytest.approx(1.5 ** 2)


def test_gphiphi_sign_changes_at_ctc_threshold():
    """g_{phi phi} > 0 below r_CTC and < 0 above."""
    g = GodelUniverse(a=1.0)
    r_ctc = g.ctc_threshold_radius
    assert g.gphiphi(0.5 * r_ctc) > 0
    assert g.gphiphi(2.0 * r_ctc) < 0
    # Exact zero at the threshold
    assert g.gphiphi(r_ctc) == pytest.approx(0.0, abs=1e-12)


def test_local_ctc_detection():
    g = GodelUniverse(a=1.0)
    assert not g.has_local_ctc(r=0.5)
    assert g.has_local_ctc(r=1.5)


def test_dust_density_and_cosmological_constant():
    """rho = 1/(8 pi a^2), Lambda = -1/(2 a^2)."""
    g = GodelUniverse(a=2.0)
    assert g.dust_density == pytest.approx(1.0 / (8.0 * np.pi * 4.0), rel=1e-12)
    assert g.cosmological_constant == pytest.approx(-1.0 / 8.0, rel=1e-12)


def test_angular_velocity_inverse_of_a():
    g = GodelUniverse(a=2.0)
    assert g.angular_velocity == pytest.approx(0.5)


def test_find_ctc_radial_interval():
    g = GodelUniverse(a=1.0)
    interval = g.find_ctc_radial_interval(r_max=3.0)
    assert interval is not None
    r_in, r_out = interval
    assert r_in == pytest.approx(g.ctc_threshold_radius)
    assert r_out == pytest.approx(3.0)


def test_no_ctc_region_below_threshold():
    g = GodelUniverse(a=1.0)
    assert g.find_ctc_radial_interval(r_max=0.5) is None


def test_proper_phi_circumference_zero_at_threshold():
    """sqrt(|g_phiphi|) = 0 exactly at r = r_CTC."""
    g = GodelUniverse(a=1.0)
    assert g.proper_phi_circumference(g.ctc_threshold_radius) == pytest.approx(0.0, abs=1e-10)
