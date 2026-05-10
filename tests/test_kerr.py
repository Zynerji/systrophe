"""Kerr spacetime tests."""

import numpy as np
import pytest

from systrophe.spacetimes import GodelUniverse  # to ensure spacetimes package
from systrophe.spacetimes.kerr import KerrSpacetime


def test_construction_validates_M_positive():
    KerrSpacetime(M=1.0, a=0.5)
    with pytest.raises(ValueError):
        KerrSpacetime(M=0.0, a=0.5)
    with pytest.raises(ValueError):
        KerrSpacetime(M=-1.0, a=0.5)


def test_subextremal_horizons():
    """For a < M: horizons at r_+- = M +- sqrt(M^2 - a^2)."""
    M, a = 1.0, 0.5
    k = KerrSpacetime(M=M, a=a)
    horizons = k.event_horizons
    assert horizons is not None
    r_plus, r_minus = horizons
    expected_plus = M + np.sqrt(M ** 2 - a ** 2)
    expected_minus = M - np.sqrt(M ** 2 - a ** 2)
    assert r_plus == pytest.approx(expected_plus)
    assert r_minus == pytest.approx(expected_minus)
    assert not k.is_overextremal
    assert not k.is_extremal


def test_overextremal_no_horizons():
    """For a > M: no horizons (naked singularity)."""
    k = KerrSpacetime(M=1.0, a=2.0)
    assert k.event_horizons is None
    assert k.is_overextremal


def test_extremal():
    k = KerrSpacetime(M=1.0, a=1.0)
    assert k.is_extremal
    horizons = k.event_horizons
    assert horizons is not None
    r_plus, r_minus = horizons
    assert r_plus == pytest.approx(1.0)
    assert r_minus == pytest.approx(1.0)


def test_schwarzschild_limit_no_ctc():
    """a = 0 -> Schwarzschild; no CTCs anywhere."""
    k = KerrSpacetime(M=1.0, a=0.0)
    assert k.schwarzschild_limit()
    # Test at multiple (r, theta)
    for r in [-2.0, -0.5, 0.5, 2.0, 10.0]:
        for theta in [0.5, np.pi / 2, np.pi - 0.5]:
            assert k.gphiphi(r, theta) >= 0
            assert not k.has_local_ctc(r, theta)


def test_no_ctc_for_positive_r():
    """For r > 0, g_phiphi >= 0 everywhere (in any sub-/super-extremal case)."""
    for a in [0.3, 1.0, 1.5]:
        k = KerrSpacetime(M=1.0, a=a)
        for r in [0.5, 1.0, 2.0, 10.0]:
            for theta in [0.4, np.pi / 2, np.pi - 0.4]:
                assert k.gphiphi(r, theta) >= 0, f"a={a}, r={r}, theta={theta}"


def test_ctc_in_negative_r_region_close_to_ring():
    """CTCs appear in the equatorial r < 0 region close to the ring (r -> 0^-)."""
    k = KerrSpacetime(M=1.0, a=0.9)
    # Close to the ring singularity (small |r|): CTCs
    assert k.has_local_ctc(r=-0.5, theta=np.pi / 2)
    assert k.has_local_ctc(r=-0.1, theta=np.pi / 2)
    # Far negative r: g_phiphi grows as r^2; no CTC
    assert not k.has_local_ctc(r=-3.0, theta=np.pi / 2)
    assert not k.has_local_ctc(r=-5.0, theta=np.pi / 2)


def test_equatorial_ctc_threshold_is_real_negative_root():
    """The equatorial CTC threshold solves r^3 + a^2 r + 2 M a^2 = 0."""
    k = KerrSpacetime(M=1.0, a=0.5)
    r_ctc = k.equatorial_ctc_radius()
    assert r_ctc is not None
    assert r_ctc < 0
    # Verify it satisfies the cubic
    residual = r_ctc ** 3 + (0.5 ** 2) * r_ctc + 2.0 * (0.5 ** 2)
    assert abs(residual) < 1e-9


def test_equatorial_threshold_separates_CTC_and_no_CTC():
    """Above the threshold (closer to ring), CTC; below threshold, no CTC.

    Convention: r_CTC < 0 is the largest negative root. CTC region is
    (r_CTC, 0); for r < r_CTC the spacetime is normal.
    """
    k = KerrSpacetime(M=1.0, a=0.5)
    r_ctc = k.equatorial_ctc_radius()
    eps = 0.01
    # Closer to ring (r > r_ctc, still negative): CTC holds
    assert k.has_local_ctc(r=r_ctc + eps, theta=np.pi / 2)
    # Farther from ring (r < r_ctc): no CTC
    assert not k.has_local_ctc(r=r_ctc - eps, theta=np.pi / 2)


def test_schwarzschild_no_equatorial_ctc():
    """Schwarzschild (a=0) has no real negative root; method returns None."""
    k = KerrSpacetime(M=1.0, a=0.0)
    assert k.equatorial_ctc_radius() is None


def test_metric_components_finite_off_singularity():
    """At any (r, theta) with Sigma > 0, metric components are finite."""
    k = KerrSpacetime(M=1.0, a=0.5)
    for r in [-2.0, -0.5, 0.5, 2.0]:
        for theta in [np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            assert np.isfinite(k.gtt(r, theta))
            assert np.isfinite(k.gtphi(r, theta))
            assert np.isfinite(k.gphiphi(r, theta))
