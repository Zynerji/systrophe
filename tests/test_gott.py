"""Gott pair (cosmic strings) CTC tests."""

from math import asin, cos, pi, sin, sqrt, tan

import pytest

from systrophe.spacetimes import GottPair, gott_critical_mu, gott_critical_velocity


def test_construction_validates_mu():
    GottPair(mu=0.05, v=0.5)  # OK
    with pytest.raises(ValueError):
        GottPair(mu=0.0, v=0.5)
    with pytest.raises(ValueError):
        GottPair(mu=0.2, v=0.5)  # mu >= 1/8
    with pytest.raises(ValueError):
        GottPair(mu=-0.01, v=0.5)


def test_construction_validates_v():
    GottPair(mu=0.05, v=0.0)  # zero v allowed
    with pytest.raises(ValueError):
        GottPair(mu=0.05, v=1.0)  # luminal
    with pytest.raises(ValueError):
        GottPair(mu=0.05, v=-0.1)


def test_alpha_and_deficit_angle():
    """alpha = 4 pi mu; deficit = 8 pi mu = 2 alpha."""
    g = GottPair(mu=0.05, v=0.5)
    assert g.alpha == pytest.approx(4.0 * pi * 0.05)
    assert g.deficit_angle == pytest.approx(8.0 * pi * 0.05)
    assert g.deficit_angle == pytest.approx(2.0 * g.alpha)


def test_lorentz_factor():
    g = GottPair(mu=0.05, v=0.6)
    assert g.gamma == pytest.approx(1.0 / sqrt(1.0 - 0.36))
    assert g.gamma == pytest.approx(1.25)


def test_critical_velocity_formula():
    """v_crit = sin(4 pi mu)."""
    mu = 0.05
    v_crit = gott_critical_velocity(mu)
    assert v_crit == pytest.approx(sin(4.0 * pi * mu))
    g = GottPair(mu=mu, v=0.5)
    assert g.critical_velocity == pytest.approx(v_crit)


def test_critical_gamma_formula():
    """gamma_crit = sec(4 pi mu) = 1 / cos(4 pi mu)."""
    mu = 0.05
    g = GottPair(mu=mu, v=0.5)
    assert g.critical_gamma == pytest.approx(1.0 / cos(4.0 * pi * mu))


def test_at_threshold_gamma_v_equals_tan_alpha():
    """At v = v_crit, gamma * v = tan(alpha) (the Gott threshold)."""
    mu = 0.05
    v = gott_critical_velocity(mu)
    g = GottPair(mu=mu, v=v)
    expected = tan(4.0 * pi * mu)
    assert g.gamma * g.v == pytest.approx(expected, rel=1e-12)


def test_below_threshold_no_ctc():
    """Below v_crit, no CTCs."""
    mu = 0.05
    v_crit = gott_critical_velocity(mu)
    g = GottPair(mu=mu, v=0.5 * v_crit)
    assert not g.has_ctc()
    assert g.ctc_margin() < 0


def test_above_threshold_ctc():
    """Above v_crit, CTCs exist."""
    mu = 0.05
    v_crit = gott_critical_velocity(mu)
    # Pick v > v_crit but still < 1
    v = min(0.5 * (v_crit + 1.0), 0.99)
    g = GottPair(mu=mu, v=v)
    assert g.has_ctc()
    assert g.ctc_margin() > 0


def test_critical_mu_inverts_critical_velocity():
    """gott_critical_mu(sin(4 pi mu)) == mu."""
    for mu in [0.01, 0.05, 0.1]:
        v = gott_critical_velocity(mu)
        mu_recovered = gott_critical_mu(v)
        assert mu_recovered == pytest.approx(mu, rel=1e-12)


def test_zero_velocity_never_gives_ctc():
    """Static strings (v = 0) do not produce CTCs regardless of mu."""
    g = GottPair(mu=0.1, v=0.0)
    assert not g.has_ctc()
