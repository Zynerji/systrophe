"""Tipler sinusoid tests: log-periodic structure and fit recovery."""

import numpy as np
import pytest

from systrophe.sinusoid import TiplerSinusoid, fit_log_periodic


def test_alpha_formula():
    s = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    assert s.alpha == pytest.approx(np.sqrt(3.0))


def test_subcritical_construction_rejected():
    with pytest.raises(ValueError):
        TiplerSinusoid(R=1.0, a=0.4, A=1.0, delta=0.0)


def test_log_periodicity():
    """L(r * exp(2 pi / alpha)) has the same cosine phase."""
    s = TiplerSinusoid(R=1.0, a=1.5, A=1.0, delta=0.3, p=0.0)
    r0 = 2.0
    period = np.exp(2 * np.pi / s.alpha)
    # With p=0 the prefactor is just r * 1 = r; log-period multiplies r by `period`,
    # cosine returns to the same value, but L scales by r ratio.
    L0 = s.L(r0)
    L1 = s.L(r0 * period)
    # cosine phase identical -> ratio equals r ratio
    assert L1 / L0 == pytest.approx(period, rel=1e-10)


def test_zeros_are_log_uniform():
    """Zeros of cos(alpha ln(r/R) + delta) are uniformly spaced in ln r."""
    s = TiplerSinusoid(R=1.0, a=2.0, A=1.0, delta=0.0)
    zs = s.zeros(0.1, 100.0)
    log_zs = np.log(zs / s.R)
    diffs = np.diff(log_zs)
    expected = np.pi / s.alpha
    assert np.allclose(diffs, expected, rtol=1e-10)


def test_fit_recovers_synthetic_params():
    """fit_log_periodic recovers (A, delta, p) from clean synthetic L(r)."""
    truth = TiplerSinusoid(R=1.0, a=1.5, A=0.8, delta=0.7, p=1.2)
    rs = np.linspace(1.05, 10.0, 200)
    Ls = truth.L(rs)
    fitted = fit_log_periodic(rs, Ls, R=1.0, a=1.5)
    assert fitted.A == pytest.approx(truth.A, rel=1e-6)
    assert fitted.delta == pytest.approx(truth.delta, abs=1e-6)
    assert fitted.p == pytest.approx(truth.p, rel=1e-6)


def test_fit_refuses_underdetermined():
    rs = np.array([1.1, 1.2, 1.3, 1.4, 1.5])  # only 5 points
    Ls = np.zeros(5)
    with pytest.raises(ValueError):
        fit_log_periodic(rs, Ls, R=1.0, a=1.5)
