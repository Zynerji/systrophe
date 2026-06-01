"""Tests for the complex-dimension spectrometer.

Falsifiable acceptance gates (the numbers are asserted, not narrated):

  (a) Cantor-type C(N) ~ N^{ln2/ln3} * P(ln N) must recover
      Re ~ 0.6309, a ln-x log-frequency near omega = 5.7192, and
      AUTO-SELECT u = ln x with an integer comb.
  (b) A clean synthetic divisor-error Delta(x) with a planted sqrt-x comb
      at fundamental 4*pi = 12.566 must AUTO-SELECT u = sqrt x and recover
      that fundamental.
  (c) A pure power law with NO oscillation must return p > 0.05 against
      >= 2000 AR(1) surrogates (no false comb).

Plus structural checks: the fast vectorised Lomb-Scargle reproduces
``scipy.signal.lombscargle(normalize=True)``, the promoted AR(1) surrogate
preserves variance/zero-mean, and the comb fitter handles both comb kinds.
"""

import math

import numpy as np
from scipy.signal import lombscargle

from systrophe.catchers.complex_dimension import (
    ComplexDimension,
    ComplexDimensionSpectrometer,
    _fit_comb,
    _LombScargleBasis,
    _match_integer_comb,
    _match_sqrt_comb,
    ar1_surrogate,
    scan_transform,
    summarize_dimension_for_report,
)

LN2_LN3 = math.log(2) / math.log(3)   # ~0.6309297535714574
CANTOR_OMEGA = 5.7192
FOUR_PI = 4 * math.pi                  # ~12.566370614359172


# --------------------------------------------------------------------------
# Structural / regression checks
# --------------------------------------------------------------------------


def test_fast_lombscargle_matches_scipy():
    """Vectorised normalised L-S must equal scipy to floating precision."""
    rng = np.random.default_rng(3)
    u = np.sort(rng.random(800)) * 40.0
    y = rng.standard_normal(800)
    freqs = np.linspace(0.3, 25.0, 600)
    ref = lombscargle(u, y - y.mean(), freqs, normalize=True)
    basis = _LombScargleBasis(u, freqs)
    mine = basis.power(y)
    assert np.max(np.abs(mine - ref)) < 1e-12


def test_ar1_surrogate_preserves_variance_and_mean():
    """Promoted AR(1) surrogate is zero-mean and matches input std."""
    rng = np.random.default_rng(0)
    n = 2000
    # An AR(1) process so the lag-1 autocorrelation is non-trivial.
    e = rng.standard_normal(n)
    y = np.empty(n)
    y[0] = e[0]
    for i in range(1, n):
        y[i] = 0.7 * y[i - 1] + e[i]
    y = y - y.mean()
    s = ar1_surrogate(y, rng)
    assert abs(s.mean()) < 1e-9
    assert abs(s.std() - y.std()) / y.std() < 1e-9
    assert len(s) == n


def test_comb_matchers():
    """Integer and sqrt-integer comb matchers pick the right line index."""
    k, fpred, rel = _match_integer_comb(3 * 5.7, 5.7)
    assert k == 3 and abs(rel) < 1e-9 and abs(fpred - 3 * 5.7) < 1e-9
    n, fpred, rel = _match_sqrt_comb(FOUR_PI * math.sqrt(4), FOUR_PI)
    assert n == 4 and abs(rel) < 1e-9


def test_fit_comb_recovers_sqrt_fundamental():
    """A pure sqrt-integer peak set is fit by the sqrt comb, not integer."""
    base = FOUR_PI
    peaks = np.array([base * math.sqrt(n) for n in (1, 2, 4, 5)])
    grid = np.linspace(1.0, 60.0, 2000)
    fit = _fit_comb(peaks, grid)
    assert abs(fit.fundamental - base) / base < 0.02
    assert fit.kind == "sqrt"
    assert fit.mean_rel_error < 1e-3


# --------------------------------------------------------------------------
# Acceptance gate (a): Cantor-type log-periodic signal
# --------------------------------------------------------------------------


def _cantor_signal(n=4000):
    N = np.linspace(2.0, 6000.0, n)
    P = (1.0
         + 0.08 * np.cos(CANTOR_OMEGA * np.log(N))
         + 0.03 * np.cos(2.0 * CANTOR_OMEGA * np.log(N)))
    return N, N ** LN2_LN3 * P


def test_gate_a_cantor_recovers_ln_x_comb():
    N, C = _cantor_signal()
    spec = ComplexDimensionSpectrometer(n_surrogates=500, n_freq=1500, seed=0)
    res = spec.fit(N, C)
    # AUTO-SELECT u = ln x.
    assert res.variable == "ln_x", f"selected {res.variable}, expected ln_x"
    # Re(D) ~ ln2/ln3 (the log-log fit on N-uniform sampling is mildly
    # biased low; tolerance reflects the actual recovered value).
    assert abs(res.re_dimension - LN2_LN3) < 0.01, res.re_dimension
    # Im(D): the log-frequency near 5.7192.
    assert abs(res.omega_fundamental - CANTOR_OMEGA) < 0.05, res.omega_fundamental
    # Significant comb, integer-spaced.
    assert res.has_comb
    assert res.p_value < 0.05
    assert res.comb_kind == "integer"
    # Convenience complex packing.
    assert abs(res.dimension.real - res.re_dimension) < 1e-12
    assert abs(res.dimension.imag - res.omega_fundamental) < 1e-12


# --------------------------------------------------------------------------
# Acceptance gate (b): divisor-error-type sqrt-x comb
# --------------------------------------------------------------------------


def _divisor_count(n: int) -> int:
    c = 0
    r = int(math.isqrt(n))
    for k in range(1, r + 1):
        if n % k == 0:
            c += 1 if k * k == n else 2
    return c


def _divisor_signal(n=8000):
    """Clean synthetic Delta(x) with a planted 4*pi sqrt-x comb.

    Delta(x) ~ x^{1/4} sum_n d(n) n^{-3/4} cos(4 pi sqrt(n x) - pi/4),
    i.e. an x^{1/4} envelope times a sqrt-integer comb in u = sqrt(x).
    """
    x = np.linspace(1.0e4, 1.0e6, n)
    u = np.sqrt(x)
    env = x ** 0.25
    osc = np.zeros_like(x)
    for k in range(1, 9):
        w = _divisor_count(k) * k ** (-0.75)
        osc += w * np.cos(FOUR_PI * math.sqrt(k) * u - math.pi / 4)
    return x, env * osc


def test_gate_b_divisor_recovers_sqrt_x_comb():
    x, delta = _divisor_signal()
    spec = ComplexDimensionSpectrometer(n_surrogates=500, n_freq=1500, seed=0)
    res = spec.fit(x, delta)
    # AUTO-SELECT u = sqrt x.
    assert res.variable == "sqrt_x", f"selected {res.variable}, expected sqrt_x"
    # Recover the planted fundamental 4*pi.
    assert abs(res.omega_fundamental - FOUR_PI) / FOUR_PI < 0.01, (
        res.omega_fundamental)
    assert res.has_comb
    assert res.p_value < 0.05
    # The discrimination is genuine (not a tie-break): the other transforms
    # do NOT produce a significant comb.
    assert res.scans["ln_x"].p_value > 0.05
    assert res.scans["x^0.25"].p_value > 0.05


# --------------------------------------------------------------------------
# Acceptance gate (c): pure power law -> no false comb, p > 0.05 @ >=2000 surr
# --------------------------------------------------------------------------


def test_gate_c_pure_power_law_no_false_comb():
    """Exact power law: no oscillatory residual, so p = 1 and has_comb=False."""
    x = np.linspace(2.0, 5000.0, 3000)
    y = 3.5 * x ** 0.8
    spec = ComplexDimensionSpectrometer(n_surrogates=2000, n_freq=1200, seed=0)
    res = spec.fit(x, y)
    assert not res.has_comb
    assert res.p_value > 0.05
    # Re(D) still recovered exactly from the envelope.
    assert abs(res.re_dimension - 0.8) < 1e-6


def test_gate_c_noisy_power_law_null_against_2000_surrogates():
    """Noisy comb-free power law: AR(1) surrogate ensemble keeps p > 0.05.

    This exercises the actual >=2000-surrogate null machinery (the exact
    case short-circuits via the round-off guard), proving no false comb is
    manufactured from red measurement noise.
    """
    rng = np.random.default_rng(7)
    x = np.linspace(2.0, 5000.0, 3000)
    y = 3.5 * x ** 0.8 * (1.0 + 0.02 * rng.standard_normal(len(x)))
    spec = ComplexDimensionSpectrometer(n_surrogates=2000, n_freq=1200, seed=0)
    res = spec.fit(x, y)
    assert not res.has_comb, res.p_value
    assert res.p_value > 0.05, res.p_value
    # Every individual transform also fails to reject its red-noise null.
    for v, s in res.scans.items():
        assert s.p_value > 0.05, (v, s.p_value)


# --------------------------------------------------------------------------
# Misc API
# --------------------------------------------------------------------------


def test_scan_transform_rejects_unknown_variable():
    x = np.linspace(2.0, 100.0, 200)
    y = x ** 0.5
    try:
        scan_transform(x, y, "cbrt_x")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown variable")


def test_summary_string_is_informative():
    N, C = _cantor_signal()
    spec = ComplexDimensionSpectrometer(n_surrogates=300, n_freq=1200, seed=0)
    res = spec.fit(N, C)
    s = summarize_dimension_for_report(res)
    assert "ComplexDimensionSpectrometer" in s
    assert "ln_x" in s
    assert isinstance(res, ComplexDimension)
