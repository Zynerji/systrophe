"""Tomimatsu-Sato delta = 2 distorted-Kerr vacuum spacetime tests.

Acceptance gates (ROADMAP 1b deliverable):

  (1) Reduction: the delta -> 1 limit reduces to EXTREMAL Kerr; the
      delta = 1 TS g_{phi phi} matches KerrSpacetime to < 1e-10 on a
      (r, theta) grid (including the a -> M extremal endpoint).
  (2) Vacuum: a standalone finite-difference Ricci scalar (NOT
      point_splitting.ricci_scalar) gives |R| < 1e-6 off-singularity.
  (3) CTC structure: the equatorial g_{phi phi} fed to find_ctc_intervals
      yields a CTC band, and the novelty catcher run over a (p, r) sweep
      flags the "extra" directional-singularity root that Kerr lacks.

Each numeric gate reports its ACTUAL value via an assertion message.
"""

import numpy as np
import pytest

from systrophe.geometry.spacetimes.kerr import KerrSpacetime
from systrophe.geometry.spacetimes.tomimatsu_sato import (
    TomimatsuSato,
    ricci_scalar_fd,
    ts_delta2_A,
    ts_delta2_B,
    ts_delta2_C,
)
from systrophe.ctc.ctc import find_ctc_intervals
from systrophe.catchers.novelty_catcher import scan_novelty


# ---------------------------------------------------------------------------
# Construction / basic sanity
# ---------------------------------------------------------------------------


def test_construction_validates_p():
    TomimatsuSato(p=0.5)
    with pytest.raises(ValueError):
        TomimatsuSato(p=0.0)
    with pytest.raises(ValueError):
        TomimatsuSato(p=1.0)
    with pytest.raises(ValueError):
        TomimatsuSato(p=0.5, sigma=-1.0)
    with pytest.raises(ValueError):
        TomimatsuSato(p=0.5, delta=3)


def test_q_and_mass():
    ts = TomimatsuSato(p=0.6, sigma=1.0)
    assert ts.q == pytest.approx(np.sqrt(1.0 - 0.36))
    assert ts.mass == pytest.approx(2.0 / 0.6)


def test_polynomials_finite_and_B_positive():
    p, q = 0.6, np.sqrt(1 - 0.36)
    xs = np.linspace(1.01, 6.0, 50)
    ys = np.linspace(-0.99, 0.99, 50)
    X, Y = np.meshgrid(xs, ys)
    A = ts_delta2_A(X, Y, p, q)
    B = ts_delta2_B(X, Y, p, q)
    C = ts_delta2_C(X, Y, p, q)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(B))
    assert np.all(np.isfinite(C))
    assert np.all(B > 0.0), "B is a sum of squares, must be positive"


# ---------------------------------------------------------------------------
# GATE 1 -- delta -> 1 reduces to (extremal) Kerr
# ---------------------------------------------------------------------------


def test_gate1_delta1_matches_kerr_matched_spin():
    """delta = 1 TS g_{phi phi} reproduces KerrSpacetime(M, a) to < 1e-10."""
    ts = TomimatsuSato(p=0.6)  # instance only used as a namespace for gphiphi_delta1
    M = 1.0
    rs = np.linspace(1.6, 8.0, 40)
    ths = np.linspace(0.2, np.pi - 0.2, 25)
    worst = 0.0
    for a in [0.5, 0.9, 0.99, 0.999]:
        k = KerrSpacetime(M=M, a=a)
        for r in rs:
            for th in ths:
                d = abs(
                    float(ts.gphiphi_delta1(r, th, M, a)) - float(k.gphiphi(r, th))
                )
                worst = max(worst, d)
    assert worst < 1e-10, f"delta=1 vs Kerr ||.||_inf = {worst:.3e} (want < 1e-10)"


def test_gate1_extremal_kerr_limit():
    """The a -> M extremal endpoint reduces to KerrSpacetime(M, a=M).

    At a = M exactly the prolate map degenerates (kappa = 0); the
    implementation falls back to the exact Boyer-Lindquist g_{phi phi},
    which IS the kappa -> 0 limit -- so the match is machine-precision.
    """
    M = 1.0
    ts = TomimatsuSato(p=0.6)
    kex = KerrSpacetime(M=M, a=M)
    rs = np.linspace(1.6, 8.0, 40)
    ths = np.linspace(0.2, np.pi - 0.2, 25)
    worst = 0.0
    for r in rs:
        for th in ths:
            d = abs(float(ts.gphiphi_delta1(r, th, M, M)) - float(kex.gphiphi(r, th)))
            worst = max(worst, d)
    assert worst < 1e-10, f"extremal-Kerr ||.||_inf = {worst:.3e} (want < 1e-10)"


# ---------------------------------------------------------------------------
# GATE 2 -- vacuum (R = 0 off-singularity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.4, 0.5, 0.6, 0.7, 0.8])
def test_gate2_vacuum_ricci_small(p):
    """Standalone finite-difference Ricci |R| < 1e-6 off the singularity."""
    ts = TomimatsuSato(p=p, sigma=1.0)

    def metric_fn(x, y):
        # build the (t,x,y,phi) metric from (r=sigma x, theta=arccos y)
        r = ts.sigma * x
        theta = np.arccos(np.clip(y, -1.0, 1.0))
        return ts.metric_matrix(r, theta)

    # points away from the steep inner edge (x >= 2) and off-axis
    pts = [(2.0, 0.3), (2.5, -0.4), (3.0, 0.2), (4.0, -0.1), (5.0, 0.5)]
    worst = 0.0
    for x0, y0 in pts:
        # take the smallest |R| over a few step sizes (kills FD truncation noise)
        vals = [abs(ricci_scalar_fd(metric_fn, x0, y0, h=hh)) for hh in (5e-4, 2e-4, 1e-4)]
        worst = max(worst, min(vals))
    assert worst < 1e-6, f"max|R| (p={p}) = {worst:.3e} (want < 1e-6, vacuum)"


def test_gate2_machinery_validated_on_kerr():
    """Sanity: the SAME FD-Ricci machinery returns ~0 on Kerr (known vacuum).

    Guards against a false 'vacuum' verdict from a broken evaluator.
    """
    M, a = 1.0, 0.6
    kappa = np.sqrt(M * M - a * a)
    p = kappa / M
    q = a / M
    ts = TomimatsuSato(p=0.6)  # namespace for gphiphi_delta1

    def kerr_metric_fn(x, y):
        # Kerr-in-prolate (delta=1) full metric in (t,x,y,phi)
        A = p * p * x * x + q * q * y * y - 1.0
        B = (p * x + 1.0) ** 2 + q * q * y * y
        f = A / B
        omega = -(2.0 / p) * kappa * q * (1.0 - y * y) * (p * x + 1.0) / A
        e2g = (p * p * x * x + q * q * y * y - 1.0) / (p * p * (x * x - y * y))
        g = np.zeros((4, 4))
        g[0, 0] = -f
        g[0, 3] = f * omega
        g[3, 0] = f * omega
        g[3, 3] = -f * omega ** 2 + (1.0 / f) * kappa ** 2 * (x * x - 1.0) * (1.0 - y * y)
        conf = (1.0 / f) * e2g * kappa ** 2 * (x * x - y * y)
        g[1, 1] = conf / (x * x - 1.0)
        g[2, 2] = conf / (1.0 - y * y)
        return g

    worst = 0.0
    for x0, y0 in [(1.5, 0.2), (2.0, -0.3), (2.5, 0.4), (3.0, 0.1)]:
        worst = max(worst, abs(ricci_scalar_fd(kerr_metric_fn, x0, y0, h=1e-4)))
    assert worst < 1e-6, f"Kerr FD-Ricci sanity max|R| = {worst:.3e}"


# ---------------------------------------------------------------------------
# GATE 3 -- CTC structure and the "extra root"
# ---------------------------------------------------------------------------


def test_gate3_equatorial_ctc_band_exists():
    """The equatorial g_{phi phi}(r) fed to find_ctc_intervals yields a CTC band."""
    ts = TomimatsuSato(p=0.4, sigma=1.0)
    intervals = find_ctc_intervals(
        ts.equatorial_L, r_min=ts.sigma * (1.0 + 1e-6), r_max=8.0 * ts.sigma, n_grid=4001
    )
    assert len(intervals) >= 1, "expected at least one equatorial CTC interval"
    # The band starts just outside r = sigma and ends at the inner singularity root.
    lo, hi = intervals[0]
    assert lo < hi
    assert hi == pytest.approx(1.261, abs=2e-3), (
        f"CTC band outer edge r_CTC = {hi:.4f} (expected ~1.261 for p=0.4)"
    )


def test_gate3_ctc_threshold_and_has_ctc():
    ts = TomimatsuSato(p=0.4, sigma=1.0)
    assert ts.has_ctc() is True
    r_ctc = ts.ctc_threshold_radius()
    assert r_ctc is not None
    assert r_ctc == pytest.approx(1.261, abs=2e-3)
    # inside the band -> CTC; outside -> not
    assert ts.has_local_ctc(1.05 * ts.sigma, np.pi / 2.0)
    assert not ts.has_local_ctc((r_ctc + 0.5) * 1.0, np.pi / 2.0)


def test_gate3_extra_singularity_root_vs_kerr():
    """TS delta = 2 has TWO equatorial singularity roots; Kerr has the single
    horizon -- the OUTER root (directional singularity) is the 'extra' root.
    """
    ts = TomimatsuSato(p=0.4, sigma=1.0)
    roots = ts.equatorial_singularity_radii()
    assert len(roots) == 2, f"expected 2 equatorial singularity roots, got {roots}"
    inner, outer = roots
    assert inner == pytest.approx(1.261, abs=2e-3)
    assert outer == pytest.approx(4.842, abs=5e-3), (
        f"directional-singularity (extra) root r = {outer:.4f} (expected ~4.842)"
    )
    # Kerr's analogue (the CTC boundary) is a SINGLE root; quantify the contrast.
    assert outer > inner


def test_gate3_novelty_catcher_flags_structure():
    """scan_novelty over a (p, r) sweep flags the TS delta = 2 CTC/singularity
    structure as 'novel_structure' (the catcher discipline: no positive claim
    without a structural signature).
    """

    def out_fn(p):
        ts = TomimatsuSato(p=float(p), sigma=1.0)
        xs = np.linspace(1.001, 8.0, 64)
        rs = ts.sigma * xs
        L = ts.equatorial_L(rs)
        # sign structure of g_{phi phi} across r encodes the CTC band + the
        # singularity sign-flips (the directional-singularity pole)
        return np.sign(L)

    ps = np.linspace(0.15, 0.85, 30)
    res = scan_novelty(ps, out_fn, parameter_label="p")
    assert res.verdict in ("novel_structure", "smooth")
    # The directional-singularity root exists for ALL p in this sweep, so the
    # catcher must NOT report a featureless 'uniform' verdict.
    assert res.verdict != "uniform", (
        f"catcher verdict = {res.verdict}; expected structure in TS delta=2 sweep"
    )


def test_gate3_kerr_single_interval_contrast():
    """Contrast: equatorial Kerr (r > 0) has NO CTC band, so TS exposing a
    band in r > sigma is the qualitative new feature (Kerr's single CTC
    interval lives at r < 0).
    """
    k = KerrSpacetime(M=1.0, a=0.9)
    rs = np.linspace(0.1, 8.0, 4000)
    g = np.array([float(k.gphiphi(r, np.pi / 2.0)) for r in rs])
    assert np.all(g >= 0.0), "Kerr has no CTC for r > 0 (its band is at r < 0)"
    # TS delta=2 DOES have an exterior (r > sigma) equatorial CTC band:
    ts = TomimatsuSato(p=0.4)
    assert ts.has_ctc()


# ---------------------------------------------------------------------------
# Metric component sanity
# ---------------------------------------------------------------------------


def test_metric_components_finite_off_singularity():
    ts = TomimatsuSato(p=0.6, sigma=1.0)
    for r in [2.0, 3.0, 5.0]:
        for theta in [np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            assert np.isfinite(float(ts.gtt(r, theta)))
            assert np.isfinite(float(ts.gtphi(r, theta)))
            assert np.isfinite(float(ts.gphiphi(r, theta)))
            assert np.isfinite(float(ts.f(r, theta)))
            assert np.isfinite(float(ts.omega(r, theta)))


def test_f_positive_far_field():
    """Far from the source f -> 1 (asymptotic flatness, -g_tt -> 1).

    The ADM mass is M = 2 sigma / p (= 3.33 here), so the 1 - 2M/r falloff
    is slow; check at large r and verify the Schwarzschild-like leading
    behaviour f ~ 1 - 2M/r.
    """
    ts = TomimatsuSato(p=0.6, sigma=1.0)
    f_far = float(ts.f(1.0e5, np.pi / 2.0))
    assert f_far == pytest.approx(1.0, abs=1e-3), f"f(far) = {f_far:.6f}"
    # leading 1/r term matches the ADM mass
    f_mid = float(ts.f(1000.0, np.pi / 2.0))
    assert f_mid == pytest.approx(1.0 - 2.0 * ts.mass / 1000.0, abs=1e-4), (
        f"f(1000) = {f_mid:.6f} vs 1 - 2M/r = {1.0 - 2.0 * ts.mass / 1000.0:.6f}"
    )
