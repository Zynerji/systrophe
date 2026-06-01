"""Tests for offtrace_modesum: adiabatic-subtracted WKB <T_rr> (Phase 2b).

This module is a HIGH-RISK build whose HONEST outcome is a documented
*blocker* / clean negative. The tests below assert the REAL numbers --
including the self-falsifying trace gate FAILING -- rather than faking a
pass. The physics that the tests pin down:

  * The genuine WKB machinery (clean Schrodinger potential, real tortoise
    phase) is correct and replaces the documented placeholders.
  * Obstruction #1: the entire LP exterior up to the FIRST (innermost,
    analytically cleanest) Cauchy horizon is a CTC band (g_phiphi < 0),
    so no static Boulware vacuum exists there -- the mode-sum is not a
    physical state.
  * Obstruction #2: even in the L>0 band around the second horizon (where
    a static vacuum DOES exist and propagating modes return), a
    leading-adiabatic-order subtraction does NOT reproduce K/(2880 pi^2)
    -- the trace gate FAILS, because the conformal anomaly lives in the
    sub-leading adiabatic orders whose closed form is unavailable for the
    non-spherical, frame-dragged LP cylinder.

Both obstructions make the <T_rr> horizon power physically meaningless,
so the deliverable's verdict is an honest 'partial'/blocked result, not a
4D mirror of the 2D Polyakov -1/-2 headline.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from systrophe.geometry.vanstockum import VanStockumInterior
from systrophe.qftcs.offtrace_modesum import (
    OfftraceModesumReport,
    adiabatic_counterterms_Trr,
    assembled_modesum_tensor,
    boulware_vacuum_obstruction,
    co_rotating_frequency_sq,
    fit_Trr_at_horizon,
    offtrace_modesum_novelty_scan,
    offtrace_modesum_report,
    radial_tortoise_phase,
    schrodinger_potential,
    T_rr_integrand,
    T_rr_ren,
    T_rr_unren,
    trace_gate_residual,
    uv_finiteness_report,
    wkb_mode,
    wkb_radial_derivative_sq,
)
from systrophe.qftcs.point_splitting import kretschmann_scalar


@pytest.fixture(scope="module")
def vs_super() -> VanStockumInterior:
    """omega=2, R=1: alpha=sqrt(15), three horizons (FINDINGS_PHASE2A case)."""
    return VanStockumInterior(omega=2.0, R=1.0)


# ---------------------------------------------------------------------------
# Genuine Schrodinger potential (clean, no 1/F regularisation)
# ---------------------------------------------------------------------------


def test_schrodinger_potential_uses_exact_tphi_inversion(vs_super):
    """p^2 leading omega^2 coefficient equals L/r^2 = g_phiphi/r^2 exactly.

    Confirms the (t,phi) block is inverted via F L + K^2 = r^2 (no 1/F
    regularisation): d(p^2)/d(omega^2) -> L/r^2 at large omega.
    """
    r = 2.7  # L > 0 band
    L = float(vs_super.analytic_exterior_L(np.array([r]))[0])
    lead_expected = L / (r * r)
    om1, om2 = 100.0, 200.0
    p2_1 = schrodinger_potential(vs_super, r, om1, m=0)
    p2_2 = schrodinger_potential(vs_super, r, om2, m=0)
    lead_measured = (p2_2 - p2_1) / (om2 * om2 - om1 * om1)
    assert math.isclose(lead_measured, lead_expected, rel_tol=1e-9), (
        f"leading symbol {lead_measured} != L/r^2 {lead_expected}"
    )


def test_co_rotating_frequency_carries_K_frame_drag(vs_super):
    """The m-dependent cross term 2 K omega m / r^2 is present (frame drag)."""
    r = 2.7
    base = co_rotating_frequency_sq(vs_super, r, omega=5.0, m=0)
    with_m = co_rotating_frequency_sq(vs_super, r, omega=5.0, m=2)
    K = float(vs_super.analytic_exterior_K(np.array([r]))[0])
    F = float(vs_super.analytic_exterior_F(np.array([r]))[0])
    # delta = (2 K omega m - F m^2)/r^2
    expected_delta = (2.0 * K * 5.0 * 2 - F * 4) / (r * r)
    assert math.isclose(with_m - base, expected_delta, rel_tol=1e-9)


def test_schrodinger_finite_at_cauchy_horizon(vs_super):
    """No 1/F blow-up: p^2 stays finite as F -> 0 at the horizon."""
    from systrophe.qftcs.quantum_diagnostics import cauchy_horizon_estimate
    r_H = float(cauchy_horizon_estimate(vs_super)[0])
    p2 = schrodinger_potential(vs_super, r_H, omega=5.0, m=1)
    assert np.isfinite(p2), "p^2 should be finite at F=0 (uses L,K not 1/F)"


# ---------------------------------------------------------------------------
# Genuine tortoise phase (NOT sqrt(V)*r placeholder)
# ---------------------------------------------------------------------------


def test_tortoise_phase_is_real_integral_not_sqrtV_times_r(vs_super):
    """Phase is int|p|dr', monotone in r, and != sqrt(p2)*r placeholder."""
    r = 2.7
    omega = 5.0
    phase = radial_tortoise_phase(vs_super, r, omega, m=0)
    p2 = schrodinger_potential(vs_super, r, omega, m=0)
    placeholder = math.sqrt(abs(p2)) * r  # the OLD hadamard_modesum form
    assert phase > 0
    # The genuine integral differs materially from sqrt(V)*r.
    assert abs(phase - placeholder) > 0.1 * abs(placeholder)


def test_tortoise_phase_monotone_and_additive(vs_super):
    """Phase grows monotonically outward (additive over sub-intervals)."""
    omega = 6.0
    p1 = radial_tortoise_phase(vs_super, 2.2, omega, r_ref=1.05)
    p2 = radial_tortoise_phase(vs_super, 2.6, omega, r_ref=1.05)
    p3 = radial_tortoise_phase(vs_super, 3.0, omega, r_ref=1.05)
    assert p1 < p2 < p3


def test_tortoise_phase_zero_at_reference(vs_super):
    val = radial_tortoise_phase(vs_super, 1.05, omega=5.0, r_ref=1.05)
    assert val == 0.0


def test_wkb_mode_oscillatory_in_allowed_region(vs_super):
    """In the L>0 band at high omega, p^2>0 gives a complex oscillatory mode."""
    chi = wkb_mode(vs_super, 2.7, omega=10.0, m=0)
    assert isinstance(chi, complex)
    assert abs(chi) > 0
    # p^2>0 -> nonzero imaginary part generically
    assert abs(chi.imag) > 0 or abs(chi.real) > 0


def test_wkb_radial_derivative_density_positive(vs_super):
    d = wkb_radial_derivative_sq(vs_super, 2.7, omega=10.0, m=0)
    assert d > 0


# ---------------------------------------------------------------------------
# OBSTRUCTION #1: no Boulware vacuum on the inner CTC band
# ---------------------------------------------------------------------------


def test_inner_exterior_is_ctc_band_no_vacuum(vs_super):
    """The whole exterior up to r_H[0] has g_phiphi<0: no static vacuum.

    This is the deep blocker behind the FINDINGS_PHASE2A open item.
    """
    from systrophe.qftcs.quantum_diagnostics import cauchy_horizon_estimate
    r_H = float(cauchy_horizon_estimate(vs_super)[0])
    # Sample several radii between the source and the first horizon.
    for r in np.linspace(1.1, 0.98 * r_H, 6):
        obs = boulware_vacuum_obstruction(vs_super, float(r), m_max=6)
        assert obs["is_ctc_band"], f"r={r}: expected CTC band (L<0)"
        assert not obs["vacuum_exists"], f"r={r}: no static vacuum should exist"
        assert obs["leading_omega2_symbol"] < 0, (
            f"r={r}: leading omega^2 symbol must be negative in CTC band"
        )
        # Clean marker: the axisymmetric m=0 continuum is fully evanescent
        # (no partial_t-positive-frequency radial modes). Some m!=0 modes can
        # still propagate at low omega via the frame-drag term -- that mixed
        # spectrum is itself the obstruction to a positive-frequency split.
        assert not obs["m0_has_propagating_mode"], (
            f"r={r}: m=0 continuum must be evanescent in the CTC band"
        )


def test_vacuum_returns_in_L_positive_band(vs_super):
    """Around the SECOND horizon (L>0 band) a static vacuum reappears."""
    # L>0 band is approximately (1.84, 4.14) for this case.
    for r in (2.3, 2.7, 3.0):
        obs = boulware_vacuum_obstruction(vs_super, r, m_max=6)
        assert obs["vacuum_exists"], f"r={r}: expected L>0 vacuum"
        assert obs["has_propagating_mode"], f"r={r}: modes should propagate"


# ---------------------------------------------------------------------------
# Leading-symbol-matched adiabatic subtraction cancels 16+ OOM of the UV
# ---------------------------------------------------------------------------


def test_adiabatic_subtraction_cancels_leading_UV(vs_super):
    """The leading-symbol counterterm cancels the bulk of the unren sum.

    |T_rr_ren| must be many orders of magnitude smaller than |T_rr_unren|
    -- i.e. the leading adiabatic order is genuinely subtracted (no trivial
    coefficient bug).
    """
    r = 2.7  # vacuum-exists band
    d = T_rr_ren(vs_super, r, omega_max=20.0, n_omega=60, m_max=5,
                 kz_max=20.0, n_kz=11)
    assert abs(d["T_rr_unren"]) > 0
    assert abs(d["T_rr_ren"]) < 0.1 * abs(d["T_rr_unren"]), (
        "leading-order subtraction should remove most of the unren value"
    )


# ---------------------------------------------------------------------------
# THE TRACE GATE (self-falsifying) -- documents the HONEST FAILURE
# ---------------------------------------------------------------------------


def test_trace_gate_fails_in_ctc_band(vs_super):
    """In the inner CTC band the trace gate fails by a wide margin (>>1e-3).

    With L<0 the mode-sum is not a physical state; even after the leading-
    symbol subtraction the trace residual is O(40-140), vastly above the
    1e-3 gate (target K/(2880 pi^2) is O(1e-3)). The <T_rr> power here is
    meaningless.
    """
    g = trace_gate_residual(vs_super, 1.2, omega_max=20.0, n_omega=40,
                            m_max=4, kz_max=20.0, n_kz=9)
    assert not g["passed"]
    # Residual O(40-140): finite (leading order subtracts) but far above gate.
    assert g["residual"] > 1.0, (
        f"CTC-band trace residual {g['residual']} should be >> the 1e-3 gate"
    )
    assert abs(g["trace_target"]) < 1e-2, "target K/(2880 pi^2) is O(1e-3)"


def test_trace_gate_fails_even_where_vacuum_exists(vs_super):
    """Leading-order AHS in the L>0 band still MISSES K/(2880 pi^2).

    This is the core honest negative: the conformal anomaly lives in the
    sub-leading adiabatic orders. A leading-symbol-matched subtraction
    leaves a residual of O(1-100), far above the 1e-3 gate -- so the
    <T_rr> power cannot be read off as physical.
    """
    for r in (2.3, 2.7, 3.0):
        g = trace_gate_residual(vs_super, r, omega_max=20.0, n_omega=60,
                                m_max=5, kz_max=20.0, n_kz=11)
        K = kretschmann_scalar(vs_super, r)
        target = K / (2880.0 * math.pi * math.pi)
        assert math.isclose(g["trace_target"], target, rel_tol=1e-9)
        # Gate FAILS: residual is O(1) or larger, target is O(1e-4).
        assert not g["passed"], f"r={r}: gate unexpectedly passed"
        assert g["residual"] > 1.0, (
            f"r={r}: residual {g['residual']} -- sub-leading anomaly not captured"
        )


def test_trace_gate_improved_16_orders_by_leading_match(vs_super):
    """Matching the metric's leading symbol drops the residual by >1e10.

    A flat (coefficient-1) counterterm leaves ~1e17; the L/r^2-matched
    counterterm leaves ~O(10-100). Documenting the magnitude of the
    leading-order cancellation (still insufficient for the gate).
    """
    r = 2.7
    g = trace_gate_residual(vs_super, r, omega_max=20.0, n_omega=60,
                            m_max=5, kz_max=20.0, n_kz=11)
    # Residual is finite and O(100) or smaller -- not the 1e17 of a
    # mismatched-coefficient subtraction.
    assert np.isfinite(g["residual"])
    assert g["residual"] < 1e4, (
        "leading-symbol-matched residual should be far below the 1e17 "
        "of an unmatched flat subtraction"
    )


# ---------------------------------------------------------------------------
# Assembled tensor structure
# ---------------------------------------------------------------------------


def test_assembled_tensor_has_all_diagonal_components(vs_super):
    tens = assembled_modesum_tensor(vs_super, 2.7, omega_max=16.0, n_omega=32,
                                    m_max=4, kz_max=16.0, n_kz=9)
    for key in ("T_tt", "T_rr", "T_phiphi", "T_zz", "trace"):
        assert key in tens
        assert np.isfinite(tens[key])


def test_T_rr_integrand_finite_and_real(vs_super):
    val = T_rr_integrand(vs_super, 2.7, omega=5.0, m=1, k_z=2.0)
    assert np.isfinite(val)


# ---------------------------------------------------------------------------
# UV finiteness (doubling omega_max)
# ---------------------------------------------------------------------------


def test_uv_finiteness_report_runs(vs_super):
    """UV check executes; reports the cutoff drift relative to the unren scale.

    Honest note: 'stable' here is RELATIVE to the (large) unren scale; the
    absolute drift is NOT negligible, consistent with residual sub-leading
    divergences that the leading-order subtraction leaves behind.
    """
    uv = uv_finiteness_report(vs_super, 2.7, omega_max=14.0, n_omega=28,
                              m_max=4, kz_max=14.0, n_kz=9)
    assert np.isfinite(uv["delta"])
    assert np.isfinite(uv["relative_drift"])
    assert uv["relative_drift"] >= 0.0


# ---------------------------------------------------------------------------
# <T_rr> horizon fit is GATED -- power reported but flagged non-physical
# ---------------------------------------------------------------------------


def test_Trr_fit_at_first_horizon_is_gate_blocked(vs_super):
    """Fit runs at r_H[0] but trace gate fails -> diverges flag forced False."""
    from systrophe.qftcs.quantum_diagnostics import cauchy_horizon_estimate
    r_H = float(cauchy_horizon_estimate(vs_super)[0])
    fit = fit_Trr_at_horizon(vs_super, r_H, n_samples=8, eps_min=3e-3,
                             eps_max=3e-2, omega_max=14.0, n_omega=28,
                             m_max=3, kz_max=14.0, n_kz=7)
    # Gate fails at the inner horizon -> not a trusted divergence.
    assert not fit.trace_gate_passed
    assert not fit.diverges, "power must NOT be reported as a physical divergence"
    assert np.isfinite(fit.trace_residual_at_mid)


# ---------------------------------------------------------------------------
# End-to-end report: honest BLOCKED verdict
# ---------------------------------------------------------------------------


def test_report_is_blocked_not_a_4d_mirror(vs_super):
    """The headline verdict is an honest documented blocker, not a 2D mirror."""
    rep = offtrace_modesum_report(vs_super, omega_max=14.0, n_omega=28,
                                  m_max=3, kz_max=14.0, n_kz=7)
    assert isinstance(rep, OfftraceModesumReport)
    assert not rep.trace_gate_passed
    # First horizon sits in the CTC band -> no vacuum.
    assert not rep.boulware_vacuum_exists
    assert rep.is_ctc_band
    assert "BLOCKED" in rep.verdict
    assert "CANNOT" in rep.verdict or "cannot" in rep.verdict


def test_report_requires_supercritical():
    vs_sub = VanStockumInterior(omega=0.3, R=1.0)
    with pytest.raises(ValueError):
        offtrace_modesum_report(vs_sub)


# ---------------------------------------------------------------------------
# Novelty catcher (always-on discipline)
# ---------------------------------------------------------------------------


def test_offtrace_novelty_scan_runs(vs_super):
    """Address-space catcher runs on the (T_rr_ren, residual, trace) profile."""
    result = offtrace_modesum_novelty_scan(vs_super, n_radii=16)
    assert result["verdict"] in {"smooth", "novel_structure", "uniform"}
    assert "lambda_2_at_radius" in result
