"""Tests for the spectral-chronology correspondence helpers + acceptance gate.

The acceptance gate (stated falsifiably): the spectral critical exponent
(slope of log tau vs log|a-1/2|, from the metric-derived openness g(a)) must match
the Boulware <T_tt> pole power (-1) in MAGNITUDE within tolerance, the two divergences
must be reported on their respective supercritical sides, and the address-space novelty
catcher on |lambda_2(a)| must flag a sharp feature bracketing a = 1/2.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from spectral_chronology_correspondence import (  # noqa: E402
    A_CRIT,
    alpha_log_frequency,
    fit_critical_exponent,
    lambda2_of_a,
    physical_openness,
    relaxation_time_from_lambda2,
    run_correspondence_study,
)


# --------------------------------------------------------------------------- #
# Pure-helper unit tests
# --------------------------------------------------------------------------- #
def test_alpha_log_frequency_threshold_and_asymptote():
    # subcritical / critical -> 0 (channel closed)
    assert alpha_log_frequency(0.4) == 0.0
    assert alpha_log_frequency(0.5) == 0.0
    # supercritical -> sqrt(4a^2 - 1)
    assert np.isclose(alpha_log_frequency(0.7), np.sqrt(4 * 0.49 - 1))
    # near-threshold square-root law: alpha ~ 2 sqrt(a - 1/2)
    da = 1e-4
    assert np.isclose(alpha_log_frequency(0.5 + da), 2.0 * np.sqrt(da), rtol=1e-2)


def test_physical_openness_bounds_and_monotone():
    assert physical_openness(0.4) == 0.0          # closed below threshold
    assert physical_openness(0.5) == 0.0          # closed at threshold
    g_lo = physical_openness(0.55)
    g_hi = physical_openness(2.0)
    assert 0.0 < g_lo < g_hi < 1.0                # opens monotonically into [0,1)
    # deep-supercritical limit approaches full openness
    assert physical_openness(50.0) > 0.98


def test_relaxation_time_limits():
    assert relaxation_time_from_lambda2(0.0) == 0.0          # instant resolution
    assert np.isinf(relaxation_time_from_lambda2(1.0))       # gap closed
    # monotone increasing toward the closed-gap limit
    assert (relaxation_time_from_lambda2(0.99)
            > relaxation_time_from_lambda2(0.5)
            > relaxation_time_from_lambda2(0.1))


def test_fit_critical_exponent_recovers_known_slope():
    d = np.geomspace(1e-4, 1e-1, 20)
    tau = 3.0 * d ** (-1.0)                                   # planted exponent -1
    exp, log_A, resid = fit_critical_exponent(d, tau)
    assert np.isclose(exp, -1.0, atol=1e-6)
    assert np.isclose(np.exp(log_A), 3.0, rtol=1e-6)
    assert resid < 1e-9


def test_lambda2_closes_at_threshold_opens_supercritically():
    # subcritical: channel is the identity -> lambda_2 == 1 (no resolution)
    assert np.isclose(lambda2_of_a(0.45), 1.0, atol=1e-12)
    # supercritical: gap opens (lambda_2 < 1), and opens more as a grows
    l2_near = lambda2_of_a(0.5 + 1e-3)
    l2_far = lambda2_of_a(0.9)
    assert l2_near < 1.0
    assert l2_far < l2_near                                   # deeper -> smaller |lambda_2|


# --------------------------------------------------------------------------- #
# Acceptance gate (the falsifiable claim)
# --------------------------------------------------------------------------- #
def test_acceptance_gate_spectral_exponent_matches_boulware_magnitude():
    res = run_correspondence_study()

    # Boulware pole power must be ~ -1 (the documented chronology-protection signal)
    assert np.isclose(res.boulware_power, -1.0, atol=0.05), res.boulware_power

    # Spectral exponent must be a genuine divergence (negative) and ~ -1 in magnitude
    assert res.spectral_exponent < 0.0, res.spectral_exponent
    assert np.isclose(abs(res.spectral_exponent), 1.0, atol=0.05), res.spectral_exponent
    assert res.spectral_residual_rms < 0.05, res.spectral_residual_rms

    # The duality (magnitude match) gate
    assert res.exponents_match is True, res.summary

    # Honesty: both divergences are reported on the SUPERCRITICAL side, in
    # DIFFERENT variables (|a-1/2| vs |r-r_H|) -- a duality, never an identity.
    assert "supercritical" in res.spectral_side
    assert "supercritical" in res.boulware_side
    assert "a-1/2" in res.spectral_side or "1/2" in res.spectral_side
    assert "r_H" in res.boulware_side or "r - r_H" in res.boulware_side


def test_acceptance_gate_catcher_brackets_threshold():
    res = run_correspondence_study()
    # The address-space catcher must flag novel structure with a sharp feature
    # bracketing a = 1/2 (the chronology threshold).
    assert res.catcher_verdict == "novel_structure", res.catcher_verdict
    assert res.catcher_n_sharp >= 1
    nearest = min(res.catcher_sharp_a, key=lambda x: abs(x - A_CRIT))
    assert abs(nearest - A_CRIT) <= 0.02, res.catcher_sharp_a


def test_study_runs_end_to_end_via_main():
    import spectral_chronology_correspondence as mod

    assert mod.main() == 0
