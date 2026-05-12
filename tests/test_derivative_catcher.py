"""Tests for the derivative catcher (smooth-sigmoid extension of the
address-space novelty catcher)."""

from __future__ import annotations

import numpy as np
import pytest

from systrophe.derivative_catcher import (
    _finite_diff,
    catch_smooth_transition,
    scan_novelty_derivative,
)


def test_finite_diff_first_order_linear():
    """First derivative of y = 2x is 2 everywhere."""
    p = np.linspace(0, 10, 20)
    v = 2.0 * p
    dvdp = _finite_diff(v, p, order=1)
    np.testing.assert_allclose(dvdp, 2.0, atol=1e-9)


def test_finite_diff_second_order_quadratic():
    """Second derivative of y = x^2 is 2 everywhere."""
    p = np.linspace(0, 10, 50)
    v = p ** 2
    d2vdp2 = _finite_diff(v, p, order=2)
    # Second derivative is 2 in the interior; edges may differ
    np.testing.assert_allclose(d2vdp2[5:-5], 2.0, atol=1e-2)


def test_derivative_catcher_catches_sat_style_transition():
    """A SAT-style transition (long plateau at 1.0, smooth drop near
    centre, long plateau at 0.0) is the real use case. The catcher
    should locate the transition centre within a few grid spacings.

    This mimics the actual 3-SAT P(SAT) curve where the derivative
    catcher recovered alpha_c = 4.270 (true 4.267) in
    examples/millennium_sat_phase_transition.py.
    """
    x_c_true = 5.0
    p = np.linspace(0, 10, 30)

    def sat_style(x):
        # Plateau at 1, sigmoid drop near x_c, plateau at 0
        # Add quantisation by rounding to nearest 1/60 to mimic
        # the finite-instance sampling in SAT counts
        clean = 1.0 / (1.0 + np.exp(-(x_c_true - x) * 4.0))
        return float(round(clean * 60) / 60.0)

    result = catch_smooth_transition(p, sat_style, n_bits=32)
    assert result["kind"] in ("smooth_sigmoid", "discontinuous")
    # On quantised data we may flag multiple sharps; require AT LEAST ONE
    # near the true centre rather than depending on tie-breaking for
    # which one wins as the argmax.
    sharps = result["derivative_scan"].sharp_features + result["value_scan"].sharp_features
    grid_spacing = float(p[1] - p[0])
    near_centre = [
        s for s in sharps
        if abs(float(s["parameter_value"]) - x_c_true) < 5 * grid_spacing
    ]
    assert near_centre, (
        f"No sharp feature within 5 grid spacings of x={x_c_true}. "
        f"Sharp features at: "
        f"{[float(s['parameter_value']) for s in sharps]}"
    )


def test_derivative_catcher_smooth_linear_returns_none():
    """A perfectly linear y=2x has no peak in its first derivative,
    so the catcher should NOT flag any transition centre."""
    p = np.linspace(0, 10, 30)

    def linear(x):
        return 2.0 * x

    result = catch_smooth_transition(p, linear, n_bits=32)
    assert result["kind"] == "none"
    assert result["estimated_transition_centre"] is None


def test_derivative_catcher_discontinuous_step_flagged():
    """A clean step function with quantised plateaus should be flagged
    by the catcher (value or derivative path)."""
    p = np.linspace(0, 10, 40)

    def step(x):
        return 0.0 if x < 5.0 else 1.0

    result = catch_smooth_transition(p, step, n_bits=32)
    assert result["kind"] in ("discontinuous", "smooth_sigmoid")
    centre = result["estimated_transition_centre"]
    assert centre is not None
    assert 4.0 <= centre <= 6.0


def test_scan_novelty_derivative_second_order():
    """Second-derivative scan should detect a cusp (kink in slope)."""
    p = np.linspace(0, 10, 30)

    def kink(x):
        # piecewise-linear with kink at x=5
        return x if x < 5.0 else 5.0 + 0.5 * (x - 5.0)

    result = scan_novelty_derivative(p, kink, n_bits=32, derivative_order=2)
    # Should be 'novel_structure' because the 2nd derivative has a
    # delta-like spike at the kink
    assert result.verdict == "novel_structure"


@pytest.mark.parametrize("steepness", [6.0, 10.0])
def test_derivative_catcher_quantised_steep_sigmoid(steepness: float):
    """Steep quantised sigmoid centre detection.

    Note: gradual sigmoids (steepness <= 3) are out of catcher domain
    -- the Hamming-step structure on the derivative is too uniform
    for the median+3*MAD discriminator to pinpoint the centre.
    """
    x_c_true = 3.0
    p = np.linspace(0, 6, 30)

    def sigmoid(x):
        clean = 1.0 / (1.0 + np.exp(-(x - x_c_true) * steepness))
        return float(round(clean * 60) / 60.0)

    result = catch_smooth_transition(p, sigmoid, n_bits=32)
    assert result["kind"] in ("smooth_sigmoid", "discontinuous")
    grid_spacing = float(p[1] - p[0])
    sharps = result["derivative_scan"].sharp_features + result["value_scan"].sharp_features
    near_centre = [
        s for s in sharps
        if abs(float(s["parameter_value"]) - x_c_true) < 5 * grid_spacing
    ]
    assert near_centre, (
        f"No sharp feature within 5 grid spacings of x={x_c_true} "
        f"at steepness={steepness}"
    )
