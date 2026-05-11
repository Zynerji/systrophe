"""Tests for the horn-toroidal warp bubble."""

import math

from systrophe.horn_toroidal_warp import (
    horn_T_tt_profile,
    horn_pinch_threshold,
    horn_radius,
    horn_steering_magnitude,
    horn_steering_vector,
    novelty_scan,
)


def test_horn_radius_returns_R_at_epsilon_zero():
    """epsilon=0 -> rho is independent of theta and equals R."""
    for t in (0.0, 1.0, 2.0, math.pi, 3.0):
        assert abs(horn_radius(t, R=1.0, epsilon=0.0) - 1.0) < 1e-12


def test_horn_radius_max_at_horn_axis():
    """rho is maximum at theta = theta_0."""
    R = 1.0
    eps = 0.3
    theta_0 = 1.5
    rho_at_axis = horn_radius(theta_0, R, eps, theta_0)
    rho_off_axis = horn_radius(theta_0 + math.pi, R, eps, theta_0)
    assert rho_at_axis > rho_off_axis


def test_steering_vector_zero_at_epsilon_zero():
    """No horn-twist -> no steering."""
    p_x, p_y = horn_steering_vector(R=1.0, epsilon=0.0, theta_0=0.0)
    assert abs(p_x) < 1e-6
    assert abs(p_y) < 1e-6


def test_steering_magnitude_grows_with_epsilon():
    m0 = horn_steering_magnitude(R=1.0, epsilon=0.0)
    m1 = horn_steering_magnitude(R=1.0, epsilon=0.5)
    assert m1 > m0


def test_steering_vector_orientation_follows_theta_0():
    """At theta_0 = 0 the dipole is along +x; at theta_0 = pi/2 along +y."""
    p_x_a, p_y_a = horn_steering_vector(R=1.0, epsilon=0.3, theta_0=0.0)
    p_x_b, p_y_b = horn_steering_vector(R=1.0, epsilon=0.3,
                                          theta_0=math.pi / 2)
    # When dipole is along +x, |p_x| > |p_y|; rotated by pi/2 -> |p_y| > |p_x|.
    assert abs(p_x_a) > abs(p_y_a) - 1e-6
    assert abs(p_y_b) > abs(p_x_b) - 1e-6


def test_pinch_threshold_is_unity():
    assert horn_pinch_threshold(R=1.0) == 1.0


def test_horn_T_tt_profile_finite():
    val = horn_T_tt_profile(theta=1.0, R=1.0, epsilon=0.2)
    assert math.isfinite(val)


def test_novelty_scan_returns_verdict():
    res = novelty_scan(epsilon_range=(0.0, 0.9), n_eps=15,
                        R=1.0, theta_0=0.0)
    assert "novelty_verdict" in res
    assert res["novelty_verdict"] in ("smooth", "uniform", "novel_structure")
