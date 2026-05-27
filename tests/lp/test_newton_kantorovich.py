"""Tests for Newton-Kantorovich solver and Picard comparison."""

import numpy as np
import pytest

from systrophe.lp.newton_kantorovich import (
    NKResult,
    is_convergence_rate_quadratic,
    newton_kantorovich_1d,
    newton_kantorovich_nd,
    picard_iteration_1d,
)


# ----- NK 1D on known problems ------------------------------------------

def test_nk_1d_quadratic_polynomial():
    """F(x) = x^2 - 2 should converge to sqrt(2) in <= 7 iterations."""
    F = lambda x: x * x - 2.0
    result = newton_kantorovich_1d(F, x0=1.5, tol=1e-12)
    assert result.converged
    assert result.iterations <= 7
    assert result.x[0] == pytest.approx(np.sqrt(2.0), abs=1e-10)


def test_nk_1d_with_analytical_derivative():
    """Analytical Jacobian path."""
    F = lambda x: x * x - 9.0
    dF = lambda x: 2 * x
    result = newton_kantorovich_1d(F, x0=2.0, df_analytical=dF, tol=1e-12)
    assert result.converged
    assert result.x[0] == pytest.approx(3.0, abs=1e-10)


def test_nk_1d_quadratic_convergence_rate():
    """For a well-conditioned smooth F, NK shows quadratic rate."""
    F = lambda x: x * x * x - 7.0
    result = newton_kantorovich_1d(F, x0=2.0, tol=1e-14)
    assert result.converged
    # Quadratic convergence -> very few iterations
    assert result.iterations <= 8
    # Residual drops dramatically near the solution
    assert result.residuals[-1] < 1e-14 or result.residuals[-1] < result.residuals[0] / 1e10


# ----- Picard 1D --------------------------------------------------------

def test_picard_converges_linearly():
    """x = cos(x) Picard iteration converges to Dottie number with linear rate.

    The fixed point is x* approx 0.7391, and g'(x*) = -sin(x*) approx -0.674,
    so Picard converges linearly with rate |g'(x*)| ~ 0.67.
    """
    g = np.cos
    result = picard_iteration_1d(g, x0=1.0, tol=1e-10, max_iter=200)
    assert result.converged
    assert result.x[0] == pytest.approx(0.7390851332151607, abs=1e-9)
    # Many iterations needed (linear convergence)
    assert result.iterations > 20


def test_picard_residual_decay_linear():
    """Consecutive Picard residuals have approximately constant ratio."""
    g = np.cos
    result = picard_iteration_1d(g, x0=1.0, tol=1e-12, max_iter=200)
    # Take residuals 10-15 (mid-iteration, well into asymptotic regime)
    rs = result.residuals[10:15]
    ratios = [rs[i + 1] / rs[i] for i in range(len(rs) - 1)]
    # All ratios should be close to |g'(fixed point)| ~ 0.67
    for r in ratios:
        assert 0.4 < r < 0.9


# ----- NK vs Picard direct comparison -----------------------------------

def test_nk_beats_picard_on_same_problem():
    """For F(x) = x - cos(x), NK uses far fewer iterations than Picard."""
    F = lambda x: x - np.cos(x)
    nk_result = newton_kantorovich_1d(F, x0=1.0, tol=1e-10)
    pi_result = picard_iteration_1d(np.cos, x0=1.0, tol=1e-10)
    assert nk_result.iterations < pi_result.iterations
    assert nk_result.iterations <= 10
    # Both converge to the same fixed point
    assert nk_result.x[0] == pytest.approx(pi_result.x[0], abs=1e-8)


# ----- NK ND ------------------------------------------------------------

def test_nk_nd_2d():
    """Two-dim system: F1 = x^2 + y^2 - 1; F2 = x - y."""
    def F(p):
        x, y = p
        return np.array([x * x + y * y - 1.0, x - y])
    result = newton_kantorovich_nd(F, x0=np.array([0.8, 0.7]), tol=1e-10)
    assert result.converged
    assert abs(result.x[0] - result.x[1]) < 1e-8
    s = result.x[0]
    assert s == pytest.approx(np.sqrt(0.5), abs=1e-7)


def test_nk_nd_iterations_few():
    """N-D NK on smooth problem converges in <= 10 iters."""
    def F(p):
        return np.array([p[0] ** 2 - 2.0, p[1] ** 3 - 8.0, p[2] - 5.0])
    result = newton_kantorovich_nd(F, x0=np.array([1.0, 1.5, 4.0]), tol=1e-10)
    assert result.converged
    assert result.iterations <= 10


# ----- Convergence-rate classification ---------------------------------

def test_classification_quadratic():
    """A clean NK run is classified as quadratic."""
    F = lambda x: x * x - 7.0
    result = newton_kantorovich_1d(F, x0=2.0, tol=1e-14)
    assert is_convergence_rate_quadratic(result)


def test_classification_linear_for_picard():
    """Picard is NOT classified as quadratic."""
    g = np.cos
    result = picard_iteration_1d(g, x0=1.0, tol=1e-10, max_iter=200)
    assert not is_convergence_rate_quadratic(result)
    assert result.rate == "linear"


def test_constant_step_iterator_is_classified_as_picard():
    """Direct demonstration: a 'constant per-iteration step' is linear-rate,
    which classify as 'linear' (Picard), not 'quadratic' (NK)."""
    # Construct an iterator with constant per-step difference, then check
    # how _classify_rate handles it
    history = []
    residuals = []
    r = 1.0
    for _ in range(10):
        residuals.append(r)
        r *= 0.5  # linear, ratio 0.5
    result = NKResult(
        x=np.array([0.0]), converged=True, iterations=10,
        residuals=residuals, history=history, rate="",
    )
    # Re-classify
    from systrophe.lp.newton_kantorovich import _classify_rate
    rate = _classify_rate(residuals)
    assert rate == "linear"
