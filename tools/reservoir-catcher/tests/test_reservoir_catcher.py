"""Tests pin the two headline findings plus mechanics. Small/fast config."""
import numpy as np
import pytest

from reservoir_catcher import (
    make_W, run_reservoir, spectral_radius, parity_task, evaluate, address_lambda2,
)


def test_spectral_radius_matched_across_kinds():
    rng = np.random.default_rng(0)
    for kind in ("random", "helical", "mobius"):
        W = make_W(120, kind, 0.9, rng)
        assert spectral_radius(W) == pytest.approx(0.9, abs=1e-6)


def test_reservoir_shapes_and_finite():
    rng = np.random.default_rng(1)
    u = rng.random((500, 1))
    X = run_reservoir(u, 80, "random", True, rng)
    assert X.shape == (500, 80)
    assert np.all(np.isfinite(X))


def test_address_lambda2_nonnegative():
    rng = np.random.default_rng(2)
    X = run_reservoir(rng.random((600, 1)), 100, "random", True, rng)
    assert address_lambda2(X, wash=100, rng=rng) >= 0.0


@pytest.fixture(scope="module")
def small_run():
    return evaluate(n=200, seeds=1, T=2600, split=1600, wash=150)


def test_Q1_nonlinearity_computes(small_run):
    """Nonlinear medium solves parity-3; linear medium is stuck at chance."""
    lin = small_run["linear-random"]["p3"][0]
    nl = small_run["nonlinear-random"]["p3"][0]
    assert lin < 0.65, f"linear parity-3 should be ~chance, got {lin}"
    assert nl > 0.85, f"nonlinear parity-3 should be solved, got {nl}"


def test_Q2_designed_topology_does_not_beat_random(small_run):
    """Helical/Mobius structure does not beat plain-random on parity-3."""
    rand = small_run["nonlinear-random"]["p3"][0]
    hel = small_run["nonlinear-helical"]["p3"][0]
    mob = small_run["nonlinear-mobius"]["p3"][0]
    assert hel <= rand + 0.05
    assert mob <= rand + 0.05
