"""Tests for the Krasnikov ring (Z_N-symmetric tube array)."""

import math

import numpy as np

from systrophe.geometry.krasnikov_ring import (
    krasnikov_ring_NEC_radial,
    krasnikov_ring_extinction_breakdown,
    krasnikov_ring_total_negative_energy,
    novelty_scan,
)


def test_extinction_at_N_equals_2_eps_zero():
    """N=2 with zero perturbation: anti-phase pair, exact extinction."""
    e = krasnikov_ring_total_negative_energy(N=2, epsilon=0.0)
    # With phasor sum {1, cos(pi)} = 1 - 1 = 0
    assert abs(e) < 1e-10


def test_extinction_at_N_3_eps_zero():
    """N=3 with zero perturbation: Z_3-symmetric, extinct."""
    e = krasnikov_ring_total_negative_energy(N=3, epsilon=0.0)
    assert abs(e) < 1e-10


def test_extinction_at_N_5_eps_zero():
    """N=5, Z_5-symmetric extinction."""
    e = krasnikov_ring_total_negative_energy(N=5, epsilon=0.0)
    assert abs(e) < 1e-10


def test_extinction_breaks_with_epsilon():
    """At small nonzero epsilon, |E_neg| grows away from zero."""
    e0 = krasnikov_ring_total_negative_energy(N=3, epsilon=0.0)
    e_eps = krasnikov_ring_total_negative_energy(N=3, epsilon=0.5)
    assert abs(e_eps) > abs(e0)


def test_extinction_breakdown_keys():
    out = krasnikov_ring_extinction_breakdown()
    assert "N_values" in out
    assert "epsilon_grid" in out
    assert "E_neg_grid" in out


def test_novelty_scan_runs():
    res = novelty_scan(N_values=[2, 3, 5], n_eps=15)
    assert "per_N" in res
    assert "aggregate_verdict" in res
    for N in (2, 3, 5):
        assert N in res["per_N"]
        assert "verdict" in res["per_N"][N]


def test_noise_robustness_runs():
    from systrophe.geometry.krasnikov_ring import krasnikov_ring_noise_robustness
    res = krasnikov_ring_noise_robustness(N=3, n_trials=10)
    assert "novelty_verdict" in res
    assert "residual_E_neg" in res
    assert len(res["residual_E_neg"]) > 0


def test_noise_robustness_residual_grows_with_noise():
    from systrophe.geometry.krasnikov_ring import krasnikov_ring_noise_robustness
    res = krasnikov_ring_noise_robustness(N=3, n_trials=20)
    r = res["residual_E_neg"]
    # The mean residual should grow (non-strictly) with noise amplitude.
    assert r[-1] >= r[0]


def test_noise_robustness_pair_is_fragile():
    """N=2 should produce nontrivial residual even at moderate noise."""
    from systrophe.geometry.krasnikov_ring import krasnikov_ring_noise_robustness
    import numpy as np
    res = krasnikov_ring_noise_robustness(
        N=2, noise_grid=np.linspace(0.0, math.pi, 6),
        n_trials=20,
    )
    # Residual at high noise should significantly exceed residual at zero noise
    assert res["residual_E_neg"][-1] > 10 * (res["residual_E_neg"][0] + 1e-12)
