"""Tests for the ER=EPR throat-visibility bound module."""

import math

import numpy as np
import pytest

from systrophe.foundations.erepr_throat_visibility import (
    bound_A_vN,
    bound_B_EF,
    bound_C_hybrid,
    post_throat_visibility,
    evaluate_bounds,
    throat_area_proxy,
    werner_concurrence,
    werner_eigenvalues,
    werner_entanglement_of_formation,
    werner_vn_entropy,
)


# ----- Werner state ----- --------------------------------------------------


def test_werner_eigenvalues_sum_to_one():
    for w in [0.0, 0.5, 1.0]:
        eigs = werner_eigenvalues(w)
        assert eigs.sum() == pytest.approx(1.0, abs=1e-12)
        assert (eigs >= -1e-12).all()


def test_werner_vn_entropy_endpoints():
    assert werner_vn_entropy(1.0) == pytest.approx(0.0, abs=1e-12)
    assert werner_vn_entropy(0.0) == pytest.approx(math.log(4.0), rel=1e-12)


def test_werner_concurrence_endpoints():
    assert werner_concurrence(0.0) == 0.0
    assert werner_concurrence(1.0) == pytest.approx(1.0, abs=1e-12)
    # separable below 1/3
    assert werner_concurrence(0.3) == 0.0
    # entangled above 1/3
    assert werner_concurrence(0.5) > 0.0


def test_werner_E_F_endpoints():
    # E_F = 0 for separable
    assert werner_entanglement_of_formation(0.0) == 0.0
    assert werner_entanglement_of_formation(0.3) == 0.0
    # E_F = log(2) for max-entangled (C=1)
    assert werner_entanglement_of_formation(1.0) == pytest.approx(
        math.log(2.0), rel=1e-12,
    )


def test_throat_area_positive_iff_entangled():
    # At w <= 1/3 the throat collapses (no entanglement); above, it opens.
    assert throat_area_proxy(0.2, measure="E_F") == 0.0
    assert throat_area_proxy(0.5, measure="E_F") > 0.0
    assert throat_area_proxy(1.0, measure="E_F") == pytest.approx(
        4.0 * math.log(2.0), rel=1e-12,
    )


def test_throat_area_rejects_bad_ell_P():
    with pytest.raises(ValueError):
        throat_area_proxy(0.5, ell_P=0.0)
    with pytest.raises(ValueError):
        throat_area_proxy(0.5, ell_P=-0.1)


def test_throat_area_rejects_unknown_measure():
    with pytest.raises(ValueError):
        throat_area_proxy(0.5, measure="xyz")


# ----- post-throat visibility --------------------------------------------


def test_post_throat_no_throat_full_fringes():
    # w = 0 -> ancilla decohered -> V = 1 (no which-way information)
    for theta in [0.0, math.pi / 4, math.pi / 2, math.pi]:
        assert post_throat_visibility(0.0, theta) == pytest.approx(
            1.0, abs=1e-12,
        )


def test_post_throat_perfect_throat_recovers_englert():
    # w = 1 -> textbook V = cos(theta/2)
    for theta in [0.0, math.pi / 4, math.pi / 2, math.pi]:
        assert post_throat_visibility(1.0, theta) == pytest.approx(
            math.cos(theta / 2.0), abs=1e-12,
        )


def test_post_throat_theta_zero_is_one():
    # No path-distinguishing rotation -> ancilla can't tell paths apart.
    for w in [0.0, 0.5, 1.0]:
        assert post_throat_visibility(w, 0.0) == pytest.approx(
            1.0, abs=1e-12,
        )


def test_post_throat_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = float(rng.uniform(0, 1))
        theta = float(rng.uniform(0, math.pi))
        V = post_throat_visibility(w, theta)
        assert 0.0 <= V <= 1.0


def test_post_throat_rejects_bad_inputs():
    with pytest.raises(ValueError):
        post_throat_visibility(-0.1, 0.5)
    with pytest.raises(ValueError):
        post_throat_visibility(1.1, 0.5)
    with pytest.raises(ValueError):
        post_throat_visibility(0.5, -0.1)
    with pytest.raises(ValueError):
        post_throat_visibility(0.5, math.pi + 0.1)


# ----- bound verdicts ----------------------------------------------------


def test_bound_A_falsified_at_w_zero():
    # S_vN(0) = log(4) -> exp(-log 4) = 1/4. V_post(0, theta) = 1 > 1/4.
    assert bound_A_vN(0.0, math.pi / 2) == pytest.approx(0.25, rel=1e-12)
    assert post_throat_visibility(0.0, math.pi / 2) == 1.0


def test_bound_B_falsified_at_theta_zero():
    # E_F(1) = log(2) -> exp(-log 2) = 1/2. V_post(1, 0) = cos(0) = 1 > 1/2.
    assert bound_B_EF(1.0, 0.0) == pytest.approx(0.5, rel=1e-12)
    assert post_throat_visibility(1.0, 0.0) == pytest.approx(
        1.0, abs=1e-12,
    )


def test_bound_C_hybrid_endpoint_consistency():
    # At E_F = 0 (no entanglement) bound -> 1.
    assert bound_C_hybrid(0.0, math.pi / 2) == pytest.approx(1.0, abs=1e-12)
    # At theta = 0 bound -> 1 regardless of w.
    assert bound_C_hybrid(1.0, 0.0) == pytest.approx(1.0, abs=1e-12)


def test_full_grid_returns_verdicts_for_all_three_bounds():
    r = evaluate_bounds()
    assert set(r.verdicts.keys()) == {"A_naive_vN", "B_E_F", "C_hybrid"}
    for v in r.verdicts.values():
        assert math.isfinite(v.max_overshoot)
        # All three bounds in their current form are *falsified*; document
        # this as the standing finding. Conjecture C is the closest miss.
        assert v.holds is False
    assert r.V_post.shape == (r.w_grid.size, r.theta_grid.size)


def test_bound_C_is_closest_to_holding():
    # C should beat both A and B in max-overshoot magnitude.
    r = evaluate_bounds()
    over_A = r.verdicts["A_naive_vN"].max_overshoot
    over_B = r.verdicts["B_E_F"].max_overshoot
    over_C = r.verdicts["C_hybrid"].max_overshoot
    assert over_C < over_B
    assert over_C < over_A
    # The hybrid bound is *almost* tight: the worst violation is small.
    assert over_C < 0.10
