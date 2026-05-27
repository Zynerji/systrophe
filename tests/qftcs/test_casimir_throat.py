"""Tests for Brown-Maclay throat Casimir."""

import numpy as np
import pytest

from systrophe.qftcs.casimir import standard_casimir_energy_density
from systrophe.qftcs.casimir_throat import (
    brown_maclay_at_lp_point,
    brown_maclay_energy_density,
    brown_maclay_normal_pressure,
    brown_maclay_T_minkowski,
    brown_maclay_trace,
    compare_throat_to_brown_maclay,
    topological_throat_coefficient,
    transverse_pressure,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


# ----- Brown-Maclay flat-space ------------------------------------------

def test_BM_shape():
    T = brown_maclay_T_minkowski(d=1.0)
    assert T.shape == (4, 4)


def test_BM_trace_is_zero():
    """Conformal invariance: trace = 0 in flat space."""
    T = brown_maclay_T_minkowski(d=2.5)
    assert brown_maclay_trace(T) == pytest.approx(0.0, abs=1e-14)


def test_BM_diagonal_signature():
    """T_tt = T_xx = T_yy, T_zz = -3 * T_tt."""
    T = brown_maclay_T_minkowski(d=1.5)
    assert T[0, 0] == pytest.approx(T[1, 1], rel=1e-12)
    assert T[1, 1] == pytest.approx(T[2, 2], rel=1e-12)
    assert T[3, 3] == pytest.approx(-3 * T[0, 0], rel=1e-12)


def test_BM_off_diagonal_zero():
    T = brown_maclay_T_minkowski(d=1.0)
    for i in range(4):
        for j in range(4):
            if i != j:
                assert T[i, j] == 0.0


def test_BM_energy_density_matches_standard_casimir():
    """T_{tt} = standard_casimir_energy_density(d) = -pi^2 / (720 d^4)."""
    for d in (0.5, 1.0, 2.0, 3.5):
        bm_rho = brown_maclay_energy_density(d)
        std_rho = float(standard_casimir_energy_density(d))
        assert bm_rho == pytest.approx(std_rho, rel=1e-12)


def test_BM_normal_pressure_textbook():
    """T_zz = +pi^2 / (240 d^4)."""
    for d in (1.0, 2.0):
        P = brown_maclay_normal_pressure(d)
        expected = np.pi ** 2 / (240.0 * d ** 4)
        assert P == pytest.approx(expected, rel=1e-12)


def test_BM_transverse_pressure_negative():
    """T_xx = T_yy = T_tt < 0."""
    for d in (1.0, 2.0):
        pt = transverse_pressure(d)
        assert pt < 0
        assert pt == pytest.approx(brown_maclay_energy_density(d), rel=1e-12)


def test_BM_d_scaling():
    """T(2d) = T(d) / 16 (since T ~ 1/d^4)."""
    T1 = brown_maclay_T_minkowski(d=1.0)
    T2 = brown_maclay_T_minkowski(d=2.0)
    assert np.allclose(T2, T1 / 16, atol=1e-14)


def test_BM_rejects_negative_d():
    with pytest.raises(ValueError):
        brown_maclay_T_minkowski(d=-1.0)


# ----- LP background evaluation ----------------------------------------

def test_BM_at_lp_returns_tensor(vs):
    result = brown_maclay_at_lp_point(vs, r=2.0, d=0.1)
    assert result["T_flat"].shape == (4, 4)
    assert np.all(np.isfinite(result["T_flat"]))
    assert np.isfinite(result["K_kretschmann"])


def test_BM_at_lp_flat_regime_for_small_d(vs):
    """Small d -> flat-space approximation valid (correction_scale small)."""
    result = brown_maclay_at_lp_point(vs, r=2.0, d=0.001)
    assert result["is_flat_space_regime"]


def test_BM_at_lp_curvature_correction_grows_with_d(vs):
    """Large d eventually breaks flat-space regime."""
    res_small = brown_maclay_at_lp_point(vs, r=2.0, d=0.01)
    res_large = brown_maclay_at_lp_point(vs, r=2.0, d=10.0)
    assert res_small["correction_scale"] < res_large["correction_scale"]


# ----- Topological connection ------------------------------------------

def test_topological_coefficient_at_zero_gamma():
    """At gamma = 0, topological coefficient equals casimir.py value."""
    c = topological_throat_coefficient(0.0)
    # From casimir.py: sum zeta_H(-3, b/3) / 720
    from systrophe.qftcs.casimir import topological_casimir_coefficient
    expected = float(topological_casimir_coefficient(0.0))
    assert c == pytest.approx(expected, rel=1e-12)


def test_compare_throat_to_BM_finite():
    """Ratio is finite at a non-zero throat coefficient."""
    cmp = compare_throat_to_brown_maclay(d=1.0, gamma_eff=0.5)
    assert np.isfinite(cmp["ratio_top_over_BM"])


def test_compare_throat_uses_correct_d_scaling():
    """rho_topological scales as 1/d^4."""
    cmp1 = compare_throat_to_brown_maclay(d=1.0, gamma_eff=0.5)
    cmp2 = compare_throat_to_brown_maclay(d=2.0, gamma_eff=0.5)
    assert cmp2["rho_topological"] / cmp1["rho_topological"] == pytest.approx(1 / 16, rel=1e-12)
