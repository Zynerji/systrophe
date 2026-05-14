"""Tests for the Dinos bridge (cylindrical-Kerr correspondence + Z3 mode match).

The bridge module resolves its `dinos` dependency in this order:
  1. Externally-installed dinos package (pip install or SYSTROPHE_DINOS_PATH).
  2. The vendored subset shipped at `systrophe/_dinos_vendored/`
     (kerr_corrections + mobius_z3_cover only).

Either path keeps these tests unconditional -- previously they
skipped at the module level when neither resolution mechanism was
available. The vendored fallback covers every Dinos symbol the 6
tests below need.
"""

import numpy as np
import pytest

from systrophe.vanstockum import VanStockumInterior
from systrophe.dinos_bridge import (
    CylindricalKerrMapping,
    kerr_correction_at_tipler_threshold,
    map_to_dinos_kerr,
    z3_branch_match_to_tipler_alpha,
)


def test_kerr_mapping_identifies_tau_with_omega_R():
    """tau = omega * R is the Kerr 'a' under the Dinos identification."""
    vs = VanStockumInterior(omega=1.5, R=2.0)
    m = map_to_dinos_kerr(vs)
    assert isinstance(m, CylindricalKerrMapping)
    assert m.tau == pytest.approx(3.0)
    assert m.a_kerr == pytest.approx(3.0)
    assert m.m_j == 0.0
    assert m.beta_plus_kappa == 0.0


def test_kerr_correction_zero_on_shell():
    """For m_j = 0 = mu (vacuum, scalar), on-shell condition gives zero shift."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    out = kerr_correction_at_tipler_threshold(vs)
    assert out["is_on_shell"] is True
    assert out["cp_shift"] == pytest.approx(0.0, abs=1e-12)


def test_z3_branch_zero_matches_tipler_fundamental():
    """The Z3 branch=0 lowest-mode eigenvalue equals the Tipler log-grid mode."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    out = z3_branch_match_to_tipler_alpha(vs, N=24)
    z3 = out["z3_eigenvalues"]
    assert out["best_branch_match"] == 0
    assert out["relative_residual"] == pytest.approx(0.0, abs=1e-12)
    expected_tipler = 2.0 * (1.0 - np.cos(2.0 * np.pi / 24))
    assert out["tipler_eigenvalue"] == pytest.approx(expected_tipler)


def test_z3_branch_one_lower_than_branch_zero_at_fundamental():
    """Non-trivial Z3 branches (1, 2) have lower fundamental eigenvalue than branch 0.

    This is the structural analog of the SystrophePair phase offset:
    branch != 0 corresponds to a 1/3 phase advance per N-step in the
    cyclic group, identifiable with the off-set sinusoid sector.
    """
    vs = VanStockumInterior(omega=1.5, R=1.0)
    out = z3_branch_match_to_tipler_alpha(vs, N=12)
    e0, e1, e2 = out["z3_eigenvalues"]
    assert e1 < e0
    assert e2 < e0
    # branches 1 and 2 are degenerate (complex conjugate pair)
    assert e1 == pytest.approx(e2)


def test_subcritical_z3_match_rejected():
    """Z3 / Tipler comparison requires supercritical input."""
    vs = VanStockumInterior(omega=0.3, R=1.0)
    with pytest.raises(ValueError):
        z3_branch_match_to_tipler_alpha(vs, N=12)


def test_z3_match_n_floor():
    """N must be at least 4 to be meaningful."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    with pytest.raises(ValueError):
        z3_branch_match_to_tipler_alpha(vs, N=2)
