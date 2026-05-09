"""Tests for the Dinos bridge (cylindrical-Kerr correspondence + Z3 mode match)."""

import importlib.util

import numpy as np
import pytest

from systrophe.vanstockum import VanStockumInterior

# Skip the entire module if dinos is not importable (z3 missing, etc.)
_DINOS_AVAILABLE = importlib.util.find_spec("z3") is not None
if not _DINOS_AVAILABLE:
    pytest.skip("z3-solver not installed; Dinos imports unavailable", allow_module_level=True)

# Probe dinos availability (caller may set SYSTROPHE_DINOS_PATH or have dinos on PYTHONPATH).
import os
import sys

_dinos_path = os.environ.get("SYSTROPHE_DINOS_PATH")
if _dinos_path:
    sys.path.insert(0, _dinos_path)
try:
    import dinos  # noqa: F401
except Exception:
    pytest.skip(
        "Dinos package not importable; set SYSTROPHE_DINOS_PATH or install dinos to enable bridge tests",
        allow_module_level=True,
    )

from systrophe.dinos_bridge import (  # noqa: E402
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
