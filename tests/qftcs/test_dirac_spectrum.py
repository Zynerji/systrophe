"""Dirac spectrum / bound-state tests."""

import numpy as np
import pytest

from systrophe import VanStockumInterior
from systrophe.qftcs.dirac_spectrum import (
    boundary_functional,
    find_bound_states,
    vanstockum_bound_states,
)


def test_boundary_functional_returns_finite_real():
    """boundary_functional returns a finite non-negative real."""
    F_fn = lambda r: 1.0
    K_fn = lambda r: 0.0
    L_fn = lambda r: r ** 2
    h_fn = lambda r: 1.0
    val = boundary_functional(
        F_fn, K_fn, L_fn, h_fn,
        E=1.0, m=0, k=0, mass=0.5,
        r_min=1.0, r_max=2.0,
    )
    assert np.isfinite(val)
    assert val >= 0


def test_find_bound_states_returns_array():
    """find_bound_states returns an ndarray (possibly empty)."""
    F_fn = lambda r: 1.0
    K_fn = lambda r: 0.0
    L_fn = lambda r: r ** 2
    h_fn = lambda r: 1.0
    energies = find_bound_states(
        F_fn, K_fn, L_fn, h_fn,
        m=0, k=0, mass=0.5,
        r_min=1.0, r_max=4.0,
        E_min=0.5, E_max=3.0, n_E=80,
    )
    assert isinstance(energies, np.ndarray)


def test_vanstockum_bound_states_supercritical():
    """Bound-state finder runs for the Tipler exterior."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    out = vanstockum_bound_states(vs, m=0, k=0, mass=0.5, r_max_factor=3.0,
                                  E_min=0.5, E_max=3.0, n_E=60)
    assert "energies" in out
    assert "n_bound" in out
    assert isinstance(out["energies"], np.ndarray)
    assert out["n_bound"] >= 0


def test_vanstockum_bound_states_rejects_subcritical():
    vs_sub = VanStockumInterior(omega=0.3, R=1.0)
    with pytest.raises(ValueError):
        vanstockum_bound_states(vs_sub, m=0, k=0, mass=0.5)


def test_boundary_functional_continuous_in_E():
    """Boundary functional should be continuous in E (small E change -> small change)."""
    F_fn = lambda r: 1.0
    K_fn = lambda r: 0.0
    L_fn = lambda r: r ** 2
    h_fn = lambda r: 1.0
    f1 = boundary_functional(F_fn, K_fn, L_fn, h_fn, E=1.0, m=0, k=0, mass=0.5,
                             r_min=1.0, r_max=2.0)
    f2 = boundary_functional(F_fn, K_fn, L_fn, h_fn, E=1.001, m=0, k=0, mass=0.5,
                             r_min=1.0, r_max=2.0)
    assert abs(f1 - f2) / max(abs(f1), abs(f2), 1e-9) < 0.5
