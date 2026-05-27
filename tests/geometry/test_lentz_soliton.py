"""Tests for the Lentz subluminal warp soliton."""

import numpy as np

from systrophe.geometry.lentz_soliton import (
    lentz_NEC_radial,
    lentz_energy_density,
    lentz_shift_vector,
    novelty_scan,
)


def test_shift_vanishes_at_x_zero():
    """tanh(0) = 0 so N^x at x=0 is zero."""
    assert abs(lentz_shift_vector(0.0, 1.0)) < 1e-10


def test_shift_vanishes_at_large_rho():
    """sech decays so N^x -> 0 far from bubble."""
    assert abs(lentz_shift_vector(1.0, 50.0)) < 1e-10


def test_energy_density_nonnegative():
    """Lentz claim: T_{tt} >= 0 everywhere."""
    xs = np.linspace(-2, 2, 11)
    rhos = np.linspace(0, 3, 11)
    for x in xs:
        for rho in rhos:
            val = lentz_energy_density(float(x), float(rho), v_s=0.5)
            assert val >= -1e-10


def test_NEC_subluminal_nonnegative():
    """NEC >= 0 at subluminal v_s."""
    xs = np.linspace(-2, 2, 11)
    rhos = np.linspace(0, 3, 11)
    for x in xs:
        for rho in rhos:
            val = lentz_NEC_radial(float(x), float(rho), v_s=0.5)
            assert val >= -1e-10


def test_NEC_superluminal_may_be_negative():
    """At v_s > 1 the cross-term turns on; min over a grid is negative."""
    xs = np.linspace(-2, 2, 21)
    rhos = np.linspace(0, 3, 11)
    vals = []
    for x in xs:
        for rho in rhos:
            vals.append(lentz_NEC_radial(float(x), float(rho), v_s=2.0))
    assert min(vals) <= 0


def test_novelty_scan_returns_verdict():
    res = novelty_scan(v_s_range=(0.2, 1.5), n_vs=8,
                        n_x=11, n_rho=11)
    assert "novelty_verdict" in res
    assert res["novelty_verdict"] in ("smooth", "uniform", "novel_structure")
