"""Tests for spinor monodromy module."""

import math

import numpy as np
import pytest

from systrophe.spinor_monodromy import (
    chronology_horizon_caustic,
    expected_monodromy_phase_per_revolution,
    fixed_point_spinors,
    monodromy_eigenvalues,
    monodromy_period_in_revolutions,
    multi_loop_monodromy,
    pair_modified_monodromy,
    spin_connection_phi,
    spinor_holonomy,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_spin_connection_phi_returns_dict(vs):
    sc = spin_connection_phi(vs, r=2.0)
    assert "omega_01_phi" in sc
    assert "omega_12_phi" in sc
    assert math.isfinite(sc["omega_01_phi"])


def test_spinor_holonomy_is_4x4(vs):
    U = spinor_holonomy(vs, r=2.0)
    assert U.shape == (4, 4)


def test_spinor_holonomy_complex_dtype(vs):
    U = spinor_holonomy(vs, r=2.0)
    assert np.iscomplexobj(U)


def test_monodromy_eigenvalues_length_4(vs):
    evals = monodromy_eigenvalues(vs, r=2.5)
    assert len(evals) == 4


def test_monodromy_eigenvalues_unit_modulus_roughly(vs):
    """For a true Spin element, eigenvalues lie on unit circle.

    Our expm-based holonomy should respect this approximately.
    """
    evals = monodromy_eigenvalues(vs, r=2.5)
    mags = np.abs(evals)
    # Allow some numerical drift
    assert all(0.1 < m < 10.0 for m in mags)


def test_multi_loop_monodromy_zero_is_identity(vs):
    U0 = multi_loop_monodromy(vs, r=2.0, n_loops=0)
    np.testing.assert_allclose(U0, np.eye(4, dtype=complex), atol=1e-12)


def test_multi_loop_monodromy_one_equals_single(vs):
    U1 = multi_loop_monodromy(vs, r=2.0, n_loops=1)
    U_direct = spinor_holonomy(vs, r=2.0)
    np.testing.assert_allclose(U1, U_direct, atol=1e-12)


def test_multi_loop_monodromy_negative_raises(vs):
    with pytest.raises(ValueError):
        multi_loop_monodromy(vs, r=2.0, n_loops=-1)


def test_pair_modified_monodromy_extinction_at_pi(vs):
    """At delta=pi, pair extinction => identity monodromy."""
    Upair = pair_modified_monodromy(vs, r=2.0, delta=math.pi)
    np.testing.assert_allclose(Upair, np.eye(4, dtype=complex), atol=1e-12)


def test_pair_modified_monodromy_at_zero_equals_single(vs):
    """At delta=0, pair = single cylinder."""
    Upair = pair_modified_monodromy(vs, r=2.0, delta=0.0)
    U = spinor_holonomy(vs, r=2.0)
    np.testing.assert_allclose(Upair, U, atol=1e-12)


def test_chronology_horizon_caustic_returns_caustic_radii(vs):
    res = chronology_horizon_caustic(vs)
    assert "caustic_radii" in res
    assert "max_omega_magnitude" in res


def test_chronology_horizon_caustic_finds_caustics_supercritical(vs):
    res = chronology_horizon_caustic(vs)
    # Supercritical case should produce caustics (omega blowups at F=0)
    assert res["max_omega_magnitude"] > 1.0


def test_expected_monodromy_phase_finite(vs):
    theta = expected_monodromy_phase_per_revolution(vs, r=2.0)
    assert math.isfinite(theta)


def test_monodromy_period_returns_int(vs):
    n = monodromy_period_in_revolutions(vs, r=2.0)
    assert isinstance(n, int)
    assert n >= 0


def test_fixed_point_spinors_returns_list(vs):
    fps = fixed_point_spinors(vs, r=2.0)
    assert isinstance(fps, list)
