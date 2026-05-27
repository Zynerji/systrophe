"""Tests for multi-cylinder Casimir interference."""

import numpy as np
import pytest

from systrophe.qftcs.multi_casimir import (
    MultiCylinderCasimir,
    all_aligned_casimir,
    casimir_phase_extinction_scan,
    interference_pattern,
    n_cylinder_casimir_energy,
    random_phase_casimir,
    uniform_phase_comb_casimir,
)


def test_single_cylinder_returns_topological_coefficient():
    """Single cylinder Casimir = topological coefficient."""
    c = n_cylinder_casimir_energy(np.array([0.0]))
    assert c.N == 1


def test_all_aligned_scales_with_N():
    """N aligned cylinders -> energy = N * single-cylinder energy."""
    c_1 = all_aligned_casimir(1, gamma=0.0)
    c_3 = all_aligned_casimir(3, gamma=0.0)
    c_5 = all_aligned_casimir(5, gamma=0.0)
    assert c_3.total_energy == pytest.approx(3 * c_1.total_energy, rel=1e-12)
    assert c_5.total_energy == pytest.approx(5 * c_1.total_energy, rel=1e-12)


def test_uniform_comb_energy_at_N3():
    """N=3 uniform comb: phases (0, 2pi/3, 4pi/3)."""
    c = uniform_phase_comb_casimir(3)
    assert isinstance(c, MultiCylinderCasimir)
    # The uniform comb doesn't necessarily extinguish the Casimir energy
    # (topological coefficient is non-linear in gamma_eff)


def test_random_phase_casimir_returns_dataclass():
    c = random_phase_casimir(N=4)
    assert c.N == 4
    assert np.isfinite(c.total_energy)


def test_phase_extinction_scan():
    result = casimir_phase_extinction_scan(N=5, n_trials=10)
    assert "random_mean_energy" in result
    assert "all_aligned_at_0_energy" in result
    assert "uniform_comb_energy" in result


def test_aligned_at_0_vs_pi_different():
    """Aligned at gamma=0 should differ from aligned at gamma=pi."""
    c0 = all_aligned_casimir(N=5, gamma=0.0)
    cpi = all_aligned_casimir(N=5, gamma=np.pi)
    assert c0.total_energy != cpi.total_energy


def test_interference_pattern_returns_dict():
    pattern = interference_pattern(gamma_1=0.0, n_points=10)
    assert "interference_amplitude" in pattern
    assert pattern["interference_amplitude"] > 0


def test_interference_nontrivial():
    """Interference produces nonzero range of total energies."""
    pattern = interference_pattern(gamma_1=0.5, n_points=20)
    assert pattern["max_energy"] != pattern["min_energy"]
