"""Tests for the N-cylinder beamforming inverse."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from beamforming_inverse import (
    BeamformingDesign,
    BeamformingInverseResult,
    build_forward_matrix,
    design_from_array,
    solve_beamforming_inverse,
    synthesised_array,
)
from systrophe.geometry.array import SystropheArray
from systrophe.geometry.sinusoid import TiplerSinusoid
from systrophe.geometry.vanstockum import VanStockumInterior


# ---------------------------------------------------------------------------
# build_forward_matrix
# ---------------------------------------------------------------------------


def _example_design(N=3) -> BeamformingDesign:
    R = np.array([1.0] * N, dtype=float)
    a = np.linspace(0.6, 0.9, N)  # different a → different alpha → distinct basis cols
    alpha = np.sqrt(4 * a * a - 1)
    p = np.ones(N)  # match TiplerSinusoid default
    return BeamformingDesign(R=R, alpha=alpha, a=a, p=p)


def test_forward_matrix_shape():
    design = _example_design(N=3)
    rs = np.linspace(1.05, 5.0, 50)
    G = build_forward_matrix(design, rs)
    assert G.shape == (50, 3)
    assert G.dtype == complex


def test_forward_matrix_unit_modulus():
    """|G[m,i]| = 1 for every entry (pure phase)."""
    design = _example_design(N=4)
    rs = np.linspace(1.05, 10.0, 30)
    G = build_forward_matrix(design, rs)
    np.testing.assert_allclose(np.abs(G), 1.0, atol=1e-12)


def test_forward_matrix_matches_systrophe_array_phasor():
    """G c reproduces SystropheArray.phasor_field(r) when (c = A_i e^(i δ_i)).
    """
    cyl_params = [(1.0, 0.6, 1.5, 0.2),
                   (1.0, 0.75, 0.8, -0.3),
                   (1.0, 0.9, 0.5, 1.1)]
    sinusoids = tuple(
        TiplerSinusoid(R=R, a=a, A=A, delta=d, p=1.0)
        for (R, a, A, d) in cyl_params
    )
    array = SystropheArray(sinusoids=sinusoids)
    design = design_from_array(array)
    c = np.array([s.A * np.exp(1j * s.delta) for s in sinusoids])

    rs = np.linspace(1.05, 5.0, 50)
    z_from_G = build_forward_matrix(design, rs) @ c
    z_from_array = array.phasor_field(rs)["phasor"]
    np.testing.assert_allclose(z_from_G, z_from_array, atol=1e-12)


# ---------------------------------------------------------------------------
# design_from_array
# ---------------------------------------------------------------------------


def test_design_from_array_roundtrip():
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    array = SystropheArray.uniform_phase_comb(cyl, N=3)
    design = design_from_array(array)
    assert design.N == 3
    np.testing.assert_allclose(design.R, [1.0, 1.0, 1.0])
    # All same source → all same alpha
    assert np.std(design.alpha) < 1e-12


# ---------------------------------------------------------------------------
# solve_beamforming_inverse: synthesise-then-recover
# ---------------------------------------------------------------------------


def test_recover_uniform_phase_comb():
    """The 3-comb has c_i = A * exp(i 2π i/3); the inverse should
    recover those amplitudes up to a single global phase or constant.

    Because all 3 cylinders are matched (same alpha), the forward
    matrix's columns are identical and rank-1. The min-norm solution
    distributes the target uniformly across columns. The TEST that
    survives this: G c_recovered ≈ z_target.
    """
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    array = SystropheArray.uniform_phase_comb(cyl, N=3)
    design = design_from_array(array)

    rs = np.linspace(1.05, 5.0, 50)
    z_target = array.phasor_field(rs)["phasor"]
    # 3-comb phasor sum is identically 0 (root-of-unity extinction)
    assert np.max(np.abs(z_target)) < 1e-10

    result = solve_beamforming_inverse(design, rs, z_target)
    # Should be a degenerate (rank-1) system; residual should match target (= 0)
    assert result.residual_norm < 1e-10


def test_recover_distinct_alpha_array():
    """When alpha_i differ, columns are linearly independent. Recovery
    of c should be tight."""
    cyl_params = [(1.0, 0.6, 1.5, 0.2),
                   (1.0, 0.75, 0.8, -0.3),
                   (1.0, 0.9, 0.5, 1.1)]
    sinusoids = tuple(
        TiplerSinusoid(R=R, a=a, A=A, delta=d, p=1.0)
        for (R, a, A, d) in cyl_params
    )
    array = SystropheArray(sinusoids=sinusoids)
    design = design_from_array(array)

    rs = np.linspace(1.05, 8.0, 100)  # overdetermined: M=100, N=3
    z_target = array.phasor_field(rs)["phasor"]
    result = solve_beamforming_inverse(design, rs, z_target)
    assert result.is_overdetermined
    assert result.rank == 3
    assert result.relative_residual < 1e-10

    c_true = np.array([s.A * np.exp(1j * s.delta) for s in sinusoids])
    np.testing.assert_allclose(result.cylinder_amplitudes, c_true, atol=1e-8)


def test_recover_distinct_alpha_amplitudes_and_phases():
    """Recovered A_i and delta_i match the planted values."""
    cyl_params = [(1.0, 0.6, 1.5, 0.2),
                   (1.0, 0.75, 0.8, -0.3),
                   (1.0, 0.9, 0.5, 1.1)]
    sinusoids = tuple(
        TiplerSinusoid(R=R, a=a, A=A, delta=d, p=1.0)
        for (R, a, A, d) in cyl_params
    )
    array = SystropheArray(sinusoids=sinusoids)
    design = design_from_array(array)
    rs = np.linspace(1.05, 8.0, 100)
    z_target = array.phasor_field(rs)["phasor"]
    result = solve_beamforming_inverse(design, rs, z_target)
    np.testing.assert_allclose(result.A, [1.5, 0.8, 0.5], atol=1e-6)
    np.testing.assert_allclose(result.delta, [0.2, -0.3, 1.1], atol=1e-6)


def test_synthesised_array_reproduces_target_field():
    """End-to-end: a SystropheArray built from solve_beamforming_inverse
    reproduces z_target when re-evaluated at the sample radii."""
    cyl_params = [(1.0, 0.6, 1.5, 0.2),
                   (1.0, 0.75, 0.8, -0.3),
                   (1.0, 0.9, 0.5, 1.1)]
    original_sinusoids = tuple(
        TiplerSinusoid(R=R, a=a, A=A, delta=d, p=1.0)
        for (R, a, A, d) in cyl_params
    )
    original_array = SystropheArray(sinusoids=original_sinusoids)
    design = design_from_array(original_array)
    rs = np.linspace(1.05, 8.0, 100)
    z_target = original_array.phasor_field(rs)["phasor"]
    result = solve_beamforming_inverse(design, rs, z_target)
    synth = synthesised_array(design, result)
    z_synth = synth.phasor_field(rs)["phasor"]
    np.testing.assert_allclose(z_synth, z_target, atol=1e-8)


# ---------------------------------------------------------------------------
# Underdetermined / overdetermined behaviour
# ---------------------------------------------------------------------------


def test_underdetermined_min_norm_solution():
    """N=5 cylinders, M=2 sample radii → underdetermined min-norm.

    A min-norm solution must (a) interpolate z_target exactly at those
    2 radii and (b) have the smallest ||c||_2 of any such solution.
    """
    a_vals = np.linspace(0.6, 0.95, 5)
    alpha = np.sqrt(4 * a_vals * a_vals - 1)
    design = BeamformingDesign(
        R=np.ones(5), alpha=alpha, a=a_vals, p=np.ones(5),
    )
    rs = np.array([1.5, 4.0])
    z_target = np.array([2.0 + 1.0j, -0.5 + 0.3j], dtype=complex)
    result = solve_beamforming_inverse(design, rs, z_target)
    # Underdetermined: residual should be ~ 0 (exact interpolation)
    assert not result.is_overdetermined
    assert result.relative_residual < 1e-12
    assert result.rank == 2  # rank limited by M=2


def test_overdetermined_least_squares():
    """N=2, M=50. Generally no exact interpolation; should fit best."""
    design = _example_design(N=2)
    rs = np.linspace(1.05, 5.0, 50)
    z_target = np.array([np.sin(np.log(r)) + 1j * np.cos(np.log(r))
                          for r in rs], dtype=complex)
    result = solve_beamforming_inverse(design, rs, z_target)
    assert result.is_overdetermined
    assert result.rank == 2
    assert result.residual_norm >= 0
    # Should be far from zero residual; this isn't an array-realisable target
    assert result.relative_residual > 0.0


# ---------------------------------------------------------------------------
# Shape / API errors
# ---------------------------------------------------------------------------


def test_shape_mismatch_raises():
    design = _example_design(N=3)
    with pytest.raises(ValueError):
        solve_beamforming_inverse(design, np.array([1.0, 2.0]),
                                    np.array([1+1j]))


def test_result_dtype():
    design = _example_design(N=3)
    rs = np.linspace(1.05, 5.0, 30)
    z_target = np.exp(1j * np.log(rs))
    result = solve_beamforming_inverse(design, rs, z_target)
    assert isinstance(result, BeamformingInverseResult)
    assert result.cylinder_amplitudes.dtype == complex
    assert result.A.dtype == float
    assert result.delta.dtype == float
