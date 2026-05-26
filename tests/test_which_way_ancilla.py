"""Tests for the which-way ancilla module."""

import math

import numpy as np
import pytest

from systrophe.which_way_ancilla import (
    englert_relation,
    entangle_path_ancilla,
    fringe_intensity_with_ancilla,
    path_coherence,
    reduced_path_density,
    sweep_coupling,
    which_way_distinguishability,
)


def test_coupling_zero_is_full_visibility():
    assert path_coherence(0.0) == pytest.approx(1.0, abs=1e-12)
    assert which_way_distinguishability(0.0) == pytest.approx(0.0, abs=1e-12)


def test_coupling_one_kills_fringes():
    assert path_coherence(1.0) == pytest.approx(0.0, abs=1e-12)
    assert which_way_distinguishability(1.0) == pytest.approx(1.0, abs=1e-12)


def test_half_coupling_matches_cosine_sine_half():
    V = path_coherence(0.5)
    D = which_way_distinguishability(0.5)
    assert V == pytest.approx(math.cos(math.pi / 4.0), rel=1e-12)
    assert D == pytest.approx(math.sin(math.pi / 4.0), rel=1e-12)


@pytest.mark.parametrize("kappa", [0.0, 0.1, 0.25, 0.5, 0.73, 0.9, 1.0])
def test_englert_saturated_for_pure_ancilla(kappa):
    res = englert_relation(kappa)
    assert res["V2_plus_D2"] == pytest.approx(1.0, abs=1e-12)
    assert res["slack"] == pytest.approx(0.0, abs=1e-12)


def test_joint_state_is_normalised():
    psi = entangle_path_ancilla(0.6)["psi"]
    assert np.vdot(psi, psi).real == pytest.approx(1.0, abs=1e-12)


def test_reduced_density_is_valid():
    rho = reduced_path_density(0.42)
    assert rho.shape == (2, 2)
    assert np.allclose(rho, rho.conj().T, atol=1e-12)
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-12)
    eigs = np.linalg.eigvalsh(rho)
    assert eigs.min() >= -1e-12
    assert eigs.max() <= 1.0 + 1e-12


def test_reduced_density_form():
    """rho_path == 0.5 * (I + cos(theta/2) * sigma_x)."""
    kappa = 0.37
    theta = math.pi * kappa
    expected = 0.5 * np.array(
        [[1.0, math.cos(theta / 2.0)],
         [math.cos(theta / 2.0), 1.0]],
    )
    assert np.allclose(reduced_path_density(kappa), expected, atol=1e-12)


def test_coupling_out_of_range_raises():
    with pytest.raises(ValueError):
        path_coherence(-0.1)
    with pytest.raises(ValueError):
        path_coherence(1.1)


def test_visibility_monotone_decreasing_in_coupling():
    kappas = np.linspace(0.0, 1.0, 11)
    Vs = [path_coherence(float(k)) for k in kappas]
    diffs = np.diff(Vs)
    assert np.all(diffs <= 1e-12)


def test_distinguishability_monotone_increasing_in_coupling():
    kappas = np.linspace(0.0, 1.0, 11)
    Ds = [which_way_distinguishability(float(k)) for k in kappas]
    diffs = np.diff(Ds)
    assert np.all(diffs >= -1e-12)


def test_sweep_arrays_consistent():
    sw = sweep_coupling(n=15)
    assert sw["coupling"].shape == (15,)
    assert sw["V"].shape == (15,)
    assert sw["D"].shape == (15,)
    assert np.allclose(sw["V2_plus_D2"], 1.0, atol=1e-12)


def test_fringe_intensity_uses_visibility():
    # y range spans > one fringe period (period = pi * D_arm / (k * d) ~ 0.87).
    y = np.linspace(-0.6, 0.6, 401)
    k, d, D_arm = 60.0, 0.3, 5.0
    I_full = fringe_intensity_with_ancilla(y, k, d, D_arm, 0.0, coupling=0.0)
    I_half = fringe_intensity_with_ancilla(y, k, d, D_arm, 0.0, coupling=0.5)
    I_none = fringe_intensity_with_ancilla(y, k, d, D_arm, 0.0, coupling=1.0)
    # Full coupling kills fringes: intensity is flat at 2.
    assert np.allclose(I_none, 2.0, atol=1e-12)
    # Full coherence: peak-to-trough span is 4 (modulo grid-sampling slop).
    assert I_full.max() - I_full.min() == pytest.approx(4.0, abs=1e-3)
    # Half: span equals 4 * cos(pi/4) = 2 * sqrt(2).
    assert I_half.max() - I_half.min() == pytest.approx(
        4.0 * math.cos(math.pi / 4.0), abs=1e-3,
    )
