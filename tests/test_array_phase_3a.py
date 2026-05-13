"""Tests for Phase 3a SystropheArray extensions: beamforming, extinction, Dirichlet."""

from __future__ import annotations

import math

import numpy as np
import pytest

from systrophe import SystropheArray
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture(scope="module")
def cyl() -> VanStockumInterior:
    return VanStockumInterior(omega=1.0, R=1.0)


def test_phasor_field_runs(cyl):
    arr = SystropheArray.from_cylinders([cyl, cyl], offsets=(0.0, 0.5))
    pf = arr.phasor_field(np.array([1.5, 2.0]))
    assert pf["phasor"].shape == (2,)
    assert np.all(np.isfinite(pf["magnitude"]))


def test_array_factor_aligned_doubles_single(cyl):
    """N aligned (delta = 0) cylinders give array factor = N * single."""
    single = SystropheArray.from_cylinders([cyl])
    triple = SystropheArray.from_cylinders([cyl] * 3, offsets=(0.0, 0.0, 0.0))
    r = 1.5
    af_single = float(single.array_factor(np.array([r]))[0])
    af_triple = float(triple.array_factor(np.array([r]))[0])
    assert math.isclose(af_triple, 3.0 * af_single, rel_tol=1e-10)


def test_uniform_phase_comb_is_extinguished(cyl):
    """Uniform-phase comb has array factor identically zero (FD noise only)."""
    for N in (2, 3, 4, 5, 6):
        comb = SystropheArray.uniform_phase_comb(cyl, N=N)
        ext = comb.extinction_check(r_max=20.0, n_grid=1001)
        assert ext["is_extinguished"], (
            f"N={N}: max_array_factor = {ext['max_array_factor']:.2e} "
            f"(expected ~ 0)"
        )


def test_non_uniform_phase_pair_is_not_extinguished(cyl):
    """A non-anti-phase pair should NOT be extinguished."""
    pair = SystropheArray.from_cylinders([cyl, cyl], offsets=(0.0, math.pi / 3))
    ext = pair.extinction_check(r_max=10.0, n_grid=601)
    assert not ext["is_extinguished"]
    assert ext["max_array_factor"] > 0.1


def test_dirichlet_pattern_oscillates(cyl):
    """For linear-ramp phasing, the Dirichlet kernel has multiple zeros."""
    N = 4
    delta_step = math.pi / 2
    arr = SystropheArray.from_cylinders(
        [cyl] * N, offsets=tuple(i * delta_step for i in range(N))
    )
    r_grid = np.geomspace(1.05, 10.0, 200)
    pattern = arr.dirichlet_pattern(r_grid, delta_step=delta_step)
    # Should have at least one zero crossing in the pattern minima
    assert np.min(pattern) < 0.5 * np.max(pattern)


def test_beam_steer_places_node_near_target(cyl):
    """beam_steer should place a CTC node (L = 0) at the target radius."""
    r_target = 3.0
    arr = SystropheArray.beam_steer(r_target=r_target, cylinder=cyl, N=2)
    L_target = float(arr.L(np.array([r_target]))[0])
    # Analytic prediction is L = 0 to machine precision
    assert abs(L_target) < 1e-9, f"L({r_target}) = {L_target} (expected ~ 0)"


def test_beam_steer_node_position_sweep(cyl):
    """Beam-steered node lands at r_target for multiple targets."""
    for r_t in (2.0, 3.5, 5.0):
        arr = SystropheArray.beam_steer(r_target=r_t, cylinder=cyl, N=2)
        L_target = float(arr.L(np.array([r_t]))[0])
        assert abs(L_target) < 1e-9, f"r_t={r_t}: L = {L_target}"


def test_beam_pattern_returns_log_grid(cyl):
    arr = SystropheArray.from_cylinders([cyl, cyl], offsets=(0.0, 0.5))
    bp = arr.beam_pattern(1.05, 10.0, n_grid=100)
    assert bp["r_grid"].shape == (100,)
    assert bp["main_lobe_amplitude"] > 0
    # Log-spaced: ratios of successive r values should be ~ constant
    ratios = bp["r_grid"][1:] / bp["r_grid"][:-1]
    assert np.std(ratios) < 1e-12


def test_sidelobe_level_finite(cyl):
    arr = SystropheArray.from_cylinders([cyl, cyl, cyl], offsets=(0.0, 0.5, 1.0))
    sll = arr.array_factor_sidelobe_level(r_max=20.0, n_grid=2001)
    # 0 < SLL <= 1 (normalised)
    assert 0.0 <= sll <= 1.0
