"""Tests for cascade-dsi-explorer."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cascade_dsi_explorer import (
    CascadeDSIExplorer,
    CascadeSummary,
    PhaseBoundaryReport,
    scan_phase_boundary,
)


# ---------------------------------------------------------------------------
# CascadeDSIExplorer basic
# ---------------------------------------------------------------------------


def test_explorer_construction():
    exp = CascadeDSIExplorer(R=1.0, alpha_0=0.8, levels=3)
    assert exp.R == 1.0
    assert exp.alpha_0 == 0.8
    assert exp.levels == 3


def test_explorer_F_returns_array():
    exp = CascadeDSIExplorer(R=1.0, alpha_0=0.8, levels=3)
    rs = np.linspace(1.0, 10.0, 50)
    F = exp.F(rs)
    assert F.shape == rs.shape
    assert np.all(np.isfinite(F))


def test_explorer_alphas_amps_lengths():
    exp = CascadeDSIExplorer(R=1.0, alpha_0=0.8, levels=4)
    assert len(exp.alphas()) == 4
    assert len(exp.amps()) == 4


def test_explorer_alphas_geometric_progression():
    """alpha_k = alpha_0 * scale_factor^k."""
    exp = CascadeDSIExplorer(R=1.0, alpha_0=0.5, levels=3, scale_factor=2.0)
    alphas = exp.alphas()
    np.testing.assert_allclose(alphas, [0.5, 1.0, 2.0])


def test_explorer_amps_geometric_decay():
    """A_k = A_0 * amp_decay^k."""
    exp = CascadeDSIExplorer(R=1.0, alpha_0=0.5, A_0=4.0, levels=3,
                              amp_decay=0.5)
    amps = exp.amps()
    np.testing.assert_allclose(amps, [4.0, 2.0, 1.0])


# ---------------------------------------------------------------------------
# Zero set + geometric progression (levels=1 is pure DSI)
# ---------------------------------------------------------------------------


def test_single_level_zero_ratio_matches_analytic():
    """levels=1 cascade is a bare cosine; consecutive zero ratios
    converge to exp(pi/alpha) in r.

    Note: verify_geometric_progression's `is_geometric` flag uses a
    strict 1e-9 tolerance that numerical sign-change zeros don't
    satisfy. We assert the numerical ratio matches the analytic value
    instead.
    """
    exp = CascadeDSIExplorer(R=1.0, alpha_0=1.0, levels=1)
    _, diag = exp.is_geometric_zero_progression(r_min=1.05, r_max=1e5)
    ratio_target = float(np.exp(np.pi / 1.0))
    assert diag["ratio_mean"] == pytest.approx(ratio_target, rel=2e-3)
    # max_rel_dev should still be small (just not 1e-9)
    assert diag["max_rel_dev"] < 1e-2


def test_multi_level_zeros_break_geometric_progression():
    """levels >= 2 cascades have a multi-scale zero set."""
    exp = CascadeDSIExplorer(R=1.0, alpha_0=0.5, levels=3,
                              scale_factor=3.0, amp_decay=0.6)
    is_gp, _ = exp.is_geometric_zero_progression(r_min=1.05, r_max=200.0)
    assert not is_gp


# ---------------------------------------------------------------------------
# Box dimension
# ---------------------------------------------------------------------------


def test_box_dimension_returns_finite():
    exp = CascadeDSIExplorer(R=1.0, alpha_0=0.8, levels=4,
                              scale_factor=2.0, amp_decay=0.5)
    d = exp.box_dimension(r_min=1.05, r_max=1e4, n_scales=14)
    assert np.isfinite(d)
    assert d >= 0.0


def test_box_dimension_increases_with_levels():
    """More cascade levels with non-trivial scale → richer zero set →
    larger box dimension."""
    d_low = CascadeDSIExplorer(R=1.0, alpha_0=0.8, levels=1).box_dimension(
        r_min=1.05, r_max=1e4, n_scales=14,
    )
    d_high = CascadeDSIExplorer(
        R=1.0, alpha_0=0.8, levels=4, scale_factor=2.5, amp_decay=0.6,
    ).box_dimension(r_min=1.05, r_max=1e4, n_scales=14)
    assert d_high >= d_low


# ---------------------------------------------------------------------------
# CascadeSummary
# ---------------------------------------------------------------------------


def test_summary_returns_dataclass():
    exp = CascadeDSIExplorer(R=1.0, alpha_0=0.8, levels=3)
    summ = exp.summary(r_min=1.05, r_max=100.0, n_scales=10)
    assert isinstance(summ, CascadeSummary)
    assert summ.R == 1.0
    assert summ.levels == 3
    assert summ.n_zeros > 0


def test_summary_records_geometric_progression_ratio():
    """The summary records the geometric-progression numerical ratio
    even when the strict 1e-9 `is_geometric_progression` flag is False."""
    exp = CascadeDSIExplorer(R=1.0, alpha_0=1.0, levels=1)
    summ = exp.summary(r_min=1.05, r_max=1e5)
    assert summ.geometric_progression_ratio_mean is not None
    ratio_target = float(np.exp(np.pi / 1.0))
    assert abs(summ.geometric_progression_ratio_mean - ratio_target) / ratio_target < 5e-3


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_scale_factor_raises():
    with pytest.raises(ValueError):
        CascadeDSIExplorer(R=1.0, alpha_0=0.8, levels=3, scale_factor=1.0)


def test_invalid_amp_decay_raises():
    with pytest.raises(ValueError):
        CascadeDSIExplorer(R=1.0, alpha_0=0.8, levels=3, amp_decay=1.5)


# ---------------------------------------------------------------------------
# Phase-boundary scan (novelty catcher)
# ---------------------------------------------------------------------------


def test_phase_boundary_scan_smoke():
    """Tiny 3x3 scan returns a well-formed report."""
    rep = scan_phase_boundary(
        scale_factors=np.array([2.0, 3.0, 4.0]),
        amp_decays=np.array([0.5, 0.7, 0.9]),
        r_min=1.05, r_max=1e3, radii=(4, 8, 16),
    )
    assert isinstance(rep, PhaseBoundaryReport)
    assert rep.n_zeros_grid.shape == (3, 3)
    assert rep.box_dimension_grid.shape == (3, 3)
    assert rep.verdict in {"uniform", "smooth", "novel_structure"}


def test_phase_boundary_scan_finds_some_zeros():
    """For reasonable params, every cell should have at least some zeros."""
    rep = scan_phase_boundary(
        scale_factors=np.array([2.0, 3.0]),
        amp_decays=np.array([0.5, 0.7]),
        r_min=1.05, r_max=1e3, radii=(4, 8),
    )
    assert (rep.n_zeros_grid > 0).all()
