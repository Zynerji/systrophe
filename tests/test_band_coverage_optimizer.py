"""Tests for the nested-cylinder CTC-band coverage optimizer."""

import numpy as np
import pytest

from systrophe.band_coverage_optimizer import (
    CoverageOptResult,
    covered_fraction,
    optimize_offsets,
    out_of_band_fraction,
    summarise_coverage,
)


def test_covered_fraction_merges_overlaps():
    # Two overlapping bands over [0, 10] cover [1, 6] = 5/10.
    bands = [(1.0, 4.0), (3.0, 6.0)]
    assert covered_fraction(bands, 0.0, 10.0) == pytest.approx(0.5)


def test_covered_fraction_clips_to_route():
    bands = [(-5.0, 5.0)]  # extends beyond route on the left
    assert covered_fraction(bands, 0.0, 10.0) == pytest.approx(0.5)


def test_out_of_band_in_unit_interval():
    f = out_of_band_fraction((1.0, 3.0, 9.0), (0.0, 0.0, 0.0))
    assert 0.0 <= f <= 1.0


def test_uniform_comb_extinguishes_more_than_single():
    """Sanity: the uniform phase comb (bands cancel) leaves MORE of the route
    unpaid than a single cylinder -- the interference is real, so optimization
    is necessary, not free."""
    n = 4
    radii = (1.0, 3.0, 9.0, 27.0)
    comb = out_of_band_fraction(radii, [2 * np.pi * i / n for i in range(n)])
    single = out_of_band_fraction((1.0,), (0.0,))
    assert comb > single


def test_optimizer_beats_aligned_baseline():
    """The core claim: optimizing phase offsets reduces the out-of-band
    (paid) fraction below naive aligned nesting."""
    res = optimize_offsets(radii=(1.0, 3.0, 9.0, 27.0), seed=1, n_pass=6)
    assert isinstance(res, CoverageOptResult)
    assert res.best_out_of_band < res.aligned_out_of_band
    assert res.best_out_of_band < res.single_cylinder_out_of_band


def test_history_is_monotone_non_increasing():
    res = optimize_offsets(radii=(1.0, 3.0, 9.0), seed=2, n_pass=6)
    h = res.history
    assert all(b <= a + 1e-12 for a, b in zip(h, h[1:]))


def test_coverage_saturates_and_is_phase_tuning():
    """Honest finding: coverage saturates with N and is dominated by
    single-cylinder phase tuning (a single optimized cylinder already reaches
    the saturated coverage)."""
    from systrophe.band_coverage_optimizer import coverage_scaling
    sc = coverage_scaling(max_cylinders=4, n_pass=5)
    assert sc["saturates"] is True
    assert sc["single_cylinder_optimized"] < 0.25  # one cylinder already ~good


def test_summary_and_novelty_present():
    res = optimize_offsets(radii=(1.0, 3.0, 9.0), seed=3, n_pass=4)
    s = summarise_coverage(res)
    assert "out-of-band" in s
    assert res.novelty_verdict in {"novel_structure", "smooth", "uniform"}
