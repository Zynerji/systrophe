"""Tests for the Knopp Drive traversal engineering deliverable."""

import math

from systrophe.knopp.knopp_drive import KnoppDriveConfig
from systrophe.knopp.knopp_traversal import (
    KnoppTraversalReport,
    knopp_traversal,
    summarise_traversal,
)


def test_traversal_returns_report():
    rep = knopp_traversal(distance=8.0, n_steps=30)
    assert isinstance(rep, KnoppTraversalReport)


def test_traversal_has_all_fields():
    rep = knopp_traversal(distance=8.0, n_steps=30)
    assert hasattr(rep, "coord_time_total")
    assert hasattr(rep, "exotic_matter_total")
    assert hasattr(rep, "total_energy_budget")
    assert hasattr(rep, "pfenning_ford_compatible")


def test_inside_band_fraction_in_range():
    rep = knopp_traversal(distance=12.0, n_steps=50)
    assert 0.0 <= rep.inside_band_fraction <= 1.0


def test_exotic_matter_nonneg():
    rep = knopp_traversal(distance=12.0, n_steps=50)
    assert rep.exotic_matter_total >= 0.0


def test_higher_Q_reduces_total_energy():
    cfg_lo = KnoppDriveConfig(Q=10.0)
    cfg_hi = KnoppDriveConfig(Q=100.0)
    r_lo = knopp_traversal(cfg_lo, distance=8.0, n_steps=30)
    r_hi = knopp_traversal(cfg_hi, distance=8.0, n_steps=30)
    assert r_hi.total_energy_budget < r_lo.total_energy_budget


def test_coord_time_scales_with_distance():
    r1 = knopp_traversal(distance=4.0, n_steps=20)
    r2 = knopp_traversal(distance=8.0, n_steps=20)
    ratio = r2.coord_time_total / r1.coord_time_total
    assert 1.7 < ratio < 2.3


def test_summary_string_contains_key_fields():
    rep = knopp_traversal(distance=8.0, n_steps=30)
    s = summarise_traversal(rep)
    assert "Knopp traversal" in s
    assert "t_coord" in s
    assert "E_neg" in s
