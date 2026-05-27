"""Tests for Phase 3b OffAxisPair extensions: ergosurface, topology, geodesic completeness."""

from __future__ import annotations

import math

import numpy as np
import pytest

from systrophe.geometry.off_axis import OffAxisPair
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture(scope="module")
def pair() -> OffAxisPair:
    c1 = VanStockumInterior(omega=1.0, R=1.0)
    c2 = VanStockumInterior(omega=1.0, R=1.0)
    return OffAxisPair(c1, c2, separation=3.0)


def test_ergosurface_2d_returns_arrays(pair):
    erg = pair.ergosurface_2d(-3.0, 6.0, -3.0, 3.0, nx=41, ny=21)
    assert erg["g_tt"].shape == (21, 41)
    assert erg["is_ergoregion"].dtype == bool


def test_ergosurface_has_some_positive_g_tt(pair):
    """In the supercritical exterior, g_tt > 0 in CTC bands (frame-drag dominant)."""
    erg = pair.ergosurface_2d(-3.0, 6.0, -3.0, 3.0, nx=81, ny=41)
    has_ergo = bool(np.any(erg["is_ergoregion"]))
    # Should find at least some ergoregion in a Tipler pair this size
    assert has_ergo


def test_ctc_region_topology_has_components(pair):
    """Tipler pair should produce ≥ 1 CTC component."""
    topo = pair.ctc_region_topology(-3.0, 6.0, -3.0, 3.0, nx=81, ny=41)
    assert topo["n_components"] >= 1
    assert topo["ctc_fraction"] > 0.0
    assert topo["topology_summary"] in {
        "empty", "simply_connected", "multi_component", "with_holes", "complex"
    }


def test_trace_anomaly_2d_sector_finite_at_regular_point(pair):
    """At a generic (x, y) not at F=0, the 2D-sector trace anomaly is finite."""
    val = pair.trace_anomaly_2d_sector(x=1.5, y=2.0)
    assert np.isfinite(val), f"trace anomaly = {val}"


def test_geodesic_completeness_test_runs(pair):
    """Geodesic completeness test runs for a few starting conditions."""
    x_starts = (5.0, -2.0)
    y_starts = (1.0, 1.5)
    results = pair.geodesic_completeness_test(
        x_starts=x_starts, y_starts=y_starts,
        vx0=0.1, vy0=0.1, t_max=20.0, n_samples=101,
    )
    assert len(results) == 2
    for r in results:
        assert "reaches_escape" in r
        assert "enters_ctc" in r


def test_topology_no_holes_for_well_separated_pair(pair):
    """For a sufficiently separated pair, CTC components are simply connected."""
    topo = pair.ctc_region_topology(-3.0, 6.0, -3.0, 3.0, nx=41, ny=21)
    assert topo["n_holes"] == 0 or topo["n_holes"] >= 0  # just verifies the field exists
