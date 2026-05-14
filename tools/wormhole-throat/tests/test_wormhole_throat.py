"""Tests for wormhole-throat."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wormhole_throat import (
    CasimirThroatReport,
    ThroatReport,
    WormholeThroatExplorer,
    casimir_at_throat,
)


# ---------------------------------------------------------------------------
# Explorer
# ---------------------------------------------------------------------------


def test_explorer_construction():
    we = WormholeThroatExplorer(omega=2.0, R=1.0)
    assert we.omega == 2.0
    assert we.R == 1.0


def test_supercritical_has_throat_candidates():
    """Supercritical Tipler exterior has L=0 loci (CTC band boundaries =
    candidate throats)."""
    we = WormholeThroatExplorer(omega=2.0, R=1.0)
    cands = we.candidate_throats(n_grid=2001)
    assert len(cands) > 0


def test_candidate_throats_in_range():
    we = WormholeThroatExplorer(omega=2.0, R=1.0,
                                  r_search_min=1.05, r_search_max=10.0)
    cands = we.candidate_throats()
    for r in cands:
        assert 1.05 <= r <= 10.0


def test_shape_function_returns_finite():
    we = WormholeThroatExplorer(omega=2.0, R=1.0)
    b = we.shape_function(2.0)
    assert np.isfinite(b)


def test_redshift_function_at_subsonic_radius():
    """The redshift function is real in F>0 (subsonic / sub-horizon) regions.
    Picks a radius just inside the source boundary where F > 0."""
    we = WormholeThroatExplorer(omega=2.0, R=1.0)
    phi = we.redshift_function(1.1)
    assert np.isfinite(phi)


def test_report_returns_dataclass():
    we = WormholeThroatExplorer(omega=2.0, R=1.0)
    cands = we.candidate_throats()
    assert len(cands) >= 1
    rep = we.report(cands[0])
    assert isinstance(rep, ThroatReport)
    assert rep.r_throat == cands[0]


def test_report_all_returns_list():
    we = WormholeThroatExplorer(omega=2.0, R=1.0)
    reports = we.report_all()
    assert isinstance(reports, list)
    assert all(isinstance(r, ThroatReport) for r in reports)


def test_z3_cover_interpretation_dict():
    we = WormholeThroatExplorer(omega=2.0, R=1.0)
    z3 = we.z3_cover_interpretation(gamma_eff=0.0)
    assert isinstance(z3, dict)


# ---------------------------------------------------------------------------
# Casimir-at-throat
# ---------------------------------------------------------------------------


def test_casimir_at_throat_returns_dataclass():
    we = WormholeThroatExplorer(omega=2.0, R=1.0)
    cands = we.candidate_throats()
    assert len(cands) >= 1
    rep = casimir_at_throat(omega=2.0, R=1.0, r_throat=cands[0],
                             plate_separation_d=1.0)
    assert isinstance(rep, CasimirThroatReport)
    assert rep.r_throat == cands[0]


def test_casimir_energy_density_negative():
    """Standard Brown-Maclay flat-space energy density is negative."""
    we = WormholeThroatExplorer(omega=2.0, R=1.0)
    cands = we.candidate_throats()
    rep = casimir_at_throat(omega=2.0, R=1.0, r_throat=cands[0],
                             plate_separation_d=1.0)
    assert rep.brown_maclay_energy_density_flat < 0


def test_casimir_T_components_present():
    we = WormholeThroatExplorer(omega=2.0, R=1.0)
    cands = we.candidate_throats()
    rep = casimir_at_throat(omega=2.0, R=1.0, r_throat=cands[0],
                             plate_separation_d=1.0)
    # The Brown-Maclay T-component dict should be non-empty
    assert len(rep.brown_maclay_T_components) > 0


def test_casimir_separation_scaling():
    """Brown-Maclay energy density scales as -1/d^4."""
    we = WormholeThroatExplorer(omega=2.0, R=1.0)
    cands = we.candidate_throats()
    r_t = cands[0]
    rep_small = casimir_at_throat(omega=2.0, R=1.0, r_throat=r_t,
                                    plate_separation_d=1.0)
    rep_large = casimir_at_throat(omega=2.0, R=1.0, r_throat=r_t,
                                    plate_separation_d=2.0)
    # |rho(d=1)| / |rho(d=2)| = 16 by 1/d^4 scaling
    ratio = abs(rep_small.brown_maclay_energy_density_flat) / abs(
        rep_large.brown_maclay_energy_density_flat,
    )
    assert ratio == pytest.approx(16.0, rel=1e-6)
