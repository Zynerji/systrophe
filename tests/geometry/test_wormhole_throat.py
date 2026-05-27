"""Tests for Morris-Thorne wormhole throat construction."""

import numpy as np
import pytest

from systrophe.geometry.vanstockum import VanStockumInterior
from systrophe.geometry.wormhole_throat import (
    WormholeThroatReport,
    build_wormhole_report,
    effective_redshift_function,
    effective_shape_function,
    find_candidate_throats,
    flaring_out_check,
    z3_cover_throat_interpretation,
)


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_shape_function_finite(vs):
    """b_eff finite for r > 0."""
    for r in (1.5, 2.0, 5.0):
        assert np.isfinite(effective_shape_function(vs, r))


def test_shape_function_relation(vs):
    """b_eff(r) = r - L(r) / r."""
    r = 2.0
    b = effective_shape_function(vs, r)
    L = float(vs.analytic_exterior_L(r))
    assert b == pytest.approx(r - L / r, rel=1e-12)


def test_redshift_function_returns_log_half(vs):
    """Phi(r) = (1/2) ln F(r) when F > 0."""
    r = 1.5  # likely subhorizon
    F = float(vs.analytic_exterior_F(r))
    phi = effective_redshift_function(vs, r)
    if F > 0:
        assert phi == pytest.approx(0.5 * np.log(F), rel=1e-12)


def test_redshift_nan_in_ctc_region(vs):
    """Phi NaN if F <= 0."""
    # Search for an r where F < 0 (deep in CTC region)
    rs = np.linspace(1.05, 20.0, 200)
    Fs = np.array([float(vs.analytic_exterior_F(r)) for r in rs])
    neg = rs[Fs < 0]
    if len(neg) > 0:
        phi = effective_redshift_function(vs, float(neg[0]))
        assert np.isnan(phi)


def test_find_candidate_throats_returns_list(vs):
    """Throat candidates correspond to L=0 (CTC band boundaries)."""
    throats = find_candidate_throats(vs, r_min=1.05, r_max=20.0)
    assert isinstance(throats, list)
    # In supercritical LP, multiple L=0 crossings exist
    for r_t in throats:
        L = float(vs.analytic_exterior_L(r_t))
        assert abs(L) < 0.1


def test_flaring_out_check_at_throat(vs):
    """At a candidate throat, the check returns small b_minus_r."""
    throats = find_candidate_throats(vs, r_min=1.05, r_max=20.0)
    if not throats:
        pytest.skip("no throats found in range")
    check = flaring_out_check(vs, r_throat=throats[0])
    assert abs(check["b_minus_r"]) < 0.1


def test_build_wormhole_report_classifies_throat(vs):
    """Throat report at L=0 location: is_throat True."""
    throats = find_candidate_throats(vs, r_min=1.05, r_max=20.0)
    if not throats:
        pytest.skip("no throats found in range")
    report = build_wormhole_report(vs, r_throat=throats[0])
    assert isinstance(report, WormholeThroatReport)
    assert report.is_throat


# ----- Z_3 cover throat interpretation ---------------------------------

def test_z3_throat_at_zero_gamma_is_closed():
    """gamma_eff = 0: closed cover, monodromy is just the cyclic phase."""
    result = z3_cover_throat_interpretation(gamma_eff=0.0, n_branches=3)
    assert result["closed_cover"]
    # Monodromy at gamma_eff = 0 is exp(2 pi i / 3) -> arg = 2 pi / 3
    assert result["monodromy_arg"] == pytest.approx(2 * np.pi / 3, rel=1e-12)


def test_z3_throat_nonzero_gamma_broken():
    """gamma_eff != 0: cover is open."""
    result = z3_cover_throat_interpretation(gamma_eff=0.5)
    assert not result["closed_cover"]


def test_z3_throat_monodromy_unit_norm():
    """Monodromy is a phase: |M| = 1."""
    for g in (0.0, 0.3, 1.0, 2.5):
        result = z3_cover_throat_interpretation(gamma_eff=g)
        assert result["monodromy_abs"] == pytest.approx(1.0, rel=1e-12)
