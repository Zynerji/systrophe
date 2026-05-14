"""Tests for lp_analyser."""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lp_analyser import LPAnalyser, PairAnalyser
from systrophe.vanstockum import VanStockumInterior


# ---------------------------------------------------------------------------
# LPAnalyser
# ---------------------------------------------------------------------------


def test_supercritical_basic_properties():
    a = LPAnalyser(omega=2.0, R=1.0)
    assert a.is_supercritical
    assert math.isclose(a.a, 2.0)
    # alpha = sqrt(4 a^2 - 1) = sqrt(15)
    assert abs(a.alpha - math.sqrt(15.0)) < 1e-9


def test_subcritical_alpha_is_nan():
    a = LPAnalyser(omega=0.3, R=1.0)
    assert not a.is_supercritical
    assert math.isnan(a.alpha)


def test_metric_components_obey_constraint():
    """F L + K^2 = r^2 should hold identically for the Case III closed forms."""
    a = LPAnalyser(omega=2.0, R=1.0)
    rs = np.linspace(1.1, 3.0, 8)
    F = a.F(rs); K = a.K(rs); L = a.L(rs)
    invariant = F * L + K * K
    assert np.allclose(invariant, rs ** 2, rtol=1e-8, atol=1e-9)


def test_cauchy_horizons_for_omega2():
    """omega=2, R=1 -> first three horizons ~ 1.405, 3.163, 7.118."""
    a = LPAnalyser(omega=2.0, R=1.0)
    horizons = a.cauchy_horizons()
    assert len(horizons) >= 3
    expected = np.array([1.4054, 3.1629, 7.1181])
    assert np.allclose(horizons[:3], expected, atol=1e-3)


def test_ctc_bands_exist():
    a = LPAnalyser(omega=2.0, R=1.0)
    bands = a.ctc_bands()
    # supercritical should have at least one CTC band
    assert len(bands) >= 1
    for lo, hi in bands:
        assert hi > lo > 0


def test_energy_conditions_all_hold_for_van_stockum():
    """Van Stockum dust satisfies all energy conditions for any omega > 0."""
    a = LPAnalyser(omega=2.0, R=1.0)
    rep = a.energy_conditions()
    assert rep.nec_holds
    assert rep.wec_holds
    assert rep.sec_holds
    assert rep.dec_holds


def test_surface_gravity_and_hawking_T_at_first_horizon():
    """For omega=2, R=1, all horizons have surface gravity ~ 2.0 and T_H ~ 1/pi."""
    a = LPAnalyser(omega=2.0, R=1.0)
    kappa = a.surface_gravity()
    T_H = a.hawking_temperature()
    assert abs(kappa - 2.0) < 1e-3
    assert abs(T_H - 1.0 / math.pi) < 1e-3


def test_boulware_stress_tensor_at_midpoint():
    """Boulware <T_tt> is positive (and finite) at a midpoint inside the
    first F > 0 band."""
    a = LPAnalyser(omega=2.0, R=1.0)
    r_mid = 0.5 * (a.R + a.cauchy_horizons()[0])
    T = a.stress_tensor(r_mid, state="boulware")
    assert np.isfinite(T["T_tt"])
    assert T["T_tt"] > 0


def test_hadamard_V_0_vanishes_in_vacuum_conformal():
    """V_0 = 0 exactly for massless conformally-coupled scalar on vacuum."""
    a = LPAnalyser(omega=2.0, R=1.0)
    r_mid = 0.5 * (a.R + a.cauchy_horizons()[0])
    assert abs(a.hadamard_V_0(r_mid)) < 1e-2


def test_hadamard_V_1_matches_kretschmann_over_720():
    """V_1 = K_kretsch / 720 in vacuum-conformal case."""
    a = LPAnalyser(omega=2.0, R=1.0)
    r_mid = 0.5 * (a.R + a.cauchy_horizons()[0])
    V_1 = a.hadamard_V_1(r_mid)
    K = a.kretschmann(r_mid)
    assert math.isclose(V_1, K / 720.0, rel_tol=1e-9)


def test_trace_anomaly_check_at_midpoint():
    """2 V_1 / (8 pi^2) should reproduce K / (2880 pi^2)."""
    a = LPAnalyser(omega=2.0, R=1.0)
    r_mid = 0.5 * (a.R + a.cauchy_horizons()[0])
    res = a.trace_anomaly_check(r_mid)
    assert res["relative_difference"] < 1e-6


def test_summary_returns_load_bearing_fields():
    """LPSummary captures the headline facts for the omega=2 R=1 spacetime."""
    a = LPAnalyser(omega=2.0, R=1.0)
    s = a.summary()
    assert s.is_supercritical
    assert s.n_cauchy_horizons_in_10R >= 3
    assert s.energy_conditions_all_satisfied is True
    assert s.first_horizon_r is not None
    assert abs(s.first_horizon_r - 1.405) < 1e-2
    # Boulware divergence power should be near -1 (Phase 2a verdict)
    assert s.boulware_T_tt_simple_pole_power is not None
    assert abs(s.boulware_T_tt_simple_pole_power - (-1.0)) < 0.25


# ---------------------------------------------------------------------------
# PairAnalyser
# ---------------------------------------------------------------------------


def test_pair_co_axial_array_factor_doubles_for_aligned_pair():
    """Two aligned cylinders (delta=0,0) -> array factor = 2x single."""
    c = VanStockumInterior(omega=1.0, R=1.0)
    single = PairAnalyser([c])
    pair = PairAnalyser([c, c], offsets=[0.0, 0.0])
    rs = np.array([1.5, 2.0, 2.5])
    af1 = single.array_factor(rs)
    af2 = pair.array_factor(rs)
    assert np.allclose(af2, 2.0 * af1, rtol=1e-10)


def test_pair_uniform_phase_comb_extinguished():
    """Anti-phase pair: array factor near zero everywhere."""
    c = VanStockumInterior(omega=1.0, R=1.0)
    p = PairAnalyser([c, c], offsets=[0.0, math.pi])
    ext = p.extinction_check(r_max=10.0)
    assert ext["is_extinguished"]


def test_pair_beam_steer_places_node():
    """PairAnalyser.beam_steer should produce L = 0 at r_target (machine zero)."""
    c = VanStockumInterior(omega=1.0, R=1.0)
    p = PairAnalyser.beam_steer(r_target=3.0, cylinder=c, N=2)
    L_at_target = float(p.L(np.array([3.0]))[0])
    assert abs(L_at_target) < 1e-9, f"L(r_target) = {L_at_target}"


def test_off_axis_pair_runs_topology():
    """OffAxisPair quantitative topology runs without crashing."""
    c1 = VanStockumInterior(omega=1.0, R=1.0)
    c2 = VanStockumInterior(omega=1.0, R=1.0)
    p = PairAnalyser([c1, c2], separation=3.0)
    topo = p.ctc_region_topology(-3.0, 6.0, -3.0, 3.0, nx=41, ny=21)
    # Phase 3b verdict for this canonical setup: 1 component, 2 holes.
    assert topo["n_components"] >= 1
    assert topo["ctc_fraction"] > 0.0
    assert topo["topology_summary"] in {
        "empty", "simply_connected", "multi_component", "with_holes", "complex"
    }


def test_off_axis_ergosurface_runs():
    c1 = VanStockumInterior(omega=1.0, R=1.0)
    c2 = VanStockumInterior(omega=1.0, R=1.0)
    p = PairAnalyser([c1, c2], separation=3.0)
    erg = p.ergosurface_2d(-3.0, 6.0, -3.0, 3.0, nx=41, ny=21)
    assert erg["g_tt"].shape == (21, 41)


def test_off_axis_rejects_more_than_2_cylinders():
    c = VanStockumInterior(omega=1.0, R=1.0)
    with pytest.raises(ValueError):
        PairAnalyser([c, c, c], separation=3.0)
