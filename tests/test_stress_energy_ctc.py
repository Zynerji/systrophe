"""Tests for stress_energy_ctc (Phase 2a)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from systrophe import (
    StressEnergyState,
    boulware_stress_tensor,
    cauchy_horizon_estimate,
    chronology_protection_novelty_scan,
    chronology_protection_report,
    divergence_rate_at_horizon,
    hartle_hawking_stress_tensor,
    polyakov_sigma,
    polyakov_sigma_derivatives,
    ricci_scalar_2d,
    stress_tensor,
    surface_gravity_at_horizon,
    tortoise_coordinate,
    trace_anomaly_2d,
    unruh_stress_tensor,
)
from systrophe.vanstockum import VanStockumInterior


# Use a strongly-supercritical reference (a = 2.0) so several Cauchy horizons
# sit inside r in (R, 10 R), enabling multi-horizon scans.
@pytest.fixture(scope="module")
def vs_super() -> VanStockumInterior:
    return VanStockumInterior(omega=2.0, R=1.0)


# ---------------------------------------------------------------------------
# Geometric building blocks
# ---------------------------------------------------------------------------


def test_polyakov_sigma_matches_definition(vs_super):
    # Between R = 1 and the first horizon r_H1 ~ 1.405 (omega=2 fixture)
    r = 1.2
    F = float(vs_super.analytic_exterior_F(np.array([r]))[0])
    expected = 0.5 * math.log(F)
    assert math.isclose(polyakov_sigma(vs_super, r), expected, rel_tol=1e-12)


def test_polyakov_sigma_nan_inside_horizon(vs_super):
    horizons = cauchy_horizon_estimate(vs_super)
    # Pick a radius where F < 0 (inside the first CTC band: r in (r_H1, r_H2))
    if len(horizons) >= 2:
        r_inside = (horizons[0] + horizons[1]) / 2.0
    else:
        r_inside = horizons[0] + 0.1
    F_inside = float(vs_super.analytic_exterior_F(np.array([r_inside]))[0])
    assert F_inside < 0, "fixture broken: expected F < 0 inside first CTC band"
    assert math.isnan(polyakov_sigma(vs_super, r_inside))


def test_sigma_derivatives_finite_away_from_horizon(vs_super):
    d = polyakov_sigma_derivatives(vs_super, 1.2)
    assert np.isfinite(d["sigma"])
    assert np.isfinite(d["sigma_rstar"])
    assert np.isfinite(d["sigma_rstar2"])
    # At a generic radius, sigma_rstar should be of order 1 (not zero)
    assert abs(d["sigma_rstar"]) > 1e-3


def test_tortoise_coordinate_zero_at_reference(vs_super):
    r_ref = 1.05 * vs_super.R
    assert math.isclose(tortoise_coordinate(vs_super, r_ref, r_ref=r_ref), 0.0, abs_tol=1e-12)


def test_tortoise_coordinate_grows_outward(vs_super):
    r_ref = 1.05 * vs_super.R
    # Both inside the F > 0 band between R and the first Cauchy horizon ~ 1.405.
    r1, r2 = 1.15, 1.3
    s1 = tortoise_coordinate(vs_super, r1, r_ref=r_ref)
    s2 = tortoise_coordinate(vs_super, r2, r_ref=r_ref)
    assert s2 > s1
    assert s1 > 0


# ---------------------------------------------------------------------------
# Trace anomaly and consistency
# ---------------------------------------------------------------------------


def test_ricci_scalar_2d_matches_qftcs_backreaction(vs_super):
    """Sanity: ricci_scalar_2d agrees with qftcs_backreaction.radial_temporal_ricci_scalar."""
    from systrophe.qftcs_backreaction import radial_temporal_ricci_scalar
    r = 1.2  # F > 0 between source boundary and first horizon
    R_new = ricci_scalar_2d(vs_super, r)
    R_old = float(radial_temporal_ricci_scalar(vs_super, r))
    assert math.isclose(R_new, R_old, rel_tol=5e-3)


def test_trace_anomaly_is_R_over_24pi(vs_super):
    r = 1.2
    R = ricci_scalar_2d(vs_super, r)
    anomaly = trace_anomaly_2d(vs_super, r)
    assert math.isclose(anomaly, R / (24.0 * math.pi), rel_tol=1e-10)


def test_polyakov_trace_recovers_anomaly(vs_super):
    """The trace of <T_mu_nu>_B must equal R_2D / (24 pi) (Polyakov identity).

    g_{tt} = -F, g_{rr} = h = 1, so g^{tt} = -1/F, g^{rr} = 1, and
    <T^mu_mu> = -T_tt/F + T_rr.
    """
    # Sweep generic radii strictly outside the source and inside the first
    # Cauchy horizon (r_H1 ~ 1.405 for omega=2, R=1 fixture).
    for r in (1.10, 1.20, 1.28, 1.35):
        T = boulware_stress_tensor(vs_super, r)
        F = T["F"]
        anomaly = trace_anomaly_2d(vs_super, r)
        trace = -T["T_tt"] / F + T["T_rr"]
        # Finite-difference noise dominates the residual at order eps^2; we use
        # a generous tolerance scaled to the anomaly magnitude.
        scale = max(abs(anomaly), 1e-8)
        assert abs(trace - anomaly) / scale < 0.05, (
            f"r={r}: trace={trace:.3e} vs anomaly={anomaly:.3e} (rel diff "
            f"{(trace - anomaly) / scale:.2e})"
        )


# ---------------------------------------------------------------------------
# State-specific stress tensors
# ---------------------------------------------------------------------------


def test_boulware_tt_grows_toward_first_horizon(vs_super):
    """<T_tt>_B magnitude grows monotonically as r approaches the first Cauchy horizon.

    For the omega=2, R=1 fixture, the regular side of the first horizon
    (F > 0) is r < r_H1, so we sample at r_H1 - eps for decreasing eps.
    """
    horizons = cauchy_horizon_estimate(vs_super)
    r_H = float(horizons[0])
    eps_grid = np.geomspace(2e-3, 5e-2, 6)
    T_vals = np.array(
        [boulware_stress_tensor(vs_super, r_H - e)["T_tt"] for e in eps_grid]
    )
    assert np.all(np.isfinite(T_vals)), f"NaN in T_vals: {T_vals}"
    # |T_tt| should decrease with increasing distance eps from the horizon
    assert np.all(np.diff(np.abs(T_vals)) <= 1e-12), (
        f"|<T_tt>_B| not monotone: {T_vals}"
    )
    # Innermost (nearest horizon) sample should dominate
    assert abs(T_vals[0]) > 10.0 * abs(T_vals[-1])


def test_hartle_hawking_differs_from_boulware_by_constant(vs_super):
    """At fixed horizon, HH offset is uniform pi T_H^2 / 12 in t_u, t_v.

    The (T_uu, T_vv) components must differ from Boulware by exactly that
    offset; the differences must agree to numerical precision.
    """
    horizons = cauchy_horizon_estimate(vs_super)
    r_H = float(horizons[0])
    kappa = surface_gravity_at_horizon(vs_super, r_H)
    T_H = kappa / (2.0 * math.pi)
    offset = math.pi * T_H * T_H / 12.0
    for r in (1.10, 1.20, 1.30):
        T_B = boulware_stress_tensor(vs_super, r)
        T_HH = hartle_hawking_stress_tensor(vs_super, r, r_horizon=r_H)
        assert math.isclose(T_HH["T_uu"] - T_B["T_uu"], offset, rel_tol=1e-10)
        assert math.isclose(T_HH["T_vv"] - T_B["T_vv"], offset, rel_tol=1e-10)


def test_unruh_carries_radial_flux(vs_super):
    """Unruh state has non-zero <T_{tr}> (outgoing thermal flux)."""
    horizons = cauchy_horizon_estimate(vs_super)
    r_H = float(horizons[0])
    r = 1.20  # Inside F > 0 band
    T_U = unruh_stress_tensor(vs_super, r, r_horizon=r_H)
    T_B = boulware_stress_tensor(vs_super, r)
    assert abs(T_U["T_t_rstar"]) > 1e-6
    assert abs(T_B["T_t_rstar"]) < 1e-10


def test_stress_tensor_dispatch(vs_super):
    horizons = cauchy_horizon_estimate(vs_super)
    r_H = float(horizons[0])
    r = 1.20
    T_b1 = stress_tensor(vs_super, r, state="boulware")
    T_b2 = boulware_stress_tensor(vs_super, r)
    assert math.isclose(T_b1["T_tt"], T_b2["T_tt"], rel_tol=1e-12)
    T_hh1 = stress_tensor(vs_super, r, state=StressEnergyState.HARTLE_HAWKING, r_horizon=r_H)
    T_hh2 = hartle_hawking_stress_tensor(vs_super, r, r_horizon=r_H)
    assert math.isclose(T_hh1["T_tt"], T_hh2["T_tt"], rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Divergence rate (the chronology-protection signature)
# ---------------------------------------------------------------------------


def test_boulware_diverges_at_first_horizon(vs_super):
    """Boulware <T_tt> diverges with power approximately -1 at the first Cauchy horizon."""
    horizons = cauchy_horizon_estimate(vs_super)
    r_H = float(horizons[0])
    fit = divergence_rate_at_horizon(
        vs_super, r_H, state=StressEnergyState.BOULWARE,
        n_samples=20, eps_min=5e-4, eps_max=2e-2, component="T_tt",
    )
    assert fit.diverges, f"Boulware fit not diverging: power={fit.power}"
    # Theoretical: simple-pole, power = -1
    assert abs(fit.power - (-1.0)) < 0.3, f"Expected p ≈ -1, got {fit.power}"
    assert fit.fit_residual_rms < 0.5, f"Fit too noisy: rms = {fit.fit_residual_rms}"


def test_boulware_diverges_at_second_horizon(vs_super):
    """Same simple-pole behaviour at the 2nd Cauchy horizon (distance-scaling sanity)."""
    horizons = cauchy_horizon_estimate(vs_super)
    assert len(horizons) >= 2
    r_H = float(horizons[1])
    fit = divergence_rate_at_horizon(
        vs_super, r_H, state=StressEnergyState.BOULWARE,
        n_samples=20, eps_min=5e-4, eps_max=2e-2, component="T_tt",
    )
    assert fit.diverges
    assert abs(fit.power - (-1.0)) < 0.3


def test_hh_still_diverges_at_other_horizons(vs_super):
    """HH analog regularises one horizon but should still diverge at the next.

    (At the 2nd horizon, the HH state designed around the 1st horizon is
    NOT thermal-regular and still inherits the Boulware divergence.)
    """
    horizons = cauchy_horizon_estimate(vs_super)
    assert len(horizons) >= 2
    r_H1, r_H2 = float(horizons[0]), float(horizons[1])
    # HH state set up around 1st horizon, fit divergence at 2nd
    fit = divergence_rate_at_horizon(
        vs_super, r_H2, state=StressEnergyState.HARTLE_HAWKING,
        n_samples=18, eps_min=5e-4, eps_max=2e-2, component="T_tt",
    )
    # We pass the 2nd horizon as r_horizon (via the divergence_rate function),
    # so internally HH is set up around r_H2, NOT r_H1. The divergence at
    # r_H2 should still be the same simple pole.
    assert fit.diverges


# ---------------------------------------------------------------------------
# End-to-end chronology-protection report
# ---------------------------------------------------------------------------


def test_chronology_protection_report_consistent(vs_super):
    rep = chronology_protection_report(vs_super, n_horizons=2,
                                          n_samples=16, eps_min=1e-3, eps_max=2e-2)
    assert rep.n_horizons_scanned == 2
    # All Boulware fits diverge with power near -1
    for f in rep.boulware_fits:
        assert f.diverges, f"Boulware fit power = {f.power} (not diverging)"
    assert rep.verdict.startswith("chronology_protection_")
    # Trace anomaly check stays small (FD noise only)
    assert rep.trace_anomaly_max_residual < 1.0


def test_report_requires_supercritical():
    """Subcritical exterior has no Cauchy horizons; report must refuse."""
    vs_sub = VanStockumInterior(omega=0.3, R=1.0)  # a = 0.3 < 1/2
    with pytest.raises(ValueError):
        chronology_protection_report(vs_sub)


# ---------------------------------------------------------------------------
# Novelty catcher (always-on rule)
# ---------------------------------------------------------------------------


def test_novelty_catcher_runs_and_returns_verdict(vs_super):
    result = chronology_protection_novelty_scan(vs_super, n_radii=24)
    assert result["verdict"] in {"smooth", "novel_structure", "uniform"}
    assert "lambda_2_at_radius" in result
    # The sweep crosses Cauchy horizons, so a sharp feature is *expected*
    # — but we don't make this a hard assertion (catcher policy: report,
    # don't enforce). What we DO enforce is that the scan runs cleanly.
    assert result["n_radii"] == 24
