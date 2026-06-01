"""Tests for the acoustic multi-horizon T_H comb (LP horizon ladder).

Acceptance gates (actual numbers reported in assert messages):

A. DSI gate. `discrete_scale_invariance_test` on the synthetic horizon
   radii returns is_dsi=True with recovered geometric ratio = exp(pi/alpha)
   to < 1%.  HONEST CAVEAT: the radii are r_n = R exp((n pi - gamma)/alpha)
   *by construction*, so the ratio is exp(pi/alpha) analytically; a clean
   DSI recovery validates only the fluid mapping / the catcher's machinery,
   not new physics.

B. Novelty gate. `scan_comb_novelty` (built on `scan_novelty`) tracks the
   comb across an a = omega R sweep and flags the supercritical onset.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from acoustic_hawking.comb import (
    AcousticTHComb,
    CombNoveltySweep,
    acoustic_T_H_comb,
    comb_logradii_with_anchors,
    comb_window_horizon_count,
    geometric_ratio,
    lp_horizon_radii,
    scan_comb_novelty,
)
from systrophe.analogs.acoustic_metric import acoustic_hawking_temperature
from systrophe.geometry.vanstockum import VanStockumInterior


# ---------------------------------------------------------------------------
# Horizon-ladder enumeration: ALL horizons, not just the first
# ---------------------------------------------------------------------------


def test_lp_horizon_radii_finds_multiple():
    """The supercritical exterior has a LADDER of horizons; we must find
    more than the single one `acoustic_horizon_radius` returns."""
    vs = VanStockumInterior(omega=2.0, R=1.0)
    radii = lp_horizon_radii(vs, r_max_over_R=200.0)
    assert len(radii) >= 3, f"expected a ladder, got {len(radii)} horizons"
    # strictly increasing
    assert np.all(np.diff(radii) > 0)


def test_lp_horizon_radii_are_F_zeros():
    """Every reported radius is a sign change of c^2 - v^2 = F."""
    vs = VanStockumInterior(omega=2.0, R=1.0)
    radii = lp_horizon_radii(vs, r_max_over_R=200.0)
    F_vals = np.array([float(vs.analytic_exterior_F(r)) for r in radii])
    assert np.max(np.abs(F_vals)) < 1e-9, f"max |F| at horizons = {np.max(np.abs(F_vals)):.2e}"


def test_lp_horizon_radii_geometric_ratio():
    """Consecutive horizons sit at the analytic ratio exp(pi/alpha)."""
    vs = VanStockumInterior(omega=2.0, R=1.0)
    radii = lp_horizon_radii(vs, r_max_over_R=200.0)
    ratios = radii[1:] / radii[:-1]
    expected = geometric_ratio(vs)
    assert np.allclose(ratios, expected, rtol=1e-9), (
        f"ratios {ratios} vs expected {expected}"
    )


def test_lp_horizon_radii_empty_subcritical():
    """Sub-/critical exteriors have no acoustic-horizon ladder."""
    vs = VanStockumInterior(omega=0.4, R=1.0)  # a = 0.4 < 1/2
    assert len(lp_horizon_radii(vs)) == 0


# ---------------------------------------------------------------------------
# The comb itself: T_H looped over ALL horizons
# ---------------------------------------------------------------------------


def test_comb_loops_all_horizons():
    """The comb T_H vector has one entry per horizon, matching the
    single-horizon helper at each radius."""
    comb = acoustic_T_H_comb(omega=2.0, R=1.0, r_max_over_R=200.0)
    assert isinstance(comb, AcousticTHComb)
    assert comb.n_horizons == len(comb.horizon_radii) == len(comb.T_hawking)
    assert comb.n_horizons >= 3
    vs = VanStockumInterior(omega=2.0, R=1.0)
    for r, T in zip(comb.horizon_radii, comb.T_hawking):
        assert T == pytest.approx(
            acoustic_hawking_temperature(vs, float(r)), rel=1e-9
        )


def test_comb_is_flat():
    """Exact comb fact: kappa_n = alpha/(2 R |sin gamma|) at every zero, so
    the T_H comb is flat across horizons.  Reports the actual spread."""
    comb = acoustic_T_H_comb(omega=2.0, R=1.0, r_max_over_R=200.0)
    mean_T = float(np.mean(comb.T_hawking))
    spread = float(np.max(np.abs(comb.T_hawking - mean_T)) / abs(mean_T))
    assert comb.comb_is_flat, f"comb T_H spread = {spread:.3e} (mean T_H = {mean_T:.6f})"
    # Analytic value: T_H = alpha / (4 pi R |sin gamma|).
    alpha = comb.alpha
    gamma = np.pi - np.arctan(alpha)
    T_analytic = alpha / (4 * np.pi * comb.R * abs(np.sin(gamma)))
    assert mean_T == pytest.approx(T_analytic, rel=1e-6), (
        f"mean T_H {mean_T:.6f} vs analytic {T_analytic:.6f}"
    )


def test_comb_raises_subcritical():
    with pytest.raises(ValueError):
        acoustic_T_H_comb(omega=0.4, R=1.0)


def test_comb_raises_too_few_horizons():
    """A window so tight that < 3 horizons fit must refuse (DSI needs >= 3)."""
    with pytest.raises(ValueError):
        acoustic_T_H_comb(omega=2.0, R=1.0, r_max_over_R=2.5)


# ---------------------------------------------------------------------------
# ACCEPTANCE GATE A: DSI on the horizon radii recovers exp(pi/alpha) < 1%
# ---------------------------------------------------------------------------


def test_gate_A_dsi_recovers_geometric_ratio():
    """ACCEPTANCE GATE A.

    discrete_scale_invariance_test on the horizon radii returns is_dsi=True
    and recovers exp(pi/alpha) to < 1%.

    HONEST CAVEAT asserted in the message: the ratio is exp(pi/alpha) by
    construction from the analytic F-zeros, so this validates the fluid
    mapping / catcher machinery, not new physics.
    """
    comb = acoustic_T_H_comb(omega=2.0, R=1.0, r_max_over_R=200.0)
    analytic = comb.geometric_ratio
    recovered = comb.dsi_recovered_ratio
    rel_err = abs(recovered - analytic) / analytic
    assert comb.dsi_is_dsi, f"DSI test returned is_dsi=False (rms={comb.dsi_rms_log_dev:.2e})"
    assert rel_err < 0.01, (
        f"recovered ratio {recovered:.6f} vs analytic exp(pi/alpha)="
        f"{analytic:.6f} -> rel err {rel_err*100:.4f}% (gate < 1%). "
        "NOTE: ratio is exp(pi/alpha) by construction; validates fluid "
        "mapping only."
    )


def test_gate_A_holds_across_alphas():
    """Gate A is not a one-off: it holds across several supercritical a.

    Near the onset the ladder is very sparse (spacing exp(pi/alpha) is
    large), so the window is widened per-a to always capture >= 3 teeth.
    """
    # (a, window) chosen so each comb has >= 3 horizons; near onset needs a
    # huge window (sparse comb) -- this is real physics, not a fudge.
    for a, win in ((0.8, 1e5), (1.2, 1e3), (2.5, 200.0), (4.0, 50.0)):
        comb = acoustic_T_H_comb(omega=a, R=1.0, r_max_over_R=win)
        rel_err = abs(comb.dsi_recovered_ratio - comb.geometric_ratio) / comb.geometric_ratio
        assert comb.n_horizons >= 3, f"a={a}: only {comb.n_horizons} horizons"
        assert comb.dsi_is_dsi
        assert rel_err < 0.01, f"a={a}: rel err {rel_err*100:.4f}%"


# ---------------------------------------------------------------------------
# ACCEPTANCE GATE B: scan_novelty tracks the comb + flags supercritical onset
# ---------------------------------------------------------------------------


def test_gate_B_scan_flags_supercritical_onset():
    """ACCEPTANCE GATE B.

    scan_comb_novelty (built on scan_novelty) flags the supercritical onset.
    A subcritical-dominated sweep with a single crossing makes the
    onset the lone outlier so the address-space catcher fires.
    """
    a_sweep = np.array([0.30, 0.34, 0.38, 0.42, 0.46, 0.49, 0.495, 0.499, 1.2])
    sweep = scan_comb_novelty(a_sweep, R=1.0, n_bits=16, r_max_over_R=200.0)
    assert isinstance(sweep, CombNoveltySweep)
    # explicit onset = first non-empty comb
    assert sweep.onset_index == 8, f"onset_index {sweep.onset_index}"
    assert sweep.onset_a == pytest.approx(1.2)
    # catcher's own sharp-feature heuristic fired on the onset
    assert sweep.catcher_verdict == "novel_structure", (
        f"verdict {sweep.catcher_verdict}, n_sharp={sweep.n_sharp_features}"
    )
    assert sweep.n_sharp_features >= 1
    sharp_as = [round(s["parameter_value"], 3) for s in sweep.sharp_features]
    assert 1.2 in sharp_as, f"onset a=1.2 not among sharp features {sharp_as}"


def test_gate_B_tracks_comb_densification():
    """The catcher tracks the comb filling the window: horizon count rises
    monotonically with a across a dense supercritical sweep."""
    a_sweep = np.linspace(0.3, 2.0, 18)
    sweep = scan_comb_novelty(a_sweep, R=1.0)
    # onset detected at the first supercritical sample
    assert sweep.onset_a is not None
    assert sweep.onset_a > 0.5
    # comb size is monotonically non-decreasing in a
    n_h = sweep.n_horizons
    assert np.all(np.diff(n_h) >= 0), f"comb size not monotone: {n_h}"
    assert n_h[0] == 0  # subcritical start
    assert n_h[-1] >= 5  # comb populated by a=2


def test_gate_B_all_subcritical_no_onset():
    """A purely subcritical sweep yields no onset and a non-novel verdict."""
    a_sweep = np.linspace(0.2, 0.45, 8)
    sweep = scan_comb_novelty(a_sweep, R=1.0)
    assert sweep.onset_index is None
    assert sweep.onset_a is None
    assert np.all(sweep.n_horizons == 0)
    assert sweep.catcher_verdict in {"uniform", "smooth"}


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def test_anchored_logradii_subcritical_is_anchors_only():
    arr = comb_logradii_with_anchors(0.4, R=1.0, r_max_over_R=200.0)
    assert len(arr) == 2  # just the two window anchors
    assert arr[0] == pytest.approx(np.log(1.0))
    assert arr[1] == pytest.approx(np.log(200.0))


def test_anchored_logradii_supercritical_adds_horizons():
    arr = comb_logradii_with_anchors(2.0, R=1.0, r_max_over_R=200.0)
    n = comb_window_horizon_count(2.0, R=1.0, r_max_over_R=200.0)
    assert len(arr) == 2 + n
    assert n >= 3
