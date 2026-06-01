"""Tests for the dynamical approach to the chronology horizon."""

import numpy as np

from chronology_horizon import chronology_signal, approach_sequence, chaotic_barrier, A_CRIT


def test_subcritical_has_no_horizon_or_protection():
    for a in (0.3, 0.45, 0.5):
        s = chronology_signal(a)
        assert not s["supercritical"]
        assert s["r_horizon"] is None
        assert not s["protected"]               # no CTC -> nothing to protect against


def test_supercritical_protection_signal():
    """Above threshold: <T_tt> diverges (power ~ -1) while Kretschmann stays finite."""
    for a in (0.55, 0.8, 1.2):
        s = chronology_signal(a)
        assert s["supercritical"]
        assert s["r_horizon"] is not None
        assert s["T_div_power"] < -0.7          # divergent (~ -1)
        assert s["T_diverges"]
        assert np.isfinite(s["kretschmann_at_rH"])   # classical curvature BOUNDED
        assert s["protected"]


def test_protection_power_is_universal():
    """The Boulware divergence power is ~ -1 across all supercritical a (alpha-universal)."""
    powers = [chronology_signal(a)["T_div_power"] for a in (0.6, 0.9, 1.3)]
    assert all(-1.1 < p < -0.9 for p in powers)
    assert np.std(powers) < 0.05


def test_kretschmann_bounded_but_grows_through_approach():
    """Classical Kretschmann stays finite at the horizon for all a (smooth geometry),
    even as the quantum stress tensor diverges."""
    seq = approach_sequence(np.array([0.6, 0.8, 1.0, 1.5]))
    Ks = [s["kretschmann_at_rH"] for s in seq]
    assert all(np.isfinite(K) for K in Ks)
    assert all(K < 1e6 for K in Ks)             # bounded
    assert Ks[-1] > Ks[0]                        # grows with a (but stays finite)


def test_protected_iff_supercritical():
    seq = approach_sequence(np.array([0.4, 0.5, 0.6, 0.9]))
    assert all(s["protected"] == s["supercritical"] for s in seq)


def test_chaotic_barrier_flickers_across_threshold():
    """Lorenz a(t) wandering across a = 1/2 toggles the protection barrier on/off."""
    cb = chaotic_barrier(a_center=0.62, amp=0.18)
    assert cb["a_min"] < A_CRIT < cb["a_max"]    # genuinely crosses the threshold
    assert cb["n_threshold_crossings"] > 5        # many crossings (chaotic)
    on = cb["barrier_strength"] > 0
    assert 0.1 < float(np.mean(on)) < 0.95        # barrier toggles (not always on/off)
