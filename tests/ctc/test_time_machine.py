"""Time-machine harness tests."""

import numpy as np
import pytest

from systrophe import (
    SystrophePair,
    VanStockumInterior,
    find_single_cylinder_windows,
    find_time_machine_windows,
    harness_time_loop,
)


def test_single_supercritical_finds_at_least_one_window():
    """A supercritical cylinder has CTC bands."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    windows = find_single_cylinder_windows(vs, r_min=1.001, r_max=200.0)
    assert len(windows) >= 1


def test_subcritical_rejected_for_single_cylinder():
    """No CTCs in subcritical exterior."""
    vs = VanStockumInterior(omega=0.3, R=1.0)
    with pytest.raises(ValueError):
        find_single_cylinder_windows(vs, r_min=1.001, r_max=20.0)


def test_window_log_span_nonzero():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    windows = find_single_cylinder_windows(vs, r_min=1.001, r_max=200.0)
    for w in windows:
        assert w.log_span() > 0.0


def test_window_l_min_negative():
    """Within a CTC band, the minimum L is strictly negative."""
    vs = VanStockumInterior(omega=1.5, R=1.0)
    windows = find_single_cylinder_windows(vs, r_min=1.001, r_max=200.0)
    for w in windows:
        assert w.L_min < 0


def test_harness_dt_per_revolution_matches_target():
    """harness_time_loop produces orbit with the requested dt per rev."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    w = find_single_cylinder_windows(vs, r_min=1.001, r_max=10.0)[0]
    target = -1.0
    out = harness_time_loop(w, target_dt_per_rev=target, n_revolutions=5)
    assert out["dt_per_revolution"] == pytest.approx(target, rel=1e-12)
    assert out["total_coord_time_advance"] == pytest.approx(5 * target)
    assert out["is_timelike"] is True


def test_harness_proper_time_positive():
    """Proper time advance per revolution is positive (orbit forward in tau)."""
    vs = VanStockumInterior(omega=1.5, R=1.0)
    w = find_single_cylinder_windows(vs, r_min=1.001, r_max=10.0)[0]
    out = harness_time_loop(w, target_dt_per_rev=-0.5, n_revolutions=3)
    assert out["dtau_per_revolution"] > 0.0


def test_harness_rejects_spacelike_target():
    """A target dt that lands Omega in the spacelike sector is rejected."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    w = find_single_cylinder_windows(vs, r_min=1.001, r_max=10.0)[0]
    # Omega bounds at r_min are ~ +- 1.0; pick a target that lands Omega in there
    ob_lo, ob_hi = w.omega_bounds_at_min
    Omega_inside = 0.5 * (ob_lo + ob_hi)
    target_dt_spacelike = 2.0 * np.pi / Omega_inside if Omega_inside != 0 else 1e6
    if abs(Omega_inside) > 1e-6:  # only test if non-degenerate
        with pytest.raises(ValueError):
            harness_time_loop(w, target_dt_per_rev=target_dt_spacelike, n_revolutions=1)


def test_pair_window_count_changes_with_offset():
    """Phase offset between two cylinders shifts the CTC band structure."""
    cyl = VanStockumInterior(omega=1.5, R=1.0)
    # Zero offset (constructive) and pi offset (destructive)
    pair_zero = SystrophePair.from_cylinders(cyl, cyl, delta_offset=0.0)
    pair_pi = SystrophePair.from_cylinders(cyl, cyl, delta_offset=np.pi)
    # Joint L envelope of pair_pi is identically zero (perfect cancellation),
    # so no CTC windows.
    bands_zero = pair_zero.ctc_bands(r_min=1.05, r_max=20.0)
    bands_pi = pair_pi.ctc_bands(r_min=1.05, r_max=20.0)
    assert len(bands_zero) >= 1
    assert len(bands_pi) == 0


def test_general_find_time_machine_windows():
    """Generic detector works on a hand-crafted L_fn with one CTC band."""
    # L = (r-2)(r-4): negative on (2, 4), positive outside.
    L_fn = lambda r: (np.asarray(r, dtype=float) - 2.0) * (np.asarray(r, dtype=float) - 4.0)
    F_fn = lambda r: np.full_like(np.asarray(r, dtype=float), 1.0)
    K_fn = lambda r: np.zeros_like(np.asarray(r, dtype=float))
    windows = find_time_machine_windows(L_fn, F_fn, K_fn, r_min=1.0, r_max=5.0)
    assert len(windows) == 1
    w = windows[0]
    assert w.r_inner == pytest.approx(2.0, abs=1e-3)
    assert w.r_outer == pytest.approx(4.0, abs=1e-3)
    assert w.r_min_L == pytest.approx(3.0, abs=1e-2)
    assert w.L_min == pytest.approx(-1.0, abs=1e-3)
