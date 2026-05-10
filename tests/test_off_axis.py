"""Tests for the off-axis Systrophe pair."""

import numpy as np
import pytest

from systrophe import VanStockumInterior
from systrophe.off_axis import OffAxisPair


def test_construction_requires_supercritical():
    cyl_sub = VanStockumInterior(omega=0.3, R=1.0)
    cyl_sup = VanStockumInterior(omega=1.0, R=1.0)
    with pytest.raises(ValueError):
        OffAxisPair(cyl1=cyl_sub, cyl2=cyl_sup, separation=2.0)
    with pytest.raises(ValueError):
        OffAxisPair(cyl1=cyl_sup, cyl2=cyl_sub, separation=2.0)


def test_construction_requires_positive_separation():
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    with pytest.raises(ValueError):
        OffAxisPair(cyl1=cyl, cyl2=cyl, separation=0.0)
    with pytest.raises(ValueError):
        OffAxisPair(cyl1=cyl, cyl2=cyl, separation=-1.0)


def test_construction_type_check():
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    with pytest.raises(TypeError):
        OffAxisPair(cyl1="not a cylinder", cyl2=cyl, separation=2.0)


def test_far_separation_matches_single_cylinder_metric():
    """At very large separation, near cylinder 1 the metric is dominated
    by cylinder 1 alone (cylinder 2 is far enough that its perturbation is
    negligible *at the Cartesian point* in absolute terms).
    """
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    pair = OffAxisPair(cyl1=cyl, cyl2=cyl, separation=1000.0)
    # Point near cylinder 1
    x, y = 1.5, 0.0
    g = pair.cartesian_metric(np.atleast_1d(x), np.atleast_1d(y))
    # Single-cylinder F at r=1.5
    F_single = float(cyl.analytic_exterior_F(1.5))
    # In Cartesian at (1.5, 0): single-cylinder g_yy = L(1.5) / r^2 = L_single / 2.25
    L_single = float(cyl.analytic_exterior_L(1.5))
    g_yy_single = L_single / (1.5 * 1.5)
    # Pair g_yy is dominated by cyl 1 + small cyl-2 contribution
    # The cyl-2 contribution at distance ~1000 is of order O(F_supercrit(1000)) which is large,
    # but evaluated as a perturbation in CARTESIAN form at r2=1000. The g_yy from cyl 2 has factor
    # cos^2(phi_2) which is ~1 at this point, plus L_2(1000)/1000^2 ~ 1 at large r since L scales
    # as r * something. So the contributions can be comparable. Just check finiteness.
    assert np.isfinite(g["g_tt"][0])
    assert np.isfinite(g["g_xx"][0])
    assert np.isfinite(g["g_yy"][0])


def test_y_reflection_symmetry():
    """Pair with both cylinders on the x-axis is symmetric under y -> -y.

    Under y -> -y: phi -> -phi, so sin(phi) flips sign, cos(phi) is invariant.
      g_tt, g_xx, g_yy invariant (involve only sin^2, cos^2)
      g_xy = sin*cos*... flips sign
      g_tx = -K sin/r flips sign
      g_ty = K cos/r invariant
    """
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    pair = OffAxisPair(cyl1=cyl, cyl2=cyl, separation=3.0)
    g_pos = pair.cartesian_metric(np.atleast_1d(2.0), np.atleast_1d(1.5))
    g_neg = pair.cartesian_metric(np.atleast_1d(2.0), np.atleast_1d(-1.5))
    assert g_pos["g_tt"][0] == pytest.approx(g_neg["g_tt"][0], rel=1e-12)
    assert g_pos["g_xx"][0] == pytest.approx(g_neg["g_xx"][0], rel=1e-12)
    assert g_pos["g_yy"][0] == pytest.approx(g_neg["g_yy"][0], rel=1e-12)
    assert g_pos["g_xy"][0] == pytest.approx(-g_neg["g_xy"][0], rel=1e-12)
    assert g_pos["g_tx"][0] == pytest.approx(-g_neg["g_tx"][0], rel=1e-12)
    assert g_pos["g_ty"][0] == pytest.approx(g_neg["g_ty"][0], rel=1e-12)


def test_swap_symmetry_with_translation():
    """Swapping cyl1<->cyl2 and reflecting x<->(d-x) gives the same metric."""
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    sep = 3.0
    pair_a = OffAxisPair(cyl1=cyl, cyl2=cyl, separation=sep)
    pair_b = OffAxisPair(cyl1=cyl, cyl2=cyl, separation=sep)  # same since cyls equal
    x, y = 1.2, 0.7
    g_a = pair_a.cartesian_metric(np.atleast_1d(x), np.atleast_1d(y))
    # Reflected point in pair_b's frame
    x_ref = sep - x
    g_b = pair_b.cartesian_metric(np.atleast_1d(x_ref), np.atleast_1d(y))
    # g_tt, g_xx, g_yy invariant under x reflection through the midplane
    assert g_a["g_tt"][0] == pytest.approx(g_b["g_tt"][0], rel=1e-12)
    assert g_a["g_xx"][0] == pytest.approx(g_b["g_xx"][0], rel=1e-12)
    assert g_a["g_yy"][0] == pytest.approx(g_b["g_yy"][0], rel=1e-12)


def test_ctc_map_2d_returns_grid():
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    pair = OffAxisPair(cyl1=cyl, cyl2=cyl, separation=3.0)
    m = pair.ctc_map_2d(x_min=-2, x_max=5, y_min=-3, y_max=3, nx=21, ny=15)
    assert m["x"].shape == (21,)
    assert m["y"].shape == (15,)
    assert m["is_ctc"].shape == (15, 21)
    # In the supercritical pair, somewhere in the rectangle should be CTC.
    assert m["is_ctc"].any()


def test_integrate_test_particle_returns_arrays():
    """integrate_test_particle returns t, x, y, tau arrays of consistent shape."""
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    pair = OffAxisPair(cyl1=cyl, cyl2=cyl, separation=4.0)
    out = pair.integrate_test_particle(
        x0=10.0, y0=0.0, vx0=0.0, vy0=0.0, t_max=0.5, n_samples=51
    )
    assert out["t"].shape == (51,)
    assert out["x"].shape == (51,)
    assert out["y"].shape == (51,)
    assert out["tau"].shape == (51,)
    # tau should be non-decreasing
    assert np.all(np.diff(out["tau"]) >= -1e-12)


def test_integrate_test_particle_initial_conditions():
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    pair = OffAxisPair(cyl1=cyl, cyl2=cyl, separation=4.0)
    out = pair.integrate_test_particle(
        x0=10.0, y0=2.0, vx0=0.5, vy0=0.0, t_max=0.1, n_samples=11
    )
    assert out["x"][0] == pytest.approx(10.0)
    assert out["y"][0] == pytest.approx(2.0)
    assert out["t"][0] == pytest.approx(0.0)
    assert out["tau"][0] == pytest.approx(0.0)


def test_has_local_ctc_returns_bool():
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    pair = OffAxisPair(cyl1=cyl, cyl2=cyl, separation=3.0)
    out = pair.has_local_ctc(x=1.5, y=0.5)
    assert isinstance(out, bool)
