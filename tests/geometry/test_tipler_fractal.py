"""Tests for discrete-scale invariance and cascade-DSI extension."""

import numpy as np
import pytest

from systrophe.geometry.sinusoid import TiplerSinusoid
from systrophe.geometry.tipler_fractal import (
    CascadeDSI,
    base_tipler_sinusoid_is_dsi,
    box_count_log_dimension,
    cascade_box_dimension,
    ctc_band_log_widths,
    dsi_extension_from_sinusoid,
    dsi_rescaling_factor,
    verify_geometric_progression,
    zero_geometric_ratio,
    zero_set,
)


# ----- zero-set DSI ------------------------------------------------------

def test_zero_geometric_ratio():
    """rho = exp(pi / alpha)."""
    for alpha in (0.5, 1.0, 2.0, 3.5):
        rho = zero_geometric_ratio(alpha)
        assert rho == pytest.approx(np.exp(np.pi / alpha), rel=1e-12)


def test_dsi_rescaling_factor_is_ratio_squared():
    """lambda = rho^2."""
    for alpha in (0.7, 1.4, 2.3):
        rho = zero_geometric_ratio(alpha)
        lam = dsi_rescaling_factor(alpha)
        assert lam == pytest.approx(rho ** 2, rel=1e-12)


def test_zero_set_count():
    """Zero set on [R, R*exp(N*pi/alpha)] has exactly N zeros (modulo edges)."""
    R, alpha = 1.0, 2.0
    zs = zero_set(R=R, alpha=alpha, delta=0.0, r_min=R, r_max=R * np.exp(6 * np.pi / alpha))
    # Cosine has zeros at alpha*ln(r/R) = pi/2 + k pi  =>  ln(r/R) = (1/2 + k) pi / alpha
    # For u in [0, 6 pi / alpha], k ranges from 0 to 5 ==> 6 zeros
    assert len(zs) == 6


def test_zero_set_is_geometric_progression():
    """Successive zeros form a geometric progression with ratio rho."""
    R, alpha, delta = 1.0, 1.5, 0.3
    zs = zero_set(R=R, alpha=alpha, delta=delta, r_min=R, r_max=R * 1e4)
    check = verify_geometric_progression(zs)
    assert check["is_geometric"]
    expected = zero_geometric_ratio(alpha)
    assert check["ratio_mean"] == pytest.approx(expected, rel=1e-12)


def test_base_sinusoid_is_dsi():
    s = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.4)
    result = base_tipler_sinusoid_is_dsi(s, n_periods=8)
    assert result["is_dsi"]
    assert result["rel_error"] < 1e-12


# ----- ctc-band DSI ------------------------------------------------------

def test_ctc_band_log_widths_constant():
    """CTC bands of a pure Tipler sinusoid have equal log-widths (a DSI signature)."""
    s = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    widths = ctc_band_log_widths(s, r_min=1.0, r_max=np.exp(10), n_grid=20001)
    # Expect each width close to pi / alpha (half-period in u)
    expected = np.pi / s.alpha
    assert len(widths) >= 3
    assert np.std(widths) / np.mean(widths) < 1e-3  # tight clustering
    assert np.mean(widths) == pytest.approx(expected, rel=5e-3)


# ----- box counting on discrete set -------------------------------------

def test_discrete_zero_set_has_low_box_dimension():
    """A pure geometric progression has box dim 0 in the resolved limit."""
    s = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    zs = zero_set(R=1.0, alpha=s.alpha, delta=0.0, r_min=1.0, r_max=np.exp(10))
    bc = box_count_log_dimension(zs, u_min=0.0, u_max=10.0, n_scales=14)
    # Discrete countable set: log N(eps) saturates at total point count for small eps,
    # giving slope -> 0 in the valid regime. Allow some slop because the
    # tail still has counts < n_points.
    assert bc["dimension"] < 0.5


# ----- cascade DSI -------------------------------------------------------

def test_cascade_construction():
    c = CascadeDSI(R=1.0, alpha_0=1.0, A_0=1.0, delta_0=0.0, levels=4)
    assert c.alphas().shape == (4,)
    assert c.amps().shape == (4,)
    assert c.amps()[0] == 1.0
    assert c.amps()[-1] == 0.5 ** 3
    assert c.alphas()[-1] == 1.0 * 2.0 ** 3


def test_cascade_F_is_finite():
    c = CascadeDSI(R=1.0, alpha_0=1.5, A_0=1.0, delta_0=0.5, levels=3)
    rs = np.linspace(1.0, 1000.0, 50)
    F = c.F(rs)
    assert np.all(np.isfinite(F))


def test_single_level_cascade_equals_base():
    """levels=1 with given (alpha_0, A_0, delta_0) reproduces a single cosine."""
    c = CascadeDSI(R=1.0, alpha_0=2.0, A_0=3.0, delta_0=0.7, levels=1)
    r = np.array([1.5, 4.0, 12.0])
    expected = 3.0 * np.cos(2.0 * np.log(r) + 0.7)
    assert np.allclose(c.F(r), expected, atol=1e-12)


def test_cascade_zeros_exist():
    """A multi-level cascade with strong sub-amps has more zeros than the base."""
    base = CascadeDSI(R=1.0, alpha_0=1.0, A_0=1.0, delta_0=0.0,
                      levels=1)
    cascade = CascadeDSI(R=1.0, alpha_0=1.0, A_0=1.0, delta_0=0.0,
                         levels=3, scale_factor=2.0, amp_decay=0.95)
    span_r_max = np.exp(8)
    zs_base = base.zeros(r_min=1.0, r_max=span_r_max, n_grid=50_001)
    zs_cascade = cascade.zeros(r_min=1.0, r_max=span_r_max, n_grid=50_001)
    # Higher-freq cascade levels add extra zeros when amp_decay is close to 1
    assert len(zs_cascade) > len(zs_base)
    assert len(zs_cascade) >= 8


def test_multi_level_cascade_box_dimension_nontrivial():
    """A self-similar multi-scale cascade has measurable box dimension > 0.

    This is the proper mathematical content of the informal 'fractal' claim:
    single-cosine zero sets are trivial; cascade zero sets are not.
    """
    c = CascadeDSI(R=1.0, alpha_0=0.8, A_0=1.0, delta_0=0.0,
                   levels=4, scale_factor=2.0, amp_decay=0.6)
    bc = cascade_box_dimension(c, r_min=1.0, r_max=np.exp(10), n_scales=18)
    # A finite-level cascade still has a countable zero set, so the
    # exact-limit dimension is 0; but at the finite scales tested,
    # the box-count slope is non-trivial (>0.3) because the multi-scale
    # structure populates many boxes at intermediate eps.
    assert bc["dimension"] > 0.3


def test_dsi_extension_from_sinusoid_preserves_base():
    s = TiplerSinusoid(R=2.0, a=1.5, A=1.7, delta=0.4)
    c = dsi_extension_from_sinusoid(s, levels=3, scale_factor=3.0, amp_decay=0.4)
    assert c.alpha_0 == s.alpha
    assert c.A_0 == s.A
    assert c.delta_0 == s.delta
    assert c.R == s.R
    assert c.levels == 3
    assert c.scale_factor == 3.0
    assert c.amp_decay == 0.4


def test_cascade_validates_params():
    with pytest.raises(ValueError):
        CascadeDSI(R=1.0, alpha_0=1.0, A_0=1.0, delta_0=0.0, levels=2,
                   scale_factor=1.0)  # must exceed 1
    with pytest.raises(ValueError):
        CascadeDSI(R=1.0, alpha_0=1.0, A_0=1.0, delta_0=0.0, levels=2,
                   amp_decay=0.0)  # must be in (0, 1)
    with pytest.raises(ValueError):
        CascadeDSI(R=-1.0, alpha_0=1.0, A_0=1.0, delta_0=0.0, levels=1)
