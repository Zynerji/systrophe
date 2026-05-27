"""Tests for holographic / AdS-dictionary module."""

import numpy as np
import pytest

from systrophe.quantum_info.holographic import (
    HolographicData,
    boundary_expansion_coefficients,
    boundary_two_point_correlator,
    brown_york_stress_tensor,
    dual_operator_dimensions,
    efimov_correspondence,
    holographic_entropy_strip,
    reconstruct_alpha_from_holographic_data,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_boundary_expansion_extracts_alpha(vs_super):
    data = boundary_expansion_coefficients(vs_super)
    assert data["regime"] == "supercritical"
    assert data["alpha"] == pytest.approx(vs_super.alpha, rel=1e-9)


def test_boundary_expansion_beta_positive(vs_super):
    data = boundary_expansion_coefficients(vs_super)
    assert data["beta"] is not None
    assert data["beta"] > 0


def test_boundary_expansion_subcritical_returns_none(vs_sub):
    data = boundary_expansion_coefficients(vs_sub)
    assert data["beta"] is None


def test_boundary_expansion_log_period(vs_super):
    data = boundary_expansion_coefficients(vs_super)
    expected = float(np.exp(2 * np.pi / vs_super.alpha))
    assert data["log_period"] == pytest.approx(expected, rel=1e-12)


def test_dual_operator_dimensions_complex(vs_super):
    holo = dual_operator_dimensions(vs_super)
    assert isinstance(holo, HolographicData)
    assert holo.delta_imag > 0
    assert not holo.is_unitary


def test_dual_operator_dimensions_subcritical_unitary(vs_sub):
    holo = dual_operator_dimensions(vs_sub)
    assert holo.is_unitary
    assert holo.delta_imag == 0.0


def test_dual_operator_matches_alpha_over_two(vs_super):
    holo = dual_operator_dimensions(vs_super)
    assert holo.delta_imag == pytest.approx(vs_super.alpha / 2.0, rel=1e-12)


def test_brown_york_stress_finite(vs_super):
    by = brown_york_stress_tensor(vs_super, r_boundary=10.0)
    assert np.isfinite(by["T_trace_bare"])
    assert np.isfinite(by["T_t_phi_bare"])


def test_brown_york_returns_F_K_L(vs_super):
    by = brown_york_stress_tensor(vs_super, r_boundary=8.0)
    assert "F_at_boundary" in by
    assert "K_at_boundary" in by
    assert "L_at_boundary" in by


def test_holographic_entropy_strip_positive(vs_super):
    ent = holographic_entropy_strip(vs_super, strip_width=0.5)
    assert ent["S_proxy"] > 0


def test_holographic_entropy_strip_scales_with_width(vs_super):
    e1 = holographic_entropy_strip(vs_super, strip_width=0.5)
    e2 = holographic_entropy_strip(vs_super, strip_width=2.0)
    # Larger strip => larger entropy (roughly linear in width)
    assert e2["S_proxy"] > e1["S_proxy"]


def test_boundary_two_point_correlator_decays(vs_super):
    c1 = boundary_two_point_correlator(vs_super, separation=1.0)
    c100 = boundary_two_point_correlator(vs_super, separation=100.0)
    assert abs(c1) > abs(c100)


def test_boundary_two_point_correlator_subcritical_real(vs_sub):
    c = boundary_two_point_correlator(vs_sub, separation=5.0)
    assert c.imag == 0


def test_boundary_two_point_correlator_supercritical_complex(vs_super):
    c = boundary_two_point_correlator(vs_super, separation=2.5)
    assert abs(c.imag) > 0


def test_boundary_two_point_invalid_separation(vs_super):
    with pytest.raises(ValueError):
        boundary_two_point_correlator(vs_super, separation=-1.0)


def test_efimov_correspondence_supercritical(vs_super):
    eff = efimov_correspondence(vs_super)
    assert eff["lambda_tipler"] is not None
    assert eff["lambda_tipler"] > 1.0


def test_efimov_correspondence_subcritical_returns_none(vs_sub):
    eff = efimov_correspondence(vs_sub)
    assert eff["lambda_tipler"] is None


def test_reconstruct_alpha_matches_theoretical(vs_super):
    """Reconstructing alpha from <O O> phase recovers theoretical alpha."""
    alpha_recovered = reconstruct_alpha_from_holographic_data(vs_super)
    assert alpha_recovered == pytest.approx(vs_super.alpha, rel=1e-2)


def test_reconstruct_alpha_subcritical_zero(vs_sub):
    alpha = reconstruct_alpha_from_holographic_data(vs_sub)
    assert alpha == 0.0
