"""Tests for triple-vortex BdG simulation."""

import numpy as np
import pytest

from systrophe.quantum_info.bdg_triple_vortex import (
    BdGSpectrum,
    acoustic_metric_components_2D,
    hawking_temperature_at_horizon_2d,
    phonon_spectrum_from_horizon,
    sonic_horizon_locus,
    surface_gravity_at_horizon_2d,
    triple_vortex_velocity,
    z3_symmetry_check,
)


def test_triple_vortex_velocity_at_centroid():
    """At the triangle centroid (0, 0), velocity should be small (vortex symmetry)."""
    vx, vy = triple_vortex_velocity(0.0, 0.0, separation=1.0)
    # The three vortices' contributions cancel by Z_3 symmetry
    assert abs(vx) < 1e-6
    assert abs(vy) < 1e-6


def test_triple_vortex_velocity_diverges_near_vortex():
    """Near a vortex center, |v| should be large."""
    R = 1.0 / np.sqrt(3.0)
    vx, vy = triple_vortex_velocity(R + 0.01, 0.0, separation=1.0)
    v_mag = np.sqrt(vx ** 2 + vy ** 2)
    assert v_mag > 10.0


def test_acoustic_metric_components_returns_dict():
    m = acoustic_metric_components_2D(0.5, 0.5)
    assert "c_minus_v_sq" in m
    assert "is_supersonic" in m


def test_acoustic_metric_at_origin_is_subsonic():
    """At the Z_3 centroid, v ~ 0, so the flow is subsonic."""
    m = acoustic_metric_components_2D(0.0, 0.0)
    assert m["is_subsonic"]


def test_acoustic_metric_near_vortex_supersonic():
    """Near a vortex, v is large, so the flow is supersonic."""
    R = 1.0 / np.sqrt(3.0)
    m = acoustic_metric_components_2D(R + 0.01, 0.0)
    assert m["is_supersonic"]


def test_sonic_horizon_locus_finds_points():
    """The sonic horizon should exist between subsonic centroid and supersonic vortex core."""
    horizon = sonic_horizon_locus(separation=1.0, c_sound=1.0,
                                       n_grid=51, grid_size=2.0)
    assert horizon["n_horizon_points"] > 0


def test_surface_gravity_at_horizon_positive():
    horizon = sonic_horizon_locus(separation=1.0, c_sound=1.0, n_grid=51,
                                       grid_size=2.0)
    if horizon["n_horizon_points"] == 0:
        pytest.skip("no horizon")
    x_h, y_h = horizon["horizon_points"][0]
    kappa = surface_gravity_at_horizon_2d(x_h, y_h)
    assert kappa > 0


def test_hawking_T_at_horizon_positive():
    horizon = sonic_horizon_locus(separation=1.0, c_sound=1.0, n_grid=51,
                                       grid_size=2.0)
    if horizon["n_horizon_points"] == 0:
        pytest.skip("no horizon")
    x_h, y_h = horizon["horizon_points"][0]
    T_H = hawking_temperature_at_horizon_2d(x_h, y_h)
    assert T_H > 0


def test_phonon_spectrum_returns_BdGSpectrum():
    spec = phonon_spectrum_from_horizon(separation=1.0, c_sound=1.0,
                                            n_omega=10)
    assert isinstance(spec, BdGSpectrum)


def test_phonon_spectrum_has_horizon():
    spec = phonon_spectrum_from_horizon(separation=1.0, c_sound=1.0,
                                            n_omega=10)
    assert spec.sonic_horizon_present
    assert spec.n_phonons_to_horizon > 0


def test_z3_symmetry_verified():
    """The triple-vortex flow is Z_3 symmetric by construction."""
    result = z3_symmetry_check(separation=1.0)
    assert result["is_z3_symmetric"]
    assert result["max_deviation"] < 1e-9
