"""Tests for the dodeca-in-horn-torus orientation sweep (knopp_dodeca_alignment)."""

import math

import numpy as np
import pytest

from systrophe.knopp.knopp_dodeca_alignment import (
    DEFAULT_R,
    FACE_LOCK_DEG,
    INRADIUS_RATIO,
    PHI,
    alignment_state,
    dodecahedron_face_axes,
    dodecahedron_vertices,
    feedback_equilibrium,
    horn_torus_sdf,
    mode_saturation,
    normalised_drive,
    orientation_sweep,
    summarise_sweep,
    sweep_rotation,
    vertex_up_rotation,
)


# ----- geometry --------------------------------------------------------------


def test_vertices_unit_and_distinct():
    v = dodecahedron_vertices()
    assert v.shape == (20, 3)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=2)
    assert np.min(d[~np.eye(20, dtype=bool)]) > 0.1


def test_face_axes_mutual_angle_is_arccos_inv_sqrt5():
    a = dodecahedron_face_axes()
    assert a.shape == (6, 3)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    dots = np.abs(a @ a.T)
    off = dots[~np.eye(6, dtype=bool)]
    assert np.allclose(off, 1.0 / math.sqrt(5.0), atol=1e-12)


def test_inradius_ratio_is_vertex_face_cosine():
    v0 = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    f0 = np.array([0.0, 1.0, PHI]) / math.sqrt(1.0 + PHI ** 2)
    assert math.isclose(INRADIUS_RATIO, float(v0 @ f0), rel_tol=1e-12)
    assert math.isclose(FACE_LOCK_DEG, 37.377, abs_tol=0.01)


def test_horn_torus_sdf_landmarks():
    R = DEFAULT_R
    assert math.isclose(horn_torus_sdf([[0, 0, 0]], R)[0], 0.0, abs_tol=1e-12)
    assert math.isclose(horn_torus_sdf([[R, 0, 0]], R)[0], -R, abs_tol=1e-12)
    assert math.isclose(horn_torus_sdf([[2 * R, 0, 0]], R)[0], 0.0, abs_tol=1e-12)
    assert horn_torus_sdf([[0, 0.3, 0]], R)[0] > 0  # axis funnel is outside


def test_vertex_up_rotation_orthonormal():
    B = vertex_up_rotation()
    assert np.allclose(B @ B.T, np.eye(3), atol=1e-12)
    assert math.isclose(np.linalg.det(B), 1.0, rel_tol=1e-9)
    v0 = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    assert np.allclose(B @ v0, [0, 1, 0], atol=1e-12)


def test_sweep_reaches_face_lock_at_advertised_angle():
    rot = sweep_rotation(FACE_LOCK_DEG)
    axes = dodecahedron_face_axes() @ rot.T
    best = max(np.max(axes[:, 1]), np.max(-axes[:, 1]))
    assert best > 0.999999


# ----- alignment + drive ------------------------------------------------------


def test_vertex_lock_state():
    s = alignment_state(0.0)
    assert s.vertex_align > s.face_align
    assert s.spiral == 0.0
    assert s.area_factor == 1.0


def test_face_lock_state_engages_spiral():
    s = alignment_state(FACE_LOCK_DEG)
    assert s.face_align > 0.5
    assert s.spiral == 1.0
    assert math.isclose(s.area_factor, 7.0, rel_tol=1e-9)
    assert s.min_gap < alignment_state(0.0).min_gap


def test_face_lock_drive_exceeds_vertex_lock():
    s0 = alignment_state(0.0)
    sf = alignment_state(FACE_LOCK_DEG)
    cal = 2.0 * s0.drive_raw / s0.area_factor
    assert normalised_drive(sf, cal) / normalised_drive(s0, cal) > 2.0


def test_area_factor_scales_drive_linearly():
    bare = alignment_state(FACE_LOCK_DEG, area_gain=0.0)
    amped = alignment_state(FACE_LOCK_DEG, area_gain=6.0)
    assert math.isclose(amped.drive_raw, 7.0 * bare.drive_raw, rel_tol=1e-9)


def test_alignment_state_validates_input():
    with pytest.raises(ValueError):
        alignment_state(0.0, scale=-1.0)
    with pytest.raises(ValueError):
        normalised_drive(alignment_state(0.0), calibration_raw=0.0)


# ----- mode comb --------------------------------------------------------------


def test_mode_saturation_monotone_in_drive():
    lo = mode_saturation(0.3, 0.5, 0.6, 1.0)
    hi = mode_saturation(0.8, 0.5, 0.6, 1.0)
    assert hi.saturation >= lo.saturation
    assert np.all(hi.amplitudes >= 0) and np.all(hi.amplitudes <= 1)


def test_mode_saturation_validates_input():
    with pytest.raises(ValueError):
        mode_saturation(1.5, 0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        mode_saturation(0.5, 0.5, 0.5, 0.5, Q=0.0)


def test_point_lock_is_narrowband_face_lock_is_broadband():
    fb_v = feedback_equilibrium(0.0)
    fb_f = feedback_equilibrium(FACE_LOCK_DEG)
    # point-lock rings only the pentagonal comb: partial, never full spectrum
    assert not fb_v.saturation.full_spectrum
    assert 0.0 < fb_v.saturation.saturation < 0.5
    assert fb_v.saturation.collapse == 0.0
    # face-lock + spiral pumps every mode: full spectrum, collapse engaged
    assert fb_f.saturation.full_spectrum
    assert fb_f.interior_saturated
    assert fb_f.saturation.collapse > 0.9


def test_feedback_loop_is_self_limiting():
    for beta in (0.0, FACE_LOCK_DEG, 90.0, 150.0):
        fb = feedback_equilibrium(beta)
        assert fb.self_limiting
        assert fb.saturation.saturation <= 1.0
        assert float(np.max(fb.saturation.amplitudes)) <= 1.0


# ----- sweep + catcher --------------------------------------------------------


def test_orientation_sweep_report():
    r = orientation_sweep(n_angles=31)
    assert r.beta_deg.shape == (31,)
    assert np.all((r.drive >= 0) & (r.drive <= 1))
    assert np.all(r.min_gap > 0)
    assert r.face_over_vertex > 1.0
    assert r.catcher_verdict in ("novel_structure", "smooth", "uniform")
    assert r.catcher_sharp_features >= 0
    text = summarise_sweep(r)
    assert "face-lock angle" in text and "catcher verdict" in text


def test_mechanism_survives_another_halving():
    # user requirement 2026-06-09: the dodeca must work at half the default
    # scale (0.13). Pinch at origin => smaller body sits closer; the
    # point/face contrast must survive.
    for s in (0.13, 0.065):
        fb_v = feedback_equilibrium(0.0, scale=s)
        fb_f = feedback_equilibrium(FACE_LOCK_DEG, scale=s)
        assert not fb_v.saturation.full_spectrum
        assert fb_f.saturation.full_spectrum
        assert fb_f.state.spiral == 1.0


def test_sweep_endpoints_agree_with_symmetry():
    # beta = 0 and beta = 180 are both vertex-up configurations (antipodal
    # vertex), so the drive curve must close on itself
    r = orientation_sweep(n_angles=31)
    assert math.isclose(r.drive[0], r.drive[-1], rel_tol=1e-6)
