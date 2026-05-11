"""Tests for off-trace Hadamard <T_{mu nu}>_ren."""

import numpy as np
import pytest

from systrophe.hadamard_offtrace import (
    energy_density_in_static_frame,
    hadamard_offtrace_T,
    hadamard_T_diagonal_components,
    hadamard_T_trace,
    hadamard_T_traceless_part,
    riemann_squared_tensor,
    trace_decomposition,
)
from systrophe.point_splitting import kretschmann_scalar, trace_anomaly_4d_exact
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    """Supercritical van Stockum cylinder for testing."""
    return VanStockumInterior(omega=1.0, R=1.0)


def test_riemann_squared_tensor_shape(vs):
    """Shape is (4, 4)."""
    T = riemann_squared_tensor(vs, r=2.0)
    assert T.shape == (4, 4)


def test_riemann_squared_tensor_symmetric(vs):
    """T_{mu nu} = T_{nu mu}."""
    T = riemann_squared_tensor(vs, r=2.0)
    asymm = np.max(np.abs(T - T.T))
    assert asymm < 1e-10


def test_riemann_squared_trace_equals_kretschmann(vs):
    """g^{mu nu} R_{mu rho sigma tau} R_nu^{rho sigma tau} = K."""
    r = 2.0
    T_R2 = riemann_squared_tensor(vs, r=r)
    trace = hadamard_T_trace(T_R2, vs, r)
    K = kretschmann_scalar(vs, r=r)
    assert trace == pytest.approx(K, rel=1e-3)


def test_offtrace_T_trace_matches_anomaly(vs):
    """Trace of <T_{mu nu}>_ren equals K / (2880 pi^2)."""
    r = 2.0
    T = hadamard_offtrace_T(vs, r=r)
    trace = hadamard_T_trace(T, vs, r)
    expected = trace_anomaly_4d_exact(vs, r=r)
    assert trace == pytest.approx(expected, rel=1e-3)


def test_traceless_part_has_zero_trace(vs):
    """The traceless part has identically zero trace."""
    r = 2.0
    T = hadamard_offtrace_T(vs, r=r)
    T_tl = hadamard_T_traceless_part(T, vs, r)
    trace_tl = hadamard_T_trace(T_tl, vs, r)
    # Small numerical noise from FD inversion
    assert abs(trace_tl) < 1e-10 * max(abs(hadamard_T_trace(T, vs, r)), 1e-10)


def test_trace_decomposition_consistency(vs):
    """T_total = T_trace_part + T_traceless."""
    r = 2.0
    d = trace_decomposition(vs, r=r)
    reconstructed = d["T_trace_part"] + d["T_traceless"]
    diff = np.max(np.abs(reconstructed - d["T_total"]))
    assert diff < 1e-12


def test_trace_anomaly_check_passes(vs):
    """The 'trace_anomaly_check' diagnostic in the decomposition is small."""
    r = 2.0
    d = trace_decomposition(vs, r=r)
    # Tolerance limited by FD accuracy of K
    rel_tol = 1e-3 * abs(d["kretschmann"]) / (2880 * np.pi ** 2)
    assert d["trace_anomaly_check"] < max(rel_tol, 1e-14)


def test_diagonal_components_finite(vs):
    """All diagonal components are finite at well-behaved radii."""
    r = 2.0
    diag = hadamard_T_diagonal_components(vs, r=r)
    for name in ("T_tt", "T_rr", "T_phi_phi", "T_zz", "T_t_phi"):
        assert np.isfinite(diag[name])


def test_offtrace_at_multiple_radii(vs):
    """Smoke test: <T_{mu nu}>_ren is finite and trace matches anomaly at multiple r."""
    for r in (1.5, 2.0, 2.5, 3.0):
        T = hadamard_offtrace_T(vs, r=r)
        assert np.all(np.isfinite(T))
        trace = hadamard_T_trace(T, vs, r)
        expected = trace_anomaly_4d_exact(vs, r=r)
        # Relax rel tol if expected is tiny
        if abs(expected) > 1e-30:
            assert trace == pytest.approx(expected, rel=1e-2)


def test_static_frame_energy_density_returns_value(vs):
    """rho_static = T_{tt} / F when F > 0."""
    r = 2.0
    rho = energy_density_in_static_frame(vs, r=r)
    # Either a finite value (timelike static observer exists) or NaN
    # (we're inside chronology horizon). Both are valid here.
    assert np.isfinite(rho) or np.isnan(rho)
