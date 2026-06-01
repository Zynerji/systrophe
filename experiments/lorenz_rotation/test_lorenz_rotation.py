"""Tests for the Lorenz-class rotation experiment.

Run from this directory:  python -m pytest test_lorenz_rotation.py -q
These tests are self-contained (numpy + scipy + systrophe core) and do NOT
modify the core rigid-rotation modules, so the main suite (tests/) is unaffected.
"""

import numpy as np
import pytest

from rotating_dust_lorenz import (
    RotatingDustLorenz,
    largest_lyapunov_rk4,
    lyapunov_spectrum,
    rotation_parameter_timeseries,
)
from geodesic_rotation_chaos import (
    GeodesicRotation,
    constant_omega,
    driven_omega,
    finite_time_lyapunov,
    phase_volume_divergence,
)
from chaotic_ctc import ctc_log_measure_for_a, chaotic_ctc_timeseries


# --------------------------------------------------------------------------- #
# H3: rotating-dust Lorenz reduction.
# --------------------------------------------------------------------------- #
def test_canonical_divergence_and_hopf():
    m = RotatingDustLorenz(sigma=10.0, r=28.0, b=8.0 / 3.0)
    assert m.divergence == pytest.approx(-(10.0 + 1.0 + 8.0 / 3.0))
    assert m.r_critical_hopf == pytest.approx(24.7368, abs=1e-3)
    assert m.cell_aspect == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-9)


def test_fixed_points():
    m = RotatingDustLorenz(sigma=10.0, r=28.0, b=8.0 / 3.0)
    fp = m.fixed_points()
    c = np.sqrt(8.0 / 3.0 * 27.0)
    assert np.allclose(fp["C_plus"], [c, c, 27.0])
    assert np.allclose(fp["C_minus"], [-c, -c, 27.0])
    assert np.allclose(fp["origin"], 0.0)


def test_from_dust_parameters_mapping():
    m = RotatingDustLorenz.from_dust_parameters(
        viscosity=10.0, transport=1.0, drive_supercriticality=28.0,
        cell_aspect=1.0 / np.sqrt(2.0),
    )
    assert m.sigma == pytest.approx(10.0)
    assert m.r == pytest.approx(28.0)
    assert m.b == pytest.approx(8.0 / 3.0, abs=1e-9)


def test_lyapunov_spectrum_reproduces_canonical_attractor():
    """Strange-attractor signature: +, ~0, - exponents; sum == divergence;
    Kaplan-Yorke ~ 2.06. Shorter integration -> looser but decisive bounds."""
    m = RotatingDustLorenz(sigma=10.0, r=28.0, b=8.0 / 3.0)
    spec = lyapunov_spectrum(
        m, np.array([1.0, 1.0, 1.0]), t_max=200.0, dt=0.005,
        renorm_every=20, t_transient=20.0,
    )
    exps = spec["exponents"]
    assert exps[0] == pytest.approx(0.906, abs=0.06)      # positive
    assert abs(exps[1]) < 0.05                            # neutral
    assert exps[2] < -13.0                                # strongly contracting
    assert spec["sum"] == pytest.approx(m.divergence, abs=1e-3)  # trace constraint
    assert spec["kaplan_yorke_dim"] == pytest.approx(2.06, abs=0.05)
    assert spec["kolmogorov_sinai"] > 0.0                 # chaotic


def test_largest_lyapunov_sign_below_and_above_onset():
    """Below the Hopf onset the leading exponent is non-positive; above it,
    positive. This is the chaos-onset discriminator."""
    m_lo = RotatingDustLorenz(sigma=10.0, r=15.0, b=8.0 / 3.0)
    m_hi = RotatingDustLorenz(sigma=10.0, r=28.0, b=8.0 / 3.0)
    le_lo = largest_lyapunov_rk4(m_lo, np.array([1.0, 1.0, 1.0]), t_max=100.0)
    le_hi = largest_lyapunov_rk4(m_hi, np.array([1.0, 1.0, 1.0]), t_max=100.0)
    assert le_lo < 0.02
    assert le_hi > 0.5


# --------------------------------------------------------------------------- #
# H0/H2: geodesic chaos is conservative (Liouville -> no attractor).
# --------------------------------------------------------------------------- #
def test_rigid_rotation_conserves_pt_and_is_regular():
    g = constant_omega(0.8)
    g = GeodesicRotation(g.omega_fn, g.omega_dot_fn, ell=0.6)
    s0 = g.initial_state_from_E(r0=0.7, E=1.2)
    assert g.hamiltonian(s0) == pytest.approx(-0.5, abs=1e-9)  # on mass shell
    tr = g.integrate(s0, tau_max=150.0, n_samples=3001)
    assert float(np.ptp(tr["p_t"])) < 1e-9                     # p_t conserved


def test_geodesic_flow_is_hamiltonian_zero_divergence():
    """Liouville: the geodesic phase flow has zero divergence for rigid AND
    driven rotation -> no attractor is possible, however chaotic."""
    for g in (
        GeodesicRotation(*(lambda c: (c.omega_fn, c.omega_dot_fn))(constant_omega(0.8)), ell=0.6),
        driven_omega(omega0=0.8, eps=0.5, drive_freq=1.3, ell=0.6),
    ):
        s0 = g.initial_state_from_E(r0=0.7, E=1.2)
        assert abs(phase_volume_divergence(g, s0)) < 1e-6


def test_driving_rotation_increases_lyapunov():
    """Time-dependent rotation breaks d/dt symmetry -> conservative chaos:
    FTLE grows with drive strength while remaining a Hamiltonian flow."""
    g0 = GeodesicRotation(*(lambda c: (c.omega_fn, c.omega_dot_fn))(constant_omega(0.8)), ell=0.6)
    g1 = driven_omega(omega0=0.8, eps=0.5, drive_freq=1.3, ell=0.6)
    s0 = g0.initial_state_from_E(r0=0.7, E=1.2)
    s1 = g1.initial_state_from_E(r0=0.7, E=1.2)
    le0 = finite_time_lyapunov(g0, s0, tau_max=200.0, n_renorm=200)
    le1 = finite_time_lyapunov(g1, s1, tau_max=200.0, n_renorm=200)
    assert le1 > 3.0 * le0
    # driven rotation pumps energy: p_t is no longer conserved
    tr = g1.integrate(s1, tau_max=100.0, n_samples=2001)
    assert float(np.ptp(tr["p_t"])) > 1.0


# --------------------------------------------------------------------------- #
# CTC bridge: chaotic rotation -> flickering time-machine bands.
# --------------------------------------------------------------------------- #
def test_rotation_parameter_stays_supercritical():
    m = RotatingDustLorenz(sigma=10.0, r=28.0, b=8.0 / 3.0)
    traj = m.integrate(np.array([1.0, 1.0, 1.0]), t_max=60.0, dt=0.01,
                       t_transient=20.0)
    rot = rotation_parameter_timeseries(traj, a0=1.5, eps=0.2, clip_min=0.55)
    assert rot["a_min"] >= 0.55           # exterior stays Bonnor Case III
    assert rot["a_max"] > rot["a_min"]    # genuinely time-varying


def test_ctc_measure_monotone_proxy_and_flicker():
    # larger a -> larger log-frequency alpha
    lo = ctc_log_measure_for_a(0.8, R=1.0, r_min=1.05, r_max=20.0)
    hi = ctc_log_measure_for_a(2.0, R=1.0, r_min=1.05, r_max=20.0)
    assert hi["alpha"] > lo["alpha"]
    assert lo["log_measure"] >= 0.0

    m = RotatingDustLorenz(sigma=10.0, r=28.0, b=8.0 / 3.0)
    traj = m.integrate(np.array([1.0, 1.0, 1.0]), t_max=60.0, dt=0.01,
                       t_transient=20.0)
    rot = rotation_parameter_timeseries(traj, a0=1.5, eps=0.2)
    cts = chaotic_ctc_timeseries(rot, R=1.0, r_min=1.05, r_max=20.0, stride=10)
    # the band structure genuinely flickers as the rotation wanders
    assert cts["n_bands"].max() > cts["n_bands"].min()
    assert cts["log_measure"].std() > 0.0


def test_subcritical_a_has_no_ctc():
    out = ctc_log_measure_for_a(0.4, R=1.0, r_min=1.05, r_max=20.0)
    assert out["n_bands"] == 0
    assert out["log_measure"] == 0.0


# --------------------------------------------------------------------------- #
# Expansion: multi-attractor registry.
# --------------------------------------------------------------------------- #
from attractors import registry, ChenFlow, HalvorsenFlow, RosslerFlow, lorenz_flow
from deep_tests import (
    delay_embed, correlation_dimension_scalar, pecora_carroll_sync,
    conditional_lyapunov, observable_faithfulness,
)


def test_constant_divergences():
    assert ChenFlow().divergence == pytest.approx(-10.0)
    assert HalvorsenFlow(a=1.4).divergence == pytest.approx(-4.2)


@pytest.mark.parametrize("flow", registry(), ids=lambda f: f.name)
def test_registry_flows_are_dissipative_attractors(flow):
    """Every rotation law: positive leading exponent (chaos), negative exponent
    sum (dissipative), sum == mean divergence (ergodic theorem), KY dim > 2."""
    spec = lyapunov_spectrum(flow, flow.default_s0, t_max=150.0, dt=0.005,
                             renorm_every=20, t_transient=flow.t_transient)
    traj = flow.integrate(flow.default_s0, 150.0, dt=0.01, t_transient=flow.t_transient)
    mean_div = flow.mean_divergence(traj["s"])
    assert spec["exponents"][0] > 0.02                     # chaotic
    assert spec["sum"] < 0.0                               # dissipative -> attractor
    assert spec["sum"] == pytest.approx(mean_div, abs=0.2) # Sum lambda = <tr J>
    assert spec["kaplan_yorke_dim"] > 2.0                  # strange (fractal)


# --------------------------------------------------------------------------- #
# Deep tests: synchronization + Takens faithfulness.
# --------------------------------------------------------------------------- #
def test_pecora_carroll_x_drive_locks_z_does_not():
    sx = pecora_carroll_sync(drive="x", t_max=60.0)
    sz = pecora_carroll_sync(drive="z", t_max=60.0)
    assert sx["locked"] and sx["tail_mean_error"] < 1e-6
    assert not sz["locked"]
    # x-drive conditional Lyapunov is negative (Pecora-Carroll criterion)
    assert conditional_lyapunov(drive="x")["conditional_lyapunov_max"] < 0.0


def test_delay_embed_shape():
    x = np.sin(np.linspace(0, 50, 1000))
    emb = delay_embed(x, m=3, tau=5)
    assert emb.shape == (1000 - 2 * 5, 3)


def test_ctc_observable_is_faithful_to_attractor():
    """The scalar CTC band-measure reconstructs the attractor dimension (~2)
    via Takens embedding -- the time machine faithfully encodes the rotation."""
    res = observable_faithfulness(lorenz_flow(), t_max=700.0, dt=0.01, stride=4)
    assert 1.7 < res["D2_state_space"] < 2.3
    assert 1.6 < res["D2_ctc_observable"] < 2.6   # delay-embed estimates run high
    assert res["relative_diff"] < 0.30
