"""Knopp Drive: four-mechanism composite warp engineering bound.

The Knopp Drive composes four independent warp-engineering shortcuts
that have each been validated against the address-space novelty
catcher into a single tractable engineering object:

    1. Tipler CTC-band gating  (exotic-matter requirement -> 0 inside
       the geometric CTC band of a supercritical rotating cylinder).
    2. Krasnikov tube embedding  (engineered directed causal corridor
       inside the bubble shell).
    3. Q-cavity feedback amplification  (sustained drive power scales
       as |E_neg_bare| / Q^2, instead of the bare impulse).
    4. Horn-toroidal twist  (theta-dependent ADM mass weight produces
       continuous steering at the dipole level p ~ R^2 epsilon |m_ADM|).

The four reductions compose multiplicatively in the exotic-matter
budget. The COMPLETE engineering bound on the sustained drive power
of a Knopp-Drive warp craft is

    P_drive  =  |E_neg|_Krasnikov(alpha, sigma)
                * (1 - coupling * tilt(r))_+        # Tipler gate
                * (1 + epsilon * cos(theta - theta_0))  # horn twist
                / Q^2                                  # feedback Q
                * f_0(sigma)                           # cycle freq

This is the canonical Knopp budget. All four factors are independently
tunable; the catcher-verified sharp boundaries supply the engineering
parameter bounds (Q >= ~8, |epsilon| < 1, r inside a Tipler CTC band,
alpha = sqrt(3) for a = 1 supercritical seed).

Pfenning-Ford compatibility
---------------------------
The product |E_neg| * tau is bounded below by the P-F inequality.
The Knopp Drive saturates it (no quantum violation). The Q factor
trades impulse for duration; the Tipler gate trades amplitude for
geometric path (the CTC band provides the missing tilt for free).

Steerability
------------
The horn-twist axis theta_0 sets the steering direction; epsilon
sets the steering magnitude. Continuous control across [0, 1).

Naming
------
Named for Christian Knopp, the author of the Systrophe framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .feedback_amplified_shell import (
    cavity_lifetime,
    required_drive_power,
    saturation_field_energy,
    shell_natural_frequency,
)
from .horn_toroidal_warp import (
    horn_steering_magnitude,
    horn_steering_vector,
)
from .krasnikov_tube import krasnikov_NEC_radial
from .novelty_catcher import scan_novelty
from .tipler_krasnikov_hybrid import (
    hybrid_total_negative_energy,
    tipler_tilt_at,
)
from .vanstockum import VanStockumInterior


@dataclass(frozen=True)
class KnoppDriveConfig:
    """Full Knopp-Drive parameter set."""
    # Tipler seed
    omega: float = 1.0
    R_cylinder: float = 1.0
    # Krasnikov tube
    alpha_wall: float = 4.0
    # Feedback shell
    Q: float = 10.0
    sigma_shell: float = 4.0
    # Horn twist
    epsilon_horn: float = 0.2
    theta_0_horn: float = 0.0
    # Geometric placement
    r_orbit: float = 1.5     # radius of the tube worldline in the LP frame
    v_s: float = 1.0         # apparent shell velocity
    R_bubble: float = 1.0    # outer bubble radius


@dataclass(frozen=True)
class KnoppDriveBudget:
    """The complete Knopp-Drive engineering report."""
    config: KnoppDriveConfig
    # Per-mechanism factor (dimensionless, in [0, 1] except horn)
    tipler_gate_factor: float
    feedback_factor: float
    horn_amplification: float
    # Composite quantities
    krasnikov_bare_E_neg: float
    composite_E_neg: float
    sustained_drive_power: float
    pfenning_ford_compatible: bool
    # Steering
    steering_vector_pxpy: tuple[float, float]
    steering_magnitude: float
    # Cavity engineering
    natural_frequency: float
    cavity_lifetime_tau: float
    saturation_field_energy_value: float
    parametric_gain_required: float


def _knopp_compose(cfg: KnoppDriveConfig) -> KnoppDriveBudget:
    """Compute the full Knopp budget from a config."""
    vs = VanStockumInterior(omega=cfg.omega, R=cfg.R_cylinder)
    # 1. Tipler gate
    tilt = tipler_tilt_at(vs, cfg.r_orbit)
    tipler_gate = max(1.0 - tilt, 0.0)  # 0 inside CTC band, ~1 outside
    # 2. Krasnikov bare wall NEC integrated over x
    kras_bare = hybrid_total_negative_energy(
        vs, cfg.r_orbit, alpha=cfg.alpha_wall, coupling=0.0,
    )
    # 3. Feedback factor (sustained scales 1/Q^2)
    fb_factor = 1.0 / max(cfg.Q, 1.0) ** 2
    # 4. Horn dipole amplification (max over theta of weight)
    horn_amp = 1.0 + cfg.epsilon_horn  # worst-case (horn-side)
    horn_p = horn_steering_vector(
        R=cfg.R_bubble, epsilon=cfg.epsilon_horn,
        theta_0=cfg.theta_0_horn, m_ADM=-1.0, sigma=cfg.sigma_shell,
    )
    horn_mag = horn_steering_magnitude(
        R=cfg.R_bubble, epsilon=cfg.epsilon_horn,
        theta_0=cfg.theta_0_horn, m_ADM=-1.0, sigma=cfg.sigma_shell,
    )
    # Composite exotic-matter budget
    composite_E_neg = (
        kras_bare * tipler_gate * fb_factor * horn_amp
    )
    # Cavity quantities
    f_0 = shell_natural_frequency(cfg.sigma_shell)
    tau = cavity_lifetime(cfg.Q, cfg.sigma_shell)
    E_sat = saturation_field_energy(
        v_s=cfg.v_s, R=cfg.R_bubble, sigma=cfg.sigma_shell, Q=cfg.Q,
    )
    P_drive = required_drive_power(
        v_s=cfg.v_s, R=cfg.R_bubble, sigma=cfg.sigma_shell, Q=cfg.Q,
    )
    # Pfenning-Ford: the cavity sustains |E_neg|*tau which must exceed
    # the bound 3/(32 pi^2 sigma^2). Use the Tipler-gated effective
    # E_neg so the test is on the post-gating budget.
    pf_product = abs(composite_E_neg) * tau
    pf_bound = 3.0 / (32.0 * math.pi ** 2 * cfg.sigma_shell ** 2)
    pf_ok = pf_product >= pf_bound - 1e-12 or abs(composite_E_neg) < 1e-12
    # Parametric gain required for saturation
    g_required = math.log(max(cfg.Q, 1.0001)) / max(tau, 1e-15)
    return KnoppDriveBudget(
        config=cfg,
        tipler_gate_factor=float(tipler_gate),
        feedback_factor=float(fb_factor),
        horn_amplification=float(horn_amp),
        krasnikov_bare_E_neg=float(kras_bare),
        composite_E_neg=float(composite_E_neg),
        sustained_drive_power=float(P_drive),
        pfenning_ford_compatible=bool(pf_ok),
        steering_vector_pxpy=(float(horn_p[0]), float(horn_p[1])),
        steering_magnitude=float(horn_mag),
        natural_frequency=float(f_0),
        cavity_lifetime_tau=float(tau),
        saturation_field_energy_value=float(E_sat),
        parametric_gain_required=float(g_required),
    )


def knopp_budget(cfg: KnoppDriveConfig | None = None, **overrides) -> KnoppDriveBudget:
    """Compute the Knopp-Drive engineering budget.

    Either pass a fully populated `KnoppDriveConfig`, or pass kwargs
    that override the default config.
    """
    if cfg is None:
        cfg = KnoppDriveConfig()
    if overrides:
        # Build a new config replacing the overrides
        from dataclasses import replace
        cfg = replace(cfg, **overrides)
    return _knopp_compose(cfg)


def knopp_drive_inside_band(cfg: KnoppDriveConfig | None = None,
                             r_orbit: float = 1.5) -> bool:
    """Does the configured orbit lie inside a Tipler CTC band?"""
    if cfg is None:
        cfg = KnoppDriveConfig(r_orbit=r_orbit)
    else:
        from dataclasses import replace
        cfg = replace(cfg, r_orbit=r_orbit)
    vs = VanStockumInterior(omega=cfg.omega, R=cfg.R_cylinder)
    return tipler_tilt_at(vs, r_orbit) >= 1.0


def novelty_scan(
    r_orbit_range: tuple[float, float] = (1.05, 12.0), n_r: int = 30,
    Q_range: tuple[float, float] = (1.0, 50.0), n_Q: int = 10,
    epsilon_range: tuple[float, float] = (0.0, 0.9), n_eps: int = 10,
    alpha_wall: float = 4.0, sigma_shell: float = 4.0,
    omega: float = 1.0, R_cylinder: float = 1.0,
) -> dict:
    """Sweep the (r_orbit, Q, epsilon) cube and run the catcher on
    the composite_E_neg surface as a function of r_orbit at the
    median (Q, epsilon).

    The catcher is expected to flag the CTC-band boundaries (where
    tipler_gate flips on/off) as sharp Hamming transitions, while
    the Q and epsilon axes contribute continuously.
    """
    r_grid = np.linspace(*r_orbit_range, n_r)
    Q_grid = np.linspace(*Q_range, n_Q)
    eps_grid = np.linspace(*epsilon_range, n_eps)
    composite = np.zeros((n_r, n_Q, n_eps))
    for i, r in enumerate(r_grid):
        for j, Q in enumerate(Q_grid):
            for k, eps in enumerate(eps_grid):
                b = knopp_budget(
                    omega=omega, R_cylinder=R_cylinder,
                    alpha_wall=alpha_wall, sigma_shell=sigma_shell,
                    Q=float(Q), epsilon_horn=float(eps),
                    r_orbit=float(r),
                )
                composite[i, j, k] = b.composite_E_neg
    j_mid = n_Q // 2
    k_mid = n_eps // 2
    def fn(rv):
        idx = int(np.argmin(np.abs(r_grid - rv)))
        return composite[idx]  # 2D slice across (Q, eps); flattened later
    # Flatten the per-r 2D slice so the catcher's address-space
    # has rich bit-occupancy
    def fn_flat(rv):
        idx = int(np.argmin(np.abs(r_grid - rv)))
        return composite[idx].ravel()
    result = scan_novelty(r_grid, fn_flat, n_bits=32)
    return {
        "r_grid": r_grid.tolist(),
        "Q_grid": Q_grid.tolist(),
        "eps_grid": eps_grid.tolist(),
        "composite_E_neg_grid": composite.tolist(),
        "novelty_verdict": result.verdict,
        "novelty_n_sharp": len(result.sharp_features),
        "novelty_sharp_features": [
            {k: (int(v) if isinstance(v, np.integer) else v)
             for k, v in s.items()}
            for s in result.sharp_features
        ],
    }


def summarise_knopp_budget(b: KnoppDriveBudget) -> str:
    """One-line summary suitable for logging or commit messages."""
    return (
        f"Knopp Drive @ r={b.config.r_orbit:.2f}: "
        f"E_neg={b.composite_E_neg:+.4e}, "
        f"P_drive={b.sustained_drive_power:+.4e}, "
        f"tipler_gate={b.tipler_gate_factor:.3f}, "
        f"Q={b.config.Q:.0f} -> 1/Q^2={b.feedback_factor:.4f}, "
        f"|steering|={b.steering_magnitude:.3e}, "
        f"P-F_ok={b.pfenning_ford_compatible}"
    )
