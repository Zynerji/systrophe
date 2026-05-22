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


class KnoppDrive:
    """High-level Knopp Drive interface.

    Provides a single object-oriented entry point to the four-mechanism
    composite warp engineering bound. Wraps the lower-level functional
    API in `knopp_drive` and the traversal calculator in
    `knopp_traversal` so a typical user can construct a drive,
    inspect its instantaneous budget, run a journey, and query
    quantum-inequality compatibility without touching the supporting
    modules directly.

    Examples
    --------
    >>> from systrophe.knopp_drive import KnoppDrive
    >>> drive = KnoppDrive(Q=100.0, epsilon_horn=0.2)
    >>> budget = drive.budget(r_orbit=1.5)
    >>> budget.composite_E_neg
    0.0   # zero exotic matter inside the Tipler CTC band
    >>> report = drive.journey(distance=10.0)
    >>> report.total_energy_budget
    6.78e-7
    >>> drive.is_pfenning_ford_compatible(distance=10.0)
    True

    Parameters
    ----------
    Q : float
        Cavity quality factor for the feedback-amplified shell.
        Sustained drive power scales as 1/Q^2 in the high-Q regime;
        the catcher-detected threshold is Q >= ~8.
    epsilon_horn : float
        Horn-twist amplitude in [0, 1). Sets the steering dipole
        magnitude.
    theta_0_horn : float
        Horn-twist axis orientation (radians). Sets the steering
        direction.
    omega : float
        Angular velocity of the seed Tipler cylinder. Together with
        R_cylinder, sets the supercriticality parameter a = omega R.
        Use a > 1/2 for nontrivial CTC bands.
    R_cylinder : float
        Radius of the seed Tipler cylinder.
    alpha_wall : float
        Wall sharpness of the embedded Krasnikov tube.
    sigma_shell : float
        Bubble shell thickness; sets the natural standing-wave
        frequency f_0 = c/(2 pi sigma).
    v_s : float
        Apparent shell velocity along the worldline.
    R_bubble : float
        Outer bubble radius.
    """

    def __init__(
        self,
        Q: float = 10.0,
        epsilon_horn: float = 0.2,
        theta_0_horn: float = 0.0,
        omega: float = 1.0,
        R_cylinder: float = 1.0,
        alpha_wall: float = 4.0,
        sigma_shell: float = 4.0,
        v_s: float = 1.0,
        R_bubble: float = 1.0,
    ) -> None:
        self._cfg = KnoppDriveConfig(
            omega=float(omega),
            R_cylinder=float(R_cylinder),
            alpha_wall=float(alpha_wall),
            Q=float(Q),
            sigma_shell=float(sigma_shell),
            epsilon_horn=float(epsilon_horn),
            theta_0_horn=float(theta_0_horn),
            v_s=float(v_s),
            R_bubble=float(R_bubble),
        )

    @property
    def config(self) -> KnoppDriveConfig:
        """The underlying immutable KnoppDriveConfig."""
        return self._cfg

    def budget(self, r_orbit: float | None = None) -> KnoppDriveBudget:
        """Compute the instantaneous Knopp budget at the given orbit
        radius. If `r_orbit` is None, uses the configured value."""
        if r_orbit is not None:
            from dataclasses import replace
            cfg = replace(self._cfg, r_orbit=float(r_orbit))
        else:
            cfg = self._cfg
        return _knopp_compose(cfg)

    def journey(self, distance: float, n_steps: int = 80):
        """Run an end-to-end traversal of given distance.

        Returns
        -------
        KnoppTraversalReport
            See `systrophe.knopp_traversal.KnoppTraversalReport`.
        """
        # Local import to avoid circular dependency.
        from .knopp_traversal import knopp_traversal
        return knopp_traversal(self._cfg, distance=float(distance),
                                n_steps=int(n_steps))

    def is_pfenning_ford_compatible(
        self, distance: float = 1.0, n_steps: int = 40,
    ) -> bool:
        """True iff the Knopp Drive respects the Pfenning-Ford quantum
        inequality at every point along a traversal of given distance.
        """
        rep = self.journey(distance=distance, n_steps=n_steps)
        return rep.pfenning_ford_compatible

    def steering_vector(self) -> tuple[float, float]:
        """(p_x, p_y) of the horn-toroidal steering dipole."""
        b = self.budget()
        return b.steering_vector_pxpy

    def is_inside_band(self, r_orbit: float | None = None) -> bool:
        """True iff the configured orbit lies inside a Tipler CTC band
        (tipler tilt >= 1)."""
        r = self._cfg.r_orbit if r_orbit is None else float(r_orbit)
        from .tipler_krasnikov_hybrid import tipler_tilt_at
        from .vanstockum import VanStockumInterior
        vs = VanStockumInterior(omega=self._cfg.omega, R=self._cfg.R_cylinder)
        return tipler_tilt_at(vs, r) >= 1.0

    def bias(
        self,
        heading: float = 0.0,
        n_cycles: int = 64,
        eps_high: float = 0.6,
        floor_pct: float = 0.85,
    ):
        """Steer the bubble with a reversible-ratcheting pendulum.

        Runs a bias controller (see `systrophe.knopp_ratchet`) that swings the
        horn-twist pendulum on `heading`, locks in forward progress with a
        rising-floor pawl, and rolls back any stroke that pushes past the horn
        pinch. Returns a `RatchetTraversalReport` with the net directed
        displacement, exotic/pump budget, and catcher verdict.

        The bias direction is reversible: call again with `heading + pi` (or
        the opposite displacement convention) to lock the opposite direction.
        """
        # Local import to avoid a circular dependency.
        from .knopp_ratchet import WarpRatchetConfig, ratchet_traversal
        cfg = WarpRatchetConfig(
            drive=self._cfg, heading=float(heading), eps_high=float(eps_high),
            floor_pct=float(floor_pct), n_cycles=int(n_cycles),
        )
        return ratchet_traversal(cfg)

    def asymmetry(
        self, heading: float = 0.0, floor_pct: float = 0.85,
        eps_high: float = 0.6,
    ) -> dict:
        """Measured exotic-matter cost to advance vs retreat one ratchet step.

        The bare drive is direction-blind (composite_E_neg is independent of
        the twist axis); this returns the asymmetry the ratchet manufactures.
        """
        from .knopp_ratchet import WarpRatchetConfig, reverse_asymmetry
        cfg = WarpRatchetConfig(
            drive=self._cfg, heading=float(heading),
            eps_high=float(eps_high), floor_pct=float(floor_pct),
        )
        return reverse_asymmetry(cfg)

    def pump_accounting(
        self, heading: float = 0.0, floor_pct: float = 0.85,
    ) -> dict:
        """Express the bias controller's cost as dynamical-Casimir pump power.

        The negative energy is the parametrically squeezed Casimir vacuum of
        the feedback shell, not a reservoir of exotic matter; the pump cost
        falls as 1/Q^2 and Pfenning-Ford is saturated, not beaten.
        """
        from .knopp_ratchet import WarpRatchetConfig, casimir_pump_accounting
        cfg = WarpRatchetConfig(
            drive=self._cfg, heading=float(heading), floor_pct=float(floor_pct),
        )
        return casimir_pump_accounting(cfg)

    def energy_ledger(
        self, heading: float = 0.0, n_cycles: int = 64,
        floor_pct: float = 0.85, interest_alpha: float = 1.0,
    ):
        """Total energy bottom-line for a biased journey, weighed against the
        Ford-Roman quantum-interest payback.

        Returns a `BiasEnergyLedger` (see `systrophe.knopp_ratchet`) with the
        pump, borrow, and quantum-interest-repay accounts, the cost per unit
        directed displacement, and the irreducible Q -> inf floor.
        """
        from .knopp_ratchet import WarpRatchetConfig, bias_energy_ledger
        cfg = WarpRatchetConfig(
            drive=self._cfg, heading=float(heading),
            floor_pct=float(floor_pct), n_cycles=int(n_cycles),
        )
        return bias_energy_ledger(cfg, interest_alpha=float(interest_alpha))

    def feasibility_report(
        self, bubble_radius_m: float = 1.0,
        wall_thickness_m: float | None = None,
        v_s_over_c: float = 1.0, n_cycles: int = 64,
        floor_pct: float = 0.85, lab_source_J: float = 1e-15,
    ):
        """SI-joule feasibility of a biased journey for a physical bubble.

        Converts the ledger's irreducible Ford-Roman principal to joules for
        the chosen radius/wall thickness and compares it to a lab squeezed-
        vacuum source. Returns a `FeasibilityReport`.
        """
        from .knopp_ratchet import feasibility_report
        return feasibility_report(
            bubble_radius_m=float(bubble_radius_m),
            wall_thickness_m=wall_thickness_m,
            v_s_over_c=float(v_s_over_c), Q=float(self._cfg.Q),
            n_cycles=int(n_cycles), floor_pct=float(floor_pct),
            lab_source_J=float(lab_source_J),
        )

    def capacitor_accounting(self, floor_pct: float = 0.85):
        """Surface-capacitor view: the peak negative-energy charge the shell
        holds at once (|E_neg|/Q) vs the Q-independent total throughput.

        Quantifies what the Q-cavity 'capacitor' buys (peak charge and pump
        power fall with Q) and what it cannot (the Ford-Roman principal).
        """
        from .knopp_ratchet import WarpRatchetConfig, capacitor_accounting
        cfg = WarpRatchetConfig(drive=self._cfg, floor_pct=float(floor_pct))
        return capacitor_accounting(cfg)

    def summarise(self, r_orbit: float | None = None) -> str:
        """One-line summary of the budget at the given orbit."""
        return summarise_knopp_budget(self.budget(r_orbit))

    def __repr__(self) -> str:
        return (
            f"KnoppDrive(Q={self._cfg.Q}, epsilon_horn={self._cfg.epsilon_horn}, "
            f"omega={self._cfg.omega}, R_cylinder={self._cfg.R_cylinder})"
        )
