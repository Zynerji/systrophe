"""Knopp-Drive reversible-ratcheting pendulum: a directional bias controller.

Motivation
----------
A bare Knopp Drive has no preferred direction. The composite exotic-matter
budget

    composite_E_neg = kras_bare * tipler_gate * feedback * horn_amp

(`knopp_drive._knopp_compose`) is INDEPENDENT of the horn-twist axis
``theta_0_horn``: only the steering *vector* rotates with the axis, while the
*cost* is the same forward or backward. A symmetric pendulum that swings the
twist axis back and forth therefore nets zero displacement — the warp-bubble
analogue of Purcell's scallop theorem.

To get net directed transport you must break the reciprocity of the swing.
This module ports the proven `ParetoRatchet` from the TriCameral trading stack
(`TriCameral.ai/tricameral/jdhart_v51/proven.py:154`) — specifically its
*rising-floor pawl*:

    if product > best_product:
        best_product = product
        ic_floor    = max(ic_floor,    best_ic    * floor_pct)   # one-way
        kelly_floor = max(kelly_floor, best_kelly * floor_pct)   # one-way

The floor only ever moves up. Forward progress is locked in cheaply; a stroke
that regresses below the raised floor triggers a rollback to the saved best
state. Applied to the warp bubble, this turns the pendulum into a one-way
valve: the drive accumulates forward "lean" and it costs strictly more
exotic matter to give that lean back than it cost to acquire it.

The two ratcheted metrics (the warp analogue of (IC, Kelly)) are

    displacement  d  — net forward impulse per pendulum cycle (wants eps high)
    safety        s  — pinch margin (pinch_threshold - eps_high), wants eps low

These genuinely trade off (displacement pushes eps -> 1, where the horn torus
self-intersects at the pinch threshold; safety pushes eps -> 0). The Pareto
product ``d * max(s, 1e-3)`` is what the pawl ratchets, exactly as the trading
ratchet ratchets ``ic * max(kelly, 1e-3)``.

Where the energy asymmetry comes from
-------------------------------------
Each forward pendulum cycle is a *power stroke* (twist axis on the heading,
``r_orbit`` OUTSIDE the Tipler CTC band, exotic matter spent) followed by a
*recovery stroke* routed THROUGH the band, where ``tipler_gate -> 0`` makes the
return free. To move one increment backward you forfeit that free corridor:
the reverse cycle pays a power stroke AND a non-gated recovery, plus it must
climb back over the pawl's raised floor. ``reverse_asymmetry`` measures the
ratio directly from `knopp_budget`; nothing is hard-coded.

"reversible"
------------
The bias direction is set by ``heading`` (the twist axis on the power stroke)
and the *sign convention* of the displacement metric. Re-anchoring the ratchet
with the opposite heading reverses the locked direction — so the pawl steers,
it is not a permanent one-way pin. Within a single journey it is a stable
forward bias; between journeys it is freely reversible.

This module is SPECULATIVE warp engineering. Per the project's standing rule,
its primary output is passed through the address-space novelty catcher and the
verdict is reported alongside; no result here is called "validated".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .alcubierre import alcubierre_total_negative_energy
from .feedback_amplified_shell import (
    cavity_lifetime,
    parametric_gain_to_steady,
    pfenning_ford_check,
    required_drive_power,
    saturation_field_energy,
    shell_natural_frequency,
)
from .horn_toroidal_warp import horn_pinch_threshold, horn_steering_magnitude
from .knopp_drive import KnoppDriveConfig, knopp_budget
from .novelty_catcher import scan_novelty

# --- SI physical constants (for feasibility_report) ------------------------
_C_SI = 299792458.0          # m / s
_G_SI = 6.67430e-11          # m^3 kg^-1 s^-2
_HBAR_SI = 1.054571817e-34   # J s
_JUPITER_MASS_KG = 1.898e27  # kg
# Geometrized-to-SI energy conversion: a geometrized energy of 1 metre is
# E[J] = (c^4 / G) * 1 m. (The 8 pi from G_uv = 8 pi G/c^4 T_uv is an O(1)
# convention dropped for an order-of-magnitude feasibility estimate.)
_GEOM_ENERGY_PER_METRE_J = _C_SI ** 4 / _G_SI   # ~1.21e44 J/m


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarpRatchetConfig:
    """Parameters of a reversible-ratcheting pendulum bias controller.

    The base drive is described by a `KnoppDriveConfig`; the ratchet adds the
    pendulum stroke amplitudes, the band geometry that supplies the free
    recovery, and the pawl floor fraction.
    """

    drive: KnoppDriveConfig = field(default_factory=KnoppDriveConfig)
    # Pendulum stroke amplitudes
    eps_high: float = 0.6          # horn-twist amplitude on the power stroke
    eps_low: float = 0.05          # horn-twist amplitude on the recovery stroke
    heading: float = 0.0           # twist axis on the power stroke (radians)
    # Band geometry: power stroke OUTSIDE band (cost), recovery INSIDE (free).
    # The Tipler gate is nonzero only in a finite shell (r ~ 2.5-6 for the
    # default omega=1, R=1 seed); r_out_band must sit inside that shell so the
    # power stroke actually spends exotic matter and produces gated thrust.
    r_out_band: float = 3.0        # power-stroke orbit radius (tipler_gate > 0)
    r_in_band: float = 1.5         # recovery orbit radius (tipler_gate == 0)
    # Pawl
    floor_pct: float = 0.85        # rising-floor fraction (proven.py default)
    n_cycles: int = 64


# ---------------------------------------------------------------------------
# The ratchet (faithful port of proven.py:154, warp metrics)
# ---------------------------------------------------------------------------


class WarpParetoRatchet:
    """(displacement, safety) Pareto ratchet with a rising-floor pawl.

    Direct analogue of `ParetoRatchet` from the TriCameral trading stack. The
    saved "best state" here is the (eps, heading, r_orbit) configuration that
    achieved the best Pareto product — the warp analogue of `best_state_dict`.
    """

    def __init__(
        self,
        anchor_disp: float,
        anchor_safety: float,
        floor_pct: float = 0.85,
    ) -> None:
        self.anchor_disp = anchor_disp
        self.anchor_safety = anchor_safety
        self.floor_pct = floor_pct
        self.disp_floor = anchor_disp * floor_pct
        self.safety_floor = (
            anchor_safety * floor_pct if anchor_safety > 0 else -0.05
        )
        self.best_disp = anchor_disp
        self.best_safety = anchor_safety
        self.best_product = anchor_disp * max(anchor_safety, 1e-3)
        self.best_config: tuple[float, float, float] | None = None
        self.rollback_count = 0
        self.advance_count = 0

    def check(self, disp: float, safety: float) -> str:
        """Return one of 'advance' / 'hold' / 'rollback'.

        Mirrors `ParetoRatchet.check` (proven.py): the floor only rises.
        """
        product = disp * max(safety, 1e-3)
        if product > self.best_product:
            self.best_product = product
            self.best_disp = max(self.best_disp, disp)
            self.best_safety = max(self.best_safety, safety)
            # rising-floor pawl — the one-way valve
            self.disp_floor = max(self.disp_floor, self.best_disp * self.floor_pct)
            self.safety_floor = max(
                self.safety_floor, self.best_safety * self.floor_pct
            )
            self.advance_count += 1
            return "advance"
        if disp < self.disp_floor and safety < self.safety_floor:
            self.rollback_count += 1
            return "rollback"
        self.advance_count += 1
        return "hold"

    def __repr__(self) -> str:
        return (
            f"WarpParetoRatchet(best_disp={self.best_disp:.4e}, "
            f"best_safety={self.best_safety:.4f}, "
            f"product={self.best_product:.4e}, "
            f"advances={self.advance_count}, rollbacks={self.rollback_count})"
        )


# ---------------------------------------------------------------------------
# Per-cycle physics, sourced from knopp_budget
# ---------------------------------------------------------------------------


def _pf_bound(sigma: float) -> float:
    """Pfenning-Ford lower bound 3/(32 pi^2 sigma^2)."""
    return 3.0 / (32.0 * math.pi ** 2 * sigma ** 2)


def _stroke(
    cfg: WarpRatchetConfig, eps: float, r_orbit: float,
) -> tuple[float, float, float]:
    """Evaluate one stroke. Returns (forward_impulse, exotic_cost, pf_margin).

    forward_impulse  = |steering dipole| * tipler_gate  (gated to 0 in band)
    exotic_cost      = |composite_E_neg|
    pf_margin        = |E_neg| * tau - PF_bound  (>=0 => quantum-compatible)
    """
    b = knopp_budget(
        cfg.drive,
        epsilon_horn=float(eps),
        theta_0_horn=float(cfg.heading),
        r_orbit=float(r_orbit),
    )
    # The steering dipole magnitude (independent of the gate); the gate decides
    # whether that momentum flux actually couples to translation this stroke.
    # Steering only works where there IS exotic matter to steer, so the impulse
    # is gated by the same tipler factor as the cost. Past the horn pinch
    # threshold the torus self-intersects and the dipole no longer translates.
    pinch = horn_pinch_threshold(R=cfg.drive.R_bubble)
    dipole = horn_steering_magnitude(
        R=cfg.drive.R_bubble, epsilon=float(eps),
        theta_0=float(cfg.heading), m_ADM=-1.0, sigma=cfg.drive.sigma_shell,
    )
    geom_ok = 1.0 if eps < pinch else 0.0
    forward_impulse = dipole * b.tipler_gate_factor * geom_ok
    exotic_cost = abs(b.composite_E_neg)
    if exotic_cost < 1e-12:
        pf_margin = 1.0  # free in-band coast is trivially PF-compatible
    else:
        pf_margin = exotic_cost * b.cavity_lifetime_tau - _pf_bound(
            cfg.drive.sigma_shell
        )
    return forward_impulse, exotic_cost, pf_margin


@dataclass(frozen=True)
class RatchetCycleStep:
    cycle: int
    eps_high: float
    displacement: float       # net forward impulse this cycle
    safety: float             # pinch margin
    exotic_cost: float        # exotic matter spent this cycle
    pf_margin: float          # PF headroom on the power stroke
    decision: str             # advance / hold / rollback
    cumulative_displacement: float
    locked_floor: float       # ratchet's current displacement floor


@dataclass(frozen=True)
class RatchetTraversalReport:
    config: WarpRatchetConfig
    steps: tuple[RatchetCycleStep, ...]
    net_displacement: float
    total_exotic_cost: float
    advances: int
    rollbacks: int
    pf_compatible: bool       # every power stroke respected PF
    novelty_verdict: str
    novelty_n_sharp: int


def ratchet_traversal(
    cfg: WarpRatchetConfig | None = None,
    eps_schedule: np.ndarray | None = None,
) -> RatchetTraversalReport:
    """Run the reversible-ratcheting pendulum for ``n_cycles`` and accumulate
    net forward displacement.

    Each cycle is a power stroke (eps_high, out of band) followed by a free
    Tipler-gated recovery (eps_low, in band). The ratchet locks in forward
    progress and rolls back configurations that push ``eps`` past the horn
    pinch threshold (safety below floor).

    ``eps_schedule`` optionally overrides the per-cycle power-stroke amplitude
    (e.g. a slowly ramping or oscillating pendulum); defaults to constant
    ``cfg.eps_high``.
    """
    if cfg is None:
        cfg = WarpRatchetConfig()
    pinch = horn_pinch_threshold(R=cfg.drive.R_bubble)

    if eps_schedule is None:
        eps_schedule = np.full(cfg.n_cycles, cfg.eps_high, dtype=float)
    eps_schedule = np.asarray(eps_schedule, dtype=float)

    # Anchor the ratchet at the nominal stroke.
    fwd0, _, _ = _stroke(cfg, cfg.eps_high, cfg.r_out_band)
    safety0 = max(pinch - cfg.eps_high, 0.0)
    ratchet = WarpParetoRatchet(
        anchor_disp=fwd0, anchor_safety=safety0, floor_pct=cfg.floor_pct,
    )
    ratchet.best_config = (cfg.eps_high, cfg.heading, cfg.r_out_band)

    steps: list[RatchetCycleStep] = []
    cum = 0.0
    total_cost = 0.0
    pf_ok_all = True
    cur_eps = cfg.eps_high

    for i in range(cfg.n_cycles):
        # Allow the schedule to ramp past the pinch (eps >= 1) so the geometry
        # break is modelled and triggers the ratchet's rollback rather than a
        # silent runaway; cap only at an absurd value.
        eps_try = float(min(eps_schedule[i], 2.0))
        # Power stroke (out of band) + free recovery (in band).
        fwd, cost, pf_margin = _stroke(cfg, eps_try, cfg.r_out_band)
        rec, rec_cost, _ = _stroke(cfg, cfg.eps_low, cfg.r_in_band)
        disp = fwd - rec            # rec ~ 0: the band gates the recovery away
        safety = max(pinch - eps_try, 0.0)
        cycle_cost = cost + rec_cost
        if pf_margin < 0:
            pf_ok_all = False

        decision = ratchet.check(disp, safety)
        if decision == "rollback" and ratchet.best_config is not None:
            # Restore the best locked configuration (analogue of
            # model.load_state_dict(best_state_dict)). The bubble does not
            # advance this cycle and forfeits the spent budget.
            cur_eps = ratchet.best_config[0]
        else:
            cur_eps = eps_try
            cum += max(disp, 0.0)
            total_cost += cycle_cost
            if decision == "advance":
                ratchet.best_config = (eps_try, cfg.heading, cfg.r_out_band)

        steps.append(RatchetCycleStep(
            cycle=i, eps_high=eps_try, displacement=disp, safety=safety,
            exotic_cost=cycle_cost, pf_margin=pf_margin, decision=decision,
            cumulative_displacement=cum, locked_floor=ratchet.disp_floor,
        ))

    # Address-space novelty catcher over the cumulative-displacement trace.
    cum_trace = np.array([s.cumulative_displacement for s in steps])
    floor_trace = np.array([s.locked_floor for s in steps])
    axis = np.arange(len(steps), dtype=float)

    def fn(c: float) -> np.ndarray:
        idx = int(round(c))
        idx = max(0, min(idx, len(steps) - 1))
        return np.array([cum_trace[idx], floor_trace[idx],
                         steps[idx].displacement])

    result = scan_novelty(axis, fn, n_bits=32, parameter_label="cycle")

    return RatchetTraversalReport(
        config=cfg,
        steps=tuple(steps),
        net_displacement=float(cum),
        total_exotic_cost=float(total_cost),
        advances=ratchet.advance_count,
        rollbacks=ratchet.rollback_count,
        pf_compatible=bool(pf_ok_all),
        novelty_verdict=result.verdict,
        novelty_n_sharp=len(result.sharp_features),
    )


# ---------------------------------------------------------------------------
# The directional energy asymmetry (measured, not assumed)
# ---------------------------------------------------------------------------


def reverse_asymmetry(cfg: WarpRatchetConfig | None = None) -> dict:
    """Measure the exotic-matter cost to advance vs retreat one ratchet step.

    Forward step: power stroke (out of band) + FREE Tipler-gated recovery.
    Reverse step: power stroke (out of band) + a non-gated recovery (the free
    forward corridor is forfeited) + the pawl penalty of climbing back over
    the raised floor.

    All energies come from `knopp_budget`; the ratio is reported, not assumed.
    """
    if cfg is None:
        cfg = WarpRatchetConfig()
    pinch = horn_pinch_threshold(R=cfg.drive.R_bubble)

    fwd, e_power, pf_margin = _stroke(cfg, cfg.eps_high, cfg.r_out_band)
    _, e_recovery_free, _ = _stroke(cfg, cfg.eps_low, cfg.r_in_band)   # ~ 0

    # Forward: pay the power stroke, recover for free in the band.
    e_advance = e_power + e_recovery_free

    # Reverse: power stroke (cost is direction-symmetric — composite_E_neg does
    # not depend on theta_0_horn) PLUS a non-gated recovery out of the band.
    _, e_recovery_nongated, _ = _stroke(cfg, cfg.eps_low, cfg.r_out_band)
    e_reverse_strokes = e_power + e_recovery_nongated

    # Pawl penalty: the rising floor acts as a restoring resistance against
    # backward motion. Its stiffness diverges as floor_pct -> 1 (perfect
    # one-way valve) and vanishes at floor_pct = 0 (no pawl => reciprocal
    # pendulum, asymmetry reduces to the forfeited free-recovery term alone).
    # Energy to push one step Δ=fwd backward against it, at the forward step's
    # per-displacement exotic-matter rate.
    disp_per_step = max(fwd, 1e-12)
    rate = e_advance / disp_per_step
    fp = min(max(cfg.floor_pct, 0.0), 1.0 - 1e-9)
    stiffness = fp / (1.0 - fp)
    e_pawl = stiffness * fwd * rate
    e_reverse = e_reverse_strokes + e_pawl

    return {
        "e_advance": float(e_advance),
        "e_reverse": float(e_reverse),
        "e_reverse_strokes": float(e_reverse_strokes),
        "e_pawl": float(e_pawl),
        "asymmetry_ratio": float(e_reverse / max(e_advance, 1e-30)),
        "forward_impulse_per_step": float(fwd),
        "power_stroke_pf_margin": float(pf_margin),
        "pinch_margin": float(max(pinch - cfg.eps_high, 0.0)),
    }


# ---------------------------------------------------------------------------
# Casimir / dynamical-Casimir pump accounting (the "no exotic reservoir" route)
# ---------------------------------------------------------------------------


def casimir_pump_accounting(cfg: WarpRatchetConfig | None = None) -> dict:
    """Re-express the ratchet's per-cycle cost as dynamical-Casimir pump power.

    The shell's negative energy is NOT a separate slab of exotic matter. It is
    the parametrically squeezed Casimir vacuum modelled in
    ``feedback_amplified_shell`` — degenerate parametric amplification at the
    pump frequency 2*f_0, i.e. the dynamical Casimir effect of
    ``dynamical_casimir`` (Wilson et al. 2011 observed exactly this in a
    superconducting circuit). Static parallel-plate Casimir is far too weak
    (``exotic_matter_accounting`` returns a quantitative NO); the amplified
    DCE route is what makes the budget reachable.

    No free lunch is claimed: sustaining the squeezed vacuum costs a real,
    positive-energy pump

        P_drive = |E_neg_bare| * f_0 / Q^2          (falls as 1/Q^2)

    and the Pfenning-Ford quantum inequality is SATURATED, not beaten — the
    high-Q cavity trades instantaneous amplitude (1/Q) for duration (Q),
    leaving the bounded product invariant. The power stroke pumps the shell;
    the recovery routes through the Tipler band, where the rotating-cylinder
    geometry supplies the light-cone tilt for free (zero pump).

    Returns the pump power for each half-cycle, the PF check, and the pump-
    energy asymmetry between advancing and retreating one ratchet step.
    """
    if cfg is None:
        cfg = WarpRatchetConfig()
    d = cfg.drive
    f_0 = shell_natural_frequency(d.sigma_shell)
    p_power = required_drive_power(
        v_s=d.v_s, R=d.R_bubble, sigma=d.sigma_shell, Q=d.Q,
    )
    pf = pfenning_ford_check(
        v_s=d.v_s, R=d.R_bubble, sigma=d.sigma_shell, Q=d.Q,
    )
    asym = reverse_asymmetry(cfg)
    return {
        "power_stroke_pump_power": float(p_power),         # real lab pump
        "recovery_pump_power": 0.0,                        # free Tipler coast
        "shell_natural_frequency_f0": float(f_0),
        "Q": float(d.Q),
        "pfenning_ford_compatible": bool(pf["compatible"]),
        "pf_product": float(pf["stored_energy_times_tau"]),
        "pf_bound": float(pf["pfenning_ford_lower_bound"]),
        "pump_asymmetry_ratio": float(asym["asymmetry_ratio"]),
        "negative_energy_source": "parametrically squeezed Casimir vacuum (DCE)",
        "needs_exotic_matter_reservoir": False,
    }


# ---------------------------------------------------------------------------
# Surface-capacitor accounting (peak charge vs total throughput)
# ---------------------------------------------------------------------------


def capacitor_accounting(cfg: WarpRatchetConfig | None = None) -> dict:
    """The feedback shell as a surface 'capacitor': peak charge vs throughput.

    The high-Q standing wave on the bubble shell is an energy accumulator. It
    charges up (rings up) under the parametric/DCE pump as

        E_shell(t) = E_shell(0) * exp(g t),   g = ln(Q) / tau,

    and at saturation holds only

        E_peak = |E_neg_bare| / Q          (the instantaneous capacitor charge)

    sustained over the cavity lifetime tau = Q / f_0. So the negative energy
    *present on the surface at any instant* is reduced by a factor Q: you never
    marshal the whole requirement at once — you trickle-charge a much smaller
    standing charge and let it act over the long cavity lifetime.

    What the capacitor does NOT change is the time-integrated energy that must
    flow and be repaid under Ford-Roman: the throughput equals the ledger
    floor and is Q-INDEPENDENT (see `bias_energy_ledger`). The capacitor
    solves the PEAK-charge and POWER problems; it cannot lower the irreducible
    TOTAL. Both truths, quantified.
    """
    if cfg is None:
        cfg = WarpRatchetConfig()
    d = cfg.drive
    e_neg_bare = abs(alcubierre_total_negative_energy(
        v_s=d.v_s, R=d.R_bubble, sigma=d.sigma_shell,
    ))
    e_peak = saturation_field_energy(
        v_s=d.v_s, R=d.R_bubble, sigma=d.sigma_shell, Q=d.Q,
    )
    tau = cavity_lifetime(d.Q, d.sigma_shell)
    g = parametric_gain_to_steady(
        v_s=d.v_s, R=d.R_bubble, sigma=d.sigma_shell, Q=d.Q,
    )
    throughput = bias_energy_ledger(cfg).irreducible_floor
    return {
        "peak_charge_natural": float(e_peak),               # held at once
        "bare_requirement_natural": float(e_neg_bare),      # Q=1, no cavity
        "peak_reduction_factor": float(d.Q),                # = Q
        "ringup_time_tau": float(tau),                      # charge time
        "parametric_gain_g": float(g),
        "total_throughput_natural": float(throughput),      # = ledger floor
        "throughput_is_Q_independent": True,
        "what_the_capacitor_buys": "peak instantaneous charge ↓ by 1/Q, "
                                   "and trickle-charge power ↓ by 1/Q^2",
        "what_it_cannot_buy": "the time-integrated Ford-Roman principal "
                              "(the Jupiter-mass floor) is unchanged",
    }


# ---------------------------------------------------------------------------
# Bias energy ledger with Ford-Roman quantum-interest payback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BiasEnergyLedger:
    """Full energy bottom-line for a biased journey, in natural units.

    Three accounts, all per the amplified-Casimir route:
      * pump      — real positive-energy input that sustains the squeezed
                    Casimir vacuum (P_drive * tau per advancing stroke).
      * borrow    — negative energy held in the shell each stroke.
      * repay     — Ford-Roman quantum-interest repayment: principal + premium.

    The bottom line ``e_ledger_total = e_pump_total + e_repay_total`` is the
    total energy that has to flow to make the journey, and
    ``energy_per_unit_displacement`` is the cost of directed transport.
    """

    n_cycles: int
    n_advancing_strokes: int
    net_displacement: float
    e_neg_bare_per_stroke: float        # |E_neg| before 1/Q feedback
    # Pump account
    e_pump_per_stroke: float
    e_pump_total: float
    # Borrow / Ford-Roman repay account
    e_borrow_per_stroke: float
    e_borrow_total: float
    hold_to_pulse_ratio: float          # tau_hold / tau_pulse  (= Q)
    interest_alpha: float
    interest_rate: float                # r in E_repay = E_borrow (1 + r)
    e_interest_total: float             # the quantum-interest premium only
    e_repay_total: float                # principal + interest
    # Bottom line
    e_ledger_total: float
    energy_per_unit_displacement: float
    irreducible_floor: float            # Q -> inf asymptote of e_ledger_total
    reverse_undo_energy: float          # ledger cost to unwind, via asymmetry
    pfenning_ford_compatible: bool
    novelty_verdict: str
    novelty_n_sharp: int


def bias_energy_ledger(
    cfg: WarpRatchetConfig | None = None,
    interest_alpha: float = 1.0,
) -> BiasEnergyLedger:
    """Integrate the total energy of a biased journey and weigh it against
    the Ford-Roman quantum-interest payback.

    Energy bookkeeping (feedback-shell / dynamical-Casimir route)
    ------------------------------------------------------------
    Per advancing power stroke the shell holds borrowed negative energy

        E_borrow = saturation_field_energy = |E_neg_bare| / Q

    pumped in over one cavity fill time tau_hold = Q / f_0 at the sustained
    rate P_drive = |E_neg_bare| f_0 / Q^2, so the real positive-energy pump
    input is E_pump = P_drive * tau_hold = |E_neg_bare| / Q.

    Ford-Roman quantum interest (Ford & Roman, PRD 60 (1999) 104018)
    ---------------------------------------------------------------
    Negative energy borrowed for a time tau_hold must be repaid by a LARGER
    positive pulse; the overpayment ("interest") grows with how long it is
    held relative to the pulse width tau_pulse = 1/f_0. This is modelled as

        E_repay = E_borrow * (1 + r),  r = (tau_hold/tau_pulse)^alpha - 1
                                          = Q^alpha - 1

    with ``interest_alpha`` a documented model exponent (NOT an exact theorem;
    Ford-Roman fix the existence and monotonicity of the premium, not this
    closed form). At the default alpha = 1 the per-stroke debt is conserved:

        E_repay = (|E_neg_bare|/Q) * Q = |E_neg_bare|   (Q-independent)

    so the bottom line is

        E_ledger = E_pump + E_repay = |E_neg_bare| (1 + 1/Q)  per stroke,

    which falls toward the irreducible floor |E_neg_bare| as Q -> inf:
    amplification removes the *pump overhead* but never the quantum-interest
    *principal*. For alpha > 1 (superlinear interest) the ledger has an
    interior optimal Q instead.
    """
    if cfg is None:
        cfg = WarpRatchetConfig()
    d = cfg.drive

    rep = ratchet_traversal(cfg)
    n_adv = sum(1 for s in rep.steps if s.decision != "rollback")
    net = rep.net_displacement

    f_0 = shell_natural_frequency(d.sigma_shell)
    tau_hold = cavity_lifetime(d.Q, d.sigma_shell)
    tau_pulse = 1.0 / f_0
    e_neg_bare = abs(alcubierre_total_negative_energy(
        v_s=d.v_s, R=d.R_bubble, sigma=d.sigma_shell,
    ))

    e_borrow = saturation_field_energy(
        v_s=d.v_s, R=d.R_bubble, sigma=d.sigma_shell, Q=d.Q,
    )
    e_pump = required_drive_power(
        v_s=d.v_s, R=d.R_bubble, sigma=d.sigma_shell, Q=d.Q,
    ) * tau_hold

    hold_ratio = tau_hold / tau_pulse                       # == Q
    r = max(hold_ratio ** interest_alpha - 1.0, 0.0)
    e_interest_stroke = e_borrow * r
    e_repay_stroke = e_borrow + e_interest_stroke
    e_ledger_stroke = e_pump + e_repay_stroke

    e_pump_total = e_pump * n_adv
    e_borrow_total = e_borrow * n_adv
    e_interest_total = e_interest_stroke * n_adv
    e_repay_total = e_repay_stroke * n_adv
    e_ledger_total = e_ledger_stroke * n_adv
    floor = e_neg_bare * n_adv                              # Q -> inf asymptote

    epud = e_ledger_total / net if net > 1e-30 else float("inf")
    asym = reverse_asymmetry(cfg)["asymmetry_ratio"]
    reverse_undo = e_ledger_total * asym

    # Address-space catcher over the cumulative ledger-energy curve.
    cum = 0.0
    cum_trace = []
    for s in rep.steps:
        if s.decision != "rollback":
            cum += e_ledger_stroke
        cum_trace.append(cum)
    cum_trace = np.array(cum_trace) if cum_trace else np.array([0.0])
    axis = np.arange(len(cum_trace), dtype=float)

    def fn(c: float) -> np.ndarray:
        idx = max(0, min(int(round(c)), len(cum_trace) - 1))
        return np.array([cum_trace[idx]])

    nov = scan_novelty(axis, fn, n_bits=32, parameter_label="cycle")

    return BiasEnergyLedger(
        n_cycles=cfg.n_cycles,
        n_advancing_strokes=int(n_adv),
        net_displacement=float(net),
        e_neg_bare_per_stroke=float(e_neg_bare),
        e_pump_per_stroke=float(e_pump),
        e_pump_total=float(e_pump_total),
        e_borrow_per_stroke=float(e_borrow),
        e_borrow_total=float(e_borrow_total),
        hold_to_pulse_ratio=float(hold_ratio),
        interest_alpha=float(interest_alpha),
        interest_rate=float(r),
        e_interest_total=float(e_interest_total),
        e_repay_total=float(e_repay_total),
        e_ledger_total=float(e_ledger_total),
        energy_per_unit_displacement=float(epud),
        irreducible_floor=float(floor),
        reverse_undo_energy=float(reverse_undo),
        pfenning_ford_compatible=bool(rep.pf_compatible),
        novelty_verdict=nov.verdict,
        novelty_n_sharp=len(nov.sharp_features),
    )


def ledger_Q_sweep(
    Q_range: tuple[float, float] = (1.0, 200.0),
    n_Q: int = 24,
    cfg: WarpRatchetConfig | None = None,
    interest_alpha: float = 1.0,
) -> dict:
    """Sweep Q and report total ledger energy + cost-per-displacement.

    At alpha = 1 the ledger falls monotonically toward the irreducible floor;
    at alpha > 1 the catcher should flag the interior optimum where pump
    overhead and quantum-interest premium cross.
    """
    if cfg is None:
        cfg = WarpRatchetConfig()
    from dataclasses import replace
    Qs = np.linspace(*Q_range, n_Q)
    total = np.zeros(n_Q)
    epud = np.zeros(n_Q)
    for i, Q in enumerate(Qs):
        c = replace(cfg, drive=replace(cfg.drive, Q=float(Q)))
        led = bias_energy_ledger(c, interest_alpha=interest_alpha)
        total[i] = led.e_ledger_total
        epud[i] = led.energy_per_unit_displacement

    def fn(qv: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(Qs - qv)))
        return np.array([total[idx], epud[idx]])

    nov = scan_novelty(Qs, fn, n_bits=32, parameter_label="Q")
    opt_idx = int(np.argmin(epud))
    return {
        "Q_grid": Qs.tolist(),
        "e_ledger_total": total.tolist(),
        "energy_per_unit_displacement": epud.tolist(),
        "optimal_Q": float(Qs[opt_idx]),
        "interest_alpha": interest_alpha,
        "novelty_verdict": nov.verdict,
        "novelty_n_sharp": len(nov.sharp_features),
    }


# ---------------------------------------------------------------------------
# SI feasibility report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeasibilityReport:
    """The bias-journey energy floor in SI joules, with a reality check.

    The natural-units ledger is dimensionless; this puts a real number on the
    irreducible Ford-Roman principal for a chosen physical bubble radius and
    wall thickness, and compares it to a lab squeezed-vacuum source.
    """

    bubble_radius_m: float
    wall_thickness_m: float
    v_s_over_c: float
    Q: float
    n_advancing_strokes: int
    alcubierre_coefficient: float
    # SI energies
    e_neg_bare_per_stroke_J: float
    irreducible_floor_J: float       # n_adv * per-stroke principal (Q->inf)
    pump_overhead_J: float           # floor / Q
    e_ledger_total_J: float          # floor + pump overhead
    # Human-scale comparisons
    floor_mass_equivalent_kg: float
    floor_jupiter_masses: float
    # Reality check
    lab_source_J: float
    shortfall_factor: float          # floor / lab source
    shortfall_orders_of_magnitude: float
    pfenning_ford_compatible: bool
    verdict: str


def feasibility_report(
    bubble_radius_m: float = 1.0,
    wall_thickness_m: float | None = None,
    v_s_over_c: float = 1.0,
    Q: float = 100.0,
    n_cycles: int = 64,
    floor_pct: float = 0.85,
    lab_source_J: float = 1e-15,
    alcubierre_coefficient: float = 1.0 / 12.0,
) -> FeasibilityReport:
    """Convert the bias-energy-ledger floor to SI joules and reality-check it.

    The per-stroke negative-energy principal uses the textbook geometrized
    Alcubierre estimate

        E_geom = kappa * (v_s/c)^2 * R^2 / sigma          [metres]
        E_SI   = E_geom * c^4 / G                          [joules]

    with kappa = ``alcubierre_coefficient`` (~1/12, an O(1) literature value;
    the order-of-magnitude conclusion is insensitive to it). This deliberately
    does NOT route the repo's `alcubierre_total_negative_energy` through c^4/G:
    that integrator scales as R^2*sigma (a relative comparator), not as
    geometrized energy, so it is not a calibrated absolute. The ledger is used
    only for the journey structure (number of advancing strokes, Q).

    ``lab_source_J`` is the negative energy a state-of-the-art squeezed-vacuum
    / dynamical-Casimir source can marshal (default 1e-15 J, a generous
    placeholder ~ 1e9 microwave quanta; override with a real figure).
    """
    if wall_thickness_m is None:
        wall_thickness_m = bubble_radius_m

    # Journey structure from the natural-units ledger (scale-invariant).
    cfg = WarpRatchetConfig(
        drive=KnoppDriveConfig(Q=float(Q)),
        floor_pct=float(floor_pct), n_cycles=int(n_cycles),
    )
    led = bias_energy_ledger(cfg, interest_alpha=1.0)
    n_adv = led.n_advancing_strokes

    # SI per-stroke principal via the geometrized Alcubierre estimate.
    e_geom_m = (alcubierre_coefficient * v_s_over_c ** 2
                * bubble_radius_m ** 2 / wall_thickness_m)
    e_stroke_J = abs(e_geom_m) * _GEOM_ENERGY_PER_METRE_J

    floor_J = n_adv * e_stroke_J
    pump_overhead_J = floor_J / max(Q, 1e-30)
    total_J = floor_J + pump_overhead_J

    mass_kg = floor_J / _C_SI ** 2
    jupiter = floor_J / (_JUPITER_MASS_KG * _C_SI ** 2)

    shortfall = floor_J / max(lab_source_J, 1e-300)
    oom = math.log10(shortfall) if shortfall > 0 else float("-inf")
    verdict = (
        f"INFEASIBLE with current sources: irreducible principal exceeds a "
        f"lab squeezed-vacuum budget by ~{oom:.0f} orders of magnitude "
        f"({jupiter:.2g} Jupiter mass-energies)."
    )

    return FeasibilityReport(
        bubble_radius_m=float(bubble_radius_m),
        wall_thickness_m=float(wall_thickness_m),
        v_s_over_c=float(v_s_over_c),
        Q=float(Q),
        n_advancing_strokes=int(n_adv),
        alcubierre_coefficient=float(alcubierre_coefficient),
        e_neg_bare_per_stroke_J=float(e_stroke_J),
        irreducible_floor_J=float(floor_J),
        pump_overhead_J=float(pump_overhead_J),
        e_ledger_total_J=float(total_J),
        floor_mass_equivalent_kg=float(mass_kg),
        floor_jupiter_masses=float(jupiter),
        lab_source_J=float(lab_source_J),
        shortfall_factor=float(shortfall),
        shortfall_orders_of_magnitude=float(oom),
        pfenning_ford_compatible=bool(led.pfenning_ford_compatible),
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Novelty sweep over the pawl strength
# ---------------------------------------------------------------------------


def novelty_scan(
    floor_range: tuple[float, float] = (0.0, 0.99),
    n_floor: int = 30,
    cfg: WarpRatchetConfig | None = None,
) -> dict:
    """Sweep the pawl floor fraction and catch where the ratchet's locked
    behaviour changes character.

    Predicted catcher hit near floor_pct -> 1 (the pawl becomes so strict that
    every cycle rolls back — the bias freezes) and near floor_pct -> 0 (no
    pawl: reciprocal pendulum, net drift collapses).
    """
    if cfg is None:
        cfg = WarpRatchetConfig()
    floors = np.linspace(*floor_range, n_floor)
    net = np.zeros(n_floor)
    rollbacks = np.zeros(n_floor)
    ratio = np.zeros(n_floor)
    for i, fp in enumerate(floors):
        from dataclasses import replace
        c = replace(cfg, floor_pct=float(fp))
        rep = ratchet_traversal(c)
        net[i] = rep.net_displacement
        rollbacks[i] = rep.rollbacks
        ratio[i] = reverse_asymmetry(c)["asymmetry_ratio"]

    def fn(fv: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(floors - fv)))
        return np.array([net[idx], rollbacks[idx], ratio[idx]])

    result = scan_novelty(floors, fn, n_bits=32, parameter_label="floor_pct")
    return {
        "floor_grid": floors.tolist(),
        "net_displacement": net.tolist(),
        "rollbacks": rollbacks.tolist(),
        "asymmetry_ratio": ratio.tolist(),
        "novelty_verdict": result.verdict,
        "novelty_n_sharp": len(result.sharp_features),
        "novelty_sharp_features": [
            {k: (int(v) if isinstance(v, np.integer) else v)
             for k, v in s.items()}
            for s in result.sharp_features
        ],
    }


def summarise_ratchet(rep: RatchetTraversalReport) -> str:
    """One-line summary suitable for logging."""
    return (
        f"WarpRatchet {rep.config.n_cycles} cycles: "
        f"net_disp={rep.net_displacement:.3e}, "
        f"E_total={rep.total_exotic_cost:.3e}, "
        f"advances={rep.advances}, rollbacks={rep.rollbacks}, "
        f"PF_ok={rep.pf_compatible}, novelty={rep.novelty_verdict}"
    )
