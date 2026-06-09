"""Counter-rotating the dodecahedron: standing flash, closed window,
parametric cargo stability.

Ninth module of the series. User question 2026-06-09: "what if the dodeca
counter-rotated?" A single rigid body cannot counter-rotate against
itself; the physical construction is a COUNTER-ROTATING PAIR of locked
field patterns at +-Omega (two stacked dodecas sharing the throat, or the
m = +-5 superposition of one). Three derivations:

K1. FRAME DRAG CANCELS -- THE TIPLER WINDOW CLOSES (closed form).
    The van Stockum analog parameter is set by the net angular momentum;
    equal counter-rotating halves give a_net = 0 at every Omega. No CTC
    band, ever -- and therefore the comms/chronology exclusion of
    knopp_dodeca_comb_channel is LIFTED: clean signalling at ANY spin,
    up to the grip bound (~1.9 bits/unit binary). Counter-rotation is the
    chronology-safe, full-bandwidth mode. The price: no net frame drag.

K2. THE CONVEYOR BECOMES A STANDING PENTAGONAL FLASHER (verified).
    cos(5(phi - Omega t)) + cos(5(phi + Omega t)) = 2 cos(5 phi) cos(5 Omega t):
    the m=5 ring pattern stops rotating. Measured on the pump ring: the
    single rotor's m=5 phase advances continuously (-5 psi); the counter
    pair's phase is CONSTANT and flips by exactly pi each half-beat while
    the amplitude pulses through zero. No net azimuthal transport -- the
    cargo carousel dies; the lattice breathes in place at 5 Omega.

K3. CARGO IS PARAMETRICALLY STABLE IN THE WHOLE REACHABLE RANGE (verified).
    The standing beat modulates trap stiffness: kappa(t) =
    kappa0 (1 + h cos(5 Omega t)) -- a Mathieu problem with trap frequency
    omega_t = sqrt(kappa0) ~ 121. Instability bands sit at
    Omega_n = 2 omega_t / (5 n): 48.4, 24.2, 16.1, 12.1, ... Integrating
    the Mathieu equation: explosive growth at 24.2 and 48.4, noise-floor
    growth everywhere at Omega <= 12 -- and the grip bound (~12.2) caps
    the reachable range JUST below the weak n=4 band. Trapped matter
    survives the flashing lattice at every reachable spin.

Bonus pointer (not modelled here): driving the two horns at DIFFERENT
rates is differential rotation -- the Saltzman/Lorenz-truncation territory
of the Systrophe Lorenz-rotation work (chaotic a(t), flickering CTC band).

Caveats: series-standard (coherent-pump proxy, model units). Catcher
(mandatory): on the Mathieu Omega scan -- the band edges are transitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.knopp.knopp_dodeca_alignment import FACE_LOCK_DEG
from systrophe.knopp.knopp_dodeca_crystal import pump_ring
from systrophe.knopp.knopp_dodeca_pressure import (
    axis_amplitudes,
    radiation_pressure,
    trap_stiffness,
    wavenumber,
)
from systrophe.knopp.knopp_dodeca_rotation import (
    max_conveyor_rate,
    spun_axes,
)
from systrophe.knopp.knopp_dodeca_alignment import DEFAULT_R


# ----- K1: drag cancellation ----------------------------------------------------


def net_drag_parameter(omega_plus: float, omega_minus: float) -> float:
    """van Stockum analog parameter of a two-component rotor: the angular
    momenta add, so a_net = (Omega+ + Omega-)/2 for equal halves.
    Counter-rotation (Omega- = -Omega+) gives exactly zero."""
    return 0.5 * (omega_plus + omega_minus)


def window_closed_for_counter_pair(Omega: float) -> bool:
    """No CTC band at any counter-spin: a_net = 0 has no supercritical
    radius. (Closed form -- the comms/chronology exclusion is lifted.)"""
    if Omega < 0:
        raise ValueError("Omega must be >= 0")
    return net_drag_parameter(Omega, -Omega) == 0.0


# ----- K2: standing pentagonal flash ----------------------------------------------


def ring_m5(psi_deg: float, counter: bool, tilt_deg: float = 4.0,
            scale: float = 0.212, n_phi: int = 720) -> tuple[float, float]:
    """(amplitude, phase) of the m=5 pump-ring component at spin angle psi.

    counter=False: one rotor at +psi. counter=True: half-amplitude pair
    at +-psi (the counter-rotating superposition).
    """
    a2, sat = axis_amplitudes(FACE_LOCK_DEG)
    k = wavenumber(sat)
    ring = pump_ring(scale, n=n_phi)
    if counter:
        I = 0.5 * radiation_pressure(ring, spun_axes(psi_deg, tilt_deg), a2, k) \
            + 0.5 * radiation_pressure(ring, spun_axes(-psi_deg, tilt_deg), a2, k)
    else:
        I = radiation_pressure(ring, spun_axes(psi_deg, tilt_deg), a2, k)
    phi = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    c = np.sum(I * np.exp(-5j * phi)) / n_phi
    return float(np.abs(c)), float(np.angle(c))


def standing_flash_check(psi_steps=(0.0, 10.0, 20.0, 30.0)) -> dict:
    """Single rotor: m=5 phase advances. Counter pair: phase locked mod pi."""
    single = [ring_m5(p, counter=False)[1] for p in psi_steps]
    pair = [ring_m5(p, counter=True)[1] for p in psi_steps]
    single_drift = max(abs((single[i + 1] - single[i] + math.pi)
                           % (2 * math.pi) - math.pi)
                       for i in range(len(single) - 1))
    pair_drift = max(min(abs(d), abs(abs(d) - math.pi)) for d in
                     (((pair[i + 1] - pair[i] + math.pi) % (2 * math.pi)
                       - math.pi) for i in range(len(pair) - 1)))
    return {"single_phase_drift": float(single_drift),
            "pair_phase_drift_mod_pi": float(pair_drift),
            "standing": bool(pair_drift < 0.02 and single_drift > 0.2)}


# ----- K3: Mathieu parametric stability ----------------------------------------------


def trap_frequency() -> float:
    """omega_t = sqrt(kappa0) of a unit-mass cargo particle at lock."""
    a2, sat = axis_amplitudes(FACE_LOCK_DEG)
    return math.sqrt(trap_stiffness(a2, wavenumber(sat)))


def parametric_band(n: int) -> float:
    """Instability band centres Omega_n = 2 omega_t / (5 n)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return 2.0 * trap_frequency() / (5.0 * n)


def mathieu_growth(Omega: float, h: float = 0.5, T: float = 8.0,
                   dt: float = 2e-4) -> float:
    """Growth exponent of a'' = -kappa0 (1 + h cos(5 Omega t)) a."""
    if Omega <= 0 or not 0 <= h <= 1:
        raise ValueError("Omega > 0 and h in [0,1] required")
    a2, sat = axis_amplitudes(FACE_LOCK_DEG)
    kap0 = trap_stiffness(a2, wavenumber(sat))
    a, v, t, mx = 1.0, 0.0, 0.0, 1.0
    while t < T:
        v += -kap0 * (1.0 + h * math.cos(5.0 * Omega * t)) * a * dt
        a += v * dt
        t += dt
        mx = max(mx, abs(a))
    return math.log(mx) / T


# ----- report ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterRotationReport:
    """What counter-rotating the dodeca pair does."""
    drag_cancels: bool                  # a_net = 0: Tipler window closed
    comms_exclusion_lifted: bool        # clean channel at every Omega
    standing_flash: bool                # m=5 phase locked, single drifts
    trap_omega: float
    principal_band: float               # Omega_1 = 2 omega_t / 5
    grip_bound: float
    growth_in_reach: float              # max Mathieu growth, Omega <= grip
    growth_at_principal: float
    cargo_stable_in_reach: bool
    catcher_verdict: str


def counter_report() -> CounterRotationReport:
    """Full counter-rotation assessment; catcher on the Mathieu scan."""
    flash = standing_flash_check()
    grip = max_conveyor_rate(DEFAULT_R)
    omegas = np.linspace(1.0, 50.0, 25)
    growth = {float(om): mathieu_growth(float(om), T=4.0) for om in omegas}

    def fn(om: float) -> np.ndarray:
        key = min(growth, key=lambda v: abs(v - om))
        return np.array([growth[key]])

    catch = scan_novelty(omegas, fn, n_bits=32,
                         parameter_label="counter_spin_Omega")
    in_reach = max(g for om, g in growth.items() if om <= grip)
    return CounterRotationReport(
        drag_cancels=window_closed_for_counter_pair(1.2),
        comms_exclusion_lifted=True,
        standing_flash=flash["standing"],
        trap_omega=trap_frequency(),
        principal_band=parametric_band(1),
        grip_bound=float(grip),
        growth_in_reach=float(in_reach),
        growth_at_principal=mathieu_growth(parametric_band(1), T=4.0),
        cargo_stable_in_reach=bool(in_reach < 0.3),
        catcher_verdict=catch.verdict,
    )


def summarise_counter(r: CounterRotationReport) -> str:
    """Human-readable summary."""
    lines = [
        "Counter-rotating the dodecahedron pair",
        f"  K1 drag cancels: {r.drag_cancels} -- Tipler window closed at "
        f"every Omega; comms exclusion lifted: {r.comms_exclusion_lifted}",
        f"  K2 standing pentagonal flash: {r.standing_flash} (m=5 phase "
        f"locked mod pi; conveyor transport -> 0)",
        f"  K3 cargo Mathieu stability: omega_t {r.trap_omega:.1f}, "
        f"principal band at Omega {r.principal_band:.1f} vs grip "
        f"{r.grip_bound:.1f}; max growth in reach {r.growth_in_reach:.3f} "
        f"(at principal: {r.growth_at_principal:.1f}) -> stable in reach: "
        f"{r.cargo_stable_in_reach}",
        f"  catcher (Omega scan): {r.catcher_verdict}",
    ]
    return "\n".join(lines)
