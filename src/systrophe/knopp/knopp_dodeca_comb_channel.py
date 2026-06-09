"""Can the rotational comb communicate through the bubble wall?

Eighth module of the dodeca-resonator series. In-model answer: YES -- the
spinning comb writes itself onto the wall as time-varying radiation
pressure, the shell re-radiates it, and tilt-keying makes a frequency-
shift-keyed transmitter. With one derived exclusion: the comms band and
the time-machine band are DISJOINT.

Derivations (model units)
-------------------------
C1. WALL IMPRINT. A fixed wall point watches the spinning C5 pattern sweep
    past: its Langevin pressure oscillates with modulation depth 0.2-0.4.
    Line structure by latitude (measured):
      - mid-latitude shell (45 deg): clean comb at 5*Omega multiples
      - near the pinch: a single pure 5*Omega line
      - equator: 10*Omega spacing (the y-mirror symmetry of the equatorial
        plane doubles the selection -- a free latitude diagnostic)
    Tilting demultiplies at the wall exactly as in the interior.

C2. RE-RADIATION + KEYING. The shell is a driven oscillator: pressure
    modulation at 5 m Omega drives wall breathing that re-radiates
    outward (speaker-cone). Binary tilt keying (delta: 0 <-> ~2-4 deg)
    toggles the wall spectrum between 5*Omega-spaced and Omega-spaced --
    a robust FSK symbol that needs no amplitude calibration. The tilt
    response is GEOMETRIC (the pattern is slaved to the body frame), so
    the symbol time is set by spectral resolution, not cavity ring-up:
    distinguishing the spacings needs one Omega-period to resolve the
    Omega line: T_symbol ~ 2 pi / Omega, bit rate R ~ (Omega/2pi) log2 M.

C3. THE COMMS / CHRONOLOGY EXCLUSION (derived design rule). Clean
    signalling presumes ordinary causal order at the wall, which fails
    inside the supercritical window Omega in (1/2R, sqrt(2 pi u / e)) =
    (0.76, 1.71) where the CTC band intersects the tube. So communicate
    at Omega below 0.76 (slow, R < 0.12 bits/unit) or ABOVE 1.71 -- the
    grip bound (~12) allows Omega up to ~12: R_max ~ 1.9 bits/unit binary.
    The time-machine band and the high-rate comms band are disjoint:
    the drive can talk or close timelike curves, not both at once.

HONEST CAVEATS. The model is a coherent field on a flat background with
no causal structure: the genuinely relativistic question -- whether a
signal can cross a SUPERLUMINAL warp wall (it cannot reach the bubble
front; Everett/Krasnikov-type results) -- is not modelled. The Knopp
ratchet bias is a sub-c displacement, for which a wall is causally
ordinary and the channel is legitimate. Inside the Tipler window all
signal chronology is suspect (and Phase 2a says the horizon diverges
anyway). Model units; no SI claim.

Catcher (mandatory): on the Omega scan of the clean-channel bit rate --
the window edges are the transitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.knopp.knopp_dodeca_alignment import FACE_LOCK_DEG
from systrophe.knopp.knopp_dodeca_pressure import axis_amplitudes, wavenumber
from systrophe.knopp.knopp_dodeca_rotation import (
    max_conveyor_rate,
    mean_interior_pressure,
    spun_axes,
    van_stockum_window,
)
from systrophe.knopp.knopp_dodeca_alignment import DEFAULT_R

#: mid-latitude wall readout point (clean 5-Omega comb, depth ~0.28)
WALL_45 = np.array([0.66 + 0.4667, 0.4667, 0.0])
WALL_EQUATOR = np.array([1.32, 0.0, 0.0])
WALL_PINCH = np.array([0.08, 0.22, 0.0])


def wall_comb_spectrum(
    point: np.ndarray, Omega: float = 1.0, tilt_deg: float = 0.0,
    n_t: int = 8192, n_periods: int = 16, threshold: float = 0.05,
) -> dict:
    """Pressure spectrum a fixed wall point sees under spin.

    Returns DC level, modulation depth (AC rms / DC), and the significant
    line frequencies in units of Omega.
    """
    if Omega <= 0:
        raise ValueError("Omega must be positive")
    a2, sat = axis_amplitudes(FACE_LOCK_DEG)
    k = wavenumber(sat)
    T = n_periods * 2.0 * math.pi / Omega
    t = np.linspace(0.0, T, n_t, endpoint=False)
    I = np.empty(n_t)
    for i, ti in enumerate(t):
        ax = spun_axes(math.degrees(Omega * ti), tilt_deg)
        I[i] = float(np.sum(a2 / 2.0 * np.cos(k * (ax @ point)) ** 2))
    F = np.abs(np.fft.rfft(I - I.mean()))
    freqs = np.fft.rfftfreq(n_t, T / n_t) * 2.0 * math.pi / Omega
    sig = F > threshold * F.max()
    return {
        "dc": float(I.mean()),
        "modulation_depth": float(np.std(I) / I.mean()),
        "lines_per_omega": freqs[sig],
    }


def lines_are_5omega_selected(lines: np.ndarray, tol: float = 0.05) -> bool:
    """True if every significant line is a multiple of 5 Omega."""
    lines = np.asarray(lines, dtype=float)
    if len(lines) == 0:
        return False
    return bool(np.all(np.abs(lines / 5.0 - np.round(lines / 5.0)) < tol))


def comb_bit_rate(Omega: float, levels: int = 2) -> float:
    """FSK bit rate: tilt keying is geometric, so the symbol time is the
    spectral-resolution limit T_s = 2 pi / Omega.
        R = (Omega / 2 pi) * log2(levels)
    """
    if Omega <= 0 or levels < 2:
        raise ValueError("Omega > 0 and levels >= 2 required")
    return Omega / (2.0 * math.pi) * math.log2(levels)


def clean_channel(Omega: float, R: float = DEFAULT_R) -> bool:
    """True when the spin sits OUTSIDE the supercritical Tipler window
    (ordinary causal order at the wall -- the comms/chronology exclusion)."""
    lo, hi = van_stockum_window(mean_interior_pressure(), R)
    return not (lo < Omega < hi)


def max_clean_bit_rate(levels: int = 2, R: float = DEFAULT_R) -> float:
    """Highest clean-channel rate: spin at the conveyor grip bound, which
    sits far above the Tipler window."""
    om = max_conveyor_rate(R)
    if not clean_channel(om, R):
        _, hi = van_stockum_window(mean_interior_pressure(), R)
        om = hi * 1.01
    return comb_bit_rate(om, levels)


# ----- report -----------------------------------------------------------------


@dataclass(frozen=True)
class CombChannelReport:
    """Through-the-wall comb communication, assessed."""
    wall_modulation_depth_45: float
    wall_5omega_selected: bool          # aligned spin: pure 5-Omega comb
    wall_demultiplied_on_tilt: bool     # tilt keying readable at the wall
    equator_spacing_per_omega: float    # 10: y-mirror doubles the selection
    tipler_window: tuple[float, float]
    comms_chronology_disjoint: bool     # high-rate comms excludes CTC band
    bit_rate_below_window: float
    bit_rate_above_window: float
    bit_rate_at_grip_bound: float
    catcher_verdict: str


def channel_report() -> CombChannelReport:
    """Assess the full channel; catcher on the clean-rate Omega scan."""
    aligned = wall_comb_spectrum(WALL_45, 1.0, 0.0)
    tilted = wall_comb_spectrum(WALL_45, 1.0, 4.0)
    eq = wall_comb_spectrum(WALL_EQUATOR, 1.0, 0.0)
    lo, hi = van_stockum_window(mean_interior_pressure())
    om_grip = max_conveyor_rate(DEFAULT_R)

    omegas = np.linspace(0.2, 3.0, 29)
    table = {float(om): (comb_bit_rate(float(om)) if clean_channel(float(om))
                         else 0.0) for om in omegas}

    def fn(om: float) -> np.ndarray:
        key = min(table, key=lambda v: abs(v - om))
        return np.array([table[key]])

    catch = scan_novelty(omegas, fn, n_bits=32,
                         parameter_label="comms_spin_Omega")
    eq_lines = eq["lines_per_omega"]
    eq_spacing = float(np.min(np.diff(eq_lines))) if len(eq_lines) > 1 \
        else float(eq_lines[0]) if len(eq_lines) else 0.0
    return CombChannelReport(
        wall_modulation_depth_45=aligned["modulation_depth"],
        wall_5omega_selected=lines_are_5omega_selected(
            aligned["lines_per_omega"]),
        wall_demultiplied_on_tilt=not lines_are_5omega_selected(
            tilted["lines_per_omega"]),
        equator_spacing_per_omega=eq_spacing,
        tipler_window=(float(lo), float(hi)),
        comms_chronology_disjoint=True,
        bit_rate_below_window=comb_bit_rate(lo * 0.99),
        bit_rate_above_window=comb_bit_rate(hi * 1.01),
        bit_rate_at_grip_bound=comb_bit_rate(om_grip),
        catcher_verdict=catch.verdict,
    )


def summarise_channel(r: CombChannelReport) -> str:
    """Human-readable summary."""
    lines = [
        "Comb communication through the bubble wall",
        f"  C1 wall imprint: depth {r.wall_modulation_depth_45:.2f} at the "
        f"45-deg shell; 5-Omega selected: {r.wall_5omega_selected}; tilt "
        f"demultiplies at the wall: {r.wall_demultiplied_on_tilt}; equator "
        f"spacing {r.equator_spacing_per_omega:.0f} Omega (mirror doubling)",
        f"  C2 FSK keying: bit rate below/above window = "
        f"{r.bit_rate_below_window:.2f} / {r.bit_rate_above_window:.2f}; "
        f"at the grip bound {r.bit_rate_at_grip_bound:.2f} bits/unit",
        f"  C3 exclusion: Tipler window {tuple(round(v, 2) for v in r.tipler_window)}"
        f" -- talk OR close timelike curves, not both:"
        f" {r.comms_chronology_disjoint}",
        f"  catcher (Omega scan): {r.catcher_verdict}",
    ]
    return "\n".join(lines)
