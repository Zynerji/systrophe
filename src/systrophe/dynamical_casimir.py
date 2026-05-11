"""Dynamical Casimir effect: photon flux from a modulated cavity wall.

Validates speculative item I.5 from `docs/INTERPRETATIONS.md`:
"DCE photon flux at off-resonance."

For a cavity with one wall moving as

    d(t) = d_0 (1 + epsilon sin(Omega t))

with small modulation amplitude epsilon << 1 and drive frequency
Omega, the Bogoliubov transformation between in and out vacua
produces a photon pair flux. On-resonance (Omega = 2 omega_n for
a cavity mode of frequency omega_n = n pi / d_0):

    <N_n>_resonant = sinh^2(epsilon omega_n t / 4),

growing exponentially in time. Off-resonance, the flux is
exponentially suppressed by the cavity Q-factor

    <N_n>_off = epsilon^2 / (Q^2 (Omega/Omega_n - 1)^2)

(perturbative; valid for |Omega/Omega_n - 1| > 1/Q).

This module implements both regimes and provides:

- `on_resonance_photon_number(epsilon, omega_n, t)`: textbook DCE flux
- `off_resonance_photon_number(epsilon, Omega, omega_n, Q)`: detuned
  perturbative result
- `cavity_mode_frequency(n, d_0)`: standard 1D-cavity dispersion
- `dce_flux_estimate(...)`: full pipeline picking the right regime

The off-resonance behaviour is quantitatively predictable here given
(Q, epsilon, detuning). The earlier speculative claim that the flux
was negligible at delta = pi was based on an unspecified cavity Q;
that claim is now testable for any given Q.

References
----------
- V. V. Dodonov, "Current status of the dynamical Casimir effect",
  Phys. Scr. 82 (2010) 038105.
- C. M. Wilson et al., "Observation of the dynamical Casimir effect
  in a superconducting circuit", Nature 479 (2011) 376.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np


def cavity_mode_frequency(n: int, d_0: float, c: float = 1.0) -> float:
    """nth cavity-mode frequency: omega_n = n pi c / d_0."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if d_0 <= 0:
        raise ValueError("d_0 must be positive")
    return float(n * pi * c / d_0)


def resonance_frequency_for_mode(omega_n: float) -> float:
    """Resonance drive frequency: Omega_res = 2 omega_n (parametric DCE)."""
    return 2.0 * omega_n


def on_resonance_photon_number(
    epsilon: float, omega_n: float, t: float
) -> float:
    """<N_n>_resonant = sinh^2(epsilon omega_n t / 4).

    Grows exponentially in t for small epsilon.
    """
    if epsilon < 0 or omega_n < 0 or t < 0:
        raise ValueError("inputs must be non-negative")
    arg = epsilon * omega_n * t / 4.0
    return float(np.sinh(arg) ** 2)


def off_resonance_photon_number(
    epsilon: float, Omega: float, omega_n: float, Q: float
) -> float:
    """<N_n>_off = epsilon^2 / (Q^2 (Omega/Omega_n - 1)^2).

    Perturbative; valid for |Omega/Omega_n - 1| > 1/Q.
    """
    if Q <= 0:
        raise ValueError("Q must be positive")
    if omega_n <= 0:
        raise ValueError("omega_n must be positive")
    Omega_n = resonance_frequency_for_mode(omega_n)
    delta = abs(Omega / Omega_n - 1.0)
    if delta < 1e-12:
        # Effectively on resonance; perturbative formula diverges
        return float("inf")
    return float(epsilon ** 2 / (Q ** 2 * delta ** 2))


def linewidth_at_resonance(omega_n: float, Q: float) -> float:
    """FWHM around resonance: Gamma = Omega_n / Q."""
    return resonance_frequency_for_mode(omega_n) / Q


@dataclass(frozen=True)
class DCEFluxResult:
    """Result of a DCE photon-flux calculation."""

    epsilon: float
    Omega: float
    omega_n: float
    d_0: float
    Q: float
    t: float
    Omega_resonance: float
    fractional_detuning: float
    on_resonance: bool
    photon_number: float
    regime: str  # "on_resonance" or "off_resonance"


def dce_flux_estimate(
    epsilon: float,
    Omega_drive: float,
    n: int = 1,
    d_0: float = 1.0,
    Q: float = 100.0,
    t: float = 1.0,
    c: float = 1.0,
) -> DCEFluxResult:
    """End-to-end DCE photon-flux estimate for a 1D cavity.

    Picks the appropriate regime (on/off resonance) based on the
    detuning relative to the cavity linewidth Omega_n / Q.
    """
    omega_n = cavity_mode_frequency(n=n, d_0=d_0, c=c)
    Omega_res = resonance_frequency_for_mode(omega_n)
    detuning = abs(Omega_drive / Omega_res - 1.0)
    linewidth_fraction = 1.0 / Q
    on_res = detuning < linewidth_fraction
    if on_res:
        N = on_resonance_photon_number(epsilon, omega_n, t)
        regime = "on_resonance"
    else:
        N = off_resonance_photon_number(epsilon, Omega_drive, omega_n, Q)
        regime = "off_resonance"
    return DCEFluxResult(
        epsilon=epsilon,
        Omega=Omega_drive,
        omega_n=omega_n,
        d_0=d_0,
        Q=Q,
        t=t,
        Omega_resonance=Omega_res,
        fractional_detuning=detuning,
        on_resonance=on_res,
        photon_number=N,
        regime=regime,
    )


def dce_spectrum(
    epsilon: float,
    Omega_min: float,
    Omega_max: float,
    n_points: int = 200,
    n_mode: int = 1,
    d_0: float = 1.0,
    Q: float = 100.0,
    t: float = 1.0,
) -> dict:
    """Sweep Omega and return the resulting photon-number spectrum."""
    Omegas = np.linspace(Omega_min, Omega_max, n_points)
    Ns = np.zeros_like(Omegas)
    for i, Om in enumerate(Omegas):
        r = dce_flux_estimate(epsilon, float(Om), n=n_mode, d_0=d_0,
                                Q=Q, t=t)
        Ns[i] = r.photon_number
    return {"Omega_drive": Omegas, "photon_number": Ns,
            "Omega_resonance": resonance_frequency_for_mode(
                cavity_mode_frequency(n_mode, d_0))}
