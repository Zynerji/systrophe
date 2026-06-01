"""Rotational superradiance in van Stockum / Tipler CTC bands.

The acoustic identification ``c^2 - v^2 = F`` (see ``analogs.acoustic_metric``)
means an ``F < 0`` band of the supercritical van Stockum exterior is a
**rotational ergoregion**: ``\\partial_t`` is spacelike, co-rotating timelike
orbits can carry *negative* Killing energy (this is exactly what
``penrose_extraction`` already computes), and a probe wave scattered off the
frame-dragging potential there should be **amplified** -- rotational
superradiance, the wave analogue of the Penrose process.

This module turns that qualitative statement into a *measurable amplitude
observable* via the standard quantum-optics dictionary for superradiant /
parametric amplification:

* A superradiant scatterer is a **two-mode squeezer**.  Acting on the vacuum
  it spontaneously creates a signal/idler **pair** with probability set by the
  squeezing parameter -- the amplitude analogue of spontaneous Hawking/Unruh
  emission.  A passive (``F > 0``, sub-threshold) scatterer is a beam splitter:
  it creates **no** quanta from the vacuum.
* Truncating each bosonic mode to its lowest two levels ``{|0>, |1>}`` (the
  occupation encoding used for small gate-model devices), the two-mode squeezer
  ``exp(theta (a^dag b^dag - a b))`` restricted to ``span{|00>, |11>}`` acts as
  ``|00> -> cos(theta)|00> + sin(theta)|11>``.  The **pair-creation
  probability** is therefore::

      P_pair(theta) = sin^2(theta)        (truncated Bogoliubov |beta|^2)

  which is ``> 0`` in an ergoregion band and ``== 0`` in a passive band.

The squeezing angle is set by the band physics through the Penrose extraction
efficiency ``eta = |E_min| / E_max`` (``penrose_extraction``):

    theta(r) = arctan(eta(r)),   eta(r) > 0  <=>  F(r) < 0  (ergoregion)

so ``P_pair`` is a monotone proxy for the superradiant gain that is **exactly
zero outside the ergoregion**.  ``theta = arctan(eta)`` is a modelling
dictionary, not a first-principles Bogoliubov derivation; the falsifiable
content is (i) ``P_pair`` reproduced to machine precision by the truncated
two-mode-squeezing circuit, (ii) ``P_pair > 0`` *iff* ``F < 0``, localised at
the ergosurface, and (iii) survival of an ``F``-sign-label surrogate.

References
----------
- Ya. B. Zel'dovich, JETP Lett. 14 (1971) 180 (rotational superradiance).
- W. G. Unruh, Phys. Rev. D 14 (1976) 870.
- R. Penrose, Rev. Nuovo Cim. 1 (1969) 252.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systrophe.qftcs.penrose_extraction import negative_energy_in_band
from systrophe.analogs.synchrotron_analog import orbital_frequency


@dataclass(frozen=True)
class SuperradianceBand:
    """Superradiance descriptor at a single radius."""

    r: float
    F: float
    Omega: float          # co-rotating orbital frequency
    energy_min: float     # most-negative Killing energy on a timelike orbit
    energy_max: float     # most-positive Killing energy
    efficiency: float     # eta = |E_min| / E_max  (0 outside the ergoregion)
    is_ergoregion: bool   # F < 0
    squeezing_angle: float  # theta = arctan(eta)
    pair_probability: float  # P_pair = sin^2(theta) = eta^2 / (1 + eta^2)


def squeezing_angle(efficiency: float) -> float:
    """Truncated two-mode-squeezing angle from the Penrose efficiency.

    ``theta = arctan(eta)`` maps ``eta in [0, inf)`` to ``[0, pi/2)`` so the
    pair-creation probability ``sin^2(theta) = eta^2 / (1 + eta^2)`` stays in
    ``[0, 1)`` and is exactly ``0`` when ``eta == 0`` (passive band).
    """
    eta = max(0.0, float(efficiency))
    return float(np.arctan(eta))


def predicted_pair_probability(theta: float) -> float:
    """Truncated Bogoliubov pair-creation probability ``P_pair = sin^2(theta)``.

    This is the value the noiseless two-mode-squeezing circuit
    ``Ry(2 theta) on q0; CX q0->q1`` must reproduce when measured in the
    computational basis: ``P(|11>) = sin^2(theta)``.
    """
    return float(np.sin(float(theta)) ** 2)


def band_superradiance_at(vs, r: float, n_omega: int = 50) -> SuperradianceBand:
    """Superradiance descriptor at radius ``r`` of cylinder ``vs``."""
    F = float(vs.analytic_exterior_F(r))
    E_min, E_max = negative_energy_in_band(vs, r, n_omega=n_omega)
    try:
        Omega = float(orbital_frequency(vs, r))
    except Exception:
        Omega = float("nan")

    if (not np.isfinite(E_min)) or (not np.isfinite(E_max)) or E_max <= 0:
        eta = 0.0
    elif E_min < 0:
        eta = abs(E_min) / E_max
    else:
        eta = 0.0

    theta = squeezing_angle(eta)
    return SuperradianceBand(
        r=float(r),
        F=F,
        Omega=Omega,
        energy_min=float(E_min),
        energy_max=float(E_max),
        efficiency=float(eta),
        is_ergoregion=bool(F < 0.0),
        squeezing_angle=theta,
        pair_probability=predicted_pair_probability(theta),
    )


def superradiance_scan(vs, radii, n_omega: int = 50) -> list[SuperradianceBand]:
    """Superradiance descriptors over a radius array."""
    return [band_superradiance_at(vs, float(r), n_omega=n_omega) for r in radii]


def amplification_follows_F_sign(bands: list[SuperradianceBand]) -> dict:
    """Consistency check: spontaneous pair creation occurs only where F < 0.

    Returns a dict with the per-band agreement and the worst violation, used
    by the experiment's surrogate test (true F-sign labels must explain
    ``pair_probability > 0`` far better than permuted labels).
    """
    agree = [
        (b.pair_probability > 1e-12) == b.is_ergoregion
        for b in bands
    ]
    ergo = [b.pair_probability for b in bands if b.is_ergoregion]
    passive = [b.pair_probability for b in bands if not b.is_ergoregion]
    return {
        "n_bands": len(bands),
        "n_agree": int(sum(agree)),
        "all_agree": bool(all(agree)),
        "mean_pair_ergoregion": float(np.mean(ergo)) if ergo else 0.0,
        "max_pair_passive": float(np.max(passive)) if passive else 0.0,
    }
