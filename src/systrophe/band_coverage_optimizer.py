"""Nested-cylinder CTC-band coverage optimizer.

Purpose
-------
The Knopp-Drive bubble-shell exotic matter (the Ford-Roman quantum-interest
*principal*, ~Jupiter mass-energy per metre-bubble) is paid only on the
OUT-OF-BAND fraction of a travel route -- the part of the worldline that lies
*outside* the CTC bands of the rotating-cylinder source, where the geometry
does not supply the cone-tilt for free (see ``knopp_ratchet`` and
``FINDINGS_KNOPP_RATCHET.md``). Nesting concentric supercritical cylinders
makes the union of their CTC bands tile more of the route, shrinking that
paid fraction -- but the bands interfere, so they can *cancel* as readily as
they add (the uniform-phase comb extinguishes them entirely). Choosing the
phase offsets is therefore a real optimization problem.

This module optimizes the phase offsets of an N-cylinder ``SystropheArray`` to
MINIMIZE the out-of-band fraction of a route.

Method
------
Amplitude-modulation multi-pass culling, ported from the ResonantQ
optimization paradigm: score each candidate offset-vector by its "energy"
E = out_of_band_fraction, reweight the population by amplitudes
a = exp(-alpha * E / E_max), keep the elite, and resample around it with a
perturbation that shrinks each pass. This is a black-box optimizer over the
phase torus [0, 2pi)^N; it uses only the verified ``array.ctc_bands`` physics.

HONEST SCOPE
------------
This reduces the *out-of-band (paid) fraction* -- an ENGINEERING-reducible
term that relocates the per-journey cost toward reusable infrastructure. It
does NOT and cannot reduce the irreducible Ford-Roman principal, which is a
quantum-inequality conservation bound. No optimizer overcomes a conservation
law. The headline output is "fraction of route still paid", not "energy
created".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .array import SystropheArray
from .novelty_catcher import scan_novelty
from .vanstockum import VanStockumInterior


def _bands_measure(bands: list[tuple[float, float]]) -> float:
    return float(sum(hi - lo for lo, hi in bands))


def covered_fraction(
    bands: list[tuple[float, float]], r_min: float, r_max: float
) -> float:
    """Fraction of [r_min, r_max] covered by the (possibly overlapping) bands."""
    if r_max <= r_min:
        return 0.0
    clipped = [
        (max(lo, r_min), min(hi, r_max))
        for lo, hi in bands
        if hi > r_min and lo < r_max
    ]
    # Merge overlaps so coverage is not double-counted.
    if not clipped:
        return 0.0
    clipped.sort()
    merged = [list(clipped[0])]
    for lo, hi in clipped[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    total = sum(hi - lo for lo, hi in merged)
    return float(total / (r_max - r_min))


def out_of_band_fraction(
    radii: Sequence[float],
    offsets: Sequence[float],
    r_min: float = 1.05,
    r_max: float = 60.0,
    a: float = 1.0,
    n_grid: int = 2001,
) -> float:
    """Fraction of the route that lies OUTSIDE every CTC band.

    Each cylinder is supercriticality-matched at ``a = omega*R`` (default 1.0)
    by setting ``omega = a / R``. Returns 1 - covered_fraction in [0, 1].
    """
    cyls = [VanStockumInterior(omega=a / float(R), R=float(R)) for R in radii]
    arr = SystropheArray.from_cylinders(cyls, offsets=[float(o) for o in offsets])
    bands = arr.ctc_bands(r_min=r_min, r_max=r_max, n_grid=n_grid)
    return float(1.0 - covered_fraction(bands, r_min, r_max))


@dataclass(frozen=True)
class CoverageOptResult:
    radii: tuple[float, ...]
    route: tuple[float, float]
    best_offsets: tuple[float, ...]
    best_out_of_band: float
    aligned_out_of_band: float          # offsets all 0 (naive nesting)
    single_cylinder_out_of_band: float  # innermost cylinder alone
    uniform_comb_out_of_band: float     # phase comb (extinction control)
    history: tuple[float, ...]          # best-so-far per pass
    novelty_verdict: str
    novelty_n_sharp: int


def optimize_offsets(
    radii: Sequence[float] = (1.0, 3.0, 9.0, 27.0),
    r_min: float = 1.05,
    r_max: float = 60.0,
    a: float = 1.0,
    population: int = 48,
    n_pass: int = 8,
    alpha: float = 6.0,
    n_grid: int = 2001,
    seed: int = 0,
) -> CoverageOptResult:
    """Optimize phase offsets to minimize the out-of-band (paid) fraction.

    Amplitude-modulation culling (ResonantQ paradigm). Deterministic given
    ``seed``. The aligned, single-cylinder, and uniform-comb baselines are
    reported alongside so the gain is auditable.
    """
    radii = tuple(float(R) for R in radii)
    n = len(radii)
    rng = np.random.default_rng(seed)

    def E(offsets) -> float:
        return out_of_band_fraction(radii, offsets, r_min, r_max, a, n_grid)

    # Baselines -----------------------------------------------------------
    aligned = E([0.0] * n)
    single = out_of_band_fraction([radii[0]], [0.0], r_min, r_max, a, n_grid)
    comb_offsets = [2.0 * np.pi * i / n for i in range(n)]
    comb = E(comb_offsets)

    # Initial population: include the aligned config, rest random.
    pop = rng.uniform(0.0, 2.0 * np.pi, size=(population, n))
    pop[0, :] = 0.0
    energies = np.array([E(o) for o in pop])

    best_idx = int(np.argmin(energies))
    best_off = pop[best_idx].copy()
    best_E = float(energies[best_idx])
    history = [best_E]

    for p in range(n_pass):
        e_max = float(np.max(energies)) + 1e-12
        amps = np.exp(-alpha * energies / e_max)
        # Elite = top fraction by amplitude (lowest energy).
        n_elite = max(2, population // 4)
        elite_idx = np.argsort(-amps)[:n_elite]
        elite = pop[elite_idx]
        mean = elite.mean(axis=0)
        # Perturbation shrinks each pass (multi-pass refinement).
        scale = (2.0 * np.pi) * 0.5 / (p + 1)
        new_pop = (mean[np.newaxis, :]
                   + rng.normal(0.0, scale, size=(population, n))) % (2.0 * np.pi)
        new_pop[0, :] = best_off  # always retain the incumbent
        pop = new_pop
        energies = np.array([E(o) for o in pop])
        idx = int(np.argmin(energies))
        if energies[idx] < best_E:
            best_E = float(energies[idx])
            best_off = pop[idx].copy()
        history.append(best_E)

    # Novelty catcher over the best-so-far descent curve (standing rule).
    hist = np.array(history)
    axis = np.arange(len(hist), dtype=float)

    def fn(t: float) -> np.ndarray:
        i = max(0, min(int(round(t)), len(hist) - 1))
        return np.array([hist[i]])

    nov = scan_novelty(axis, fn, n_bits=32, parameter_label="pass")

    return CoverageOptResult(
        radii=radii,
        route=(float(r_min), float(r_max)),
        best_offsets=tuple(float(o) for o in best_off),
        best_out_of_band=best_E,
        aligned_out_of_band=float(aligned),
        single_cylinder_out_of_band=float(single),
        uniform_comb_out_of_band=float(comb),
        history=tuple(float(h) for h in history),
        novelty_verdict=nov.verdict,
        novelty_n_sharp=len(nov.sharp_features),
    )


def coverage_scaling(
    max_cylinders: int = 8,
    base_ratio: float = 3.0,
    r_min: float = 1.05,
    r_max: float = 60.0,
    seed: int = 1,
    n_pass: int = 6,
) -> dict:
    """Out-of-band fraction vs number of nested cylinders (optimized phases).

    HONEST FINDING (measured): the coverage win SATURATES almost immediately
    and is dominated by *single-cylinder phase tuning*, not by nesting -- a
    single cylinder with an optimized phase already reaches the saturated
    coverage (~86% here), and adding cylinders does not improve it. This
    revises the earlier "nesting tiles the route" framing: the lever is the
    phase, and extra cylinders are redundant for coverage (they remain useful
    only for placing bands at specific radii a single cylinder cannot reach).
    """
    out = {}
    for n in range(1, max_cylinders + 1):
        radii = tuple(float(base_ratio ** i) for i in range(n))
        res = optimize_offsets(radii=radii, r_min=r_min, r_max=r_max,
                               seed=seed, n_pass=n_pass, population=40)
        out[n] = float(res.best_out_of_band)
    vals = list(out.values())
    return {
        "out_of_band_by_n": out,
        "saturates": bool(max(vals) - min(vals) < 0.05),
        "single_cylinder_optimized": out[1],
        "interpretation": "coverage win is single-cylinder phase tuning; "
                          "nesting saturates",
    }


def summarise_coverage(res: CoverageOptResult) -> str:
    """One-line summary of an optimization result."""
    return (
        f"BandCoverageOpt radii={res.radii}: out-of-band "
        f"single={res.single_cylinder_out_of_band:.3f} "
        f"aligned={res.aligned_out_of_band:.3f} "
        f"-> optimized={res.best_out_of_band:.3f} "
        f"(comb={res.uniform_comb_out_of_band:.3f}); novelty={res.novelty_verdict}"
    )
