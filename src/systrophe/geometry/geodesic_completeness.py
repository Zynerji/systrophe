"""Geodesic completeness diagnostic for the LP exterior.

Tests whether CTC regions in the rotating-cylinder spacetime are
*physically accessible* from outside, or whether they are topologically
isolated regions requiring extension across chronology horizons.

The key question: can a future-directed timelike geodesic from a
"regular" external observer (r >> all CTC bands) reach a target point
inside a CTC band? If yes, the CTC is genuinely physical; if not, it
is a coordinate artifact of the analytic continuation.

Tests provided:

- `is_orbit_timelike(vs, r)`: can a future-directed orbit exist at r?
  (Tests the timelike-Omega bounds.)
- `chronology_horizon_radii(vs, r_min, r_max)`: F = 0 surfaces.
- `radial_geodesic_reaches(vs, r_start, r_target)`: existence test for
  a radial geodesic from r_start reaching r_target without divergent
  proper time.
- `causal_diamond_extent(vs, r_obs)`: maximum r-range an observer at
  r_obs can reach via timelike geodesics.
- `geodesic_completeness_report(vs)`: full diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systrophe.geometry.geodesic import timelike_omega_bounds


@dataclass(frozen=True)
class GeodesicCompletenessReport:
    """Diagnostic for the LP exterior's geodesic accessibility."""

    n_ctc_bands: int
    ctc_band_ranges: list[tuple[float, float]]
    n_chronology_horizons: int
    horizon_radii: list[float]
    all_orbits_timelike: bool
    ctc_accessibility: list[bool]
    is_geodesically_complete: bool


def is_orbit_timelike(vs, r: float) -> bool:
    """True iff there exists a future-directed timelike circular orbit at r.

    Tests whether the (F, K, L) at r admit a real Omega satisfying
    F - 2 K Omega - L Omega^2 > 0.
    """
    F = float(vs.analytic_exterior_F(r))
    K = float(vs.analytic_exterior_K(r))
    L = float(vs.analytic_exterior_L(r))
    try:
        om_lo, om_hi = timelike_omega_bounds(F, K, L, r)
        return bool(om_lo < om_hi)
    except Exception:
        return False


def chronology_horizon_radii(vs, r_min: float = 1.05, r_max: float = 50.0,
                                 n_grid: int = 2001) -> list[float]:
    """Find F = 0 zero crossings: chronology horizons in [r_min, r_max]."""
    rs = np.linspace(r_min, r_max, n_grid)
    Fs = np.array([float(vs.analytic_exterior_F(r)) for r in rs])
    flips = np.where(np.diff(np.sign(Fs)) != 0)[0]
    horizons = []
    for i in flips:
        r1, r2 = rs[i], rs[i + 1]
        F1, F2 = Fs[i], Fs[i + 1]
        if abs(F2 - F1) < 1e-30:
            horizons.append(float(0.5 * (r1 + r2)))
        else:
            horizons.append(float(r1 - F1 * (r2 - r1) / (F2 - F1)))
    return horizons


def ctc_band_radii(vs, r_min: float = 1.05, r_max: float = 50.0,
                       n_grid: int = 2001) -> list[tuple[float, float]]:
    """Find (r_inner, r_outer) ranges where L(r) < 0 (CTC bands)."""
    rs = np.linspace(r_min, r_max, n_grid)
    Ls = np.array([float(vs.analytic_exterior_L(r)) for r in rs])
    in_ctc = Ls < 0
    bands = []
    if not in_ctc.any():
        return bands
    # Group consecutive Trues
    diff = np.diff(in_ctc.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if in_ctc[0]:
        starts = np.insert(starts, 0, 0)
    if in_ctc[-1]:
        ends = np.append(ends, len(in_ctc) - 1)
    for s, e in zip(starts, ends):
        bands.append((float(rs[s]), float(rs[e])))
    return bands


def radial_geodesic_reaches(
    vs, r_start: float, r_target: float, max_distance: float = 100.0,
) -> bool:
    """Heuristic test: can a timelike geodesic from r_start reach r_target?

    Test: are all radial positions between r_start and r_target
    capable of supporting a timelike orbit? If yes, by continuity a
    radial-coordinate geodesic connecting them exists.
    """
    if abs(r_start - r_target) > max_distance:
        return False
    n_test = 50
    rs = np.linspace(r_start, r_target, n_test) if r_start < r_target else np.linspace(r_target, r_start, n_test)
    for r in rs:
        if not is_orbit_timelike(vs, float(r)):
            return False
    return True


def ctc_accessibility(vs, r_observer: float | None = None,
                          r_max: float = 50.0) -> dict:
    """For each CTC band, test if it is reachable from a distant observer.

    r_observer defaults to r_max (asymptotic). Returns dict per band:
      {(r_in, r_out): is_accessible}.
    """
    if r_observer is None:
        r_observer = r_max
    bands = ctc_band_radii(vs, r_max=r_max)
    accessibility = {}
    for band in bands:
        r_target = 0.5 * (band[0] + band[1])  # band centre
        accessible = radial_geodesic_reaches(vs, r_observer, r_target)
        accessibility[band] = accessible
    return accessibility


def causal_diamond_extent(vs, r_obs: float, search_max: float = 50.0) -> dict:
    """Range of r-coordinates reachable from r_obs via timelike geodesics.

    Returns (r_min_reach, r_max_reach), the extreme r-coordinates that
    are connectable via timelike radial paths from r_obs.
    """
    # Outward: walk r upward until we hit a non-timelike-orbit region
    n_test = 200
    rs_out = np.linspace(r_obs, search_max, n_test)
    r_max_reach = r_obs
    for r in rs_out:
        if is_orbit_timelike(vs, float(r)):
            r_max_reach = float(r)
        else:
            break
    # Inward
    rs_in = np.linspace(r_obs, 1.05, n_test)
    r_min_reach = r_obs
    for r in rs_in:
        if is_orbit_timelike(vs, float(r)):
            r_min_reach = float(r)
        else:
            break
    return {"r_min_reach": r_min_reach, "r_max_reach": r_max_reach,
            "r_observer": r_obs}


def geodesic_completeness_report(vs, r_min: float = 1.05, r_max: float = 50.0,
                                       n_grid: int = 2001) -> GeodesicCompletenessReport:
    """Complete diagnostic of geodesic accessibility on LP exterior."""
    horizons = chronology_horizon_radii(vs, r_min, r_max, n_grid)
    bands = ctc_band_radii(vs, r_min, r_max, n_grid)
    # Check accessibility for each CTC band
    accessibility = ctc_accessibility(vs, r_observer=r_max, r_max=r_max)
    accessibility_list = [accessibility.get(b, False) for b in bands]

    # All-orbits-timelike check across r range
    test_rs = np.linspace(r_min, r_max, 200)
    all_timelike = all(is_orbit_timelike(vs, float(r)) for r in test_rs)

    # Geodesically complete: all CTC bands accessible from asymptotic observer
    is_complete = all(accessibility_list) if accessibility_list else True

    return GeodesicCompletenessReport(
        n_ctc_bands=len(bands),
        ctc_band_ranges=bands,
        n_chronology_horizons=len(horizons),
        horizon_radii=horizons,
        all_orbits_timelike=all_timelike,
        ctc_accessibility=accessibility_list,
        is_geodesically_complete=is_complete,
    )
