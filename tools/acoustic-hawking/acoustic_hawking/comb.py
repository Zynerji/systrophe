"""Multi-horizon acoustic Hawking-temperature *comb* over the LP ladder.

`acoustic_metric` / `horizon.AnalogHorizonAnalyser` find and characterise
only the *first* acoustic horizon (the first zero of c^2 - v^2 = F).  But
the supercritical (Bonnor Case III) van Stockum exterior has a whole
**ladder** of acoustic horizons at the analytic F-zeros

    r_n = R * exp((n pi - gamma) / alpha),   n = 1, 2, 3, ...

with alpha = sqrt(4 a^2 - 1), gamma = pi - arctan(alpha), a = omega R.
Consecutive horizons sit at a fixed geometric ratio

    r_{n+1} / r_n = exp(pi / alpha)

which is the discrete-scale-invariance (DSI) ratio of the Tipler sinusoid.
Each horizon carries its own acoustic surface gravity and acoustic Hawking
temperature; this module loops the existing single-horizon
`acoustic_surface_gravity` / `acoustic_hawking_temperature` over *all* of
them and returns the per-horizon T_H **comb** plus a report.

Two exact, falsifiable comb facts (both verified in the tests):

1. **DSI ratio.**  Running `discrete_scale_invariance_test` on the horizon
   radius set recovers the geometric ratio exp(pi / alpha).  This only
   validates the *fluid mapping* -- the ratio is exp(pi / alpha) *by
   construction* from the analytic F-zeros, so a clean DSI recovery is a
   self-consistency check, not new physics.

2. **Flat T_H comb.**  Because F'(r_n) = (1/R) alpha (+/-1) / sin(gamma) at
   every zero (the sin term vanishes), the acoustic surface gravity
   kappa_n = alpha / (2 R |sin gamma|) is *the same at every horizon*, so
   the comb is flat:  T_H_n = alpha / (4 pi R |sin gamma|) for all n.

The address-space novelty catcher (`scan_novelty`) is run over an alpha
sweep on the comb's anchored log-radii (which the catcher hashes into a
fixed-window positional occupancy); it tracks the comb appearing and
densifying as alpha grows, flags the supercritical onset (a -> 1/2+, first
horizon appears) as a sharp feature, and the helper additionally reports
the onset index explicitly.

No draining-tank forward model, no "first measurable alpha" claim is made
here: this is the kinematic comb of the analytic LP exterior only.

References
----------
- W. G. Unruh, Phys. Rev. Lett. 46 (1981) 1351.
- F. J. Tipler, Phys. Rev. D 9 (1974) 2203 (oscillatory exterior).
- W. B. Bonnor, J. Phys. A 13 (1980) 2121 (Case III).
- D. Sornette, Phys. Rep. 297 (1998) 239 (discrete scale invariance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systrophe.analogs.acoustic_metric import (
    acoustic_hawking_temperature,
    acoustic_surface_gravity,
)
from systrophe.catchers.dsi_observables import discrete_scale_invariance_test
from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.geometry.vanstockum import VanStockumInterior


# ---------------------------------------------------------------------------
# Horizon-ladder enumeration
# ---------------------------------------------------------------------------


def lp_horizon_radii(vs: VanStockumInterior, r_max_over_R: float = 50.0) -> np.ndarray:
    """All acoustic / chronology horizons (F-zeros) in (R, r_max].

    The supercritical exterior F(r) = (r/R) sin(alpha u + gamma)/sin(gamma)
    (u = ln(r/R)) vanishes at u_n = (n pi - gamma)/alpha, n = 1, 2, ...
    Returns the closed-form ladder r_n = R exp(u_n) for every u_n in
    (0, ln(r_max_over_R)].  Empty array if the exterior is sub-/critical
    (no real alpha, hence no F-zeros).

    These are *exactly* the radii where c^2 - v^2 = F changes sign, i.e. the
    sign changes of the acoustic-cone factor -- all of them, not just the
    first (which is all `acoustic_horizon_radius` returns).
    """
    if not vs.is_supercritical():
        return np.array([], dtype=float)
    alpha = float(vs.alpha)
    gamma = float(np.pi - np.arctan(alpha))
    upper_u = float(np.log(r_max_over_R))
    radii = []
    n = 1
    while True:
        u_n = (n * np.pi - gamma) / alpha
        if u_n > upper_u:
            break
        if u_n > 0:
            radii.append(float(vs.R * np.exp(u_n)))
        n += 1
    return np.array(radii, dtype=float)


def geometric_ratio(vs: VanStockumInterior) -> float:
    """Analytic comb ratio r_{n+1}/r_n = exp(pi / alpha)."""
    return float(np.exp(np.pi / float(vs.alpha)))


# ---------------------------------------------------------------------------
# The comb itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcousticTHComb:
    """Per-horizon acoustic Hawking-temperature comb on the LP ladder."""

    omega: float
    R: float
    alpha: float
    horizon_radii: np.ndarray  # r_n
    surface_gravities: np.ndarray  # kappa_n = (1/2)|F'(r_n)|
    T_hawking: np.ndarray  # T_H_n = kappa_n / (2 pi)
    geometric_ratio: float  # exp(pi / alpha) (analytic)
    dsi_recovered_ratio: float  # ratio recovered from the radii by DSI test
    dsi_is_dsi: bool
    dsi_rms_log_dev: float
    comb_is_flat: bool  # T_H_n equal across n to < flat_rtol
    n_horizons: int

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AcousticTHComb(omega={self.omega}, R={self.R}, "
            f"n_horizons={self.n_horizons}, ratio={self.geometric_ratio:.4f}, "
            f"flat={self.comb_is_flat}, dsi={self.dsi_is_dsi})"
        )


def acoustic_T_H_comb(
    omega: float,
    R: float,
    r_max_over_R: float = 50.0,
    eps: float = 1e-5,
    flat_rtol: float = 1e-6,
) -> AcousticTHComb:
    """Loop surface-gravity / T_H over *all* LP horizons -> the T_H comb.

    Parameters
    ----------
    omega, R : van Stockum cylinder parameters (must be supercritical,
        a = omega R > 1/2, for the comb to exist).
    r_max_over_R : enumerate horizons out to this multiple of R.
    eps : finite-difference step for the acoustic surface gravity.
    flat_rtol : tolerance for the "flat comb" verdict (max relative spread
        of T_H_n about the mean).

    Returns
    -------
    AcousticTHComb

    Raises
    ------
    ValueError if the exterior is sub-/critical (no acoustic horizons) or
    if fewer than 3 horizons are found in the window (DSI test needs >= 3).
    """
    vs = VanStockumInterior(omega=float(omega), R=float(R))
    if not vs.is_supercritical():
        raise ValueError(
            "acoustic T_H comb is defined only for the supercritical exterior "
            f"a = omega*R > 1/2 (got a = {vs.a:.4f}); sub-/critical exteriors "
            "have no acoustic-horizon ladder."
        )
    radii = lp_horizon_radii(vs, r_max_over_R=r_max_over_R)
    if len(radii) < 3:
        raise ValueError(
            f"need >= 3 horizons in (R, {r_max_over_R} R] for the comb / DSI test; "
            f"found {len(radii)}. Increase r_max_over_R."
        )

    # Per-horizon acoustic surface gravity and Hawking temperature, by
    # looping the *existing* single-horizon helpers over every F-zero.
    kappas = np.array(
        [acoustic_surface_gravity(vs, float(r), eps=eps) for r in radii],
        dtype=float,
    )
    T_H = np.array(
        [acoustic_hawking_temperature(vs, float(r), eps=eps) for r in radii],
        dtype=float,
    )

    # DSI on the radius set (reuse the catcher). Centre a fine candidate grid
    # on the analytic ratio so the recovered ratio is not limited by the
    # default coarse [1.05, 10] scan.
    ratio_analytic = geometric_ratio(vs)
    candidates = np.linspace(0.5 * ratio_analytic, 1.5 * ratio_analytic, 4001)
    dsi = discrete_scale_invariance_test(radii, candidate_ratios=candidates)

    # Flat-comb test.
    mean_T = float(np.mean(T_H))
    spread = float(np.max(np.abs(T_H - mean_T)) / max(abs(mean_T), 1e-30))
    flat = bool(spread < flat_rtol)

    return AcousticTHComb(
        omega=float(vs.omega),
        R=float(vs.R),
        alpha=float(vs.alpha),
        horizon_radii=radii,
        surface_gravities=kappas,
        T_hawking=T_H,
        geometric_ratio=float(ratio_analytic),
        dsi_recovered_ratio=float(dsi["best_ratio"]),
        dsi_is_dsi=bool(dsi["is_dsi"]),
        dsi_rms_log_dev=float(dsi["rms_log_dev"]),
        comb_is_flat=flat,
        n_horizons=int(len(radii)),
    )


# ---------------------------------------------------------------------------
# Novelty-catcher: comb across an alpha (a = omega R) sweep
# ---------------------------------------------------------------------------


def comb_logradii_with_anchors(
    a: float,
    R: float,
    r_max_over_R: float,
) -> np.ndarray:
    """Horizon-comb log-radii plus the two fixed window anchors.

    Returns ``[log(R), log(r_max_over_R * R), log(r_1), log(r_2), ...]`` --
    the comb's horizon log-radii in (R, r_max_over_R * R] with the two
    window endpoints prepended.  Sub-/critical (no horizons) -> just the two
    anchors.

    The anchors pin a *global* value range so that, when the novelty
    catcher hashes each output independently with `real_array_to_address`
    (which bins by [v_min, v_max]), every sweep point shares the same
    log-radius window.  The resulting address is then the comb's fixed-window
    *positional* occupancy: empty combs collapse to the same all-anchor
    address (zero Hamming step), and the supercritical onset is a genuine,
    position-resolved jump that the catcher can flag.
    """
    log_lo = float(np.log(R))
    log_hi = float(np.log(r_max_over_R * R))
    out = [log_lo, log_hi]
    vs = VanStockumInterior(omega=float(a) / float(R), R=float(R))
    if not vs.is_supercritical():
        return np.array(out, dtype=float)
    alpha = float(vs.alpha)
    gamma = float(np.pi - np.arctan(alpha))
    log_win = float(np.log(r_max_over_R))
    n = 1
    while True:
        u_n = (n * np.pi - gamma) / alpha
        if u_n > log_win:
            break
        if u_n > 0:
            out.append(log_lo + u_n)
        n += 1
    return np.array(out, dtype=float)


def comb_window_horizon_count(
    a: float,
    R: float,
    r_max_over_R: float,
) -> int:
    """Number of acoustic horizons in (R, r_max_over_R * R] at rotation a."""
    vs = VanStockumInterior(omega=float(a) / float(R), R=float(R))
    if not vs.is_supercritical():
        return 0
    return int(len(lp_horizon_radii(vs, r_max_over_R=r_max_over_R)))


@dataclass(frozen=True)
class CombNoveltySweep:
    """Novelty-catcher result for the comb across an alpha sweep."""

    a_values: np.ndarray  # the a = omega R sweep
    n_horizons: np.ndarray  # comb size in the window at each a
    onset_index: int | None  # first index with a horizon (supercritical onset)
    onset_a: float | None  # a at that index
    catcher_verdict: str
    n_sharp_features: int
    sharp_features: list
    lambda_2_at_radius: dict


def scan_comb_novelty(
    a_values: np.ndarray,
    R: float = 1.0,
    n_bits: int = 16,
    r_max_over_R: float = 200.0,
) -> CombNoveltySweep:
    """Run the address-space novelty catcher over the comb vs a = omega R.

    For each `a`, the catcher hashes the comb's anchored log-radii
    (`comb_logradii_with_anchors`) into a fixed-window positional occupancy
    address, so `scan_novelty` tracks the comb appearing (supercritical
    onset) and densifying.  The supercritical onset is *additionally*
    reported explicitly (the first non-empty comb), independent of the
    catcher's sharp-feature heuristic.

    `data_adaptive=False` so identical sub-critical (anchors-only) combs
    hash to the identical address (zero Hamming step), making the onset the
    genuine first non-zero step.
    """
    a_arr = np.asarray(a_values, dtype=float)

    def output_fn(a: float) -> np.ndarray:
        return comb_logradii_with_anchors(a, R=R, r_max_over_R=r_max_over_R)

    result = scan_novelty(
        a_arr,
        output_fn,
        n_bits=n_bits,
        parameter_label="a = omega * R",
        data_adaptive=False,
    )

    # Comb size and explicit supercritical-onset detection.
    n_h = np.array(
        [comb_window_horizon_count(a, R=R, r_max_over_R=r_max_over_R)
         for a in a_arr],
        dtype=int,
    )
    nz = np.nonzero(n_h > 0)[0]
    onset_index = int(nz[0]) if len(nz) else None
    onset_a = float(a_arr[onset_index]) if onset_index is not None else None

    return CombNoveltySweep(
        a_values=a_arr,
        n_horizons=n_h,
        onset_index=onset_index,
        onset_a=onset_a,
        catcher_verdict=result.verdict,
        n_sharp_features=len(result.sharp_features),
        sharp_features=list(result.sharp_features),
        lambda_2_at_radius=dict(result.lambda_2_at_radius),
    )
