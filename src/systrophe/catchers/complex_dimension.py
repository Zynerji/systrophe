"""Complex-dimension spectrometer: ONE instrument for both DSI codepaths.

Discrete-scale invariance (DSI) advertises a *complex* fractal dimension

    D = Re(D) + i * Im(D),

where ``Re(D)`` is the leading power-law (box-counting) exponent and
``Im(D)`` is a log-periodic angular frequency: the observable carries an
oscillation that is periodic in some scaling variable ``u(x)``. The catch
is that *which* variable linearises the comb is structure-dependent:

  * Euler-product / renormalisation-group cascades (Cantor sets, crash
    precursors, Systrophe's Tipler sinusoid) oscillate in ``u = ln x``
    with an INTEGER comb ``omega = k * omega_fund``.

  * Voronoi / Hardy lattice-error terms (Dirichlet divisor ``Delta(x)``,
    Gauss circle ``P(x)``) oscillate in ``u = sqrt(x)`` with a
    SQRT-INTEGER comb ``omega = omega_fund * sqrt(n)`` (peaks at
    ``4 pi sqrt(n)`` resp. ``2 pi sqrt(n)``).

Historically these lived in two disjoint codepaths: the ln-x detector in
``catchers.dsi_observables`` (``lomb_scargle_log_periodicity`` +
``box_count_dimension_1d``) and an orphaned sqrt-x detector buried in
``examples/erdos/erdos_sqrtn_lattice_dsi.py``. If you point the ln-x
detector at a divisor error it (correctly, but uselessly) returns a null,
and vice versa. A user does not know a-priori which world the signal is
in.

``ComplexDimensionSpectrometer`` unifies them. For each candidate scaling
variable ``u in {ln x, sqrt x, x^{1/4}}`` it

  1. fits the power-law envelope (exponent ``Re(D)`` via a log|y| vs log x
     regression, the same machinery as ``growth_catcher.catch_growth``)
     and divides it out, leaving a centred oscillation;
  2. runs ``scipy.signal.lombscargle`` on ``(u, residual)``;
  3. scores comb-harmonicity as a CALIBRATED p-value: the strongest comb
     peak versus a per-transform AR(1) phase-preserving surrogate null
     (promoted here from the erdos example), so the score is comparable
     ACROSS transforms even though their abscissae have different units;
  4. fits the fundamental against BOTH an integer comb and a sqrt-integer
     comb so the divisor comb is not wrongly rejected.

The transform with the SMALLEST calibrated p-value is selected. The
spectrometer returns

    D = Re(D) + i * omega_fundamental ,

the selected variable, and the calibrated p-value. A pure power law with
no oscillation returns ``p > p_threshold`` against the surrogate ensemble
and is reported as ``has_comb = False`` (no false comb).

Novelty-catcher culture: every positive verdict is gated on the AR(1)
surrogate null. We never report a comb without the calibrated p-value that
falsifies the red-noise hypothesis.

References
----------
- D. Sornette, "Discrete-scale invariance and complex dimensions",
  Phys. Rep. 297 (1998) 239.
- G. H. Hardy / G. Voronoi lattice-point error expansions (the 4 pi vs
  2 pi sqrt-integer combs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.signal import find_peaks, lfilter

# --------------------------------------------------------------------------
# Fast vectorised normalised Lomb-Scargle periodogram.
#
# ``scipy.signal.lombscargle`` is O(n_freq * n_samples) per call and, for
# the grid sizes we need (~2000 freqs x ~thousands of samples), costs
# several seconds. The AR(1) surrogate null needs thousands of evaluations
# on the SAME abscissa ``u`` and frequency grid, so we precompute the
# (tau-shifted) cos/sin design matrices ONCE and evaluate each surrogate's
# power as two matrix-vector products. This reproduces scipy's
# ``normalize=True`` periodogram to ~3e-16 (regression-tested) while
# turning thousands of L-S calls into a handful of matmuls.
# --------------------------------------------------------------------------


class _LombScargleBasis:
    """Precomputed normalised Lomb-Scargle design for fixed ``u`` and freqs.

    Reproduces ``scipy.signal.lombscargle(u, y, freqs, normalize=True)``:
    for each angular frequency ``w`` a phase offset ``tau`` orthogonalises
    the cos/sin pair, and the normalised power is

        P(w) = [ (C.y)^2 / (C.C) + (S.y)^2 / (S.S) ] / (y.y),

    with ``y`` mean-subtracted (the caller centres the residual).
    """

    def __init__(self, u: np.ndarray, freqs: np.ndarray) -> None:
        u = np.asarray(u, dtype=float)
        freqs = np.asarray(freqs, dtype=float)
        self.freqs = freqs
        wt = 2.0 * freqs[:, None] * u[None, :]
        s2 = np.sin(wt).sum(axis=1)
        c2 = np.cos(wt).sum(axis=1)
        tau = 0.5 * np.arctan2(s2, c2) / freqs
        arg = freqs[:, None] * u[None, :] - (freqs * tau)[:, None]
        self._C = np.cos(arg)
        self._S = np.sin(arg)
        self._cc = (self._C * self._C).sum(axis=1)
        self._ss = (self._S * self._S).sum(axis=1)
        # Guard against exact zeros (can happen at degenerate frequencies).
        self._cc = np.where(self._cc > 0, self._cc, np.inf)
        self._ss = np.where(self._ss > 0, self._ss, np.inf)

    def power(self, y: np.ndarray) -> np.ndarray:
        """Normalised Lomb-Scargle power spectrum of ``y`` (mean removed)."""
        yc = np.asarray(y, dtype=float)
        yc = yc - yc.mean()
        denom = float(yc.dot(yc))
        if denom <= 0:
            return np.zeros(len(self.freqs))
        a = self._C @ yc
        b = self._S @ yc
        return (a * a / self._cc + b * b / self._ss) / denom


# --------------------------------------------------------------------------
# AR(1) phase-preserving surrogate null (promoted from
# examples/erdos/erdos_dsi_sweep.py into the package).
# --------------------------------------------------------------------------


def ar1_surrogate(y0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """AR(1) red-noise surrogate matching the lag-1 autocorrelation of ``y0``.

    Builds a phase-randomised surrogate by driving a first-order
    autoregressive filter whose coefficient ``phi`` equals the lag-1
    autocorrelation of the (mean-subtracted) input, then rescaling to the
    input standard deviation. Preserving the AR(1) structure is the
    appropriate null for log-periodicity: it asks whether a comb peak can
    be manufactured by ordinary red noise of the same colour.

    Parameters
    ----------
    y0
        Mean-centred 1D residual (the detrended oscillation).
    rng
        A seeded ``numpy.random.Generator``.

    Notes
    -----
    Promoted verbatim (modulo signature/typing) from the orphaned erdos
    DSI example so both DSI codepaths share one calibrated null.
    """
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    if n < 3:
        return y0.copy()
    phi = np.corrcoef(y0[1:], y0[:-1])[0, 1]
    phi = float(np.clip(phi if np.isfinite(phi) else 0.0, 0.0, 0.98))
    e = rng.standard_normal(n)
    s = lfilter([math.sqrt(1.0 - phi ** 2)], [1.0, -phi], e)
    s = s - s.mean()
    sd = s.std()
    return s * (y0.std() / sd) if sd > 0 else s


# --------------------------------------------------------------------------
# Comb-harmonicity scoring helpers
# --------------------------------------------------------------------------


def _match_integer_comb(freq: float, base: float) -> tuple[int, float, float]:
    """Nearest integer-comb fit ``omega = k * base``.

    Returns ``(k, predicted_freq, signed_relative_error)`` where the
    relative error is ``(freq - predicted) / predicted``.
    """
    k = max(1, int(round(freq / base))) if base > 0 else 1
    f_pred = base * k
    rel = (freq - f_pred) / f_pred if f_pred > 0 else float("inf")
    return k, f_pred, rel


def _match_sqrt_comb(freq: float, base: float) -> tuple[int, float, float]:
    """Nearest sqrt-integer-comb fit ``omega = base * sqrt(n)``.

    Returns ``(n, predicted_freq, signed_relative_error)``. This is the
    lattice / Voronoi comb (the ``4 pi sqrt(n)`` divisor comb, the
    ``2 pi sqrt(n)`` circle comb); without it a sqrt-x detector wrongly
    rejects the divisor error.
    """
    if base <= 0:
        return 1, base, float("inf")
    n_est = (freq / base) ** 2
    n = max(1, int(round(n_est)))
    f_pred = base * math.sqrt(n)
    rel = (freq - f_pred) / f_pred if f_pred > 0 else float("inf")
    return n, f_pred, rel


def _comb_residual(peaks: np.ndarray, base: float, kind: str) -> float:
    """Mean absolute relative error of ``peaks`` against a comb of ``base``.

    ``kind`` is ``"integer"`` or ``"sqrt"``. Lower is a cleaner comb.
    """
    if len(peaks) == 0 or base <= 0:
        return float("inf")
    errs = []
    for f in peaks:
        if kind == "integer":
            _, _, rel = _match_integer_comb(float(f), base)
        else:
            _, _, rel = _match_sqrt_comb(float(f), base)
        errs.append(abs(rel))
    return float(np.mean(errs))


@dataclass(frozen=True)
class CombFit:
    """Best comb (integer or sqrt-integer) fit to a set of peak frequencies."""

    fundamental: float          # omega_fund
    kind: str                   # "integer" | "sqrt"
    mean_rel_error: float       # mean |rel error| of peaks vs the comb
    n_peaks: int


def _fit_comb(peak_freqs: np.ndarray, freq_grid: np.ndarray) -> CombFit:
    """Estimate the comb fundamental that best explains ``peak_freqs``.

    Tries BOTH an integer comb (``omega = k * fund``) and a sqrt-integer
    comb (``omega = fund * sqrt(n)``). The fundamental candidates are
    derived from the lowest observed peak (interpreting it as the n=1 /
    k=1 line) plus a fine scan around it; the comb model with the smaller
    mean relative error wins. Returns a :class:`CombFit`.
    """
    peak_freqs = np.sort(np.asarray(peak_freqs, dtype=float))
    if len(peak_freqs) == 0:
        return CombFit(float("nan"), "integer", float("inf"), 0)

    f_lo = float(peak_freqs[0])
    df = float(freq_grid[1] - freq_grid[0]) if len(freq_grid) > 1 else 0.0
    # Candidate fundamentals: the lowest peak interpreted as the first comb
    # line, and a +/- window around it (the lowest peak may itself be a
    # higher harmonic if the n=1 line is weak).
    cand = [f_lo]
    for d in (1, 2, 3):
        if f_lo / (d + 1) > 0:
            cand.append(f_lo / (d + 1))  # lowest peak is the (d+1)-th line
    # Fine local scan around f_lo to absorb grid quantisation.
    span = max(df * 3.0, 0.02 * f_lo)
    cand.extend(np.linspace(f_lo - span, f_lo + span, 25).tolist())
    cand = [c for c in cand if c > 0]

    best = CombFit(f_lo, "integer", float("inf"), len(peak_freqs))
    for base in cand:
        for kind in ("integer", "sqrt"):
            res = _comb_residual(peak_freqs, base, kind)
            if res < best.mean_rel_error:
                best = CombFit(float(base), kind, float(res), len(peak_freqs))
    return best


# --------------------------------------------------------------------------
# Per-transform DSI scan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformScan:
    """Result of the DSI scan in a single scaling variable ``u``."""

    variable: str               # "ln_x" | "sqrt_x" | "x^0.25"
    re_dimension: float         # power-law exponent Re(D)
    peak_omega: float           # strongest Lomb-Scargle angular frequency
    peak_power: float           # normalised L-S power at that omega
    comb_fundamental: float     # omega_fund of the best comb fit
    comb_kind: str              # "integer" | "sqrt"
    comb_rel_error: float       # mean |rel error| of top peaks vs comb
    p_value: float              # calibrated p vs AR(1) surrogate ensemble
    top_omegas: tuple           # the top-K peak frequencies


# Built-in scaling-variable transforms. Each maps positive x -> u.
_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "ln_x": np.log,
    "sqrt_x": np.sqrt,
    "x^0.25": lambda x: np.power(x, 0.25),
}

# Default upper bound on the scanned angular frequency (rad). The DSI combs
# we target sit well below this; capping here keeps the low-frequency
# fundamentals resolved on a fixed-size grid regardless of sampling density.
_DEFAULT_OMEGA_CEILING = 60.0

# Minimum relative oscillation amplitude (centred residual std / de-enveloped
# level) for a transform to be scanned at all. Below this the "oscillation"
# is detrending round-off, not signal: an EXACT power law sits at ~1e-16,
# while any real log-periodic ripple is >= O(1e-3). Set well above the
# float floor and well below any physical comb.
_MIN_REL_OSCILLATION = 1e-9


def _power_law_exponent(x: np.ndarray, y: np.ndarray) -> float:
    """Leading power-law exponent Re(D): slope of log|y| vs log x.

    Mirrors ``growth_catcher.catch_growth``'s log-linear envelope fit but
    in log-log coordinates (the box-counting / Re(D) exponent). Uses a
    floor to guard against zero amplitudes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pos = x > 0
    x, y = x[pos], y[pos]
    floor = 1e-12 * (np.max(np.abs(y)) + 1e-300)
    ly = np.log(np.maximum(np.abs(y), floor))
    lx = np.log(x)
    slope, _ = np.polyfit(lx, ly, 1)
    return float(slope)


def _detrend_power_law(x: np.ndarray, y: np.ndarray, exponent: float) -> np.ndarray:
    """Divide out the power-law envelope ``x**exponent``, then centre.

    Leaves the oscillatory residual P(u) whose comb we are after. This is
    the package generalisation of the erdos example's ``err / sqrt(u)``
    (which removes the ``x^{1/4}`` divisor envelope by hand).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    env = np.power(np.maximum(x, 1e-300), exponent)
    r = y / env
    return r - r.mean()


def scan_transform(
    x: np.ndarray,
    y: np.ndarray,
    variable: str,
    *,
    n_freq: int = 2000,
    omega_max: float | None = None,
    top_k: int = 6,
    peak_height_frac: float = 0.15,
    n_surrogates: int = 2000,
    seed: int = 0,
) -> TransformScan:
    """Run the DSI scan for one scaling variable ``u = variable(x)``.

    Detrends the power law, Lomb-Scargles the residual against a uniform
    grid in ``u``, extracts the top-K peaks, fits the comb (integer OR
    sqrt-integer), and calibrates a p-value against an AR(1) surrogate
    ensemble. ``omega_max`` defaults to a fraction of the Nyquist-like
    frequency implied by the ``u`` sampling.
    """
    if variable not in _TRANSFORMS:
        raise ValueError(f"unknown variable {variable!r}; "
                         f"choose from {sorted(_TRANSFORMS)}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pos = x > 0
    x, y = x[pos], y[pos]

    re_dim = _power_law_exponent(x, y)
    resid = _detrend_power_law(x, y, re_dim)

    # Relative oscillation amplitude: the centred residual std as a fraction
    # of the de-enveloped level. For an EXACT power law this collapses to the
    # floating-point floor (~1e-16); any genuine log-periodic ripple is many
    # orders larger. Guard against a spurious comb manufactured from
    # detrending round-off (a pure power law must return no comb).
    env = np.power(np.maximum(x, 1e-300), re_dim)
    level = float(np.mean(np.abs(y / env))) + 1e-300
    rel_amp = float(np.std(resid) / level)
    if rel_amp < _MIN_REL_OSCILLATION:
        return TransformScan(
            variable=variable, re_dimension=re_dim,
            peak_omega=float("nan"), peak_power=0.0,
            comb_fundamental=float("nan"), comb_kind="integer",
            comb_rel_error=float("inf"), p_value=1.0, top_omegas=tuple(),
        )

    u = _TRANSFORMS[variable](x)
    order = np.argsort(u)
    u, resid = u[order], resid[order]

    # Frequency grid. The mean sample spacing in u sets a Nyquist-like cap,
    # but we also impose a physically-motivated ceiling so that a dense
    # ``u`` grid does not blow ``omega_max`` up to hundreds (which would
    # leave the low-frequency comb under-resolved). The DSI combs of
    # interest live below ~60 rad (ln-x fundamentals ~5-6; the sqrt-x
    # divisor/circle combs run to ~4*pi*sqrt(10) ~ 40), matching the erdos
    # sweep's omega in (2, 60).
    du = float(np.mean(np.diff(u))) if len(u) > 1 else 1.0
    nyq = math.pi / du if du > 0 else 50.0
    if omega_max is None:
        omega_max = min(_DEFAULT_OMEGA_CEILING, 0.5 * nyq)
    omega_min = max(2.0 * math.pi / (u[-1] - u[0]), 1e-6) if u[-1] > u[0] else 1e-6
    freqs = np.linspace(omega_min, omega_max, n_freq)

    # One precomputed basis serves both the observed spectrum and the whole
    # AR(1) surrogate ensemble (same u, same freqs).
    basis = _LombScargleBasis(u, freqs)
    power = basis.power(resid)
    peak_idx = int(np.argmax(power))
    peak_omega = float(freqs[peak_idx])
    peak_power = float(power[peak_idx])

    # Top-K peaks for comb fitting.
    pk, _ = find_peaks(power, height=peak_power * peak_height_frac)
    if len(pk) == 0:
        pk = np.array([peak_idx])
    order_pk = np.argsort(power[pk])[::-1]
    top = pk[order_pk][:top_k]
    top_freqs = freqs[np.sort(top)]
    comb = _fit_comb(top_freqs, freqs)

    # Calibrated p-value: max surrogate L-S power vs observed peak power.
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_surrogates)
    for b in range(n_surrogates):
        s = ar1_surrogate(resid, rng)
        null_max[b] = basis.power(s).max()
    p_value = float((np.sum(null_max >= peak_power) + 1) / (n_surrogates + 1))

    return TransformScan(
        variable=variable,
        re_dimension=re_dim,
        peak_omega=peak_omega,
        peak_power=peak_power,
        comb_fundamental=comb.fundamental,
        comb_kind=comb.kind,
        comb_rel_error=comb.mean_rel_error,
        p_value=p_value,
        top_omegas=tuple(float(f) for f in top_freqs),
    )


# --------------------------------------------------------------------------
# The spectrometer
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ComplexDimension:
    """Full complex-dimension fit returned by the spectrometer.

    Attributes
    ----------
    re_dimension
        ``Re(D)``: the leading power-law / box-counting exponent.
    omega_fundamental
        ``Im(D)``: the log/sqrt angular frequency of the comb fundamental.
    variable
        The auto-selected scaling variable that linearised the comb
        (``"ln_x"``, ``"sqrt_x"`` or ``"x^0.25"``).
    comb_kind
        ``"integer"`` (``omega = k * fund``) or ``"sqrt"``
        (``omega = fund * sqrt(n)``).
    p_value
        Calibrated p-value of the selected transform against its AR(1)
        surrogate null. Small p => a real comb.
    has_comb
        True iff ``p_value <= p_threshold`` (the comb survives the null).
    dimension
        Convenience complex number ``Re(D) + i * omega_fundamental``.
    scans
        The per-transform :class:`TransformScan` results, keyed by
        variable, for auditing.
    """

    re_dimension: float
    omega_fundamental: float
    variable: str
    comb_kind: str
    p_value: float
    has_comb: bool
    dimension: complex
    scans: dict


class ComplexDimensionSpectrometer:
    """Auto-selecting complex-dimension instrument for DSI signals.

    Fits ``D = Re(D) + i * omega_fund`` to a signal ``y(x)``, choosing the
    scaling variable (``ln x``, ``sqrt x`` or ``x^{1/4}``) whose comb is
    most significant against a per-transform AR(1) surrogate null. One
    instrument for both the Euler-product (ln-x integer comb) and the
    Voronoi/Hardy (sqrt-x sqrt-integer comb) worlds.

    Example
    -------
    >>> import numpy as np
    >>> from systrophe.catchers.complex_dimension import (
    ...     ComplexDimensionSpectrometer)
    >>> x = np.linspace(2.0, 5000.0, 6000)
    >>> # Cantor-type: N^{ln2/ln3} * (1 + small log-periodic ripple)
    >>> re = np.log(2) / np.log(3)
    >>> om = 5.7192
    >>> y = x**re * (1 + 0.05 * np.cos(om * np.log(x)))
    >>> spec = ComplexDimensionSpectrometer(n_surrogates=500)
    >>> res = spec.fit(x, y)
    >>> res.variable
    'ln_x'
    """

    def __init__(
        self,
        *,
        variables: tuple[str, ...] = ("ln_x", "sqrt_x", "x^0.25"),
        n_freq: int = 2000,
        top_k: int = 6,
        n_surrogates: int = 2000,
        p_threshold: float = 0.05,
        peak_height_frac: float = 0.15,
        seed: int = 0,
    ) -> None:
        for v in variables:
            if v not in _TRANSFORMS:
                raise ValueError(f"unknown variable {v!r}")
        self.variables = variables
        self.n_freq = n_freq
        self.top_k = top_k
        self.n_surrogates = n_surrogates
        self.p_threshold = p_threshold
        self.peak_height_frac = peak_height_frac
        self.seed = seed

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        omega_max: dict | float | None = None,
    ) -> ComplexDimension:
        """Fit the complex dimension, auto-selecting the scaling variable.

        Parameters
        ----------
        x, y
            Equal-length 1D arrays. ``x`` must be positive (a scale
            variable); ``y`` is the DSI observable (amplitude track).
        omega_max
            Optional per-variable max angular frequency. A dict keyed by
            variable name, or a single float applied to all, or None to
            auto-derive from the ``u`` sampling.

        Returns
        -------
        ComplexDimension
            The fit, with the auto-selected variable and calibrated p.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) != len(y):
            raise ValueError("x and y must have equal length")
        if len(x) < 16:
            raise ValueError("need at least 16 samples to fit a comb")

        scans: dict[str, TransformScan] = {}
        for i, v in enumerate(self.variables):
            if isinstance(omega_max, dict):
                om = omega_max.get(v)
            else:
                om = omega_max
            scans[v] = scan_transform(
                x, y, v,
                n_freq=self.n_freq,
                omega_max=om,
                top_k=self.top_k,
                peak_height_frac=self.peak_height_frac,
                n_surrogates=self.n_surrogates,
                # Decorrelate surrogate streams per transform.
                seed=self.seed + 1009 * i,
            )

        # Select the transform with the smallest calibrated p-value. Ties
        # (e.g. both p = 1/(N+1) flooring) are broken by the cleaner comb
        # (smaller mean relative error).
        best_v = min(
            self.variables,
            key=lambda v: (scans[v].p_value, scans[v].comb_rel_error),
        )
        best = scans[best_v]
        has_comb = best.p_value <= self.p_threshold

        return ComplexDimension(
            re_dimension=best.re_dimension,
            omega_fundamental=best.comb_fundamental,
            variable=best_v,
            comb_kind=best.comb_kind,
            p_value=best.p_value,
            has_comb=has_comb,
            dimension=complex(best.re_dimension, best.comb_fundamental),
            scans=scans,
        )


def summarize_dimension_for_report(res: ComplexDimension) -> str:
    """One-line summary for inclusion in a paper / commit message."""
    return (
        f"ComplexDimensionSpectrometer: D = {res.re_dimension:.4f} "
        f"+ i*{res.omega_fundamental:.4f}  "
        f"(var={res.variable}, comb={res.comb_kind}, "
        f"p={res.p_value:.4g}, has_comb={res.has_comb})"
    )
