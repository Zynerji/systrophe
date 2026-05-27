"""Tipler sinusoid: log-periodic cosine envelope of the supercritical exterior.

In Bonnor's Case III (a = omega R > 1/2), the vacuum exterior of a van
Stockum cylinder has metric components that oscillate as functions of
log(r / R) with frequency

    alpha = sqrt(4 a^2 - 1).

We model g_{phi phi}(r) in the exterior, r > R, as

    L(r) = r * (r/R)^p * [ A * cos(alpha * ln(r/R) + delta) ]

where p is a slowly-varying real exponent (default 1, set by the leading
asymptotics of the Lewis-Papapetrou system) and (A, delta) are the
matching constants determined by continuity at r = R.

A "Tipler sinusoid" with phase offset delta is the object that, when
combined with another such sinusoid (see `pair.py`), produces interference
in the exterior CTC structure.

Note
----
The exact (A, delta, p) for a single van Stockum cylinder come from
matching the exterior vacuum solution to the interior dust at r = R.
This module exposes them as constructor parameters; callers are expected
to either supply matched values from the literature, fit them from
numerical integration, or use the canonical fit-from-data utility below.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, log, sqrt

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class TiplerSinusoid:
    """Log-periodic cosine envelope of L(r) = g_{phi phi}(r) in the exterior.

    Parameters
    ----------
    R : float
        Cylinder radius (defines log origin).
    a : float
        Source rotation parameter omega * R; must exceed 1/2.
    A : float
        Amplitude of the log-periodic component (matching constant).
    delta : float
        Phase offset (matching constant; "the off-set").
    p : float, optional
        Power-law prefactor exponent; default 1.0 (leading asymptotics).

    Attributes
    ----------
    alpha : float
        Log-frequency sqrt(4 a^2 - 1).
    """

    R: float
    a: float
    A: float
    delta: float
    p: float = 1.0

    def __post_init__(self) -> None:
        if self.R <= 0:
            raise ValueError("R must be positive")
        if self.a <= 0.5:
            raise ValueError(
                f"Tipler sinusoid is defined only for a > 1/2 (got a = {self.a})"
            )

    @property
    def alpha(self) -> float:
        return sqrt(4.0 * self.a * self.a - 1.0)

    def L(self, r: float | np.ndarray) -> np.ndarray:
        """Evaluate the log-periodic g_{phi phi} envelope at radius r."""
        r = np.asarray(r, dtype=float)
        if np.any(r <= 0):
            raise ValueError("r must be positive")
        u = np.log(r / self.R)
        return r * (r / self.R) ** self.p * self.A * np.cos(self.alpha * u + self.delta)

    def zeros(self, r_min: float, r_max: float) -> np.ndarray:
        """Return r-values in [r_min, r_max] where L(r) = 0.

        Zeros mark transitions between timelike and spacelike phi-orbits;
        intervals where L < 0 are CTC regions in this leading-order model.
        """
        if r_min <= 0 or r_max <= r_min:
            raise ValueError("require 0 < r_min < r_max")
        # Cosine zeros: alpha*ln(r/R) + delta = pi/2 + k*pi
        u_min = log(r_min / self.R)
        u_max = log(r_max / self.R)
        k_min = int(np.ceil((self.alpha * u_min + self.delta - np.pi / 2) / np.pi))
        k_max = int(np.floor((self.alpha * u_max + self.delta - np.pi / 2) / np.pi))
        ks = np.arange(k_min, k_max + 1)
        u = (np.pi / 2 + ks * np.pi - self.delta) / self.alpha
        return self.R * np.exp(u)


def fit_log_periodic(
    r: np.ndarray,
    L_values: np.ndarray,
    R: float,
    a: float,
    p_init: float = 1.0,
) -> TiplerSinusoid:
    """Fit (A, delta, p) to numerical L(r) data with alpha fixed by source.

    alpha is fixed by the source: alpha = sqrt(4 a^2 - 1). Free parameters
    are (A, delta, p). Reduces to 3 knobs vs len(r) observables; will
    refuse to fit if len(r) < 6.
    """
    if a <= 0.5:
        raise ValueError("a must exceed 1/2")
    if len(r) < 6:
        raise ValueError(f"need at least 6 data points to fit 3 params, got {len(r)}")
    alpha = sqrt(4.0 * a * a - 1.0)

    def model(r_, A, delta, p):
        u = np.log(r_ / R)
        return r_ * (r_ / R) ** p * A * np.cos(alpha * u + delta)

    popt, _ = curve_fit(model, r, L_values, p0=[1.0, 0.0, p_init], maxfev=10_000)
    A, delta, p = popt
    delta = float(np.mod(delta + np.pi, 2 * np.pi) - np.pi)  # wrap to (-pi, pi]
    return TiplerSinusoid(R=R, a=a, A=float(A), delta=delta, p=float(p))
