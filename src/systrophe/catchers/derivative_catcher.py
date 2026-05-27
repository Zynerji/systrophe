"""Derivative catcher: address-space catcher on the first derivative.

The base ``scan_novelty`` catcher is built to flag QUALITATIVE outliers
(Hamming-step jumps larger than median + 3*MAD AND above a 25% n_bits
absolute floor). This makes it the right discriminator for
discontinuous phase boundaries but blind to gradient transitions
(e.g. continuous sigmoid order-parameter curves like the 3-SAT
phase transition).

The derivative catcher resolves this by:

  1. Taking a 1D scalar output_fn(parameter) -> scalar.
  2. Numerically differentiating along the parameter axis.
  3. Feeding the derivative array to scan_novelty.

For a sigmoid transition s(x) = 1 / (1 + exp(-k(x - x_c))), the first
derivative is a clean peak at x = x_c. The catcher's median + 3*MAD
discriminator catches the peak as a sharp feature, recovering x_c.

This module also exposes a SECOND-derivative catcher for detecting
inflection-point transitions where the value AND first derivative are
smooth but the second derivative shows a clear peak (e.g. cusp-like
critical points).
"""

from __future__ import annotations

import numpy as np

from systrophe.catchers.novelty_catcher import NoveltyScanResult, scan_novelty


def _finite_diff(values: np.ndarray, parameters: np.ndarray,
                  order: int = 1) -> np.ndarray:
    """Central-difference numerical derivative of values w.r.t. parameters.

    Returns an array of the SAME length as values (forward-fill at edges).
    """
    p = np.asarray(parameters, dtype=float)
    v = np.asarray(values, dtype=float)
    n = len(p)
    if n < 2:
        return np.zeros_like(v)
    dvdp = np.zeros_like(v)
    # Central diff for interior, forward/backward for edges
    dvdp[1:-1] = (v[2:] - v[:-2]) / (p[2:] - p[:-2])
    dvdp[0] = (v[1] - v[0]) / (p[1] - p[0])
    dvdp[-1] = (v[-1] - v[-2]) / (p[-1] - p[-2])
    if order == 1:
        return dvdp
    return _finite_diff(dvdp, p, order=order - 1)


def scan_novelty_derivative(
    parameter_values: np.ndarray,
    scalar_output_fn,
    n_bits: int = 32,
    radii: tuple[int, ...] = (4, 8, 12, 16),
    derivative_order: int = 1,
    parameter_label: str = "parameter",
) -> NoveltyScanResult:
    """Scan for novelty in the k-th derivative of a scalar output.

    Parameters
    ----------
    parameter_values
        1D array of parameter samples (assumed monotonic in input order).
    scalar_output_fn
        Callable parameter -> scalar (float). Wrapped to expose the
        derivative array as a real-array output to the base catcher.
    derivative_order
        1 = first derivative (sigmoid peak detection),
        2 = second derivative (cusp / inflection detection).

    Returns
    -------
    NoveltyScanResult with the derivative array as the output series.
    """
    p_arr = np.asarray(parameter_values, dtype=float)
    raw_values = np.array([float(scalar_output_fn(float(x))) for x in p_arr])
    deriv = _finite_diff(raw_values, p_arr, order=derivative_order)

    def derived_output_fn(p_val: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(p_arr - p_val)))
        return np.array([float(deriv[idx])])

    return scan_novelty(
        p_arr, derived_output_fn, n_bits=n_bits, radii=radii,
        parameter_label=f"{parameter_label}_d{derivative_order}",
    )


def catch_smooth_transition(
    parameter_values: np.ndarray,
    scalar_output_fn,
    n_bits: int = 32,
) -> dict:
    """Two-pass catcher for smooth-sigmoid transitions.

    Pass 1: base catcher on values (catches qualitative jumps).
    Pass 2: derivative catcher on the first derivative (catches
            sigmoid centres).

    Returns a dict::

        {
          "value_scan": NoveltyScanResult (verdict, sharp_features, ...),
          "derivative_scan": NoveltyScanResult,
          "estimated_transition_centre": <parameter_value> or None,
          "kind": "discontinuous" | "smooth_sigmoid" | "none",
        }

    The estimated centre is the parameter at the sharpest derivative
    sharp feature; if neither scan flags anything, ``kind == "none"``.
    """
    p_arr = np.asarray(parameter_values, dtype=float)

    def value_fn(p_val: float) -> np.ndarray:
        return np.array([float(scalar_output_fn(float(p_val)))])

    value_scan = scan_novelty(p_arr, value_fn, n_bits=n_bits)
    derivative_scan = scan_novelty_derivative(
        p_arr, scalar_output_fn, n_bits=n_bits, derivative_order=1,
    )

    kind = "none"
    centre = None
    if value_scan.verdict == "novel_structure" and value_scan.sharp_features:
        kind = "discontinuous"
        sharp = max(value_scan.sharp_features,
                    key=lambda s: s.get("excess_over_median", 0))
        centre = float(sharp["parameter_value"])
    elif derivative_scan.verdict == "novel_structure" and derivative_scan.sharp_features:
        kind = "smooth_sigmoid"
        sharp = max(derivative_scan.sharp_features,
                    key=lambda s: s.get("excess_over_median", 0))
        centre = float(sharp["parameter_value"])

    return {
        "value_scan": value_scan,
        "derivative_scan": derivative_scan,
        "estimated_transition_centre": centre,
        "kind": kind,
    }
