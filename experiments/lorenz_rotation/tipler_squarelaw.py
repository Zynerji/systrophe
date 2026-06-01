"""Square-law (intensity) detection of the Tipler log-periodic structure.

Companion to the GW-cross-polarization 'folding' result. There, an even (quadratic)
observable of a Z2-symmetric strange attractor under-reconstructed it (Takens genericity
failure). Here the SAME principle is applied to the Tipler/Systrophe object itself: the
log-periodic g_phiphi envelope (the 'Tipler sinusoid')

    L(u) = A cos(alpha u + delta),     u = ln(r/R),

carries the CTC information in its SIGN (CTC band <=> L < 0) and its discrete-scale-
invariance (DSI) in alpha. A square-law / intensity detector measures L^2:

    L^2 = (A^2/2)[1 + cos(2 alpha u + 2 delta)],

so it (i) DOUBLES the log-frequency (alpha -> 2 alpha), hence reports the apparent DSI
rescaling ratio as sqrt(lambda) = exp(pi/alpha) instead of lambda = exp(2 pi/alpha); and
(ii) DESTROYS the sign, so the chronology (CTC vs non-CTC) information is unrecoverable.

This is a concrete, testable caveat for any intensity-type readout of the band structure
(e.g. a square-law detector in the Knopp-drive band-gating experiments would measure 2 alpha,
not alpha). It also generalises the attractor 'folding' from the Z2 (Lorenz) case to the
log-periodic (DSI) case: square-law observation maps lambda -> sqrt(lambda).
"""

from __future__ import annotations

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior


def _zero_crossings(sig: np.ndarray) -> int:
    s = np.sign(sig - np.mean(sig))
    s[s == 0] = 1
    return int(np.sum(np.abs(np.diff(s)) > 0))


def effective_logfrequency(sig: np.ndarray, u: np.ndarray) -> float:
    """alpha_eff from zero-crossing density of the de-meaned signal: n ~ alpha U/pi."""
    U = float(u[-1] - u[0])
    return _zero_crossings(sig) * np.pi / U


def square_law_experiment(a: float = 1.0, R: float = 1.0,
                          u_min: float = 0.2, u_max: float = 20.0, n: int = 40000) -> dict:
    """Compare linear (L) vs square-law (L^2) detection of the Tipler sinusoid."""
    vs = VanStockumInterior(omega=a / R, R=R)
    alpha = float(vs.alpha)
    # The CTC sign and the DSI live in the log-periodic factor cos(alpha u + delta);
    # the matched-sinusoid phase delta is taken from the analytic L (it does not
    # affect frequency-doubling). This factor IS the sign of the full g_phiphi.
    delta = float(vs.tipler_sinusoid_L().delta)
    u = np.linspace(u_min, u_max, n)
    L = np.cos(alpha * u + delta)            # Tipler log-periodic structure
    I = L * L                                # square-law / intensity observable

    alpha_L = effective_logfrequency(L, u)
    alpha_I = effective_logfrequency(I, u)
    lam_true = float(np.exp(2 * np.pi / alpha))            # DSI rescaling from L
    lam_app = float(np.exp(2 * np.pi / alpha_I))           # apparent, from L^2
    return {
        "a": a, "alpha": alpha,
        "alpha_from_L": alpha_L, "alpha_from_Lsq": alpha_I,
        "freq_doubling_ratio": alpha_I / alpha_L,
        "dsi_lambda_true": lam_true,
        "dsi_lambda_apparent_from_Lsq": lam_app,
        "sqrt_lambda_check": float(np.sqrt(lam_true)),
        "ctc_fraction_from_L": float(np.mean(L < 0)),       # recoverable
        "ctc_fraction_from_Lsq": float(np.mean(I < 0)),     # 0 -> sign destroyed
        "ctc_sign_recoverable_from_Lsq": bool(np.any(I < 0)),
    }


def square_law_sweep(a_values) -> list:
    return [square_law_experiment(float(a)) for a in a_values]
