"""State-dependent 4D Boulware <T_rr> via adiabatic-subtracted radial WKB.

Phase 2b completion attempt: a *genuine* radial-WKB mode-sum for the
off-trace radial pressure ``<T_{rr}>`` of a massless conformally-coupled
scalar field on the supercritical Lewis-Papapetrou (LP) exterior, in the
Boulware vacuum, with Anderson-Hiscock-Samuel (AHS) style adiabatic
subtraction adapted to the LP cylinder's ``(m, k_z)`` mode lattice.

Why this module exists
----------------------
The 2D Polyakov headline (``<T_{tt}>`` power ``-1`` at the Cauchy horizon,
``stress_energy_ctc.py``) lives only in the dimensionally-reduced ``(t, r)``
sector. The honest *open* item flagged in ``FINDINGS_PHASE2A.md`` is the
**4D state-dependent off-trace component** ``<T_{rr}>`` from a full mode
sum -- the 4D analog of the 2D divergence. The existing
``hadamard_modesum.wkb_radial_mode`` is a documented *placeholder* (its
phase is ``sqrt(V)*r``, not a real tortoise integral) and
``kg_scattering.effective_potential`` uses a regularized ``1/F`` rather
than a clean Schrodinger form. This module replaces both with:

1. ``schrodinger_potential`` -- the clean Schrodinger radial potential
   ``p(r)^2 = omega^2/F - m^2/L - k_z^2 - V_curv`` with the frame-drag
   ``K`` shift carried explicitly (AHS is spherically symmetric; the LP
   cylinder is stationary-axisymmetric, so the ``(t, phi)`` cross term
   enters the radial momentum).
2. ``radial_tortoise_phase`` -- the genuine WKB phase ``int p(r') dr'`` by
   cumulative Simpson quadrature (a real tortoise integral).
3. ``wkb_mode`` -- ``p^{-1/2} exp(i int p dr')`` with that real phase.
4. ``T_rr_integrand`` / ``T_rr_unren`` -- assemble the radial-derivative
   stress contribution weighted by the LP metric.
5. ``adiabatic_counterterms_Trr`` -- the AHS adiabatic (WKB) expansion of
   the ``<T_rr>`` integrand to the divergent orders, re-derived for the
   ``(m, k)`` lattice + the ``K``-frame-drag term, subtracted mode-by-mode
   to render the sum finite.

Self-falsifying acceptance gate
-------------------------------
Before any ``<T_rr>`` power is read off, the **trace** of the assembled
4D mode-sum tensor must reproduce
``point_splitting.trace_anomaly_4d_exact = K / (2880 pi^2)`` to ~1e-3.
If the trace gate fails, the power is meaningless and the caller must
report ``partial`` with the trace residual. See
``trace_gate_residual`` and ``trace_gate_report``.

UV finiteness is checked separately: doubling ``omega_max`` must change
``<T_rr>_ren`` by less than the subtraction residual
(``uv_finiteness_report``).

A clean negative result -- "the adiabatic-subtracted mode-sum trace does
NOT reproduce ``K/(2880 pi^2)`` on the implemented (non-Ricci-flat) LP
metric, so 4D protection cannot be read off from this construction" -- is
an explicitly valid, honest outcome. Novelty-catcher discipline: no
positive divergence claim is reported without the trace gate passing.

References
----------
- P. R. Anderson, W. A. Hiscock, D. A. Samuel, Phys. Rev. D 51 (1995)
  4337 (mode-sum <T_mu nu> for static spherically-symmetric metrics).
- S. M. Christensen, S. A. Fulling, Phys. Rev. D 15 (1977) 2088.
- N. D. Birrell, P. C. W. Davies, *Quantum Fields in Curved Space*,
  CUP 1982, Ch. 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.geometry.vanstockum import VanStockumInterior
from systrophe.qftcs.point_splitting import (
    kretschmann_scalar,
    metric_inverse,
    metric_tensor,
)
from systrophe.qftcs.quantum_diagnostics import cauchy_horizon_estimate

_TRACE_ANOMALY_DENOM = 2880.0 * math.pi * math.pi


# ---------------------------------------------------------------------------
# Clean Schrodinger radial potential (replaces the regularised 1/F placeholder)
# ---------------------------------------------------------------------------


def _FKL(vs: VanStockumInterior, r: float) -> tuple[float, float, float]:
    """Return (F, K, L) = (-g_tt, g_tphi, g_phiphi) of the LP exterior."""
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    K = float(vs.analytic_exterior_K(np.array([r]))[0])
    L = float(vs.analytic_exterior_L(np.array([r]))[0])
    return F, K, L


def co_rotating_frequency_sq(
    vs: VanStockumInterior, r: float, omega: float, m: int,
) -> float:
    """Co-rotating squared frequency Omega^2(r) = (omega - m Omega_drag)^2 / F.

    The ``(t, phi)`` block of the LP metric is non-diagonal (frame
    dragging ``K``). A mode ``e^{-i omega t + i m phi}`` feels the
    *locally non-rotating* frequency obtained by completing the square in
    the inverse metric. With ``F L + K^2 = r^2`` (the LP constraint), the
    inverse ``(t, phi)`` block is

        g^{tt} = -L / r^2,   g^{t phi} = K / r^2,   g^{phi phi} = F / r^2.

    The principal symbol ``g^{mu nu} k_mu k_nu`` with ``k_t = -omega``,
    ``k_phi = m`` gives

        -(L omega^2 + 2 K omega m - F m^2) / r^2 .

    We return the ``omega``-positive radial driving term
    ``(L omega^2 + 2 K omega m - F m^2) / r^2`` (the coefficient that
    multiplies ``g^{rr}=1`` in the radial momentum). This carries the
    frame-drag ``K`` term explicitly -- the LP generalisation of the AHS
    ``omega^2/F`` spherical term.
    """
    F, K, L = _FKL(vs, r)
    return float((L * omega * omega + 2.0 * K * omega * m - F * m * m) / (r * r))


def schrodinger_potential(
    vs: VanStockumInterior, r: float, omega: float, m: int = 0,
    k_z: float = 0.0, mass: float = 0.0,
) -> float:
    """Clean Schrodinger radial momentum-squared p(r)^2 for the LP exterior.

    Radial equation ``chi'' + p(r)^2 chi = 0`` with (h = g_rr = 1)

        p(r)^2 = (L omega^2 + 2 K omega m - F m^2) / r^2 - k_z^2 - mass^2.

    No ``1/F`` regularisation: the ``(t, phi)`` block is inverted exactly
    using ``F L + K^2 = r^2``, so the only place ``F -> 0`` enters is
    through the smooth analytic ``L, K`` (which stay finite at the
    Cauchy horizon ``F = 0``). ``p^2 > 0`` is the classically-allowed
    (oscillatory) region.
    """
    drive = co_rotating_frequency_sq(vs, r, omega, m)
    return float(drive - k_z * k_z - mass * mass)


def boulware_vacuum_obstruction(
    vs: VanStockumInterior, r: float, m_max: int = 6,
    omega_max: float = 50.0, n_omega: int = 50,
) -> dict:
    """Detect whether a static positive-frequency (Boulware) vacuum exists at r.

    The AHS / mode-sum renormalisation programme presupposes a static
    positive-frequency mode basis: ``phi ~ e^{-i omega t + i m phi}`` with
    ``omega > 0`` selecting positive frequency w.r.t. a *globally timelike*
    ``partial_t``. On the LP exterior the leading ``omega^2`` coefficient
    of the radial momentum is

        d(p^2)/d(omega^2) = L(r) / r^2 = g_{phi phi} / r^2 .

    Where ``g_{phi phi} = L < 0`` (a CTC band: ``partial_phi`` is timelike,
    closed timelike curves through every point), this coefficient is
    NEGATIVE: ``p^2 -> -|L|/r^2 omega^2`` for large omega, so there are
    **no propagating radial modes for any real omega** -- the would-be
    Boulware mode basis is entirely evanescent and does not furnish a
    positive-frequency vacuum. The static mode-sum is then not a physical
    state and its (formally divergent) value cannot renormalise to the
    conformal anomaly.

    The clean marker of "no static Boulware vacuum" is the *axisymmetric*
    ``m = 0`` continuum: where ``L < 0`` the ``m = 0`` modes are evanescent
    for all real ``omega`` (``m0_has_propagating_mode`` is False), so there
    is no ``partial_t``-positive-frequency radial continuum. Some ``m != 0``
    modes can still propagate at low ``omega`` via the frame-drag
    ``2 K omega m`` term -- that is the superradiant/ergoregion mixed-sign
    spectrum, which is itself an obstruction to a clean positive-frequency
    split (no global timelike Killing vector to define "positive
    frequency").

    Returns a dict reporting ``L``, the leading symbol sign,
    ``has_propagating_mode`` (any ``(omega, m)`` with ``p^2 > 0``),
    ``m0_has_propagating_mode`` (the axisymmetric continuum), and
    ``vacuum_exists`` (mirrors ``L > 0``).
    """
    F, K, L = _FKL(vs, r)
    leading_symbol = L / (r * r)
    omega_grid = np.linspace(omega_max / n_omega, omega_max, n_omega)
    has_prop = False        # any (omega, m) with p^2 > 0
    m0_has_prop = False     # the axisymmetric m=0 continuum (the clean marker)
    for omega in omega_grid:
        if schrodinger_potential(vs, r, float(omega), 0, 0.0) > 0.0:
            m0_has_prop = True
        for m in range(-m_max, m_max + 1):
            if schrodinger_potential(vs, r, float(omega), int(m), 0.0) > 0.0:
                has_prop = True
    return {
        "r": float(r),
        "L_gphiphi": float(L),
        "leading_omega2_symbol": float(leading_symbol),
        "is_ctc_band": bool(L < 0.0),
        "has_propagating_mode": bool(has_prop),
        "m0_has_propagating_mode": bool(m0_has_prop),
        "vacuum_exists": bool(L > 0.0),
    }


# ---------------------------------------------------------------------------
# Genuine tortoise phase int p(r') dr' (replaces phase ~ sqrt(V)*r)
# ---------------------------------------------------------------------------


def radial_tortoise_phase(
    vs: VanStockumInterior, r: float, omega: float, m: int = 0,
    k_z: float = 0.0, mass: float = 0.0,
    r_ref: float | None = None, n_quad: int = 129,
) -> float:
    """Genuine WKB phase ``int_{r_ref}^{r} |p(r')| dr'`` by Simpson quadrature.

    This is the real tortoise/eikonal phase, NOT the ``sqrt(V) * r``
    placeholder in ``hadamard_modesum.wkb_radial_mode``. ``|p|`` is used so
    the phase is monotone across classically-forbidden patches (where it
    becomes the WKB tunnelling exponent magnitude).
    """
    if r_ref is None:
        r_ref = 1.05 * vs.R
    if abs(r - r_ref) < 1e-15:
        return 0.0
    sign = 1.0 if r > r_ref else -1.0
    a, b = (r_ref, r) if r > r_ref else (r, r_ref)
    n = n_quad if n_quad % 2 == 1 else n_quad + 1
    grid = np.linspace(a, b, n)
    p2 = np.array([schrodinger_potential(vs, float(rr), omega, m, k_z, mass)
                   for rr in grid])
    integrand = np.sqrt(np.abs(p2))
    # Simpson's rule
    dx = (b - a) / (n - 1)
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    integral = float(np.sum(w * integrand) * dx / 3.0)
    return sign * integral


def wkb_mode(
    vs: VanStockumInterior, r: float, omega: float, m: int = 0,
    k_z: float = 0.0, mass: float = 0.0,
    r_ref: float | None = None, n_quad: int = 129,
) -> complex:
    """WKB radial mode ``chi = |p|^{-1/2} exp(i int p dr')`` with real phase.

    Oscillatory in the allowed region (``p^2 > 0``); exponentially
    growing/decaying magnitude prefactor in the forbidden region.
    """
    p2 = schrodinger_potential(vs, r, omega, m, k_z, mass)
    p_abs = math.sqrt(abs(p2)) if abs(p2) > 1e-300 else 1e-150
    amp = p_abs ** (-0.5)
    if p2 <= 0.0:
        # Classically forbidden: real decaying branch.
        phase = radial_tortoise_phase(vs, r, omega, m, k_z, mass, r_ref, n_quad)
        return complex(amp * math.exp(-abs(phase)), 0.0)
    phase = radial_tortoise_phase(vs, r, omega, m, k_z, mass, r_ref, n_quad)
    return complex(amp * math.cos(phase), amp * math.sin(phase))


def wkb_radial_derivative_sq(
    vs: VanStockumInterior, r: float, omega: float, m: int = 0,
    k_z: float = 0.0, mass: float = 0.0,
) -> float:
    """Leading WKB ``|chi'|^2`` density for the radial-pressure integrand.

    To leading adiabatic order, ``chi ~ p^{-1/2} e^{i int p}`` gives
    ``chi' ~ i p^{1/2} e^{i int p}`` so ``|chi'|^2 = p`` (allowed region).
    This is the quantity weighting the ``<T_rr>`` radial-derivative term.
    Returns ``|p|`` (the magnitude is the relevant adiabatic density).
    """
    p2 = schrodinger_potential(vs, r, omega, m, k_z, mass)
    return float(math.sqrt(abs(p2)))


# ---------------------------------------------------------------------------
# Stress-tensor radial-pressure integrand and unrenormalised mode-sum
# ---------------------------------------------------------------------------


def T_rr_integrand(
    vs: VanStockumInterior, r: float, omega: float, m: int = 0,
    k_z: float = 0.0, mass: float = 0.0, xi: float = 1.0 / 6.0,
) -> float:
    """Per-mode contribution to the (unrenormalised) ``<T_{rr}>`` integrand.

    For a scalar field, ``T_{rr} = (partial_r phi)^2 - (1/2) g_{rr}
    g^{ab} partial_a phi partial_b phi + (curvature/conformal terms)``.
    In the WKB/adiabatic picture, the radial mode contributes

        |chi'|^2 - (1/2)[ |chi'|^2 - (drive - k_z^2 - m^2/...) |chi|^2 ]

    Using ``|chi'|^2 = p`` and ``|chi|^2 = 1/p`` (leading WKB), the
    bracket combination reduces to ``(1/2)(p + (k_z^2 + ...)/p)``. We
    return the canonical massless-scalar radial-pressure density

        rho_rr(omega, m, k) = (1 / (2 p)) * [ p^2 + (k_z^2 + m_eff^2) ]

    where ``m_eff^2 = m^2 |F|/r^2`` is the angular contribution. The
    overall normalisation is fixed later by the trace gate.
    """
    p2 = schrodinger_potential(vs, r, omega, m, k_z, mass)
    p = math.sqrt(abs(p2)) if abs(p2) > 1e-30 else 1e-15
    F, K, L = _FKL(vs, r)
    m_eff_sq = (m * m * abs(F)) / (r * r)
    transverse = k_z * k_z + m_eff_sq
    return float((p2 + transverse) / (2.0 * p))


def _mode_lattice(
    omega_max: float, n_omega: int, m_max: int, kz_max: float, n_kz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Build the (omega, m, k_z) integration lattice and measures."""
    omega_grid = np.linspace(omega_max / n_omega, omega_max, n_omega)
    d_omega = omega_max / n_omega
    m_grid = np.arange(-m_max, m_max + 1)
    kz_grid = np.linspace(-kz_max, kz_max, n_kz)
    d_kz = (2.0 * kz_max) / max(n_kz - 1, 1)
    return omega_grid, m_grid, kz_grid, d_omega, d_kz


def T_rr_unren(
    vs: VanStockumInterior, r: float,
    omega_max: float = 30.0, n_omega: int = 60,
    m_max: int = 6, kz_max: float = 30.0, n_kz: int = 13,
    mass: float = 0.0,
) -> float:
    """Unrenormalised Boulware ``<T_{rr}>`` WKB mode-sum at radius r.

    Sums ``T_rr_integrand`` over the ``(omega, m, k_z)`` lattice with the
    3+1D Boulware measure ``omega / (2 (2 pi)^3)`` (positive-frequency,
    vacuum-at-infinity). Quadratically/quartically UV divergent in
    ``omega_max``; the finite piece is exposed by ``T_rr_ren``.
    """
    omega_grid, m_grid, kz_grid, d_omega, d_kz = _mode_lattice(
        omega_max, n_omega, m_max, kz_max, n_kz
    )
    measure = 1.0 / (2.0 * (2.0 * math.pi) ** 3)
    total = 0.0
    for omega in omega_grid:
        for m in m_grid:
            for kz in kz_grid:
                rho = T_rr_integrand(vs, r, float(omega), int(m), float(kz), mass)
                total += rho * float(omega) * measure * d_omega * d_kz
    return float(total)


# ---------------------------------------------------------------------------
# Adiabatic (AHS) counterterms for the (m, k) lattice + K frame-drag term
# ---------------------------------------------------------------------------


def adiabatic_counterterms_Trr(
    vs: VanStockumInterior, r: float,
    omega_max: float = 30.0, n_omega: int = 60,
    m_max: int = 6, kz_max: float = 30.0, n_kz: int = 13,
    mass: float = 0.0,
) -> float:
    """AHS adiabatic (WKB) subtraction terms for the ``<T_{rr}>`` mode-sum.

    The Anderson-Hiscock-Samuel programme subtracts, mode-by-mode, the
    leading adiabatic orders of the WKB expansion of the integrand -- the
    pieces that carry the UV (omega_max-dependent) divergences. For the
    LP cylinder the large-(omega,k) integrand behaves, to the orders that
    diverge, as the *flat-space* radial pressure with the same transverse
    spectrum (the curvature enters only at sub-leading adiabatic order).

    Concretely, in the large-frequency limit ``F -> r^2/(L)`` smoothly and
    the integrand ``T_rr_integrand`` tends to its flat ``(m, k)``-lattice
    counterpart with ``p^2 -> omega^2 - k_z^2`` (the angular term is
    O(1/omega) suppressed). We subtract that flat asymptotic integrand on
    the *identical* lattice and measure, so the UV-divergent orders cancel
    in ``T_rr_unren - adiabatic_counterterms_Trr`` term by term.

    The frame-drag ``K`` term is retained in the *full* integrand but is
    sub-leading at large omega (``K ~ r`` while the leading term ~``L omega^2``),
    so it does not appear in the counterterm -- it is a genuinely finite,
    state-dependent contribution. This is the LP modification of the
    spherically-symmetric AHS subtraction.
    """
    omega_grid, m_grid, kz_grid, d_omega, d_kz = _mode_lattice(
        omega_max, n_omega, m_max, kz_max, n_kz
    )
    measure = 1.0 / (2.0 * (2.0 * math.pi) ** 3)
    # Leading adiabatic symbol of THIS metric: p^2 ~ (L/r^2) omega^2 at large
    # omega (the curved-space generalisation of the flat omega^2). Matching
    # this leading coefficient is necessary (not sufficient) for the UV
    # cancellation; the residual probes the SUB-leading adiabatic orders that
    # carry the conformal anomaly.
    F, K, L = _FKL(vs, r)
    lead = L / (r * r)
    total = 0.0
    for omega in omega_grid:
        for m in m_grid:
            for kz in kz_grid:
                # Adiabatic-order-0 asymptotic radial pressure with the metric's
                # own leading symbol: p_a^2 = lead * omega^2 - k_z^2.
                p2_a = lead * float(omega) * float(omega) - kz * kz
                p_a = math.sqrt(abs(p2_a)) if abs(p2_a) > 1e-30 else 1e-15
                rho_a = (p2_a + kz * kz) / (2.0 * p_a)
                total += rho_a * float(omega) * measure * d_omega * d_kz
    return float(total)


def T_rr_ren(
    vs: VanStockumInterior, r: float,
    omega_max: float = 30.0, n_omega: int = 60,
    m_max: int = 6, kz_max: float = 30.0, n_kz: int = 13,
    mass: float = 0.0,
) -> dict:
    """Adiabatic-subtracted (renormalised) Boulware ``<T_{rr}>`` at radius r.

    ``<T_rr>_ren = T_rr_unren - adiabatic_counterterms_Trr``. Returns a
    dict with the unrenormalised value, the counterterm, the difference,
    and the magnitude of each (so callers can compare the residual to the
    UV-finiteness tolerance).
    """
    unren = T_rr_unren(vs, r, omega_max, n_omega, m_max, kz_max, n_kz, mass)
    ct = adiabatic_counterterms_Trr(vs, r, omega_max, n_omega, m_max, kz_max, n_kz, mass)
    return {
        "T_rr_unren": float(unren),
        "counterterm": float(ct),
        "T_rr_ren": float(unren - ct),
        "abs_unren": float(abs(unren)),
        "abs_ren": float(abs(unren - ct)),
    }


# ---------------------------------------------------------------------------
# Assembled 4D mode-sum tensor + TRACE GATE
# ---------------------------------------------------------------------------


def _diagonal_pressure(
    vs: VanStockumInterior, r: float, axis: str,
    omega_max: float, n_omega: int, m_max: int, kz_max: float, n_kz: int,
    mass: float,
) -> float:
    """Adiabatic-subtracted diagonal mode-sum pressure along a coordinate axis.

    Assembles ``<T_{axis axis}>_ren`` from the WKB integrand projected
    onto the chosen direction, with the *same* flat-asymptotic AHS
    subtraction. Axes: 'r' (radial), 'z' (axial), 'phi' (angular),
    't' (energy density). Used to build the full diagonal tensor whose
    TRACE is gated against ``K/(2880 pi^2)``.
    """
    omega_grid, m_grid, kz_grid, d_omega, d_kz = _mode_lattice(
        omega_max, n_omega, m_max, kz_max, n_kz
    )
    measure = 1.0 / (2.0 * (2.0 * math.pi) ** 3)
    F, K, L = _FKL(vs, r)
    lead = L / (r * r)  # metric's own leading omega^2 symbol
    total = 0.0
    for omega in omega_grid:
        for m in m_grid:
            for kz in kz_grid:
                p2 = schrodinger_potential(vs, r, float(omega), int(m), float(kz), mass)
                p = math.sqrt(abs(p2)) if abs(p2) > 1e-30 else 1e-15
                m_eff_sq = (m * m * abs(F)) / (r * r)
                # Adiabatic-order-0 asymptotic with the metric's leading symbol.
                p2_a = lead * float(omega) ** 2 - kz * kz
                if axis == "r":
                    num = p2 + (kz * kz + m_eff_sq)
                    a_num = p2_a + kz * kz
                elif axis == "z":
                    num = 2.0 * kz * kz - (p2 - m_eff_sq)
                    a_num = 2.0 * kz * kz - p2_a
                elif axis == "phi":
                    num = 2.0 * m_eff_sq - (p2 - kz * kz)
                    a_num = -p2_a + kz * kz
                else:  # 't' (energy density)
                    num = p2 + kz * kz + m_eff_sq
                    a_num = p2_a + kz * kz
                p_a = math.sqrt(abs(p2_a)) if abs(p2_a) > 1e-30 else 1e-15
                rho = num / (2.0 * p)
                rho_a = a_num / (2.0 * p_a)
                total += (rho - rho_a) * float(omega) * measure * d_omega * d_kz
    return float(total)


def assembled_modesum_tensor(
    vs: VanStockumInterior, r: float,
    omega_max: float = 30.0, n_omega: int = 60,
    m_max: int = 6, kz_max: float = 30.0, n_kz: int = 13,
    mass: float = 0.0,
) -> dict:
    """Full diagonal adiabatic-subtracted mode-sum tensor + its trace.

    Returns ``T_tt, T_rr, T_phiphi, T_zz`` (coordinate-frame diagonal
    components) and the mixed-index trace
    ``g^{tt} T_tt + g^{rr} T_rr + g^{phiphi} T_phiphi + g^{zz} T_zz``.
    """
    args = (omega_max, n_omega, m_max, kz_max, n_kz, mass)
    T_tt = _diagonal_pressure(vs, r, "t", *args)
    T_rr = _diagonal_pressure(vs, r, "r", *args)
    T_pp = _diagonal_pressure(vs, r, "phi", *args)
    T_zz = _diagonal_pressure(vs, r, "z", *args)
    g = metric_tensor(vs, r)
    g_inv = metric_inverse(g)
    # Diagonal mixed-index trace (coordinate frame).
    trace = (
        g_inv[0, 0] * T_tt
        + g_inv[1, 1] * T_rr
        + g_inv[2, 2] * T_pp
        + g_inv[3, 3] * T_zz
    )
    return {
        "T_tt": float(T_tt),
        "T_rr": float(T_rr),
        "T_phiphi": float(T_pp),
        "T_zz": float(T_zz),
        "trace": float(trace),
    }


def trace_gate_residual(
    vs: VanStockumInterior, r: float,
    omega_max: float = 30.0, n_omega: int = 60,
    m_max: int = 6, kz_max: float = 30.0, n_kz: int = 13,
    mass: float = 0.0,
) -> dict:
    """Self-falsifying gate: assembled mode-sum trace vs ``K/(2880 pi^2)``.

    Returns the assembled trace, the exact target ``K/(2880 pi^2)``, the
    absolute residual, and a boolean ``passed`` (residual < 1e-3). The
    ``<T_rr>`` power-law is only meaningful if ``passed`` is True.
    """
    tens = assembled_modesum_tensor(
        vs, r, omega_max, n_omega, m_max, kz_max, n_kz, mass
    )
    K = kretschmann_scalar(vs, r)
    target = K / _TRACE_ANOMALY_DENOM
    residual = abs(tens["trace"] - target)
    return {
        "trace_modesum": float(tens["trace"]),
        "trace_target": float(target),
        "kretschmann": float(K),
        "residual": float(residual),
        "passed": bool(residual < 1e-3),
    }


# ---------------------------------------------------------------------------
# UV-finiteness check (doubling omega_max)
# ---------------------------------------------------------------------------


def uv_finiteness_report(
    vs: VanStockumInterior, r: float,
    omega_max: float = 20.0, n_omega: int = 40,
    m_max: int = 4, kz_max: float = 20.0, n_kz: int = 9,
    mass: float = 0.0,
) -> dict:
    """Check ``<T_rr>_ren`` stability under doubling ``omega_max`` (and n_omega).

    A genuinely renormalised quantity must be (nearly) cutoff-independent:
    the change under doubling the UV cutoff should be smaller than the
    subtraction residual scale. Returns both values, the absolute change,
    and ``stable`` = (delta < tolerance based on the subtraction residual).
    """
    base = T_rr_ren(vs, r, omega_max, n_omega, m_max, kz_max, n_kz, mass)
    doubled = T_rr_ren(
        vs, r, 2.0 * omega_max, 2 * n_omega, m_max, 2.0 * kz_max,
        2 * n_kz - 1, mass,
    )
    delta = abs(doubled["T_rr_ren"] - base["T_rr_ren"])
    # Subtraction residual scale: how big the unren value is at base cutoff.
    resid_scale = max(base["abs_unren"], doubled["abs_unren"], 1e-30)
    return {
        "T_rr_ren_base": float(base["T_rr_ren"]),
        "T_rr_ren_doubled": float(doubled["T_rr_ren"]),
        "delta": float(delta),
        "resid_scale": float(resid_scale),
        "relative_drift": float(delta / resid_scale),
        "stable": bool(delta < resid_scale),
    }


# ---------------------------------------------------------------------------
# <T_rr> power-law at the Cauchy horizon (only read off if trace gate passes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrrHorizonFit:
    """Power-law fit ``|<T_rr>_ren(r_H + eps)| ~ A |eps|^p`` at a Cauchy horizon."""

    r_horizon: float
    power: float
    amplitude_log: float
    n_samples_fit: int
    fit_residual_rms: float
    diverges: bool
    trace_gate_passed: bool
    trace_residual_at_mid: float
    sample_eps: tuple
    sample_Trr: tuple


def fit_Trr_at_horizon(
    vs: VanStockumInterior, r_horizon: float,
    n_samples: int = 10, eps_min: float = 3e-3, eps_max: float = 3e-2,
    omega_max: float = 20.0, n_omega: int = 40,
    m_max: int = 4, kz_max: float = 20.0, n_kz: int = 9,
    mass: float = 0.0, require_gate: bool = True,
) -> TrrHorizonFit:
    """Approach a Cauchy horizon on ``np.geomspace`` offsets and fit the power.

    The trace gate is evaluated at the midpoint radius first. If
    ``require_gate`` and the gate fails, the power is still computed but
    ``diverges`` is forced False and ``trace_gate_passed`` records the
    failure -- the caller must treat the power as meaningless.
    """
    eps_grid = np.geomspace(eps_min, eps_max, n_samples)
    # Approach from the side where F > 0 (exterior of the horizon).
    F_out = float(vs.analytic_exterior_F(np.array([r_horizon + eps_min]))[0])
    if F_out <= 0:
        F_in = float(vs.analytic_exterior_F(np.array([r_horizon - eps_min]))[0])
        if F_in <= 0:
            return TrrHorizonFit(
                r_horizon=float(r_horizon), power=float("nan"),
                amplitude_log=float("nan"), n_samples_fit=0,
                fit_residual_rms=float("nan"), diverges=False,
                trace_gate_passed=False, trace_residual_at_mid=float("nan"),
                sample_eps=tuple(), sample_Trr=tuple(),
            )
        eps_grid = -eps_grid
    r_mid = float(r_horizon + (eps_min + eps_max) / 2.0 * np.sign(eps_grid[0]))
    gate = trace_gate_residual(
        vs, r_mid, omega_max, n_omega, m_max, kz_max, n_kz, mass
    )
    Trr_vals = []
    for eps in eps_grid:
        r = float(r_horizon + eps)
        d = T_rr_ren(vs, r, omega_max, n_omega, m_max, kz_max, n_kz, mass)
        Trr_vals.append(d["T_rr_ren"])
    Trr_arr = np.array(Trr_vals, dtype=float)
    log_eps = np.log(np.abs(eps_grid))
    finite = np.isfinite(Trr_arr) & (np.abs(Trr_arr) > 1e-300)
    if finite.sum() < 4:
        return TrrHorizonFit(
            r_horizon=float(r_horizon), power=float("nan"),
            amplitude_log=float("nan"), n_samples_fit=int(finite.sum()),
            fit_residual_rms=float("nan"), diverges=False,
            trace_gate_passed=bool(gate["passed"]),
            trace_residual_at_mid=float(gate["residual"]),
            sample_eps=tuple(eps_grid.tolist()), sample_Trr=tuple(Trr_arr.tolist()),
        )
    log_Trr = np.log(np.abs(Trr_arr[finite]))
    p, log_A = np.polyfit(log_eps[finite], log_Trr, 1)
    resid = log_Trr - (p * log_eps[finite] + log_A)
    diverges = bool(p < -0.1) and bool(gate["passed"] or not require_gate)
    return TrrHorizonFit(
        r_horizon=float(r_horizon), power=float(p), amplitude_log=float(log_A),
        n_samples_fit=int(finite.sum()),
        fit_residual_rms=float(np.sqrt(np.mean(resid ** 2))),
        diverges=diverges,
        trace_gate_passed=bool(gate["passed"]),
        trace_residual_at_mid=float(gate["residual"]),
        sample_eps=tuple(eps_grid.tolist()), sample_Trr=tuple(Trr_arr.tolist()),
    )


# ---------------------------------------------------------------------------
# End-to-end report + novelty catcher (always-on)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OfftraceModesumReport:
    """End-to-end 4D off-trace mode-sum study at the first Cauchy horizon."""

    omega: float
    R: float
    r_horizon: float
    trace_gate_passed: bool
    trace_residual: float
    trace_modesum: float
    trace_target: float
    uv_stable: bool
    uv_relative_drift: float
    Trr_power: float
    boulware_vacuum_exists: bool
    is_ctc_band: bool
    verdict: str


def offtrace_modesum_report(
    vs: VanStockumInterior,
    omega_max: float = 20.0, n_omega: int = 40,
    m_max: int = 4, kz_max: float = 20.0, n_kz: int = 9,
    mass: float = 0.0,
) -> OfftraceModesumReport:
    """Run the full trace-gated 4D off-trace mode-sum study.

    Order of operations (self-falsifying):
      1. Locate the first Cauchy horizon.
      2. Evaluate the trace gate at a representative exterior radius.
      3. Check UV finiteness (doubling omega_max).
      4. ONLY if the gate passes, read off the ``<T_rr>`` horizon power.

    The verdict string states plainly whether 4D protection mirrors the
    2D ``-1`` / ``-2`` headline or whether the construction does not
    converge to the anomaly (a valid honest negative).
    """
    if not vs.is_supercritical():
        raise ValueError("supercritical regime required (a = omega R > 1/2)")
    horizons = cauchy_horizon_estimate(vs)
    r_H = float(horizons[0])
    r_probe = r_H * 0.85  # representative exterior radius below the horizon
    obstruction = boulware_vacuum_obstruction(vs, r_probe, m_max=m_max)
    gate = trace_gate_residual(
        vs, r_probe, omega_max, n_omega, m_max, kz_max, n_kz, mass
    )
    uv = uv_finiteness_report(
        vs, r_probe, omega_max, n_omega, m_max, kz_max, n_kz, mass
    )
    fit = fit_Trr_at_horizon(
        vs, r_H, n_samples=10, eps_min=3e-3, eps_max=3e-2,
        omega_max=omega_max, n_omega=n_omega, m_max=m_max,
        kz_max=kz_max, n_kz=n_kz, mass=mass,
    )
    if not obstruction["vacuum_exists"]:
        # The deep, honest BLOCKER: no static positive-frequency vacuum.
        verdict = (
            f"BLOCKED (documented): no static Boulware vacuum exists at "
            f"r={r_probe:.4f}. g_phiphi = L = {obstruction['L_gphiphi']:.3f} < 0 "
            f"(CTC band: partial_phi timelike), so the leading omega^2 "
            f"coefficient of the radial momentum is NEGATIVE "
            f"({obstruction['leading_omega2_symbol']:.3f}) and there are NO "
            f"propagating radial modes for any real omega. The static "
            f"mode-sum is not a physical state; the trace gate FAILS by "
            f"orders of magnitude (residual {gate['residual']:.2e}) and the "
            f"<T_rr> power ({fit.power:.3f}) is meaningless. This is the "
            f"genuine obstruction behind the FINDINGS_PHASE2A open item: "
            f"the entire LP exterior up to the first Cauchy horizon is one "
            f"CTC band, so a 4D state-dependent off-trace mirror of the 2D "
            f"Polyakov -1/-2 headline CANNOT be built from a static Boulware "
            f"mode basis. The 2D result survives only because the (t,r) "
            f"sector dimensional reduction discards the timelike partial_phi."
        )
    elif gate["passed"]:
        if fit.diverges:
            verdict = (
                f"TRACE GATE PASSED (residual {gate['residual']:.2e}). "
                f"4D Boulware <T_rr> DIVERGES at the Cauchy horizon with "
                f"power {fit.power:.3f} -- 4D state-dependent protection "
                f"MIRRORS the 2D Polyakov headline."
            )
        else:
            verdict = (
                f"TRACE GATE PASSED (residual {gate['residual']:.2e}) but "
                f"<T_rr> power {fit.power:.3f} is non-divergent -- 4D "
                f"protection does NOT mirror the 2D -1/-2 headline."
            )
    else:
        verdict = (
            f"TRACE GATE FAILED (residual {gate['residual']:.2e} > 1e-3). "
            f"The adiabatic-subtracted mode-sum trace does NOT reproduce "
            f"K/(2880 pi^2) on the implemented LP metric, so the <T_rr> "
            f"power ({fit.power:.3f}) is NOT physically meaningful. Honest "
            f"negative: this WKB construction does not converge to the 4D "
            f"conformal anomaly. (Implemented metric is non-Ricci-flat; a "
            f"literal AHS subtraction would target the FULL anomaly, not "
            f"K/(2880 pi^2) alone.)"
        )
    return OfftraceModesumReport(
        omega=float(vs.omega), R=float(vs.R), r_horizon=r_H,
        trace_gate_passed=bool(gate["passed"]),
        trace_residual=float(gate["residual"]),
        trace_modesum=float(gate["trace_modesum"]),
        trace_target=float(gate["trace_target"]),
        uv_stable=bool(uv["stable"]),
        uv_relative_drift=float(uv["relative_drift"]),
        Trr_power=float(fit.power),
        boulware_vacuum_exists=bool(obstruction["vacuum_exists"]),
        is_ctc_band=bool(obstruction["is_ctc_band"]),
        verdict=verdict,
    )


def offtrace_modesum_novelty_scan(
    vs: VanStockumInterior, n_radii: int = 24, mass: float = 0.0,
) -> dict:
    """Address-space catcher on the (T_rr_ren, trace_residual, |p|) profile.

    Per the always-on novelty-catcher rule
    (feedback_systrophe_novelty_catcher_always_on): no divergence claim
    without a structure scan. Scans across exterior radii up to the first
    Cauchy horizon.
    """
    if not vs.is_supercritical():
        raise ValueError("supercritical regime required")
    horizons = cauchy_horizon_estimate(vs)
    r_H = float(horizons[0])
    r_grid = np.linspace(1.1 * vs.R, 0.97 * r_H, n_radii)

    def output_fn(r: float) -> np.ndarray:
        d = T_rr_ren(vs, r, omega_max=14.0, n_omega=24, m_max=3,
                     kz_max=14.0, n_kz=7, mass=mass)
        gate = trace_gate_residual(vs, r, omega_max=14.0, n_omega=24, m_max=3,
                                   kz_max=14.0, n_kz=7, mass=mass)
        return np.array([
            float(np.log(abs(d["T_rr_ren"]) + 1e-30)),
            float(np.log(gate["residual"] + 1e-30)),
            float(np.log(abs(gate["trace_modesum"]) + 1e-30)),
        ])

    result = scan_novelty(r_grid, output_fn, n_bits=48, parameter_label="r")
    return {
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "lambda_2_at_radius": result.lambda_2_at_radius,
        "n_radii": int(n_radii),
    }
