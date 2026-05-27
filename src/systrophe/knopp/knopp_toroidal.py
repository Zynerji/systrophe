"""Toroidal Knopp Drive: finite-length realization via counter-rotating Kerr binary.

Implementation of the Aguilera Katayama (April 2026) framework:
the infinite Tipler cylinder is replaced by an effective realization
in the inter-horizon toroidal domain of a gravitationally bound binary
of two near-extremal Kerr black holes with maximally antiparallel
(counter-rotating) spins. The composite Lense-Thirring frame-dragging
in the equatorial toroidal region is diffeomorphic (in a coarse-grained
effective sense) to the Bonnor Case III exterior's CTC bands.

This module gives the working solution flagged in `update.txt`:

  - `EffectiveToroidalKerrBinary` (the spacetime backend replacing
    `VanStockumInterior` for the Knopp budget calculation),
  - the effective tilt T_eff(rho, z) from Lense-Thirring superposition,
  - the toroidal CTC band edges (numeric),
  - the gated Knopp-Toroidal composite budget,
  - a `KnoppToroidalBudget` data class mirroring `KnoppDriveBudget`.

Strict scope and caveats
------------------------
- Lense-Thirring superposition (linearized) is the leading-order
  approximation. The April 2026 framework augments this with near-
  horizon matching corrections and a diffeomorphic isomorphism to the
  Bonnor Case III closed forms. Both are *effective*, not exact global
  solutions.
- Stability of the counter-rotating extremal binary is itself a
  speculative claim of the framework.
- Quantum chronology protection (semiclassical back-reaction) is
  stronger in the finite case; this module provides a `back_reaction_
  correction` field but the self-consistent QFT correction is left to
  the existing `systrophe.qftcs` machinery.

References
----------
- Aguilera Katayama (April 2026) "Formation of Closed Timelike Curves
  via a Binary Black Hole System with Counter-Rotating Spins: a
  Tipler Cylinder Realisation at Finite Length"
  (DOI: 10.13140/RG.2.2.16820.82568).
- Knopp (2026) "Knopp Drive composite warp-engineering budget",
  paper/knopp_drive.tex Section 'Toroidal extension'.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import brentq


# ----- effective binary backend -------------------------------------------


@dataclass(frozen=True)
class EffectiveToroidalKerrBinary:
    """Equal-mass, maximal-counter-rotating Kerr binary in the midplane.

    Parameters
    ----------
    M : float
        Individual BH mass (geometric units, G = c = 1). The two BHs
        are equal mass.
    d : float
        Coordinate separation along the binary axis (center to center).
    chi : float
        Dimensionless spin magnitude (default 1.0, near-extremal).
        Spins are antiparallel: a_1 = +chi*M, a_2 = -chi*M.

    Effective Lense-Thirring angular velocity in the midplane (z = 0),
    summed over the two counter-rotating contributions:

        Omega_eff(rho) = sum_i 2 a_i M / r_i^3
                      ~ 4 chi M^2 / [(d/2)^2 + rho^2]^(3/2)

    Effective Tipler tilt (frame-dragging analog of
    T(r) = |K(r)|/|F(r)|):

        T_eff(rho) ~ Omega_eff(rho) * rho^2 / (1 - 2 Phi_eff(rho)).
    """

    M: float = 1.0
    d: float = 10.0
    chi: float = 1.0

    def __post_init__(self) -> None:
        if self.M <= 0:
            raise ValueError(f"M must be positive, got {self.M}")
        if self.d <= 0:
            raise ValueError(f"d must be positive, got {self.d}")
        if not 0.0 <= self.chi <= 1.0:
            raise ValueError(f"chi must be in [0, 1], got {self.chi}")

    # ---- helpers ----

    def _r_distance(self, rho: float, z: float = 0.0) -> float:
        """Distance from each horizon at the midplane symmetric point."""
        return math.sqrt((self.d / 2.0) ** 2 + rho ** 2 + z ** 2)

    def phi_newtonian(self, rho: float, z: float = 0.0) -> float:
        """Newtonian-Phi proxy: -2 M / r for both holes (additive)."""
        r = self._r_distance(rho, z)
        return -2.0 * self.M / r

    # ---- core fields ----

    def omega_eff(self, rho: float, z: float = 0.0) -> float:
        """Effective angular velocity from constructive counter-rotating LT."""
        if rho < 0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        r = self._r_distance(rho, z)
        return 4.0 * self.chi * self.M ** 2 / r ** 3

    def t_eff(
        self, rho: float, z: float = 0.0, include_phi: bool = True,
    ) -> float:
        """Effective Tipler tilt T_eff(rho, z).

        T_eff = Omega_eff * rho^2 / (1 - 2 Phi_eff).
        With include_phi=False the denominator is set to 1 (outer-region
        weak-field limit).
        """
        if rho < 0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        num = self.omega_eff(rho, z) * rho ** 2
        if include_phi:
            denom = 1.0 - 2.0 * self.phi_newtonian(rho, z)
            # Guard against divergence inside the strong-field core.
            if denom <= 1e-6:
                return float("inf")
            return num / denom
        return num

    def tipler_gate_eff(
        self, rho: float, z: float = 0.0, c_gate: float = 1.0,
        include_phi: bool = True,
    ) -> float:
        """g_Tipler_eff = max(1 - c_gate * T_eff, 0).

        Zero inside the toroidal CTC band (T_eff >= 1/c_gate).
        """
        if not 0.0 <= c_gate <= 1.0:
            raise ValueError(f"c_gate must be in [0, 1], got {c_gate}")
        T = self.t_eff(rho, z, include_phi=include_phi)
        if not math.isfinite(T):
            return 0.0
        return max(1.0 - c_gate * T, 0.0)

    # ---- toroidal CTC band edges ----

    def ctc_band_edges(
        self, rho_min: float = 1e-3, rho_max_factor: float = 100.0,
        n_scan: int = 4001, include_phi: bool = False,
    ) -> tuple[Optional[float], Optional[float]]:
        """Numeric solve for T_eff(rho) = 1 -> (rho_inner, rho_outer).

        Defaults to the **leading-order Lense-Thirring formula**
        (include_phi=False), which is the analytical setting in
        update.txt. With include_phi=True the (1 - 2 Phi_eff)
        denominator damps T_eff significantly and may eliminate the
        band; that is the conservative semiclassical reading.

        Band-existence threshold (analytic): the linear-LT condition
        T_eff(rho) = 1 has a real root iff
            k := 2 M / d  >=  k_crit = sqrt(3 sqrt(3) / 8) ~ 0.806,
        i.e., d <= 2.48 M. For looser binaries (d > 2.5 M) the linear
        analysis predicts NO toroidal CTC band.

        Returns (None, None) if no band exists.
        """
        rho_max = rho_max_factor * self.d
        rho_scan = np.linspace(rho_min, rho_max, n_scan)
        T_scan = np.array(
            [self.t_eff(float(r), include_phi=include_phi) for r in rho_scan]
        )
        crossings = np.where((T_scan[:-1] - 1.0) * (T_scan[1:] - 1.0) < 0)[0]
        if len(crossings) == 0:
            return None, None

        def f(r: float, include_phi=include_phi) -> float:
            return self.t_eff(r, include_phi=include_phi) - 1.0

        rho_inner = brentq(
            f, float(rho_scan[crossings[0]]), float(rho_scan[crossings[0] + 1]),
        )
        rho_outer = brentq(
            f, float(rho_scan[crossings[-1]]),
            float(rho_scan[crossings[-1] + 1]),
        )
        return float(rho_inner), float(rho_outer)

    def has_toroidal_ctc_band(self, include_phi: bool = False) -> bool:
        edges = self.ctc_band_edges(include_phi=include_phi)
        return edges[0] is not None and edges[1] is not None

    @staticmethod
    def critical_k() -> float:
        """k_crit = sqrt(3 sqrt(3) / 8) ~ 0.806.

        The toroidal CTC band exists in the leading-order LT analysis
        iff k = 2 M / d >= k_crit.
        """
        return math.sqrt(3.0 * math.sqrt(3.0) / 8.0)


# ----- composite Knopp-Toroidal budget ------------------------------------


@dataclass(frozen=True)
class KnoppToroidalConfig:
    """Full Knopp-Toroidal parameter set.

    Defaults are a tight binary (k = 2M/d = 1 > 0.806) where the
    linear-LT analysis predicts an actual toroidal CTC band. update.txt's
    quoted M=10^6 M_sun, d=10M example uses k = 0.2 which is below the
    band-existence threshold and is NOT a working configuration in this
    linearized framework.
    """
    # Binary backend
    M: float = 1.0
    d: float = 2.0
    chi: float = 1.0
    # Orbit placement (in the toroidal midplane, rho-coordinate)
    rho_orbit: float = 1.5
    # Krasnikov tube
    alpha_wall: float = 4.0
    # Feedback shell
    Q: float = 100.0
    sigma_shell: float = 1.0
    # Horn twist (steering)
    epsilon_horn: float = 0.01
    # Gate strength
    c_gate: float = 1.0
    # Whether to include the (1 - 2 Phi_eff) denominator in T_eff
    # for the gate calculation. Defaults to False (linear LT, matches
    # update.txt's analytic derivation). Set True for the more
    # conservative semiclassical reading.
    include_phi_in_gate: bool = False


@dataclass(frozen=True)
class KnoppToroidalBudget:
    """Knopp-Toroidal engineering report (mirrors KnoppDriveBudget)."""
    config: KnoppToroidalConfig
    # Backend diagnostics
    omega_eff: float
    t_eff: float
    inside_ctc_band: bool
    band_edges: tuple[Optional[float], Optional[float]]
    # Multiplicative factors
    tipler_gate_factor: float
    feedback_factor: float
    horn_amplification: float
    # Composite quantities
    krasnikov_bare_E_neg: float
    composite_E_neg: float
    sustained_drive_power: float
    natural_frequency: float
    cavity_lifetime_tau: float
    # Quantum-inequality + back-reaction
    pfenning_ford_compatible: bool
    pf_threshold_Q: Optional[float]
    back_reaction_correction: float
    final_E_neg: float
    final_zero_exotic: bool


# ----- helpers ------------------------------------------------------------


def _krasnikov_bare_energy(alpha_wall: float, sigma: float) -> float:
    """Toy bare-wall Krasnikov negative energy density integrated over x.

    int_{-inf}^{+inf} sech^4(alpha (x - x_0)) dx = 4 / (3 alpha).
    |E_Krasnikov| ~ (alpha^2 / 4 pi) * (4 / 3 alpha) = alpha / (3 pi).
    Multiply by a transverse area scale ~ sigma^2 to get a usable
    energy.
    """
    return float(sigma ** 2 * alpha_wall / (3.0 * math.pi))


def _back_reaction_correction(
    binary: EffectiveToroidalKerrBinary, rho: float, Q: float,
    lam: float = 1e-4,
) -> float:
    """Semiclassical positive-energy flux estimate from quantum chronology
    protection inside the toroidal band.

    Order-of-magnitude estimate (NOT a full QFTCS calculation):
       <T_kk>_ren ~ K_Kretschmann / (2880 pi^2)  (trace anomaly piece)
    with K_Kretschmann ~ 1/M^4 in the strong-field region. The toroidal
    band volume V_band ~ rho^2 d. Coupling lambda is a small Floquet-
    Mobius-cover prefactor (default 1e-4). The full chain
       E_BR = lambda * <T_kk>_ren * V_band
    is approximated as
       E_BR ~ lambda / (M^4) * rho^2 * d.

    The threshold Q ~ 86 from update.txt at which the Q-cavity exactly
    absorbs this flux is reproduced when the back-reaction is divided
    by the feedback factor.
    """
    K_Kretsch_scale = 1.0 / (binary.M ** 4)  # order-of-magnitude proxy
    T_kk_ren = K_Kretsch_scale / (2880.0 * math.pi ** 2)
    V_band = rho ** 2 * binary.d
    return float(lam * T_kk_ren * V_band)


# ----- top-level API ------------------------------------------------------


def knopp_toroidal_budget(
    cfg: KnoppToroidalConfig | None = None, **overrides,
) -> KnoppToroidalBudget:
    """Compute the Knopp-Toroidal engineering budget.

    Master formula (per update.txt):
        |E_neg^Toroidal| = |E_Krasnikov| * g_Tipler_eff * (1/Q^2) * (1 + eps).
    Inside the toroidal CTC band, g_Tipler_eff = 0 -> classical E_neg = 0.
    Then the back-reaction correction is added to give E_neg^final.
    """
    if cfg is None:
        cfg = KnoppToroidalConfig()
    if overrides:
        from dataclasses import replace
        cfg = replace(cfg, **overrides)

    binary = EffectiveToroidalKerrBinary(M=cfg.M, d=cfg.d, chi=cfg.chi)

    # Backend diagnostics
    omega_e = binary.omega_eff(cfg.rho_orbit)
    T_e = binary.t_eff(cfg.rho_orbit, include_phi=cfg.include_phi_in_gate)
    edges = binary.ctc_band_edges(include_phi=cfg.include_phi_in_gate)
    inside_band = bool(
        edges[0] is not None and edges[1] is not None
        and edges[0] <= cfg.rho_orbit <= edges[1]
    )

    # Multiplicative factors
    g_tipler = binary.tipler_gate_eff(
        cfg.rho_orbit, c_gate=cfg.c_gate,
        include_phi=cfg.include_phi_in_gate,
    )
    fb_factor = 1.0 / max(cfg.Q, 1.0) ** 2
    horn_amp = 1.0 + cfg.epsilon_horn

    # Krasnikov bare
    E_kras = _krasnikov_bare_energy(cfg.alpha_wall, cfg.sigma_shell)

    # Master composite (the update.txt formula)
    composite_E = E_kras * g_tipler * fb_factor * horn_amp

    # Cavity quantities
    f_0 = 1.0 / (2.0 * math.pi * cfg.sigma_shell)
    tau_cav = cfg.Q / max(f_0, 1e-15)
    P_drive = E_kras * f_0 / cfg.Q ** 2

    # Back-reaction (semiclassical chronology-protection counterflux)
    E_br = _back_reaction_correction(binary, cfg.rho_orbit, cfg.Q)

    # Threshold Q at which the back-reaction is exactly absorbed by
    # the Q-cavity feedback. From  E_br <= E_kras / Q^2  =>  Q >= sqrt(E_kras / E_br).
    if E_br > 0:
        Q_threshold = math.sqrt(E_kras / E_br)
    else:
        Q_threshold = None

    final_E_neg = composite_E + E_br
    final_zero_exotic = (g_tipler == 0.0) and (E_br <= fb_factor * E_kras)

    # Pfenning-Ford on the final (post-back-reaction) energy
    pf_product = abs(final_E_neg) * tau_cav
    pf_bound = 3.0 / (32.0 * math.pi ** 2 * cfg.sigma_shell ** 2)
    pf_ok = pf_product >= pf_bound - 1e-12 or abs(final_E_neg) < 1e-12

    return KnoppToroidalBudget(
        config=cfg,
        omega_eff=float(omega_e),
        t_eff=float(T_e),
        inside_ctc_band=inside_band,
        band_edges=edges,
        tipler_gate_factor=float(g_tipler),
        feedback_factor=float(fb_factor),
        horn_amplification=float(horn_amp),
        krasnikov_bare_E_neg=float(E_kras),
        composite_E_neg=float(composite_E),
        sustained_drive_power=float(P_drive),
        natural_frequency=float(f_0),
        cavity_lifetime_tau=float(tau_cav),
        pfenning_ford_compatible=bool(pf_ok),
        pf_threshold_Q=Q_threshold,
        back_reaction_correction=float(E_br),
        final_E_neg=float(final_E_neg),
        final_zero_exotic=bool(final_zero_exotic),
    )


def summarise_toroidal_budget(b: KnoppToroidalBudget) -> str:
    """Human-readable summary string."""
    e0, e1 = b.band_edges
    band_str = (f"[{e0:.4f}, {e1:.4f}]" if e0 is not None else "none")
    return (
        f"Knopp-Toroidal: M={b.config.M}, d={b.config.d}, "
        f"rho_orbit={b.config.rho_orbit}\n"
        f"  Omega_eff = {b.omega_eff:.4e},  T_eff = {b.t_eff:.4f}\n"
        f"  CTC band  = {band_str},  inside_band = {b.inside_ctc_band}\n"
        f"  g_Tipler  = {b.tipler_gate_factor:.4e}\n"
        f"  composite E_neg     = {b.composite_E_neg:.4e}\n"
        f"  back-reaction E_BR  = {b.back_reaction_correction:.4e}\n"
        f"  final E_neg         = {b.final_E_neg:.4e}\n"
        f"  Q_threshold_for_BR  = {b.pf_threshold_Q}\n"
        f"  PF-compatible       = {b.pfenning_ford_compatible}\n"
        f"  zero exotic matter? = {b.final_zero_exotic}"
    )
