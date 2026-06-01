"""Validate the derived coupled linearized vacuum equations on van Stockum.

Three checks:
  R  Regularity: no component has a 1/F0 denominator -> the equations are regular
     at ergosurfaces (F0 = 0) in metric variables. (The single-variable model's
     instability came from the variable choice omega = K/F, not from physics.)
  B  Background vacuum: G0_{mu nu} ~ 0 on the numerical van Stockum background
     (validates the background + its conformal factor h).
  Z  Static-family zero mode: d_a(van Stockum) is a static (d_t = 0) solution of
     the linearized equations, so G1_{mu nu} ~ 0 there (validates the equations).
"""

from __future__ import annotations

import warnings

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior
from systrophe.geometry.lewis_papapetrou import integrate_lp_exterior

from derive_coupled import get_system

warnings.filterwarnings("ignore")

# Ergosurface regularization: in Lewis-Papapetrou coordinates the conformal
# factor h = g_rr = g_zz diverges (~ e^700) at ergosurfaces F0 = 0, a COORDINATE
# pathology. In the linearized equations every h-dependent term carries 1/h or
# 1/h^2, so it -> 0 there (a regular limit); the only problem is float OVERFLOW
# of h^2. Capping h at H_CAP realises that limit numerically without overflow.
H_CAP = 1e40


# --- background builders --------------------------------------------------- #
def _d_dr(y, r):
    """4th-order-ish first derivative on a uniform grid (np.gradient, 2nd order)."""
    return np.gradient(y, r, edge_order=2)


def background_fields(a0, R, r):
    """F0,K0,L0 (analytic) and h0,S0 (LP integrator) plus r-derivatives on grid r."""
    vs = VanStockumInterior(omega=a0 / R, R=R)
    F0 = np.asarray(vs.analytic_exterior_F(r), float)
    K0 = np.asarray(vs.analytic_exterior_K(r), float)
    L0 = np.asarray(vs.analytic_exterior_L(r), float)
    lp = integrate_lp_exterior(omega_dust=a0 / R, R=R, r_max=float(r[-1]) + 0.5,
                               n_samples=8000)
    h0 = np.clip(np.interp(r, lp.r, lp.h), None, H_CAP)   # ergosurface regularization
    S0 = h0.copy()
    d = {}
    for nm, y in [("F0", F0), ("K0", K0), ("L0", L0), ("h0", h0), ("S0", S0)]:
        yr = _d_dr(y, r)
        yrr = _d_dr(yr, r)
        d[nm] = y; d[nm + "r"] = yr; d[nm + "rr"] = yrr
    return d


def static_zero_mode(a0, R, r, da=1e-3):
    """(dF,dK,dL,dh,dS) = d/da of the van Stockum solution (a static perturbation),
    plus their r-derivatives. Time derivatives are zero."""
    def fields(a):
        vs = VanStockumInterior(omega=a / R, R=R)
        F = np.asarray(vs.analytic_exterior_F(r), float)
        K = np.asarray(vs.analytic_exterior_K(r), float)
        L = np.asarray(vs.analytic_exterior_L(r), float)
        lp = integrate_lp_exterior(omega_dust=a / R, R=R, r_max=float(r[-1]) + 0.5,
                                   n_samples=8000)
        h = np.clip(np.interp(r, lp.r, lp.h), None, H_CAP)   # ergosurface regularization
        return F, K, L, h, h.copy()

    Fp, Kp, Lp, hp, Sp = fields(a0 + da)
    Fm, Km, Lm, hm, Sm = fields(a0 - da)
    out = {}
    for nm, yp, ym in [("dF", Fp, Fm), ("dK", Kp, Km), ("dL", Lp, Lm),
                       ("dh", hp, hm), ("dS", Sp, Sm)]:
        dval = (yp - ym) / (2 * da)
        out[nm] = dval
        out[nm + "_r"] = _d_dr(dval, r)
        out[nm + "_rr"] = _d_dr(out[nm + "_r"], r)
        out[nm + "_t"] = np.zeros_like(dval)
        out[nm + "_tt"] = np.zeros_like(dval)
        out[nm + "_tr"] = np.zeros_like(dval)
    return out


def main():
    a0, R = 1.5, 1.0
    sysd = get_system()
    order, lambdas = sysd["order"], sysd["lambdas"]
    bg_order, bg_lambdas = sysd["bg_order"], sysd["bg_lambdas"]

    idx = {(0, 0): "tt", (0, 1): "tr", (0, 2): "tphi", (1, 1): "rr",
           (1, 2): "rphi", (2, 2): "phiphi", (3, 3): "zz"}

    # --- R: numerical regularity ACROSS an ergosurface (F0=0 at ~1.54) -------
    print("\n=== R: regularity across an ergosurface (F0=0) ===")
    r_erg = np.linspace(1.40, 1.70, 600)          # straddles the F0=0 surface ~1.54
    rb_e = background_fields(a0, R, r_erg)
    zm_e = static_zero_mode(a0, R, r_erg)
    val_e = {o: rb_e[o] for o in bg_order}
    for nm in ["dF", "dK", "dL", "dh", "dS"]:
        for suf in ["", "_t", "_r", "_tt", "_rr", "_tr"]:
            val_e[nm + suf] = zm_e[nm + suf]
    args_e = [val_e[o] for o in order]
    F0e = rb_e["F0"]
    i_erg = int(np.argmin(np.abs(F0e)))
    print(f"  min|F0| on grid = {np.abs(F0e)[i_erg]:.2e} at r={r_erg[i_erg]:.3f}")
    for key, fn in lambdas.items():
        g1 = np.asarray(fn(*args_e), float)
        finite = np.all(np.isfinite(g1))
        print(f"  G1[{idx.get(key, key):6s}] : finite across ergosurface = {finite}  "
              f"(value at F0~0: {g1[i_erg]:.2e})")

    # evaluation grid: ergosurfaces for a0=1.5 are ~1.54, 4.69; sample safe points
    vs = VanStockumInterior(omega=a0 / R, R=R)
    r = np.linspace(2.0, 8.0, 400)
    F0r = np.asarray(vs.analytic_exterior_F(r), float)
    safe = np.abs(F0r) > 0.05                      # away from ergosurfaces
    rb = background_fields(a0, R, r)
    zm = static_zero_mode(a0, R, r)

    print("\n=== B: background vacuum  G0 ~ 0 ===")
    bg_vals = [rb[o] for o in bg_order]
    for key, fn in bg_lambdas.items():
        g0 = np.asarray(fn(*bg_vals), float)
        # normalise by a curvature scale ~ |F0''|+|K0''|+|L0''|
        scale = np.abs(rb["F0rr"]) + np.abs(rb["K0rr"]) + np.abs(rb["L0rr"]) + 1e-12
        rel = np.abs(g0)[safe] / scale[safe]
        print(f"  G0[{idx.get(key, key):6s}] : max|G0|/scale = {np.max(rel):.2e}")

    print("\n=== Z: static-family zero mode  G1 ~ 0 ===")
    val = {o: rb[o] for o in bg_order}
    for nm in ["dF", "dK", "dL", "dh", "dS"]:
        for suf in ["", "_t", "_r", "_tt", "_rr", "_tr"]:
            val[nm + suf] = zm[nm + suf]
    args = [val[o] for o in order]
    scale = (np.abs(zm["dF_rr"]) + np.abs(zm["dK_rr"]) + np.abs(zm["dL_rr"]) + 1e-12)
    worst = 0.0
    for key, fn in lambdas.items():
        g1 = np.asarray(fn(*args), float)
        rel = np.abs(g1)[safe] / scale[safe]
        m = float(np.nanmax(rel))
        worst = max(worst, m)
        print(f"  G1[{idx.get(key, key):6s}] : max|G1|/scale = {m:.2e}")
    print(f"\n  worst static-zero-mode residual = {worst:.2e}  "
          f"({'PASS' if worst < 5e-2 else 'CHECK'})")


if __name__ == "__main__":
    main()
