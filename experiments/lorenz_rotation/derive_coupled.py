"""Symbolic derivation of the coupled linearized vacuum equations for the
z-independent (t, phi)-sector perturbations of a static van Stockum cylinder.

Metric ansatz (even/rotational sector; the odd z-twist sector decouples at
linear order by z -> -z parity):

    ds^2 = -F dt^2 + 2 K dt dphi + L dphi^2 + h dr^2 + S dz^2

with F, K, L, h, S functions of (t, r). Background is the static van Stockum
exterior (functions of r only, S0 = h0). We compute the Einstein tensor with
sympy, linearize in the perturbations (dF, dK, dL, dh, dS), and classify each
linearized component G^(1)_{mu nu} = 0 as an EVOLUTION equation (contains
second time derivatives) or a CONSTRAINT (does not).

Output: pickled lambdified coefficient functions used by evolve_coupled.py.
This is a one-shot derivation; run it directly to (re)generate the equations and
print their structure.
"""

from __future__ import annotations

import itertools
import sympy as sp


def build_linearized():
    t, r, phi, z, eps = sp.symbols("t r phi z epsilon", real=True)
    coords = [t, r, phi, z]

    # background functions of r, perturbations functions of (t, r)
    F0 = sp.Function("F0")(r); K0 = sp.Function("K0")(r); L0 = sp.Function("L0")(r)
    h0 = sp.Function("h0")(r); S0 = sp.Function("S0")(r)
    dF = sp.Function("dF")(t, r); dK = sp.Function("dK")(t, r); dL = sp.Function("dL")(t, r)
    dh = sp.Function("dh")(t, r); dS = sp.Function("dS")(t, r)

    F = F0 + eps * dF; K = K0 + eps * dK; L = L0 + eps * dL
    h = h0 + eps * dh; S = S0 + eps * dS

    g = sp.Matrix([
        [-F, 0, K, 0],
        [0,  h, 0, 0],
        [K,  0, L, 0],
        [0,  0, 0, S],
    ])
    # analytic block inverse (avoid the very slow sp.simplify(g.inv()))
    den = F * L + K * K          # FL + K^2  (= r^2 for the background)
    ginv = sp.Matrix([
        [-L / den, 0, K / den, 0],
        [0, 1 / h, 0, 0],
        [K / den, 0, F / den, 0],
        [0, 0, 0, 1 / S],
    ])

    # Christoffel symbols
    n = 4
    Gamma = [[[sp.S(0)] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                s = sp.S(0)
                for d in range(n):
                    s += ginv[a, d] * (sp.diff(g[d, b], coords[c])
                                       + sp.diff(g[d, c], coords[b])
                                       - sp.diff(g[b, c], coords[d]))
                Gamma[a][b][c] = sp.Rational(1, 2) * s

    # Ricci tensor
    Ric = sp.zeros(n, n)
    for b in range(n):
        for c in range(n):
            s = sp.S(0)
            for a in range(n):
                s += sp.diff(Gamma[a][b][c], coords[a]) - sp.diff(Gamma[a][b][a], coords[c])
                for d in range(n):
                    s += Gamma[a][a][d] * Gamma[d][b][c] - Gamma[a][c][d] * Gamma[d][b][a]
            Ric[b, c] = s

    Rscalar = sum(ginv[a, b] * Ric[a, b] for a in range(n) for b in range(n))
    Ein = Ric - sp.Rational(1, 2) * g * Rscalar

    # linearize: first-order coefficient = d/d(eps) at eps=0  (fast, exact)
    def lin(expr):
        return sp.expand(sp.diff(expr, eps).subs(eps, 0))

    print("Computing & linearizing Einstein tensor components...")
    G1 = {}
    G0 = {}
    for a, b in itertools.combinations_with_replacement(range(n), 2):
        comp = lin(Ein[a, b])                       # O(eps^1): fast (diff+subs)
        bg = sp.expand(Ein[a, b].subs(eps, 0))      # background only (no eps): smaller
        if comp != 0:
            G1[(a, b)] = comp
        if bg != 0:
            G0[(a, b)] = bg
        print(f"  G1[{a}{b}] done (len={len(str(comp))})")
    return {"coords": coords, "G1": G1, "G0": G0,
            "fields": (dF, dK, dL, dh, dS),
            "background": (F0, K0, L0, h0, S0)}


def lambdify_system(out):
    """Replace all Derivative/Function objects by plain symbols and lambdify each
    linearized component. Returns (symbol_names, lambdas) where lambdas[(a,b)] is a
    numeric function of the symbol list.

    Symbol order (45):
      background: F0,F0r,F0rr,K0,K0r,K0rr,L0,L0r,L0rr,h0,h0r,h0rr,S0,S0r,S0rr
      perturbation (for f in dF,dK,dL,dh,dS): f,f_t,f_r,f_tt,f_rr,f_tr
    """
    t, r = out["coords"][0], out["coords"][1]
    F0, K0, L0, h0, S0 = out["background"]
    dF, dK, dL, dh, dS = out["fields"]

    syms = {}
    subs = {}
    for fn, nm in [(F0, "F0"), (K0, "K0"), (L0, "L0"), (h0, "h0"), (S0, "S0")]:
        s0 = sp.Symbol(nm); s1 = sp.Symbol(nm + "r"); s2 = sp.Symbol(nm + "rr")
        subs[sp.Derivative(fn, (r, 2))] = s2
        subs[sp.Derivative(fn, r)] = s1
        subs[fn] = s0
        syms[nm] = s0; syms[nm + "r"] = s1; syms[nm + "rr"] = s2
    for fn, nm in [(dF, "dF"), (dK, "dK"), (dL, "dL"), (dh, "dh"), (dS, "dS")]:
        s = {k: sp.Symbol(nm + suf) for k, suf in
             [("", ""), ("_t", "_t"), ("_r", "_r"), ("_tt", "_tt"),
              ("_rr", "_rr"), ("_tr", "_tr")]}
        subs[sp.Derivative(fn, (t, 2))] = s["_tt"]
        subs[sp.Derivative(fn, (r, 2))] = s["_rr"]
        subs[sp.Derivative(fn, t, r)] = s["_tr"]
        subs[sp.Derivative(fn, t)] = s["_t"]
        subs[sp.Derivative(fn, r)] = s["_r"]
        subs[fn] = s[""]
        for k in s:
            syms[nm + k] = s[k]

    order = ["F0", "F0r", "F0rr", "K0", "K0r", "K0rr", "L0", "L0r", "L0rr",
             "h0", "h0r", "h0rr", "S0", "S0r", "S0rr"]
    for nm in ["dF", "dK", "dL", "dh", "dS"]:
        order += [nm, nm + "_t", nm + "_r", nm + "_tt", nm + "_rr", nm + "_tr"]
    symlist = [syms[o] for o in order]

    # cse=True splits each large expression into small assignments -> fast compile
    lambdas = {}
    for key, expr in out["G1"].items():
        e2 = expr.subs(subs)
        lambdas[key] = sp.lambdify(symlist, e2, "numpy", cse=True)

    bg_order = ["F0", "F0r", "F0rr", "K0", "K0r", "K0rr", "L0", "L0r", "L0rr",
                "h0", "h0r", "h0rr", "S0", "S0r", "S0rr"]
    bg_symlist = [syms[o] for o in bg_order]
    bg_lambdas = {}
    for key, expr in out.get("G0", {}).items():
        bg_lambdas[key] = sp.lambdify(bg_symlist, expr.subs(subs), "numpy", cse=True)

    return {"order": order, "lambdas": lambdas, "bg_order": bg_order,
            "bg_lambdas": bg_lambdas}


def get_system(cache_path="_coupled_system.dill", rebuild=False):
    """Return the lambdified linearized system, loading from a dill cache if
    present (the symbolic derivation + cse-lambdify costs minutes, so cache it)."""
    import os
    import dill

    if (not rebuild) and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return dill.load(f)
    sysd = lambdify_system(build_linearized())
    with open(cache_path, "wb") as f:
        dill.dump(sysd, f)
    return sysd


def classify(out):
    """Print which linearized components contain second time derivatives."""
    t = out["coords"][0]
    fields = out["fields"]
    names = ["tt", "tr", "tphi", "tz", "rr", "rphi", "rz", "phiphi", "phiz", "zz"]
    idx = {(0, 0): "tt", (0, 1): "tr", (0, 2): "tphi", (0, 3): "tz",
           (1, 1): "rr", (1, 2): "rphi", (1, 3): "rz",
           (2, 2): "phiphi", (2, 3): "phiz", (3, 3): "zz"}
    print("\n=== linearized component classification ===")
    for key, expr in out["G1"].items():
        has_tt = any(expr.has(sp.diff(f, t, t)) for f in fields)
        kind = "EVOLUTION (has d_tt)" if has_tt else "CONSTRAINT"
        present = [str(f.func) for f in fields if expr.has(f)]
        print(f"  G1[{idx[key]:6s}] : {kind:22s} fields={present}")


if __name__ == "__main__":
    out = build_linearized()
    classify(out)
