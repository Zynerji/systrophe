"""Derive vacuum Einstein equations for cylindrical WLP metric (SymPy).

Metric ansatz (Lewis 1932, van Stockum 1937, Bonnor 1980):

    ds^2 = -F(r) dt^2 + 2 K(r) dt dphi + L(r) dphi^2 + h(r) (dr^2 + dz^2)

with three commuting Killing vectors ∂_t, ∂_phi, ∂_z. All metric functions
depend on r alone. Computes the Ricci tensor in vacuum and extracts the
independent ODEs.

Run once and verify; the resulting ODE system is then encoded in
`systrophe.geometry.lewis_papapetrou` for runtime use without a SymPy dependency.

Usage
-----
    python tools/derive_lewis_papapetrou.py
"""

from __future__ import annotations

import sympy as sp


def main() -> None:
    t, r, phi, z = sp.symbols("t r phi z", real=True)
    coords = [t, r, phi, z]
    F = sp.Function("F")(r)
    K = sp.Function("K")(r)
    L = sp.Function("L")(r)
    h = sp.Function("h")(r)

    g = sp.Matrix(
        [
            [-F, 0, K, 0],
            [0, h, 0, 0],
            [K, 0, L, 0],
            [0, 0, 0, h],
        ]
    )
    g_inv = sp.simplify(g.inv())

    n = 4
    Gamma = [[[sp.S(0)] * n for _ in range(n)] for _ in range(n)]
    for mu in range(n):
        for nu in range(n):
            for rho in range(n):
                s = sp.S(0)
                for sigma in range(n):
                    s += g_inv[mu, sigma] * (
                        sp.diff(g[rho, sigma], coords[nu])
                        + sp.diff(g[nu, sigma], coords[rho])
                        - sp.diff(g[nu, rho], coords[sigma])
                    )
                Gamma[mu][nu][rho] = sp.simplify(s / 2)

    Ricci = sp.zeros(n, n)
    for mu in range(n):
        for nu in range(n):
            s = sp.S(0)
            for lam in range(n):
                s += sp.diff(Gamma[lam][mu][nu], coords[lam])
                s -= sp.diff(Gamma[lam][mu][lam], coords[nu])
                for sig in range(n):
                    s += Gamma[lam][lam][sig] * Gamma[sig][mu][nu]
                    s -= Gamma[lam][nu][sig] * Gamma[sig][mu][lam]
            Ricci[mu, nu] = sp.simplify(s)

    print("=" * 70)
    print("Vacuum Einstein equations R_{mu nu} = 0 for cylindrical WLP")
    print("=" * 70)
    for mu in range(n):
        for nu in range(mu, n):
            if Ricci[mu, nu] != 0:
                print(f"\nR[{mu},{nu}] = 0:")
                sp.pprint(sp.simplify(Ricci[mu, nu]))

    # Constraint check: FL + K^2 = r^2 (canonical Weyl coord)
    # Substitute L = (r^2 - K^2)/F and re-simplify
    print("\n" + "=" * 70)
    print("After substituting constraint L = (r^2 - K^2)/F (canonical Weyl):")
    print("=" * 70)
    L_sub = (r**2 - K**2) / F
    for mu in range(n):
        for nu in range(mu, n):
            R_subbed = Ricci[mu, nu].subs(L, L_sub).doit()
            R_simp = sp.simplify(R_subbed)
            if R_simp != 0:
                print(f"\nR[{mu},{nu}] [constrained] = 0:")
                sp.pprint(R_simp)


if __name__ == "__main__":
    main()
