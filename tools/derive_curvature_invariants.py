"""Derive symbolic curvature invariants for the Lewis-Papapetrou vacuum metric.

Computes Christoffel, Riemann, Ricci, R, and the Kretschmann scalar
R_{mu nu rho sigma} R^{mu nu rho sigma} for the metric

    ds^2 = -F dt^2 + 2 K dt dphi + L dphi^2 + h(dr^2 + dz^2)

with F, K, L, h all functions of r alone.

Vacuum constraint FL + K^2 = r^2 may be optionally substituted to
simplify the result. The Kretschmann scalar in vacuum equals the
square of the Weyl tensor and feeds the 4D trace anomaly of a
conformally-coupled scalar field.

Usage
-----
    python tools/derive_curvature_invariants.py
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
    print("metric g_{mu nu}:")
    sp.pprint(g)

    g_inv = sp.simplify(g.inv())

    n = 4
    # Christoffel symbols
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

    # Riemann tensor R^mu_{nu rho sigma}
    Riemann = [[[[sp.S(0)] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for mu in range(n):
        for nu in range(n):
            for rho in range(n):
                for sigma in range(n):
                    s = sp.diff(Gamma[mu][nu][sigma], coords[rho])
                    s -= sp.diff(Gamma[mu][nu][rho], coords[sigma])
                    for lam in range(n):
                        s += Gamma[mu][lam][rho] * Gamma[lam][nu][sigma]
                        s -= Gamma[mu][lam][sigma] * Gamma[lam][nu][rho]
                    Riemann[mu][nu][rho][sigma] = sp.simplify(s)

    # Ricci tensor R_{nu sigma} = R^mu_{nu mu sigma}
    Ricci = sp.zeros(n, n)
    for nu in range(n):
        for sigma in range(n):
            s = sp.S(0)
            for mu in range(n):
                s += Riemann[mu][nu][mu][sigma]
            Ricci[nu, sigma] = sp.simplify(s)
    print("\nRicci tensor R_{mu nu}:")
    sp.pprint(Ricci)

    # Ricci scalar
    R_scalar = sp.S(0)
    for mu in range(n):
        for nu in range(n):
            R_scalar += g_inv[mu, nu] * Ricci[mu, nu]
    R_scalar = sp.simplify(R_scalar)
    print("\nR (scalar):")
    sp.pprint(R_scalar)

    # Lower-index Riemann R_{mu nu rho sigma} = g_{mu lambda} R^lambda_{nu rho sigma}
    Riemann_lower = [[[[sp.S(0)] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for mu in range(n):
        for nu in range(n):
            for rho in range(n):
                for sigma in range(n):
                    s = sp.S(0)
                    for lam in range(n):
                        s += g[mu, lam] * Riemann[lam][nu][rho][sigma]
                    Riemann_lower[mu][nu][rho][sigma] = s

    # Kretschmann K = R_{mu nu rho sigma} R^{mu nu rho sigma}
    K_kretschmann = sp.S(0)
    for mu in range(n):
        for nu in range(n):
            for rho in range(n):
                for sigma in range(n):
                    R_lower = Riemann_lower[mu][nu][rho][sigma]
                    R_upper = sp.S(0)
                    for a in range(n):
                        for b in range(n):
                            for c in range(n):
                                for d in range(n):
                                    R_upper += (
                                        g_inv[mu, a] * g_inv[nu, b]
                                        * g_inv[rho, c] * g_inv[sigma, d]
                                        * Riemann_lower[a][b][c][d]
                                    )
                    K_kretschmann += R_lower * R_upper
    K_kretschmann = sp.simplify(K_kretschmann)
    print("\nKretschmann K = R_{mu nu rho sigma} R^{mu nu rho sigma}:")
    sp.pprint(K_kretschmann)


if __name__ == "__main__":
    main()
