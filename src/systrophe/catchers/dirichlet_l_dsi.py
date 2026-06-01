"""Dirichlet L-function DSI helper: characters, twisted Chebyshev psi,
the explicit-formula cascade, and verified low zeros of L(s, chi).

This generalises the zeta-only prime-DSI machinery
(``examples/millennium/millennium_primes_dsi_inverse.py``) to Dirichlet
L-functions L(s, chi). The bridge is the SAME log-periodic cascade, but
the prime weights ``log p`` are multiplied by a Dirichlet character
``chi(p)``.

Why the comb is *cleaner* than zeta
-----------------------------------
For a non-principal character chi mod q, L(s, chi) is entire: there is no
pole at s = 1, so the explicit formula for the twisted Chebyshev function

    psi(x; chi) = sum_{p^k <= x} chi(p^k) log p

has NO main term ``x`` to subtract. The full content is the oscillatory
sum over the non-trivial zeros rho = 1/2 + i*gamma of L(s, chi):

    psi(x; chi) = - sum_rho x^rho / rho + (lower-order trivial terms).

Pairing rho = 1/2 + i*gamma with its conjugate (chi_3 and chi_4 are REAL
quadratic characters, so their zeros come in conjugate pairs +/- gamma)
and writing u = ln x gives the normalised fluctuation

    f_chi(u) := psi(e^u; chi) / e^{u/2}
              = - sum_{gamma > 0} 2 [ 0.5 cos(gamma u) + gamma sin(gamma u) ]
                                  / (0.25 + gamma^2).

That is *identical in form* to the zeta cascade in
``millennium_primes_dsi_inverse.explicit_formula_cascade`` -- only the set
of log-periodic frequencies ``gamma`` changes: they are now the zeros of
L(s, chi) rather than zeta. So each Dirichlet character produces its own
distinct comb, disjoint from zeta's and from each other.

Characters implemented
----------------------
``chi_3`` (the odd quadratic character mod 3) and ``chi_4`` (the odd
quadratic character mod 4). Both are real, so the conjugate-pairing above
applies verbatim.

The verified low zeros are computed via the Hurwitz-zeta representation

    L(s, chi) = q^{-s} sum_{a=1}^{q} chi(a) zeta(s, a/q)

root-found on the critical line (``low_zeros_via_hurwitz``), or returned
from a checked hardcode (``VERIFIED_LOW_ZEROS``) so the example runs with
no mpmath dependency at import time.
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------
# Dirichlet characters (real quadratic, conductors 3 and 4)
# --------------------------------------------------------------------------

#: chi_4 mod 4: the unique non-principal character mod 4 (odd, quadratic).
#: chi_4(n) = +1 if n == 1 (mod 4), -1 if n == 3 (mod 4), 0 if even.
_CHI4 = {1: 1, 3: -1}

#: chi_3 mod 3: the non-principal character mod 3 (odd, quadratic).
#: chi_3(n) = +1 if n == 1 (mod 3), -1 if n == 2 (mod 3), 0 if 3 | n.
_CHI3 = {1: 1, 2: -1}

_CHAR_TABLES = {3: _CHI3, 4: _CHI4}


def dirichlet_character(q: int):
    """Return the real quadratic character chi mod q (q in {3, 4}).

    The returned callable maps an integer n to chi(n) in {-1, 0, +1}.
    chi is completely multiplicative and q-periodic, so chi(n) depends
    only on ``n % q``.
    """
    if q not in _CHAR_TABLES:
        raise ValueError(f"only q in {sorted(_CHAR_TABLES)} are implemented, got {q}")
    table = _CHAR_TABLES[q]

    def chi(n: int) -> int:
        return table.get(int(n) % q, 0)

    return chi


def character_vector(q: int, values: np.ndarray) -> np.ndarray:
    """Vectorised chi(values) for the real character mod q.

    ``values`` is an integer array; returns a float array of {-1, 0, +1}.
    """
    chi = dirichlet_character(q)
    res = np.array([chi(int(v)) for v in np.asarray(values).ravel()],
                   dtype=float)
    return res.reshape(np.asarray(values).shape)


# --------------------------------------------------------------------------
# Verified low zeros of L(s, chi) (ground truth)
# --------------------------------------------------------------------------

#: First non-trivial zeros gamma of L(s, chi) on the critical line, found
#: by complex root-finding of the Hurwitz-zeta representation at 30 digits
#: (all returned with Re(rho) = 0.5 to machine precision). Disjoint between
#: q = 3 and q = 4 and from the zeta zero 14.134725.
VERIFIED_LOW_ZEROS = {
    4: [6.020949, 10.243770, 12.988098, 16.342607, 18.291993, 21.450611],
    3: [8.039737, 11.249206, 15.704619, 18.261997, 20.455771],
}


def low_zeros(q: int, n: int | None = None) -> np.ndarray:
    """Verified low imaginary parts gamma of the L(s, chi_q) zeros."""
    z = np.array(VERIFIED_LOW_ZEROS[q], dtype=float)
    return z if n is None else z[:n]


def low_zeros_via_hurwitz(q: int, n: int, seeds=None, dps: int = 30):
    """Recompute the low zeros of L(s, chi_q) from scratch with mpmath.

    L(s, chi) = q^{-s} sum_{a=1}^{q} chi(a) * Hurwitz-zeta(s, a/q), root
    found in the complex plane near 0.5 + i*seed. Used to certify
    ``VERIFIED_LOW_ZEROS``; the example itself does not need mpmath.
    """
    from mpmath import mp, mpc, mpf, findroot, zeta as mp_zeta

    mp.dps = dps
    chi = dirichlet_character(q)

    def L(s):
        s = mpc(s)
        tot = mpc(0)
        for a in range(1, q + 1):
            c = chi(a)
            if c:
                tot += c * mp_zeta(s, mpf(a) / q)
        return q ** (-s) * tot

    if seeds is None:
        seeds = VERIFIED_LOW_ZEROS[q][:n]
    out = []
    for s in seeds[:n]:
        r = findroot(L, mpc(0.5, float(s)))
        out.append((float(r.real), float(r.imag)))
    return out


# --------------------------------------------------------------------------
# Twisted Chebyshev psi(x; chi) and the prime race psi(x; q, a)
# --------------------------------------------------------------------------

def primes_up_to(n: int) -> np.ndarray:
    """Boolean-sieve the primes <= n as an int64 array."""
    if n < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(math.isqrt(n)) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.flatnonzero(sieve).astype(np.int64)


def twisted_prime_power_steps(x_max: int, q: int):
    """Sorted prime-power positions p^k <= x_max and their chi-twisted
    psi-weights ``chi(p) * log p``.

    Because chi is completely multiplicative, chi(p^k) = chi(p)^k, so the
    weight at p^k is chi(p)^k * log p. Returns (positions_sorted,
    cumulative_weight) so psi(x; chi) is a single ``searchsorted``.
    Prime powers with chi(p) = 0 (p | q) contribute zero weight.
    """
    primes = primes_up_to(x_max)
    chi_p = character_vector(q, primes)  # chi(p) for each prime
    positions = [primes.astype(float)]
    weights = [chi_p * np.log(primes.astype(float))]
    k = 2
    while True:
        limit = x_max ** (1.0 / k)
        mask = primes <= limit
        sel = primes[mask]
        if sel.size == 0:
            break
        chi_sel = chi_p[mask]
        positions.append(sel.astype(float) ** k)
        # chi(p^k) = chi(p)^k
        weights.append((chi_sel ** k) * np.log(sel.astype(float)))
        k += 1
    pos = np.concatenate(positions)
    w = np.concatenate(weights)
    order = np.argsort(pos)
    pos = pos[order]
    cum = np.cumsum(w[order])
    return pos, cum


def make_twisted_psi(x_max: int, q: int):
    """Vectorised psi(x; chi) = sum_{p^k <= x} chi(p^k) log p."""
    pos, cum = twisted_prime_power_steps(x_max, q)

    def psi(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(pos, x, side="right")
        return np.where(idx > 0, cum[np.clip(idx - 1, 0, len(cum) - 1)], 0.0)

    return psi


def twisted_fluctuation(psi, u: np.ndarray) -> np.ndarray:
    """f_chi(u) = psi(e^u; chi) / e^{u/2}.

    No ``x`` main term and no log corrections are subtracted, because
    L(s, chi) is entire (no pole): the twisted psi is already a pure
    oscillation about zero.
    """
    x = np.exp(u)
    return psi(x) / np.sqrt(x)


def dirichlet_cascade(u: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """Truncated explicit-formula reconstruction of f_chi(u) from the zeros.

    f_model(u) = - sum_gamma 2 [ 0.5 cos(gamma u) + gamma sin(gamma u) ]
                              / (0.25 + gamma^2),

    identical in form to the zeta cascade; only ``gammas`` differ (zeros of
    L(s, chi) instead of zeta). Valid for real chi, whose zeros pair as
    +/- gamma.
    """
    g = np.asarray(gammas, dtype=float)[:, None]
    uu = np.asarray(u, dtype=float)[None, :]
    terms = 2.0 * (0.5 * np.cos(g * uu) + g * np.sin(g * uu)) / (0.25 + g ** 2)
    return -terms.sum(axis=0)


def class_psi_steps(x_max: int, q: int, a: int):
    """Positions and cumulative log-weights of prime powers p^k <= x_max
    with p^k == a (mod q), for the prime race psi(x; q, a).

    psi(x; q, a) = sum_{p^k <= x, p^k == a (mod q)} log p (untwisted).
    Only residue classes ``a`` coprime to q carry primes (beyond the ramified
    prime p | q, which is dropped for cleanliness of the race).
    """
    primes = primes_up_to(x_max)
    positions = []
    weights = []
    # k = 1 and higher powers; residue of p^k mod q.
    k = 1
    while True:
        limit = x_max ** (1.0 / k)
        sel = primes[primes <= limit]
        if sel.size == 0:
            break
        pows = sel.astype(np.int64) ** k
        res = (pows % q)
        m = (res == (a % q)) & ((sel % q) != 0)  # drop ramified prime p | q
        if m.any():
            positions.append(pows[m].astype(float))
            weights.append(np.log(sel[m].astype(float)))
        k += 1
    if not positions:
        return np.array([]), np.array([])
    pos = np.concatenate(positions)
    w = np.concatenate(weights)
    order = np.argsort(pos)
    return pos[order], np.cumsum(w[order])


def make_class_psi(x_max: int, q: int, a: int):
    """Vectorised psi(x; q, a) for the residue class a mod q."""
    pos, cum = class_psi_steps(x_max, q, a)

    def psi(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(pos) == 0:
            return np.zeros_like(x, dtype=float)
        idx = np.searchsorted(pos, x, side="right")
        return np.where(idx > 0, cum[np.clip(idx - 1, 0, len(cum) - 1)], 0.0)

    return psi
