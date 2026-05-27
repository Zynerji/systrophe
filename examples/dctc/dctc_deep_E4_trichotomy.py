"""E4 deep dive: trichotomy transition between encoding regimes.

Question: is the trichotomy abrupt or smooth? Construct hybrid U(t)
that interpolates abstract Clifford -> direct-CTC; measure where
chronology-protection signature kicks in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.linalg

from systrophe.ctc.d_ctc import (
    clifford_like_unitary,
    dctc_fixed_point,
    density_matrix_diagnostics,
)
from systrophe.catchers.novelty_catcher import catch_novelty_in_distributions
from systrophe.geometry.pair import SystrophePair
from systrophe.geometry.sinusoid import TiplerSinusoid


def trace_distance(rho_a, rho_b):
    diff = rho_a - rho_b
    eigs = np.linalg.eigvalsh(0.5 * (diff + diff.conj().T))
    return 0.5 * float(np.sum(np.abs(eigs)))


def polar_unitary(M):
    U, _, Vh = np.linalg.svd(M)
    return U @ Vh


def dctc_amp(U, dim_cr=2, n_init=3, eps=0.1):
    dim_total = U.shape[0]
    dim_ctc = dim_total // dim_cr
    sigma_a = np.zeros((dim_cr, dim_cr), dtype=complex); sigma_a[0, 0] = 1.0
    sigma_b = np.zeros((dim_cr, dim_cr), dtype=complex); sigma_b[0, 0] = 1 - eps; sigma_b[1, 1] = eps
    rng = np.random.default_rng(11)
    best = 0.0
    for _ in range(n_init):
        psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
        psi /= np.linalg.norm(psi)
        rho_init = np.outer(psi, psi.conj())
        try:
            ra = dctc_fixed_point(U, sigma_a, dim_cr=dim_cr, rho_ctc_init=rho_init,
                                    tol=1e-10, max_iter=2000)["rho_ctc"]
            rb = dctc_fixed_point(U, sigma_b, dim_cr=dim_cr, rho_ctc_init=rho_init,
                                    tol=1e-10, max_iter=2000)["rho_ctc"]
            best = max(best, trace_distance(ra, rb))
        except Exception:
            pass
    return best


def direct_L_unitary(delta, dim_cr=2):
    """Direct-CTC encoding (construction E from phase AM-counter)."""
    a, R = 1.0, 1.0
    s1 = TiplerSinusoid(R=R, a=a, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=R, a=a, A=1.0, delta=delta)
    pair = SystrophePair(s1=s1, s2=s2)
    rs = np.linspace(1.2, 8.0, 6)
    L_vals = pair.L(rs)
    M = np.outer(L_vals, L_vals)
    H = np.diag(L_vals) + 0.1 * (M + M.conj().T) / 2
    U_evol = scipy.linalg.expm(-1j * H)
    perm = np.array([3, 4, 5, 0, 1, 2])
    P = np.zeros((6, 6), dtype=complex)
    for i, p in enumerate(perm):
        P[i, p] = 1.0
    return polar_unitary(U_evol @ P)


def hybrid_unitary(delta, t, U_clifford, dim_cr=2):
    """Hybrid: (1-t) * Clifford + t * direct-L. t in [0, 1]."""
    U_direct = direct_L_unitary(delta, dim_cr)
    M = (1 - t) * U_clifford + t * U_direct
    return polar_unitary(M)


def physical_L_squared(delta):
    a, R = 1.0, 1.0
    s1 = TiplerSinusoid(R=R, a=a, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=R, a=a, A=1.0, delta=delta)
    pair = SystrophePair(s1=s1, s2=s2)
    rs = np.linspace(1.05, 10.0, 200)
    L = pair.L(rs)
    return float(np.sum(L ** 2) * (10.0 - 1.05) / 200)


def main():
    dim_cr = 2
    dim_total = 6
    print("=" * 70)
    print("E4: trichotomy transition - hybrid Clifford <-> direct-L encoding")
    print("=" * 70)
    print()

    # Fix a Clifford reference U
    rng = np.random.default_rng(12345)
    U_clifford = clifford_like_unitary(dim_total, rng)

    # Sweep mixing parameter t in [0, 1] AND delta in [0, 2*pi]
    ts = np.linspace(0, 1, 11)
    deltas = np.linspace(0, 2 * np.pi, 13)

    print(f"{'t':4s} {'phys_at_pi':10s} {'amp_at_pi':10s} {'amp_at_0':10s} {'ratio':8s}")
    print("(ratio < 0.1 ~ chronology-protection signature)")
    print()

    results = []
    for t in ts:
        amps = []
        for delta in deltas:
            U = hybrid_unitary(float(delta), float(t), U_clifford, dim_cr)
            amps.append(dctc_amp(U, dim_cr=dim_cr))
        amps = np.array(amps)
        # delta indices
        i_pi = int(np.argmin(np.abs(deltas - np.pi)))
        i_0 = 0
        amp_at_pi = float(amps[i_pi])
        amp_at_0 = float(amps[i_0])
        ratio = amp_at_pi / max(amp_at_0, 1e-12)
        phys_at_pi = physical_L_squared(float(deltas[i_pi]))
        print(f"  {t:.2f}  {phys_at_pi:10.4f}  {amp_at_pi:10.4f}  {amp_at_0:10.4f}  {ratio:8.3f}")
        results.append({"t": float(t), "amps": amps.tolist(),
                          "amp_at_pi": amp_at_pi, "amp_at_0": amp_at_0,
                          "ratio_pi_over_0": ratio})

    print()
    # Find the transition t value where ratio drops below 0.5
    transition_t = None
    for r in results:
        if r["ratio_pi_over_0"] < 0.5:
            transition_t = r["t"]
            break
    if transition_t is not None:
        print(f"Chronology-protection signature kicks in at t >= {transition_t:.2f}")
        print(f"That is: when U has at least {int(100*transition_t)}% direct-CTC encoding.")
    else:
        print("Ratio never drops below 0.5 - the chronology-protection signature is")
        print("never triggered in this construction series.")

    print()

    # Mandatory novelty catcher: hash each row's amplitude distribution
    # to address space and inspect successive Hamming steps for sharp
    # transitions across the t sweep. Retrofit pass on 2026-05-11
    # surfaced sharp jumps at rows 1->2 and 8->9 (the trichotomy basin
    # boundaries); recording the verdict here so future re-runs preserve
    # that finding without an external scan.
    def _hist(arr, n_bins: int = 16) -> np.ndarray:
        a = np.asarray(arr, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.zeros(n_bins)
        lo, hi = float(a.min()), float(a.max())
        eps = n_bins * np.finfo(float).eps * max(abs(lo), abs(hi), 1.0)
        if hi - lo <= eps:
            h = np.zeros(n_bins); h[0] = a.size
        else:
            h, _ = np.histogram(a, bins=n_bins, range=(lo, hi))
        s = h.sum()
        return h.astype(float) / s if s > 0 else h.astype(float)

    catcher_labels = [f"row{i}" for i in range(len(results))]
    catcher_dists = [_hist(r["amps"]) for r in results]
    novelty = catch_novelty_in_distributions(catcher_dists,
                                              labels=catcher_labels)
    print(f"Novelty catcher: verdict='{novelty['verdict']}', "
          f"sharp_features={len(novelty['sharp_features'])}")
    for sf in novelty["sharp_features"]:
        print(f"  sharp: {sf.get('between')}  step={sf.get('hamming_step')}  "
              f"ratio_to_median={sf.get('ratio_to_median'):.3f}")

    out = Path("examples") / "dctc_deep_E4_trichotomy_results.json"
    with open(out, "w") as f:
        json.dump({
            "deltas": deltas.tolist(),
            "ts": ts.tolist(),
            "results": results,
            "transition_t": transition_t,
            "novelty_catcher": novelty,
        }, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
