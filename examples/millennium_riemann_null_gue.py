"""Null reference for the Riemann catcher: GUE random-matrix spacings.

Run the same address-space catcher on synthetic GUE-distributed
spacings (Wigner surmise) at the same N as the zeta-zero analysis.
If the catcher returns the SAME verdict on synthetic GUE as on zeta
zeros, the Riemann result is `finite-N consistent with GUE` (i.e.
RH-consistent at the level of the catcher's sensitivity).

If the catcher returns SMOOTH on synthetic GUE but NOVEL on zeta zeros,
the zeta zero spacings deviate from GUE in a catcher-detectable way at
finite N. That would be the actual flag worth following up.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systrophe.novelty_catcher import (
    catch_novelty_per_quantity,
    scan_novelty,
)


def gue_eigenvalue_spacings(n_eigs: int, seed: int = 11) -> np.ndarray:
    """Generate spacings of a GUE matrix (mean-normalised).

    Sample an n_eigs x n_eigs Hermitian matrix with iid complex
    Gaussian entries (Hermitised), diagonalise, sort, return spacings
    normalised to mean=1.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (n_eigs, n_eigs)) + 1j * rng.normal(0, 1, (n_eigs, n_eigs))
    h = (a + a.conj().T) / 2
    eigs = np.linalg.eigvalsh(h)
    eigs = np.sort(eigs)
    # Use the bulk (middle 60%) to avoid edge effects
    lo = int(0.2 * n_eigs)
    hi = int(0.8 * n_eigs)
    bulk = eigs[lo:hi]
    sp = np.diff(bulk)
    return sp / np.mean(sp)


def run_catcher_on_spacings_array(s_norm: np.ndarray, label: str) -> dict:
    """Apply the same catcher pipeline as the Riemann analysis."""
    def fn(idx_float):
        i = int(round(idx_float))
        i = max(0, min(len(s_norm) - 1, i))
        return np.array([float(s_norm[i])])
    n_pts = min(50, len(s_norm))
    indices = np.arange(n_pts, dtype=float)
    scan = scan_novelty(indices, fn, n_bits=32)
    third = len(s_norm) // 3
    per_q = {
        "spacings": {
            "first_third":  s_norm[:third],
            "middle_third": s_norm[third:2 * third],
            "last_third":   s_norm[2 * third:],
        }
    }
    pq = catch_novelty_per_quantity(per_q, n_bins=32)
    return {
        "label": label,
        "n_spacings": int(len(s_norm)),
        "mean": float(np.mean(s_norm)),
        "std":  float(np.std(s_norm)),
        "scan_verdict": scan.verdict,
        "scan_n_sharp": len(scan.sharp_features),
        "per_quantity_verdict": pq["aggregate_verdict"],
        "per_quantity_n_sharp_spacings": len(
            pq["per_quantity"]["spacings"].get("sharp_features", [])
        ),
    }


def main() -> None:
    print("=" * 70)
    print("Riemann null reference: catcher on synthetic GUE spacings")
    print("=" * 70)
    print()

    # GUE size needs to be ~5x the desired bulk size since we take middle 60%
    # For 200 bulk spacings, use 250 eigenvalues * 0.6 = 150 — too few.
    # Solve forward: bulk = 0.6 * N - 1 spacings. For bulk=500: N=836.
    targets = [50, 100, 200, 500]
    results_per_seed = {}
    for seed in (11, 17, 23, 41, 53):
        per_n = []
        for target_n_spacings in targets:
            n_eigs = int(target_n_spacings / 0.6) + 2
            sp = gue_eigenvalue_spacings(n_eigs, seed=seed)
            r = run_catcher_on_spacings_array(sp, label=f"GUE_N{target_n_spacings}_seed{seed}")
            per_n.append(r)
            print(f"  seed={seed} N={target_n_spacings} "
                  f"scan={r['scan_verdict']}/{r['scan_n_sharp']}sharp  "
                  f"per_quantity={r['per_quantity_verdict']}/{r['per_quantity_n_sharp_spacings']}sharp")
        results_per_seed[f"seed_{seed}"] = per_n
        print()

    out_path = Path(__file__).parent / "millennium_riemann_null_gue_results.json"
    out_path.write_text(json.dumps(results_per_seed, indent=2))
    print(f"Wrote {out_path}")
    print()
    print("Interpretation")
    print("==============")
    print("Compare the verdict at each N to the actual Riemann result:")
    print("  N=50:  Riemann scan=smooth/0sharp,  per_q=smooth")
    print("  N=100: Riemann scan=novel/1sharp,   per_q=smooth")
    print("  N=200: Riemann scan=novel/1sharp,   per_q=smooth")
    print("  N=500: Riemann scan=novel/1sharp,   per_q=NOVEL_STRUCTURE/1sharp")
    print()
    print("If synthetic GUE shows the same per_quantity verdict trend (smooth")
    print("at small N, novel at N=500), the Riemann N=500 flag is a finite-N")
    print("artifact -- not a deviation from GUE/RH.")


if __name__ == "__main__":
    main()
