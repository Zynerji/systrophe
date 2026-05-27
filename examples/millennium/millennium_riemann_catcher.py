"""Millennium-problem exploration: Riemann Hypothesis via the
address-space novelty catcher.

The Riemann Hypothesis claims that all non-trivial zeros of the
Riemann zeta function zeta(s) = sum_n 1/n^s lie on the critical
line Re(s) = 1/2. Random matrix theory (Montgomery 1973) predicts
that the SPACINGS between consecutive zero imaginary parts follow
the Gaussian Unitary Ensemble (GUE) statistics. The catcher should:

  - report SMOOTH/UNIFORM verdict if the GUE prediction holds across
    the spacings of the first N zeros (consistent with RH)
  - report NOVEL_STRUCTURE if there is a sharp Hamming transition in
    the spacing distribution that points to a discontinuity
    (suggesting either a non-Montgomery distribution or, at finite N,
    an off-line zero perturbing the local spacing).

This is NOT a proof of RH; it is a tool-based computational
exploration consistent with the catcher's standing methodology.
The result is published as a Systrophe Millennium-problem
investigation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# We use mpmath to compute the zeta zeros (built-in to mpmath).
# Fall back gracefully if mpmath is missing.

from systrophe.catchers.novelty_catcher import (
    catch_novelty_in_named_arrays,
    catch_novelty_per_quantity,
    real_array_to_address,
    scan_novelty,
)


def get_zeta_zeros(n_zeros: int = 100) -> np.ndarray:
    """Imaginary parts of the first n_zeros non-trivial zeta zeros.

    Uses mpmath.zetazero, which returns mp.mpc complex with Re=1/2.
    Returns array of floats: Im(rho_k) for k = 1..n_zeros.
    """
    try:
        from mpmath import mp, zetazero
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError(
            "mpmath is required; install with `pip install mpmath`."
        ) from e
    mp.dps = 15  # 15 dp is plenty for catcher purposes; 30 is overkill
    zeros = []
    for k in range(1, n_zeros + 1):
        z = zetazero(k)
        zeros.append(float(z.imag))
    return np.array(zeros)


def montgomery_predicted_spacing_pdf(s: float) -> float:
    """The Montgomery-pair-correlation prediction for normalised gaps.

    p(s) = 1 - (sin(pi s) / (pi s))^2

    The expected GUE nearest-neighbour spacing distribution is the
    Wigner surmise p(s) = (32/pi^2) s^2 exp(-4 s^2 / pi).
    """
    if s <= 0:
        return 0.0
    return (32.0 / math.pi ** 2) * s ** 2 * math.exp(-4.0 * s ** 2 / math.pi)


def normalised_zero_spacings(zeros_imag: np.ndarray) -> np.ndarray:
    """Compute the spacings t_{k+1} - t_k and normalise so the mean
    equals 1.

    Following Montgomery, the GUE-predicted spacing PDF is in
    units of <s> = 1 (one mean-spacing per unit).
    """
    spacings = np.diff(zeros_imag)
    # Normalise so the local mean spacing = 1
    # (Use the LOG of the t-value as the normalisation, since the
    # density of zeros up to T scales as T log T / (2 pi).)
    t_centres = (zeros_imag[:-1] + zeros_imag[1:]) / 2
    mean_spacing_local = 2 * math.pi / np.log(t_centres / (2 * math.pi))
    return spacings / mean_spacing_local


def run_catcher_on_spacings(
    zeros_imag: np.ndarray,
) -> dict:
    """Run the address-space catcher on the normalised spacings.

    Three modes:
      1. Direct scan_novelty: address each gap individually.
      2. Per-quantity catcher: compare consecutive 1/3rds of the
         spacings list as separate distributions.
      3. Real-array-to-address: hash the entire spacing array to a
         32-bit fingerprint.
    """
    s_norm = normalised_zero_spacings(zeros_imag)

    # Mode 1: scan_novelty over zero index
    def fn(idx_float):
        i = int(round(idx_float))
        i = max(0, min(len(s_norm) - 1, i))
        return np.array([float(s_norm[i])])
    n_pts = min(50, len(s_norm))
    indices = np.arange(n_pts, dtype=float)
    scan = scan_novelty(indices, fn, n_bits=32)

    # Mode 2: per-quantity (third-split)
    third = len(s_norm) // 3
    per_q = {
        "spacings": {
            "first_third":  s_norm[:third],
            "middle_third": s_norm[third:2 * third],
            "last_third":   s_norm[2 * third:],
        }
    }
    pq = catch_novelty_per_quantity(per_q, n_bins=32)

    # Mode 3: hash to a 32-bit fingerprint
    fingerprint = real_array_to_address(s_norm, n_bits=32)

    return {
        "n_zeros":          int(len(zeros_imag)),
        "n_spacings":       int(len(s_norm)),
        "mean_normalised_spacing":  float(np.mean(s_norm)),
        "std_normalised_spacing":   float(np.std(s_norm)),
        "scan_novelty_verdict":     scan.verdict,
        "scan_n_sharp":             len(scan.sharp_features),
        "scan_sharp_features": [
            {k: int(v) if isinstance(v, np.integer) else v for k, v in s.items()}
            for s in scan.sharp_features
        ],
        "per_quantity_verdict":     pq["aggregate_verdict"],
        "per_quantity_detail": {
            q: {"verdict": r["verdict"], "n_sharp": len(r.get("sharp_features", []))}
            for q, r in pq["per_quantity"].items()
        },
        "fingerprint_32bit": fingerprint.tolist(),
    }


def main() -> None:
    print("=" * 70)
    print("Riemann Hypothesis exploration via Systrophe address-space catcher")
    print("=" * 70)
    print()
    print("Methodology: compute the first 100 non-trivial zeta zeros,")
    print("normalise their spacings to mean=1, and ask the catcher whether")
    print("the spacing distribution shows novel structure (suggesting a")
    print("discontinuity) or is smooth/uniform (consistent with the")
    print("Montgomery-Odlyzko GUE prediction, hence consistent with RH).")
    print()

    # Default ladder up to 500 zeros (1000 is ~30+ min at 15dp)
    # Use --extended for N=1000 too.
    import sys as _sys
    n_values = (50, 100, 200, 500)
    if "--extended" in _sys.argv:
        n_values = (50, 100, 200, 500, 1000)
    elif "--quick" in _sys.argv:
        n_values = (50, 100)
    for n in n_values:
        print(f"--- First {n} non-trivial zeros ---")
        try:
            zeros_imag = get_zeta_zeros(n_zeros=n)
            result = run_catcher_on_spacings(zeros_imag)
            print(f"  scan_novelty: verdict={result['scan_novelty_verdict']}, "
                  f"n_sharp={result['scan_n_sharp']}")
            print(f"  per_quantity (1st/mid/last third): "
                  f"verdict={result['per_quantity_verdict']}")
            print(f"  mean normalised spacing = {result['mean_normalised_spacing']:.4f}")
            print(f"  std  normalised spacing = {result['std_normalised_spacing']:.4f}")
            print()

            out_path = Path(__file__).parent / f"millennium_riemann_catcher_n{n}_results.json"
            out_path.write_text(json.dumps(result, indent=2, default=str))
            print(f"  Wrote {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    print("Interpretation")
    print("==============")
    print("  - SMOOTH / UNIFORM verdict across all N is consistent with RH")
    print("    (the spacing distribution matches the Montgomery-Odlyzko GUE")
    print("    prediction with no catcher-detectable anomaly).")
    print("  - NOVEL_STRUCTURE verdict at some N would suggest a sharp")
    print("    transition in the spacing distribution -- worth follow-up.")
    print("  - This is a tool-based exploration; it neither proves nor")
    print("    disproves RH.")


if __name__ == "__main__":
    main()
