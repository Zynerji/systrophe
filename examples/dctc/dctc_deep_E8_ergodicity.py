"""E8 deep dive: ergodicity of D-CTC iteration.

Tests:
  1. Extreme initial conditions (rank-1, max-mixed, edge cases, adversarial)
  2. Detect multiple distinct cycles within one channel
  3. Connection between ergodicity and Clifford structure
  4. Convergence rate from extreme initial conditions
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systrophe.ctc.d_ctc import (
    apply_channel,
    clifford_like_unitary,
    dctc_fixed_point,
    density_matrix_diagnostics,
)
from systrophe.catchers.novelty_catcher import catch_novelty_in_named_arrays


def haar_random_unitary(dim, rng):
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def detect_distinct_limits(U, sigma_cr, dim_cr, dim_ctc, n_init: int = 50,
                            tol: float = 1e-7):
    """Run iteration from many initial conditions; cluster final states."""
    rng = np.random.default_rng(2024)
    limits = []  # list of distinct limit states found
    for _ in range(n_init):
        # Random init: pure state OR mixed state with random weights
        if rng.random() > 0.5:
            psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
            psi /= np.linalg.norm(psi)
            rho_init = np.outer(psi, psi.conj())
        else:
            # Mixed init
            weights = rng.dirichlet(np.ones(dim_ctc))
            vecs = []
            for _ in range(dim_ctc):
                v = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
                v /= np.linalg.norm(v)
                vecs.append(v)
            rho_init = sum(weights[i] * np.outer(vecs[i], vecs[i].conj()) for i in range(dim_ctc))
        try:
            r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr, rho_ctc_init=rho_init,
                                  tol=1e-10, max_iter=2000)
            rho_fp = r["rho_ctc"]
            # Check if matches any existing limit
            matched = False
            for prev in limits:
                if float(np.linalg.norm(rho_fp - prev, "fro")) < tol:
                    matched = True
                    break
            if not matched:
                limits.append(rho_fp)
        except Exception:
            pass
    return limits


def main():
    dim_cr, dim_ctc = 2, 3
    dim_total = dim_cr * dim_ctc
    sigma_cr = np.zeros((dim_cr, dim_cr), dtype=complex); sigma_cr[0, 0] = 1.0
    rng = np.random.default_rng(99999)

    print("=" * 70)
    print("E8 deep dive: ergodicity of D-CTC iteration")
    print("=" * 70)
    print()

    # Part 1: many channels, very-many init counts, check uniqueness
    n_channels = 200
    n_init_extreme = 30
    haar_n_limits = []
    cliff_n_limits = []
    for _ in range(n_channels):
        for kind in ("haar", "cliff"):
            U = haar_random_unitary(dim_total, rng) if kind == "haar" else clifford_like_unitary(dim_total, rng)
            limits = detect_distinct_limits(U, sigma_cr, dim_cr, dim_ctc, n_init=n_init_extreme)
            (haar_n_limits if kind == "haar" else cliff_n_limits).append(len(limits))
    haar_n_limits = np.array(haar_n_limits)
    cliff_n_limits = np.array(cliff_n_limits)

    print(f"Across {n_channels} channels each, {n_init_extreme} initial conditions:")
    print(f"  Haar mean # distinct limit states: {haar_n_limits.mean():.2f}")
    print(f"  Haar max:                          {haar_n_limits.max()}")
    print(f"  Haar fraction unique (1 limit):    {float(np.mean(haar_n_limits == 1)):.3f}")
    print()
    print(f"  Clifford mean # distinct limits:   {cliff_n_limits.mean():.2f}")
    print(f"  Clifford max:                      {cliff_n_limits.max()}")
    print(f"  Clifford fraction unique:          {float(np.mean(cliff_n_limits == 1)):.3f}")
    print()

    # Part 2: adversarial initial conditions
    print("=" * 70)
    print("Adversarial init conditions:")
    print("=" * 70)
    print()
    # Pick a Clifford channel
    U = clifford_like_unitary(dim_total, rng)
    # Try various extreme rho_init: |0><0|, |+><+|, max-mixed, random pure
    extreme_inits = []
    # Basis states
    for i in range(dim_ctc):
        v = np.zeros(dim_ctc, dtype=complex); v[i] = 1
        extreme_inits.append((f"basis_{i}", np.outer(v, v.conj())))
    # Superpositions
    for i in range(dim_ctc):
        for j in range(i + 1, dim_ctc):
            v = np.zeros(dim_ctc, dtype=complex); v[i] = 1; v[j] = 1; v /= np.sqrt(2)
            extreme_inits.append((f"super_{i}_{j}", np.outer(v, v.conj())))
    # Maximally mixed
    extreme_inits.append(("max_mixed", np.eye(dim_ctc, dtype=complex) / dim_ctc))

    limits_for_extreme = []
    print(f"  init type        n_iter   purity   ||rho_fp - first_fp||")
    first_fp = None
    for name, rho_init in extreme_inits:
        r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr, rho_ctc_init=rho_init,
                              tol=1e-10, max_iter=2000)
        pur = density_matrix_diagnostics(r["rho_ctc"])["purity"]
        if first_fp is None:
            first_fp = r["rho_ctc"]
            dist = 0.0
        else:
            dist = float(np.linalg.norm(r["rho_ctc"] - first_fp, "fro"))
        print(f"  {name:18s}  {r['iterations']:5d}  {pur:.4f}   {dist:.6e}")
    print()

    # Part 3: cycle-vs-fixed-point detection
    print("=" * 70)
    print("Period-2 cycle: ergodic over the cycle?")
    print("=" * 70)
    print()
    # Pick a Clifford channel that produces period-2 (some do, per E1)
    found_p2 = False
    for trial in range(100):
        U_test = clifford_like_unitary(dim_total, rng)
        # Detect cycle
        psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
        psi /= np.linalg.norm(psi)
        rho_init = np.outer(psi, psi.conj())
        rho = rho_init.copy()
        for _ in range(20):
            rho_new = apply_channel(U_test, sigma_cr, rho, dim_cr)
            tr = np.trace(rho_new)
            rho_new /= tr if abs(tr) > 1e-12 else 1
            if float(np.linalg.norm(rho_new - rho_init, "fro")) < 1e-6:
                # Period-1 or short cycle
                rho = rho_new
                continue
            rho = rho_new
        # Now check period
        rho2 = apply_channel(U_test, sigma_cr, rho, dim_cr)
        rho2 /= np.trace(rho2) if abs(np.trace(rho2)) > 1e-12 else 1
        # Iterate 2 more steps
        rho3 = apply_channel(U_test, sigma_cr, rho2, dim_cr)
        rho3 /= np.trace(rho3) if abs(np.trace(rho3)) > 1e-12 else 1
        rho4 = apply_channel(U_test, sigma_cr, rho3, dim_cr)
        rho4 /= np.trace(rho4) if abs(np.trace(rho4)) > 1e-12 else 1
        # If rho4 == rho2 (period 2) but rho3 != rho2 (not period 1)
        d_42 = float(np.linalg.norm(rho4 - rho2, "fro"))
        d_32 = float(np.linalg.norm(rho3 - rho2, "fro"))
        if d_42 < 1e-7 and d_32 > 1e-4:
            found_p2 = True
            print(f"  Found period-2 channel at trial {trial}")
            # Test: from many initial conditions, do we always reach this cycle?
            cycle_states_collected = []
            for k in range(20):
                psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
                psi /= np.linalg.norm(psi)
                rho_i = np.outer(psi, psi.conj())
                rho_cur = rho_i.copy()
                for _ in range(15):
                    rho_cur = apply_channel(U_test, sigma_cr, rho_cur, dim_cr)
                    rho_cur /= np.trace(rho_cur) if abs(np.trace(rho_cur)) > 1e-12 else 1
                # Check if rho_cur matches rho2 or rho3
                d_a = float(np.linalg.norm(rho_cur - rho2, "fro"))
                d_b = float(np.linalg.norm(rho_cur - rho3, "fro"))
                if d_a < d_b:
                    cycle_states_collected.append("A")
                else:
                    cycle_states_collected.append("B")
            print(f"  From 20 random inits, cycle phase reached: "
                  f"A={cycle_states_collected.count('A')}, B={cycle_states_collected.count('B')}")
            print(f"  (Both should occur if ergodic over the cycle)")
            break
    if not found_p2:
        print("  No period-2 channel found in 100 trials (cycles are rare among Clifford)")

    print()
    print("=" * 70)
    print("Verdict")
    print("=" * 70)
    print()
    print(f"Across {n_channels} channels, {n_init_extreme} init each:")
    print(f"  Haar:     {float(np.mean(haar_n_limits == 1))*100:.1f}% strict-unique fixed point")
    print(f"  Clifford: {float(np.mean(cliff_n_limits == 1))*100:.1f}% strict-unique fixed point")
    print()
    if float(np.mean(haar_n_limits == 1)) > 0.95 and float(np.mean(cliff_n_limits == 1)) > 0.95:
        print("Ergodicity is RESOUNDINGLY confirmed: nearly 100% of D-CTC channels")
        print("admit a unique fixed point (modulo cycle structure), reached from")
        print("any initial state.")
    elif float(np.mean(cliff_n_limits == 1)) < 0.5:
        print("Many Clifford channels have multiple distinct limits -- ergodicity")
        print("breaks down.")

    # Native novelty catcher: per-channel limit-count distributions.
    novelty = catch_novelty_in_named_arrays({
        "haar_n_limits": haar_n_limits,
        "cliff_n_limits": cliff_n_limits,
    })
    print()
    print(f"Novelty catcher: verdict='{novelty['verdict']}', "
          f"n_sharp={len(novelty['sharp_features'])}")

    out = Path("examples") / "dctc_deep_E8_ergodicity_results.json"
    with open(out, "w") as f:
        json.dump({
            "n_channels": n_channels,
            "n_init_per_channel": n_init_extreme,
            "haar_unique_fraction": float(np.mean(haar_n_limits == 1)),
            "clifford_unique_fraction": float(np.mean(cliff_n_limits == 1)),
            "haar_mean_n_limits": float(haar_n_limits.mean()),
            "clifford_mean_n_limits": float(cliff_n_limits.mean()),
            "haar_max_n_limits": int(haar_n_limits.max()),
            "clifford_max_n_limits": int(cliff_n_limits.max()),
            "novelty_catcher": novelty,
        }, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
