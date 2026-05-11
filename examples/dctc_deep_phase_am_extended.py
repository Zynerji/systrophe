"""D-CTC Phase AM extension: robustness across U(delta) constructions.

The original Phase AM result -- D-CTC amplification is bimodal at
delta = 0 and delta = pi, independent of physical CTC content -- may
depend on the specific U(delta) construction. Here we test the finding
across FOUR alternative parametrisations, each physically motivated:

Construction A: original (Tipler-sinusoid + delta-dependent coupling)
Construction B: linear interpolation U(delta) = polar((1-t)U_0 + t*U_pi)
Construction C: anomaly-inflow generator (Z_3 cycle shift with
                delta-dependent branch phases)
Construction D: Floquet propagator under periodic drive at omega = 1

If all four give bimodal D-CTC amplification co-locating at the
extrema of physical CTC content, the independence finding is robust.
If only A gives bimodal, the original was an artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.linalg

from systrophe.anomaly_inflow import z3_branch_twists
from systrophe.d_ctc import dctc_fixed_point, density_matrix_diagnostics
from systrophe.floquet_mobius import z3_cycle_shift
from systrophe.pair import SystrophePair
from systrophe.sinusoid import TiplerSinusoid


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def physical_ctc_content(delta, omega=1.0, R=1.0, r_min=1.05, r_max=10.0,
                           n_grid=200):
    a = omega * R
    s1 = TiplerSinusoid(R=R, a=a, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=R, a=a, A=1.0, delta=delta)
    pair = SystrophePair(s1=s1, s2=s2)
    rs = np.linspace(r_min, r_max, n_grid)
    L = pair.L(rs)
    return float(np.sum(L ** 2) * (r_max - r_min) / n_grid)


def trace_distance(rho_a, rho_b):
    diff = rho_a - rho_b
    eigs = np.linalg.eigvalsh(0.5 * (diff + diff.conj().T))
    return 0.5 * float(np.sum(np.abs(eigs)))


def dctc_amplification(U, dim_cr=2, eps=0.1, n_init=5):
    dim_total = U.shape[0]
    dim_ctc = dim_total // dim_cr
    sigma_a = np.zeros((dim_cr, dim_cr), dtype=complex); sigma_a[0, 0] = 1.0
    sigma_b = np.zeros((dim_cr, dim_cr), dtype=complex)
    sigma_b[0, 0] = 1 - eps; sigma_b[1, 1] = eps
    rng = np.random.default_rng(11)
    best_amp = 0.0
    for _ in range(n_init):
        psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
        psi = psi / np.linalg.norm(psi)
        rho_init = np.outer(psi, psi.conj())
        try:
            r_a = dctc_fixed_point(U, sigma_a, dim_cr=dim_cr, rho_ctc_init=rho_init,
                                     tol=1e-10, max_iter=2000)["rho_ctc"]
            r_b = dctc_fixed_point(U, sigma_b, dim_cr=dim_cr, rho_ctc_init=rho_init,
                                     tol=1e-10, max_iter=2000)["rho_ctc"]
            amp = trace_distance(r_a, r_b)
            best_amp = max(best_amp, amp)
        except Exception:
            continue
    return best_amp


def polar_unitary(M):
    """Project a square matrix to the nearest unitary via SVD."""
    U, _, Vh = np.linalg.svd(M)
    return U @ Vh


# -----------------------------------------------------------------
# Four constructions
# -----------------------------------------------------------------

def construction_A_original(delta, gamma_eff=0.0, dim_cr=2):
    """Original Phase AM construction."""
    dim_total = dim_cr * 3
    H = np.zeros((dim_total, dim_total), dtype=complex)
    for cr in range(dim_cr):
        for b in range(3):
            i = cr * 3 + b
            alpha = 1.5
            H[i, i] = np.cos(alpha * delta + 2 * np.pi * b / 3) + 0.3 * cr
    coupling = 0.5 * np.sin(delta)
    for b1 in range(3):
        for b2 in range(3):
            if b1 != b2:
                H[0 * 3 + b1, 1 * 3 + b2] = coupling * np.exp(1j * 2 * np.pi * (b2 - b1) / 3)
                H[1 * 3 + b2, 0 * 3 + b1] = H[0 * 3 + b1, 1 * 3 + b2].conj()
    perm = np.array([2, 5, 1, 4, 0, 3])
    Pref = np.zeros((6, 6), dtype=complex)
    for i, p in enumerate(perm):
        Pref[i, p] = 1.0
    Dref = np.diag([1, 1j, -1, -1j, 1, 1j])
    U_clifford = Pref @ Dref
    return scipy.linalg.expm(-1j * H) @ U_clifford


def construction_B_interpolation(delta, gamma_eff=0.0, dim_cr=2):
    """Linear interpolation between two reference unitaries."""
    dim_total = dim_cr * 3
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(22)
    perm_a = rng_a.permutation(dim_total)
    perm_b = rng_b.permutation(dim_total)
    P_a = np.zeros((dim_total, dim_total), dtype=complex)
    P_b = np.zeros((dim_total, dim_total), dtype=complex)
    for i in range(dim_total):
        P_a[i, perm_a[i]] = 1.0
        P_b[i, perm_b[i]] = 1.0
    D_a = np.diag(rng_a.choice([1, -1, 1j, -1j], dim_total))
    D_b = np.diag(rng_b.choice([1, -1, 1j, -1j], dim_total))
    U_a = (P_a @ D_a).astype(complex)
    U_b = (P_b @ D_b).astype(complex)
    t = (delta % (2 * np.pi)) / (2 * np.pi)
    # Smooth periodic interpolation: t in [0, 0.5] -> U_a -> U_b, [0.5, 1] -> U_b -> U_a
    if t <= 0.5:
        s = 2 * t
    else:
        s = 2 * (1 - t)
    M = (1 - s) * U_a + s * U_b
    return polar_unitary(M)


def construction_C_anomaly_flow(delta, gamma_eff=0.0, dim_cr=2):
    """Anomaly-inflow generator: Z_3 cycle shift with delta-dependent branch phases."""
    twists = z3_branch_twists(gamma_eff)
    H_branch = np.diag(twists * 2 * np.pi + delta * np.array([0, 1, 2]) / 3)
    # CR-CTC coupling via off-diagonal entanglement
    dim_total = dim_cr * 3
    S = z3_cycle_shift()
    # Joint: H = sigma_x_cr (x) H_branch + delta * cycle-shift block
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    H_full = np.kron(np.eye(dim_cr, dtype=complex), H_branch) + delta * 0.3 * np.kron(sigma_x, S + S.conj().T)
    U_evol = scipy.linalg.expm(-1j * H_full)
    # Base Clifford
    perm = np.array([5, 0, 3, 1, 4, 2])
    Pref = np.zeros((6, 6), dtype=complex)
    for i, p in enumerate(perm):
        Pref[i, p] = 1.0
    Dref = np.diag([1, -1, 1j, -1j, 1, -1j])
    return U_evol @ Pref @ Dref


def construction_D_floquet(delta, gamma_eff=0.0, dim_cr=2,
                              omega=1.0, n_steps=50):
    """Floquet propagator under time-periodic drive with phase shift delta."""
    dim_total = dim_cr * 3
    twists = z3_branch_twists(gamma_eff)
    H_static = np.kron(np.eye(dim_cr, dtype=complex), np.diag(twists * 2 * np.pi))
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    H_drive_op = np.kron(sigma_x, z3_cycle_shift() + z3_cycle_shift().conj().T)
    T = 2 * np.pi / omega
    dt = T / n_steps
    U = np.eye(dim_total, dtype=complex)
    for k in range(n_steps):
        t = (k + 0.5) * dt
        # Drive: sin(omega t + delta) -- delta enters as phase shift
        drive = 0.5 * np.sin(omega * t + delta)
        H_t = H_static + drive * H_drive_op
        U = scipy.linalg.expm(-1j * H_t * dt) @ U
    return U


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    print("=" * 70)
    print("Phase AM-extended: 4 U(delta) constructions")
    print("=" * 70)
    print()

    deltas = np.linspace(0, 2 * np.pi, 21)
    constructions = {
        "A_original":      construction_A_original,
        "B_interpolation": construction_B_interpolation,
        "C_anomaly_flow":  construction_C_anomaly_flow,
        "D_floquet":       construction_D_floquet,
    }

    results = {}
    physical = [physical_ctc_content(float(d)) for d in deltas]

    print(f"Sweep delta in [0, 2pi] with {len(deltas)} grid points")
    print()

    for name, fn in constructions.items():
        amps = []
        for d in deltas:
            U = fn(float(d), gamma_eff=0.0, dim_cr=2)
            amp = dctc_amplification(U, dim_cr=2)
            amps.append(amp)
        amps = np.array(amps)
        idx_max = int(np.argmax(amps))
        idx_min = int(np.argmin(amps))
        # Find peaks: locations where amp > 0.5 * max_amp
        peaks_idx = [i for i in range(len(deltas))
                      if amps[i] > 0.5 * amps.max() and amps[i] > 0.01]
        # Correlation
        pearson_phys = float(np.corrcoef(physical, amps)[0, 1])
        log_amps = np.log(np.maximum(amps, 1e-12))
        log_phys = np.log(np.maximum(np.array(physical), 1e-12))
        pearson_log = float(np.corrcoef(log_phys, log_amps)[0, 1])

        results[name] = {
            "deltas": deltas.tolist(),
            "physical_ctcs": physical,
            "amplifications": amps.tolist(),
            "max_amp": float(amps.max()),
            "min_amp": float(amps.min()),
            "max_delta": float(deltas[idx_max]),
            "min_delta": float(deltas[idx_min]),
            "n_peaks": len(peaks_idx),
            "peak_deltas": [float(deltas[i]) for i in peaks_idx],
            "pearson_phys_vs_amp": pearson_phys,
            "pearson_log_vs_log": pearson_log,
        }

        print(f"--- {name} ---")
        print(f"  max amp: {amps.max():.4f} at delta = {deltas[idx_max]:.3f}")
        print(f"  min amp: {amps.min():.4f} at delta = {deltas[idx_min]:.3f}")
        print(f"  n peaks (amp > 50% of max): {len(peaks_idx)}")
        if len(peaks_idx) > 0:
            print(f"  peak deltas: {[f'{deltas[i]:.3f}' for i in peaks_idx]}")
        print(f"  Pearson(physical_CTC, amp): {pearson_phys:+.3f}")
        print(f"  Pearson(log phys, log amp): {pearson_log:+.3f}")
        print()

    print("=" * 70)
    print("Synthesis")
    print("=" * 70)
    print()
    print("Does the bimodal pattern (peaks at delta = 0 AND pi) repeat across all four constructions?")
    print()
    for name, r in results.items():
        # Check if peaks are near 0 and pi
        peaks = r["peak_deltas"]
        has_0   = any(abs(p) < 0.5 or abs(p - 2 * np.pi) < 0.5 for p in peaks)
        has_pi  = any(abs(p - np.pi) < 0.5 for p in peaks)
        verdict = "BIMODAL (peaks at 0 + pi)" if (has_0 and has_pi) else (
            "PEAK NEAR 0 ONLY" if has_0 else (
            "PEAK NEAR PI ONLY" if has_pi else "OTHER STRUCTURE"))
        print(f"  {name:18s}: {verdict}")
    print()

    # Compute overall correlation
    pearsons = [r["pearson_phys_vs_amp"] for r in results.values()]
    print(f"Pearson correlations across constructions: {[f'{p:+.3f}' for p in pearsons]}")
    mean_pearson = float(np.mean(pearsons))
    print(f"Mean Pearson: {mean_pearson:+.3f}")
    if abs(mean_pearson) < 0.3:
        print()
        print("ROBUST INDEPENDENCE: across all four constructions, D-CTC")
        print("amplification is essentially uncorrelated with physical CTC content.")
        print("The chronology-protection x D-CTC independence finding is")
        print("ROBUST across U(delta) parametrisation.")
    elif mean_pearson > 0.5:
        print()
        print("STRONG POSITIVE: D-CTC amplification tracks physical CTC content.")
        print("Chronology protection DOES suppress D-CTC amplification.")
    elif mean_pearson < -0.5:
        print()
        print("STRONG NEGATIVE: D-CTC amplification anti-correlates with physical CTC.")
    else:
        print()
        print(f"Intermediate correlation (mean {mean_pearson:+.3f}).")

    out = Path("examples") / "dctc_deep_phase_am_extended_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
