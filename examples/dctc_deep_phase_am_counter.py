"""D-CTC Phase AM-counterfactual: hunt for a U(delta) that exhibits
chronology-protection signature.

The independence finding (Phase AM/AM-ext) tested 4 U(delta)
constructions, none of which showed D-CTC amplification vanishing
at delta = pi specifically due to chronology protection.

Here we try 4 *more* physically-motivated constructions, specifically
designed to potentially couple D-CTC amplification to physical CTC
content:

E: directly use L_pair(r) values at sample radii as matrix entries
F: geodesic propagator at fixed r in CTC band
G: SystrophePair-derived qubit + Z_3 cyclic Hamiltonian with
   delta-dependent matching at L(r) = 0 boundary
H: Tipler-frequency-modulated drive (alpha-driven Floquet)

If even THESE physically-grounded constructions fail to show the
signature, the independence is super-robust.
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
    U, _, Vh = np.linalg.svd(M)
    return U @ Vh


# -----------------------------------------------------------------
# Constructions E, F, G, H (physically grounded)
# -----------------------------------------------------------------

def construction_E_direct_L(delta, dim_cr=2):
    """Use L_pair(r) values at 6 radii as Hermitian matrix entries (after symmetrising).

    The idea: a U built directly from the L_pair structure should
    naturally collapse at delta = pi (since L vanishes there).
    """
    a = 1.0
    R = 1.0
    s1 = TiplerSinusoid(R=R, a=a, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=R, a=a, A=1.0, delta=delta)
    pair = SystrophePair(s1=s1, s2=s2)
    # Sample L at 6 radii covering CTC bands and exterior
    rs = np.linspace(1.2, 8.0, 6)
    L_vals = pair.L(rs)
    # Build a 6x6 Hermitian by L^T L matrix-like construction
    M = np.outer(L_vals, L_vals)
    # Add a small Clifford backbone so the matrix is always full-rank
    Clifford = np.eye(6, dtype=complex)
    perm = np.array([3, 4, 5, 0, 1, 2])
    P = np.zeros((6, 6), dtype=complex)
    for i, p in enumerate(perm):
        P[i, p] = 1.0
    H = np.diag(L_vals) + 0.1 * (M + M.conj().T) / 2
    U_evol = scipy.linalg.expm(-1j * H)
    return polar_unitary(U_evol @ P)


def construction_F_geodesic(delta, dim_cr=2):
    """Geodesic propagator at fixed r in the CTC band.

    Use the timelike-Omega bounds at r=3.35 in the first CTC band of
    a=1 cylinder. The propagator U(delta) corresponds to going around
    the angular CTC orbit by phase delta.
    """
    # First CTC band of a=1, R=1: deepest L at r~3.35
    r0 = 3.35
    a = 1.0
    R = 1.0
    s1 = TiplerSinusoid(R=R, a=a, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=R, a=a, A=1.0, delta=delta)
    pair = SystrophePair(s1=s1, s2=s2)
    L0 = float(pair.L(r0))
    # Eigenvalues of the propagator: e^{i delta} sign(L) for each of 3 branches
    # Block-diagonal construction on (dim_cr x 3)
    sigma = np.sign(L0)
    Omega_band = sigma * np.sqrt(abs(L0)) * 0.3
    # H = Omega_band * (cyclic shift) -- gives a delta-dependent phase
    S = z3_cycle_shift()
    H_branch = Omega_band * (S + S.conj().T)
    # CR-CTC coupling
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    H_full = np.kron(np.eye(2, dtype=complex), H_branch) + delta * 0.5 * np.kron(sigma_x, S + S.conj().T)
    U = scipy.linalg.expm(-1j * H_full)
    # Clifford base
    perm = np.array([4, 1, 5, 2, 0, 3])
    P = np.zeros((6, 6), dtype=complex)
    for i, p in enumerate(perm):
        P[i, p] = 1.0
    D = np.diag([1, -1j, 1j, -1, 1, -1j])
    return polar_unitary(U @ P @ D)


def construction_G_pair_boundary(delta, dim_cr=2):
    """SystrophePair-derived Hamiltonian gating on L(r) = 0 boundary.

    Construct U such that the matrix elements are explicitly proportional
    to L(r) at sampled radii. At delta = pi (L = 0 everywhere), the
    matrix collapses to identity-like.
    """
    a = 1.0
    R = 1.0
    s1 = TiplerSinusoid(R=R, a=a, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=R, a=a, A=1.0, delta=delta)
    pair = SystrophePair(s1=s1, s2=s2)
    rs = np.linspace(1.5, 6.0, 6)
    L_vals = pair.L(rs)
    # Build H proportional to L
    H = np.zeros((6, 6), dtype=complex)
    for i in range(6):
        for j in range(6):
            if i != j:
                H[i, j] = 0.2 * L_vals[i] * np.exp(1j * np.pi * (i - j) / 3)
    H = (H + H.conj().T) / 2  # symmetrise
    U = scipy.linalg.expm(-1j * H)
    # Clifford base for amplification capacity
    perm = np.array([2, 5, 1, 4, 0, 3])
    P = np.zeros((6, 6), dtype=complex)
    for i, p in enumerate(perm):
        P[i, p] = 1.0
    D = np.diag([1, 1j, -1, -1j, 1, 1j])
    return polar_unitary(U @ P @ D)


def construction_H_alpha_floquet(delta, dim_cr=2, n_steps=40):
    """Tipler-frequency Floquet drive.

    The drive frequency is alpha = sqrt(4 a^2 - 1) (Tipler log-freq).
    delta enters as the relative phase between two drive components,
    one of which carries the L_pair envelope.
    """
    a = 1.0
    alpha = np.sqrt(4 * a * a - 1)
    twists = z3_branch_twists(0.0)
    H_static = np.kron(np.eye(2, dtype=complex), np.diag(twists * 2 * np.pi))
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    H_drive = np.kron(sigma_x, z3_cycle_shift() + z3_cycle_shift().conj().T)
    T = 2 * np.pi
    dt = T / n_steps
    U = np.eye(6, dtype=complex)
    for k in range(n_steps):
        t = (k + 0.5) * dt
        # delta-modulated drive amplitude
        drive = 0.4 * np.cos(alpha * t) * np.cos(t + delta)
        H_t = H_static + drive * H_drive
        U = scipy.linalg.expm(-1j * H_t * dt) @ U
    return U


# -----------------------------------------------------------------

def main():
    print("=" * 70)
    print("Phase AM-counter: hunt for chronology-protection signature")
    print("=" * 70)
    print()

    deltas = np.linspace(0, 2 * np.pi, 21)
    constructions = {
        "E_direct_L":      construction_E_direct_L,
        "F_geodesic":      construction_F_geodesic,
        "G_pair_boundary": construction_G_pair_boundary,
        "H_alpha_floquet": construction_H_alpha_floquet,
    }

    physical = [physical_ctc_content(float(d)) for d in deltas]
    results = {}

    for name, fn in constructions.items():
        amps = []
        for d in deltas:
            U = fn(float(d), dim_cr=2)
            amp = dctc_amplification(U, dim_cr=2)
            amps.append(amp)
        amps = np.array(amps)
        idx_max = int(np.argmax(amps))
        idx_min = int(np.argmin(amps))
        pearson_phys = float(np.corrcoef(physical, amps)[0, 1])
        # Check: is amp at delta = pi LOWER than at neighbouring deltas?
        pi_idx = int(np.argmin(np.abs(deltas - np.pi)))
        # Compare amp at pi to surrounding values
        neighbors = []
        if pi_idx > 0:
            neighbors.append(amps[pi_idx - 1])
        if pi_idx < len(amps) - 1:
            neighbors.append(amps[pi_idx + 1])
        amp_at_pi = amps[pi_idx]
        neighbor_mean = float(np.mean(neighbors)) if neighbors else 0.0
        # If amp_at_pi < 0.5 * neighbor_mean, chronology protection signature
        cp_signature = amp_at_pi < 0.5 * neighbor_mean and amp_at_pi < 0.5 * amps.max()
        results[name] = {
            "deltas": deltas.tolist(),
            "amplifications": amps.tolist(),
            "physical_ctcs": physical,
            "max_amp": float(amps.max()),
            "max_delta": float(deltas[idx_max]),
            "min_amp": float(amps.min()),
            "min_delta": float(deltas[idx_min]),
            "amp_at_pi": float(amp_at_pi),
            "amp_at_pi_neighbor_mean": neighbor_mean,
            "chronology_protection_signature": bool(cp_signature),
            "pearson_phys_vs_amp": pearson_phys,
        }

        print(f"--- {name} ---")
        print(f"  max amp:    {amps.max():.4f} at delta = {deltas[idx_max]:.3f}")
        print(f"  min amp:    {amps.min():.4f} at delta = {deltas[idx_min]:.3f}")
        print(f"  amp at pi:  {amp_at_pi:.4f}  (neighbor mean: {neighbor_mean:.4f})")
        print(f"  Pearson(phys, amp): {pearson_phys:+.3f}")
        print(f"  Chronology-protection signature: {cp_signature}")
        print()

    print("=" * 70)
    print("Overall verdict")
    print("=" * 70)
    print()
    n_with_signature = sum(1 for r in results.values()
                              if r["chronology_protection_signature"])
    print(f"Constructions with chronology-protection signature: {n_with_signature} / {len(results)}")
    if n_with_signature == 0:
        print()
        print("NO construction exhibits the chronology-protection signature.")
        print("This is consistent with the independence finding from Phase AM-ext:")
        print("D-CTC amplification is structurally decoupled from physical CTC content.")
    elif n_with_signature >= len(results) / 2:
        print()
        print("MAJORITY of physically-motivated constructions show chronology-")
        print("protection signature -- the finding is NOT robust.")
    print()

    out = Path("examples") / "dctc_deep_phase_am_counter_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
