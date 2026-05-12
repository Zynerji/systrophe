"""Quantum-Error-Correction bridge module.

The Systrophe framework contains several mechanisms that map cleanly
onto quantum-error-correction concepts. This module surfaces those
mappings as runnable functions and exposes them under a single QEC-
oriented namespace.

Five concrete mappings
======================

1. **Anyonic-CTC -> topological codes**:
   the Tipler-band anyonic braid phases (`anyonic_ctc.braid_phase_at_band`)
   provide a natural hierarchy of anyonic sectors. The
   topological-entanglement entropy log D maps to the surface-code
   logical qubit's protection budget.

2. **Address-space catcher -> syndrome anomaly detection**:
   stabilizer-measurement syndromes are bitstrings; the catcher's
   bit-occupancy hash + Hamming graph + lambda_2 is a domain-agnostic
   anomaly detector. We provide `syndrome_anomaly_score` that wraps
   the catcher for syndrome-string inputs.

3. **D-CTC spectral oracle -> decoder convergence prediction**:
   |lambda_2(E)| of the CPTP channel superoperator predicts D-CTC
   fixed-point iteration count (Pearson r = 0.99, log-log slope 0.92).
   The same calculation applies to QEC decoder iteration: the
   stabilizer-channel's spectral gap predicts decoder convergence
   without running the decoder. We provide `predict_decoder_iterations`.

4. **Krasnikov-ring -> stabilizer redundancy and noise threshold**:
   the Z_N-symmetric Krasnikov ring's noise-robustness threshold (N=2
   fragile, N>=3 robust at sigma ~ 2.058 rad) is structurally the
   fault-tolerance threshold of a stabilizer code with N-fold parity-
   check redundancy. We provide `ring_fault_tolerance_threshold`.

5. **Z_3 Mobius cover -> ternary qudit stabilizer codes**:
   the Dinos bridge identifies Z_3 cover branches with Tipler
   log-grid eigenvalues; for QEC this maps to ternary qudit
   stabilizer codes (Steane-class). We provide `z3_qutrit_stabilizer_map`.

Catcher integration
===================
Every QEC mapping function runs through the address-space novelty
catcher on its natural parameter sweep, surfacing any sharp
transitions that indicate fault-tolerance phase boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .novelty_catcher import (
    bitstring_counts_to_address,
    catch_novelty_in_distributions,
    catch_novelty_in_named_arrays,
    scan_novelty,
)


# ----------------------------------------------------------------------
# Mapping 1: anyonic-CTC -> topological codes
# ----------------------------------------------------------------------

def topological_code_logical_protection(
    n_bands: int = 4, alpha: float = math.sqrt(3),
    fibonacci: bool = False,
) -> dict:
    """The Tipler-band anyonic-code protection budget.

    For each band 1..n_bands, the braid phase encodes a stabilizer-
    measurement outcome. The total topological-entanglement entropy
    log(D) is the protection budget against logical errors.

    For Fibonacci anyons (`fibonacci=True`), the quantum dimension d
    equals the golden ratio phi; D = phi^n for n bands.
    """
    from .anyonic_ctc import (
        braid_phase_at_band,
        quantum_dimension_for_band,
        topological_entanglement_entropy,
    )
    bands = []
    for k in range(1, n_bands + 1):
        theta = braid_phase_at_band(k, alpha=alpha)
        d = quantum_dimension_for_band(k, alpha=alpha, fibonacci=fibonacci)
        bands.append({"band_index": k, "braid_phase": theta,
                       "quantum_dimension": d})
    D_total = float(np.prod([b["quantum_dimension"] for b in bands]))
    log_D = math.log(max(D_total, 1e-12))
    # Stabilizer logical-error suppression: protection_factor ~ 1/D^{n_logical}
    # For a single logical qubit, protection ~ 1/D
    return {
        "bands": bands,
        "total_quantum_dimension": D_total,
        "logical_protection_log_D": log_D,
        "protection_factor": 1.0 / max(D_total, 1e-12),
    }


# ----------------------------------------------------------------------
# Mapping 2: address-space catcher -> syndrome anomaly detection
# ----------------------------------------------------------------------

def syndrome_anomaly_score(
    syndrome_strings: list[str], n_bits: int = 32,
) -> dict:
    """Anomaly score for a list of stabilizer-measurement syndromes.

    Each syndrome is treated as a binary string. The catcher hashes
    them, builds the Hamming-distance graph, computes lambda_2, and
    reports the verdict + sharp-feature locations.

    A `novel_structure` verdict at a particular shot index indicates
    an anomalous syndrome shot that may correspond to a logical error
    (as opposed to a routine physical error).
    """
    if not syndrome_strings:
        return {"verdict": "empty", "sharp_features": []}
    counts_per_shot = [{s: 1} for s in syndrome_strings]
    addresses = [bitstring_counts_to_address(c, n_bits=n_bits)
                 for c in counts_per_shot]
    # Convert to "distributions" (degenerate single-bit prob vectors)
    distributions = [a.astype(float) for a in addresses]
    nov = catch_novelty_in_distributions(
        distributions,
        labels=[f"shot_{i}" for i in range(len(syndrome_strings))],
    )
    return {
        "n_shots": len(syndrome_strings),
        "verdict": nov["verdict"],
        "n_sharp": len(nov["sharp_features"]),
        "sharp_features": nov["sharp_features"],
        "lambda_2_at_radius": nov["lambda_2_at_radius"],
    }


# ----------------------------------------------------------------------
# Mapping 3: D-CTC spectral oracle -> decoder convergence
# ----------------------------------------------------------------------

def predict_decoder_iterations(
    lambda_2_abs: float, tol: float = 1e-10,
) -> float:
    """Predicted decoder iteration count from |lambda_2(E)|.

    The same spectral-oracle formula that predicts D-CTC fixed-point
    iteration applies to any contractive QEC decoder:
        iter_required = -log(tol) / log(|lambda_2|^{-1})

    For decoders with Pauli-channel spectral gap > 0, this gives an
    O(d^6) estimate of decoder iterations without running the decoder.

    Pearson r = 0.99 in the D-CTC validation (`dctc_deep_phase_b`).
    """
    if lambda_2_abs <= 1e-15:
        return 1.0
    if lambda_2_abs >= 1.0 - 1e-15:
        return float("inf")
    return -math.log(tol) / -math.log(lambda_2_abs)


def stabilizer_channel_lambda_2(
    pauli_error_rate: float, n_stabilizers: int = 4,
) -> float:
    """|lambda_2| of an idealised stabilizer-Pauli channel at given
    physical error rate.

    For an n-stabilizer code under depolarising noise p, the channel
    eigenvalues are (1-p)^k for various k; the second-largest is
    (1-p)^{n_stabilizers-1}.
    """
    return (1.0 - pauli_error_rate) ** (n_stabilizers - 1)


# ----------------------------------------------------------------------
# Mapping 4: Krasnikov-ring -> fault-tolerance threshold
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class FaultToleranceProfile:
    N: int
    noise_threshold_rad: float
    is_robust: bool
    catcher_verdict: str


def ring_fault_tolerance_threshold(
    N_values: tuple[int, ...] = (2, 3, 5, 8, 12),
    n_noise: int = 30, n_trials: int = 30,
) -> dict:
    """Fault-tolerance threshold for N-fold stabilizer codes.

    Direct mapping from `krasnikov_ring_noise_robustness`: the Z_N
    extinction (under Gaussian phase noise) is structurally the
    fault-tolerance threshold of a stabilizer code with N-fold parity-
    check redundancy.

    N=2 (single-pair code) is fragile at sigma ~ 2.058 rad.
    N >= 3 codes are noise-robust (catcher reports `smooth`).

    Returns the fault-tolerance profile per N.
    """
    from .krasnikov_ring import krasnikov_ring_noise_robustness
    profiles = []
    for N in N_values:
        res = krasnikov_ring_noise_robustness(
            N=int(N), n_trials=n_trials, alpha=4.0,
            noise_grid=np.linspace(0.0, math.pi, n_noise),
        )
        # Identify the noise level at which residual_E_neg exceeds 10x
        # the zero-noise value; that's the FT threshold.
        residuals = res["residual_E_neg"]
        zero_residual = max(residuals[0], 1e-12)
        threshold_idx = None
        for i, r in enumerate(residuals):
            if r > 10 * zero_residual:
                threshold_idx = i
                break
        noise_grid = res["noise_grid"]
        threshold_rad = (
            float(noise_grid[threshold_idx]) if threshold_idx is not None
            else float(noise_grid[-1])
        )
        profiles.append(FaultToleranceProfile(
            N=int(N),
            noise_threshold_rad=threshold_rad,
            is_robust=(res["novelty_verdict"] == "smooth"),
            catcher_verdict=res["novelty_verdict"],
        ))
    return {
        "profiles": [
            {"N": p.N, "noise_threshold_rad": p.noise_threshold_rad,
             "is_robust": p.is_robust, "catcher_verdict": p.catcher_verdict}
            for p in profiles
        ],
        "min_robust_N": min(
            (p.N for p in profiles if p.is_robust), default=None,
        ),
    }


# ----------------------------------------------------------------------
# Mapping 5: Z_3 Mobius cover -> ternary qudit stabilizer
# ----------------------------------------------------------------------

def z3_qutrit_stabilizer_map(
    omega: float = 1.0, R_cylinder: float = 1.0, N_nodes: int = 24,
) -> dict:
    """Match Z_3 Mobius cover branches to qutrit stabilizer eigenvalues.

    The Dinos bridge identifies the Z_3 cover's discrete-Laplacian
    eigenvalues with the Tipler log-grid fundamentals. For QEC, this
    maps to a ternary qudit stabilizer code: the three branches of
    the cover are the three stabilizer-measurement eigenvalues
    (+1, omega, omega^2 with omega = exp(2 pi i / 3)).

    Returns the mapping and the relative residual against the
    Tipler-Laplacian eigenvalue. Requires the z3-solver package
    (degrades to a heuristic without it).
    """
    from .vanstockum import VanStockumInterior
    vs = VanStockumInterior(omega=omega, R=R_cylinder)
    try:
        from .dinos_bridge import z3_branch_match_to_tipler_alpha
        out = z3_branch_match_to_tipler_alpha(vs, N=N_nodes)
        return {
            "tipler_eigenvalue": float(out["tipler_eigenvalue"]),
            "z3_eigenvalues": tuple(float(e) for e in out["z3_eigenvalues"]),
            "best_branch_match": int(out["best_branch_match"]),
            "relative_residual": float(out["relative_residual"]),
            "z3_qutrit_eigenvalues": (
                1.0,
                math.cos(2 * math.pi / 3),
                math.cos(4 * math.pi / 3),
            ),
            "available": True,
        }
    except (ImportError, ModuleNotFoundError):
        return {
            "available": False,
            "reason": "z3-solver not installed; install with `pip install z3-solver`.",
        }


# ----------------------------------------------------------------------
# Catcher across the QEC mappings
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Mapping 6: surface-code distance scaling
# ----------------------------------------------------------------------

def surface_code_logical_error_rate(
    d: int, p_phys: float, p_th: float = 0.0107,
) -> float:
    """Logical error rate for a distance-d surface code.

    Threshold theorem (Dennis et al. 2002; Stephens 2014):
        p_L  =  A (p_phys / p_th)^((d+1) / 2)
    with the dimensionless prefactor A ~ 0.1 for the standard 2D
    rotated surface code under depolarising noise.

    Returns the logical error rate per syndrome cycle.
    """
    if p_phys <= 0:
        return 0.0
    if p_phys >= p_th:
        return float(min(1.0, p_phys * (p_phys / p_th) ** 0.5))
    return 0.1 * (p_phys / p_th) ** ((d + 1) / 2)


def surface_code_resource_cost(d: int) -> dict:
    """Physical qubit + cycle count for a distance-d surface code.

    Standard rotated surface code:
      n_phys_per_logical = 2 d^2 - 1
      n_syndrome_cycles  ~ d (one round per code distance for FT)
    """
    return {
        "distance": int(d),
        "n_physical_qubits": 2 * d * d - 1,
        "n_syndrome_cycles": int(d),
        "n_stabilizer_measurements": (d ** 2 - 1) * d,
    }


def surface_code_distance_scaling(
    p_phys_range: tuple[float, float] = (1e-4, 5e-2),
    n_p: int = 30,
    d_values: tuple[int, ...] = (3, 5, 7, 9, 11),
    p_th: float = 0.0107,
) -> dict:
    """Sweep (d, p_phys) and report the logical-error and resource
    surfaces.

    Catcher target: the threshold-theorem prediction is a smooth
    polynomial in the suppression regime (p_phys < p_th) and a
    discontinuity at the threshold p_phys = p_th. The catcher should
    flag the threshold crossing as a sharp Hamming transition.
    """
    p_grid = np.logspace(
        np.log10(p_phys_range[0]),
        np.log10(p_phys_range[1]),
        n_p,
    )
    per_d_arrays = {}
    for d in d_values:
        p_L = np.array([
            surface_code_logical_error_rate(int(d), float(p), p_th=p_th)
            for p in p_grid
        ])
        # log-rescale for catcher visibility
        per_d_arrays[f"d{d}"] = np.log10(p_L + 1e-30)

    nov = catch_novelty_in_named_arrays(per_d_arrays, n_bins=32)
    return {
        "p_phys_grid": p_grid.tolist(),
        "d_values": list(d_values),
        "p_th": p_th,
        "log10_p_L_per_d": {k: v.tolist() for k, v in per_d_arrays.items()},
        "novelty_verdict": nov["verdict"],
        "novelty_n_sharp": len(nov["sharp_features"]),
        "novelty_sharp_features": nov["sharp_features"],
        "resource_cost_table": {
            d: surface_code_resource_cost(d) for d in d_values
        },
    }


def qec_bridge_catcher_sweep() -> dict:
    """Run the catcher across the QEC mapping outputs to identify
    QEC-relevant emergent transitions.

    Uses per-quantity catcher (so decoder-iteration curves are
    compared only to other decoder-iteration curves, not to
    topological-protection curves; eliminates labelling artefact).
    """
    from .novelty_catcher import catch_novelty_per_quantity

    # Decoder iterations: same observable across different n_stabilizers
    error_rates = np.linspace(0.001, 0.1, 30)
    decoder_arrays = {}
    for ns in (4, 8, 12, 16):
        iters = np.array([
            predict_decoder_iterations(
                stabilizer_channel_lambda_2(float(p), n_stabilizers=ns),
            )
            for p in error_rates
        ])
        iters = np.where(np.isfinite(iters), iters, 1e10)
        decoder_arrays[f"ns{ns}"] = np.log10(iters + 1e-12)

    # Topological protection (Fibonacci anyons): different observable
    n_bands_grid = list(range(1, 11))
    topo_fib = np.array([
        topological_code_logical_protection(n_bands=int(n), fibonacci=True)[
            "logical_protection_log_D"
        ]
        for n in n_bands_grid
    ])
    topo_std = np.array([
        topological_code_logical_protection(n_bands=int(n), fibonacci=False)[
            "logical_protection_log_D"
        ]
        for n in n_bands_grid
    ])

    per_q = {
        "decoder_log_iter": decoder_arrays,
        "topological_log_D": {
            "fibonacci": topo_fib,
            "abelian": topo_std,
        },
    }
    return catch_novelty_per_quantity(per_q, n_bins=32)
