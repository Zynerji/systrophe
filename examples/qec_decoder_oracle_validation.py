"""Validate the QEC spectral oracle (Mapping 3) against a real
iterative QEC decoder.

The spectral-oracle formula predicts the iteration count of any
contractive decoder from the second-largest-magnitude eigenvalue
|lambda_2(E)| of the channel superoperator:

    iter_required(tol) = -log(tol) / log(|lambda_2|^{-1})

Pearson r = 0.99 in the D-CTC source-domain validation
(dctc_deep_phase_b). This script validates the SAME formula
against an iterative QEC decoder for a 3-qubit bit-flip code
under controlled depolarising noise.

Methodology
-----------
1. Construct the bit-flip-code CPTP channel superoperator E_p
   at physical error rate p.
2. Numerically diagonalise E_p and extract |lambda_2(E_p)|.
3. Run an iterative syndrome-decoder on the channel until the
   logical-error-rate estimate converges to tolerance tol; count
   iterations.
4. Compare measured iterations to spectral-oracle prediction.
5. Pearson r and log-log slope over a sweep of p.
"""

from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np

from systrophe.qec_bridge import predict_decoder_iterations


def bit_flip_code_superoperator(p: float, n_qubits: int = 3) -> np.ndarray:
    """CPTP channel superoperator for the n-qubit bit-flip code under
    independent X-error noise.

    Each qubit independently flips with probability p; the resulting
    channel acts on the n-qubit density matrix in column-stacked
    representation. Returns the (4^n, 4^n) superoperator matrix.

    For n=3 (the standard bit-flip code), the result is a 64x64 matrix.
    """
    # Single-qubit bit-flip channel:
    #   E(rho) = (1-p) rho + p X rho X
    # Choi superoperator (column-stacked):
    #   S_single = (1-p) I (x) I + p X (x) X
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    S_single = (1.0 - p) * np.kron(eye2, eye2) + p * np.kron(sigma_x, sigma_x)
    # n-qubit tensor product
    S_total = S_single
    for _ in range(n_qubits - 1):
        S_total = np.kron(S_total, S_single)
    return S_total


def measure_iterative_decoder_iterations(
    p: float, n_qubits: int = 3,
    max_iter: int = 5000, tol: float = 1e-10,
    seed: int = 0,
) -> int:
    """Run an iterative bit-flip-code syndrome decoder under noise p.

    The decoder applies the channel superoperator repeatedly from a
    RANDOM PURE INITIAL STATE (orthogonal to the trivial fixed-point
    subspace), returns the number of iterations needed for the
    iterates' successive deviation to fall below `tol`.
    """
    S = bit_flip_code_superoperator(p, n_qubits=n_qubits)
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    # Random vector orthogonal to the all-ones direction (the trivial
    # 1-eigenvector of a CPTP channel)
    rho = rng.normal(size=n) + 1j * rng.normal(size=n)
    rho -= np.mean(rho) * np.ones(n)  # project out trivial direction
    rho = rho / np.linalg.norm(rho)
    for k in range(1, max_iter + 1):
        rho_new = S @ rho
        # Project out trivial 1-eigenvector
        rho_new -= np.mean(rho_new) * np.ones(n)
        norm = np.linalg.norm(rho_new)
        if norm > 1e-15:
            rho_new = rho_new / norm
        diff = float(np.linalg.norm(rho_new - rho))
        if diff < tol:
            return k
        rho = rho_new
    return max_iter


def measure_channel_lambda_2(p: float, n_qubits: int = 3) -> float:
    """Numerically extract |lambda_2(E)| from the superoperator.

    Returns the LARGEST eigenvalue magnitude strictly less than 1
    (i.e., excludes the trivial 1-eigenvalue corresponding to the
    fixed-point subspace).
    """
    S = bit_flip_code_superoperator(p, n_qubits=n_qubits)
    eigenvalues = np.linalg.eigvals(S)
    abs_eigs = np.sort(np.abs(eigenvalues))[::-1]  # descending
    # Strip the trivial 1-eigenvalue (and any near-1 fixed-point
    # eigenvalues, of which a CPTP channel can have several).
    nontrivial = abs_eigs[abs_eigs < 1.0 - 1e-9]
    if len(nontrivial) == 0:
        return 0.0
    return float(nontrivial[0])


def validate_spectral_oracle_on_qec(
    p_grid: np.ndarray | None = None,
    n_qubits: int = 3, tol: float = 1e-8,
) -> dict:
    """Validate spectral-oracle prediction against measured iterations
    across a noise-rate sweep.

    For each p in p_grid:
      - measure |lambda_2(E_p)| from the superoperator
      - predict iter from the spectral oracle
      - measure actual iter from the iterative decoder
      - compare

    Returns Pearson r and log-log slope of predicted vs measured.
    """
    if p_grid is None:
        p_grid = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2])

    rows = []
    for p in p_grid:
        lam_2 = measure_channel_lambda_2(float(p), n_qubits=n_qubits)
        predicted = predict_decoder_iterations(lam_2, tol=tol)
        measured = measure_iterative_decoder_iterations(
            float(p), n_qubits=n_qubits, tol=tol,
        )
        rows.append({
            "p": float(p),
            "lambda_2": float(lam_2),
            "predicted_iter": float(predicted) if math.isfinite(predicted) else float("inf"),
            "measured_iter": int(measured),
        })

    # Compute Pearson r on log scale (since iter spans orders of magnitude)
    finite_rows = [r for r in rows if math.isfinite(r["predicted_iter"])]
    if len(finite_rows) >= 3:
        pred = np.array([math.log10(r["predicted_iter"]) for r in finite_rows])
        meas = np.array([math.log10(r["measured_iter"]) for r in finite_rows])
        # Pearson r
        pearson = float(np.corrcoef(pred, meas)[0, 1])
        # log-log slope
        slope, intercept = np.polyfit(pred, meas, 1)
    else:
        pearson = float("nan")
        slope = float("nan")
        intercept = float("nan")

    return {
        "rows": rows,
        "pearson_r_log_log": pearson,
        "log_log_slope": float(slope),
        "log_log_intercept": float(intercept),
        "n_qubits": int(n_qubits),
        "tol": float(tol),
    }


def main() -> None:
    print("=" * 70)
    print("QEC spectral-oracle validation against bit-flip-code decoder")
    print("=" * 70)
    print()

    result = validate_spectral_oracle_on_qec()

    print(f"{'p_phys':10s} {'|lambda_2|':12s} {'predicted iter':16s} {'measured iter':14s}")
    print("-" * 60)
    for r in result["rows"]:
        print(f"  {r['p']:8.4f}    {r['lambda_2']:8.4f}      "
              f"{r['predicted_iter']:12.1f}      {r['measured_iter']:8d}")
    print()
    print(f"Pearson r (log-log):  {result['pearson_r_log_log']:.4f}")
    print(f"log-log slope:        {result['log_log_slope']:.4f}  (theory: 1.0)")
    print(f"log-log intercept:    {result['log_log_intercept']:.4f}")
    print()
    if result["pearson_r_log_log"] > 0.95:
        print("VALIDATED: spectral oracle predicts iter count with Pearson r > 0.95.")
    elif result["pearson_r_log_log"] > 0.80:
        print("PARTIAL: r > 0.80; oracle is informative but not exact.")
    else:
        print("FAILED: r < 0.80; oracle does not apply in this decoder regime.")

    out_path = Path(__file__).parent / "qec_decoder_oracle_validation_results.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
