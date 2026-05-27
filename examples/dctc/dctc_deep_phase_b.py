"""D-CTC deep exploration --- Phase B: spectral characterization.

For each Haar-random U, builds the explicit channel superoperator
E: rho_CTC -> Tr_CR[U(sigma_CR (x) rho_CTC)U^dag] as a
(d_ctc^2 x d_ctc^2) matrix.

Computes lambda_2(E) (second-largest-magnitude eigenvalue). The
spectral mixing-time prediction is

    iter_required(epsilon) ~ -log(epsilon) / log|lambda_2|^{-1}
                          ~  1 / (1 - |lambda_2|)            (slow-mixing limit)

Question: does the empirical iteration count correlate with the
predicted spectral mixing time? If yes, this gives an O(d^6) spectral
oracle that predicts iteration count without running the iteration.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from systrophe.ctc.d_ctc import dctc_fixed_point, density_matrix_diagnostics


def haar_random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    diag_R = np.diag(R)
    phases = diag_R / np.abs(diag_R)
    return Q * phases


def channel_superoperator(
    U: np.ndarray, sigma_cr: np.ndarray, dim_cr: int, dim_ctc: int
) -> np.ndarray:
    """Build the channel E: rho_CTC -> Tr_CR[U (sigma_CR (x) rho_CTC) U^dag]
    as a (d_ctc^2, d_ctc^2) matrix acting on vec(rho_CTC).

    Uses the explicit action on the d_ctc^2 basis matrices |i><j|.
    """
    d2 = dim_ctc * dim_ctc
    M = np.zeros((d2, d2), dtype=complex)
    for k in range(d2):
        i, j = divmod(k, dim_ctc)
        rho_basis = np.zeros((dim_ctc, dim_ctc), dtype=complex)
        rho_basis[i, j] = 1.0
        # E(rho_basis) = Tr_CR[U (sigma_CR (x) rho_basis) U^dag]
        joint = np.kron(sigma_cr, rho_basis)
        out = U @ joint @ U.conj().T
        out_resh = out.reshape((dim_cr, dim_ctc, dim_cr, dim_ctc))
        result = np.einsum("aiaj->ij", out_resh)
        M[:, k] = result.reshape(d2)
    return M


def spectral_predict_iter(lambda_2_abs: float, tol: float = 1e-10) -> float:
    """Predicted iteration count to reach `tol` based on |lambda_2|."""
    if lambda_2_abs <= 1e-15:
        return 1.0
    if lambda_2_abs >= 1.0 - 1e-15:
        return float("inf")
    return -np.log(tol) / -np.log(lambda_2_abs)


def main():
    print("=" * 70)
    print("D-CTC Phase B: spectral characterization")
    print("=" * 70)
    print()

    configs = [(2, 3), (4, 3), (2, 4), (3, 4)]
    n_samples = 80
    print(f"Configs: {configs}, {n_samples} samples each")
    print()

    all_results = {}
    t0 = time.time()
    for dim_cr, dim_ctc in configs:
        print(f"--- dim_CR={dim_cr}, dim_CTC={dim_ctc} ---")
        rng = np.random.default_rng(31 + 100 * dim_cr + dim_ctc)
        dim_total = dim_cr * dim_ctc

        l2_list = []
        iter_list = []
        purity_list = []
        spectral_predict_list = []

        for k in range(n_samples):
            U = haar_random_unitary(dim_total, rng)
            sigma_cr = np.zeros((dim_cr, dim_cr), dtype=complex)
            sigma_cr[0, 0] = 1.0
            psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
            psi = psi / np.linalg.norm(psi)
            rho_init = np.outer(psi, psi.conj())

            # Empirical iteration count
            r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                                  rho_ctc_init=rho_init, tol=1e-10,
                                  max_iter=5000)
            iter_list.append(r["iterations"])
            diag = density_matrix_diagnostics(r["rho_ctc"])
            purity_list.append(diag["purity"])

            # Channel superoperator
            M = channel_superoperator(U, sigma_cr, dim_cr, dim_ctc)
            eigs = np.linalg.eigvals(M)
            eigs_abs = np.abs(eigs)
            order = np.argsort(-eigs_abs)
            # lambda_1 should be ~1 (CPTP map); lambda_2 is the spectral gap
            l1 = eigs_abs[order[0]]
            l2 = eigs_abs[order[1]] if len(eigs_abs) > 1 else 0.0
            l2_list.append(l2)
            spectral_predict_list.append(spectral_predict_iter(l2))

        l2_arr = np.array(l2_list)
        iter_arr = np.array(iter_list)
        purity_arr = np.array(purity_list)
        pred_arr = np.array(spectral_predict_list)

        # Drop infinities for correlation
        finite = np.isfinite(pred_arr) & (iter_arr < 5000)
        if finite.sum() < 5:
            print(f"  too few finite samples ({finite.sum()}); skipping correlation")
            continue
        pearson = float(np.corrcoef(iter_arr[finite], pred_arr[finite])[0, 1])
        # log-log slope (theory says slope = 1 in log-log space)
        log_emp = np.log(np.clip(iter_arr[finite], 1, None))
        log_pred = np.log(np.clip(pred_arr[finite], 1, None))
        valid_loglog = (log_pred > 0) & (log_emp > 0)
        slope, intercept = np.polyfit(log_pred[valid_loglog], log_emp[valid_loglog], 1)

        print(f"  mean |lambda_2| = {l2_arr.mean():.4f}, max = {l2_arr.max():.4f}, min = {l2_arr.min():.4f}")
        print(f"  iter median = {np.median(iter_arr):.1f}, max = {iter_arr.max()}")
        print(f"  Pearson(iter, predicted) = {pearson:.4f}")
        print(f"  log-log slope            = {slope:.4f} (theory: 1.0)")
        print(f"  log-log intercept        = {intercept:.4f}")

        # Where the prediction breaks (lambda_2 -> 1 cases)
        slow_idx = np.argsort(-l2_arr)[:5]
        print(f"  top-5 slow-mixing |lambda_2|: {[f'{l2_arr[i]:.4f}' for i in slow_idx]}")
        print(f"      their iter counts        : {[int(iter_arr[i]) for i in slow_idx]}")
        print(f"      their purities           : {[f'{purity_arr[i]:.3f}' for i in slow_idx]}")

        all_results[f"{dim_cr}x{dim_ctc}"] = {
            "dim_cr": dim_cr, "dim_ctc": dim_ctc,
            "lambda_2": l2_arr.tolist(),
            "iter": iter_arr.tolist(),
            "purity": purity_arr.tolist(),
            "spectral_predict": pred_arr.tolist(),
            "pearson_correlation": pearson,
            "loglog_slope": float(slope),
            "loglog_intercept": float(intercept),
            "n_samples": n_samples,
            "mean_lambda_2": float(l2_arr.mean()),
            "max_lambda_2": float(l2_arr.max()),
        }
        print()

    print(f"Total compute: {time.time() - t0:.1f}s")
    print()
    print("=" * 70)
    print("Interpretation")
    print("=" * 70)
    print()
    print("If Pearson correlation > 0.9 and log-log slope ~ 1, then")
    print("|lambda_2(E)| is a strong predictor of iteration count.")
    print()
    print("This gives an O(d_total^6) one-shot spectral oracle that predicts")
    print("convergence rate without iterating --- the empirical iteration")
    print("counts emerge from a single eigendecomposition.")
    print()

    out_path = Path("examples") / "dctc_deep_phase_b_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
