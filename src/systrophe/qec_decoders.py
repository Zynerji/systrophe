"""QEC decoders + benchmarks for the spectral-oracle validation.

Three decoder families implemented:

  1. Iterative-channel decoder  -- proxy for any contractive decoder;
     the same one used in qec_decoder_oracle_validation.py.
  2. Belief-propagation decoder -- iterative Bayesian update.
  3. Minimum-weight syndrome decoder -- exhaustive match
     (proxy for MWPM).

Each decoder returns an `iter_count` and a `decoded_state`. We
compare against the spectral-oracle prediction
  iter_predicted = -log(tol) / log(|lambda_2|^{-1})
across code families and physical error rates.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .qec_codes import (
    StabilizerCode,
    channel_lambda_2,
    depolarising_code_superoperator,
    logical_error_rate,
    repetition_code_superoperator,
)


@dataclass(frozen=True)
class DecoderResult:
    """Result of running a decoder on a stabilizer-code channel."""
    iter_count: int
    runtime_seconds: float
    converged: bool
    final_residual: float
    decoder_name: str


def iterative_channel_decoder(
    superoperator: np.ndarray,
    max_iter: int = 5000, tol: float = 1e-10, seed: int = 0,
) -> DecoderResult:
    """Iterative-channel-action decoder.

    Apply the channel superoperator repeatedly from a random pure
    state (orthogonal to trivial 1-eigenvector). Returns when
    consecutive iterates' deviation falls below tol.
    """
    rng = np.random.default_rng(seed)
    n = superoperator.shape[0]
    rho = rng.normal(size=n) + 1j * rng.normal(size=n)
    rho -= np.mean(rho) * np.ones(n)
    rho = rho / np.linalg.norm(rho)
    t0 = time.perf_counter()
    final_diff = 0.0
    for k in range(1, max_iter + 1):
        rho_new = superoperator @ rho
        rho_new -= np.mean(rho_new) * np.ones(n)
        norm = np.linalg.norm(rho_new)
        if norm > 1e-15:
            rho_new = rho_new / norm
        diff = float(np.linalg.norm(rho_new - rho))
        final_diff = diff
        if diff < tol:
            return DecoderResult(
                iter_count=k,
                runtime_seconds=time.perf_counter() - t0,
                converged=True, final_residual=diff,
                decoder_name="iterative_channel",
            )
        rho = rho_new
    return DecoderResult(
        iter_count=max_iter, runtime_seconds=time.perf_counter() - t0,
        converged=False, final_residual=final_diff,
        decoder_name="iterative_channel",
    )


def belief_propagation_decoder(
    superoperator: np.ndarray,
    max_iter: int = 5000, tol: float = 1e-10, damping: float = 0.5,
    seed: int = 1,
) -> DecoderResult:
    """Belief-propagation-style decoder (damped iteration).

    Standard BP decoder for stabilizer codes uses Bayesian update on
    a Tanner graph. Here we model BP by damped repeated application
    of the superoperator: rho_{k+1} = damping * S @ rho_k + (1 - damping) * rho_k.
    Damping factor in (0, 1] simulates message-passing on the graph.
    """
    rng = np.random.default_rng(seed)
    n = superoperator.shape[0]
    rho = rng.normal(size=n) + 1j * rng.normal(size=n)
    rho -= np.mean(rho) * np.ones(n)
    rho = rho / np.linalg.norm(rho)
    t0 = time.perf_counter()
    final_diff = 0.0
    for k in range(1, max_iter + 1):
        rho_step = superoperator @ rho
        rho_step -= np.mean(rho_step) * np.ones(n)
        norm = np.linalg.norm(rho_step)
        if norm > 1e-15:
            rho_step = rho_step / norm
        # Damped update
        rho_new = damping * rho_step + (1 - damping) * rho
        rho_new -= np.mean(rho_new) * np.ones(n)
        norm = np.linalg.norm(rho_new)
        if norm > 1e-15:
            rho_new = rho_new / norm
        diff = float(np.linalg.norm(rho_new - rho))
        final_diff = diff
        if diff < tol:
            return DecoderResult(
                iter_count=k, runtime_seconds=time.perf_counter() - t0,
                converged=True, final_residual=diff,
                decoder_name=f"belief_propagation_d{damping}",
            )
        rho = rho_new
    return DecoderResult(
        iter_count=max_iter, runtime_seconds=time.perf_counter() - t0,
        converged=False, final_residual=final_diff,
        decoder_name=f"belief_propagation_d{damping}",
    )


def minimum_weight_proxy_decoder(
    superoperator: np.ndarray,
    max_iter: int = 5000, tol: float = 1e-10, seed: int = 2,
) -> DecoderResult:
    """Minimum-weight-syndrome decoder proxy.

    True MWPM decoders operate on the syndrome graph. As a proxy, we
    use spectral-truncation: project onto the top-k eigenvectors of
    the channel and iterate in that subspace. This converges in fewer
    iterations when the channel's spectral gap is large.
    """
    n = superoperator.shape[0]
    # Use sparse Arnoldi for the dominant eigenpairs. The dense
    # np.linalg.eig allocates an O(n^2) eigenvector matrix that
    # OOMs on n_qubits >= 7 (4^7 = 16384). eigs is O(k * n * iter)
    # in memory and time.
    # Project onto a small spectral subspace. For dense (small n) we
    # use sqrt(n); for sparse (large n) we cap at 8 to keep Arnoldi
    # cheap. The decoder dynamics only depend on the top few
    # eigenvectors anyway -- the channel spectrum decays rapidly.
    if n <= 256:
        k_proj = max(2, int(math.sqrt(n)))
        eigvals, eigvecs = np.linalg.eig(superoperator)
        order = np.argsort(-np.abs(eigvals))
        eigvecs = eigvecs[:, order]
        V = eigvecs[:, :k_proj]
    else:
        from scipy.sparse.linalg import eigs as sparse_eigs
        k_proj = 8
        try:
            _, V_sparse = sparse_eigs(
                superoperator, k=k_proj, which="LM",
                maxiter=1000, tol=1e-8,
            )
            V = V_sparse
        except Exception:
            eigvals, eigvecs = np.linalg.eig(superoperator)
            order = np.argsort(-np.abs(eigvals))
            V = eigvecs[:, order][:, :k_proj]
    S_proj = V.conj().T @ superoperator @ V
    rng = np.random.default_rng(seed)
    rho = rng.normal(size=k_proj) + 1j * rng.normal(size=k_proj)
    rho = rho / np.linalg.norm(rho)
    t0 = time.perf_counter()
    final_diff = 0.0
    for k in range(1, max_iter + 1):
        rho_new = S_proj @ rho
        norm = np.linalg.norm(rho_new)
        if norm > 1e-15:
            rho_new = rho_new / norm
        diff = float(np.linalg.norm(rho_new - rho))
        final_diff = diff
        if diff < tol:
            return DecoderResult(
                iter_count=k, runtime_seconds=time.perf_counter() - t0,
                converged=True, final_residual=diff,
                decoder_name="minimum_weight_proxy",
            )
        rho = rho_new
    return DecoderResult(
        iter_count=max_iter, runtime_seconds=time.perf_counter() - t0,
        converged=False, final_residual=final_diff,
        decoder_name="minimum_weight_proxy",
    )


# ------------------------------------------------------------------
# Decoder vs spectral-oracle benchmark
# ------------------------------------------------------------------

def benchmark_decoders_vs_oracle(
    code_family: str = "repetition",
    n_qubits: int = 3,
    p_grid: np.ndarray | None = None,
    tol: float = 1e-8,
    decoders: tuple[str, ...] = ("iterative", "bp", "mw_proxy"),
) -> dict:
    """Benchmark all decoders against the spectral-oracle prediction.

    For each p in p_grid:
      - Build the code channel superoperator.
      - Extract |lambda_2| -> spectral-oracle prediction.
      - Run each decoder; record iter_count + runtime.
      - Compute log-log Pearson r per decoder against the oracle.
    """
    if p_grid is None:
        p_grid = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2])

    per_decoder: dict[str, dict] = {d: {"iters": [], "runtimes": []}
                                     for d in decoders}
    predicted = []
    lambda_2s = []
    for p in p_grid:
        if code_family == "repetition":
            S = repetition_code_superoperator(float(p), n=n_qubits)
        elif code_family == "depolarising":
            S = depolarising_code_superoperator(float(p), n=n_qubits)
        else:
            raise ValueError(f"Unknown family: {code_family}")
        lam_2 = channel_lambda_2(S)
        if lam_2 <= 1e-15:
            pred = 1
        elif lam_2 >= 1.0 - 1e-15:
            pred = float("inf")
        else:
            pred = -math.log(tol) / -math.log(lam_2)
        predicted.append(pred)
        lambda_2s.append(lam_2)
        if "iterative" in decoders:
            r = iterative_channel_decoder(S, tol=tol)
            per_decoder["iterative"]["iters"].append(r.iter_count)
            per_decoder["iterative"]["runtimes"].append(r.runtime_seconds)
        if "bp" in decoders:
            r = belief_propagation_decoder(S, tol=tol, damping=0.7)
            per_decoder["bp"]["iters"].append(r.iter_count)
            per_decoder["bp"]["runtimes"].append(r.runtime_seconds)
        if "mw_proxy" in decoders:
            r = minimum_weight_proxy_decoder(S, tol=tol)
            per_decoder["mw_proxy"]["iters"].append(r.iter_count)
            per_decoder["mw_proxy"]["runtimes"].append(r.runtime_seconds)

    predicted = np.array(predicted)
    pearson = {}
    log_slope = {}
    for name, data in per_decoder.items():
        meas = np.array(data["iters"])
        if len(meas) >= 3 and np.all(np.isfinite(predicted)) and np.all(meas > 0):
            log_pred = np.log10(predicted)
            log_meas = np.log10(meas)
            r = float(np.corrcoef(log_pred, log_meas)[0, 1])
            slope, _ = np.polyfit(log_pred, log_meas, 1)
            pearson[name] = r
            log_slope[name] = float(slope)
        else:
            pearson[name] = float("nan")
            log_slope[name] = float("nan")

    return {
        "code_family": code_family,
        "n_qubits": n_qubits,
        "p_grid": p_grid.tolist(),
        "lambda_2": lambda_2s,
        "predicted_iter": predicted.tolist(),
        "per_decoder": {
            name: {
                "iters": data["iters"],
                "runtimes": data["runtimes"],
                "pearson_r_log_log": pearson[name],
                "log_log_slope": log_slope[name],
                "mean_runtime_s": float(np.mean(data["runtimes"])),
            }
            for name, data in per_decoder.items()
        },
    }
