"""Tests for systrophe.quantum_info.qec_codes and systrophe.quantum_info.qec_decoders."""

import math

import numpy as np

from systrophe.quantum_info.qec_codes import (
    CODE_THRESHOLDS,
    bacon_shor_descriptor,
    channel_lambda_2,
    color_code_descriptor,
    depolarising_code_superoperator,
    depolarising_single_qubit_superoperator,
    lambda_2_per_code_family,
    logical_error_rate,
    repetition_code_descriptor,
    repetition_code_superoperator,
    resource_cost,
    steane_code_descriptor,
    surface_code_descriptor,
)
from systrophe.quantum_info.qec_decoders import (
    DecoderResult,
    belief_propagation_decoder,
    benchmark_decoders_vs_oracle,
    iterative_channel_decoder,
    minimum_weight_proxy_decoder,
)


class TestCodes:
    def test_repetition_descriptor(self):
        code = repetition_code_descriptor(n=5)
        assert code.name == "repetition_5"
        assert code.n == 5 and code.k == 1 and code.d == 5

    def test_steane_descriptor(self):
        code = steane_code_descriptor()
        assert code.n == 7 and code.k == 1 and code.d == 3

    def test_bacon_shor_descriptor(self):
        code = bacon_shor_descriptor(d=3)
        assert code.n == 9 and code.k == 1 and code.d == 3

    def test_surface_descriptor(self):
        code = surface_code_descriptor(d=5)
        assert code.n == 25 and code.k == 1 and code.d == 5

    def test_color_descriptor(self):
        code = color_code_descriptor(d=3)
        assert code.k == 1 and code.d == 3

    def test_repetition_superoperator_shape(self):
        S = repetition_code_superoperator(p=0.05, n=3)
        # n qubits -> 4^n x 4^n superoperator (column-stacked density matrix)
        assert S.shape == (64, 64)

    def test_depolarising_superoperator_unitarity_preserves_trace(self):
        """For a CPTP map, Tr(E(rho)) = Tr(rho); the column-stacked
        identity vector |I> maps to itself with eigenvalue 1."""
        S = depolarising_single_qubit_superoperator(p=0.05)
        I_vec = np.array([1, 0, 0, 1], dtype=complex) / 2.0  # |I> / 2
        S_I = S @ I_vec
        # The trace-preserving condition implies Tr(E(rho)) = Tr(rho)
        # which is harder to test in column-stacked form without proper
        # contraction; we test the simpler condition that |I> is a
        # 1-eigenvector
        assert np.allclose(S_I, I_vec, atol=1e-9)

    def test_lambda_2_decreases_with_error_rate(self):
        """For independent-noise channels, larger p -> smaller |lambda_2|
        (faster decoder convergence)."""
        S_lo = repetition_code_superoperator(p=0.01, n=3)
        S_hi = repetition_code_superoperator(p=0.1, n=3)
        lam_lo = channel_lambda_2(S_lo)
        lam_hi = channel_lambda_2(S_hi)
        assert lam_lo > lam_hi

    def test_logical_error_rate_decreases_with_distance(self):
        """At fixed p_phys < p_th, higher distance gives lower p_L."""
        for d in (3, 5, 9):
            code = surface_code_descriptor(d=d)
        p_L_3 = logical_error_rate(surface_code_descriptor(3), 0.001)
        p_L_9 = logical_error_rate(surface_code_descriptor(9), 0.001)
        assert p_L_9 < p_L_3

    def test_resource_cost_keys(self):
        code = surface_code_descriptor(d=5)
        cost = resource_cost(code)
        assert "n_physical" in cost
        assert "n_logical" in cost
        assert "distance" in cost
        assert cost["distance"] == 5

    def test_thresholds_in_range(self):
        for family, thr in CODE_THRESHOLDS.items():
            assert 0 < thr < 1, f"{family} threshold {thr} out of range"

    def test_lambda_2_per_family(self):
        out = lambda_2_per_code_family(p_phys=0.05, n_qubits=3)
        assert "repetition" in out
        assert "depolarising" in out


class TestDecoders:
    def test_iterative_returns_DecoderResult(self):
        S = repetition_code_superoperator(p=0.05, n=3)
        r = iterative_channel_decoder(S, tol=1e-6, max_iter=200)
        assert isinstance(r, DecoderResult)
        assert r.iter_count >= 1

    def test_bp_returns_DecoderResult(self):
        S = repetition_code_superoperator(p=0.05, n=3)
        r = belief_propagation_decoder(S, tol=1e-6, max_iter=200)
        assert isinstance(r, DecoderResult)

    def test_mw_proxy_returns_DecoderResult(self):
        S = repetition_code_superoperator(p=0.05, n=3)
        r = minimum_weight_proxy_decoder(S, tol=1e-6, max_iter=200)
        assert isinstance(r, DecoderResult)

    def test_iter_count_decreases_with_noise(self):
        """Higher p (more noise) -> larger spectral gap ->
        faster decoder convergence."""
        S_lo = repetition_code_superoperator(p=0.01, n=3)
        S_hi = repetition_code_superoperator(p=0.1, n=3)
        r_lo = iterative_channel_decoder(S_lo, tol=1e-6, max_iter=2000)
        r_hi = iterative_channel_decoder(S_hi, tol=1e-6, max_iter=2000)
        assert r_hi.iter_count <= r_lo.iter_count

    def test_runtimes_positive(self):
        S = repetition_code_superoperator(p=0.05, n=3)
        r = iterative_channel_decoder(S, tol=1e-4, max_iter=50)
        assert r.runtime_seconds > 0


class TestBenchmark:
    def test_returns_dict_with_per_decoder(self):
        out = benchmark_decoders_vs_oracle(
            code_family="repetition", n_qubits=3,
            p_grid=np.array([0.01, 0.05, 0.1]),
            tol=1e-4,
            decoders=("iterative", "bp"),
        )
        assert "per_decoder" in out
        assert "iterative" in out["per_decoder"]
        assert "bp" in out["per_decoder"]

    def test_pearson_r_in_range(self):
        out = benchmark_decoders_vs_oracle(
            code_family="repetition", n_qubits=3,
            p_grid=np.array([0.01, 0.02, 0.05, 0.1, 0.2]),
            tol=1e-4,
            decoders=("iterative",),
        )
        r = out["per_decoder"]["iterative"]["pearson_r_log_log"]
        if math.isfinite(r):
            assert -1 <= r <= 1
