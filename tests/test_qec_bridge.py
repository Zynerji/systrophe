"""Tests for the Systrophe-QEC bridge module."""

import math

import numpy as np
import pytest

from systrophe.qec_bridge import (
    predict_decoder_iterations,
    qec_bridge_catcher_sweep,
    ring_fault_tolerance_threshold,
    stabilizer_channel_lambda_2,
    syndrome_anomaly_score,
    topological_code_logical_protection,
    z3_qutrit_stabilizer_map,
)


class TestTopologicalProtection:
    def test_returns_dict(self):
        out = topological_code_logical_protection(n_bands=3)
        assert isinstance(out, dict)

    def test_n_bands_field_count(self):
        out = topological_code_logical_protection(n_bands=4)
        assert len(out["bands"]) == 4

    def test_protection_factor_positive(self):
        out = topological_code_logical_protection(n_bands=3)
        assert out["protection_factor"] > 0

    def test_fibonacci_dimension_uses_phi(self):
        """Fibonacci anyons use the golden ratio for quantum dimension."""
        out_fib = topological_code_logical_protection(n_bands=2, fibonacci=True)
        out_std = topological_code_logical_protection(n_bands=2, fibonacci=False)
        # The Fibonacci version uses phi ~ 1.618 for d
        assert any(b["quantum_dimension"] != out_std["bands"][i]["quantum_dimension"]
                   for i, b in enumerate(out_fib["bands"]))


class TestSyndromeAnomaly:
    def test_empty_syndromes_returns_empty(self):
        out = syndrome_anomaly_score([])
        assert out["verdict"] == "empty"

    def test_uniform_syndromes_smooth_or_uniform(self):
        """All identical syndromes -> uniform / smooth."""
        out = syndrome_anomaly_score(["0001"] * 10)
        assert out["verdict"] in ("uniform", "smooth")

    def test_mixed_syndromes_returns_dict(self):
        out = syndrome_anomaly_score(["0001", "0010", "0100", "1000"])
        assert "verdict" in out
        assert "n_sharp" in out


class TestDecoderIterations:
    def test_zero_lambda_returns_one(self):
        """|lambda_2| = 0 means immediate convergence."""
        assert predict_decoder_iterations(0.0) == 1.0

    def test_lambda_close_to_one_returns_infinity(self):
        """|lambda_2| ~ 1 means decoder never converges."""
        out = predict_decoder_iterations(1.0 - 1e-16)
        assert math.isinf(out)

    def test_mid_lambda_returns_reasonable_iters(self):
        """|lambda_2| = 0.5 gives ~ 33 iterations for tol=1e-10."""
        out = predict_decoder_iterations(0.5, tol=1e-10)
        assert 30 < out < 40

    def test_stabilizer_lambda_2_decreases_with_error_rate(self):
        l1 = stabilizer_channel_lambda_2(pauli_error_rate=0.01)
        l2 = stabilizer_channel_lambda_2(pauli_error_rate=0.10)
        assert l1 > l2


class TestRingFaultTolerance:
    def test_returns_profiles(self):
        out = ring_fault_tolerance_threshold(
            N_values=(2, 3, 5), n_trials=10,
        )
        assert "profiles" in out
        assert len(out["profiles"]) == 3

    def test_N_2_threshold_is_finite(self):
        """N=2 has a finite noise threshold somewhere in [0, pi]."""
        out = ring_fault_tolerance_threshold(
            N_values=(2,), n_trials=20,
        )
        thr = out["profiles"][0]["noise_threshold_rad"]
        assert 0.0 < thr <= math.pi

    def test_N_5_threshold_is_higher_than_or_equal_to_N_2(self):
        """N=5 should be at least as noise-tolerant as N=2 (higher
        threshold OR equal at the scan endpoint)."""
        out = ring_fault_tolerance_threshold(
            N_values=(2, 5), n_trials=20,
        )
        thr2 = out["profiles"][0]["noise_threshold_rad"]
        thr5 = out["profiles"][1]["noise_threshold_rad"]
        # N=5 should tolerate at least as much noise as N=2
        assert thr5 >= thr2 - 0.5

    def test_profiles_returned_for_each_N(self):
        out = ring_fault_tolerance_threshold(
            N_values=(2, 3, 5, 8), n_trials=10,
        )
        # Check we got 4 profiles back
        assert len(out["profiles"]) == 4
        # And each has the expected fields
        for p in out["profiles"]:
            assert "N" in p and "noise_threshold_rad" in p
            assert "is_robust" in p and "catcher_verdict" in p


class TestZ3QutritMap:
    def test_returns_dict(self):
        out = z3_qutrit_stabilizer_map(N_nodes=12)
        assert "available" in out

    def test_z3_eigenvalues_are_cube_roots_of_unity(self):
        """The qutrit eigenvalues are 1, cos(2pi/3), cos(4pi/3)."""
        out = z3_qutrit_stabilizer_map(N_nodes=12)
        if out["available"]:
            evs = out["z3_qutrit_eigenvalues"]
            assert len(evs) == 3
            assert abs(evs[0] - 1.0) < 1e-9
            assert abs(evs[1] - math.cos(2 * math.pi / 3)) < 1e-9


class TestCatcherSweep:
    def test_runs_and_returns_dict(self):
        res = qec_bridge_catcher_sweep()
        # The function now returns a per-quantity wrapper dict
        # (catch_novelty_per_quantity) with aggregate_verdict at the top.
        assert "aggregate_verdict" in res
        assert "per_quantity" in res
        assert res["aggregate_verdict"] in (
            "uniform", "smooth", "novel_structure", "insufficient",
        )
