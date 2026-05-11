"""Tests for the address-space novelty catcher."""

import numpy as np
import pytest

from systrophe.novelty_catcher import (
    NoveltyScanResult,
    bitstring_counts_to_address,
    catch_novelty_in_distributions,
    hamming_distance,
    lambda_2_of_hamming_graph,
    probability_vector_to_address,
    real_array_to_address,
    scan_novelty,
    summarize_novelty_for_report,
)


def test_probability_vector_address_basic():
    p = np.array([0.05, 0.4, 0.1, 0.45])
    addr = probability_vector_to_address(p, n_bits=4, threshold=0.25)
    assert addr.tolist() == [0, 1, 0, 1]


def test_real_array_address_pads():
    arr = np.array([0.1, 0.5, 0.9])
    addr = real_array_to_address(arr, n_bits=8, v_min=0.0, v_max=1.0)
    assert len(addr) == 8


def test_real_array_address_handles_empty():
    addr = real_array_to_address(np.array([]), n_bits=4)
    assert addr.tolist() == [0, 0, 0, 0]


def test_bitstring_counts_address():
    counts = {"00": 100, "01": 50, "10": 25, "11": 5}
    addr = bitstring_counts_to_address(counts, n_bits=4, rate_threshold=0.1)
    # 00 -> 100/180 = 0.56 > 0.1 -> 1
    # 01 -> 50/180 = 0.28 > 0.1 -> 1
    # 10 -> 25/180 = 0.14 > 0.1 -> 1
    # 11 -> 5/180 = 0.03 < 0.1 -> 0
    assert addr.tolist() == [1, 1, 1, 0]


def test_hamming_distance_simple():
    a = np.array([1, 0, 1, 0])
    b = np.array([1, 1, 1, 1])
    assert hamming_distance(a, b) == 2


def test_lambda_2_of_disconnected_graph_zero():
    """Two isolated addresses at distance > radius -> graph disconnected."""
    addrs = [np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0])]
    l2 = lambda_2_of_hamming_graph(addrs, radius=1)
    assert l2 == 0.0


def test_lambda_2_connected_graph_positive():
    """Two adjacent addresses -> single edge -> λ₂ = 2."""
    addrs = [np.array([1, 1, 0, 0]), np.array([1, 1, 0, 1])]
    l2 = lambda_2_of_hamming_graph(addrs, radius=2)
    assert l2 > 0


def test_scan_novelty_returns_NoveltyScanResult():
    def f(p):
        return np.array([p * 0.5, 1 - p * 0.5])
    params = np.linspace(0.0, 1.0, 5)
    result = scan_novelty(params, f, n_bits=8)
    assert isinstance(result, NoveltyScanResult)


def test_scan_novelty_uniform_detected_non_adaptive():
    """All outputs identical with data_adaptive=False -> 'uniform' verdict.

    Note: with data_adaptive=True (default), rank-thermometer encoding
    will assign distinct addresses even to identical values, so we test
    the legacy path explicitly here.
    """
    def f(p):
        return np.array([0.5, 0.5])
    params = np.linspace(0.0, 1.0, 4)
    result = scan_novelty(params, f, n_bits=4, data_adaptive=False)
    assert result.verdict == "uniform"


def test_scan_novelty_smooth_for_smooth_function():
    def f(p):
        return np.array([p, 1 - p])
    params = np.linspace(0.1, 0.9, 5)
    result = scan_novelty(params, f, n_bits=8)
    assert result.verdict in ("smooth", "uniform")


def test_catch_novelty_in_distributions_returns_dict():
    dists = [np.array([0.5, 0.5]),
             np.array([0.8, 0.2]),
             np.array([0.2, 0.8])]
    res = catch_novelty_in_distributions(dists, n_bits=4)
    assert "verdict" in res
    assert "lambda_2_at_radius" in res
    assert "sharp_features" in res


def test_summarize_novelty_returns_string():
    def f(p):
        return np.array([p, 1 - p])
    params = np.linspace(0.1, 0.9, 4)
    result = scan_novelty(params, f, n_bits=8)
    s = summarize_novelty_for_report(result)
    assert isinstance(s, str)
    assert "Novelty catcher" in s


def test_catcher_on_counts_dicts():
    def f(p):
        return {"00": int(100 * p), "11": int(100 * (1 - p))}
    params = np.linspace(0.1, 0.9, 4)
    result = scan_novelty(params, f, n_bits=4)
    assert isinstance(result, NoveltyScanResult)
