"""Tests for square-law (intensity) detection of the Tipler log-periodic structure."""

import numpy as np

from tipler_squarelaw import square_law_experiment, square_law_sweep


def test_square_law_doubles_log_frequency():
    """A square-law detector measures 2 alpha (the cos -> cos^2 identity)."""
    for a in (0.7, 1.0, 1.5):
        r = square_law_experiment(a)
        assert abs(r["alpha_from_L"] - r["alpha"]) / r["alpha"] < 0.05
        assert 1.85 < r["freq_doubling_ratio"] < 2.05


def test_apparent_dsi_ratio_is_sqrt_of_true():
    """L^2 reports the DSI rescaling as sqrt(lambda) instead of lambda."""
    r = square_law_experiment(1.0)
    # apparent lambda ~ sqrt(true lambda) (finite-range zero-count tolerance)
    assert abs(r["dsi_lambda_apparent_from_Lsq"] - r["sqrt_lambda_check"]) \
        / r["sqrt_lambda_check"] < 0.15


def test_ctc_sign_destroyed_by_square_law():
    """CTC bands (L<0) are recoverable from L (~half the range) but NOT from L^2."""
    r = square_law_experiment(1.0)
    assert 0.4 < r["ctc_fraction_from_L"] < 0.6
    assert r["ctc_fraction_from_Lsq"] == 0.0
    assert not r["ctc_sign_recoverable_from_Lsq"]


def test_sweep_consistent():
    # exact doubling is 2.0; finite-range zero-counting gives ~10% scatter
    rows = square_law_sweep([0.8, 1.2, 2.0])
    assert all(1.7 < row["freq_doubling_ratio"] < 2.2 for row in rows)
