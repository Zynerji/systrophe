"""Tests for stochastic LP module."""

import math

import numpy as np
import pytest

from systrophe.lp.stochastic_lp import (
    StochasticResult,
    diffusion_coefficient_estimate,
    escape_probability_in_finite_time,
    mean_first_passage_to_CH,
    novelty_scan,
    stationary_distribution,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_diffusion_coefficient_positive(vs_super):
    D = diffusion_coefficient_estimate(vs_super)
    assert D > 0


def test_mfpt_returns_StochasticResult(vs_super):
    res = mean_first_passage_to_CH(vs_super, r_start=1.3)
    assert isinstance(res, StochasticResult)


def test_mfpt_finite_in_band(vs_super):
    res = mean_first_passage_to_CH(vs_super, r_start=1.3)
    assert math.isfinite(res.mean_first_passage_time)
    assert res.mean_first_passage_time > 0


def test_mfpt_subcritical_infinite(vs_sub):
    res = mean_first_passage_to_CH(vs_sub, r_start=1.5)
    assert res.mean_first_passage_time == float("inf")


def test_stationary_distribution_normalized(vs_super):
    res = stationary_distribution(vs_super)
    rho = res["rho_stationary"]
    total = float(np.trapezoid(rho, res["r_grid"]))
    assert total == pytest.approx(1.0, rel=1e-2)


def test_escape_probability_in_unit(vs_super):
    p = escape_probability_in_finite_time(vs_super, r_start=1.3, T_window=10.0)
    assert 0 <= p <= 1


def test_escape_probability_grows_with_time(vs_super):
    p1 = escape_probability_in_finite_time(vs_super, r_start=1.3, T_window=1.0)
    p100 = escape_probability_in_finite_time(vs_super, r_start=1.3, T_window=100.0)
    assert p100 >= p1


def test_escape_probability_subcritical_zero(vs_sub):
    p = escape_probability_in_finite_time(vs_sub, r_start=1.5, T_window=10.0)
    assert p == 0.0


def test_novelty_scan_returns_verdict():
    res = novelty_scan(n_r_values=10)
    assert "verdict" in res
