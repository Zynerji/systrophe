"""Tests for Page curve CTC module."""

import math

import numpy as np
import pytest

from systrophe.page_curve_ctc import (
    PageCurveData,
    all_ctc_band_page_curves,
    bekenstein_bound_estimate,
    enclosed_region_volume,
    entanglement_entropy_proxy,
    information_paradox_resolved_at_page_time,
    page_curve,
    page_time_estimate,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_enclosed_region_volume_positive(vs_super):
    V = enclosed_region_volume(vs_super, 1.83, 11.23)
    assert V > 0


def test_enclosed_region_volume_invalid_range(vs_super):
    with pytest.raises(ValueError):
        enclosed_region_volume(vs_super, 5.0, 2.0)


def test_bekenstein_bound_positive(vs_super):
    S_max = bekenstein_bound_estimate(vs_super, 1.83, 11.23)
    assert S_max > 0


def test_page_time_finite(vs_super):
    pt = page_time_estimate(vs_super, 1.83, 11.23)
    assert math.isfinite(pt)
    assert pt > 0


def test_page_time_zero_emission_infinite(vs_super):
    pt = page_time_estimate(vs_super, 1.83, 11.23, emission_rate=0.0)
    assert pt == float("inf")


def test_entropy_proxy_zero_at_zero_time(vs_super):
    S = entanglement_entropy_proxy(vs_super, 1.83, 11.23, t=0.0)
    assert S == 0.0


def test_entropy_proxy_peaks_at_page_time(vs_super):
    pt = page_time_estimate(vs_super, 1.83, 11.23)
    S_at_pt = entanglement_entropy_proxy(vs_super, 1.83, 11.23, t=pt)
    S_before = entanglement_entropy_proxy(vs_super, 1.83, 11.23, t=pt / 2)
    S_after = entanglement_entropy_proxy(vs_super, 1.83, 11.23, t=1.5 * pt)
    # Peak at Page time
    assert S_at_pt >= S_before - 1e-9
    assert S_at_pt >= S_after - 1e-9


def test_page_curve_returns_PageCurveData(vs_super):
    curve = page_curve(vs_super, 1.83, 11.23)
    assert isinstance(curve, PageCurveData)
    assert len(curve.t_grid) == len(curve.s_at_times)


def test_page_curve_s_starts_at_zero(vs_super):
    curve = page_curve(vs_super, 1.83, 11.23)
    assert curve.s_at_times[0] == pytest.approx(0.0, abs=1e-9)


def test_page_curve_s_ends_at_zero(vs_super):
    curve = page_curve(vs_super, 1.83, 11.23)
    assert curve.s_at_times[-1] == pytest.approx(0.0, abs=1e-9)


def test_page_curve_max_at_max_entropy(vs_super):
    curve = page_curve(vs_super, 1.83, 11.23)
    # Numerical peak <= analytical max_entropy
    assert max(curve.s_at_times) <= curve.max_entropy + 1e-6


def test_information_paradox_resolved(vs_super):
    res = information_paradox_resolved_at_page_time(vs_super, 1.83, 11.23)
    assert "resolved" in res
    # Page curve symmetric -> entropy at 2*pt should equal entropy at pt
    # For our piecewise-linear proxy, 2*pt is at the END (= 0), so it
    # *should* be < entropy(pt), i.e. resolved=True.
    # (entropy_at_2x_page_time should be ~ 0 since 2*pt = T_evap)
    assert res["resolved"] is True


def test_all_ctc_band_page_curves_returns_list(vs_super):
    curves = all_ctc_band_page_curves(vs_super, n_bands=2)
    assert isinstance(curves, list)
    assert all(isinstance(c, PageCurveData) for c in curves)


def test_all_ctc_band_page_curves_subcritical_empty(vs_sub):
    curves = all_ctc_band_page_curves(vs_sub)
    assert curves == []
