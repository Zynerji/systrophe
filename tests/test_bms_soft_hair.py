"""Tests for BMS soft hair module."""

import math

import pytest

from systrophe.bms_soft_hair import (
    SoftHairMode,
    novelty_scan,
    pair_extinction_soft_hair,
    soft_hair_information_content,
    superrotation_charge,
    supertranslation_mode_amplitudes,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_supertranslation_modes_returns_list(vs_super):
    modes = supertranslation_mode_amplitudes(vs_super, n_modes=5)
    assert len(modes) == 5
    assert all(isinstance(m, SoftHairMode) for m in modes)


def test_supertranslation_modes_subcritical_empty(vs_sub):
    modes = supertranslation_mode_amplitudes(vs_sub)
    assert modes == []


def test_supertranslation_amplitudes_decay(vs_super):
    modes = supertranslation_mode_amplitudes(vs_super, n_modes=4)
    amps = [m.amplitude for m in modes]
    for a1, a2 in zip(amps[:-1], amps[1:]):
        assert a2 < a1


def test_superrotation_charge_finite(vs_super):
    Q = superrotation_charge(vs_super, n=1)
    assert math.isfinite(Q)
    assert Q > 0


def test_superrotation_subcritical_zero(vs_sub):
    Q = superrotation_charge(vs_sub)
    assert Q == 0.0


def test_soft_hair_information_content_finite(vs_super):
    res = soft_hair_information_content(vs_super, n_modes=5)
    assert res["total_bits"] >= 0
    assert "n_modes_used" in res


def test_soft_hair_subcritical_zero_bits(vs_sub):
    res = soft_hair_information_content(vs_sub)
    assert res["total_bits"] == 0.0


def test_pair_extinction_at_pi_returns_zero_hair(vs_super):
    res = pair_extinction_soft_hair(vs_super, delta=math.pi)
    assert all(a == 0 for a in res["scaled_amplitudes"])
    assert res["hair_removed_at_pi"] is True


def test_pair_extinction_at_zero_full_hair(vs_super):
    res = pair_extinction_soft_hair(vs_super, delta=0.0)
    # extinction_factor = 1
    base = supertranslation_mode_amplitudes(vs_super, n_modes=5)
    for sa, m in zip(res["scaled_amplitudes"], base):
        assert sa == pytest.approx(m.amplitude, rel=1e-12)


def test_novelty_scan_returns_verdict():
    res = novelty_scan(n_modes=15)
    assert "verdict" in res
    assert "sharp_features" in res
