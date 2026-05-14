"""Tests for warp-drive-zoo."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from warp_drive_zoo import (
    AlcubierreDrive,
    BobrickMartireDrive,
    LentzDrive,
    WarpDriveComparison,
    compare_drives,
)


# ---------------------------------------------------------------------------
# AlcubierreDrive
# ---------------------------------------------------------------------------


def test_alcubierre_NEC_violated_at_wall():
    """Alcubierre is the canonical NEC-violating drive."""
    d = AlcubierreDrive(v_s=1.0, R=1.0, sigma=8.0)
    nec = d.nec_radial(*d.wall_location())
    assert nec < 0


def test_alcubierre_total_negative_energy_scales_with_v_s():
    """Pfenning-Ford: total exotic mass scales ~ v_s² (energy density
    ~ v_s²; total integral scales with v_s²)."""
    d_slow = AlcubierreDrive(v_s=0.5)
    d_fast = AlcubierreDrive(v_s=1.0)
    E_slow = abs(d_slow.total_negative_energy())
    E_fast = abs(d_fast.total_negative_energy())
    # Should scale roughly as v_s^2, so E_fast / E_slow ≈ 4
    ratio = E_fast / E_slow
    assert 3.0 < ratio < 5.0


def test_alcubierre_total_negative_energy_is_negative():
    d = AlcubierreDrive()
    assert d.total_negative_energy() < 0


def test_alcubierre_wall_at_R():
    d = AlcubierreDrive(R=1.5)
    x_w, rho_w = d.wall_location()
    assert x_w == pytest.approx(1.5)
    assert rho_w == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BobrickMartireDrive
# ---------------------------------------------------------------------------


def test_bm_positive_mass_no_exotic_matter():
    """Bobrick-Martire with m_ADM > 0 respects NEC at the wall."""
    d = BobrickMartireDrive(m_ADM=1.0)
    nec = d.nec_radial(*d.wall_location())
    # NEC violated iff nec < 0; for m_ADM > 0, expect NEC respected
    assert nec >= 0 or abs(nec) < 1e-6


def test_bm_negative_mass_exotic_matter():
    """Bobrick-Martire with m_ADM < 0 violates NEC at the wall."""
    d = BobrickMartireDrive(m_ADM=-1.0)
    nec = d.nec_radial(*d.wall_location())
    assert nec <= 0


def test_bm_energy_density_sign_follows_mass():
    """T_tt = m_ADM * F; sign tracks m_ADM."""
    d_pos = BobrickMartireDrive(m_ADM=1.0)
    d_neg = BobrickMartireDrive(m_ADM=-1.0)
    rho_pos = d_pos.energy_density(0.5, 0.0)
    rho_neg = d_neg.energy_density(0.5, 0.0)
    assert np.sign(rho_pos) == -np.sign(rho_neg)


# ---------------------------------------------------------------------------
# LentzDrive
# ---------------------------------------------------------------------------


def test_lentz_energy_density_returns_finite():
    d = LentzDrive()
    rho = d.energy_density(0.0, 1.0)
    assert np.isfinite(rho)


def test_lentz_nec_at_wall_finite():
    d = LentzDrive()
    nec = d.nec_radial(*d.wall_location())
    assert np.isfinite(nec)


# ---------------------------------------------------------------------------
# Head-to-head comparison
# ---------------------------------------------------------------------------


def test_comparison_returns_dataclass():
    drives = [
        AlcubierreDrive(),
        BobrickMartireDrive(m_ADM=1.0),
        LentzDrive(),
    ]
    cmp_ = compare_drives(drives, box_half_size=2.0, n_grid=16)
    assert isinstance(cmp_, WarpDriveComparison)
    assert set(cmp_.drives) == {"alcubierre", "bobrick_martire", "lentz"}
    for name in cmp_.drives:
        assert name in cmp_.nec_at_wall
        assert name in cmp_.total_negative_energy


def test_comparison_alcubierre_most_exotic():
    """Among Alcubierre + positive-mass BM, Alcubierre has more |negative E|."""
    drives = [
        AlcubierreDrive(v_s=1.0),
        BobrickMartireDrive(m_ADM=1.0),  # positive mass — no exotic
    ]
    cmp_ = compare_drives(drives, box_half_size=2.0, n_grid=16)
    # Positive-mass BM should have ~zero exotic; Alcubierre should have nonzero
    bm_neg = abs(cmp_.total_negative_energy["bobrick_martire"])
    al_neg = abs(cmp_.total_negative_energy["alcubierre"])
    assert al_neg > bm_neg


def test_comparison_ranking_ordered():
    """Ranking should be sorted ascending by |negative energy|."""
    drives = [
        AlcubierreDrive(),
        BobrickMartireDrive(m_ADM=1.0),
        LentzDrive(),
    ]
    cmp_ = compare_drives(drives, box_half_size=2.0, n_grid=16)
    vals = [v for (_, v) in cmp_.ranking_by_exotic_matter]
    assert vals == sorted(vals)


def test_comparison_nec_violated_flags():
    drives = [
        AlcubierreDrive(),
        BobrickMartireDrive(m_ADM=-1.0),  # negative mass — exotic
    ]
    cmp_ = compare_drives(drives, box_half_size=2.0, n_grid=16)
    assert cmp_.nec_violated["alcubierre"] is True
    # Negative-mass BM: NEC violated
    assert cmp_.nec_violated["bobrick_martire"] is True


def test_pfenning_ford_scaling_reproduced():
    """Alcubierre total negative energy scales as ~v_s^2 (re-test via compare)."""
    drives = [
        AlcubierreDrive(v_s=0.5),
        AlcubierreDrive(v_s=1.0),
    ]
    cmp_ = compare_drives(drives, box_half_size=2.0, n_grid=16)
    # Both have name "alcubierre" — only last survives; switch to single-call.
    d_slow = AlcubierreDrive(v_s=0.5)
    d_fast = AlcubierreDrive(v_s=1.0)
    ratio = abs(d_fast.total_negative_energy()) / abs(d_slow.total_negative_energy())
    assert 3.0 < ratio < 5.0
