"""SystropheArray (N-cylinder phased array) tests."""

import numpy as np
import pytest

from systrophe import VanStockumInterior, SystropheArray, TiplerSinusoid


def test_construction_validates_min_one():
    with pytest.raises(ValueError):
        SystropheArray(sinusoids=())


def test_construction_validates_supercritical():
    """All sinusoids must be a > 1/2."""
    s_super = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    SystropheArray(sinusoids=(s_super,))  # OK


def test_single_cylinder_array_equals_cylinder():
    """N=1 array L(r) equals single cylinder L(r)."""
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    s = cyl.tipler_sinusoid()
    arr = SystropheArray(sinusoids=(s,))
    rs = np.array([1.5, 2.0, 3.0, 5.0])
    np.testing.assert_allclose(arr.L(rs), s.L(rs), rtol=1e-14)


def test_aligned_pair_doubles_amplitude():
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    arr = SystropheArray.from_cylinders([cyl, cyl], offsets=[0.0, 0.0])
    s = cyl.tipler_sinusoid()
    rs = np.array([1.5, 2.0, 3.0])
    np.testing.assert_allclose(arr.L(rs), 2.0 * s.L(rs), rtol=1e-14)


def test_uniform_phase_comb_extinguishes_for_N_3():
    """Three cylinders at phases 0, 2pi/3, 4pi/3 sum to zero (roots of unity)."""
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    arr = SystropheArray.uniform_phase_comb(cyl, N=3)
    rs = np.linspace(1.05, 5.0, 50)
    np.testing.assert_allclose(arr.L(rs), 0.0, atol=1e-10)


def test_uniform_phase_comb_extinguishes_for_N_4():
    cyl = VanStockumInterior(omega=1.5, R=1.0)
    arr = SystropheArray.uniform_phase_comb(cyl, N=4)
    rs = np.linspace(1.05, 5.0, 50)
    np.testing.assert_allclose(arr.L(rs), 0.0, atol=1e-10)


def test_uniform_phase_comb_no_ctc_bands():
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    arr = SystropheArray.uniform_phase_comb(cyl, N=3)
    bands = arr.ctc_bands(r_min=1.05, r_max=20.0)
    assert bands == []


def test_phasor_collapse_recovers_aligned_amplitude():
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    arr = SystropheArray.from_cylinders([cyl, cyl, cyl], offsets=[0.0, 0.0, 0.0])
    collapsed = arr.to_single_sinusoid()
    s = cyl.tipler_sinusoid()
    assert collapsed.A == pytest.approx(3.0 * s.A, rel=1e-12)


def test_phasor_collapse_phase_correct():
    """Two equal-amplitude sinusoids at phases 0 and pi/2 collapse to amplitude sqrt(2),
    phase pi/4 (clockwise)."""
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=np.pi / 2.0)
    arr = SystropheArray(sinusoids=(s1, s2))
    c = arr.to_single_sinusoid()
    assert c.A == pytest.approx(np.sqrt(2.0), rel=1e-12)
    assert c.delta == pytest.approx(np.pi / 4.0, rel=1e-12)


def test_phasor_collapse_requires_matched():
    """alpha-beat blocks the phasor sum."""
    cyl_a = VanStockumInterior(omega=1.0, R=1.0)
    cyl_b = VanStockumInterior(omega=2.0, R=1.0)  # different a
    arr = SystropheArray.from_cylinders([cyl_a, cyl_b])
    with pytest.raises(ValueError):
        arr.to_single_sinusoid()


def test_from_cylinders_validates_offset_length():
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    with pytest.raises(ValueError):
        SystropheArray.from_cylinders([cyl, cyl, cyl], offsets=[0.0, 0.5])  # length mismatch


def test_uniform_phase_comb_requires_N_at_least_2():
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    with pytest.raises(ValueError):
        SystropheArray.uniform_phase_comb(cyl, N=1)
