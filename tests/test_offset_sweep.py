"""Tests for the SystrophePair convenience constructors and offset sweep."""

import numpy as np
import pytest

from systrophe import SystrophePair, VanStockumInterior


def test_from_cylinders_zero_offset_doubles_amplitude():
    """Zero offset between two identical cylinders gives 2x the F amplitude."""
    cyl1 = VanStockumInterior(omega=1.0, R=1.0)
    cyl2 = VanStockumInterior(omega=1.0, R=1.0)
    pair = SystrophePair.from_cylinders(cyl1, cyl2, delta_offset=0.0)
    rs = np.array([1.5, 2.0, 3.0])
    L_pair = pair.L(rs)
    # Single cylinder F values
    s = cyl1.tipler_sinusoid()
    L_single = s.L(rs)
    np.testing.assert_allclose(L_pair, 2.0 * L_single, rtol=1e-10)


def test_from_cylinders_pi_offset_kills_pair():
    """delta_offset = pi between identical cylinders cancels the envelope."""
    cyl1 = VanStockumInterior(omega=1.5, R=1.0)
    cyl2 = VanStockumInterior(omega=1.5, R=1.0)
    pair = SystrophePair.from_cylinders(cyl1, cyl2, delta_offset=np.pi)
    rs = np.linspace(1.5, 5.0, 50)
    np.testing.assert_allclose(pair.L(rs), 0.0, atol=1e-10)


def test_offset_sweep_pi_local_minimum():
    """Sweeping the offset, log_measure of CTC bands has a minimum near pi.

    Anti-phase superposition kills the Tipler sinusoid; CTC content vanishes.
    """
    cyl1 = VanStockumInterior(omega=1.2, R=1.0)
    cyl2 = VanStockumInterior(omega=1.2, R=1.0)
    pair = SystrophePair.from_cylinders(cyl1, cyl2, delta_offset=0.0)
    offsets = np.linspace(0.0, 2 * np.pi, 17)
    out = pair.offset_sweep(r_min=1.05, r_max=20.0, offsets=offsets)
    measures = out["log_measures"]
    # Index of the global minimum should be at or near pi (index ~8 of 17)
    i_min = int(np.argmin(measures))
    assert 6 <= i_min <= 10, f"min log_measure at i={i_min}, offset={offsets[i_min]}"
    # And the minimum should be much smaller than the zero-offset measure
    assert measures[i_min] < 0.05 * measures[0]


def test_total_ctc_log_measure_zero_offset_positive():
    """Zero-offset pair has CTC bands and a positive total log measure."""
    cyl = VanStockumInterior(omega=1.5, R=1.0)
    pair = SystrophePair.from_cylinders(cyl, cyl, delta_offset=0.0)
    measure = pair.total_ctc_log_measure(r_min=1.05, r_max=50.0)
    assert measure > 0.5


def test_from_cylinders_rejects_subcritical():
    cyl_sub = VanStockumInterior(omega=0.3, R=1.0)
    cyl_sup = VanStockumInterior(omega=1.0, R=1.0)
    with pytest.raises(ValueError):
        SystrophePair.from_cylinders(cyl_sub, cyl_sup)


def test_from_cylinders_type_check():
    with pytest.raises(TypeError):
        SystrophePair.from_cylinders("not_a_cylinder", VanStockumInterior(omega=1.0, R=1.0))
