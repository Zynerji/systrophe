"""Tests for the Tipler-Krasnikov composite warp drive."""

import numpy as np

from systrophe.tipler_krasnikov_hybrid import (
    hybrid_NEC_radial,
    hybrid_total_negative_energy,
    novelty_scan,
    tipler_tilt_at,
)
from systrophe.vanstockum import VanStockumInterior


def test_tilt_zero_inside_cylinder():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    assert tipler_tilt_at(vs, 0.5) == 0.0


def test_tilt_nonnegative_outside():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    rs = np.linspace(1.05, 8.0, 20)
    for r in rs:
        assert tipler_tilt_at(vs, float(r)) >= 0.0


def test_hybrid_NEC_reduces_below_pure_krasnikov():
    """At a radius where Tipler tilt is non-zero, hybrid |NEC| <= pure Krasnikov |NEC|."""
    from systrophe.krasnikov_tube import krasnikov_NEC_radial
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r_inside_band = 3.0  # within first CTC band for a=1
    nec_pure = krasnikov_NEC_radial(0.0, t=1.0, alpha=4.0)
    nec_hybrid = hybrid_NEC_radial(vs, 0.0, 1.0, r_inside_band,
                                     alpha=4.0, coupling=1.0)
    assert abs(nec_hybrid) <= abs(nec_pure) + 1e-12


def test_hybrid_zero_coupling_equals_pure_krasnikov():
    """coupling=0 -> hybrid reduces to pure Krasnikov NEC exactly."""
    from systrophe.krasnikov_tube import krasnikov_NEC_radial
    vs = VanStockumInterior(omega=1.0, R=1.0)
    nec_pure = krasnikov_NEC_radial(0.0, t=1.0, alpha=4.0)
    nec_hybrid = hybrid_NEC_radial(vs, 0.0, 1.0, 3.0,
                                     alpha=4.0, coupling=0.0)
    assert abs(nec_hybrid - nec_pure) < 1e-12


def test_hybrid_total_energy_nonpositive():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    e = hybrid_total_negative_energy(vs, 3.0, alpha=4.0, coupling=1.0)
    assert e <= 0


def test_novelty_scan_returns_verdict():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    res = novelty_scan(vs, r_range=(1.05, 8.0), n_r=20,
                        alpha_range=(2.0, 8.0), n_alpha=6,
                        coupling=1.0)
    assert "novelty_verdict" in res
    assert res["novelty_verdict"] in ("smooth", "uniform", "novel_structure")
