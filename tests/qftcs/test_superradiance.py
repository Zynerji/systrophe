"""Tests for rotational-superradiance band physics (numpy-only)."""

from __future__ import annotations

import numpy as np
import pytest

from systrophe.geometry.vanstockum import VanStockumInterior
from systrophe.qftcs.superradiance import (
    SuperradianceBand,
    amplification_follows_F_sign,
    band_superradiance_at,
    predicted_pair_probability,
    squeezing_angle,
    superradiance_scan,
)


@pytest.fixture
def vs():
    # omega R = 2 supercritical: 5 ergosurfaces, clean alternating bands.
    return VanStockumInterior(omega=2.0, R=1.0)


def test_squeezing_angle_zero_for_passive():
    assert squeezing_angle(0.0) == 0.0
    assert squeezing_angle(-3.0) == 0.0  # clamped


def test_squeezing_angle_monotone_and_bounded():
    etas = np.linspace(0.0, 10.0, 50)
    thetas = np.array([squeezing_angle(e) for e in etas])
    assert np.all(np.diff(thetas) >= 0)            # monotone non-decreasing
    assert np.all(thetas < np.pi / 2)              # bounded below pi/2


def test_pair_probability_identity():
    # P_pair = sin^2(theta) = eta^2/(1+eta^2)
    for eta in [0.0, 0.2, 0.5, 1.0, 3.0]:
        theta = squeezing_angle(eta)
        p = predicted_pair_probability(theta)
        assert p == pytest.approx(eta ** 2 / (1.0 + eta ** 2), abs=1e-12)
        assert 0.0 <= p < 1.0


def test_ergoregion_has_pair_creation_passive_has_none(vs):
    # An ergoregion radius (F<0) with finite negative energy.
    erg = band_superradiance_at(vs, 2.4)
    assert erg.F < 0
    assert erg.is_ergoregion
    assert erg.energy_min < 0
    assert erg.pair_probability > 0

    # A passive radius (F>0) just outside the second ergosurface.
    pas = band_superradiance_at(vs, 3.4)
    assert pas.F > 0
    assert not pas.is_ergoregion
    assert pas.pair_probability == 0.0


def test_pair_creation_iff_ergoregion_over_scan(vs):
    # Over a scan, pair_probability > 0 must coincide exactly with F < 0
    # *when* the negative-energy band is well defined (finite E_min<0).
    radii = np.linspace(1.6, 6.0, 45)
    bands = superradiance_scan(vs, radii)
    for b in bands:
        if b.pair_probability > 0:
            # spontaneous emission only inside an ergoregion
            assert b.is_ergoregion
            assert b.energy_min < 0


def test_amplification_follows_F_sign_summary(vs):
    radii = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7]
    bands = superradiance_scan(vs, radii)
    summary = amplification_follows_F_sign(bands)
    assert summary["all_agree"]
    # Ergoregion band creates pairs; passive control stays dark.
    assert summary["mean_pair_ergoregion"] > 0.05
    assert summary["max_pair_passive"] == 0.0


def test_band_descriptor_types(vs):
    b = band_superradiance_at(vs, 2.4)
    assert isinstance(b, SuperradianceBand)
    assert np.isfinite(b.squeezing_angle)
    assert b.pair_probability == predicted_pair_probability(b.squeezing_angle)


@pytest.mark.parametrize("r", [2.2, 2.4, 2.6])
def test_truncated_tms_circuit_matches_prediction_if_qiskit(vs, r):
    """Noiseless wiring check: the Ry(2 theta)+CX circuit yields P(11)=sin^2 theta."""
    qi = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    b = band_superradiance_at(vs, r)
    qc = qi.QuantumCircuit(2)
    qc.ry(2.0 * b.squeezing_angle, 0)
    qc.cx(0, 1)
    probs = Statevector.from_instruction(qc).probabilities_dict()
    p11 = probs.get("11", 0.0)
    assert p11 == pytest.approx(b.pair_probability, abs=1e-9)
