"""Tests for the ER=EPR quantum information channel."""

import numpy as np
import pytest

from systrophe.quantum_info.erepr_channel import (
    EREPRChannelReport,
    build_channel_report,
    channel_entanglement_fidelity,
    gjw_coupling_scan,
    summarise_channel,
    teleport_qubit,
    _single_qubit_state,
)


def test_single_qubit_teleport_is_perfect():
    rng = np.random.default_rng(0)
    worst = 1.0
    for _ in range(100):
        th = rng.uniform(0, np.pi)
        ph = rng.uniform(0, 2 * np.pi)
        _, F = teleport_qubit(_single_qubit_state(th, ph))
        worst = min(worst, F)
    assert worst > 0.999999  # deterministic, fidelity 1


def test_entanglement_fidelity_beats_classical_and_is_perfect():
    f1 = channel_entanglement_fidelity(1)
    f2 = channel_entanglement_fidelity(2)
    assert f1 == pytest.approx(1.0, abs=1e-9)
    assert f2 == pytest.approx(1.0, abs=1e-9)
    assert f1 > 0.5 and f2 > 0.5  # above the classical bound


def test_multi_qubit_register_preserved():
    """A 3-qubit entangled register transmits with unit entanglement fidelity."""
    assert channel_entanglement_fidelity(3) == pytest.approx(1.0, abs=1e-9)


def test_channel_is_not_ftl():
    r = build_channel_report()
    assert isinstance(r, EREPRChannelReport)
    assert r.is_faster_than_light is False
    assert r.requires_coupling_between_mouths is True


def test_gjw_haar_signature_is_honestly_absent():
    """A generic Haar scrambler does NOT activate the GJW channel (no
    size-winding) -- fidelity stays near the no-coupling baseline."""
    scan = gjw_coupling_scan(n=3)
    assert scan["gjw_signature_present"] is False
    assert scan["peak_fidelity"] < 0.55       # stays near baseline


def test_real_syk_scrambler_activates_the_channel():
    """A real SYK scrambler ACTIVATES coupling-mediated transmission where Haar
    does not: the size-winding lift is large and >> the Haar lift."""
    from systrophe.quantum_info.erepr_channel import syk_vs_haar_activation
    a = syk_vs_haar_activation(nq=3, seeds=(1, 2, 3))
    assert a["syk_activates_channel"] is True
    assert a["syk_lift"] > 0.05
    assert a["syk_lift"] > 3 * abs(a["haar_lift"])


def test_syk_falls_short_of_classical_bound_at_small_N():
    """Honest: at classically-simulable N the SYK wormhole transmission does not
    reach the 0.5 classical bound (needs larger N / engineered winding)."""
    from systrophe.quantum_info.erepr_channel import syk_vs_haar_activation
    a = syk_vs_haar_activation(nq=3, seeds=(1, 2, 3))
    assert a["reaches_classical_bound"] is False
    assert "SYK" in a["note"]


def test_syk_transmission_runs():
    from systrophe.quantum_info.erepr_channel import syk_wormhole_transmission
    f = syk_wormhole_transmission(nq=3, t=0.5, g=4.0, seed=1)
    assert 0.0 <= f <= 1.0


def test_summary():
    s = summarise_channel(build_channel_report())
    assert "ER=EPR channel" in s and "FTL=False" in s
