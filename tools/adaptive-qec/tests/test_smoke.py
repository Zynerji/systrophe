"""adaptive-qec tests."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools" / "dijkstra-mwpm"))

from adaptive_qec import (
    AnomalyGatedDecoder,
    GateDecision,
    SyndromeWindowStats,
    SyntheticShot,
    generate_shots,
    gating_threshold_default,
)
from dijkstra_mwpm import (
    decode_with_dijkstra_mwpm,
    decode_with_naive_mwpm,
)


# ---------------------------------------------------------------------------
# Synthetic shot generator
# ---------------------------------------------------------------------------


def test_generate_shots_shape_d3():
    shots = generate_shots(d=3, p_error=0.01, n_shots=5, n_rounds=2)
    assert len(shots) == 5
    for s in shots:
        assert isinstance(s, SyntheticShot)
        assert len(s.data_bits) == 9
        assert len(s.syndromes_per_round) == 2
        assert all(len(r) == 4 for r in s.syndromes_per_round)  # d=3 has 4 Z-stabs


def test_generate_shots_no_noise_is_clean():
    """p_error = 0 => no flips => data and syndromes all zero."""
    shots = generate_shots(d=3, p_error=0.0, n_shots=3, n_rounds=2)
    for s in shots:
        assert s.n_errors_total == 0
        assert all(b == 0 for b in s.data_bits)
        assert all(all(b == 0 for b in r) for r in s.syndromes_per_round)


def test_generate_shots_higher_p_more_errors():
    s_low = generate_shots(d=5, p_error=0.001, n_shots=50, n_rounds=3, seed=0)
    s_high = generate_shots(d=5, p_error=0.05, n_shots=50, n_rounds=3, seed=0)
    avg_low = sum(s.n_errors_total for s in s_low) / len(s_low)
    avg_high = sum(s.n_errors_total for s in s_high) / len(s_high)
    assert avg_high > avg_low


def test_generate_invalid_distance_raises():
    with pytest.raises(ValueError):
        generate_shots(d=4, p_error=0.01, n_shots=1)


# ---------------------------------------------------------------------------
# AnomalyGatedDecoder basic
# ---------------------------------------------------------------------------


def test_decoder_constructs():
    dec = AnomalyGatedDecoder(d=5, window_size=32)
    assert dec.d == 5
    assert dec.window_size == 32
    assert dec.per_shot_threshold == 10  # 2 * 5
    assert dec.window_threshold == 2.0    # 0.4 * 5


def test_decoder_invalid_distance_raises():
    with pytest.raises(ValueError):
        AnomalyGatedDecoder(d=4)


def test_decoder_returns_0_or_1():
    dec = AnomalyGatedDecoder(d=3)
    shots = generate_shots(d=3, p_error=0.01, n_shots=5, n_rounds=2)
    for s in shots:
        out = dec.decode(s.data_bits, list(s.syndromes_per_round))
        assert out in (0, 1)


def test_decoder_zero_noise_zero_logical():
    """No noise => no syndromes => fast path => logical = 0."""
    dec = AnomalyGatedDecoder(d=3)
    shots = generate_shots(d=3, p_error=0.0, n_shots=10, n_rounds=2)
    for s in shots:
        out = dec.decode(s.data_bits, list(s.syndromes_per_round))
        assert out == 0
    assert dec.fraction_slow == 0.0


def test_decoder_high_noise_triggers_slow_path():
    """At very high noise the per-shot or window threshold should fire."""
    dec = AnomalyGatedDecoder(d=3, window_size=8)
    shots = generate_shots(d=3, p_error=0.2, n_shots=20, n_rounds=3, seed=1)
    for s in shots:
        dec.decode(s.data_bits, list(s.syndromes_per_round))
    assert dec.fraction_slow > 0.3, (
        f"expected gate to fire on high-noise; fraction_slow={dec.fraction_slow}"
    )


def test_decoder_history_recorded():
    dec = AnomalyGatedDecoder(d=3)
    shots = generate_shots(d=3, p_error=0.05, n_shots=15, n_rounds=2)
    for s in shots:
        dec.decode(s.data_bits, list(s.syndromes_per_round))
    assert len(dec.history) == 15
    assert all(isinstance(h, GateDecision) for h in dec.history)


def test_decoder_window_stats():
    dec = AnomalyGatedDecoder(d=3, window_size=8)
    shots = generate_shots(d=3, p_error=0.05, n_shots=12, n_rounds=2)
    for s in shots:
        dec.decode(s.data_bits, list(s.syndromes_per_round))
    s = dec.stats()
    assert isinstance(s, SyndromeWindowStats)
    assert s.n_filled == 8  # window_size, after sending 12 shots through
    assert s.window_size == 8


def test_decoder_reset_clears_state():
    dec = AnomalyGatedDecoder(d=3, window_size=8)
    shots = generate_shots(d=3, p_error=0.05, n_shots=5, n_rounds=2)
    for s in shots:
        dec.decode(s.data_bits, list(s.syndromes_per_round))
    assert len(dec.history) == 5
    dec.reset()
    assert len(dec.history) == 0
    assert dec.stats().n_filled == 0


def test_decoder_batch_matches_per_shot():
    dec = AnomalyGatedDecoder(d=3)
    shots = generate_shots(d=3, p_error=0.05, n_shots=10, n_rounds=2)
    db_list = [s.data_bits for s in shots]
    syn_list = [list(s.syndromes_per_round) for s in shots]
    batch = dec.decode_batch(db_list, syn_list)
    dec2 = AnomalyGatedDecoder(d=3)
    one_at_a_time = [
        dec2.decode(s.data_bits, list(s.syndromes_per_round)) for s in shots
    ]
    assert batch == one_at_a_time


# ---------------------------------------------------------------------------
# End-to-end benchmark: gated must be accurate at high p, fast at low p
# ---------------------------------------------------------------------------


def _acc_with_fn(fn, shots, d) -> float:
    correct = 0
    for s in shots:
        if fn(s.data_bits, list(s.syndromes_per_round), d) == 0:
            correct += 1
    return correct / len(shots)


def _acc_with_decoder(dec, shots) -> float:
    correct = 0
    for s in shots:
        if dec.decode(s.data_bits, list(s.syndromes_per_round)) == 0:
            correct += 1
    return correct / len(shots)


def test_gated_at_least_naive_accuracy_at_moderate_p():
    """At moderate noise the gated decoder must NOT do worse than naive."""
    d = 3
    shots = generate_shots(d=d, p_error=0.06, n_shots=200, n_rounds=2, seed=7)
    acc_naive = _acc_with_fn(decode_with_naive_mwpm, shots, d)
    dec = AnomalyGatedDecoder(d=d, window_size=16, per_shot_threshold=2,
                                window_threshold=0.5)
    acc_gated = _acc_with_decoder(dec, shots)
    assert acc_gated >= acc_naive - 0.02, (
        f"gated {acc_gated} < naive {acc_naive}"
    )


def test_gated_stays_on_fast_at_clean_data():
    """Clean data => gate should NOT fire."""
    d = 5
    shots = generate_shots(d=d, p_error=0.0, n_shots=50, n_rounds=2, seed=0)
    dec = AnomalyGatedDecoder(d=d, window_size=16)
    for s in shots:
        dec.decode(s.data_bits, list(s.syndromes_per_round))
    assert dec.fraction_slow == 0.0


def test_gating_threshold_default():
    assert gating_threshold_default() == 3.0
