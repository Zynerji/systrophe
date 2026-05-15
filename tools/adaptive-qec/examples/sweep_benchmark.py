"""Headline benchmark: naive vs Dijkstra vs gated, sweeping the physical
error rate.

For each `p`:
  - generate N synthetic shots
  - decode with naive MWPM, full Dijkstra-MWPM, and the AnomalyGatedDecoder
  - report logical-error rate (1 - accuracy) and wall clock per decoder

Expected outcome (the design target):
  * At low p (p ~ 1e-3): all three decoders ~equal accuracy; gated stays
    on naive >90% of the time => near-naive wall clock.
  * At high p (p ~ 5e-2): Dijkstra clearly beats naive; gated trips most
    shots to Dijkstra => accuracy ~= Dijkstra, wall clock between.
"""

from __future__ import annotations

import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3] / "tools" / "dijkstra-mwpm"))

from adaptive_qec import AnomalyGatedDecoder, generate_shots
from dijkstra_mwpm import (
    decode_with_dijkstra_mwpm,
    decode_with_naive_mwpm,
)


def acc_and_time(decoder_fn, shots, d):
    t0 = time.perf_counter()
    correct = 0
    for s in shots:
        if decoder_fn(s.data_bits, list(s.syndromes_per_round), d) == 0:
            correct += 1
    dt = time.perf_counter() - t0
    return correct / len(shots), dt


def acc_and_time_decoder(dec, shots):
    t0 = time.perf_counter()
    correct = 0
    for s in shots:
        if dec.decode(s.data_bits, list(s.syndromes_per_round)) == 0:
            correct += 1
    dt = time.perf_counter() - t0
    return correct / len(shots), dt


def main():
    d = 3
    n_shots = 300
    n_rounds = 2
    p_list = [1e-4, 1e-3, 5e-3, 1e-2, 3e-2, 6e-2, 1e-1]

    print(f"adaptive-qec headline benchmark (d={d}, n_shots={n_shots}, "
          f"n_rounds={n_rounds})")
    print()
    print(f"{'p':>8}  {'naive_acc':>10}  {'dijk_acc':>10}  {'gated_acc':>10}  "
          f"{'gate_slow_frac':>14}  {'naive_t':>9}  {'dijk_t':>9}  "
          f"{'gated_t':>9}")
    print("-" * 100)
    for p in p_list:
        shots = generate_shots(d=d, p_error=p, n_shots=n_shots,
                                n_rounds=n_rounds, seed=42)
        acc_n, t_n = acc_and_time(decode_with_naive_mwpm, shots, d)
        acc_d, t_d = acc_and_time(decode_with_dijkstra_mwpm, shots, d)
        dec = AnomalyGatedDecoder(d=d, window_size=16, per_shot_threshold=2,
                                    window_threshold=0.5)
        acc_g, t_g = acc_and_time_decoder(dec, shots)
        print(f"{p:>8.0e}  {acc_n:>10.3f}  {acc_d:>10.3f}  {acc_g:>10.3f}  "
              f"{dec.fraction_slow:>14.2f}  {t_n:>9.3f}s  {t_d:>9.3f}s  "
              f"{t_g:>9.3f}s")
    print()
    print("Reading the table:")
    print("  - At low p, naive_acc ~= dijk_acc, gated stays on naive (slow_frac low)")
    print("  - At high p, naive_acc drops, dijk_acc holds; gated tracks dijk")
    print("  - gated_t between naive_t and dijk_t")


if __name__ == "__main__":
    main()
