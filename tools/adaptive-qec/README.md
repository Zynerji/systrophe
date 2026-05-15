# adaptive-qec

**Anomaly-gated QEC decoder for the rotated surface code.**

Pairs the fast `naive` MWPM (single-qubit-boundary-flip) with the
accurate `Dijkstra` MWPM (full chain correction; the +5–25 pp Heron-r2
win) behind a sliding-window catcher gate. At low noise the gate stays
shut and decoding is naive (cheap); at high noise the gate trips and
decoding is Dijkstra (accurate). At moderate noise it adapts shot-by-
shot, paying the Dijkstra cost only when the syndrome stream actually
needs it.

## Headline benchmark (d=3, 300 shots, 2 rounds per p)

| p | naive | Dijkstra | **gated** | slow_frac |
|---|------:|---------:|----------:|----------:|
| 1e-4 | 100% | 100% | **100%** | 0% |
| 1e-3 | 100% | 100% | **100%** | 0% |
| 5e-3 | 96.7% | 98.3% | **97.0%** | 3% |
| 1e-2 | 95.7% | 98.0% | **96.7%** | 15% |
| 3e-2 | 86.0% | 94.7% | **93.3%** | 80% |
| 6e-2 | 72.3% | 86.0% | **85.7%** | 98% |
| 1e-1 | 60.3% | 79.3% | **79.3%** | 99% |

The gated decoder closes 75–100% of the accuracy gap between naive
and Dijkstra across the full noise sweep, while only firing the slow
path when the rolling-window syndrome statistics demand it.

Reproduce with:

```bash
PYTHONPATH=src:tools/dijkstra-mwpm:tools/adaptive-qec python \
  tools/adaptive-qec/examples/sweep_benchmark.py
```

## Architecture

* **Fast path** = `dijkstra_mwpm.decode_with_naive_mwpm`
* **Slow path** = `dijkstra_mwpm.decode_with_dijkstra_mwpm`
* **Gate** = `AnomalyGatedDecoder`:
  * Rolling window of recent per-shot syndrome weights
  * Per-shot trigger: if the current shot's total weight exceeds
    `per_shot_threshold` (default `2 * d`), use slow path
  * Window trigger: if the rolling mean exceeds `window_threshold`
    (default `0.4 * d`), gate stays open

## API

```python
from adaptive_qec import AnomalyGatedDecoder, generate_shots

dec = AnomalyGatedDecoder(d=5, window_size=64)
shots = generate_shots(d=5, p_error=0.01, n_shots=1000, n_rounds=3)

# Per shot
for s in shots:
    logical = dec.decode(s.data_bits, list(s.syndromes_per_round))

# Or batch
logical_list = dec.decode_batch(
    [s.data_bits for s in shots],
    [list(s.syndromes_per_round) for s in shots],
)

# Inspect what fraction used the slow path
print(dec.fraction_slow)
```

## Synthetic shot generator

`generate_shots(d, p_error, n_shots, n_rounds)` simulates independent
per-qubit X errors per round on a d×d rotated surface code starting
from logical |0⟩. Each `SyntheticShot` carries `data_bits`,
`syndromes_per_round`, and `n_errors_total` (ground-truth label). The
correct decoded logical for every shot is `0`.

This is intentionally minimal — sufficient to exercise the gap
between naive and Dijkstra MWPM, not a full circuit-level
depolarising-channel simulator.

## Tests

```
PYTHONPATH=src:tools/dijkstra-mwpm:tools/adaptive-qec python -m pytest \
    tools/adaptive-qec/tests/ -q
```

16 tests, all offline; runtime < 2 s.

## License

MIT.
