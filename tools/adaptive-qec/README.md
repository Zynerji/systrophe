# adaptive-qec

**Anomaly-gated QEC decoder: catcher-watched Dijkstra-MWPM with full-MWPM fallback.**

Extends `tools/dijkstra-mwpm/` with an online catcher gate:

1. Run Dijkstra-MWPM as the fast default decoder
2. The address-space novelty catcher monitors the syndrome stream's λ₂ in a sliding window
3. When λ₂ spikes beyond a calibrated threshold, the syndrome distribution has drifted into a regime where Dijkstra under-decodes — switch to full MWPM (or a neural decoder) for those rounds
4. The Phase 2a power-law-fit machinery characterizes the syndrome-stream divergence (sub-threshold vs near-threshold regime)

## Why this exists

Dijkstra-MWPM trades optimality for speed and wins +5–25 pp on synthetic
benchmarks against full MWPM in some regimes — but degrades in others.
Adaptive-qec keeps the speed win in the common regime and the accuracy
win in anomalous rounds, gated automatically by the catcher.

## Status

**Scaffold.** Not yet implemented. Smoke tests pass on the package
structure. Implementation: TBD.

## Layout

```
adaptive-qec/
├── README.md
├── adaptive_qec/
│   ├── __init__.py
│   └── gating.py     (TODO: AnomalyGatedDecoder)
└── tests/
    └── test_smoke.py
```

## License

MIT.
