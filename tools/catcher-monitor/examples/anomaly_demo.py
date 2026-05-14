"""Anomaly-detection demo: 99 Gaussian samples + 1 outlier.

The outlier should rank top-1 in the address-space distance score.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from catcher_monitor import find_anomalies


def main():
    rng = np.random.default_rng(42)
    normal = rng.normal(loc=0.0, scale=1.0, size=(99, 16))
    outlier = np.full((1, 16), 12.0)
    samples = np.vstack([normal, outlier])

    res = find_anomalies(samples, top_k=5)
    print(f"n samples:    {res.n_samples}")
    print(f"top-5 most anomalous indices (truth = 99):")
    for rank, idx in enumerate(res.anomaly_indices, 1):
        marker = "  <-- planted outlier" if idx == 99 else ""
        print(f"  #{rank}  idx={idx}  score={res.scores[idx]:.3f}{marker}")
    print()
    print(f"score distribution: min={res.scores.min():.3f}  "
          f"median={np.median(res.scores):.3f}  max={res.scores.max():.3f}")


if __name__ == "__main__":
    main()
