"""adaptive-qec: anomaly-gated QEC decoder.

The fast path is naive MWPM (single-qubit-boundary-flip); the slow
path is full Dijkstra-MWPM (the +5-25pp Heron-r2 win). The gate is a
sliding-window catcher on the syndrome stream. When the window's
mean syndrome weight crosses a threshold, subsequent shots use the
slow / accurate decoder.
"""

from __future__ import annotations

from .gating import (
    AnomalyGatedDecoder,
    GateDecision,
    SyndromeWindowStats,
)
from .synthetic import (
    SyntheticShot,
    generate_shots,
)

__version__ = "0.1.0"

__all__ = [
    "AnomalyGatedDecoder",
    "GateDecision",
    "SyndromeWindowStats",
    "SyntheticShot",
    "generate_shots",
    "gating_threshold_default",
]


def gating_threshold_default() -> float:
    """Default mean-syndrome-weight gate threshold above which the
    decoder falls back from naive to Dijkstra. Calibrated for d=5 on
    synthetic depolarising noise."""
    return 3.0
