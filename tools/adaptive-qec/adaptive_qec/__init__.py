"""adaptive-qec: anomaly-gated QEC decoder (catcher + Dijkstra-MWPM + full MWPM).

Scaffold. Implementation pending.
"""

__version__ = "0.0.0-scaffold"


def gating_threshold_default() -> float:
    """Default catcher λ_2 jump threshold beyond which the decoder
    falls back from fast Dijkstra to full MWPM."""
    return 0.5


__all__ = ["gating_threshold_default"]
