"""AnomalyGatedDecoder (TODO).

Intended interface:

    decoder = AnomalyGatedDecoder(
        fast=DijkstraMWPM(...),
        slow=FullMWPM(...),
        catcher_window=64,
        gate_threshold=0.5,
    )
    correction = decoder.decode(syndrome_round)

Catcher runs on the rolling-window syndrome-address stream; if the
window's λ_2 jumps above gate_threshold, the next decode uses
`slow`. Otherwise `fast`.
"""

from __future__ import annotations


class AnomalyGatedDecoder:
    """Placeholder; implementation pending."""

    def __init__(self, fast=None, slow=None, catcher_window: int = 64,
                 gate_threshold: float = 0.5) -> None:
        self.fast = fast
        self.slow = slow
        self.catcher_window = int(catcher_window)
        self.gate_threshold = float(gate_threshold)

    def decode(self, syndrome) -> bytes:
        raise NotImplementedError(
            "AnomalyGatedDecoder.decode is not yet implemented. "
            "Scaffold only.",
        )
