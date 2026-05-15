"""AnomalyGatedDecoder: sliding-window anomaly gate over a syndrome stream.

Architecture
------------
* Fast path = naive MWPM (single-qubit-boundary-flip)
* Slow path = Dijkstra-MWPM (full chain correction; Heron-r2 +5-25pp win)
* Gate      = rolling window of recent syndromes; per-window mean
              syndrome weight + per-shot weight thresholds. When either
              triggers, the current shot uses the slow path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from dijkstra_mwpm import (
    MWPMDecoder,
    decode_with_dijkstra_mwpm,
    decode_with_naive_mwpm,
)


@dataclass(frozen=True)
class SyndromeWindowStats:
    """Snapshot of the gate's rolling-window state."""
    window_size: int
    n_filled: int
    mean_weight: float
    max_weight: int


@dataclass(frozen=True)
class GateDecision:
    """Per-shot decision record."""
    used_slow_path: bool
    per_shot_weight: int
    window_mean_weight: float
    reason: str


class AnomalyGatedDecoder:
    """Catcher-gated QEC decoder.

    Parameters
    ----------
    d : int
        Surface code distance (odd, >= 3).
    window_size : int
        Number of recent shots tracked for the rolling-window mean.
    per_shot_threshold : int
        If the current shot's total syndrome weight exceeds this,
        always use the slow path. Defaults to ``2 * d`` (heuristic).
    window_threshold : float
        If the rolling-window MEAN syndrome weight exceeds this, the
        gate stays open (slow path) for subsequent shots until the
        mean drops back below.
    """

    def __init__(
        self,
        d: int,
        window_size: int = 64,
        per_shot_threshold: int | None = None,
        window_threshold: float | None = None,
    ) -> None:
        if d < 3 or d % 2 == 0:
            raise ValueError(f"d must be odd and >= 3 (got {d})")
        self.d = int(d)
        self.window_size = int(window_size)
        self._dijkstra = MWPMDecoder(d=self.d)
        self._z_stabs = self._dijkstra.Z_stabs
        if per_shot_threshold is None:
            per_shot_threshold = 2 * self.d
        if window_threshold is None:
            # Mean weight scales with d (more stabs => more background
            # weight even at low noise)
            window_threshold = 0.4 * self.d
        self.per_shot_threshold = int(per_shot_threshold)
        self.window_threshold = float(window_threshold)
        self._window: deque[int] = deque(maxlen=self.window_size)
        self._history: list[GateDecision] = []

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> SyndromeWindowStats:
        if not self._window:
            return SyndromeWindowStats(
                window_size=self.window_size, n_filled=0,
                mean_weight=0.0, max_weight=0,
            )
        return SyndromeWindowStats(
            window_size=self.window_size,
            n_filled=len(self._window),
            mean_weight=float(np.mean(self._window)),
            max_weight=int(np.max(self._window)),
        )

    @property
    def history(self) -> list[GateDecision]:
        return list(self._history)

    @property
    def fraction_slow(self) -> float:
        if not self._history:
            return 0.0
        return sum(1 for h in self._history if h.used_slow_path) / len(self._history)

    # ------------------------------------------------------------------
    # Core decode
    # ------------------------------------------------------------------

    def _total_syndrome_weight(
        self, syndromes_per_round: list[tuple[int, ...]] | list[list[int]],
    ) -> int:
        """Sum of all violated Z-stabs across all rounds for one shot."""
        return int(sum(sum(row) for row in syndromes_per_round))

    def decode(
        self,
        data_bits: tuple[int, ...] | list[int],
        syndromes_per_round: list[tuple[int, ...]] | list[list[int]] | None = None,
    ) -> int:
        """Decode a single shot. Updates the rolling window + gate state."""
        syndromes_per_round = syndromes_per_round or []
        w = self._total_syndrome_weight(syndromes_per_round)
        self._window.append(w)
        stats = self.stats()

        if w > self.per_shot_threshold:
            use_slow, reason = True, "per_shot_weight_exceeded"
        elif (stats.mean_weight > self.window_threshold
              and stats.n_filled >= max(8, self.window_size // 4)):
            use_slow, reason = True, "window_mean_weight_exceeded"
        else:
            use_slow, reason = False, "below_threshold"

        if use_slow:
            result = decode_with_dijkstra_mwpm(
                data_bits, syndromes_per_round, self.d,
            )
        else:
            result = decode_with_naive_mwpm(
                data_bits, syndromes_per_round, self.d,
            )

        self._history.append(GateDecision(
            used_slow_path=bool(use_slow),
            per_shot_weight=int(w),
            window_mean_weight=stats.mean_weight,
            reason=reason,
        ))
        return int(result)

    def decode_batch(
        self,
        data_bits_list: list[tuple[int, ...]],
        syndromes_list: list[list[tuple[int, ...]]],
    ) -> list[int]:
        if len(data_bits_list) != len(syndromes_list):
            raise ValueError("data_bits_list and syndromes_list must have same length")
        return [self.decode(d, s) for d, s in zip(data_bits_list, syndromes_list)]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._window.clear()
        self._history.clear()
