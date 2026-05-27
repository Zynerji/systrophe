"""Streaming TrainingMonitor: detect sharp transitions in an online scalar series.

Intended for training-loss monitoring, sensor streams, or any scalar
time-series where you want to be alerted when a phase-transition-like
event happens. Maintains a rolling window of the last `window` samples,
detrends a linear baseline, and runs the Systrophe value-level
`scan_novelty` over the residuals. Detrending is important because the
catcher is built for non-monotonic parameter scans; a smoothly
decreasing loss + the rank-thermometer encoder + the derivative-scan
in `catch_smooth_transition` produces false-positive sigmoid hits on
any windowed monotonic signal.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty


@dataclass
class MonitorEvent:
    """Per-update report from `TrainingMonitor.update`."""
    step: int
    value: float
    is_anomaly: bool                    # True when a sharp transition was detected in the window
    transition_kind: str                # "discontinuous" | "smooth_sigmoid" | "none"
    transition_at_step: int | None      # absolute step (not window-relative) where the transition centre was estimated
    window_size: int
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingMonitor:
    """Online catcher for a scalar series.

    Parameters
    ----------
    window:
        Rolling-window size. The catcher only sees the most recent
        `window` samples. Larger windows give more statistical power
        but smear sharp transitions.
    refresh_every:
        Run the catcher only every Nth update (cheaper for long
        training runs). Default 10.
    min_samples:
        Don't run the catcher until at least this many samples are in
        the window. Default = max(16, window // 4).
    n_bits:
        Address width for the catcher.
    cooldown:
        After flagging an anomaly, suppress further alerts for this
        many updates (avoids flapping when an event spans many steps).
    """

    def __init__(
        self,
        window: int = 200,
        refresh_every: int = 10,
        min_samples: int | None = None,
        n_bits: int = 32,
        cooldown: int = 20,
        magnitude_z: float = 4.0,
    ) -> None:
        self.window = int(window)
        self.refresh_every = int(refresh_every)
        self.min_samples = int(min_samples) if min_samples is not None else max(16, window // 4)
        self.n_bits = int(n_bits)
        self.cooldown = int(cooldown)
        # Catcher flags can fire on noise patterns in a detrended window.
        # We post-gate: require the flagged sample's residual to be at
        # least `magnitude_z` * (median absolute deviation) above the
        # window's residual baseline. 4 MAD is robust to ~1% false-
        # positive rate on Gaussian noise.
        self.magnitude_z = float(magnitude_z)

        self._values: deque = deque(maxlen=window)
        self._steps: deque = deque(maxlen=window)
        self._update_count: int = 0
        self._cooldown_remaining: int = 0
        self._last_alert: MonitorEvent | None = None

    def update(self, value: float, step: int | None = None,
                  metadata: dict | None = None) -> MonitorEvent:
        """Push a new sample and (cheaply) check for an anomaly.

        Returns the event for THIS update, with `is_anomaly` True iff a
        sharp transition was detected in the current window. Events
        during the cooldown period after a prior alert always have
        `is_anomaly=False` (regardless of the catcher result).
        """
        self._update_count += 1
        if step is None:
            step = self._update_count
        self._values.append(float(value))
        self._steps.append(int(step))
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # Default: no anomaly reported on this step
        event = MonitorEvent(
            step=int(step),
            value=float(value),
            is_anomaly=False,
            transition_kind="none",
            transition_at_step=None,
            window_size=len(self._values),
            metadata=dict(metadata or {}),
        )

        # Run catcher only every Nth update and only when we have enough samples.
        should_run = (
            self._update_count % self.refresh_every == 0
            and len(self._values) >= self.min_samples
            and self._cooldown_remaining == 0
        )
        if not should_run:
            return event

        params = np.arange(len(self._values), dtype=float)
        vals = np.array(self._values, dtype=float)

        # Detrend: subtract linear least-squares fit. The catcher fires
        # false positives on smoothly-monotonic windows; detrending
        # removes the trend so only residual jumps survive.
        slope, intercept = np.polyfit(params, vals, 1)
        residual = vals - (slope * params + intercept)

        def fn_(p: float) -> np.ndarray:
            i = int(round(p))
            i = max(0, min(len(residual) - 1, i))
            return np.array([float(residual[i])])

        res = scan_novelty(params, fn_, n_bits=self.n_bits,
                              parameter_label="step")
        if res.verdict == "novel_structure" and res.sharp_features:
            best = max(res.sharp_features,
                       key=lambda s: s.get("excess_over_median", 0.0))
            window_idx = int(best["between_indices"][1])
            window_idx = max(0, min(len(self._steps) - 1, window_idx))
            # Magnitude gate: the flagged sample's residual must be
            # well outside the noise floor measured by the median
            # absolute deviation of the rest of the window.
            mad = float(np.median(np.abs(residual - np.median(residual))))
            if mad == 0.0:
                mad = 1e-12
            z = abs(float(residual[window_idx])) / (1.4826 * mad)
            if z >= self.magnitude_z:
                absolute_step = self._steps[window_idx]
                event.is_anomaly = True
                event.transition_kind = "discontinuous"
                event.transition_at_step = int(absolute_step)
                event.metadata.setdefault("magnitude_z", z)
                self._cooldown_remaining = self.cooldown
                self._last_alert = event

        return event

    def reset(self) -> None:
        """Forget all observed history."""
        self._values.clear()
        self._steps.clear()
        self._update_count = 0
        self._cooldown_remaining = 0
        self._last_alert = None

    @property
    def last_alert(self) -> MonitorEvent | None:
        return self._last_alert
