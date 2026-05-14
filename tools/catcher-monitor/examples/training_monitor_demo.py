"""TrainingMonitor demo: smooth loss decrease + a synthetic instability event."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from catcher_monitor import TrainingMonitor


def main():
    rng = np.random.default_rng(0)
    losses = []
    for step in range(200):
        if step < 100:
            v = 2.0 - 0.01 * step + 0.03 * rng.normal()           # steady descent
        elif step < 110:
            v = 1.0 - 0.005 * (step - 100) + 0.03 * rng.normal()  # plateau
        else:
            # Sudden instability: loss jumps to 4.0
            v = 4.0 - 0.005 * (step - 110) + 0.05 * rng.normal()
        losses.append(v)

    mon = TrainingMonitor(window=80, refresh_every=5, min_samples=32)
    n_alerts = 0
    alert_steps = []
    for step, v in enumerate(losses):
        ev = mon.update(float(v), step=step)
        if ev.is_anomaly:
            n_alerts += 1
            alert_steps.append((step, ev.transition_at_step,
                                  ev.metadata.get("magnitude_z", 0.0)))

    print(f"Loss trace: 200 steps with a 1.0 -> 4.0 jump at step 110.")
    print(f"Total alerts: {n_alerts}")
    for s, at, z in alert_steps[:5]:
        print(f"  flagged at step {s}, transition placed at step {at}, z={z:.2f}")
    print()
    if any(95 <= at <= 130 for _, at, _ in alert_steps):
        print("PASS: at least one alert placed the transition near the truth (step ~110).")
    else:
        print("FAIL: no alert near the truth jump location.")


if __name__ == "__main__":
    main()
