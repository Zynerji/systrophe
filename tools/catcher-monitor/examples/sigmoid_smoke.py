"""Smoke example: phase-transition detection on a quantised sigmoid.

Runs in milliseconds; useful to verify the tool is wired correctly.
"""

from __future__ import annotations

import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from catcher_monitor import find_phase_transition


def main():
    centre_truth = 2.5
    params = np.linspace(0.0, 5.0, 60)

    def fn(x: float) -> float:
        return round(1.0 / (1.0 + math.exp(-50.0 * (x - centre_truth))), 2)

    t0 = time.time()
    res = find_phase_transition(params, fn)
    elapsed = time.time() - t0

    print(f"truth:    {centre_truth}")
    print(f"detected: {res.transition_at}")
    print(f"kind:     {res.kind}")
    print(f"conf:     {res.confidence:.3f}")
    print(f"err:      {abs(res.transition_at - centre_truth):.4f}")
    print(f"elapsed:  {elapsed:.4f} s")


if __name__ == "__main__":
    main()
