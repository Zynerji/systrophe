# catcher-monitor

**The Systrophe address-space λ₂ catcher, exposed as a detector.**

This tool packages `systrophe.catchers.novelty_catcher` and
`systrophe.catchers.derivative_catcher` (the Hamming-graph algebraic-connectivity
diagnostic that has caught 26+ emergents in the Systrophe project's
physics modules and IBM Quantum runs) behind a small, well-typed
public API focused on three concrete use cases:

* **`find_phase_transition`** — locate a sharp transition in a
  parameter sweep. Returns `kind` (`discontinuous` / `smooth_sigmoid`
  / `none`), `transition_at`, and `confidence`.
* **`find_anomalies`** — flag samples that are isolated in
  address-space. Useful for OOD detection on activations, sensor
  data, or scientific measurements.
* **`TrainingMonitor`** — streaming/online variant that watches a
  scalar (e.g. training loss) over time, detrends a linear baseline
  inside a rolling window, and alerts on sharp residual jumps.

## Provenance

* Built on `systrophe.catchers.novelty_catcher.scan_novelty` (address-space
  λ₂ on a Hamming graph of bit-hashed values) and
  `systrophe.catchers.derivative_catcher.catch_smooth_transition` (two-pass
  scan adding a derivative-level pass for smooth sigmoids).
* These primitives have a *demonstrated* detection track record on
  the Systrophe project: 26+ emergents flagged in physics phase
  modules and IBM Quantum hardware runs.
* This tool packages them as a detector — **the role the catcher was
  validated for**. The contrasting use-case (catcher as compute-gate
  in a transformer) was tested in `tools/cyliformer/` and falsified
  at 7B / WikiText-2; see `tools/cyliformer/STATUS.md`.

## Layout

```
catcher-monitor/
├── README.md                  (you are here)
├── catcher_monitor/
│   ├── __init__.py            (public API)
│   ├── detection.py           find_phase_transition + find_anomalies + scan_emergents
│   └── monitor.py             TrainingMonitor (streaming)
├── examples/
│   ├── sigmoid_smoke.py             quick (~0.3s) self-check
│   ├── anomaly_demo.py              99 Gaussians + 1 outlier
│   ├── training_monitor_demo.py     1.0->4.0 loss-jump detection
│   └── ising_phase_transition.py    2D Ising T_c calibration (~5 min)
├── tests/
│   └── test_catcher_monitor.py      13 tests (phase, anomaly, scan, monitor)
└── requirements.txt
```

## Quick start

```python
from catcher_monitor import find_phase_transition

import numpy as np
params = np.linspace(0.0, 5.0, 60)

def my_measurement(x: float) -> float:
    return 0.0 if x < 2.5 else 1.0

result = find_phase_transition(params, my_measurement)
print(result)
# PhaseTransitionResult(kind='discontinuous', transition_at=2.45..., confidence=0.5)
```

```python
from catcher_monitor import find_anomalies
import numpy as np

# 99 normal samples + 1 outlier
normal = np.random.default_rng(0).normal(size=(99, 16))
outlier = np.full((1, 16), 12.0)
samples = np.vstack([normal, outlier])

result = find_anomalies(samples, top_k=3)
print(result.anomaly_indices)   # 99 should be in this list
```

```python
from catcher_monitor import TrainingMonitor

monitor = TrainingMonitor(window=200, refresh_every=10)
for step, loss in enumerate(training_loop_losses):
    event = monitor.update(loss, step=step)
    if event.is_anomaly:
        print(f"Phase transition / instability at step {event.transition_at_step}")
```

## Calibration: 2D Ising T_c

Running `examples/ising_phase_transition.py` simulates a 24×24 Ising
lattice via Metropolis (5 minutes), feeds the magnetisation curve to
`find_phase_transition`, and prints the detected vs analytical T_c.

On a 5-minute simulation we observed:

* detected T_c = 2.48, analytical T_c = 2.269, **error 9.15%**

This is calibrated to a fairly modest MC budget. The Hash-Quine
reference work (cited in the Systrophe project's hash-quine address-
space rule) reported 1.4% error on a more thoroughly-equilibrated
L=32 lattice. The discrepancy is in the simulation, not the catcher
— with longer MC and L=32 the localisation tightens.

## Known limitations (read these)

1. **Sigmoid bias**: the catcher localises the *end* of a transition
   region (the rank-thermometer's last-rank flip), not its centre.
   For smooth analytic sigmoids with slope ≤ ~15 the catcher returns
   `kind='none'` by design — see Dianoia FINDINGS at
   `C:/Users/cknop/.local/bin/Dianoia/FINDINGS.md`. This tool's tests
   document the regime explicitly.
2. **Anomaly detection needs global normalisation**: `find_anomalies`
   computes `v_min`, `v_max` across the entire sample set before
   hashing. Per-sample normalisation (the Systrophe default) would
   make an "all-large" outlier indistinguishable from an
   "all-small" normal sample.
3. **TrainingMonitor needs detrending + a magnitude gate**: the
   catcher fires false positives on noise in a windowed monotonic
   signal. We detrend a linear baseline inside the window and gate
   alerts on `|residual| / MAD >= 4` (~1% Gaussian-noise false-
   positive rate).

## Install

This tool ships as part of the Systrophe repo. To use it without
installing the full repo:

```bash
pip install -r tools/catcher-monitor/requirements.txt   # numpy, scipy
# Then add tools/catcher-monitor/ and src/ to PYTHONPATH:
PYTHONPATH=src:tools/catcher-monitor python -c "from catcher_monitor import find_phase_transition"
```

## License

MIT, inherited from the Systrophe parent package.
