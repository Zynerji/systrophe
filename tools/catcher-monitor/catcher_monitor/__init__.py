"""catcher-monitor: the Systrophe address-space lambda_2 catcher exposed as a detector.

Provenance: this tool packages `systrophe.novelty_catcher` and
`systrophe.derivative_catcher` (the Hamming-graph algebraic-connectivity
diagnostic that has caught 26+ emergents in the Systrophe project's
physics modules and IBM Quantum runs) behind a small, well-typed
public API aimed at three concrete use cases:

  * `find_phase_transition`  -- detect a sharp transition in a
    parameter sweep. Returns a single transition centre + kind
    ("discontinuous" / "smooth_sigmoid" / "none") + confidence.
  * `find_anomalies`         -- flag samples that are disconnected
    from the rest in address-space. Useful for out-of-distribution
    detection on activations, sensor readings, or scientific data.
  * `TrainingMonitor`        -- streaming/online variant for
    monitoring a scalar (e.g. training loss) over time; reports
    sharp transitions in a rolling window.

Distinction from the Systrophe core:
  * The core's `scan_novelty` is a low-level parametric scanner.
    This tool wraps it for the most-common cases and returns
    plain dataclasses rather than the internal `NoveltyScanResult`.
  * The core does not include sample-set anomaly detection
    (no parameter axis); this tool builds that on top of the same
    Hamming-graph primitive.
  * The core does not include streaming monitoring; this tool
    adds a windowed wrapper.
"""

from .detection import (
    AnomalyResult,
    Emergent,
    PhaseTransitionResult,
    find_anomalies,
    find_phase_transition,
    scan_emergents,
)
from .monitor import (
    MonitorEvent,
    TrainingMonitor,
)

__all__ = [
    "AnomalyResult",
    "Emergent",
    "PhaseTransitionResult",
    "find_anomalies",
    "find_phase_transition",
    "scan_emergents",
    "MonitorEvent",
    "TrainingMonitor",
]

__version__ = "0.1.0"
