"""gw-burst-catcher: an unmodeled gravitational-wave burst detector built on
the Systrophe address-space lambda_2 catcher.

Pipeline:
  1. Acquire strain (real GWOSC fetch or synthetic injection).
  2. PSD-whiten in the frequency domain.
  3. Constant-Q band-pass time-frequency representation.
  4. Slide a window across the time-frequency grid; for each window
     bit-encode the per-band log-energy via rank-thermometer hashing
     and run the address-space catcher.
  5. Report per-window Hamming step + detection statistic.

This is the catcher's GW use case -- "detect a sharp transition in
the time-frequency content of a strain stream, with no template
bank". The same primitive that flags emergents in physics modules
flags unmodeled-burst candidates here.

Provenance: ported from `src/systrophe/gw_catcher.py` (266 LOC,
authored mid-session; previously referenced by the
`O3 unsearched-sample scan` 5000-segment dry-run in commit 3999ef6).
The promotion to a proper tool adds:

  * **Synthetic-injection mode** (`synthetic.py`) so tests don't
    need network access to GWOSC.
  * **Test suite** that injects a known chirp into Gaussian noise
    and asserts the catcher flags it at the right time.
  * Separation of acquisition / preprocessing / detection into
    independently-importable modules.

The full GWOSC fetcher (`fetch_strain_gwosc`) is still here for
real-event runs; it just lives behind a `gwpy` optional dep.
"""

from .events import KNOWN_EVENTS
from .preprocess import (
    q_transform,
    whiten_strain,
)
from .detection import (
    BurstDetectionResult,
    catcher_scan_qtile,
    run_event_catcher,
)
from .synthetic import (
    SyntheticInjection,
    inject_chirp,
    make_gaussian_noise,
)

__all__ = [
    "KNOWN_EVENTS",
    "q_transform",
    "whiten_strain",
    "BurstDetectionResult",
    "catcher_scan_qtile",
    "run_event_catcher",
    "SyntheticInjection",
    "inject_chirp",
    "make_gaussian_noise",
]

__version__ = "0.1.0"
