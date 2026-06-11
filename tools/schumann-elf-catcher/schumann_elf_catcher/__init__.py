"""schumann-elf-catcher: Systrophe catcher stack reparameterized for the
Schumann-resonance / ELF band, with a surrogate (phase-randomized) null.

Targets: transient bursts, slow amplitude trends, spectral regime change-points,
and open-ended anomaly flagging on raw ELF magnetometer recordings.
"""

from .io import ELFRecording, load_sierra_nevada
from .adapter import (
    SCHUMANN_MODES,
    BurstCandidate,
    ELFScanResult,
    burst_scan,
    burst_null_test,
    phase_randomize,
    band_rms_track,
    trend_scan,
    regime_change_scan,
)

__all__ = [
    "ELFRecording",
    "load_sierra_nevada",
    "SCHUMANN_MODES",
    "BurstCandidate",
    "ELFScanResult",
    "burst_scan",
    "burst_null_test",
    "phase_randomize",
    "band_rms_track",
    "trend_scan",
    "regime_change_scan",
]

__version__ = "0.1.0"
