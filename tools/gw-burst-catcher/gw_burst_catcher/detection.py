"""Burst-detection front end: slide a window across the Q-transform,
hash each window's per-band log-energy to an address, run the Systrophe
catcher, and report per-window Hamming-step + summary detection statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty

from .preprocess import q_transform, whiten_strain
from .events import KNOWN_EVENTS


@dataclass
class BurstDetectionResult:
    """Per-detector burst-detection report.

    Attributes
    ----------
    verdict
        Catcher verdict: "novel_structure" | "smooth" | "uniform".
    n_sharp_features
        Number of windows the catcher flagged as Hamming-step outliers.
    sharp_features
        Per-flagged-window metadata: indices, Hamming step, centre time
        in seconds within the strain segment.
    max_hamming_step
        Largest Hamming step between consecutive windows -- the
        load-bearing detection statistic.
    max_hamming_step_time_s
        Time within the segment (seconds, from t=0) of the largest
        Hamming step.
    n_windows
        Number of sliding windows the catcher processed.
    window_centres_s
        Centre times (s) of every window, for plotting.
    hamming_steps_trajectory
        Hamming step between window k-1 and window k, for k=1..n-1.
    """
    verdict: str
    n_sharp_features: int
    sharp_features: list[dict]
    max_hamming_step: float
    max_hamming_step_time_s: float
    n_windows: int
    window_centres_s: list[float]
    hamming_steps_trajectory: list[int]


def catcher_scan_qtile(
    qt_grid: np.ndarray,
    sample_rate: int,
    window_ms: float = 250.0,
    hop_ms: float = 50.0,
    n_bits: int = 32,
) -> BurstDetectionResult:
    """Slide a window across `qt_grid`, run the catcher, summarise.

    Parameters
    ----------
    qt_grid
        Time-frequency grid from `preprocess.q_transform`, shape
        `(n_freq, n_time)`.
    sample_rate
        Sample rate of the underlying strain in Hz.
    window_ms, hop_ms
        Sliding-window size and stride in milliseconds.
    n_bits
        Address width passed through to the catcher.
    """
    n_freq, n_time = qt_grid.shape
    window_samples = int(window_ms * 1e-3 * sample_rate)
    hop_samples = max(1, int(hop_ms * 1e-3 * sample_rate))
    n_windows = max(1, (n_time - window_samples) // hop_samples + 1)

    window_features = np.zeros((n_windows, n_freq))
    window_centres = np.zeros(n_windows)
    for w in range(n_windows):
        i0 = w * hop_samples
        i1 = min(i0 + window_samples, n_time)
        window_features[w] = np.log(
            np.sum(qt_grid[:, i0:i1] ** 2, axis=1) + 1e-12
        )
        window_centres[w] = (i0 + i1) / 2 / sample_rate

    def fn(idx_float):
        i = int(round(idx_float))
        i = max(0, min(n_windows - 1, i))
        return window_features[i]

    indices = np.arange(n_windows, dtype=float)
    scan = scan_novelty(
        indices, fn, n_bits=n_bits,
        data_adaptive=True,
        parameter_label="window_idx",
    )

    addresses = scan.addresses
    hamming_steps = []
    for k in range(1, len(addresses)):
        diff = int(np.sum(addresses[k] != addresses[k - 1]))
        hamming_steps.append(diff)
    if hamming_steps:
        max_idx = int(np.argmax(hamming_steps))
        max_step = float(hamming_steps[max_idx])
        max_time = float(window_centres[max_idx + 1])
    else:
        max_step = 0.0
        max_time = 0.0

    sharp = [
        {
            "between_indices": list(s["between_indices"]),
            "window_centre_s": float(window_centres[s["between_indices"][1]]),
            "hamming_step": int(s["hamming_step"]),
        }
        for s in scan.sharp_features
    ]

    return BurstDetectionResult(
        verdict=scan.verdict,
        n_sharp_features=len(scan.sharp_features),
        sharp_features=sharp,
        max_hamming_step=max_step,
        max_hamming_step_time_s=max_time,
        n_windows=int(n_windows),
        window_centres_s=window_centres.tolist(),
        hamming_steps_trajectory=hamming_steps,
    )


def detect_burst_in_strain(
    strain: np.ndarray,
    sample_rate: int,
    psd_segment: tuple[float, float] | None = None,
    f_min: float = 32.0,
    f_max: float = 512.0,
    n_freq: int = 64,
    Q: float = 8.0,
    window_ms: float = 250.0,
    hop_ms: float = 50.0,
    n_bits: int = 32,
) -> BurstDetectionResult:
    """End-to-end on a single 1-D strain array: whiten + Q-transform + catcher."""
    whitened = whiten_strain(strain, sample_rate=sample_rate, psd_segment=psd_segment)
    qt_grid, _freqs = q_transform(
        whitened, sample_rate=sample_rate,
        f_min=f_min, f_max=f_max, n_freq=n_freq, Q=Q,
    )
    return catcher_scan_qtile(
        qt_grid, sample_rate=sample_rate,
        window_ms=window_ms, hop_ms=hop_ms, n_bits=n_bits,
    )


# ---------------------------------------------------------------------------
# Optional real-event runner (needs gwpy + internet to GWOSC)
# ---------------------------------------------------------------------------


def fetch_strain_gwosc(
    event_name: str, detector: str = "H1",
    duration_s: float = 32.0, sample_rate: int = 4096,
) -> tuple[np.ndarray, float, dict]:
    """Fetch strain around a known event from GWOSC. Requires `gwpy`.

    Returns `(strain_array, t0_gps, metadata)`. The strain is centred
    on the event GPS time (event at t = duration / 2 within the array).
    """
    if event_name not in KNOWN_EVENTS:
        raise ValueError(f"Unknown event: {event_name}")
    info = KNOWN_EVENTS[event_name]
    gps_centre = info["gps"]
    t0 = gps_centre - duration_s / 2
    t1 = t0 + duration_s

    try:
        from gwpy.timeseries import TimeSeries
    except ImportError as exc:
        raise ImportError(
            "fetch_strain_gwosc requires gwpy. Install with "
            "`pip install gwpy` or use detect_burst_in_strain on "
            "your own pre-loaded strain array."
        ) from exc

    ts = TimeSeries.fetch_open_data(
        detector, t0, t1, sample_rate=sample_rate, cache=True,
    )
    return np.asarray(ts.value), float(t0), {
        "detector": detector,
        "event": event_name,
        "sample_rate": int(sample_rate),
        "gps_centre": float(gps_centre),
        "duration_s": float(duration_s),
    }


def run_event_catcher(
    event_name: str = "GW150914",
    detectors: tuple[str, ...] = ("H1", "L1"),
    duration_s: float = 32.0,
    sample_rate: int = 4096,
) -> dict:
    """End-to-end catcher run on a known GWOSC event across multiple detectors.

    Reports per-detector burst detection + a time-of-flight
    coincidence check between the first two detectors in the list
    (in milliseconds).
    """
    out: dict = {
        "event": event_name, "duration_s": duration_s,
        "sample_rate": sample_rate, "per_detector": {},
    }
    for detector in detectors:
        strain, t0_gps, _meta = fetch_strain_gwosc(
            event_name, detector=detector,
            duration_s=duration_s, sample_rate=sample_rate,
        )
        res = detect_burst_in_strain(strain, sample_rate=sample_rate)
        event_t = duration_s / 2
        out["per_detector"][detector] = {
            "verdict": res.verdict,
            "n_sharp_features": res.n_sharp_features,
            "sharp_features": res.sharp_features,
            "max_hamming_step": res.max_hamming_step,
            "max_hamming_step_time_s": res.max_hamming_step_time_s,
            "n_windows": res.n_windows,
            "t0_gps": t0_gps,
            "event_time_in_window_s": event_t,
            "max_hamming_offset_from_event_s": (
                res.max_hamming_step_time_s - event_t
            ),
        }
    if len(detectors) >= 2:
        t1 = out["per_detector"][detectors[0]]["max_hamming_step_time_s"]
        t2 = out["per_detector"][detectors[1]]["max_hamming_step_time_s"]
        out["tof_coincidence_ms"] = float(abs(t1 - t2) * 1000.0)
    return out
