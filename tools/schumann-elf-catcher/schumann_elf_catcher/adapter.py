"""ELF / Schumann-resonance adapter for the Systrophe catcher stack.

This reparameterizes the ``gw-burst-catcher`` time-frequency pipeline for the
Schumann band (~4-48 Hz at ~256 Hz sample rate) and adds three things the GW
tool lacks but that honest ELF anomaly hunting requires:

  1. **Chunked streaming.** The address-space catcher builds an O(n^2) Hamming
     matrix, so a whole hour at a 100 ms hop (~36k windows) is infeasible. We
     whiten the hour once, then slide the catcher over short chunks and
     aggregate candidate transients with absolute timestamps.

  2. **A surrogate (phase-randomized) null.** Phase randomization preserves the
     power spectrum exactly -- the induction-coil response, the Schumann comb,
     mains lines -- but destroys phase coherence, so genuine transients vanish.
     If the real recording's largest Hamming step exceeds the surrogate
     distribution, the transient is real and not a spectral-shape artifact.
     Without this, a "detection" means nothing (Systrophe rule: no verdict
     without a null run).

  3. **Trend + regime scans.** ``growth_catcher`` on a per-bin band-RMS track
     catches slow amplitude drift; ``scan_novelty`` on per-bin spectra catches
     regime change-points. Together with the burst scan these cover the four
     targets: transient bursts, slow trends, regime changes, open-ended.

What this is NOT: a calibrated SR observatory pipeline. No instrument-response
deconvolution, no glitch veto, no absolute FAR. Every flag is a *candidate*
that the surrogate null either supports or rejects.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

# Reuse the gw-burst-catcher primitives (sibling tool, not pip-installed).
_TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GW = os.path.join(_TOOLS_DIR, "gw-burst-catcher")
if _GW not in sys.path:
    sys.path.insert(0, _GW)

from gw_burst_catcher.preprocess import whiten_strain, q_transform  # noqa: E402

from systrophe.catchers.growth_catcher import catch_growth  # noqa: E402
from systrophe.catchers.novelty_catcher import scan_novelty  # noqa: E402

# Canonical Schumann modes (Hz), modes 1-5.
SCHUMANN_MODES = (7.83, 14.3, 20.8, 27.3, 33.8)


@dataclass
class BurstCandidate:
    """One transient candidate, located in absolute recording time."""

    time_s: float          # seconds from start of recording
    hamming_step: int      # address-space step size (detection statistic)
    chunk_index: int


@dataclass
class ELFScanResult:
    """Aggregate result of the chunked burst scan over a recording."""

    n_chunks: int
    chunk_s: float
    f_band: tuple[float, float]
    max_hamming_step: int
    max_step_time_s: float
    candidates: list[BurstCandidate] = field(default_factory=list)
    null_p_value: float | None = None
    null_max_steps: list[int] = field(default_factory=list)
    null_threshold_95: float | None = None


# --------------------------------------------------------------------------- #
# Burst scan (chunked) + surrogate null
# --------------------------------------------------------------------------- #

def _prep(samples: np.ndarray) -> np.ndarray:
    """Remove DC offset; return float64."""
    x = np.asarray(samples, dtype=float)
    return x - x.mean()


def _encode_and_step(feats: np.ndarray, n_bits: int):
    """Value-thermometer address encoding + consecutive Hamming steps.

    Address-space encoding in the spirit of
    ``systrophe.catchers.novelty_catcher.real_array_to_address``: each band's
    per-window log-energy is binned by VALUE into a thermometer over the
    band's global [min, max] across the chunk, addresses are concatenated and
    truncated to ``n_bits``. Value-binning (vs the rank-thermometer used for
    smooth parametric scans) is the right choice for noisy time series: flat
    background stays in one bin (small Hamming steps) while a genuine
    transient jumps bins in many bands at once (large step). We compute only
    the *consecutive* Hamming steps (the burst statistic), not the full
    O(n^2) matrix or lambda_2, so the surrogate null stays tractable.
    """
    n_windows, n_comp = feats.shape
    M = feats.astype(float, copy=True)
    for c in range(n_comp):
        col = M[:, c]
        bad = ~np.isfinite(col)
        if bad.any():
            good = col[~bad]
            col[bad] = float(np.median(good)) if good.size else 0.0
    bpc = max(1, n_bits // max(n_comp, 1))
    vmin = M.min(axis=0)
    vmax = M.max(axis=0)
    span = vmax - vmin
    span[span <= 0] = 1.0
    bin_idx = np.clip(((M - vmin) / span * bpc).astype(int), 0, bpc)
    thresholds = np.arange(bpc)
    addr = (bin_idx[:, :, None] > thresholds[None, None, :]).astype(np.int8)
    addr = addr.reshape(n_windows, n_comp * bpc)
    if addr.shape[1] > n_bits:
        addr = addr[:, :n_bits]
    if n_windows < 2:
        return addr, np.zeros(0, dtype=int)
    steps = np.sum(addr[1:] != addr[:-1], axis=1).astype(int)
    return addr, steps


def _fast_qtile_scan(
    qt_grid: np.ndarray, sample_rate: float,
    window_ms: float, hop_ms: float, n_bits: int,
):
    """Vectorized replacement for ``catcher_scan_qtile`` (consecutive steps
    only). Returns (verdict, max_step, max_step_time_s, sharp_features,
    window_centres_s, steps).
    """
    n_freq, n_time = qt_grid.shape
    window_samples = max(1, int(window_ms * 1e-3 * sample_rate))
    hop_samples = max(1, int(hop_ms * 1e-3 * sample_rate))
    n_windows = max(1, (n_time - window_samples) // hop_samples + 1)

    qt2 = qt_grid ** 2
    feats = np.empty((n_windows, n_freq))
    centres = np.empty(n_windows)
    for w in range(n_windows):
        i0 = w * hop_samples
        i1 = min(i0 + window_samples, n_time)
        feats[w] = np.log(np.sum(qt2[:, i0:i1], axis=1) + 1e-12)
        centres[w] = (i0 + i1) / 2 / sample_rate

    _addr, steps = _encode_and_step(feats, n_bits)

    sharp: list[dict] = []
    if n_windows >= 3 and len(steps) > 0:
        median_step = float(np.median(steps))
        mad = float(np.median(np.abs(steps - median_step)))
        absolute_floor = max(int(0.25 * n_bits), 4)
        outlier_floor = max(3.0 * mad, float(absolute_floor))
        for k in range(len(steps)):
            step = int(steps[k])
            excess = step - median_step
            if step >= absolute_floor and excess >= outlier_floor:
                sharp.append({
                    "window_centre_s": float(centres[k + 1]),
                    "hamming_step": step,
                })

    if len(steps) == 0 or np.all(steps == 0):
        verdict = "uniform"
        sharp = []
        max_step, max_time = 0, 0.0
    else:
        verdict = "novel_structure" if sharp else "smooth"
        j = int(np.argmax(steps))
        max_step = int(steps[j])
        max_time = float(centres[j + 1])
    return verdict, max_step, max_time, sharp, centres, steps


def burst_scan(
    samples: np.ndarray,
    sample_rate: float,
    f_min: float = 6.0,
    f_max: float = 45.0,
    n_freq: int = 24,
    Q: float = 4.0,
    chunk_s: float = 60.0,
    window_ms: float = 500.0,
    hop_ms: float = 100.0,
    n_bits: int = 64,
    whiten: bool = True,
    sharp_only: bool = False,
) -> ELFScanResult:
    """Whiten once, then slide the catcher over ``chunk_s`` chunks.

    Returns transient candidates (the catcher's sharp Hamming-step
    windows) located in absolute recording time, plus the single largest
    Hamming step -- the headline detection statistic that the surrogate
    null is calibrated against.
    """
    x = _prep(samples)
    if whiten:
        x = whiten_strain(x, sample_rate=int(round(sample_rate)))

    chunk_n = int(round(chunk_s * sample_rate))
    n_chunks = max(1, len(x) // chunk_n)

    candidates: list[BurstCandidate] = []
    global_max = 0
    global_max_time = 0.0

    for c in range(n_chunks):
        seg = x[c * chunk_n:(c + 1) * chunk_n]
        if len(seg) < int(0.5 * chunk_n):
            continue
        qt_grid, _f = q_transform(
            seg, sample_rate=int(round(sample_rate)),
            f_min=f_min, f_max=f_max, n_freq=n_freq, Q=Q,
        )
        _verdict, max_step, max_time, sharp, _c, _s = _fast_qtile_scan(
            qt_grid, sample_rate=sample_rate,
            window_ms=window_ms, hop_ms=hop_ms, n_bits=n_bits,
        )
        offset = c * chunk_s
        if max_step > global_max:
            global_max = int(max_step)
            global_max_time = offset + max_time
        if not sharp_only:
            for s in sharp:
                candidates.append(BurstCandidate(
                    time_s=offset + s["window_centre_s"],
                    hamming_step=int(s["hamming_step"]),
                    chunk_index=c,
                ))

    candidates.sort(key=lambda b: b.hamming_step, reverse=True)
    return ELFScanResult(
        n_chunks=n_chunks, chunk_s=chunk_s, f_band=(f_min, f_max),
        max_hamming_step=global_max, max_step_time_s=global_max_time,
        candidates=candidates,
    )


def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomized surrogate: identical PSD, destroyed phase coherence."""
    n = len(x)
    X = np.fft.rfft(x)
    mag = np.abs(X)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=mag.shape)
    phase[0] = 0.0                      # keep DC real
    if n % 2 == 0:
        phase[-1] = 0.0                 # keep Nyquist real
    return np.fft.irfft(mag * np.exp(1j * phase), n=n)


def burst_null_test(
    samples: np.ndarray,
    sample_rate: float,
    n_surrogates: int = 60,
    seed: int = 0,
    **scan_kwargs,
) -> ELFScanResult:
    """Run ``burst_scan`` on the real data and on phase-randomized surrogates.

    Populates ``null_p_value`` (fraction of surrogates whose largest Hamming
    step >= the real one), the surrogate max-step samples, and the 95th
    percentile threshold. A small p-value means the transient structure is
    *not* explained by the recording's power spectrum alone.
    """
    real = burst_scan(samples, sample_rate, **scan_kwargs)

    x = _prep(samples)
    rng = np.random.default_rng(seed)
    null_steps: list[int] = []
    for _ in range(n_surrogates):
        surr = phase_randomize(x, rng)
        r = burst_scan(surr, sample_rate, sharp_only=True, **scan_kwargs)
        null_steps.append(int(r.max_hamming_step))

    null_arr = np.asarray(null_steps, dtype=float)
    p = float((np.sum(null_arr >= real.max_hamming_step) + 1) / (n_surrogates + 1))
    real.null_p_value = p
    real.null_max_steps = null_steps
    real.null_threshold_95 = float(np.percentile(null_arr, 95)) if len(null_arr) else None
    return real


# --------------------------------------------------------------------------- #
# Trend scan (slow drift) + regime change-point scan
# --------------------------------------------------------------------------- #

def band_rms_track(
    samples: np.ndarray,
    sample_rate: float,
    band: tuple[float, float] = (6.0, 45.0),
    bin_s: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin RMS amplitude inside ``band``. Returns (bin_centre_s, rms)."""
    x = _prep(samples)
    bin_n = int(round(bin_s * sample_rate))
    n_bins = max(1, len(x) // bin_n)
    centres = np.empty(n_bins)
    rms = np.empty(n_bins)
    freqs = np.fft.rfftfreq(bin_n, 1.0 / sample_rate)
    bmask = (freqs >= band[0]) & (freqs <= band[1])
    for i in range(n_bins):
        seg = x[i * bin_n:(i + 1) * bin_n]
        X = np.fft.rfft(seg * np.hanning(len(seg)))
        # Parseval-style band power -> RMS amplitude in band.
        power = np.sum(np.abs(X[bmask]) ** 2)
        rms[i] = np.sqrt(power) / len(seg)
        centres[i] = (i + 0.5) * bin_s
    return centres, rms


def trend_scan(
    samples: np.ndarray,
    sample_rate: float,
    band: tuple[float, float] = (6.0, 45.0),
    bin_s: float = 30.0,
    n_perm: int = 400,
):
    """Catch a sustained amplitude drift in ``band`` (growth_catcher)."""
    t, rms = band_rms_track(samples, sample_rate, band=band, bin_s=bin_s)
    return catch_growth(
        t, rms, n_perm=n_perm, parameter_label=f"t_s[{band[0]}-{band[1]}Hz]",
    ), (t, rms)


def regime_change_scan(
    samples: np.ndarray,
    sample_rate: float,
    band: tuple[float, float] = (6.0, 45.0),
    n_subbands: int = 16,
    bin_s: float = 30.0,
    n_bits: int = 32,
):
    """Scan per-bin sub-band spectra for sharp regime change-points."""
    x = _prep(samples)
    bin_n = int(round(bin_s * sample_rate))
    n_bins = max(1, len(x) // bin_n)
    freqs = np.fft.rfftfreq(bin_n, 1.0 / sample_rate)
    edges = np.linspace(band[0], band[1], n_subbands + 1)

    spectra = np.zeros((n_bins, n_subbands))
    for i in range(n_bins):
        seg = x[i * bin_n:(i + 1) * bin_n]
        X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        for b in range(n_subbands):
            m = (freqs >= edges[b]) & (freqs < edges[b + 1])
            spectra[i, b] = np.log(np.sum(X[m]) + 1e-12)

    def fn(idx_float):
        i = int(round(idx_float))
        i = max(0, min(n_bins - 1, i))
        return spectra[i]

    scan = scan_novelty(
        np.arange(n_bins, dtype=float), fn, n_bits=n_bits,
        data_adaptive=True, parameter_label="time_bin",
    )
    # Map flagged bin indices to absolute times.
    change_points = [
        {"time_s": float(s["between_indices"][1] * bin_s),
         "hamming_step": int(s["hamming_step"])}
        for s in scan.sharp_features
    ]
    return scan.verdict, change_points, (spectra, bin_s)
