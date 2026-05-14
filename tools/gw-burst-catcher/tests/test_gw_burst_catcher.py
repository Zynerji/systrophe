"""Tests for gw-burst-catcher. No network access required; everything
runs on synthetic strain produced by `gw_burst_catcher.synthetic`."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gw_burst_catcher import (
    BurstDetectionResult,
    KNOWN_EVENTS,
    SyntheticInjection,
    catcher_scan_qtile,
    inject_chirp,
    make_gaussian_noise,
    q_transform,
    whiten_strain,
)
from gw_burst_catcher.detection import detect_burst_in_strain


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


def test_known_events_have_gps_and_detectors():
    assert "GW150914" in KNOWN_EVENTS
    for name, info in KNOWN_EVENTS.items():
        assert "gps" in info and isinstance(info["gps"], (int, float))
        assert "detectors" in info and len(info["detectors"]) >= 1


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------


def test_make_gaussian_noise_shape():
    sr = 1024
    noise = make_gaussian_noise(duration_s=4.0, sample_rate=sr, seed=0)
    assert noise.shape == (4 * sr,)
    assert np.isfinite(noise).all()
    # Reasonable amplitude scale (not all zeros, not blown up)
    assert 1e-3 < float(np.std(noise)) < 1e3


def test_make_gaussian_noise_seed_reproducible():
    a = make_gaussian_noise(2.0, 1024, seed=42)
    b = make_gaussian_noise(2.0, 1024, seed=42)
    np.testing.assert_array_equal(a, b)


def test_inject_chirp_adds_signal():
    sr = 1024
    noise = make_gaussian_noise(4.0, sr, seed=0)
    pre = noise.copy()
    info = inject_chirp(noise, sample_rate=sr, t_inject_s=2.0,
                          f_start=50.0, f_end=250.0,
                          amplitude=5.0, duration_s=0.2)
    assert isinstance(info, SyntheticInjection)
    # Strain has been modified somewhere near the injection time
    diff = noise - pre
    assert np.any(diff != 0)
    # Energy concentrated near the injection
    n = len(noise)
    inj_idx = int(2.0 * sr)
    near_inj = np.sum(diff[inj_idx - 200: inj_idx + 200] ** 2)
    elsewhere = float(np.sum(diff ** 2) - near_inj)
    assert near_inj > elsewhere


def test_inject_chirp_outside_strain_raises():
    sr = 256
    noise = make_gaussian_noise(1.0, sr, seed=0)
    with pytest.raises(ValueError):
        inject_chirp(noise, sample_rate=sr, t_inject_s=10.0,
                       f_start=20, f_end=100, amplitude=1.0, duration_s=0.1)


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------


def test_whiten_strain_preserves_length():
    sr = 1024
    noise = make_gaussian_noise(4.0, sr, seed=0)
    w = whiten_strain(noise, sample_rate=sr)
    assert w.shape == noise.shape
    assert np.isfinite(w).all()


def test_whiten_strain_flattens_spectrum_roughly():
    """After whitening, the band-averaged spectral shape should be
    much flatter than the coloured-noise input.

    The coloured noise has a 3x bump in [100, 300] Hz vs the [30, 100]
    Hz baseline; after whitening, that bump should be largely gone.
    Use band averages of |FFT| rather than per-bin max/min, which is
    Rayleigh-noisy regardless of whitening.
    """
    sr = 1024
    noise = make_gaussian_noise(8.0, sr, seed=1)
    s_raw = np.abs(np.fft.rfft(noise))
    s_white = np.abs(np.fft.rfft(whiten_strain(noise, sr)))
    freqs = np.fft.rfftfreq(len(noise), 1.0 / sr)

    def band_avg(spec, lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(np.mean(spec[m])) if np.any(m) else 0.0

    raw_bump_ratio = band_avg(s_raw, 100.0, 300.0) / max(band_avg(s_raw, 30.0, 100.0), 1e-12)
    white_bump_ratio = band_avg(s_white, 100.0, 300.0) / max(band_avg(s_white, 30.0, 100.0), 1e-12)
    assert white_bump_ratio < raw_bump_ratio, (
        f"whitening did not flatten the spectrum: "
        f"raw bump/baseline = {raw_bump_ratio:.2f}, "
        f"whitened = {white_bump_ratio:.2f}"
    )


def test_q_transform_shape():
    sr = 1024
    noise = make_gaussian_noise(4.0, sr, seed=0)
    qt, freqs = q_transform(noise, sample_rate=sr,
                              f_min=30.0, f_max=400.0, n_freq=32)
    assert qt.shape == (32, len(noise))
    assert freqs.shape == (32,)
    # Monotonic log-spaced frequencies
    assert np.all(np.diff(freqs) > 0)


# ---------------------------------------------------------------------------
# Catcher detection
# ---------------------------------------------------------------------------


def test_catcher_flags_injection_in_noise():
    """An injected chirp should produce a max-Hamming-step centred near
    the injection time."""
    sr = 1024
    duration = 8.0
    t_inj = 4.0
    rng_seed = 0
    strain = make_gaussian_noise(duration, sr, seed=rng_seed)
    inject_chirp(strain, sample_rate=sr, t_inject_s=t_inj,
                   f_start=50.0, f_end=250.0,
                   amplitude=10.0, duration_s=0.3)
    res = detect_burst_in_strain(strain, sample_rate=sr,
                                    f_min=30.0, f_max=400.0, n_freq=48,
                                    window_ms=200.0, hop_ms=50.0)
    assert isinstance(res, BurstDetectionResult)
    # Detection time should be within +/-0.5 s of the truth injection
    assert abs(res.max_hamming_step_time_s - t_inj) < 0.5, (
        f"max-hamming at {res.max_hamming_step_time_s}s, truth {t_inj}s"
    )
    # The detection statistic should be clearly above zero
    assert res.max_hamming_step > 0


def test_catcher_quiet_on_pure_noise():
    """Pure Gaussian noise should NOT produce a strong narrow detection."""
    sr = 1024
    strain = make_gaussian_noise(8.0, sr, seed=2)
    res = detect_burst_in_strain(strain, sample_rate=sr,
                                    f_min=30.0, f_max=400.0, n_freq=32,
                                    window_ms=200.0, hop_ms=50.0)
    # The verdict can be "smooth" or "uniform"; key check is that the
    # max-step is at most modestly above the typical hamming step.
    if res.hamming_steps_trajectory:
        median = float(np.median(res.hamming_steps_trajectory))
        # Allow up to 2.5x median for chance fluctuations on synthetic noise
        assert res.max_hamming_step <= 2.5 * max(median, 1.0) + 5


def test_catcher_scan_returns_dataclass():
    sr = 1024
    strain = make_gaussian_noise(4.0, sr, seed=3)
    qt, _f = q_transform(strain, sample_rate=sr, n_freq=24,
                            f_min=30, f_max=400)
    res = catcher_scan_qtile(qt, sample_rate=sr, window_ms=200,
                                hop_ms=80, n_bits=32)
    assert isinstance(res, BurstDetectionResult)
    assert isinstance(res.window_centres_s, list)
    assert isinstance(res.hamming_steps_trajectory, list)
    assert res.n_windows > 0


def test_max_hamming_step_time_in_segment():
    """The reported time should be within [0, duration]."""
    sr = 1024
    duration = 4.0
    strain = make_gaussian_noise(duration, sr, seed=4)
    res = detect_burst_in_strain(strain, sample_rate=sr, n_freq=24,
                                    window_ms=200, hop_ms=100)
    assert 0.0 <= res.max_hamming_step_time_s <= duration


# ---------------------------------------------------------------------------
# Run-event runner refuses without gwpy / network
# ---------------------------------------------------------------------------


def test_fetch_strain_unknown_event_raises():
    from gw_burst_catcher.detection import fetch_strain_gwosc
    with pytest.raises(ValueError):
        fetch_strain_gwosc("not_a_real_event")
