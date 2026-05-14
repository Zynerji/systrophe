"""Synthetic strain generators for offline testing.

The Systrophe project's MEMORY note about `feedback_test_before_long_runs`
applies here: real-event GW data lives behind a GWOSC fetcher that
needs network access + `gwpy`. Tests and CI should use synthetic
injections so they're deterministic and fast.

We provide:

  * `make_gaussian_noise(duration_s, sample_rate, asd, seed)` --
    Gaussian noise with a frequency-domain shape that mimics a
    LIGO-like detector amplitude spectral density (a simple bucket
    around 200 Hz; not a calibrated curve, but enough to exercise the
    whitening + Q-transform + catcher pipeline).

  * `inject_chirp(strain, sample_rate, t_inject_s, f_start, f_end,
                   amplitude, duration_s)` -- in-place chirp injection
    that ramps from `f_start` to `f_end` over `duration_s` centred on
    `t_inject_s`.

The chirp's parameters are loose imitations of a stellar-mass binary
inspiral; we are NOT trying to match a particular waveform family
(no PN inspiral, no IMR), just to put a clean broadband transient
into the strain that the catcher must locate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticInjection:
    """Record of what was injected; returned by `inject_chirp`."""
    t_inject_s: float
    f_start: float
    f_end: float
    amplitude: float
    duration_s: float


def make_gaussian_noise(
    duration_s: float,
    sample_rate: int,
    asd_floor: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Coloured Gaussian noise with a LIGO-like ASD bump around 200 Hz.

    Returns a 1-D float array of length `int(duration_s * sample_rate)`.
    The shape is hand-tuned to behave reasonably under whitening; it
    is NOT a calibrated detector ASD.
    """
    n = int(duration_s * sample_rate)
    rng = np.random.default_rng(seed)
    # White noise in time domain
    white = rng.normal(size=n)
    # Move to frequency, apply a soft bandpass envelope, back to time
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    # Bandpass envelope: rise from 20 Hz, peak ~200 Hz, fall by 1 kHz
    envelope = np.ones_like(freqs) * asd_floor
    envelope = np.where(freqs < 10.0, 1e-3, envelope)
    envelope = np.where(
        (freqs > 100.0) & (freqs < 300.0),
        envelope * 3.0,
        envelope,
    )
    envelope = np.where(freqs > 1500.0, asd_floor * 0.5, envelope)
    coloured = spectrum * envelope
    out = np.fft.irfft(coloured, n=n)
    return out


def inject_chirp(
    strain: np.ndarray,
    sample_rate: int,
    t_inject_s: float,
    f_start: float = 50.0,
    f_end: float = 250.0,
    amplitude: float = 5.0,
    duration_s: float = 0.2,
    envelope: str = "tanh_window",
) -> SyntheticInjection:
    """Add a quadratic-chirp burst to `strain` in place.

    The chirp signal is
        s(t) = amplitude * w(t) * sin(2 pi (f0 t + 0.5 alpha t^2))
    with `f0 = f_start`, `alpha = (f_end - f_start) / duration_s`, and
    `w(t)` a smooth window (default tanh-edged plateau).
    """
    n = len(strain)
    t_inject_samples = int(round(t_inject_s * sample_rate))
    inj_samples = int(round(duration_s * sample_rate))
    if inj_samples < 4:
        raise ValueError(f"duration_s={duration_s} too short at sr={sample_rate}")
    half = inj_samples // 2
    i0 = max(0, t_inject_samples - half)
    i1 = min(n, t_inject_samples + half)
    if i1 - i0 < 4:
        raise ValueError(
            f"Injection extends outside strain: t_inject_s={t_inject_s}, "
            f"strain duration = {n / sample_rate} s"
        )
    t = np.arange(i1 - i0) / sample_rate
    alpha = (f_end - f_start) / duration_s
    phase = 2 * np.pi * (f_start * t + 0.5 * alpha * t * t)
    raw = amplitude * np.sin(phase)

    # Smooth window so the burst has well-defined edges
    if envelope == "tanh_window":
        edge = max(2, len(t) // 10)
        w = np.ones_like(t)
        w[:edge] = 0.5 * (1.0 + np.tanh(np.linspace(-3.0, 3.0, edge)))
        w[-edge:] = 0.5 * (1.0 + np.tanh(np.linspace(3.0, -3.0, edge)))
    elif envelope == "hann":
        w = np.hanning(len(t))
    else:
        raise ValueError(f"unknown envelope: {envelope!r}")

    strain[i0:i1] += w * raw
    return SyntheticInjection(
        t_inject_s=t_inject_s, f_start=f_start, f_end=f_end,
        amplitude=amplitude, duration_s=duration_s,
    )
