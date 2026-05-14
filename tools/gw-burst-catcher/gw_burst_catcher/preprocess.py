"""Strain pre-processing: PSD whitening + constant-Q band-pass."""

from __future__ import annotations

import numpy as np


def whiten_strain(
    strain: np.ndarray,
    sample_rate: int,
    psd_segment: tuple[float, float] | None = None,
    welch_nperseg: int = 2048,
) -> np.ndarray:
    """Whiten `strain` by dividing by sqrt(PSD) in the frequency domain.

    Parameters
    ----------
    strain
        1-D float array of the time-domain strain to whiten.
    sample_rate
        Sample rate in Hz.
    psd_segment
        `(start_frac, end_frac)` range of the strain to use as the
        quiet baseline for PSD estimation. Default: first 25% of the
        segment. Pick a window that is event-free for best whitening.
    welch_nperseg
        Welch PSD segment length. Defaults to 2048; clipped to the
        baseline length if shorter.
    """
    if psd_segment is None:
        psd_segment = (0.0, 0.25)
    n = len(strain)
    start_idx = int(psd_segment[0] * n)
    end_idx = int(psd_segment[1] * n)
    baseline = strain[start_idx:end_idx]
    if len(baseline) < 16:
        raise ValueError(
            f"PSD baseline too short ({len(baseline)} samples); "
            f"widen psd_segment or use a longer input"
        )

    from scipy.signal import welch
    f_psd, psd = welch(
        baseline, fs=sample_rate,
        nperseg=min(welch_nperseg, len(baseline)),
    )

    spectrum = np.fft.rfft(strain)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)

    psd_interp = np.interp(freqs, f_psd, psd, left=psd[0], right=psd[-1])
    # Avoid divide-by-zero at DC
    if len(psd_interp) > 1:
        psd_interp[0] = psd_interp[1]
    else:
        psd_interp[0] = 1.0
    whitened_spec = spectrum / np.sqrt(psd_interp)
    return np.fft.irfft(whitened_spec, n=n)


def q_transform(
    strain: np.ndarray,
    sample_rate: int,
    f_min: float = 32.0,
    f_max: float = 512.0,
    n_freq: int = 64,
    Q: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Absolute-value constant-Q time-frequency representation.

    For each of `n_freq` log-spaced centre frequencies between
    `f_min` and `f_max`, bandpass-filter `strain` with bandwidth
    `f_c / Q` and take the absolute value. Returns:

      * `time_freq_grid`: shape `(n_freq, n_time)`, each row a
        band-pass envelope.
      * `freqs`: shape `(n_freq,)`, centre frequencies in Hz.
    """
    from scipy.signal import butter, sosfilt
    n_time = len(strain)
    freqs = np.geomspace(f_min, f_max, n_freq)
    out = np.zeros((n_freq, n_time))
    nyq = sample_rate / 2.0 - 1.0
    for i, fc in enumerate(freqs):
        bw = fc / Q
        f_lo = max(0.5, fc - bw / 2)
        f_hi = min(nyq, fc + bw / 2)
        if f_hi <= f_lo:
            continue
        sos = butter(4, [f_lo, f_hi], btype="bandpass",
                     fs=sample_rate, output="sos")
        out[i] = np.abs(sosfilt(sos, strain))
    return out, freqs
