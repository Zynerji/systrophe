"""Synthetic validation of the ELF catcher adapter.

These do not need network access. They check that:
  * a known injected ELF transient is localized in time;
  * the surrogate null SUPPORTS a real injected burst (small p); and
  * the surrogate null REJECTS pure colored noise (large p) -- i.e. the
    detector does not cry wolf on a signal whose only structure is its PSD.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schumann_elf_catcher import (  # noqa: E402
    burst_scan, burst_null_test, phase_randomize, trend_scan,
)

FS = 256.0


def _schumann_like_noise(duration_s=300.0, seed=0):
    """Colored noise with a rising induction-coil-like response + SR comb."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * FS)
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    resp = np.ones_like(f) * 1e-3
    band = (f > 4) & (f < 45)
    resp[band] = (f[band] / 30.0)            # coil response rising to ~30 Hz
    for fc in (7.83, 14.3, 20.8, 27.3, 33.8):  # SR comb bumps
        resp += 0.5 * np.exp(-0.5 * ((f - fc) / 0.6) ** 2)
    return np.fft.irfft(X * resp, n=n)


def _inject_transient(x, t_s, amp=8.0, dur_s=0.4, f0=8.0, f1=40.0):
    n = len(x)
    i0 = int((t_s - dur_s / 2) * FS)
    i1 = int((t_s + dur_s / 2) * FS)
    t = np.arange(i1 - i0) / FS
    chirp = amp * np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / dur_s * t * t))
    w = np.hanning(len(t))
    x[i0:i1] += w * chirp * x.std()
    return x


def test_phase_randomize_preserves_psd():
    x = _schumann_like_noise(seed=1)
    s = phase_randomize(x, np.random.default_rng(0))
    px = np.abs(np.fft.rfft(x)) ** 2
    ps = np.abs(np.fft.rfft(s)) ** 2
    # PSD preserved to numerical precision; phases differ.
    assert np.allclose(px, ps, rtol=1e-6, atol=px.max() * 1e-8)


def test_injected_burst_localized():
    # Inject mid-chunk (150 s into 60 s chunks -> 30 s into chunk 2) so the
    # burst is not split across a chunk boundary.
    x = _schumann_like_noise(seed=2)
    _inject_transient(x, 150.0, amp=12.0)
    res = burst_scan(x, FS, chunk_s=60.0)
    assert res.max_hamming_step > 0
    assert abs(res.max_step_time_s - 150.0) < 2.5


def test_null_supports_injection_and_rejects_noise():
    # Injected burst -> should beat the PSD-matched null (small p).
    x = _schumann_like_noise(seed=3)
    _inject_transient(x, 150.0, amp=16.0)
    hit = burst_null_test(x, FS, n_surrogates=40, chunk_s=60.0, seed=7)
    assert hit.null_p_value <= 0.10

    # Pure colored noise -> null should NOT be exceeded (large p).
    y = _schumann_like_noise(seed=99)
    miss = burst_null_test(y, FS, n_surrogates=40, chunk_s=60.0, seed=7)
    assert miss.null_p_value > 0.10


def test_trend_detects_ramp():
    # Amplitude ramp over the segment -> growth catcher should say 'growing'.
    x = _schumann_like_noise(seed=5)
    t = np.arange(len(x)) / FS
    x *= (1.0 + 1.5 * t / t[-1])
    g, _ = trend_scan(x, FS, bin_s=10.0)
    assert g.verdict == "growing"
