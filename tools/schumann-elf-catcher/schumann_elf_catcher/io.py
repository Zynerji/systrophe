"""Loaders for public ELF / Schumann-resonance raw waveforms.

Currently supports the Sierra Nevada ELF station archive (Rodriguez-Camacho
et al., 2022; Zenodo 6348691 etc.). Each hourly recording is a flat array of
little-endian int16 samples; the companion ``*_info.txt`` gives the sampling
period and the first-sample UTC timestamp.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ELFRecording:
    """One contiguous ELF magnetometer recording."""

    samples: np.ndarray      # 1-D float array (raw ADC counts, DC not removed)
    sample_rate: float       # Hz
    t0_utc: str              # ISO-ish timestamp of the first sample
    sensor: str              # "0" (NS) | "1" (EW) | "unknown"
    source_path: str

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


def _parse_sierra_info(info_text: str) -> dict:
    """Parse a Sierra Nevada ``*_info.txt`` block into a dict."""
    out: dict = {}
    for line in info_text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        out[key.strip().lower()] = val.strip()
    return out


def load_sierra_nevada(
    data_path: str, info_path: str | None = None,
) -> ELFRecording:
    """Load one Sierra Nevada hourly recording.

    Parameters
    ----------
    data_path
        Path to the raw (extension-less) sample file, e.g.
        ``.../smplGRTU1_sensor_1_1403081849``.
    info_path
        Path to the companion ``*_info.txt``. Defaults to
        ``data_path + "_info.txt"`` if present; otherwise a 256 Hz
        fallback sample rate is assumed.
    """
    samples = np.fromfile(data_path, dtype="<i2").astype(float)

    if info_path is None:
        cand = data_path + "_info.txt"
        info_path = cand if os.path.exists(cand) else None

    sample_rate = 256.0
    t0 = "unknown"
    if info_path and os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8", errors="replace") as f:
            info = _parse_sierra_info(f.read())
        period_us = info.get("sampling period (usec)")
        if period_us:
            sample_rate = 1.0e6 / float(period_us)
        t0 = info.get("1st sample timestamp", "unknown")

    m = re.search(r"sensor_([01])", os.path.basename(data_path))
    sensor = m.group(1) if m else "unknown"

    return ELFRecording(
        samples=samples, sample_rate=sample_rate, t0_utc=t0,
        sensor=sensor, source_path=data_path,
    )
