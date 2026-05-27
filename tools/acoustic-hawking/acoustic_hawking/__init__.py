"""acoustic-hawking: analog gravity tool for the Lewis-Papapetrou exterior.

Wraps `systrophe.analogs.acoustic_metric` (Unruh's acoustic-LP identification)
and `systrophe.analogs.acoustic_hawking_spectrum` (phonon spectral density,
Steinhauer 2019 benchmark) in a single LPAnalyser-style object.

Headline use: predict the acoustic Hawking temperature of a rotating
van Stockum exterior as if it were a flowing BEC, and compare to the
Steinhauer 2019 measurement of 0.124 ± 0.012 nK in his analog-BH
apparatus.
"""

from __future__ import annotations

from .horizon import (
    AnalogHorizonReport,
    AnalogHorizonAnalyser,
)
from .spectrum import (
    SpectrumReport,
    compute_phonon_spectrum,
)
from .bench import (
    SteinhauerBenchmark,
    benchmark_against_steinhauer_2019,
)

__all__ = [
    "AnalogHorizonReport",
    "AnalogHorizonAnalyser",
    "SpectrumReport",
    "compute_phonon_spectrum",
    "SteinhauerBenchmark",
    "benchmark_against_steinhauer_2019",
]
