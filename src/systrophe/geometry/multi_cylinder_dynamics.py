"""Multi-cylinder array dynamics: N > 2 SystrophePair extensions.

Extends ``SystrophePair`` and ``SystropheArray`` with:

- Mixed-frequency arrays (different alpha_i for each cylinder)
- Beating-pattern analysis when alpha_beats != 0
- Random-phase array statistics
- N-fold extinction generalisations (which phase patterns extinguish)
- Asymptotic CTC density vs N

For matched-frequency arrays, the phasor extinction theorem gives:
A_eff = |sum_i exp(i delta_i)| = 0 iff the delta_i are the N-th roots
of unity (or any uniform 2 pi k/N pattern). For mixed-frequency arrays
there is generically no clean extinction at any delta pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systrophe.ctc.ctc import find_ctc_intervals
from systrophe.geometry.sinusoid import TiplerSinusoid


@dataclass(frozen=True)
class MultiCylinderArray:
    """N cylinders with possibly different alpha_i values."""

    sinusoids: list[TiplerSinusoid]

    def __post_init__(self) -> None:
        if len(self.sinusoids) < 1:
            raise ValueError("need at least 1 sinusoid")

    def L(self, r):
        r = np.asarray(r, dtype=float)
        return sum(s.L(r) for s in self.sinusoids)

    def ctc_bands(self, r_min: float, r_max: float, n_grid: int = 8001) -> list:
        return find_ctc_intervals(self.L, r_min=r_min, r_max=r_max, n_grid=n_grid)

    def total_ctc_log_measure(self, r_min: float, r_max: float,
                                  n_grid: int = 8001) -> float:
        bands = self.ctc_bands(r_min, r_max, n_grid)
        if not bands:
            return 0.0
        return float(sum(np.log(b / a) for a, b in bands))

    @property
    def N(self) -> int:
        return len(self.sinusoids)

    @property
    def alphas(self) -> np.ndarray:
        return np.array([s.alpha for s in self.sinusoids])

    @property
    def is_matched_frequency(self) -> bool:
        """True iff all alpha_i are equal."""
        alphas = self.alphas
        return bool(np.std(alphas) < 1e-12)


def random_phase_array(
    N: int, base_alpha: float = 1.0, R: float = 1.0,
    rng_seed: int = 0,
) -> MultiCylinderArray:
    """Random-phase N-cylinder array at matched alpha."""
    rng = np.random.default_rng(rng_seed)
    phases = rng.uniform(0, 2 * np.pi, N)
    a = float(np.sqrt((base_alpha ** 2 + 1) / 4))
    sinusoids = [TiplerSinusoid(R=R, a=a, A=1.0, delta=float(d)) for d in phases]
    return MultiCylinderArray(sinusoids=sinusoids)


def uniform_phase_array(N: int, base_alpha: float = 1.0,
                          R: float = 1.0) -> MultiCylinderArray:
    """N-cylinder array with delta_i = 2 pi i / N (extinction comb)."""
    a = float(np.sqrt((base_alpha ** 2 + 1) / 4))
    phases = [2 * np.pi * i / N for i in range(N)]
    sinusoids = [TiplerSinusoid(R=R, a=a, A=1.0, delta=p) for p in phases]
    return MultiCylinderArray(sinusoids=sinusoids)


def mixed_frequency_array(
    N: int, alpha_min: float = 1.0, alpha_max: float = 3.0,
    R: float = 1.0, rng_seed: int = 0,
) -> MultiCylinderArray:
    """N-cylinder array with random alpha_i in [alpha_min, alpha_max]."""
    rng = np.random.default_rng(rng_seed)
    alphas = rng.uniform(alpha_min, alpha_max, N)
    phases = rng.uniform(0, 2 * np.pi, N)
    sinusoids = []
    for alpha, delta in zip(alphas, phases):
        a = float(np.sqrt((alpha ** 2 + 1) / 4))
        sinusoids.append(TiplerSinusoid(R=R, a=a, A=1.0, delta=float(delta)))
    return MultiCylinderArray(sinusoids=sinusoids)


def beat_frequency_pair(s1: TiplerSinusoid, s2: TiplerSinusoid) -> float:
    """Beat frequency |alpha_1 - alpha_2| between two sinusoids."""
    return float(abs(s1.alpha - s2.alpha))


def beat_log_period(alpha_beat: float) -> float:
    """Log-period in u = ln(r) corresponding to beat frequency."""
    if alpha_beat < 1e-12:
        return float("inf")
    return float(2 * np.pi / alpha_beat)


def phasor_sum(deltas: np.ndarray, amps: np.ndarray | None = None) -> complex:
    """Complex phasor sum A_eff exp(i delta_eff) = sum A_i exp(i delta_i)."""
    deltas = np.asarray(deltas, dtype=float)
    if amps is None:
        amps = np.ones_like(deltas)
    amps = np.asarray(amps, dtype=float)
    return complex(np.sum(amps * np.exp(1j * deltas)))


def phasor_extinction_check(deltas: np.ndarray,
                                amps: np.ndarray | None = None,
                                tol: float = 1e-9) -> bool:
    """True iff the phasor sum is (near-)zero."""
    z = phasor_sum(deltas, amps)
    return bool(abs(z) < tol)


def n_cylinder_extinction_phases(N: int) -> np.ndarray:
    """Canonical uniform-phase extinction pattern: delta_i = 2 pi i / N."""
    return np.array([2 * np.pi * i / N for i in range(N)])


@dataclass(frozen=True)
class ScalingStudyResult:
    """Result of CTC measure vs N study."""

    N_values: list[int]
    uniform_measures: list[float]
    random_measures: list[float]
    random_std: list[float]


def ctc_measure_vs_N(
    N_range: list[int], r_min: float = 1.05, r_max: float = 10.0,
    n_random_samples: int = 20, rng_seed: int = 7,
) -> ScalingStudyResult:
    """Study CTC log-measure as a function of N.

    For each N:
      - uniform-phase comb: should give exact extinction (measure = 0)
      - random-phase: distribution of measures

    The expectation: uniform extinction holds for all N, random
    measure scales as ~ sqrt(N) / N = 1/sqrt(N) (random-walk argument).
    """
    rng = np.random.default_rng(rng_seed)
    uniform_measures = []
    random_means = []
    random_stds = []
    for N in N_range:
        # Uniform
        arr_uniform = uniform_phase_array(N)
        u_meas = arr_uniform.total_ctc_log_measure(r_min, r_max)
        uniform_measures.append(u_meas)
        # Random
        rand_measures = []
        for _ in range(n_random_samples):
            arr_rand = random_phase_array(N, rng_seed=int(rng.integers(1e9)))
            rand_measures.append(arr_rand.total_ctc_log_measure(r_min, r_max))
        random_means.append(float(np.mean(rand_measures)))
        random_stds.append(float(np.std(rand_measures)))
    return ScalingStudyResult(
        N_values=list(N_range),
        uniform_measures=uniform_measures,
        random_measures=random_means,
        random_std=random_stds,
    )
