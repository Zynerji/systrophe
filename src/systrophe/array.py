"""N-cylinder Systrophe phased array.

Generalises the two-cylinder ``SystrophePair`` to an arbitrary number of
co-axial co-rotating cylinders, each with its own phase offset. The
joint exterior g_{phi phi} is the linearised sum of N Tipler-sinusoid
contributions:

    L_array(r) = sum_{i=1}^N A_i * r * cos(alpha_i * u + delta_i)

where u = ln(r/R_i). For *matched* cylinders (all same a, R, p), the
sum collapses to a single effective sinusoid via phasor sum:

    A_eff * exp(i * delta_eff) = sum_i A_i * exp(i * delta_i),

and the CTC band locations are tunable by the choice of phase pattern.

Special configurations
----------------------
- **All-aligned** (delta_i = delta for all i): N-fold constructive
  interference, A_eff = N * A. CTC band log-measure is roughly N times
  the single-cylinder value.
- **Uniform phase comb** (delta_i = 2 pi i / N): the phasor sum is
  exactly zero (geometric sum of N-th roots of unity); the joint
  exterior has *no* CTC bands. An N-fold topological off-switch
  generalising the pair anti-phase result.
- **Linear ramp** (delta_i = i * Delta): arithmetic progression of
  phasor angles; produces a Dirichlet-kernel-like interference
  pattern.

Caveats
-------
- Linearised. Cross-terms in the metric appear at O(G^2) and are not
  modelled.
- Co-axial only.
- For unmatched cylinders (different a or R), the phasor collapse does
  not apply; only numerical evaluation is meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .sinusoid import TiplerSinusoid


@dataclass(frozen=True)
class SystropheArray:
    """N-cylinder co-axial phased array.

    Parameters
    ----------
    sinusoids : tuple[TiplerSinusoid, ...]
        N (N >= 1) Tipler sinusoids representing N co-axial cylinders.
        All must be supercritical (a > 1/2).
    """

    sinusoids: tuple[TiplerSinusoid, ...]

    def __post_init__(self) -> None:
        if len(self.sinusoids) < 1:
            raise ValueError("array must contain at least one sinusoid")
        for s in self.sinusoids:
            if s.a <= 0.5:
                raise ValueError("all sinusoids must be supercritical (a > 1/2)")

    @property
    def N(self) -> int:
        return len(self.sinusoids)

    @property
    def alphas(self) -> tuple[float, ...]:
        return tuple(s.alpha for s in self.sinusoids)

    @property
    def all_matched(self) -> bool:
        """True iff all sinusoids have identical (R, a, p) -- phasor collapse applies."""
        first = self.sinusoids[0]
        for s in self.sinusoids[1:]:
            if abs(s.R - first.R) > 1e-12 * max(abs(s.R), abs(first.R), 1.0):
                return False
            if abs(s.a - first.a) > 1e-12:
                return False
            if abs(s.p - first.p) > 1e-12:
                return False
        return True

    def L(self, r: float | np.ndarray) -> np.ndarray:
        """Combined L envelope sum_i L_i(r)."""
        r_arr = np.asarray(r, dtype=float)
        total = np.zeros_like(r_arr)
        for s in self.sinusoids:
            total = total + s.L(r_arr)
        return total

    def to_single_sinusoid(self) -> TiplerSinusoid:
        """Phasor-collapse the array to a single equivalent sinusoid.

        Requires all matched sinusoids (same R, a, p). The phasor sum of
        amplitudes and phases gives the effective single-sinusoid
        representation with amplitude |sum_i A_i exp(i delta_i)| and
        phase arg(sum_i A_i exp(i delta_i)).
        """
        if not self.all_matched:
            raise ValueError("phasor collapse requires all matched (R, a, p)")
        first = self.sinusoids[0]
        z = sum(
            s.A * np.exp(1j * s.delta) for s in self.sinusoids
        )
        if not isinstance(z, complex):
            z = complex(z)
        return TiplerSinusoid(
            R=first.R,
            a=first.a,
            A=float(abs(z)),
            delta=float(np.angle(z)),
            p=first.p,
        )

    def ctc_bands(
        self, r_min: float, r_max: float, n_grid: int = 4001, atol: float = 1e-12
    ) -> list[tuple[float, float]]:
        """CTC bands of the joint envelope on [r_min, r_max]."""
        from .ctc import find_ctc_intervals

        return find_ctc_intervals(self.L, r_min=r_min, r_max=r_max, n_grid=n_grid, atol=atol)

    @classmethod
    def from_cylinders(
        cls,
        cylinders: Iterable,
        offsets: Iterable[float] | None = None,
    ) -> "SystropheArray":
        """Construct an array from N VanStockumInterior cylinders + relative offsets."""
        from .vanstockum import VanStockumInterior

        cyls = tuple(cylinders)
        if not all(isinstance(c, VanStockumInterior) for c in cyls):
            raise TypeError("all inputs must be VanStockumInterior instances")
        if offsets is None:
            offsets = (0.0,) * len(cyls)
        offsets = tuple(float(o) for o in offsets)
        if len(offsets) != len(cyls):
            raise ValueError("len(offsets) must equal len(cylinders)")
        sinusoids = []
        for c, off in zip(cyls, offsets):
            s = c.tipler_sinusoid()
            sinusoids.append(
                TiplerSinusoid(R=s.R, a=s.a, A=s.A, delta=s.delta + off, p=s.p)
            )
        return cls(sinusoids=tuple(sinusoids))

    @classmethod
    def uniform_phase_comb(cls, cylinder, N: int) -> "SystropheArray":
        """N copies of the same cylinder with phases evenly spaced at 2 pi i / N.

        Phasor-collapsing this gives zero: the sum of N-th roots of unity
        is identically zero. So the resulting array has no CTC bands --
        an N-fold topological extinction.
        """
        if N < 2:
            raise ValueError("N must be >= 2")
        offsets = tuple(2.0 * np.pi * i / N for i in range(N))
        return cls.from_cylinders([cylinder] * N, offsets=offsets)

    # -----------------------------------------------------------------
    # Phase 3a: beamforming + extinction diagnostics
    # -----------------------------------------------------------------

    def phasor_field(self, r: float | np.ndarray) -> dict:
        """Complex phasor sum sum_i A_i exp(i (alpha_i u_i + delta_i)) at r.

        For matched cylinders this is the geometric "antenna" phasor whose
        magnitude controls L(r) and whose phase controls the CTC band
        position. For unmatched arrays each term carries its own log-
        frequency and the sum is a multi-frequency beat.

        Returns dict with 'phasor' (complex array), 'magnitude', 'phase'.
        """
        r_arr = np.asarray(r, dtype=float)
        z = np.zeros_like(r_arr, dtype=complex)
        for s in self.sinusoids:
            u = np.log(r_arr / s.R)
            arg = s.alpha * u + s.delta
            z = z + s.A * np.exp(1j * arg)
        return {
            "phasor": z,
            "magnitude": np.abs(z),
            "phase": np.angle(z),
        }

    def array_factor(self, r: float | np.ndarray) -> np.ndarray:
        """Magnitude of the phasor sum at radius r.

        Maximum N * A at constructive interference, zero at extinction.
        Direct analog of antenna-array array factor (linear-phase-progression
        steering).
        """
        return np.asarray(self.phasor_field(r)["magnitude"])

    def extinction_check(self, r_min: float = None, r_max: float = None,
                            n_grid: int = 4001, atol: float = 1e-10) -> dict:
        """Verify N-fold topological extinction: max |phasor| < atol.

        For a uniform_phase_comb, the phasor sum is geometric N-th root
        of unity sum = 0 identically. Numerically this means
        max(|phasor(r)|) over [r_min, r_max] is ~ 0 (limited by FD noise).
        """
        if r_min is None:
            r_min = 1.05 * self.sinusoids[0].R
        if r_max is None:
            r_max = 10.0 * self.sinusoids[0].R
        r_grid = np.linspace(r_min, r_max, n_grid)
        af = self.array_factor(r_grid)
        max_af = float(np.max(np.abs(af)))
        is_extinguished = bool(max_af < atol * self.N * max(s.A for s in self.sinusoids))
        return {
            "max_array_factor": max_af,
            "is_extinguished": is_extinguished,
            "r_max": float(r_max),
            "atol": float(atol),
            "N": int(self.N),
        }

    def dirichlet_pattern(self, r: float | np.ndarray, delta_step: float = None
                            ) -> np.ndarray:
        """Dirichlet-kernel array factor for linear-ramp phasing.

        For matched sinusoids with delta_i = i * delta_step, the array
        factor is the geometric sum
            sum_{i=0}^{N-1} exp(i alpha u + i delta_step)
                = exp(i (N-1)(alpha u + delta_step)/2)
                  * sin(N (alpha u + delta_step) / 2)
                  / sin((alpha u + delta_step) / 2)

        Direct analog of the antenna-array Dirichlet kernel; useful for
        beam-steering CTC location.
        """
        if not self.all_matched:
            raise ValueError("Dirichlet pattern requires matched array")
        r_arr = np.asarray(r, dtype=float)
        first = self.sinusoids[0]
        if delta_step is None:
            # Infer delta_step from second sinusoid
            if self.N < 2:
                raise ValueError("need N >= 2 to infer delta_step")
            delta_step = self.sinusoids[1].delta - first.delta
        u = np.log(r_arr / first.R)
        psi = first.alpha * u + delta_step
        denom = np.sin(psi / 2.0)
        denom_safe = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
        return np.asarray(np.abs(np.sin(self.N * psi / 2.0) / denom_safe))

    @staticmethod
    def beam_steer(r_target: float, cylinder, N: int = 2) -> "SystropheArray":
        """Solve inverse problem: choose offsets to place an L-node at r_target.

        The TiplerSinusoid stores L(r) = r (r/R)^p A cos(alpha u + delta_L)
        where delta_L = gamma - psi is the *intrinsic* matching phase of
        the L-mode for this cylinder. For a pair with offsets
        (0, delta_aim) the joint L at r_target is

            A [cos(alpha u_t + delta_L) + cos(alpha u_t + delta_L + delta_aim)]
              = 2 A cos(delta_aim/2) cos(alpha u_t + delta_L + delta_aim/2)

        Setting the second factor to zero gives the local node:

            delta_aim = pi - 2 alpha ln(r_target / R) - 2 delta_L   (mod 2 pi)

        Returns the resulting SystropheArray of N cylinders with
        alternating delta = 0 and delta = delta_aim phases. (For N > 2
        the simple alternation still produces a local L-node at r_target;
        the depth of the node deepens with N.)
        """
        cyl_sin = cylinder.tipler_sinusoid()
        alpha = cyl_sin.alpha
        R = cyl_sin.R
        delta_L = cyl_sin.delta
        u_target = math.log(r_target / R)
        delta_aim = float(math.pi - 2.0 * alpha * u_target - 2.0 * delta_L)
        offsets = tuple(0.0 if (i % 2 == 0) else delta_aim for i in range(N))
        return SystropheArray.from_cylinders([cylinder] * N, offsets=offsets)

    def beam_pattern(self, r_min: float, r_max: float, n_grid: int = 200,
                       normalise: bool = True) -> dict:
        """Beam-pattern array factor across r in [r_min, r_max].

        Returns the array factor (and optionally normalised to its maximum)
        on a log-spaced radial grid -- the natural geometry for log-periodic
        sources.
        """
        r_grid = np.geomspace(r_min, r_max, n_grid)
        af = self.array_factor(r_grid)
        if normalise and np.max(af) > 0:
            af_n = af / np.max(af)
        else:
            af_n = af
        return {
            "r_grid": r_grid,
            "array_factor": af,
            "array_factor_normalised": af_n,
            "main_lobe_r": float(r_grid[int(np.argmax(af))]),
            "main_lobe_amplitude": float(np.max(af)),
        }

    def array_factor_sidelobe_level(self, r_min: float = None, r_max: float = None,
                                       n_grid: int = 4001) -> float:
        """Ratio of strongest side-lobe to main lobe.

        For an unsteered uniform-amplitude array, this is the standard
        antenna-side-lobe-level diagnostic. Lower (more negative dB) is
        sharper beam.
        """
        if r_min is None:
            r_min = 1.05 * self.sinusoids[0].R
        if r_max is None:
            r_max = 100.0 * self.sinusoids[0].R
        bp = self.beam_pattern(r_min, r_max, n_grid=n_grid, normalise=True)
        af = bp["array_factor_normalised"]
        # Find local maxima (strictly between neighbours)
        peaks = []
        for i in range(1, len(af) - 1):
            if af[i] > af[i - 1] and af[i] > af[i + 1]:
                peaks.append(float(af[i]))
        if not peaks:
            return 0.0
        peaks.sort(reverse=True)
        # Main lobe is peaks[0]; biggest side-lobe is peaks[1] if it exists
        if len(peaks) < 2:
            return 0.0
        return float(peaks[1] / peaks[0])
