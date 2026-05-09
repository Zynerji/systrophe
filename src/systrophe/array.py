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
