"""Two-cylinder superposition of Tipler sinusoids.

For two co-rotating ("top-spin"), positive-mass ("dual-positive") source
cylinders sharing a common axis (or treated co-axially in leading order),
the linearised exterior g_{phi phi} is the sum of the individual
log-periodic envelopes plus a relative phase offset:

    L_pair(r) = L_1(r) + L_2(r)

where L_i = TiplerSinusoid(R_i, a_i, A_i, delta_i, p_i).L. The "off-set
Tipler sinusoid" is then the combined log-periodic envelope, whose CTC
region structure is determined by the relative phase

    Delta_delta = delta_2 - delta_1

and, when the two sources have different a (different alpha), by the
beat between the two log-frequencies.

Caveats
-------
- This is leading-order linearised superposition. The exact
  two-cylinder GR problem has no closed-form solution; non-linear
  effects appear at O(G^2) and are absent here.
- Co-axial geometry is assumed. Off-axis pairs require a different
  ansatz (the linearised metric has azimuthal-multipole content).
- Both sources must individually be supercritical (a > 1/2) for the
  Tipler-sinusoid representation to apply.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systrophe.geometry.sinusoid import TiplerSinusoid


@dataclass(frozen=True)
class SystrophePair:
    """Linearised superposition of two co-axial Tipler sinusoids.

    Parameters
    ----------
    s1, s2 : TiplerSinusoid
        The two source sinusoids. Must share a common log origin (same R)
        OR have R values within numerical tolerance; otherwise the log
        offset is absorbed into the relative delta and a warning is emitted.
    """

    s1: TiplerSinusoid
    s2: TiplerSinusoid

    def __post_init__(self) -> None:
        if not (self.s1.a > 0.5 and self.s2.a > 0.5):
            raise ValueError("Both sources must be supercritical (a > 1/2)")

    @property
    def phase_offset(self) -> float:
        """Relative phase delta_2 - delta_1, wrapped to (-pi, pi]."""
        d = self.s2.delta - self.s1.delta
        d = float(np.mod(d + np.pi, 2 * np.pi) - np.pi)
        # Convention: half-open interval (-pi, pi]; remap -pi -> +pi.
        if d <= -np.pi + 1e-12:
            d = np.pi
        return d

    @property
    def alpha_beat(self) -> float:
        """Log-frequency beat: alpha_2 - alpha_1.

        Zero iff sources have identical a; in that case the pair behaves
        as a single sinusoid with combined amplitude and phase.
        """
        return self.s2.alpha - self.s1.alpha

    def L(self, r: float | np.ndarray) -> np.ndarray:
        """Combined log-periodic envelope L_1(r) + L_2(r)."""
        return self.s1.L(r) + self.s2.L(r)

    def to_single_sinusoid(self) -> TiplerSinusoid:
        """Collapse to a single TiplerSinusoid when alpha_beat == 0.

        Two co-frequency cosines sum to a single cosine via
            A1 cos(x + d1) + A2 cos(x + d2)
                = A_eff cos(x + d_eff)
        with A_eff and d_eff computed from the phasor sum. Requires
        matched (R, a, p); raises if any differ.
        """
        if abs(self.alpha_beat) > 1e-12:
            raise ValueError("alpha_beat != 0; pair has no single-sinusoid collapse")
        if abs(self.s1.R - self.s2.R) > 1e-12 * max(self.s1.R, self.s2.R):
            raise ValueError("R mismatch; cannot collapse")
        if abs(self.s1.p - self.s2.p) > 1e-12:
            raise ValueError("p mismatch; cannot collapse")
        z1 = self.s1.A * np.exp(1j * self.s1.delta)
        z2 = self.s2.A * np.exp(1j * self.s2.delta)
        z = z1 + z2
        return TiplerSinusoid(
            R=self.s1.R,
            a=self.s1.a,
            A=float(np.abs(z)),
            delta=float(np.angle(z)),
            p=self.s1.p,
        )

    @classmethod
    def from_cylinders(
        cls,
        cyl_1,
        cyl_2,
        delta_offset: float = 0.0,
    ) -> "SystrophePair":
        """Construct a pair from two VanStockumInterior cylinders.

        Each cylinder generates its matched TiplerSinusoid via
        VanStockumInterior.tipler_sinusoid(); the second cylinder is
        rotated by an additional `delta_offset` relative phase, modelling
        the "off-set" between the two top-spin sources.

        Both cylinders must be supercritical.
        """
        from systrophe.geometry.vanstockum import VanStockumInterior

        if not isinstance(cyl_1, VanStockumInterior) or not isinstance(cyl_2, VanStockumInterior):
            raise TypeError("Both inputs must be VanStockumInterior instances")
        s1 = cyl_1.tipler_sinusoid()
        s2 = cyl_2.tipler_sinusoid()
        # Apply additional offset to s2's phase
        s2_off = TiplerSinusoid(
            R=s2.R, a=s2.a, A=s2.A, delta=s2.delta + float(delta_offset), p=s2.p
        )
        return cls(s1=s1, s2=s2_off)

    def ctc_bands(
        self, r_min: float, r_max: float, n_grid: int = 4001, atol: float = 1e-12
    ) -> list[tuple[float, float]]:
        """CTC bands of the combined envelope L_pair < 0 in [r_min, r_max]."""
        from systrophe.ctc.ctc import find_ctc_intervals

        return find_ctc_intervals(self.L, r_min=r_min, r_max=r_max, n_grid=n_grid, atol=atol)

    def total_ctc_log_measure(self, r_min: float, r_max: float, n_grid: int = 4001) -> float:
        """Total ln(r) span occupied by CTC bands in [r_min, r_max].

        Useful as a single-number diagnostic of how much CTC content the
        offset pair retains; analogous to the total time the off-set sinusoid
        spends in the time-loop sector.
        """
        bands = self.ctc_bands(r_min, r_max, n_grid=n_grid)
        return float(sum(np.log(b / a) for a, b in bands))

    def offset_sweep(
        self,
        r_min: float,
        r_max: float,
        offsets: np.ndarray,
        s1: TiplerSinusoid | None = None,
    ) -> dict:
        """Sweep `delta_offset` and report CTC log-measure.

        For visual / diagnostic use: how does the CTC band content change as
        the second cylinder's phase is rotated through the supplied offsets?

        Parameters
        ----------
        r_min, r_max : float
            Range to characterize.
        offsets : np.ndarray
            Phases (radians) to sweep.
        s1 : TiplerSinusoid, optional
            Reference sinusoid. Defaults to self.s1; the second sinusoid is
            constructed from self.s2 with delta replaced by self.s2.delta + offset.

        Returns
        -------
        dict with keys 'offsets', 'log_measures', 'n_bands'.
        """
        if s1 is None:
            s1 = self.s1
        offsets = np.asarray(offsets, dtype=float)
        log_measures = np.zeros_like(offsets)
        n_bands = np.zeros_like(offsets, dtype=int)
        for i, off in enumerate(offsets):
            s2_shifted = TiplerSinusoid(
                R=self.s2.R,
                a=self.s2.a,
                A=self.s2.A,
                delta=self.s2.delta + float(off),
                p=self.s2.p,
            )
            pair = SystrophePair(s1=s1, s2=s2_shifted)
            bands = pair.ctc_bands(r_min, r_max)
            log_measures[i] = float(sum(np.log(b / a) for a, b in bands)) if bands else 0.0
            n_bands[i] = len(bands)
        return {"offsets": offsets, "log_measures": log_measures, "n_bands": n_bands}
