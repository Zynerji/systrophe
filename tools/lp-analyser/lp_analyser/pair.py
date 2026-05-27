"""PairAnalyser: Phase 3a/3b for SystrophePair + SystropheArray + OffAxisPair."""

from __future__ import annotations

from typing import Any

import numpy as np

from systrophe.geometry.array import SystropheArray
from systrophe.geometry.off_axis import OffAxisPair
from systrophe.geometry.pair import SystrophePair
from systrophe.geometry.sinusoid import TiplerSinusoid
from systrophe.geometry.vanstockum import VanStockumInterior


class PairAnalyser:
    """Drive Phase 3a (array beam-forming) and Phase 3b (off-axis pair).

    Parameters
    ----------
    cylinders
        List of `VanStockumInterior` instances. Length 1 -> single
        cylinder. Length 2 -> co-axial pair (Phase 3a) or off-axis
        pair (Phase 3b) depending on `separation`. Length > 2 -> array.
    offsets
        Relative phase offsets per cylinder (rad). Defaults to all zero.
    separation
        Perpendicular distance between cylinder axes, for the
        Phase 3b off-axis pair. If None, treat the cylinders as
        co-axial (Phase 3a).
    """

    def __init__(
        self,
        cylinders: list[VanStockumInterior],
        offsets: list[float] | None = None,
        separation: float | None = None,
    ) -> None:
        if len(cylinders) == 0:
            raise ValueError("need at least one cylinder")
        for c in cylinders:
            if not isinstance(c, VanStockumInterior):
                raise TypeError("cylinders must be VanStockumInterior instances")
        self.cylinders = list(cylinders)
        if offsets is None:
            offsets = [0.0] * len(cylinders)
        if len(offsets) != len(cylinders):
            raise ValueError("len(offsets) must match len(cylinders)")
        self.offsets = list(offsets)
        self.separation = float(separation) if separation is not None else None

        if self.separation is None:
            # Co-axial array (Phase 3a)
            self.array = SystropheArray.from_cylinders(self.cylinders, offsets=self.offsets)
            self.off_axis = None
        elif len(cylinders) == 2:
            # Off-axis pair (Phase 3b)
            self.array = None
            self.off_axis = OffAxisPair(cylinders[0], cylinders[1], separation=self.separation)
        else:
            raise ValueError("off-axis configuration requires exactly 2 cylinders")

    # ------------------------------------------------------------------
    # Phase 3a: co-axial array
    # ------------------------------------------------------------------
    def L(self, r: float | np.ndarray) -> np.ndarray:
        """Joint L = sum_i L_i(r) envelope. L < 0 marks CTC bands;
        L = 0 marks beam-steered nodes."""
        if self.array is None:
            raise RuntimeError("L only defined for co-axial configurations")
        return np.asarray(self.array.L(np.asarray(r)))

    def array_factor(self, r: float | np.ndarray) -> np.ndarray:
        """|sum_i A_i exp(i (alpha_i u + delta_i))| at radius r."""
        if self.array is None:
            raise RuntimeError("array_factor only defined for co-axial configurations")
        return np.asarray(self.array.array_factor(np.asarray(r)))

    def extinction_check(self, r_max: float | None = None) -> dict:
        """Verify N-fold extinction (uniform-phase comb collapses to ~0).

        Returns dict with `max_array_factor`, `is_extinguished`, `N`.
        """
        if self.array is None:
            raise RuntimeError("extinction_check only defined for co-axial configurations")
        return self.array.extinction_check(r_max=r_max)

    @staticmethod
    def beam_steer(r_target: float, cylinder: VanStockumInterior,
                    N: int = 2) -> "PairAnalyser":
        """Construct an array with phases chosen so the joint L vanishes
        at r_target (Phase 3a inverse problem)."""
        arr = SystropheArray.beam_steer(r_target=r_target, cylinder=cylinder, N=N)
        # Recover cylinders + offsets from the array's sinusoids
        offsets = [float(s.delta) - float(arr.sinusoids[0].delta) for s in arr.sinusoids]
        return PairAnalyser([cylinder] * N, offsets=offsets, separation=None)

    # ------------------------------------------------------------------
    # Phase 3b: off-axis pair
    # ------------------------------------------------------------------
    def cartesian_metric(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Joint Cartesian metric at (x, y) in the off-axis pair."""
        if self.off_axis is None:
            raise RuntimeError("cartesian_metric requires an off-axis pair")
        return self.off_axis.cartesian_metric(x, y)

    def ctc_map_2d(self, x_min: float, x_max: float, y_min: float, y_max: float,
                       nx: int = 81, ny: int = 41) -> dict:
        """2D CTC region map for the off-axis pair."""
        if self.off_axis is None:
            raise RuntimeError("ctc_map_2d requires an off-axis pair")
        return self.off_axis.ctc_map_2d(x_min, x_max, y_min, y_max, nx=nx, ny=ny)

    def ctc_region_topology(self, x_min: float, x_max: float, y_min: float, y_max: float,
                                nx: int = 81, ny: int = 41) -> dict:
        """Connected-component + hole count of the joint CTC region.

        The Phase 3b headline finding (Systrophe v0.21.0): the
        canonical omega=1, R=1, d=3 pair has 1 connected component
        with 2 holes -- a non-trivial 2D topology.
        """
        if self.off_axis is None:
            raise RuntimeError("ctc_region_topology requires an off-axis pair")
        return self.off_axis.ctc_region_topology(
            x_min, x_max, y_min, y_max, nx=nx, ny=ny,
        )

    def ergosurface_2d(self, x_min: float, x_max: float, y_min: float, y_max: float,
                          nx: int = 81, ny: int = 41) -> dict:
        """2D ergosurface (g_tt = 0) map for the off-axis pair."""
        if self.off_axis is None:
            raise RuntimeError("ergosurface_2d requires an off-axis pair")
        return self.off_axis.ergosurface_2d(x_min, x_max, y_min, y_max, nx=nx, ny=ny)

    def geodesic_completeness_test(self, x_starts: list[float], y_starts: list[float],
                                      vx0: float = 0.05, vy0: float = 0.05,
                                      t_max: float = 30.0,
                                      n_samples: int = 251) -> list[dict]:
        """Integrate timelike test particles from given starts, check escape."""
        if self.off_axis is None:
            raise RuntimeError("geodesic_completeness_test requires an off-axis pair")
        return self.off_axis.geodesic_completeness_test(
            tuple(x_starts), tuple(y_starts), vx0=vx0, vy0=vy0,
            t_max=t_max, n_samples=n_samples,
        )
