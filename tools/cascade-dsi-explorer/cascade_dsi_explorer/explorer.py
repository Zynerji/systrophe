"""CascadeDSIExplorer: one object per cascade, exposes the diagnostic stack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systrophe.tipler_fractal import (
    CascadeDSI,
    cascade_box_dimension,
    verify_geometric_progression,
)


@dataclass(frozen=True)
class CascadeSummary:
    """Compact characterisation of a cascade-DSI zero set."""
    R: float
    alpha_0: float
    A_0: float
    levels: int
    scale_factor: float
    amp_decay: float
    n_zeros: int
    box_dimension: float
    geometric_progression_ratio_mean: float | None
    geometric_progression_max_rel_dev: float | None
    is_geometric_progression: bool

    def __repr__(self) -> str:
        return (
            f"CascadeSummary(levels={self.levels}, sf={self.scale_factor}, "
            f"ad={self.amp_decay:.3f}, n_zeros={self.n_zeros}, "
            f"dim={self.box_dimension:.4f}, "
            f"geom={self.is_geometric_progression})"
        )


class CascadeDSIExplorer:
    """Wrap a CascadeDSI and expose the diagnostic stack.

    Parameters
    ----------
    R : float
        Reference radius (log origin).
    alpha_0 : float
        Base log-frequency.
    A_0 : float
        Base amplitude.
    delta_0 : float
        Base phase offset (default 0).
    levels : int
        Number of cascade levels (levels=1 is the bare sinusoid).
    scale_factor : float
        alpha_{k+1} / alpha_k (must exceed 1).
    amp_decay : float
        A_{k+1} / A_k (must be in (0, 1)).
    """

    def __init__(self, R: float, alpha_0: float, A_0: float = 1.0,
                 delta_0: float = 0.0, levels: int = 4,
                 scale_factor: float = 2.0, amp_decay: float = 0.5) -> None:
        self.cascade = CascadeDSI(
            R=float(R), alpha_0=float(alpha_0), A_0=float(A_0),
            delta_0=float(delta_0), levels=int(levels),
            scale_factor=float(scale_factor), amp_decay=float(amp_decay),
        )

    @property
    def R(self) -> float:
        return self.cascade.R

    @property
    def levels(self) -> int:
        return self.cascade.levels

    @property
    def scale_factor(self) -> float:
        return self.cascade.scale_factor

    @property
    def amp_decay(self) -> float:
        return self.cascade.amp_decay

    @property
    def alpha_0(self) -> float:
        return self.cascade.alpha_0

    @property
    def A_0(self) -> float:
        return self.cascade.A_0

    def F(self, r):
        """Cascade envelope sum_k A_k cos(alpha_k ln(r/R) + delta_0)."""
        return self.cascade.F(r)

    def alphas(self) -> np.ndarray:
        return self.cascade.alphas()

    def amps(self) -> np.ndarray:
        return self.cascade.amps()

    def zeros(self, r_min: float, r_max: float,
              n_grid: int = 100_001) -> np.ndarray:
        """Approximate zeros of F on [r_min, r_max] (sign-change + lerp)."""
        return self.cascade.zeros(r_min=float(r_min), r_max=float(r_max),
                                    n_grid=int(n_grid))

    def box_dimension(self, r_min: float, r_max: float,
                      n_scales: int = 18) -> float:
        """Box-counting fractal dimension of the zero set in u = ln(r) coords."""
        bc = cascade_box_dimension(self.cascade, r_min=float(r_min),
                                    r_max=float(r_max), n_scales=int(n_scales))
        return float(bc["dimension"])

    def is_geometric_zero_progression(self, r_min: float, r_max: float
                                       ) -> tuple[bool, dict]:
        """True iff the zero set is a (single-level) geometric progression.

        Returns (flag, full_diagnostic_dict).
        """
        zs = self.zeros(r_min=r_min, r_max=r_max)
        if len(zs) < 3:
            return False, {"n_zeros": len(zs)}
        diag = verify_geometric_progression(zs)
        return bool(diag["is_geometric"]), diag

    def summary(self, r_min: float, r_max: float,
                n_scales: int = 18) -> CascadeSummary:
        """One-shot characterisation: zeros + dimension + geometric check."""
        zs = self.zeros(r_min=r_min, r_max=r_max)
        n = len(zs)
        if n >= 3:
            gp = verify_geometric_progression(zs)
            gp_mean = float(gp["ratio_mean"])
            gp_dev = float(gp["max_rel_dev"])
            is_gp = bool(gp["is_geometric"])
        else:
            gp_mean = None
            gp_dev = None
            is_gp = False
        dim = self.box_dimension(r_min=r_min, r_max=r_max,
                                   n_scales=n_scales) if n > 0 else 0.0
        return CascadeSummary(
            R=self.R,
            alpha_0=self.alpha_0,
            A_0=self.A_0,
            levels=self.levels,
            scale_factor=self.scale_factor,
            amp_decay=self.amp_decay,
            n_zeros=int(n),
            box_dimension=float(dim),
            geometric_progression_ratio_mean=gp_mean,
            geometric_progression_max_rel_dev=gp_dev,
            is_geometric_progression=is_gp,
        )
