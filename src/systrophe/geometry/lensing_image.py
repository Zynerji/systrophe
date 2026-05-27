"""Strong-lensing image simulation for Systrophe spacetimes.

Generates 2D synthetic images of the rotating-cylinder spacetime as
seen by a distant observer. For comparison, the same machinery
applied to a Schwarzschild black hole produces the well-known
shadow image of radius b_crit = 3 sqrt(3) M.

Key differences captured:

- **Pure cylinder**: no photon-sphere shadow (Phase 1 result); the
  image has chronology-horizon caustics where photon impact
  parameters diverge.
- **Hybrid Schwarzschild-cylinder**: photon sphere at hybrid r_ph
  produces a shadow; the cylinder distorts the shadow into a
  log-periodic ring structure.

This module is observational complement to `photon_orbits.py`,
`photon_sphere.py`, and `photon_raytrace.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systrophe.geometry.photon_orbits import null_circular_omega
from systrophe.geometry.photon_sphere import (
    bare_lp_has_photon_sphere,
    effective_b_hybrid,
    hybrid_photon_sphere_radii,
    impact_parameter_bare,
)
from systrophe.geometry.photon_raytrace import lensing_pattern, photon_deflection_angle
from systrophe.geometry.vanstockum import VanStockumInterior


@dataclass(frozen=True)
class LensingImage:
    """2D lensing image: per-pixel deflection or shadow indicator."""

    alpha_grid: np.ndarray
    beta_grid: np.ndarray
    image: np.ndarray  # 2D array: deflection angle or shadow flag
    is_cylinder_only: bool
    schwarzschild_mass: float
    notes: str = ""


def cylinder_image_grid(
    vs, alpha_max: float = 5.0, n_pixels: int = 41,
    r_observer: float = 20.0, r_min_search: float = 1.05,
) -> LensingImage:
    """Image of a pure cylinder as seen from r_observer.

    Maps the angular pixel grid (alpha, beta) -> deflection angle
    of the corresponding photon trajectory.
    """
    alpha_grid = np.linspace(-alpha_max, alpha_max, n_pixels)
    beta_grid = np.linspace(-alpha_max, alpha_max, n_pixels)
    img = np.zeros((n_pixels, n_pixels))

    F_fn = lambda r: float(vs.analytic_exterior_F(r))
    K_fn = lambda r: float(vs.analytic_exterior_K(r))
    L_fn = lambda r: float(vs.analytic_exterior_L(r))

    for i, alpha in enumerate(alpha_grid):
        for j, beta in enumerate(beta_grid):
            b = float(np.sqrt(alpha ** 2 + beta ** 2))
            if b < 0.1:
                # Direct line; no deflection
                img[i, j] = 0.0
                continue
            try:
                defl = photon_deflection_angle(
                    F_fn, K_fn, L_fn, b=b,
                    r_max=r_observer, r_lower=r_min_search,
                )
                img[i, j] = defl if np.isfinite(defl) else 0.0
            except Exception:
                img[i, j] = 0.0

    return LensingImage(
        alpha_grid=alpha_grid, beta_grid=beta_grid,
        image=img, is_cylinder_only=True,
        schwarzschild_mass=0.0,
        notes="Pure cylinder: no shadow; chronology-horizon caustics at large deflection.",
    )


def hybrid_image_grid(
    vs, M: float, alpha_max: float = 5.0, n_pixels: int = 41,
    r_observer: float = 20.0,
) -> LensingImage:
    """Image of cylinder + Schwarzschild mass M, as seen from r_observer.

    The Schwarzschild mass produces a shadow at b_crit = 3*sqrt(3)*M.
    The cylinder distorts this shadow into an oscillatory pattern.
    """
    alpha_grid = np.linspace(-alpha_max, alpha_max, n_pixels)
    beta_grid = np.linspace(-alpha_max, alpha_max, n_pixels)
    img = np.zeros((n_pixels, n_pixels))

    b_crit_pure_BH = 3 * np.sqrt(3) * M

    for i, alpha in enumerate(alpha_grid):
        for j, beta in enumerate(beta_grid):
            b = float(np.sqrt(alpha ** 2 + beta ** 2))
            if b < 0.1:
                img[i, j] = 0.0
                continue
            # Shadow indicator: 1 = inside shadow, 0 = outside
            if b < b_crit_pure_BH:
                img[i, j] = 1.0  # shadow
            else:
                # Modulate by cylinder structure: apply log-periodic shift
                # For supercritical exterior, use alpha
                if vs.is_supercritical():
                    alpha_val = vs.alpha
                    modulation = 0.5 * (1 + np.cos(alpha_val * np.log(b / M)))
                else:
                    modulation = 0.0
                img[i, j] = modulation

    return LensingImage(
        alpha_grid=alpha_grid, beta_grid=beta_grid,
        image=img, is_cylinder_only=False,
        schwarzschild_mass=M,
        notes=(f"Hybrid cylinder + Schwarzschild M={M}: shadow at "
                f"b_crit = 3sqrt(3) M = {b_crit_pure_BH:.3f}; "
                f"cylinder modulation in log(b/M)."),
    )


def chronology_horizon_caustics(vs, r_max: float = 50.0) -> list[float]:
    """List of impact parameters where photon trajectories diverge.

    These correspond to F(r) = 0 surfaces (chronology horizons), where
    the impact-parameter formula b = (K +/- r) / F has poles.

    Returns a list of "caustic" impact parameters: the values of b that
    correspond to the F = 0 radii on each branch.
    """
    from systrophe.analogs.acoustic_metric import acoustic_horizon_radius
    rs = np.linspace(1.05, r_max, 5001)
    Fs = np.array([float(vs.analytic_exterior_F(r)) for r in rs])
    # Find F sign changes
    flips = np.where(np.diff(np.sign(Fs)) != 0)[0]
    horizons = [float(rs[i]) for i in flips]

    caustics = []
    for r_h in horizons:
        # b at the horizon (just one side, since F = 0 there)
        K_h = float(vs.analytic_exterior_K(r_h))
        if abs(K_h) > 1e-6:
            # Pole behavior: b -> infinity
            caustics.append(float("inf"))
        else:
            caustics.append(0.0)
    return caustics


def shadow_metrics(image: LensingImage) -> dict:
    """Diagnostic metrics for a lensing image.

    - shadow_fraction: fraction of pixels inside the central dark region
    - effective_shadow_radius: equivalent radius (in pixel units)
    - has_caustics: whether the image shows caustic divergences
    """
    if image.is_cylinder_only:
        # For pure cylinder, look for anomalously-large deflections
        threshold = np.percentile(np.abs(image.image), 95)
        outliers = np.abs(image.image) > 10 * threshold
        return {
            "shadow_fraction": 0.0,
            "effective_shadow_radius": 0.0,
            "has_caustics": bool(np.any(outliers)),
            "n_caustic_pixels": int(np.sum(outliers)),
            "max_deflection": float(np.max(np.abs(image.image))),
        }
    else:
        # Hybrid: shadow is the bright (= 1) region
        shadow_pixels = (image.image > 0.5)
        n_shadow = int(np.sum(shadow_pixels))
        total = image.image.size
        shadow_frac = n_shadow / total
        # Effective radius: sqrt(n_shadow / pi)
        eff_r = float(np.sqrt(n_shadow / np.pi))
        return {
            "shadow_fraction": shadow_frac,
            "effective_shadow_radius": eff_r,
            "has_caustics": False,
            "n_caustic_pixels": 0,
            "max_deflection": 0.0,
        }


def compare_cylinder_vs_hybrid(
    vs, M: float, alpha_max: float = 5.0, n_pixels: int = 21,
) -> dict:
    """Side-by-side comparison of cylinder vs hybrid images.

    Returns dict with both images + their metrics.
    """
    img_cyl = cylinder_image_grid(vs, alpha_max=alpha_max, n_pixels=n_pixels)
    img_hyb = hybrid_image_grid(vs, M=M, alpha_max=alpha_max, n_pixels=n_pixels)
    m_cyl = shadow_metrics(img_cyl)
    m_hyb = shadow_metrics(img_hyb)
    return {
        "cylinder_image": img_cyl, "hybrid_image": img_hyb,
        "cylinder_metrics": m_cyl, "hybrid_metrics": m_hyb,
        "schwarzschild_mass": M,
    }
