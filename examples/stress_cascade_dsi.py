"""Stress Test 2: cascade-DSI dimension surface.

Sweeps (scale_factor sigma, amp_decay rho) on a grid and computes the
box-counting dimension of the cascade-DSI zero set. Looks for phase
transitions, plateaus, scaling collapses.

Setup
-----
F_cascade(r) = Sum_{k=0..L-1} A_k cos(alpha_k ln r + delta)
  with alpha_k = alpha_0 * sigma^k
       A_k    = A_0    * rho^k.

For sigma close to 1 OR rho close to 0: only the lowest-frequency
component contributes, dimension -> 0.

For sigma large AND rho close to 1: all scales contribute, dimension
can exceed 0 in box-counting sense.

Output
------
30x30 dimension matrix saved to examples/stress_cascade_dsi_results.json.
Phase-transition diagnostics + scaling-collapse fits.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from systrophe.tipler_fractal import CascadeDSI, cascade_box_dimension


def dimension_surface(
    sigmas: np.ndarray,
    rhos: np.ndarray,
    levels: int = 4,
    r_min: float = 1.0,
    r_max: float = np.exp(10),
    n_scales: int = 18,
) -> np.ndarray:
    """Compute box-counting dimension over a (sigma, rho) grid."""
    D = np.zeros((len(sigmas), len(rhos)))
    for i, sigma in enumerate(sigmas):
        for j, rho in enumerate(rhos):
            cascade = CascadeDSI(
                R=1.0, alpha_0=0.8, A_0=1.0, delta_0=0.0,
                levels=levels,
                scale_factor=float(sigma), amp_decay=float(rho),
            )
            try:
                bc = cascade_box_dimension(cascade, r_min=r_min, r_max=r_max,
                                             n_scales=n_scales)
                D[i, j] = bc["dimension"]
            except Exception:
                D[i, j] = np.nan
    return D


def find_phase_transition(
    D: np.ndarray, sigmas: np.ndarray, rhos: np.ndarray, threshold: float = 0.3
) -> dict:
    """Locate the (sigma, rho) boundary where dimension first exceeds threshold."""
    boundary = []
    for i in range(len(sigmas)):
        # Walk j (rho axis) until D[i, j] > threshold
        j_star = None
        for j in range(len(rhos)):
            if D[i, j] > threshold:
                j_star = j
                break
        if j_star is not None:
            boundary.append((float(sigmas[i]), float(rhos[j_star])))
    return {"boundary_points": boundary, "threshold": threshold}


def scaling_collapse_test(
    D: np.ndarray, sigmas: np.ndarray, rhos: np.ndarray
) -> dict:
    """Test if D(sigma, rho) = f(sigma * rho^alpha) for some alpha.

    Heuristic: search for alpha such that the dimension surface collapses
    onto a 1D curve under the rescaling x = sigma * rho^alpha.
    """
    flat_sig, flat_rho, flat_D = [], [], []
    for i in range(len(sigmas)):
        for j in range(len(rhos)):
            if np.isfinite(D[i, j]) and D[i, j] > 1e-3:
                flat_sig.append(sigmas[i])
                flat_rho.append(rhos[j])
                flat_D.append(D[i, j])
    if len(flat_D) < 10:
        return {"alpha_best": None, "rms_after": float("inf")}
    flat_sig = np.array(flat_sig); flat_rho = np.array(flat_rho); flat_D = np.array(flat_D)
    best_alpha = None
    best_rms = float("inf")
    for alpha in np.linspace(-3, 3, 121):
        if alpha == 0:
            continue
        x = flat_sig * flat_rho ** alpha
        # Bin x, take mean D in each bin, compute spread
        x_log = np.log(np.maximum(x, 1e-10))
        order = np.argsort(x_log)
        sorted_D = flat_D[order]
        # Compute residual from a moving-average fit
        from numpy.lib.stride_tricks import sliding_window_view
        if len(sorted_D) < 8:
            continue
        w = 5
        windowed = sliding_window_view(sorted_D, w)
        local_mean = windowed.mean(axis=1)
        local_resid = sorted_D[w//2:-(w//2)] - local_mean
        rms = float(np.sqrt(np.mean(local_resid ** 2)))
        if rms < best_rms:
            best_rms = rms
            best_alpha = alpha
    return {"alpha_best": best_alpha, "rms_after": best_rms}


def main():
    print("=" * 60)
    print("Cascade-DSI dimension surface stress test")
    print("=" * 60)
    print()

    sigmas = np.linspace(1.5, 8.0, 20)
    rhos = np.linspace(0.4, 0.99, 20)
    print(f"Grid: {len(sigmas)} x {len(rhos)} = {len(sigmas)*len(rhos)} points")
    print(f"sigma in [{sigmas[0]:.2f}, {sigmas[-1]:.2f}]")
    print(f"rho   in [{rhos[0]:.2f}, {rhos[-1]:.2f}]")
    print()

    t0 = time.time()
    D = dimension_surface(sigmas, rhos, levels=4)
    elapsed = time.time() - t0
    print(f"Computed in {elapsed:.1f}s")
    print()

    finite_D = D[np.isfinite(D)]
    print(f"Dimension statistics:")
    print(f"  min:  {finite_D.min():.4f}")
    print(f"  max:  {finite_D.max():.4f}")
    print(f"  mean: {finite_D.mean():.4f}")
    print(f"  std:  {finite_D.std():.4f}")
    print()

    # Phase transition boundary
    pt = find_phase_transition(D, sigmas, rhos, threshold=0.3)
    print(f"Phase transition (D > 0.3) boundary points: {len(pt['boundary_points'])}")
    if pt["boundary_points"]:
        print(f"  first: sigma={pt['boundary_points'][0][0]:.3f}, rho={pt['boundary_points'][0][1]:.3f}")
        print(f"  last : sigma={pt['boundary_points'][-1][0]:.3f}, rho={pt['boundary_points'][-1][1]:.3f}")
    print()

    # Scaling collapse
    sc = scaling_collapse_test(D, sigmas, rhos)
    print(f"Scaling collapse:")
    print(f"  best alpha: {sc['alpha_best']}")
    print(f"  rms after : {sc['rms_after']:.4f}")
    print()

    # Surface contour summary
    print("Dimension matrix (sigma rows x rho cols, both ascending):")
    for i in range(0, len(sigmas), 4):
        row_str = "  ".join(f"{D[i, j]:.2f}" for j in range(0, len(rhos), 4))
        print(f"  sigma={sigmas[i]:.2f}: {row_str}")

    payload = {
        "sigmas": sigmas.tolist(),
        "rhos": rhos.tolist(),
        "dimension_matrix": D.tolist(),
        "stats": {
            "min": float(finite_D.min()),
            "max": float(finite_D.max()),
            "mean": float(finite_D.mean()),
        },
        "phase_transition": pt,
        "scaling_collapse": sc,
    }
    out_path = Path("examples") / "stress_cascade_dsi_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
