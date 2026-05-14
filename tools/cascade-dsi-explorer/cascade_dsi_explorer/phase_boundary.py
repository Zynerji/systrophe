"""Phase-boundary scanner for cascade-DSI parameter space.

Sweeps (scale_factor, amp_decay) on a 2D grid, hashes each cascade's
zero set into a bit address, builds a Hamming-radius graph and reports
the algebraic-connectivity λ_2. Sharp jumps in λ_2 are candidate
phase boundaries (e.g. transition between geometric-progression and
fractal-cascade regimes).

Follows the protocol in `examples/stress_cascade_novelty.py` (the
mandated address-space rule). Wired as a standalone tool so other
deliverables can call it without reproducing the boilerplate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .explorer import CascadeDSIExplorer


@dataclass(frozen=True)
class PhaseBoundaryReport:
    """Result of a 2D phase-boundary scan.

    Attributes
    ----------
    scale_factors, amp_decays : np.ndarray of shape (N_sf,), (N_ad,)
        Swept axis values.
    lambda_2_by_radius : dict[int, float]
        λ_2 of the full pool's Hamming graph at each radius cutoff.
    n_zeros_grid : np.ndarray of shape (N_sf, N_ad)
        Number of zeros per (sf, ad) cell.
    box_dimension_grid : np.ndarray of shape (N_sf, N_ad)
        Box-counting dimension per cell.
    max_lambda_2_jump : float
        Largest |λ_2(radius_{i+1}) - λ_2(radius_i)| across the radius
        sweep. Heuristic threshold for "novel structure" verdict.
    verdict : str
        "uniform" | "smooth" | "novel_structure"
    """
    scale_factors: np.ndarray
    amp_decays: np.ndarray
    lambda_2_by_radius: dict
    n_zeros_grid: np.ndarray
    box_dimension_grid: np.ndarray
    max_lambda_2_jump: float
    verdict: str


def _zero_set_to_address(zeros: np.ndarray, n_bits: int = 64,
                          u_min: float = -1.0, u_max: float = 10.0) -> np.ndarray:
    """Hash a zero set to a binary occupancy vector in u = ln r coordinates."""
    if len(zeros) == 0:
        return np.zeros(n_bits, dtype=int)
    u = np.log(np.maximum(zeros, 1e-12))
    valid = u[(u >= u_min) & (u < u_max)]
    if len(valid) == 0:
        return np.zeros(n_bits, dtype=int)
    bin_idx = np.floor((valid - u_min) / (u_max - u_min) * n_bits).astype(int)
    bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < n_bits)]
    addr = np.zeros(n_bits, dtype=int)
    addr[bin_idx] = 1
    return addr


def _lambda_2_of_pool(addresses: list[np.ndarray], radius: int) -> float:
    """λ_2 of the Hamming-radius graph on a list of addresses."""
    n = len(addresses)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = int(np.sum(addresses[i] != addresses[j]))
            if d <= radius:
                A[i, j] = 1
                A[j, i] = 1
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    eigs = np.sort(np.linalg.eigvalsh(L))
    return float(eigs[1]) if n >= 2 else 0.0


def scan_phase_boundary(
    R: float = 1.0,
    alpha_0: float = 0.8,
    A_0: float = 1.0,
    delta_0: float = 0.0,
    levels: int = 4,
    scale_factors: np.ndarray | None = None,
    amp_decays: np.ndarray | None = None,
    r_min: float = 1.0,
    r_max: float = 1e6,
    n_bits: int = 64,
    radii: tuple[int, ...] = (4, 8, 12, 16, 24),
    novel_threshold: float = 0.5,
    smooth_threshold: float = 0.1,
) -> PhaseBoundaryReport:
    """Sweep (scale_factor, amp_decay), hash zero sets, run the catcher.

    Default sweep ranges are conservative (sf in (1.5, 8.0),
    ad in (0.4, 0.99)) and match the published Systrophe novelty
    example. Override via the explicit arrays.
    """
    if scale_factors is None:
        scale_factors = np.linspace(1.5, 8.0, 8)
    if amp_decays is None:
        amp_decays = np.linspace(0.4, 0.99, 8)
    scale_factors = np.asarray(scale_factors, dtype=float)
    amp_decays = np.asarray(amp_decays, dtype=float)

    addresses: list[np.ndarray] = []
    n_zeros_grid = np.zeros(
        (len(scale_factors), len(amp_decays)), dtype=int,
    )
    dim_grid = np.zeros_like(n_zeros_grid, dtype=float)
    u_min_addr = float(np.log(max(r_min, 1e-12)))
    u_max_addr = float(np.log(r_max))
    for i, sf in enumerate(scale_factors):
        for j, ad in enumerate(amp_decays):
            try:
                exp = CascadeDSIExplorer(
                    R=R, alpha_0=alpha_0, A_0=A_0, delta_0=delta_0,
                    levels=levels, scale_factor=float(sf),
                    amp_decay=float(ad),
                )
                zs = exp.zeros(r_min=r_min, r_max=r_max, n_grid=20_001)
                n_zeros_grid[i, j] = len(zs)
                if len(zs) > 1:
                    dim_grid[i, j] = exp.box_dimension(
                        r_min=r_min, r_max=r_max, n_scales=14,
                    )
                addr = _zero_set_to_address(
                    zs, n_bits=n_bits, u_min=u_min_addr, u_max=u_max_addr,
                )
            except ValueError:
                addr = np.zeros(n_bits, dtype=int)
            addresses.append(addr)

    lambda_2_by_radius = {
        int(r): _lambda_2_of_pool(addresses, radius=int(r))
        for r in radii
    }
    l2_vals = list(lambda_2_by_radius.values())
    jumps = [abs(l2_vals[i + 1] - l2_vals[i]) for i in range(len(l2_vals) - 1)]
    max_jump = float(max(jumps)) if jumps else 0.0

    if max_jump > novel_threshold:
        verdict = "novel_structure"
    elif max_jump > smooth_threshold:
        verdict = "smooth"
    else:
        verdict = "uniform"

    return PhaseBoundaryReport(
        scale_factors=scale_factors,
        amp_decays=amp_decays,
        lambda_2_by_radius=lambda_2_by_radius,
        n_zeros_grid=n_zeros_grid,
        box_dimension_grid=dim_grid,
        max_lambda_2_jump=max_jump,
        verdict=verdict,
    )
