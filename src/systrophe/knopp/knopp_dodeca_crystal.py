"""Crystal-lock of the dodeca-in-horn-torus resonator: the (scale, beta) sweep.

Fourth module of the series (alignment -> first principles -> pressure ->
crystal). User direction 2026-06-09: "it just needs the last bit of
orientation to lock into a crystal-like analogy." This module makes that
analogy precise and finds the lock.

The crystal analogy, derived
----------------------------
The saturated interior field is a 6-axis icosahedral interference pattern --
the density-wave structure of an ICOSAHEDRAL QUASICRYSTAL (Shechtman class),
not a periodic crystal. Three registry conditions define the lock:

1. SOURCE REGISTRY (closed form, orientation-invariant).
   Face centres sit at r_i = r_in * s * a_i; axis j's standing wave there
   has phase x*G_ij with x = k * r_in * s and G_ij = a_i . a_j in
   {1, +-1/sqrt(5)}. With equal axis amplitudes:
       Lambda(x) = [cos^2(x) + 5 cos^2(x / sqrt(5))] / 6.
   The two frequencies are INCOMMENSURATE (ratio sqrt(5)), so perfect
   registry (Lambda = 1) is impossible at x > 0: the system can only lock
   at the continued-fraction convergents of sqrt(5) = [2; 4, 4, 4, ...]
   (m/n = 2/1, 9/4, 38/17, ...), i.e. QUASICRYSTAL APPROXIMANT locking,
   the Fibonacci-approximant ladder of icosahedral phases. First practical
   rung: x ~ 2.12 pi (Lambda 0.956); deep rung: x ~ 8.97 pi (Lambda 0.997).

2. PUMP-RING REGISTRY (numeric, scale-dependent).
   The Casimir pump zone is the funnel-wall ring facing the aligned
   pentagon: height h = r_in*s, radius rho_w = R - sqrt(R^2 - h^2).
   P = <I>_ring / I_ceiling peaks when an antinode SHELL lies on the ring
   (laser-cavity self-locking: field antinode on the gain medium).

3. RING COHERENCE (numeric, ORIENTATION-dependent -- "the last bit").
   Uc = 1 - std(I)/mean(I) on the pump ring. Away from face-lock the
   icosahedral pattern sweeps obliquely across the ring (contrast ~ 0.18);
   at beta = FACE_LOCK_DEG the pentagon's C5 axis registers on the torus
   axis and the ring becomes a single equi-phase antinode ring
   (contrast ~ 0.001, Uc ~ 0.999). This is the Bragg-ring condition that
   "locks the crystal" in orientation.

Crystal order parameter: X = Lambda * P * Uc, gated x0.2 unless the comb is
full-spectrum (the lock must sit on the saturated state).

Found lock (computed in crystal_lock_sweep, verified in tests):
    scale* ~ 0.215, beta* = FACE_LOCK_DEG = 37.377 deg, X ~ 0.72,
    Q-independent from Q = 30 to 240.

Caveats: same series caveats (coherent-pump proxy, model units). The
catcher runs on the scale slice of X at the lock orientation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.knopp.knopp_dodeca_alignment import (
    DEFAULT_R,
    FACE_LOCK_DEG,
    INRADIUS_RATIO,
    alignment_state,
    mode_saturation,
    normalised_drive,
)
from systrophe.knopp.knopp_dodeca_pressure import (
    field_axes,
    radiation_pressure,
    wavenumber,
)

SQRT5 = math.sqrt(5.0)


def source_registry(x: float | np.ndarray) -> np.ndarray:
    """Lambda(x) = [cos^2 x + 5 cos^2(x/sqrt5)]/6 -- face-centre registry.

    Quasiperiodic (frequencies 2 and 2/sqrt5 incommensurate): Lambda < 1
    for all x > 0; peaks at sqrt(5)-convergent approximants.
    """
    x = np.asarray(x, dtype=float)
    return (np.cos(x) ** 2 + 5.0 * np.cos(x / SQRT5) ** 2) / 6.0


def registry_ladder(x_max: float = 40.0, n: int = 16000) -> list[dict]:
    """Locking rungs of Lambda(x): local maxima above 0.8, sorted by x."""
    x = np.linspace(0.05, x_max, n)
    lam = source_registry(x)
    pk = (lam[1:-1] > lam[:-2]) & (lam[1:-1] > lam[2:]) & (lam[1:-1] > 0.8)
    idx = np.where(pk)[0] + 1
    return [{"x": float(x[i]), "lambda": float(lam[i]),
             "m_over_pi": float(x[i] / math.pi),
             "n_over_pi": float(x[i] / SQRT5 / math.pi)} for i in idx]


def pump_ring(scale: float, R: float = DEFAULT_R, n: int = 64) -> np.ndarray:
    """Funnel-wall ring facing the aligned pentagon (the Casimir pump zone)."""
    if scale <= 0 or R <= 0:
        raise ValueError("scale and R must be positive")
    h = INRADIUS_RATIO * scale
    rho = R - math.sqrt(max(R * R - h * h, 1e-12))
    phi = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.stack([rho * np.sin(phi), np.full(n, h), rho * np.cos(phi)],
                    axis=1)


@dataclass(frozen=True)
class CrystalOrder:
    """Crystal order parameter at one (scale, beta) configuration."""
    scale: float
    beta_deg: float
    x_commensuration: float    # k * r_in * scale
    source_lambda: float       # face-centre registry, closed form
    pump_registry: float       # <I>_ring / ceiling
    ring_coherence: float      # 1 - std/mean on the ring (1 = equi-phase)
    full_spectrum: bool
    order: float               # Lambda * P * Uc, x0.2 if not full spectrum


def crystal_order(
    beta_deg: float, scale: float, R: float = DEFAULT_R, Q: float = 60.0,
) -> CrystalOrder:
    """Evaluate the crystal order parameter (self-consistent k via saturation)."""
    ref = alignment_state(0.0, scale, R)
    cal = 2.0 * ref.drive_raw / ref.area_factor
    st = alignment_state(beta_deg, scale, R)
    d = normalised_drive(st, cal)
    sr = mode_saturation(d, st.vertex_align, st.face_align, st.spiral, Q=Q)
    k = wavenumber(sr.saturation)
    a2 = np.full(6, float(np.mean(sr.amplitudes ** 2)))
    ceiling = float(np.sum(a2) / 2.0)
    I = radiation_pressure(pump_ring(scale, R), field_axes(beta_deg), a2, k)
    P = float(np.mean(I) / ceiling)
    Uc = max(0.0, 1.0 - float(np.std(I)) / max(float(np.mean(I)), 1e-12))
    x = k * INRADIUS_RATIO * scale
    lam = float(source_registry(x))
    gate = 1.0 if sr.full_spectrum else 0.2
    return CrystalOrder(
        scale=float(scale), beta_deg=float(beta_deg),
        x_commensuration=float(x), source_lambda=lam,
        pump_registry=P, ring_coherence=Uc,
        full_spectrum=bool(sr.full_spectrum),
        order=float(lam * P * Uc * gate),
    )


@dataclass(frozen=True)
class CrystalLockReport:
    """Result of the full (scale, beta) crystal-lock sweep."""
    scale_axis: np.ndarray
    beta_axis: np.ndarray
    order_grid: np.ndarray         # n_scale x n_beta
    lock_scale: float
    lock_beta_deg: float
    lock_order: float
    lock_at_face_angle: bool       # |beta* - FACE_LOCK_DEG| < grid step
    lock: CrystalOrder
    ladder: list[dict]             # sqrt(5)-approximant rungs of Lambda
    catcher_verdict: str


def crystal_lock_sweep(
    scales: np.ndarray | None = None,
    betas: np.ndarray | None = None,
    R: float = DEFAULT_R,
    Q: float = 60.0,
) -> CrystalLockReport:
    """Full 2D sweep; the lock is the global argmax of the order parameter."""
    if scales is None:
        scales = np.linspace(0.10, 0.45, 29)
    if betas is None:
        betas = np.linspace(20.0, 60.0, 33)
    scales = np.asarray(scales, dtype=float)
    betas = np.asarray(betas, dtype=float)
    grid = np.zeros((len(scales), len(betas)))
    for i, s in enumerate(scales):
        for j, b in enumerate(betas):
            grid[i, j] = crystal_order(b, s, R, Q).order
    i0, j0 = np.unravel_index(int(np.argmax(grid)), grid.shape)
    # polish: evaluate at the exact face-lock angle for the winning scale
    polished = crystal_order(FACE_LOCK_DEG, float(scales[i0]), R, Q)
    if polished.order >= grid[i0, j0]:
        lock = polished
    else:
        lock = crystal_order(float(betas[j0]), float(scales[i0]), R, Q)
    step = float(betas[1] - betas[0]) if len(betas) > 1 else 1.0

    s_slice = grid[:, j0]
    table = dict(zip(scales, s_slice))

    def fn(s: float) -> np.ndarray:
        key = min(table, key=lambda v: abs(v - s))
        return np.array([table[key]])

    catch = scan_novelty(scales, fn, n_bits=32,
                         parameter_label="dodeca_scale_at_lock_beta")
    return CrystalLockReport(
        scale_axis=scales, beta_axis=betas, order_grid=grid,
        lock_scale=lock.scale, lock_beta_deg=lock.beta_deg,
        lock_order=lock.order,
        lock_at_face_angle=bool(abs(lock.beta_deg - FACE_LOCK_DEG) <= step),
        lock=lock, ladder=registry_ladder(),
        catcher_verdict=catch.verdict,
    )


def summarise_crystal(r: CrystalLockReport) -> str:
    """Human-readable summary."""
    rungs = ", ".join(f"x={d['x']:.2f} (L {d['lambda']:.3f})"
                      for d in r.ladder[:5])
    lines = [
        "Crystal-lock sweep (icosahedral quasicrystal approximant registry)",
        f"  LOCK: scale = {r.lock_scale:.3f}, beta = {r.lock_beta_deg:.3f} deg "
        f"(face angle: {r.lock_at_face_angle})",
        f"  order X = {r.lock_order:.4f}  [Lambda {r.lock.source_lambda:.3f} "
        f"x ring {r.lock.pump_registry:.3f} x coherence "
        f"{r.lock.ring_coherence:.3f}]",
        f"  commensuration x = k r_in s = {r.lock.x_commensuration:.3f} "
        f"({r.lock.x_commensuration / math.pi:.3f} pi)",
        f"  sqrt(5)-approximant ladder: {rungs}",
        f"  catcher: {r.catcher_verdict}",
    ]
    return "\n".join(lines)
