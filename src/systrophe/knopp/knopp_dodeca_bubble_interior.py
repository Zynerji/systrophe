"""What happens INSIDE the bubble at crystal lock.

Fifth module of the dodeca-resonator series (alignment -> first principles
-> pressure -> crystal -> interior). At the crystal lock (scale 0.212,
beta = FACE_LOCK_DEG, full-spectrum saturation, k = 38) the tube interior
carries the locked icosahedral standing-wave field. This module derives what
that field DOES to matter and waves inside the bubble.

Derivations (model units, coherent classical field -- no GR claims)
-------------------------------------------------------------------
B0. THE AXIS POTENTIAL IS THE REGISTRY FUNCTION (exact identity).
    Along the bubble axis y at face-lock, the aligned face axis projects at
    1 and the other five at exactly +-1/sqrt(5) (the icosahedral G_ij).
    The Gor'kov potential along the axis is therefore
        V(x) = (V0/6) [cos(2 k x) + 5 cos(2 k x / sqrt 5)]
             = 2 V0 (Lambda(k x) - 1/2),
    an affine map of the crystal-registry function Lambda itself: the same
    sqrt(5)-incommensurate two-tone quasiperiodicity that sets the lock
    ladder also rules the interior. This is the canonical quasiperiodic
    (Aubry-Andre-class) potential.

B1. TRAP LATTICE (matter): the Gor'kov minima form an icosahedral
    QUASILATTICE -- discrete multiple nearest-neighbour shells (>= 2),
    unlike the single NN spacing of a periodic (cubic-control) lattice.
    At lock: ~10^3 deep traps per (0.9)^3 central volume, median depth
    ~ 0.7 x ceiling, median spacing ~ pi/k..sqrt5 pi/2k. Classical test
    matter loads onto these sites (acoustic-levitation physics).

B2. WAVE LOCALIZATION (honest negative with derived threshold): probe
    waves in the axis potential undergo an Aubry-Andre-like localization
    transition at V0* of order the recoil energy E_r = (2k/sqrt5)^2 ~ 1155.
    Numerically the inverse-participation ratio jumps between V0 ~ 1e3 and
    2e3. The lock amplitude is V0 = ceiling ~ 2.5 -- a factor ~10^2.7 BELOW
    threshold: at crystal lock the interior is TRANSPARENT to waves (modes
    extended), not a wave cage. Localization would require boosting the
    stored field by ~650x.

B3. PAYLOAD GRIP: during directional collapse the m=1 tilt exerts force
    density ~ sat * eps * ceiling / R on trapped matter; the trap escape
    gradient is ~ max|grad U| ~ k * sum A^2. The grip ratio
        G = max|grad U| / (eps * ceiling / R) ~ O(10^2)
    so payload stays PINNED to the lattice while the bubble contracts:
    the locked crystal acts as a rigid cargo rack riding the collapse.

Net picture at crystal lock: the bubble interior is a transparent
icosahedral quasicrystal "cargo lattice" -- matter is held on ~10^3 pinned
quasilattice sites with O(10^2) grip margin against the collapse tilt,
while radiation passes through freely (no localization at lock amplitude).

Catcher (mandatory): runs on the localization V0-scan -- a genuine
transition candidate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.spatial import cKDTree

from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.knopp.knopp_dodeca_alignment import FACE_LOCK_DEG, DEFAULT_R
from systrophe.knopp.knopp_dodeca_crystal import source_registry
from systrophe.knopp.knopp_dodeca_pressure import (
    axis_amplitudes,
    field_axes,
    gorkov_gradient,
    gorkov_potential,
    tube_interior_points,
    wavenumber,
)

SQRT5 = math.sqrt(5.0)


# ----- B0: the axis potential is the registry function ------------------------


def axis_projections(beta_deg: float = FACE_LOCK_DEG) -> np.ndarray:
    """|a_j . y| for the 6 face axes: at face-lock exactly {1, 1/sqrt5 x5}."""
    return np.abs(field_axes(beta_deg)[:, 1])


def axis_potential(x: np.ndarray, V0: float, k: float) -> np.ndarray:
    """V(x) along the bubble axis at face-lock (exact, from projections)."""
    x = np.asarray(x, dtype=float)
    return V0 / 6.0 * (np.cos(2.0 * k * x) + 5.0 * np.cos(2.0 * k * x / SQRT5))


def registry_identity_residual(k: float = 38.0, n: int = 2000) -> float:
    """max |V(x)/V0 - 2(Lambda(kx) - 1/2)| -- the B0 identity, numerically."""
    x = np.linspace(0.0, 2.0, n)
    lhs = axis_potential(x, 1.0, k)
    rhs = 2.0 * (source_registry(k * x) - 0.5)
    return float(np.max(np.abs(lhs - rhs)))


# ----- B2: wave localization scan ---------------------------------------------


def recoil_energy(k: float = 38.0) -> float:
    """E_r = (2k/sqrt5)^2: the slow-tone recoil scale setting the threshold."""
    return (2.0 * k / SQRT5) ** 2


def participation(V0: float, k: float = 38.0, L: float = 8.0,
                  n: int = 5000, n_states: int = 60) -> float:
    """Mean normalised IPR (n * IPR) of the lowest states in V(x).

    ~ O(1) for extended states; >> 1 when localized.
    """
    if V0 < 0 or n < 100:
        raise ValueError("V0 must be >= 0 and n >= 100")
    x = np.linspace(0.0, L, n)
    dx = x[1] - x[0]
    V = axis_potential(x, V0, k)
    diag = 2.0 / dx ** 2 + V
    off = -np.ones(n - 1) / dx ** 2
    _, vecs = eigh_tridiagonal(diag, off, select="i",
                               select_range=(0, n_states - 1))
    p = vecs ** 2 / np.sum(vecs ** 2, axis=0)
    return float(np.mean(np.sum(p ** 2, axis=0)) * n)


def localization_scan(
    V0_values: np.ndarray | None = None, k: float = 38.0,
    jump_factor: float = 10.0,
) -> dict:
    """IPR vs amplitude: locate the localization threshold V0*."""
    if V0_values is None:
        V0_values = np.array([2.5, 25.0, 100.0, 400.0, 1000.0,
                              2000.0, 4000.0, 8000.0])
    V0_values = np.asarray(V0_values, dtype=float)
    ipr = np.array([participation(v, k) for v in V0_values])
    base = ipr[0]
    above = np.where(ipr > jump_factor * base)[0]
    v_star = float(V0_values[above[0]]) if len(above) else float("inf")
    return {"V0": V0_values, "n_ipr": ipr, "V0_star": v_star,
            "recoil": recoil_energy(k)}


# ----- B1: trap lattice census --------------------------------------------------


@dataclass(frozen=True)
class TrapCensus:
    """Deep Gor'kov minima in the central interior volume."""
    n_traps: int
    median_depth: float          # negative; ceiling-relative magnitude ~0.7
    median_spacing: float
    nn_shell_count: int          # discrete NN shells (>=2 = quasilattice)
    shells: tuple[float, ...]


def _census(axes: np.ndarray, a2: np.ndarray, k: float,
            extent: float, n: int, depth_cut: float) -> TrapCensus:
    g = np.linspace(-extent, extent, n)
    P = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1)
    U = gorkov_potential(P.reshape(-1, 3), axes, a2, k).reshape(n, n, n)
    m = np.ones((n, n, n), bool)
    m[[0, -1], :, :] = m[:, [0, -1], :] = m[:, :, [0, -1]] = False
    for s in product((-1, 0, 1), repeat=3):
        if s == (0, 0, 0):
            continue
        m &= U < np.roll(U, s, (0, 1, 2))
    ceiling = float(np.sum(a2) / 2.0)
    deep = m & (U < depth_cut * ceiling)
    pts = P[deep]
    if len(pts) < 4:
        return TrapCensus(len(pts), 0.0, 0.0, 0, ())
    depths = U[deep]
    d, _ = cKDTree(pts).query(pts, k=2)
    nn = d[:, 1]
    hist, edges = np.histogram(nn, bins=60, range=(0.0, 0.25))
    shells = tuple(
        float(0.5 * (edges[i] + edges[i + 1]))
        for i in range(1, 59)
        if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]
        and hist[i] > 0.15 * hist.max())
    return TrapCensus(
        n_traps=int(len(pts)),
        median_depth=float(np.median(depths)),
        median_spacing=float(np.median(nn)),
        nn_shell_count=len(shells),
        shells=shells,
    )


def trap_census(beta_deg: float = FACE_LOCK_DEG, extent: float = 0.45,
                n: int = 160, depth_cut: float = -0.3) -> TrapCensus:
    """Census of deep traps in the locked icosahedral field."""
    a2, sat = axis_amplitudes(beta_deg)
    return _census(field_axes(beta_deg), a2, wavenumber(sat), extent, n,
                   depth_cut)


def cubic_control_census(extent: float = 0.45, n: int = 160,
                         depth_cut: float = -0.3) -> TrapCensus:
    """Periodic control: 3 orthogonal axes -> simple cubic, ONE NN shell."""
    a2, sat = axis_amplitudes(FACE_LOCK_DEG)
    a2c = np.full(3, float(np.mean(a2)))
    return _census(np.eye(3), a2c, wavenumber(sat), extent, n, depth_cut)


# ----- B3: payload grip ----------------------------------------------------------


def payload_grip(beta_deg: float = FACE_LOCK_DEG, eps: float = 0.22,
                 R: float = DEFAULT_R) -> float:
    """Trap escape gradient / collapse tilt force density.

    >> 1 means matter stays pinned to the lattice during directional
    collapse. Tilt force density ~ sat * eps * ceiling / R (the m=1 stress
    scale from first-principles D5).
    """
    if eps <= 0:
        raise ValueError("eps must be positive (no tilt -> no collapse)")
    a2, sat = axis_amplitudes(beta_deg)
    k = wavenumber(sat)
    pts = tube_interior_points(R, n=32)
    gmax = float(np.max(np.linalg.norm(
        gorkov_gradient(pts, field_axes(beta_deg), a2, k), axis=1)))
    ceiling = float(np.sum(a2) / 2.0)
    tilt = sat * eps * ceiling / R
    return gmax / tilt


# ----- report ---------------------------------------------------------------------


@dataclass(frozen=True)
class BubbleInteriorReport:
    """The inside of the bubble at crystal lock."""
    registry_identity_residual: float   # B0 exact identity check
    axis_projection_check: bool         # {1, 1/sqrt5 x5} at face-lock
    census: TrapCensus                  # B1 quasilattice
    cubic_control_shells: int           # periodic control: 1 shell
    is_quasilattice: bool               # >= 2 shells vs control
    lock_amplitude: float               # V0 at lock = ceiling
    localization_V0_star: float         # B2 derived threshold
    localization_shortfall: float       # V0*/lock (waves stay extended)
    waves_extended_at_lock: bool
    grip_ratio: float                   # B3
    payload_pinned: bool
    catcher_verdict: str


def bubble_interior_report(
    beta_deg: float = FACE_LOCK_DEG, eps: float = 0.22,
    census_n: int = 160,
) -> BubbleInteriorReport:
    """Derive the full interior picture at crystal lock."""
    a2, sat = axis_amplitudes(beta_deg)
    k = wavenumber(sat)
    proj = np.sort(axis_projections(beta_deg))
    proj_ok = bool(
        abs(proj[-1] - 1.0) < 1e-6
        and np.allclose(proj[:-1], 1.0 / SQRT5, atol=1e-6))

    scan = localization_scan(k=k)
    table = dict(zip(scan["V0"], scan["n_ipr"]))

    def fn(v: float) -> np.ndarray:
        key = min(table, key=lambda u: abs(u - v))
        return np.array([table[key]])

    catch = scan_novelty(scan["V0"], fn, n_bits=32,
                         parameter_label="field_amplitude_V0")

    census = trap_census(beta_deg, n=census_n)
    control = cubic_control_census(n=census_n)
    lock_amp = float(np.sum(a2) / 2.0)
    grip = payload_grip(beta_deg, eps)
    return BubbleInteriorReport(
        registry_identity_residual=registry_identity_residual(k),
        axis_projection_check=proj_ok,
        census=census,
        cubic_control_shells=control.nn_shell_count,
        is_quasilattice=bool(census.nn_shell_count >= 2
                             and census.nn_shell_count
                             > control.nn_shell_count),
        lock_amplitude=lock_amp,
        localization_V0_star=float(scan["V0_star"]),
        localization_shortfall=float(scan["V0_star"] / lock_amp),
        waves_extended_at_lock=bool(scan["V0_star"] > lock_amp),
        grip_ratio=float(grip),
        payload_pinned=bool(grip > 10.0),
        catcher_verdict=catch.verdict,
    )


def summarise_interior(r: BubbleInteriorReport) -> str:
    """Human-readable summary."""
    lines = [
        "Inside the bubble at crystal lock",
        f"  B0 axis potential == registry function: residual "
        f"{r.registry_identity_residual:.1e}; projections "
        f"{{1, 1/sqrt5 x5}}: {r.axis_projection_check}",
        f"  B1 trap quasilattice: {r.census.n_traps} deep traps, median "
        f"depth {r.census.median_depth:.2f}, spacing "
        f"{r.census.median_spacing:.3f}, {r.census.nn_shell_count} NN shells "
        f"(cubic control: {r.cubic_control_shells}) -> quasilattice: "
        f"{r.is_quasilattice}",
        f"  B2 wave localization: V0* ~ {r.localization_V0_star:.0f} vs lock "
        f"amplitude {r.lock_amplitude:.2f} (shortfall "
        f"{r.localization_shortfall:.0f}x) -> waves extended at lock: "
        f"{r.waves_extended_at_lock}",
        f"  B3 payload grip: escape/tilt = {r.grip_ratio:.0f} -> pinned "
        f"during collapse: {r.payload_pinned}",
        f"  catcher (V0 scan): {r.catcher_verdict}",
    ]
    return "\n".join(lines)
