"""Dodecahedral Casimir resonator in the horn-torus throat: orientation sweep.

Hypothesis under test (2026-06-09, demo parity:
``~/.local/bin/dinos_systrophe_dodeca_demo.html``):

    Seat a regular dodecahedron at the pinch of the Knopp horn torus with its
    points reaching toward the inner horns. The Casimir gap between the horn
    surface and the polyhedron pumps harmonics that radiate from the 12
    pentagonal faces into the torus volume. Sweeping the orientation from
    point-aligned to face-aligned lets the horn stretch and flatten to tile
    the pentagon as a five-armed spiral, multiplying the facing area ->
    more vacuum noise -> stronger standing waves -> a rising-frequency
    feedback loop, until the tube interior is saturated with a full-spectrum
    standing-wave field that directionally collapses the warp bubble.

What this module actually adjudicates (model units, NOT SI physics):

1. Exact geometry: signed gap between dodecahedron surface samples (20
   vertices + 12 face centres) and the horn torus, as a function of the
   orientation angle beta along the vertex->face geodesic (face lock at
   beta = arccos(r_in/r_circ) = 37.377 deg).
2. A proximity-force-approximation (PFA) drive proxy: sum_i A_i/(g_i+g0)^3
   with a parallel-plate alignment weight on faces. Dimensionless.
3. The spiral-tiling hypothesis encoded as an area factor (1 + gain*spiral)
   that engages on face lock. The module tests its CONSEQUENCES (drive
   amplification, broadband pumping), not its derivation -- there is no
   first-principles model here of a horn surface tiling a pentagon.
4. A 24-mode standing-wave comb ODE: point alignment pumps a sparse
   pentagonal comb (every 5th mode); face lock + spiral pumps broadband.
   Full-spectrum saturation (all modes above half amplitude) is the stated
   requirement for directional bubble collapse.
5. Feedback self-limitation: gaps are bounded below by contact, mode
   amplitudes by 1, so the loop saturates -- it does not run away.

HONEST CAVEATS (consistent with knopp_toroidal_casimir_dodecahedron):
- The drive is a coherent-pump proxy. Vacuum fluctuations are not a free
  energy reservoir; "Casimir noise rings up the cavity" conflates vacuum
  modes with pumped modes exactly as flagged in the nested-cavity module.
  Geometry and alignment are the only claims adjudicated here.
- No SI feasibility is claimed; Ford-Roman bookkeeping lives in
  knopp_ratchet.bias_energy_ledger and is not re-derived.
- Per the Systrophe deliverable rule, the address-space novelty catcher
  runs on every orientation sweep and its verdict is part of the report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty

PHI = (1.0 + math.sqrt(5.0)) / 2.0
#: dodecahedron inradius / circumradius == cos(vertex->face geodesic angle)
INRADIUS_RATIO = PHI ** 2 / math.sqrt(3.0 * (PHI ** 2 + 1.0))
#: orientation angle (deg) at which the up-vertex gives way to the up-face
FACE_LOCK_DEG = math.degrees(math.acos(INRADIUS_RATIO))

DEFAULT_R = 0.66          # horn-torus major == tube radius (demo parity)
DEFAULT_SCALE = 0.26      # dodeca circumradius (demo parity; halved 2026-06-09
                          # per user -- smaller dodeca sits CLOSER to the pinch)
DEFAULT_G0 = 0.02         # PFA gap regulariser (contact floor)
DEFAULT_AREA_GAIN = 6.0   # spiral-tiling area factor: 1 + gain at full engage
FACE_AREA_WEIGHT = 2.5    # relative PFA area of a pentagonal face
VERTEX_AREA_WEIGHT = 0.4  # relative PFA area of a point


# ----- geometry -------------------------------------------------------------


def dodecahedron_vertices() -> np.ndarray:
    """The 20 unit vertices: (+-1,+-1,+-1), (0,+-1/phi,+-phi) + cyclic."""
    ip = 1.0 / PHI
    verts = []
    for x in (-1, 1):
        for y in (-1, 1):
            for z in (-1, 1):
                verts.append((x, y, z))
    for a, b in ((ip, PHI), (-ip, PHI), (ip, -PHI), (-ip, -PHI)):
        verts.append((0.0, a, b))
        verts.append((a, b, 0.0))
        verts.append((b, 0.0, a))
    v = np.asarray(verts, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def dodecahedron_face_axes() -> np.ndarray:
    """The 6 unique face axes (12 faces = +-axes), unit length."""
    axes = np.asarray(
        [(0, 1, PHI), (0, 1, -PHI), (1, PHI, 0),
         (1, -PHI, 0), (PHI, 0, 1), (-PHI, 0, 1)], dtype=float)
    return axes / np.linalg.norm(axes, axis=1, keepdims=True)


def horn_torus_sdf(points: np.ndarray, R: float = DEFAULT_R) -> np.ndarray:
    """Signed distance to the horn torus (major == tube radius == R).

    Negative inside the tube. The pinch point is the origin.
    """
    p = np.atleast_2d(np.asarray(points, dtype=float))
    rho = np.hypot(p[:, 0], p[:, 2])
    return np.hypot(rho - R, p[:, 1]) - R


def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    x, y, z = ax
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def vertex_up_rotation() -> np.ndarray:
    """Rotation mapping the vertex (1,1,1)/sqrt(3) onto +y (point into horn)."""
    v0 = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    b1 = np.cross(v0, [0.0, 0.0, 1.0])
    b1 = b1 / np.linalg.norm(b1)
    b2 = np.cross(b1, v0)
    return np.vstack([b1, v0, b2])


def sweep_rotation(beta_deg: float) -> np.ndarray:
    """Orientation at sweep angle beta along the vertex->face geodesic.

    beta = 0: vertex up. beta = FACE_LOCK_DEG: adjacent face axis up.
    """
    B = vertex_up_rotation()
    f_world = B @ (np.array([0.0, 1.0, PHI]) / math.sqrt(1.0 + PHI ** 2))
    # axis = f x y so +beta carries the face axis TOWARD +y (face up at 37.377)
    sweep_axis = np.cross(f_world, [0.0, 1.0, 0.0])
    return _rotation_about(sweep_axis, math.radians(beta_deg)) @ B


# ----- alignment + PFA drive ------------------------------------------------


@dataclass(frozen=True)
class AlignmentState:
    """Geometry + drive at one dodecahedron orientation."""
    beta_deg: float
    min_gap: float            # smallest |gap| over all surface samples
    vertex_align: float       # 0..1 best point-toward-horn alignment
    face_align: float         # 0..1 best face-toward-horn alignment
    spiral: float             # 0..1 spiral tiling engagement (from face lock)
    area_factor: float        # 1 + area_gain * spiral
    drive_raw: float          # PFA sum, area factor applied (model units)


def alignment_state(
    beta_deg: float,
    scale: float = DEFAULT_SCALE,
    R: float = DEFAULT_R,
    g0: float = DEFAULT_G0,
    area_gain: float = DEFAULT_AREA_GAIN,
) -> AlignmentState:
    """Compute gaps, alignment scores and the PFA drive at one orientation."""
    if scale <= 0 or R <= 0 or g0 <= 0:
        raise ValueError("scale, R and g0 must be positive")
    rot = sweep_rotation(beta_deg)
    verts = (dodecahedron_vertices() @ rot.T) * scale
    gaps_v = np.abs(horn_torus_sdf(verts, R))
    raw = float(np.sum(VERTEX_AREA_WEIGHT / (gaps_v + g0) ** 3))
    vert_align = float(np.max(
        np.abs(verts[:, 1] / scale) ** 8 * np.exp(-gaps_v / 0.25)))

    axes = dodecahedron_face_axes() @ rot.T
    face_align = 0.0
    min_gap = float(np.min(gaps_v))
    h = 1e-3
    for sign in (1.0, -1.0):
        for n in sign * axes:
            c = n * INRADIUS_RATIO * scale
            d = float(horn_torus_sdf(c[None, :], R)[0])
            g = abs(d)
            min_gap = min(min_gap, g)
            grad = np.array([
                float(horn_torus_sdf((c + h * e)[None, :], R)[0]
                      - horn_torus_sdf((c - h * e)[None, :], R)[0])
                for e in np.eye(3)]) / (2 * h)
            grad = grad / max(np.linalg.norm(grad), 1e-12)
            to_surf = grad * (1.0 if d < 0 else -1.0)
            af = max(0.0, float(np.dot(n, to_surf)))   # parallel-plate factor
            raw += FACE_AREA_WEIGHT * (0.2 + 0.8 * af ** 2) / (g + g0) ** 3
            fa = abs(n[1]) ** 8 * math.exp(-g / 0.25)
            face_align = max(face_align, fa)
    face_align = min(1.0, face_align)
    vert_align = min(1.0, vert_align)

    spiral = min(1.0, max(0.0, (face_align - 0.25) / 0.3))
    area_factor = 1.0 + area_gain * spiral
    return AlignmentState(
        beta_deg=float(beta_deg),
        min_gap=float(min_gap),
        vertex_align=vert_align,
        face_align=face_align,
        spiral=float(spiral),
        area_factor=float(area_factor),
        drive_raw=float(raw * area_factor),
    )


def normalised_drive(state: AlignmentState, calibration_raw: float) -> float:
    """Soft-normalise the PFA drive to 0..1 against a calibration raw value
    (demo parity: calibration = 2x the vertex-up raw drive)."""
    if calibration_raw <= 0:
        raise ValueError("calibration_raw must be positive")
    return state.drive_raw / (state.drive_raw + calibration_raw)


# ----- standing-wave mode comb ----------------------------------------------


@dataclass(frozen=True)
class SaturationReport:
    """Steady saturation of the 24-mode standing-wave comb."""
    drive: float
    saturation: float          # fraction of modes above half amplitude
    full_spectrum: bool        # all modes above half amplitude
    amplitudes: np.ndarray
    collapse: float            # sat * spiral: directional collapse engaged


def mode_saturation(
    drive: float,
    vertex_align: float,
    face_align: float,
    spiral: float,
    Q: float = 60.0,
    n_modes: int = 24,
    t_end: float = 40.0,
    dt: float = 0.02,
) -> SaturationReport:
    """Integrate the mode-comb ODE to steady state.

    da_k/dt = drive * coup_k * 2.2 * (1 - a_k) - a_k * 1.5/sqrt(Q)

    coup_k = vertex_align * comb_k * 0.6 + face_align * (0.3 + 0.7*spiral)
    comb_k = 1 on every 5th mode (pentagonal comb), 0.15 otherwise: point
    alignment is narrowband, face lock + spiral is broadband.
    """
    if Q <= 0:
        raise ValueError("Q must be positive")
    if not (0 <= drive <= 1):
        raise ValueError("drive must be in [0, 1]")
    k = np.arange(n_modes)
    comb = np.where(k % 5 == 0, 1.0, 0.15)
    coup = vertex_align * comb * 0.6 + face_align * (0.3 + 0.7 * spiral)
    amps = np.zeros(n_modes)
    decay = 1.5 / math.sqrt(Q)
    for _ in range(int(t_end / dt)):
        amps += dt * (drive * coup * 2.2 * (1.0 - amps) - amps * decay)
        np.clip(amps, 0.0, 1.0, out=amps)
    sat = float(np.mean(amps > 0.5))
    return SaturationReport(
        drive=float(drive),
        saturation=sat,
        full_spectrum=bool(np.all(amps > 0.5)),
        amplitudes=amps,
        collapse=float(sat * spiral),
    )


# ----- orientation sweep + catcher -------------------------------------------


@dataclass(frozen=True)
class OrientationSweepReport:
    """Full sweep of the dodecahedron orientation with catcher verdict."""
    beta_deg: np.ndarray
    drive: np.ndarray              # normalised 0..1
    face_align: np.ndarray
    vertex_align: np.ndarray
    min_gap: np.ndarray
    face_lock_deg: float
    peak_beta_deg: float
    drive_at_face_lock: float
    drive_at_vertex_lock: float
    face_over_vertex: float        # drive ratio, spiral hypothesis engaged
    catcher_verdict: str
    catcher_sharp_features: int


def orientation_sweep(
    n_angles: int = 181,
    scale: float = DEFAULT_SCALE,
    R: float = DEFAULT_R,
    area_gain: float = DEFAULT_AREA_GAIN,
) -> OrientationSweepReport:
    """Sweep beta over [0, 180] deg; run the novelty catcher on the curve."""
    betas = np.linspace(0.0, 180.0, n_angles)
    states = [alignment_state(b, scale, R, area_gain=area_gain)
              for b in betas]
    cal = 2.0 * states[0].drive_raw / states[0].area_factor
    drive = np.array([normalised_drive(s, cal) for s in states])
    fa = np.array([s.face_align for s in states])
    va = np.array([s.vertex_align for s in states])
    mg = np.array([s.min_gap for s in states])

    i_face = int(np.argmin(np.abs(betas - FACE_LOCK_DEG)))
    table = {float(b): np.array([d, f, v, g])
             for b, d, f, v, g in zip(betas, drive, fa, va, mg)}

    def fn(b: float) -> np.ndarray:
        key = min(table, key=lambda x: abs(x - b))
        return table[key]

    catch = scan_novelty(betas, fn, n_bits=32,
                         parameter_label="orientation_beta_deg")
    return OrientationSweepReport(
        beta_deg=betas,
        drive=drive,
        face_align=fa,
        vertex_align=va,
        min_gap=mg,
        face_lock_deg=FACE_LOCK_DEG,
        peak_beta_deg=float(betas[int(np.argmax(drive))]),
        drive_at_face_lock=float(drive[i_face]),
        drive_at_vertex_lock=float(drive[0]),
        face_over_vertex=float(drive[i_face] / max(drive[0], 1e-12)),
        catcher_verdict=catch.verdict,
        catcher_sharp_features=len(catch.sharp_features),
    )


# ----- feedback loop ----------------------------------------------------------


@dataclass(frozen=True)
class FeedbackReport:
    """Closed-loop state at one orientation: alignment -> spiral -> drive ->
    saturation -> collapse, with the self-limitation check."""
    state: AlignmentState
    saturation: SaturationReport
    self_limiting: bool
    interior_saturated: bool       # >= 95% of modes ringing
    directional_collapse: float    # collapse * m=1 gate (horn-twist eps)


def feedback_equilibrium(
    beta_deg: float,
    scale: float = DEFAULT_SCALE,
    R: float = DEFAULT_R,
    Q: float = 60.0,
    area_gain: float = DEFAULT_AREA_GAIN,
    eps_twist: float = 0.25,
) -> FeedbackReport:
    """Run the full loop at a fixed orientation.

    The loop is self-limiting by construction: gaps are bounded below by
    the contact floor g0, the area factor by (1 + area_gain), and mode
    amplitudes by 1 -- saturation asymptotes instead of running away.
    The report verifies those bounds numerically.

    Directionality (knopp_dodeca_first_principles D5): the saturated C_5
    field has zero m=1 moment, so DIRECTIONAL collapse is gated by the
    horn-twist steering lobe: directional_collapse = sat*spiral*min(1, 4*eps).
    eps_twist = 0 gives a saturated but directionless interior.
    """
    st = alignment_state(beta_deg, scale, R, area_gain=area_gain)
    ref = alignment_state(0.0, scale, R, area_gain=area_gain)
    cal = 2.0 * ref.drive_raw / ref.area_factor
    d = normalised_drive(st, cal)
    satrep = mode_saturation(d, st.vertex_align, st.face_align, st.spiral, Q=Q)
    self_limiting = (
        d <= 1.0
        and st.area_factor <= 1.0 + area_gain + 1e-12
        and float(np.max(satrep.amplitudes)) <= 1.0
        and satrep.saturation <= 1.0
    )
    if eps_twist < 0:
        raise ValueError("eps_twist must be >= 0")
    m1_gate = min(1.0, 4.0 * eps_twist)
    return FeedbackReport(
        state=st,
        saturation=satrep,
        self_limiting=bool(self_limiting),
        interior_saturated=bool(satrep.saturation >= 0.95),
        directional_collapse=float(satrep.collapse * m1_gate),
    )


@dataclass(frozen=True)
class ScaleSweepReport:
    """Sweep of the dodecahedron size: find the optimal scale.

    Objective: the weakest-mode saturation margin at face-lock,
        margin = gamma * drive * coup_min * sqrt(Q) / kappa_0,
    (margin > 1 means even the weakest comb mode rings past half amplitude:
    full spectrum with headroom). The optimum is the unconstrained argmax:
    near-zero gaps are not a failure mode here -- the PFA proxy carries a
    contact floor g0, and the warp shell is a metric feature, not a wall.
    `contact_free` (worst orientation gap >= clearance) is reported as a
    DIAGNOSTIC: across a live orientation sweep some surface sample always
    rides within ~1e-3 of the wall, at every scale.
    Point-lock saturation is scale-pinned by the self-calibration (drive
    = 1/3 by construction), so the contrast story is carried by face-lock.
    """
    scale: np.ndarray
    drive_face: np.ndarray         # normalised face-lock drive
    margin_face: np.ndarray        # weakest-mode margin (>1 = full spectrum)
    saturation_face: np.ndarray
    min_gap_face: np.ndarray
    worst_gap_any_beta: np.ndarray  # min gap over the orientation sweep
    contact_free: np.ndarray        # worst gap >= clearance
    clearance: float
    optimal_scale: float
    optimal_margin: float
    catcher_verdict: str


def scale_sweep(
    scales: np.ndarray | None = None,
    R: float = DEFAULT_R,
    Q: float = 60.0,
    area_gain: float = DEFAULT_AREA_GAIN,
    clearance: float = 0.005,
    n_beta_check: int = 37,
) -> ScaleSweepReport:
    """Sweep the dodeca circumradius and locate the optimal size."""
    if scales is None:
        scales = np.linspace(0.05, 0.90, 35)
    scales = np.asarray(scales, dtype=float)
    if np.any(scales <= 0):
        raise ValueError("scales must be positive")
    betas = np.linspace(0.0, 180.0, n_beta_check)
    gamma, kappa0 = 2.2, 1.5

    drive_f = np.empty_like(scales)
    margin_f = np.empty_like(scales)
    sat_f = np.empty_like(scales)
    gap_f = np.empty_like(scales)
    worst = np.empty_like(scales)
    for i, s in enumerate(scales):
        ref = alignment_state(0.0, s, R, area_gain=area_gain)
        st = alignment_state(FACE_LOCK_DEG, s, R, area_gain=area_gain)
        cal = 2.0 * ref.drive_raw / ref.area_factor
        d = normalised_drive(st, cal)
        coup_min = st.vertex_align * 0.15 * 0.6 \
            + st.face_align * (0.3 + 0.7 * st.spiral)
        drive_f[i] = d
        margin_f[i] = gamma * d * coup_min * math.sqrt(Q) / kappa0
        sat_f[i] = mode_saturation(
            d, st.vertex_align, st.face_align, st.spiral, Q=Q).saturation
        gap_f[i] = st.min_gap
        worst[i] = min(alignment_state(b, s, R, area_gain=area_gain).min_gap
                       for b in betas)
    free = worst >= clearance
    i_opt = int(np.argmax(margin_f))
    table = dict(zip(scales, margin_f))

    def fn(s: float) -> np.ndarray:
        key = min(table, key=lambda x: abs(x - s))
        return np.array([table[key]])

    catch = scan_novelty(scales, fn, n_bits=32, parameter_label="dodeca_scale")
    return ScaleSweepReport(
        scale=scales,
        drive_face=drive_f,
        margin_face=margin_f,
        saturation_face=sat_f,
        min_gap_face=gap_f,
        worst_gap_any_beta=worst,
        contact_free=free,
        clearance=float(clearance),
        optimal_scale=float(scales[i_opt]),
        optimal_margin=float(margin_f[i_opt]),
        catcher_verdict=catch.verdict,
    )


def summarise_scale_sweep(r: ScaleSweepReport) -> str:
    """Human-readable summary."""
    n_free = int(np.sum(r.contact_free))
    lines = [
        "Dodeca scale sweep (objective: face-lock weakest-mode margin)",
        f"  scales swept               = {len(r.scale)} in "
        f"[{r.scale[0]:.3f}, {r.scale[-1]:.3f}]",
        f"  contact-free diag (gap >= {r.clearance}) = {n_free}/{len(r.scale)}",
        f"  OPTIMAL scale              = {r.optimal_scale:.3f} "
        f"(margin {r.optimal_margin:.2f}, >1 = full spectrum w/ headroom)",
        f"  catcher verdict            = {r.catcher_verdict}",
    ]
    return "\n".join(lines)


def summarise_sweep(r: OrientationSweepReport) -> str:
    """Human-readable summary."""
    lines = [
        "Dodeca-in-horn-torus orientation sweep (PFA proxy, model units)",
        f"  face-lock angle            = {r.face_lock_deg:.3f} deg",
        f"  drive at vertex lock (0deg)= {r.drive_at_vertex_lock:.4f}",
        f"  drive at face lock         = {r.drive_at_face_lock:.4f}",
        f"  face/vertex drive ratio    = {r.face_over_vertex:.3f}",
        f"  global peak at beta        = {r.peak_beta_deg:.1f} deg",
        f"  catcher verdict            = {r.catcher_verdict} "
        f"({r.catcher_sharp_features} sharp features)",
    ]
    return "\n".join(lines)
