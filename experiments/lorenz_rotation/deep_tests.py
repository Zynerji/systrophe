"""Deeper tests that exploit the novel construct (chaotically-gated CTC / time machine).

The construct: a dissipative chaotic flow drives the rotation a(t) of a Tipler cylinder,
which drives a time-varying CTC band structure. Two consequences are worth testing because
they are specific to *this* construct, not to the bare attractor:

1.  OBSERVABLE FAITHFULNESS (Takens).  The CTC band log-measure M(t) is a single scalar
    readout of the time machine. If the construct is sound, M(t) is a smooth observable of
    the attractor, so by Takens' embedding theorem a delay embedding of M(t) reconstructs
    the attractor and its correlation dimension equals the attractor's own. I.e. *the time
    machine's band-flicker is a faithful encoding of the chaotic rotation* -- you can read
    the attractor out of the spacetime alone.

2.  CHAOS SYNCHRONIZATION OF A TIPLER PAIR (Pecora-Carroll).  Two cylinders with Lorenz
    rotations, master -> slave coupled through one shared variable, synchronize iff the
    conditional (sub-system) Lyapunov exponents are negative. Synchronized, the pair's
    relative rotation a_2(t) - a_1(t) -> 0: two chaotic time machines lock phase. This
    exploits the SystrophePair structure directly.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from rotating_dust_lorenz import (
    rotation_parameter_timeseries,
    largest_lyapunov_rk4,
)
from chaotic_ctc import chaotic_ctc_timeseries


# --------------------------------------------------------------------------- #
# 1. Delay embedding + correlation dimension of a scalar observable.
# --------------------------------------------------------------------------- #
def autocorr_decorrelation_lag(x: np.ndarray, max_lag: int = 2000) -> int:
    """First lag where the autocorrelation drops below 1/e (a Takens delay)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x)
    if var == 0:
        return 1
    for k in range(1, min(max_lag, n - 1)):
        ac = np.dot(x[:-k], x[k:]) / var
        if ac < np.exp(-1.0):
            return max(k, 1)
    return max(1, min(max_lag, n // 10))


def delay_embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Takens delay-coordinate embedding of scalar x into m dimensions."""
    x = np.asarray(x, dtype=float)
    n = len(x) - (m - 1) * tau
    if n <= 0:
        raise ValueError("series too short for requested embedding")
    return np.column_stack([x[i * tau:i * tau + n] for i in range(m)])


def _gp_dimension(pts: np.ndarray, theiler: int = 0, n_eps: int = 30) -> float:
    """Grassberger-Procaccia correlation dimension of a point cloud.

    With a Theiler window: exclude pairs whose time indices are within `theiler`
    of each other (removes spurious temporal correlation that deflates D2).
    """
    n = len(pts)
    if theiler > 0:
        # pairwise distances honouring the Theiler window (vectorised upper tri)
        dists = []
        for i in range(n):
            j0 = i + 1 + theiler
            if j0 >= n:
                continue
            dd = np.linalg.norm(pts[j0:] - pts[i], axis=1)
            dists.append(dd)
        if not dists:
            return float("nan")
        d = np.concatenate(dists)
    else:
        d = pdist(pts)
    d = d[d > 0]
    dmin, dmax = np.percentile(d, 0.5), np.percentile(d, 99)
    eps = np.logspace(np.log10(dmin), np.log10(dmax), n_eps)
    npair = len(d)
    C = np.array([np.count_nonzero(d < e) / npair for e in eps])
    valid = C > 0
    log_eps, log_C = np.log(eps[valid]), np.log(C[valid])
    band = (C[valid] >= 3e-3) & (C[valid] <= 0.15)
    if band.sum() < 4:
        lo, hi = int(0.2 * len(log_eps)), int(0.6 * len(log_eps))
        band = np.zeros(len(log_eps), dtype=bool)
        band[lo:hi] = True
    slope, _ = np.polyfit(log_eps[band], log_C[band], 1)
    return float(slope)


def correlation_dimension_scalar(
    x: np.ndarray, m: int = 5, tau: int | None = None,
    n_points: int = 5000, theiler: int | None = None,
) -> dict:
    """Correlation dimension of a scalar time series via delay embedding."""
    if tau is None:
        tau = autocorr_decorrelation_lag(x)
    emb = delay_embed(x, m, tau)
    idx = np.linspace(0, len(emb) - 1, min(n_points, len(emb))).astype(int)
    pts = emb[idx]
    if theiler is None:
        theiler = 0  # subsampling already decorrelates
    D2 = _gp_dimension(pts, theiler=theiler)
    return {"D2": D2, "tau": int(tau), "m": int(m), "n_points": len(pts)}


def observable_faithfulness(
    flow, a0: float = 1.5, eps: float = 0.2,
    t_max: float = 1200.0, dt: float = 0.01, stride: int = 2,
    R: float = 1.0, r_min: float = 1.05, r_max: float = 20.0,
) -> dict:
    """Compare D2 of the attractor (state space) to D2 reconstructed from the
    scalar CTC band-measure M(t) alone.

    A match certifies that the time-machine band-flicker faithfully encodes the
    chaotic rotation (the construct is a smooth observable of the attractor).
    """
    traj = flow.integrate(flow.default_s0, t_max, dt=dt, t_transient=flow.t_transient)
    # state-space D2 (3-D point cloud)
    idx = np.linspace(0, len(traj["s"]) - 1, 5000).astype(int)
    D2_state = _gp_dimension(traj["s"][idx])

    # CTC observable: a(t) -> M(t)
    rot = rotation_parameter_timeseries(traj, a0=a0, eps=eps, clip_min=0.55)
    cts = chaotic_ctc_timeseries(rot, R=R, r_min=r_min, r_max=r_max, stride=stride)
    M = cts["log_measure"]
    D2_obs = correlation_dimension_scalar(M, m=5)

    return {
        "flow": flow.name,
        "D2_state_space": D2_state,
        "D2_ctc_observable": D2_obs["D2"],
        "embed_tau": D2_obs["tau"],
        "relative_diff": abs(D2_obs["D2"] - D2_state) / max(D2_state, 1e-9),
        "n_ctc_samples": len(M),
    }


# --------------------------------------------------------------------------- #
# 2. Pecora-Carroll synchronization of two Lorenz-rotation cylinders.
# --------------------------------------------------------------------------- #
def pecora_carroll_sync(
    sigma: float = 10.0, r: float = 28.0, b: float = 8.0 / 3.0,
    t_max: float = 60.0, dt: float = 0.005, drive: str = "x",
) -> dict:
    """Master -> slave Lorenz synchronization.

    The slave integrates the full Lorenz system but its `drive` variable is
    continuously replaced by the master's (the Pecora-Carroll replacement
    scheme). For Lorenz with the x-drive, the (y, z) sub-system has negative
    conditional Lyapunov exponents, so the slave synchronizes to the master.

    Returns the synchronization-error decay and whether it locked.
    """
    def lorenz(s):
        x, y, z = s
        return np.array([sigma * (y - x), r * x - y - x * z, x * y - b * z])

    def rk4(f, s, h):
        k1 = f(s); k2 = f(s + 0.5 * h * k1)
        k3 = f(s + 0.5 * h * k2); k4 = f(s + h * k3)
        return s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    drive_idx = {"x": 0, "y": 1, "z": 2}[drive]
    m = np.array([1.0, 1.0, 1.0])           # master
    sl = np.array([10.0, -5.0, 20.0])       # slave: very different IC
    n = int(t_max / dt)
    t = np.empty(n); err = np.empty(n)
    for k in range(n):
        m = rk4(lorenz, m, dt)
        sl = rk4(lorenz, sl, dt)
        sl[drive_idx] = m[drive_idx]        # Pecora-Carroll replacement
        t[k] = (k + 1) * dt
        err[k] = float(np.linalg.norm(sl - m))
    tail = err[int(0.7 * n):]
    return {
        "t": t, "error": err, "drive": drive,
        "final_error": float(err[-1]),
        "tail_mean_error": float(tail.mean()),
        "locked": bool(tail.mean() < 1e-3),
    }


def conditional_lyapunov(
    sigma: float = 10.0, r: float = 28.0, b: float = 8.0 / 3.0,
    drive: str = "x", t_max: float = 300.0, dt: float = 0.005,
) -> dict:
    """Largest conditional (sub-system) Lyapunov exponent of the driven slave.

    Negative => guaranteed synchronization (Pecora-Carroll criterion). Computed
    from the response sub-system's Jacobian along the master trajectory.
    """
    free = [i for i in range(3) if i != {"x": 0, "y": 1, "z": 2}[drive]]

    def lorenz(s):
        x, y, z = s
        return np.array([sigma * (y - x), r * x - y - x * z, x * y - b * z])

    def jac(s):
        x, y, z = s
        return np.array([[-sigma, sigma, 0.0],
                         [r - z, -1.0, -x],
                         [y, x, -b]])

    def rk4(f, s, h):
        k1 = f(s); k2 = f(s + 0.5 * h * k1)
        k3 = f(s + 0.5 * h * k2); k4 = f(s + h * k3)
        return s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    s = np.array([1.0, 1.0, 1.0])
    for _ in range(int(20.0 / dt)):
        s = rk4(lorenz, s, dt)

    v = np.ones(len(free)); v /= np.linalg.norm(v)
    n = int(t_max / dt)
    log_sum = 0.0
    for _ in range(n):
        J = jac(s)[np.ix_(free, free)]   # response sub-Jacobian
        v = v + dt * (J @ v)             # tangent evolution (Euler on subsystem)
        s = rk4(lorenz, s, dt)
        nv = np.linalg.norm(v)
        log_sum += np.log(nv)
        v /= nv
    return {"drive": drive, "conditional_lyapunov_max": float(log_sum / (n * dt)),
            "free_subsystem": free}


# --------------------------------------------------------------------------- #
# 3. Catcher chaos-onset universality across attractors.
# --------------------------------------------------------------------------- #
def chaos_onset_scan(flow, t_le: float = 100.0) -> dict:
    """Locate each attractor's chaos onset two independent ways and compare.

    (a) Lyapunov: first bifurcation-parameter value where the largest exponent
        crosses positive.
    (b) Catcher: address-space lambda_2 sharp feature in the settling-robust
        |X|-distribution fingerprint scanned over the same parameter axis.

    Universality claim: the catcher (no dynamics, just address-space topology of
    an observable) localises the same onset the Lyapunov exponent does, for
    every attractor.
    """
    from systrophe.catchers.novelty_catcher import scan_novelty

    bif = flow.bifurcation
    scan = np.asarray(bif["scan"], dtype=float)
    s0 = flow.default_s0
    t_trans = flow.t_transient

    def le_of(p):
        f = flow.with_param(p)
        return largest_lyapunov_rk4(f, s0, t_max=t_le, dt=0.005, t_transient=t_trans)

    le_curve = np.array([le_of(p) for p in scan])
    # A non-finite LE marks an unbounded (non-attractor) parameter, not chaos;
    # exclude it so a divergent scan edge isn't mistaken for the onset.
    le_pos = np.isfinite(le_curve) & (le_curve > 0.02)
    le_onset = float(scan[int(np.argmax(le_pos))]) if le_pos.any() else None

    q = np.linspace(0.0, 1.0, 16)

    def fingerprint(p):
        f = flow.with_param(p)
        t_tr = 3.0 * t_trans
        traj = f.integrate(s0, t_max=t_tr + 150.0, dt=0.03, t_transient=t_tr)
        X = traj["s"][:, 0]
        if traj.get("diverged") or len(X) < 16:
            # unbounded / non-attractor regime: sentinel far from any bounded run
            return np.full(16, 1e6)
        return np.quantile(np.abs(X), q)

    res = scan_novelty(scan, fingerprint, n_bits=32, radii=(4, 8, 12, 16),
                       parameter_label=f"{flow.name}:{bif['param']}")
    catcher_feats = [f["parameter_value"] for f in res.sharp_features]
    # nearest catcher feature to the Lyapunov onset
    nearest = None
    if catcher_feats and le_onset is not None:
        nearest = float(min(catcher_feats, key=lambda v: abs(v - le_onset)))

    return {
        "flow": flow.name,
        "physical": getattr(flow, "physical", False),
        "param": bif["param"],
        "scan_min": float(scan.min()), "scan_max": float(scan.max()),
        "lyapunov_onset": le_onset,
        "catcher_verdict": res.verdict,
        "catcher_features": [round(x, 3) for x in catcher_feats],
        "catcher_nearest_to_le_onset": nearest,
        "onset_agreement": (None if (nearest is None or le_onset is None)
                            else abs(nearest - le_onset)),
        "lyapunov_curve": le_curve.tolist(),
        "scan": scan.tolist(),
    }
