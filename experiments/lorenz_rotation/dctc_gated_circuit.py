"""Chaotically-gated Deutsch-CTC circuit (Marrakesh-ready noise-robustness demo).

Scope (stated up front): this is a DEMONSTRATION, not a discovery. The dynamics are
2-qubit and classically exact; the only thing a quantum device adds is the test of whether
the predicted chronology-gating signature survives Heron-r2 noise. It builds on the repo's
hardware-validated D-CTC spectral oracle.

Construct
---------
A chronology-respecting (CR) qubit + a CTC qubit. The joint interaction is a gated partial
swap U(g) = SWAP^g, where g in [0,1] is the 'CTC openness':

    g(a) = clamp((a - 1/2)/width, 0, 1)        a = omega R (the rotation parameter)

so g = 1 when the cylinder is supercritical (chronology horizon present, CTC open) and g = 0
when subcritical (closed). Driving a(t) on the Lorenz attractor makes g(t) flicker across
the threshold.

D-CTC content: at g = 1 (open) U = SWAP, the D-CTC channel E(rho)=Tr_CR[SWAP(sigma(x)rho)SWAP]
= sigma resolves to its fixed point in ONE step (|lambda_2| = 0); at g = 0 (closed) U = I, the
channel is the identity (|lambda_2| = 1, no resolution). So the spectral gap 1 - |lambda_2(g)|
-- the rate at which the CTC 'computes' its self-consistent state -- flickers on the attractor.

Hardware observable: prepare CR in |1>, CTC in |0>, apply U(g), measure the CTC qubit.
P(CTC=1) interpolates 0 (closed) -> 1 (open): the chronology channel transmitting the CR
state into the CTC register iff the cylinder is supercritical. The Marrakesh run measures
P(CTC=1) along the Lorenz g(t) and checks it tracks the ideal sin^2(pi g/2) under noise.

This module builds the circuits and the ideal/classical predictions; it does NOT submit
(that needs IBM Quantum credentials + qiskit-ibm-runtime and the user's action). Attach the
novelty catcher to the measured distribution before any verdict (project rule).
"""

from __future__ import annotations

import numpy as np

from systrophe.ctc.d_ctc import apply_channel, dctc_fixed_point, predict_convergence_via_spectrum

A_CRIT = 0.5

# 2-qubit SWAP and its eigendecomposition (for the fractional power SWAP^g)
_SWAP = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=complex)


def gate_openness(a: float, width: float = 0.1) -> float:
    """CTC openness g(a) in [0,1]: 0 subcritical (a<1/2), ramps to 1 supercritical."""
    return float(np.clip((a - A_CRIT) / width, 0.0, 1.0))


def partial_swap(g: float) -> np.ndarray:
    """U(g) = SWAP^g  (g=0 -> I, g=1 -> SWAP).  Eigenvalues of SWAP are +-1."""
    w, V = np.linalg.eigh(_SWAP)
    powered = np.diag(np.exp(1j * np.pi * g * (w < 0)))  # (-1)^g = e^{i pi g} on singlet
    return V @ powered @ V.conj().T


def classical_signal(g: float) -> dict:
    """Ideal observable + D-CTC spectral diagnostics at openness g."""
    U = partial_swap(g)
    # observable: CR=|1>, CTC=|0> -> P(CTC qubit = 1) after U
    psi_in = np.zeros(4, dtype=complex); psi_in[0b10] = 1.0   # |q1 q0> = |1 0>
    psi_out = U @ psi_in
    p = psi_out.reshape(2, 2)                                  # [CR, CTC]
    p_ctc1 = float(np.sum(np.abs(p[:, 1]) ** 2))
    # D-CTC fixed point / spectral oracle on the gated channel
    sigma = np.array([[0, 0], [0, 1]], dtype=complex)          # CR fixed input |1><1|
    fp = dctc_fixed_point(U, sigma, dim_cr=2, tol=1e-10, max_iter=500)
    try:
        spec = predict_convergence_via_spectrum(U, sigma, dim_cr=2, tol=1e-10)
        lam2 = float(spec.get("lambda_2", float("nan")))
    except Exception:
        lam2 = float("nan")
    return {"g": g, "P_ctc1_ideal": p_ctc1, "lambda_2": lam2,
            "spectral_gap": (1.0 - lam2) if np.isfinite(lam2) else float("nan"),
            "dctc_iterations": int(fp["iterations"]), "dctc_converged": bool(fp["converged"])}


def lorenz_gate_trajectory(a_center: float = 0.7, amp: float = 0.18, width: float = 0.1,
                           lorenz_t_max: float = 60.0, dt: float = 0.02,
                           stride: int = 8) -> dict:
    """Lorenz a(t) -> openness g(t) and the ideal P(CTC=1) signal along it."""
    from rotating_dust_lorenz import RotatingDustLorenz, rotation_parameter_timeseries

    m = RotatingDustLorenz(sigma=10.0, r=28.0, b=8.0 / 3.0)
    traj = m.integrate(np.array([1.0, 1.0, 1.0]), t_max=lorenz_t_max, dt=dt, t_transient=20.0)
    rot = rotation_parameter_timeseries(traj, a0=a_center, eps=amp / a_center, clip_min=0.05)
    t = (rot["t"] - rot["t"][0])[::stride]
    a_t = rot["a"][::stride]
    g_t = np.array([gate_openness(float(a), width) for a in a_t])
    p_t = np.array([classical_signal(float(g))["P_ctc1_ideal"] for g in g_t])
    return {"t": t, "a": a_t, "g": g_t, "P_ctc1_ideal": p_t,
            "n_threshold_crossings": int(np.sum(np.abs(np.diff(np.sign(a_t - A_CRIT))) > 0)),
            "fraction_open": float(np.mean(g_t > 0.5))}


# --------------------------------------------------------------------------- #
# Qiskit circuit construction (for hardware submission); qiskit is optional.
# --------------------------------------------------------------------------- #
def build_circuit(g: float):
    """2-qubit circuit: prep CR=|1>, CTC=|0>, gated partial-SWAP^g, measure CTC.

    Partial SWAP via the iSWAP-family generator: U(g) = exp(-i (pi g/2)(XX+YY+ZZ)/2)
    realises SWAP^g up to a global phase (irrelevant to P(CTC=1))."""
    from qiskit import QuantumCircuit

    theta = np.pi * g / 2.0
    qc = QuantumCircuit(2, 1)
    qc.x(0)                       # CR qubit (q0) -> |1>;  CTC qubit (q1) stays |0>
    qc.rxx(theta, 0, 1)
    qc.ryy(theta, 0, 1)
    qc.rzz(theta, 0, 1)
    qc.measure(1, 0)             # measure CTC qubit
    return qc


def build_batch(g_values):
    """List of circuits over a g-trajectory, ready for transpile + submission."""
    return [build_circuit(float(g)) for g in g_values]


def simulate_batch(g_values, shots: int = 8192, seed: int = 0) -> np.ndarray:
    """Ideal (Aer) P(CTC=1) for each g -- the noiseless reference for the HW run."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    sim = AerSimulator(seed_simulator=seed)
    out = []
    for g in g_values:
        qc = build_circuit(float(g))
        res = sim.run(transpile(qc, sim), shots=shots).result()
        counts = res.get_counts()
        out.append(counts.get("1", 0) / shots)
    return np.array(out)


def calibration_note() -> str:
    return ("Before submission: pull live calibration via "
            "quantum_golden_pendulum.calibration.pull_calibration('ibm_marrakesh'); "
            "transpile opt_level=3 + dynamical decoupling (XpXm) + gate/measure twirling, "
            "8192 shots/point; then attach systrophe.catchers.novelty_catcher to the P(CTC=1) "
            "distribution before any verdict.")
