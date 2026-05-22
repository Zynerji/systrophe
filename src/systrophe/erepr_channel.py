"""ER=EPR quantum information channel — entanglement-mediated qubit transport.

This is the one genuinely buildable capability from the wormhole map
(``wormhole_map``): the ER=EPR channel. Under Maldacena–Susskind ER=EPR, the
shared entanglement between two systems IS an Einstein–Rosen bridge, and sending
a qubit through that entanglement is the operational realization of "traversing
the wormhole" (the Gao–Jafferis–Wall / Maldacena–Qi teleportation protocol; a
one-qubit version was demonstrated on a quantum processor in 2022).

What is built here (and works, fidelity 1)
------------------------------------------
A deterministic, measurement-free teleportation channel that transports an
arbitrary (possibly entangled) qubit register through pre-shared EPR pairs using
coherent corrections — i.e. the entanglement resource = the "wormhole", and the
qubit comes out the far mouth with unit fidelity.

Honest boundaries (no overclaim)
--------------------------------
1. NOT faster-than-light. The corrections require a coupling between the two
   mouths (a physical/classical connection); by the no-signaling theorem no
   information moves until that connection is used. The channel is a quantum-
   network primitive (distributed QC, entanglement-assisted comms), not FTL.
2. The full GJW "negative-energy size-winding" dynamics (scramble + e^{igV}
   coupling + unscramble, with the traversability sign-asymmetry) require
   SYK-like chaotic scrambling. ``gjw_coupling_scan`` shows — as a documented
   NEGATIVE result — that a generic Haar scrambler does NOT reproduce the
   signature; the operational channel above does not depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- minimal state-vector machinery ---------------------------------------

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
_BELL = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)


def _apply(state: np.ndarray, U: np.ndarray, qubits: list[int], N: int) -> np.ndarray:
    """Apply a k-qubit gate U to `qubits` of an N-qubit state vector."""
    k = len(qubits)
    psi = state.reshape([2] * N)
    psi = np.moveaxis(psi, qubits, range(k)).reshape(2 ** k, -1)
    psi = U @ psi
    psi = psi.reshape([2] * k + [2] * (N - k))
    psi = np.moveaxis(psi, range(k), qubits)
    return psi.reshape(-1)


def _bell(state: np.ndarray, a: int, b: int, N: int) -> np.ndarray:
    return _apply(_apply(state, _H, [a], N), _CNOT, [a, b], N)


def _reduced_density(state: np.ndarray, keep: list[int], N: int) -> np.ndarray:
    k = len(keep)
    psi = state.reshape([2] * N)
    psi = np.moveaxis(psi, keep, range(k)).reshape(2 ** k, -1)
    return psi @ psi.conj().T


def _single_qubit_state(theta: float, phi: float) -> np.ndarray:
    return np.array([np.cos(theta / 2),
                     np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


# --- the working channel: coherent EPR teleportation ----------------------


def teleport_qubit(input_state: np.ndarray) -> tuple[np.ndarray, float]:
    """Send one qubit through an EPR pair (the 'wormhole'); return (rho_out, F).

    Deterministic, measurement-free: EPR resource + coherent X/Z corrections.
    F is the fidelity of the output qubit with the input pure state.
    """
    input_state = np.asarray(input_state, dtype=complex)
    input_state = input_state / np.linalg.norm(input_state)
    # qubits: M=0 (message), A=1, B=2 (B = far mouth)
    N = 3
    s = np.zeros(2 ** N, dtype=complex)
    s[0] = 1.0
    s = _bell(s, 1, 2, N)                       # EPR resource A-B = the wormhole
    s = _apply(s, _state_inject(input_state), [0], N)  # load message into mouth
    s = _apply(s, _CNOT, [0, 1], N)
    s = _apply(s, _H, [0], N)
    s = _apply(s, _CNOT, [1, 2], N)            # X correction (coherent)
    s = _apply(s, _CZ, [0, 2], N)              # Z correction (coherent)
    rho_out = _reduced_density(s, [2], N)
    F = float(np.real(input_state.conj() @ rho_out @ input_state))
    return rho_out, F


def _state_inject(input_state: np.ndarray) -> np.ndarray:
    """1-qubit unitary mapping |0> -> input_state (to load the message)."""
    a, b = input_state
    # complete to a unitary [[a, -conj(b)],[b, conj(a)]]
    return np.array([[a, -np.conj(b)], [b, np.conj(a)]], dtype=complex)


def channel_entanglement_fidelity(m_qubits: int = 1) -> float:
    """Entanglement fidelity of the m-qubit ER=EPR channel (1.0 = perfect).

    Each message qubit is half of a Bell pair with a reference; after teleporting
    all messages to the far mouths, we measure how well the reference-output
    pairs remain Bell pairs. Beats the classical bound 1/2; perfect = 1.
    """
    m = m_qubits
    # layout: P_i (refs), Msg_i, A_i, B_i   for i in 0..m-1
    N = 4 * m
    P = list(range(0, m))
    Msg = list(range(m, 2 * m))
    A = list(range(2 * m, 3 * m))
    B = list(range(3 * m, 4 * m))
    s = np.zeros(2 ** N, dtype=complex)
    s[0] = 1.0
    for i in range(m):
        s = _bell(s, P[i], Msg[i], N)          # unknown input (ref-message Bell)
        s = _bell(s, A[i], B[i], N)            # EPR resource
    for i in range(m):                          # teleport each Msg_i -> B_i
        s = _apply(s, _CNOT, [Msg[i], A[i]], N)
        s = _apply(s, _H, [Msg[i]], N)
        s = _apply(s, _CNOT, [A[i], B[i]], N)
        s = _apply(s, _CZ, [Msg[i], B[i]], N)
    # fidelity of (P_i, B_i) pairs with Bell^{⊗m}
    keep = []
    for i in range(m):
        keep += [P[i], B[i]]
    rho = _reduced_density(s, keep, N)
    target = np.array([1.0], dtype=complex)
    for _ in range(m):
        target = np.kron(target, _BELL)
    return float(np.real(target.conj() @ rho @ target))


# --- GJW coupling diagnostic (documented NEGATIVE result for Haar) --------


def _haar(k: int, rng: np.random.Generator) -> np.ndarray:
    z = (rng.normal(size=(2 ** k, 2 ** k)) + 1j * rng.normal(size=(2 ** k, 2 ** k))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))


def _zz_coupling(state: np.ndarray, g: float, Lq: list[int], Rq: list[int],
                 N: int) -> np.ndarray:
    n = len(Lq)
    idx = np.arange(2 ** N)
    bits = (idx[:, None] >> np.arange(N)[::-1]) & 1
    sgn = 1 - 2 * bits
    phase = np.sum(sgn[:, Lq] * sgn[:, Rq], axis=1) / n
    return state * np.exp(1j * g * phase)


def gjw_coupling_scan(g_grid: np.ndarray | None = None, n: int = 3,
                      seed: int = 3) -> dict:
    """Maldacena-Qi/GJW coupling-mediated transmission vs coupling g.

    Scramble L (Haar), apply e^{igV} (V = Σ Z_L Z_R / n), unscramble R, read the
    far qubit. HONEST RESULT: a generic Haar scrambler lacks 'size winding', so
    this does NOT reproduce the traversable signature — fidelity stays at/below
    the no-coupling baseline and the sign-asymmetry is noise. The working channel
    above does not rely on this; a faithful signature needs SYK dynamics.
    """
    if g_grid is None:
        g_grid = np.linspace(-np.pi, np.pi, 41)
    rng = np.random.default_rng(seed)
    N = 2 + 2 * n
    P, M = 0, 1
    L = list(range(2, 2 + n))
    R = list(range(2 + n, 2 + 2 * n))
    U = _haar(n, rng)

    def run(g: float) -> float:
        s = np.zeros(2 ** N, dtype=complex)
        s[0] = 1.0
        for j in range(n):
            s = _bell(s, L[j], R[j], N)
        s = _bell(s, P, M, N)
        s = _apply(s, _SWAP, [M, L[0]], N)
        s = _apply(s, U, L, N)
        s = _zz_coupling(s, g, L, R, N)
        s = _apply(s, U.conj(), R, N)
        rho = _reduced_density(s, [P, R[0]], N)
        return float(np.real(_BELL.conj() @ rho @ _BELL))

    F = np.array([run(float(g)) for g in g_grid])
    baseline = run(0.0)
    # sign asymmetry F(+g) - F(-g) over positive grid
    pos = g_grid > 1e-9
    asym = 0.0
    for g in g_grid[pos]:
        asym = max(asym, abs(run(float(g)) - run(float(-g))))
    peak = float(F.max())
    return {
        "g_grid": g_grid.tolist(),
        "fidelity": F.tolist(),
        "baseline_g0": float(baseline),
        "peak_fidelity": peak,
        "max_sign_asymmetry": float(asym),
        "classical_bound": 0.5,
        # signature requires peak clearly above classical AND a real sign-asymmetry
        "gjw_signature_present": bool(peak > 0.55 and asym > 0.1),
        "note": "generic Haar scrambler: no size-winding -> no signature; "
                "SYK dynamics required for the GJW traversable effect",
    }


# --- report ----------------------------------------------------------------


@dataclass(frozen=True)
class EREPRChannelReport:
    deterministic_fidelity_1q: float
    deterministic_fidelity_2q: float
    capacity_qubits_per_epr_pair: int
    requires_coupling_between_mouths: bool
    is_faster_than_light: bool
    gjw_signature_with_haar_scrambler: bool
    gjw_note: str


def build_channel_report() -> EREPRChannelReport:
    """Characterize the ER=EPR channel: what works, what doesn't, honestly."""
    f1 = channel_entanglement_fidelity(1)
    f2 = channel_entanglement_fidelity(2)
    scan = gjw_coupling_scan()
    return EREPRChannelReport(
        deterministic_fidelity_1q=float(f1),
        deterministic_fidelity_2q=float(f2),
        capacity_qubits_per_epr_pair=1,
        requires_coupling_between_mouths=True,
        is_faster_than_light=False,
        gjw_signature_with_haar_scrambler=bool(scan["gjw_signature_present"]),
        gjw_note=scan["note"],
    )


def summarise_channel(r: EREPRChannelReport) -> str:
    return (
        f"ER=EPR channel: ent.fidelity 1q={r.deterministic_fidelity_1q:.3f}, "
        f"2q={r.deterministic_fidelity_2q:.3f} (classical bound 0.5); "
        f"capacity={r.capacity_qubits_per_epr_pair} qubit/EPR pair; "
        f"FTL={r.is_faster_than_light}; "
        f"GJW-signature(Haar)={r.gjw_signature_with_haar_scrambler}"
    )
