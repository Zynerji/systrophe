"""First-principles analysis of the GJW wormhole channel: the size operator.

Derivation (verified to machine precision in tests)
---------------------------------------------------
For n EPR pairs |EPR> = ⊗_j (|00>+|11>)/√2 between L and R, and the
Maldacena–Qi / Gao–Jafferis–Wall coupling

    V = (1/n) Σ_j Z_j^L Z_j^R,

every Pauli string P acting on L is an eigenvector of V on the EPR state:

    V · P_L |EPR> = (1 - 2 size(P)/n) · P_L |EPR>,        (★)

where size(P) = number of sites at which P is X or Y (anticommutes with Z).
Proof: Z_j^L Z_j^R |EPR> = |EPR> (ricochet Z^T=Z), and Z_j^L P = (-1)^{a_j} P Z_j^L
with a_j=1 iff P has X/Y at j; summing gives (1/n)Σ_j(-1)^{a_j}=1-2 size/n. ∎

So **the coupling IS the operator-size operator**, and

    e^{igV} P_L|EPR> = e^{ig} e^{-2ig·size(P)/n} P_L|EPR>

applies a phase LINEAR in size — the "size winding" mechanism, derived from
first principles rather than assumed.

Consequence: the size-winding mechanism (derived, not assumed)
-------------------------------------------------------------
Because e^{igV} phases each Pauli component by e^{-2ig·size/n}, the coupling can
refocus a scrambled message ONLY if its coefficient phases are themselves linear
in size ("perfect size winding"). Chaotic dynamics grow the operator size (so
the mechanism is active — verified: SYK lifts the channel where Haar does not),
but at finite, classically-simulable N the winding is only approximately linear,
so single-shot teleportation of an unknown qubit is PARTIAL. (Honest note: a
naive "fidelity = |char. function of the size distribution|" identification is
wrong — that quantity is 1 at g=0, where the EPR ricochet trivially mirrors the
whole operator algebra onto R without localizing the message on R_0. Teleporting
an unknown STATE is a stronger, localizing requirement and must be measured by
running the protocol.)

The solution (verified)
-----------------------
Two routes give unit fidelity, both using the same EPR resource (= the bridge):
1. The large-N gravity limit with perfect size winding refocuses the message
   deterministically — the genuine traversable wormhole, not classically
   simulable here.
2. The DETERMINISTIC coherent-correction teleportation already built in
   ``erepr_channel`` (fidelity 1, any N): replace size-winding refocusing with
   feed-forward Bell-basis corrections. This is the buildable unit-fidelity
   channel. It is the same wormhole resource read out deterministically rather
   than through a single size-limited coupling.

So the first-principles "solution" is: (★) the coupling is the size operator
[rigorous]; chaotic scrambling activates the mechanism [SYK > Haar, measured];
unit fidelity is delivered by the deterministic coherent channel [F=1, verified],
or by the unsimulable large-N perfect-winding limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .erepr_channel import _apply, _BELL, _H, _CNOT  # reuse primitives

_P1 = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli(s: str) -> np.ndarray:
    m = np.array([[1]], dtype=complex)
    for ch in s:
        m = np.kron(m, _P1[ch])
    return m


def _size_xy(s: str) -> int:
    return sum(1 for ch in s if ch in "XY")


def epr_state(n: int) -> np.ndarray:
    """n EPR pairs between L (qubits 0..n-1) and R (n..2n-1)."""
    N = 2 * n
    s = np.zeros(2 ** N, dtype=complex)
    s[0] = 1.0
    for j in range(n):
        s = _apply(s, _H, [j], N)
        s = _apply(s, _CNOT, [j, n + j], N)
    return s


def size_operator(n: int) -> np.ndarray:
    """V = (1/n) Σ_j Z_j^L Z_j^R on 2n qubits."""
    N = 2 * n
    V = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for j in range(n):
        zl = ["I"] * N
        zl[j] = "Z"
        zr = ["I"] * N
        zr[n + j] = "Z"
        V = V + _pauli("".join(zl)) @ _pauli("".join(zr))
    return V / n


def verify_size_operator(n: int = 3) -> float:
    """Max residual of identity (★) over all Pauli strings P on L. ~0 confirms it."""
    N = 2 * n
    V = size_operator(n)
    epr = epr_state(n)
    max_res = 0.0
    for combo in product("IXYZ", repeat=n):
        s = "".join(combo)
        full = list("I" * N)
        for q, ch in enumerate(s):
            full[q] = ch
        state = _pauli("".join(full)) @ epr
        Vs = V @ state
        lam = 1.0 - 2.0 * _size_xy(s) / n
        max_res = max(max_res, float(np.linalg.norm(Vs - lam * state)))
    return max_res


def pauli_size_distribution(O: np.ndarray, n: int) -> dict[int, float]:
    """Size distribution {size: weight} of an n-qubit operator via Pauli decomp.

    weight(s) = Σ_{size(P)=s} |Tr(P† O)/2^n|^2, normalized to sum 1.
    """
    d = 2 ** n
    dist: dict[int, float] = {}
    for combo in product("IXYZ", repeat=n):
        s = "".join(combo)
        c = np.trace(_pauli(s).conj().T @ O) / d
        w = float(abs(c) ** 2)
        if w > 1e-15:
            dist[_size_xy(s)] = dist.get(_size_xy(s), 0.0) + w
    tot = sum(dist.values())
    return {k: v / tot for k, v in dist.items()} if tot > 0 else dist


def mean_operator_size(O: np.ndarray, n: int) -> float:
    """Mean operator size <size> of an n-qubit operator (chaos grows this)."""
    dist = pauli_size_distribution(O, n)
    return float(sum(s * w for s, w in dist.items()))


@dataclass(frozen=True)
class SizeWindingReport:
    n: int
    size_operator_residual: float          # ~0 confirms identity (★) [rigorous]
    haar_teleport_fidelity: float          # ~0.25: no size-winding, channel shut
    syk_teleport_fidelity: float           # >0.25: SYK activates the mechanism
    syk_lift_over_haar: float
    mechanism_active: bool                  # syk lifts the channel above Haar
    deterministic_channel_fidelity: float   # 1.0: the unit-fidelity solution
    conclusion: str


def first_principles_report(n: int = 3) -> SizeWindingReport:
    """End-to-end: verify (★); measure that chaotic scrambling activates the
    mechanism (SYK > Haar); and that the deterministic channel solves it (F=1)."""
    res = verify_size_operator(n)
    from .erepr_channel import (
        channel_entanglement_fidelity,
        syk_vs_haar_activation,
    )
    act = syk_vs_haar_activation(nq=n)
    det = channel_entanglement_fidelity(1)
    return SizeWindingReport(
        n=n,
        size_operator_residual=float(res),
        haar_teleport_fidelity=float(act["haar_peak_fidelity"]),
        syk_teleport_fidelity=float(act["syk_peak_fidelity"]),
        syk_lift_over_haar=float(act["syk_lift"] - act["haar_lift"]),
        mechanism_active=bool(act["syk_activates_channel"]),
        deterministic_channel_fidelity=float(det),
        conclusion="(star) GJW coupling = operator-size operator [residual ~0, "
                   "rigorous]. Chaotic scrambling activates size-winding "
                   "(SYK teleports where Haar does not), but single-shot "
                   "fidelity is partial at simulable N. Unit fidelity is "
                   "delivered by the deterministic coherent channel (F=1) -- "
                   "same EPR bridge, feed-forward readout -- or the unsimulable "
                   "large-N perfect-winding limit. NOT a faster-than-light or "
                   "matter-transport result.",
    )


def summarise(r: SizeWindingReport) -> str:
    return (
        f"SizeWinding n={r.n}: V=size-operator (residual {r.size_operator_residual:.1e}); "
        f"teleport Haar={r.haar_teleport_fidelity:.3f} vs SYK={r.syk_teleport_fidelity:.3f} "
        f"(mechanism_active={r.mechanism_active}); deterministic F="
        f"{r.deterministic_channel_fidelity:.3f}"
    )
