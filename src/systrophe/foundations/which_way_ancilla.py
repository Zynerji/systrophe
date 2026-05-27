"""Which-way ancilla: derive fringe visibility from path-detector entanglement.

A two-path (slit U, slit L) qubit is entangled with an ancilla qubit by a
controlled rotation parameterised by a coupling kappa in [0, 1]:

    |Psi> = (1/sqrt(2)) [ |U> (x) |0>
                       + |L> (x) (cos(theta/2) |0> + sin(theta/2) |1>) ]

with theta = pi * kappa. kappa = 0 leaves the ancilla unmarked
(perfect coherence, full fringes); kappa = 1 maps |L> -> |1>, so the
ancilla records which slit the particle went through and the fringes
vanish.

Tracing out the ancilla gives the reduced path density matrix
    rho_path = (1/2) [ I + cos(theta/2) sigma_x ],
whose off-diagonal coherence
    V = 2 |rho_path[0, 1]| = cos(theta/2)
is the fringe visibility. The optimal which-way distinguishability
    D = sqrt(1 - |<0|a_L>|^2) = sin(theta/2)
saturates Englert's wave-particle duality relation
    V^2 + D^2 = 1
when the ancilla states are pure.

This module is intentionally self-contained (numpy only). Feed `V`
into a two-path intensity:
    I(y) = 2 + 2 V cos(k * dL(y) + phi).

References
----------
- B.-G. Englert, Phys. Rev. Lett. 77, 2154 (1996).
- M. O. Scully, B.-G. Englert, H. Walther, Nature 351, 111 (1991).
"""

from __future__ import annotations

import math

import numpy as np


def _check_coupling(coupling: float) -> None:
    if not 0.0 <= coupling <= 1.0:
        raise ValueError(f"coupling must be in [0, 1], got {coupling}")


def entangle_path_ancilla(coupling: float) -> dict:
    """Build the joint path (x) ancilla state |Psi> for which-way coupling.

    Basis order (path, ancilla): |U,0>, |U,1>, |L,0>, |L,1>.

    Parameters
    ----------
    coupling : float in [0, 1]
        kappa = 0 leaves the ancilla in |0> regardless of path;
        kappa = 1 maps |L> -> |1>, a perfect which-way record.

    Returns
    -------
    dict with keys:
      - "psi": 4-vector (numpy) in the basis above.
      - "theta": rotation angle pi * kappa.
      - "ancilla_U", "ancilla_L": 2-vectors the ancilla picks up
        conditional on the path branch.
    """
    _check_coupling(coupling)
    theta = math.pi * coupling
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    psi = np.array([1.0, 0.0, c, s], dtype=float) / math.sqrt(2.0)
    return {
        "psi": psi,
        "theta": theta,
        "ancilla_U": np.array([1.0, 0.0], dtype=float),
        "ancilla_L": np.array([c, s], dtype=float),
    }


def reduced_path_density(coupling: float) -> np.ndarray:
    """rho_path = Tr_ancilla |Psi><Psi|. 2x2 Hermitian, trace 1."""
    psi = entangle_path_ancilla(coupling)["psi"]
    # psi[2*p + a]: rows = path index, cols = ancilla index.
    psi_mat = psi.reshape(2, 2)
    return psi_mat @ psi_mat.conj().T


def path_coherence(coupling: float) -> float:
    """Fringe visibility V = 2|rho_path[0,1]| = cos(theta/2)."""
    rho = reduced_path_density(coupling)
    return float(2.0 * abs(rho[0, 1]))


def which_way_distinguishability(coupling: float) -> float:
    """Optimal which-way distinguishability D = sqrt(1 - |<a_U|a_L>|^2)."""
    state = entangle_path_ancilla(coupling)
    overlap = float(np.vdot(state["ancilla_U"], state["ancilla_L"]).real)
    return float(math.sqrt(max(0.0, 1.0 - overlap * overlap)))


def englert_relation(coupling: float) -> dict:
    """Englert wave-particle inequality V^2 + D^2 <= 1.

    Returns a dict reporting V, D, V^2+D^2, and the slack
    (= 1 - V^2 - D^2; zero for pure-ancilla which-way markers).
    """
    V = path_coherence(coupling)
    D = which_way_distinguishability(coupling)
    return {
        "coupling": float(coupling),
        "V": V,
        "D": D,
        "V2_plus_D2": V * V + D * D,
        "slack": 1.0 - V * V - D * D,
    }


def fringe_intensity_with_ancilla(
    y: np.ndarray, k: float, d: float, D_arm: float, phi: float,
    coupling: float,
) -> np.ndarray:
    """Two-slit intensity with V derived from a which-way ancilla.

    Identical to the plain double-slit formula except the visibility
    factor is `path_coherence(coupling)` instead of a free parameter.

    Parameters
    ----------
    y : array
        Detector positions on the screen.
    k : float
        Effective wavenumber.
    d : float
        Slit half-separation.
    D_arm : float
        Slits-to-screen distance.
    phi : float
        Static phase shift (e.g. an Aharonov-Bohm phase).
    coupling : float
        Which-way coupling in [0, 1].
    """
    V = path_coherence(coupling)
    L_U = np.sqrt(D_arm * D_arm + d * d) + np.sqrt(D_arm * D_arm + (y - d) ** 2)
    L_L = np.sqrt(D_arm * D_arm + d * d) + np.sqrt(D_arm * D_arm + (y + d) ** 2)
    return 2.0 + 2.0 * V * np.cos(k * (L_U - L_L) + phi)


def sweep_coupling(n: int = 21) -> dict:
    """V, D, V^2+D^2 across coupling in [0, 1] for plotting / regression."""
    kappas = np.linspace(0.0, 1.0, n)
    Vs = np.array([path_coherence(float(k)) for k in kappas])
    Ds = np.array([which_way_distinguishability(float(k)) for k in kappas])
    return {
        "coupling": kappas,
        "V": Vs,
        "D": Ds,
        "V2_plus_D2": Vs * Vs + Ds * Ds,
    }
