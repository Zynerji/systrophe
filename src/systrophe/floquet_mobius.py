"""Joint Floquet spectrum over time-circle and Z_3 branch index.

The adiabatic Floquet module computes quasi-energies on a single
time-circle. The Mobius cover introduces a *second* discrete symmetry:
the branch index b in {0, 1, 2} with twisted Z_3 boundary condition
psi(b + 3) = psi(b).

This module builds the *joint* monodromy on (t in [0, T]) x (b in Z_3)
and extracts:

- branch-resolved instantaneous energies (3-level system),
- Floquet propagator U(T) under a time-periodic drive that couples
  adjacent branches via Z_3 hopping,
- quasi-energy spectrum eps_F wrapped to the joint Brillouin zone,
- diagnostic of Floquet engineering at resonance Omega = e_1 - e_0.

Discrete-time model
-------------------
At fixed instant t, the three branches have energies (e_0, e_1, e_2)
which come from the static radial Dirac problem on the Z_3 cover at
the chosen twist parameters alpha_b = b/3 + gamma_eff / (2 pi). The
joint static Hamiltonian on the 3-dimensional branch space is

    H_static = diag(e_0, e_1, e_2) + g * [|0><1| + |1><2| + |2><0| + h.c.]

where g is a hopping strength representing branch-coupling (e.g.
particle exchange across the Mobius cover). A time-periodic drive of
strength amp and frequency Omega couples (or shifts) branches:

    V(t) = amp * sin(Omega * t) * cycle_shift_operator

The joint Floquet propagator is computed by numerical exponentiation
along the time-circle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


def z3_hopping_matrix(g: float = 1.0) -> np.ndarray:
    """Z_3 cyclic-hopping matrix on the branch space.

    H_hop = g * (|0><1| + |1><2| + |2><0|  +  h.c.)
    """
    H = np.zeros((3, 3), dtype=complex)
    for b in range(3):
        H[b, (b + 1) % 3] = g
        H[(b + 1) % 3, b] = g
    return H


def z3_cycle_shift() -> np.ndarray:
    """Z_3 cycle-shift operator: |b> -> |b+1 mod 3>."""
    S = np.zeros((3, 3), dtype=complex)
    for b in range(3):
        S[(b + 1) % 3, b] = 1.0
    return S


def joint_static_hamiltonian(
    branch_energies: np.ndarray, hopping: float = 0.0
) -> np.ndarray:
    """Static 3x3 Hamiltonian on the Z_3 branch space.

    Diagonal: branch energies (e_0, e_1, e_2). Off-diagonal: cyclic
    Z_3 hopping with strength `hopping`.
    """
    branch_energies = np.asarray(branch_energies, dtype=complex)
    if branch_energies.shape != (3,):
        raise ValueError("branch_energies must have shape (3,)")
    H = np.diag(branch_energies)
    if hopping != 0:
        H = H + z3_hopping_matrix(hopping)
    return H


def floquet_propagator(
    H_static: np.ndarray,
    drive_amp: float,
    omega_drive: float,
    n_steps: int = 200,
) -> np.ndarray:
    """Floquet monodromy U(T) for H(t) = H_static + drive_amp * sin(omega t) * S.

    `S` is the Z_3 cycle-shift operator. The propagator is computed by
    Trotter-Suzuki second-order integration along t in [0, T] with T =
    2 pi / omega.
    """
    if omega_drive <= 0:
        raise ValueError("omega_drive must be positive")
    T = 2 * np.pi / omega_drive
    dt = T / n_steps
    S = z3_cycle_shift()
    U = np.eye(3, dtype=complex)
    for k in range(n_steps):
        t_mid = (k + 0.5) * dt
        H_t = H_static + drive_amp * np.sin(omega_drive * t_mid) * (S + S.conj().T)
        U = expm(-1j * H_t * dt) @ U
    return U


def joint_floquet_spectrum(
    H_static: np.ndarray,
    drive_amp: float,
    omega_drive: float,
    n_steps: int = 200,
) -> np.ndarray:
    """Quasi-energy spectrum of the Floquet propagator.

    Returns three real numbers in (-pi/T, pi/T] (the Floquet Brillouin
    zone). Computed as -arg(eigenvalue) / T for each Floquet eigenvalue.
    """
    U = floquet_propagator(H_static, drive_amp, omega_drive, n_steps)
    eig_U = np.linalg.eigvals(U)
    T = 2 * np.pi / omega_drive
    quasi = -np.angle(eig_U) / T
    return np.sort(quasi.real)


def brillouin_zone_wrap(eps: np.ndarray, omega_drive: float) -> np.ndarray:
    """Wrap a quasi-energy array into the Floquet Brillouin zone."""
    half = omega_drive / 2.0
    return np.mod(eps + half, omega_drive) - half


@dataclass(frozen=True)
class FloquetMobiusResult:
    """Diagnostics for a joint Floquet-Mobius spectrum."""

    branch_energies: np.ndarray
    hopping: float
    drive_amp: float
    omega_drive: float
    quasi_energies: np.ndarray
    propagator: np.ndarray


def analyze_floquet_mobius(
    branch_energies: np.ndarray,
    hopping: float,
    drive_amp: float,
    omega_drive: float,
    n_steps: int = 200,
) -> FloquetMobiusResult:
    """Full analysis: build static H, compute propagator, extract spectrum."""
    H_static = joint_static_hamiltonian(branch_energies, hopping=hopping)
    U = floquet_propagator(H_static, drive_amp, omega_drive, n_steps=n_steps)
    T = 2 * np.pi / omega_drive
    eig_U = np.linalg.eigvals(U)
    quasi = brillouin_zone_wrap(-np.angle(eig_U).real / T, omega_drive)
    quasi = np.sort(quasi)
    return FloquetMobiusResult(
        branch_energies=np.asarray(branch_energies),
        hopping=hopping,
        drive_amp=drive_amp,
        omega_drive=omega_drive,
        quasi_energies=quasi,
        propagator=U,
    )


def static_limit_check(
    branch_energies: np.ndarray, omega_drive: float, n_steps: int = 200
) -> dict:
    """Static limit (drive_amp = 0): quasi-energies = (e_b mod omega).

    Returns dict with:
      - expected: branch_energies wrapped to BZ
      - obtained: numerical quasi-energies
      - max_err : max absolute difference
    """
    result = analyze_floquet_mobius(
        branch_energies, hopping=0.0, drive_amp=0.0,
        omega_drive=omega_drive, n_steps=n_steps,
    )
    expected = np.sort(brillouin_zone_wrap(np.asarray(branch_energies, dtype=float),
                                            omega_drive))
    obtained = np.sort(result.quasi_energies)
    return {
        "expected": expected,
        "obtained": obtained,
        "max_err": float(np.max(np.abs(expected - obtained))),
    }


def z3_symmetry_check(
    branch_energies: np.ndarray,
    hopping: float,
    drive_amp: float,
    omega_drive: float,
    n_steps: int = 200,
) -> dict:
    """Z_3 cyclic-permutation symmetry of the spectrum.

    When the branch energies are cyclically permuted, the quasi-energy
    spectrum (as a set) must be invariant.
    """
    spec_0 = analyze_floquet_mobius(
        branch_energies, hopping, drive_amp, omega_drive, n_steps
    ).quasi_energies
    rolled = np.roll(branch_energies, 1)
    spec_1 = analyze_floquet_mobius(
        rolled, hopping, drive_amp, omega_drive, n_steps
    ).quasi_energies
    return {
        "spec_orig": spec_0,
        "spec_rolled": spec_1,
        "max_set_diff": float(np.max(np.abs(np.sort(spec_0) - np.sort(spec_1)))),
    }
