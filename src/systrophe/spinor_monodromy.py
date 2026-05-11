"""Spinor monodromy along CTCs in the LP supercritical exterior.

A spinor parallel-transported along a closed phi-loop at constant t
and r picks up a spin holonomy U(r), an element of Spin(1,3). For the
LP background, this holonomy is computed from the spin connection
omega^{ab}_phi(r) integrated around phi in [0, 2pi]:

    U(r) = P exp[-1/4 * gamma_a gamma_b * integral omega^{ab}_phi dphi].

Since omega^{ab}_phi is phi-independent (axisymmetry), the integral
is just 2 pi * omega^{ab}_phi(r). The monodromy is then

    U(r) = exp[-pi/2 * omega^{ab}_phi(r) gamma_a gamma_b],

a complex 4x4 matrix with eigenvalues on the unit circle (Spin
elements are unitary in our flat-space gamma representation).

For r in a CTC band (F(r) < 0), the phi-loop is genuinely timelike-
oriented and the monodromy is the *temporal* holonomy: it represents
the rotation of a quantum spinor that traverses the CTC once.

Key questions this module answers:
- Is the monodromy nontrivial (i.e., U != identity)?
- Does it have a fixed-point spinor (eigenvalue +1)?
- What is the cumulative monodromy after N traversals?
- How does the pair extinction (delta = pi) modify the monodromy?

Functions
---------
- spin_connection_phi: omega^{ab}_phi at radius r
- spinor_holonomy: 4x4 SU(2,2) monodromy matrix
- monodromy_eigenvalues: spectral content
- fixed_point_spinors: spinors invariant under U
- multi_loop_monodromy: U^N for N traversals
- pair_modified_monodromy: U for Systrophe pair (delta != 0)
- chronology_horizon_caustic: blow-up of |omega^{ab}_phi| at F = 0
"""

from __future__ import annotations

import math

import numpy as np

from .vanstockum import VanStockumInterior


def _gamma_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chiral (Weyl) representation gamma matrices, signature (-,+,+,+)."""
    sigma1 = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma3 = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    Z2 = np.zeros((2, 2), dtype=complex)
    gamma0 = np.block([[Z2, I2], [I2, Z2]])
    gamma1 = np.block([[Z2, sigma1], [-sigma1, Z2]])
    gamma2 = np.block([[Z2, sigma2], [-sigma2, Z2]])
    gamma3 = np.block([[Z2, sigma3], [-sigma3, Z2]])
    return gamma0, gamma1, gamma2, gamma3


def spin_connection_phi(vs: VanStockumInterior, r: float) -> dict:
    """Compute the relevant spin-connection components omega^{ab}_phi at r.

    For the LP metric ds^2 = -F dt^2 + 2K dt dphi + L dphi^2 +
    h(dr^2 + dz^2), the orthonormal tetrad has e^0 = sqrt(F) dt -
    K/sqrt(F) dphi etc. The spin connection on a phi-loop picks up
    omega^{01}_phi, omega^{12}_phi, etc. We return finite-difference
    estimates of the dominant components.
    """
    eps = 1e-4 * max(abs(r), 1.0)
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    F_plus = float(vs.analytic_exterior_F(np.array([r + eps]))[0])
    F_minus = float(vs.analytic_exterior_F(np.array([r - eps]))[0])
    Fp = (F_plus - F_minus) / (2 * eps)

    K = float(vs.analytic_exterior_K(np.array([r]))[0])
    K_plus = float(vs.analytic_exterior_K(np.array([r + eps]))[0])
    K_minus = float(vs.analytic_exterior_K(np.array([r - eps]))[0])
    Kp = (K_plus - K_minus) / (2 * eps)

    # Dominant component (heuristic): omega^{0 1}_phi ~ K' / (2 sqrt|F|)
    abs_F = max(abs(F), 1e-30)
    w01 = float(Kp / (2.0 * math.sqrt(abs_F)))
    # omega^{1 2}_phi ~ F' / (2 sqrt|F|) on the (r, phi) plane
    w12 = float(Fp / (2.0 * math.sqrt(abs_F)))
    return {
        "r": r,
        "F": F,
        "K": K,
        "F_prime": Fp,
        "K_prime": Kp,
        "omega_01_phi": w01,
        "omega_12_phi": w12,
    }


def spinor_holonomy(vs: VanStockumInterior, r: float) -> np.ndarray:
    """4x4 Spin(1,3) holonomy U(r) for parallel transport around phi-loop.

    U(r) = exp[-pi/2 * (omega^{ab}_phi gamma_a gamma_b)].
    Here a, b in {0, 1} and {1, 2} since the metric is z-independent.
    """
    gamma0, gamma1, gamma2, gamma3 = _gamma_matrices()
    sc = spin_connection_phi(vs, r)
    w01 = sc["omega_01_phi"]
    w12 = sc["omega_12_phi"]
    sigma_01 = 0.5 * (gamma0 @ gamma1 - gamma1 @ gamma0)
    sigma_12 = 0.5 * (gamma1 @ gamma2 - gamma2 @ gamma1)
    exponent = -math.pi / 2 * (w01 * sigma_01 + w12 * sigma_12)
    from scipy.linalg import expm
    return expm(exponent)


def monodromy_eigenvalues(vs: VanStockumInterior, r: float) -> np.ndarray:
    """Eigenvalues of the 4x4 spinor holonomy U(r)."""
    U = spinor_holonomy(vs, r)
    return np.linalg.eigvals(U)


def fixed_point_spinors(vs: VanStockumInterior, r: float,
                        tol: float = 1e-6) -> list[np.ndarray]:
    """Return spinors psi with U psi = psi (i.e., eigenvectors at +1)."""
    U = spinor_holonomy(vs, r)
    evals, evecs = np.linalg.eig(U)
    fp = []
    for i, ev in enumerate(evals):
        if abs(ev - 1.0) < tol:
            fp.append(evecs[:, i])
    return fp


def multi_loop_monodromy(vs: VanStockumInterior, r: float,
                          n_loops: int) -> np.ndarray:
    """Cumulative monodromy after N traversals: U(r)^N."""
    if n_loops < 0:
        raise ValueError("n_loops must be non-negative")
    U = spinor_holonomy(vs, r)
    return np.linalg.matrix_power(U, n_loops)


def pair_modified_monodromy(vs: VanStockumInterior, r: float,
                             delta: float) -> np.ndarray:
    """Monodromy for a Systrophe pair at relative phase offset delta.

    For delta = 0 (aligned), recovers the single-cylinder U(r).
    For delta = pi (extinction), the angular momentum cancels and U
    collapses to the identity (no net spin rotation).
    """
    U = spinor_holonomy(vs, r)
    extinction_factor = 0.5 * (1.0 + math.cos(delta))
    # Interpolate between identity (delta=pi) and U (delta=0)
    return extinction_factor * U + (1.0 - extinction_factor) * np.eye(4, dtype=complex)


def chronology_horizon_caustic(vs: VanStockumInterior,
                                r_min: float = None,
                                r_max: float = None,
                                n_samples: int = 200) -> dict:
    """Locate where |omega^{ab}_phi| diverges (chronology horizons)."""
    if r_min is None:
        r_min = vs.R * 1.05
    if r_max is None:
        r_max = vs.R * 30.0
    r_grid = np.geomspace(r_min, r_max, n_samples)
    omegas = []
    for r in r_grid:
        sc = spin_connection_phi(vs, float(r))
        omegas.append(abs(sc["omega_01_phi"]) + abs(sc["omega_12_phi"]))
    omegas_arr = np.asarray(omegas)
    # Find peaks
    peaks = []
    for i in range(1, len(omegas_arr) - 1):
        if omegas_arr[i] > omegas_arr[i - 1] and omegas_arr[i] > omegas_arr[i + 1]:
            if omegas_arr[i] > 10 * np.median(omegas_arr):
                peaks.append(float(r_grid[i]))
    return {
        "r_grid": r_grid,
        "omega_magnitudes": omegas_arr,
        "caustic_radii": peaks,
        "max_omega_magnitude": float(omegas_arr.max()),
    }


def expected_monodromy_phase_per_revolution(vs: VanStockumInterior,
                                              r: float) -> float:
    """Expected phase exp(i * theta) on the unit circle per revolution.

    Returns theta in radians from eigenvalues.
    """
    evals = monodromy_eigenvalues(vs, r)
    # Average phase
    phases = np.angle(evals)
    return float(np.mean(np.abs(phases)))


def monodromy_period_in_revolutions(vs: VanStockumInterior, r: float) -> int:
    """Approximate N such that U(r)^N ~ identity.

    Returns 0 if eigenvalues are not rationally related to 2 pi
    within 100 revolutions.
    """
    theta = expected_monodromy_phase_per_revolution(vs, r)
    if abs(theta) < 1e-10:
        return 1
    for n in range(1, 101):
        if abs((n * theta) % (2 * math.pi)) < 0.05:
            return n
    return 0
