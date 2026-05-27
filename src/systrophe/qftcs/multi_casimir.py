"""Multi-cylinder Casimir interference.

Combines `casimir` (single-cylinder Z_3 cover topological Casimir) and
`multi_cylinder_dynamics` (N-cylinder array dynamics) to compute the
Casimir vacuum energy of an N-cylinder configuration.

Question: do the phase-extinction patterns (Phase 5) also extinguish
the Casimir energy?

The N-cylinder Casimir energy from the topological Z_3 mode sum
is per-cylinder:

    E_C^(i) = (1/720) Sum_{b=0,1,2} zeta_H(-3, b/3 + gamma_i / (2 pi))

For an N-cylinder phased array, the combined vacuum energy depends on
the phase pattern via interference of the individual mode sums.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systrophe.qftcs.casimir import topological_casimir_coefficient


@dataclass(frozen=True)
class MultiCylinderCasimir:
    """Casimir energy decomposition for an N-cylinder configuration."""

    N: int
    gamma_phases: np.ndarray  # gamma_eff per cylinder
    individual_energies: np.ndarray  # C(gamma_i) per cylinder
    total_energy: float
    is_extinguished: bool  # True iff total < tol (phase-comb extinction)


def n_cylinder_casimir_energy(gamma_phases: np.ndarray) -> MultiCylinderCasimir:
    """Compute Casimir energy of N-cylinder array with given phases.

    Each cylinder contributes its topological-Casimir coefficient
    independently (no cross-cylinder cross-terms at leading order).

    The total is just the sum.
    """
    gammas = np.asarray(gamma_phases, dtype=float)
    individual = np.array([float(topological_casimir_coefficient(g))
                              for g in gammas])
    total = float(np.sum(individual))
    extinguished = abs(total) < 1e-12
    return MultiCylinderCasimir(
        N=len(gammas), gamma_phases=gammas,
        individual_energies=individual,
        total_energy=total,
        is_extinguished=extinguished,
    )


def uniform_phase_comb_casimir(N: int) -> MultiCylinderCasimir:
    """N-cylinder array with gamma_i = 2 pi i / N.

    The phasor sum is zero, but the Casimir energy uses the
    topological coefficient C(gamma) which is NOT linear in
    exp(i gamma). So uniform-phase doesn't necessarily extinguish
    the Casimir energy.
    """
    gammas = np.array([2 * np.pi * i / N for i in range(N)])
    return n_cylinder_casimir_energy(gammas)


def all_aligned_casimir(N: int, gamma: float = 0.0) -> MultiCylinderCasimir:
    """All N cylinders at the same gamma_eff = constant."""
    gammas = np.full(N, gamma)
    return n_cylinder_casimir_energy(gammas)


def random_phase_casimir(N: int, seed: int = 0) -> MultiCylinderCasimir:
    """N-cylinder array with random gamma_i."""
    rng = np.random.default_rng(seed)
    gammas = rng.uniform(0, 2 * np.pi, N)
    return n_cylinder_casimir_energy(gammas)


def casimir_phase_extinction_scan(
    N: int, n_trials: int = 50,
    base_gamma_range: tuple[float, float] = (0.0, 2 * np.pi),
) -> dict:
    """Scan random N-cylinder configurations and report Casimir statistics.

    Compares to:
      - uniform-phase comb (delta_i = 2 pi i / N): does this extinguish?
      - all-aligned: maximum constructive
    """
    rng = np.random.default_rng(99)
    energies = []
    for _ in range(n_trials):
        gammas = rng.uniform(*base_gamma_range, N)
        c = n_cylinder_casimir_energy(gammas)
        energies.append(c.total_energy)
    energies = np.array(energies)
    # Uniform comb
    comb = uniform_phase_comb_casimir(N)
    # All-aligned at zero
    aligned = all_aligned_casimir(N, gamma=0.0)
    # All-aligned at pi (anti-phase)
    aligned_pi = all_aligned_casimir(N, gamma=np.pi)
    return {
        "N": N, "n_trials": n_trials,
        "random_mean_energy": float(np.mean(energies)),
        "random_std_energy": float(np.std(energies)),
        "random_max_energy": float(np.max(np.abs(energies))),
        "uniform_comb_energy": comb.total_energy,
        "uniform_comb_extinguished": comb.is_extinguished,
        "all_aligned_at_0_energy": aligned.total_energy,
        "all_aligned_at_pi_energy": aligned_pi.total_energy,
    }


def interference_pattern(
    gamma_1: float,
    gamma_2_range: tuple[float, float] = (0.0, 2 * np.pi),
    n_points: int = 100,
) -> dict:
    """Two-cylinder Casimir energy as gamma_2 varies, with gamma_1 fixed.

    Shows the interference structure of the topological Casimir
    coefficient.
    """
    gammas_2 = np.linspace(*gamma_2_range, n_points)
    energies = []
    for g2 in gammas_2:
        c = n_cylinder_casimir_energy(np.array([gamma_1, g2]))
        energies.append(c.total_energy)
    energies = np.array(energies)
    return {
        "gamma_1": gamma_1,
        "gamma_2_grid": gammas_2.tolist(),
        "total_energies": energies.tolist(),
        "min_energy": float(np.min(energies)),
        "max_energy": float(np.max(energies)),
        "interference_amplitude": float(np.max(energies) - np.min(energies)),
    }
