"""Steinhauer 2019 benchmark.

Steinhauer (Nature Physics 2019) measured T_H = 0.124 ± 0.012 nK in
his BEC analog black hole. This module compares an LP-based analog
prediction to that measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from systrophe.acoustic_hawking_spectrum import (
    bec_experimental_predictions,
    compare_to_steinhauer_2019,
)


@dataclass(frozen=True)
class SteinhauerBenchmark:
    """Comparison against Steinhauer 2019 measurement.

    Attributes
    ----------
    T_predicted_nK : float
        Predicted Hawking temperature, nK.
    T_steinhauer_nK : float
        Measured value: 0.124 nK.
    T_steinhauer_uncertainty_nK : float
        Measurement uncertainty: 0.012 nK.
    sigma_deviation : float
        |T_predicted - T_steinhauer| / T_uncertainty.
    consistent_with_measurement : bool
        True iff sigma_deviation < 3 (3-sigma agreement).
    bec_predictions : dict
        Full set of intermediate BEC-vortex predictions (healing
        length, sound speed, etc.).
    """
    T_predicted_nK: float
    T_steinhauer_nK: float
    T_steinhauer_uncertainty_nK: float
    sigma_deviation: float
    consistent_with_measurement: bool
    bec_predictions: dict


def benchmark_against_steinhauer_2019(
    omega: float, R: float, n_density: float, atom_mass: float,
    a_scattering: float = 5.3e-9,
) -> SteinhauerBenchmark:
    """Compute the BEC-analog Hawking T and compare to Steinhauer 2019.

    Parameters
    ----------
    omega : float
        BEC vortex angular velocity, 1/s.
    R : float
        BEC vortex core radius, m.
    n_density : float
        BEC density, m^-3.
    atom_mass : float
        Atomic mass, kg (Rb-87 ≈ 1.44e-25 kg).
    a_scattering : float, default 5.3e-9 m (Rb-87 s-wave length).

    Returns
    -------
    SteinhauerBenchmark
    """
    pred = bec_experimental_predictions(
        omega=float(omega), R=float(R), n_density=float(n_density),
        atom_mass=float(atom_mass), a_scattering=float(a_scattering),
    )
    T_predicted_nK = float(pred["T_H_acoustic_nK"])
    cmp_ = compare_to_steinhauer_2019(T_predicted_nK)
    return SteinhauerBenchmark(
        T_predicted_nK=T_predicted_nK,
        T_steinhauer_nK=float(cmp_["T_steinhauer_nK"]),
        T_steinhauer_uncertainty_nK=float(cmp_["T_steinhauer_uncertainty_nK"]),
        sigma_deviation=float(cmp_["sigma_deviation"]),
        consistent_with_measurement=bool(cmp_["consistent_with_measurement"]),
        bec_predictions=dict(pred),
    )
