"""Systrophe (Συστροφή): co-rotating cylinder pair, van Stockum / Tipler exterior.

The Greek word systrophe means "twisting-together" or "coiled conjunction"
and describes the physics here: two co-rotating, dual-positive-mass dust
cylinders sharing an axis, whose joint exterior contains a log-periodic
"Tipler sinusoid" structure with a relative phase offset.
"""

from .vanstockum import VanStockumInterior, vanstockum_interior_metric
from .sinusoid import TiplerSinusoid, fit_log_periodic
from .pair import SystrophePair
from .array import SystropheArray
from .ctc import find_ctc_intervals, has_ctc
from .lewis_papapetrou import LPSolution, integrate_lp_exterior
from .geodesic import (
    CircularOrbit,
    integrate_geodesic,
    is_omega_timelike,
    omega_for_target_coord_time,
    timelike_omega_bounds,
)
from .time_machine import (
    TimeMachineWindow,
    find_single_cylinder_windows,
    find_time_machine_windows,
    harness_time_loop,
)
from .lp_robust import LPRobustSolution, integrate_lp_robust
from .off_axis import OffAxisPair
from .energy_conditions import (
    EnergyConditionReport,
    energy_condition_report,
    proper_energy_density,
    total_energy_per_unit_length,
)
from .photon_orbits import (
    integrate_null_geodesic,
    null_circular_omega,
    null_impact_parameters,
    vanstockum_photon_omega,
)
from .quantum_diagnostics import (
    cauchy_horizon_estimate,
    chronology_protection_indicator,
    conformal_anomaly_2d_proxy,
    hawking_temperature_at_horizon,
    ricci_scalar,
    surface_gravity_at_horizon,
    tolman_blueshift_factor,
)
from .photon_raytrace import (
    lensing_pattern,
    photon_deflection_angle,
    photon_perihelion,
)
from .dirac import (
    LewisPapapetrouTetrad,
    gamma_matrix,
    radial_dirac_system,
    solve_radial_dirac,
    vanstockum_dirac_system,
)

__all__ = [
    "VanStockumInterior",
    "vanstockum_interior_metric",
    "TiplerSinusoid",
    "fit_log_periodic",
    "SystrophePair",
    "SystropheArray",
    "find_ctc_intervals",
    "has_ctc",
    "LPSolution",
    "integrate_lp_exterior",
    "CircularOrbit",
    "integrate_geodesic",
    "is_omega_timelike",
    "omega_for_target_coord_time",
    "timelike_omega_bounds",
    "TimeMachineWindow",
    "find_single_cylinder_windows",
    "find_time_machine_windows",
    "harness_time_loop",
    "LPRobustSolution",
    "integrate_lp_robust",
    "OffAxisPair",
    "EnergyConditionReport",
    "energy_condition_report",
    "proper_energy_density",
    "total_energy_per_unit_length",
]
__version__ = "0.7.0"
