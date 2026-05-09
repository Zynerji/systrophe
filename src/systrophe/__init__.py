"""Systrophe (Συστροφή): co-rotating cylinder pair, van Stockum / Tipler exterior.

The Greek word systrophe means "twisting-together" or "coiled conjunction"
and describes the physics here: two co-rotating, dual-positive-mass dust
cylinders sharing an axis, whose joint exterior contains a log-periodic
"Tipler sinusoid" structure with a relative phase offset.
"""

from .vanstockum import VanStockumInterior, vanstockum_interior_metric
from .sinusoid import TiplerSinusoid, fit_log_periodic
from .pair import SystrophePair
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

__all__ = [
    "VanStockumInterior",
    "vanstockum_interior_metric",
    "TiplerSinusoid",
    "fit_log_periodic",
    "SystrophePair",
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
__version__ = "0.3.1"
