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
from .dirac_spectrum import (
    boundary_functional,
    find_bound_states,
    vanstockum_bound_states,
)
from .dirac_sea import (
    chronology_horizon_pressure_divergence_rate,
    density_of_states_radial,
    dirac_sea_pressure_proxy,
    local_energy,
)
from .particle_creation import (
    bose_einstein,
    fermi_dirac,
    particle_creation_spectrum_at_horizon,
    total_emission_power_proxy,
)
from .qftcs_backreaction import (
    conformal_anomaly_trace,
    radial_temporal_ricci_scalar,
    stress_energy_at_horizon,
    vacuum_energy_density_proxy,
)
from .point_splitting import (
    dewitt_a2_coefficient,
    effective_action_volume_density,
    kretschmann_scalar,
    phi_squared_largemass_expansion,
    ricci_scalar as ricci_scalar_4d,
    riemann_tensor,
    trace_anomaly_4d_exact,
    vacuum_residual,
)
from .floquet import (
    AdiabaticFloquetSpectrum,
    adiabatic_floquet_spectrum,
    adiabatic_floquet_validity,
    nonadiabatic_floquet_correction,
    static_pair_bound_states,
)
from .casimir import (
    hurwitz_zeta_neg3,
    standard_casimir_energy_density,
    standard_casimir_force,
    topological_casimir_coefficient,
    topological_casimir_derivative,
    z3_cover_fundamental_eigenvalue,
    z3_cover_mode_density,
    z3_cover_regularised_zeta_sum,
)
from .tipler_fractal import (
    CascadeDSI,
    base_tipler_sinusoid_is_dsi,
    box_count_log_dimension,
    cascade_box_dimension,
    ctc_band_log_widths,
    dsi_extension_from_sinusoid,
    dsi_rescaling_factor,
    verify_geometric_progression,
    zero_geometric_ratio,
)
from .anomaly_inflow import (
    AXIAL_ANOMALY_COEFFICIENT,
    axial_anomaly_density,
    callan_harvey_bulk_inflow,
    callan_harvey_consistency,
    chern_simons_5form_coefficient,
    dirac_eta_invariant,
    index_density_2form,
    z3_anomaly_inflow_balance,
    z3_branch_etas,
    z3_branch_twists,
    z3_total_eta,
)
from .horned_torus import (
    HornedTorus,
    compare_horn_modes,
    horn_circumference_at_z,
    inverted_horn_profile,
    regular_horn_profile,
)
from .hadamard_offtrace import (
    energy_density_in_static_frame,
    hadamard_offtrace_T,
    hadamard_T_diagonal_components,
    hadamard_T_trace,
    hadamard_T_traceless_part,
    riemann_squared_tensor,
    trace_decomposition,
)
from .newton_kantorovich import (
    NKResult,
    is_convergence_rate_quadratic,
    newton_kantorovich_1d,
    newton_kantorovich_nd,
    picard_iteration_1d,
)
from .floquet_mobius import (
    FloquetMobiusResult,
    analyze_floquet_mobius,
    brillouin_zone_wrap,
    floquet_propagator,
    joint_floquet_spectrum,
    joint_static_hamiltonian,
    static_limit_check,
    z3_cycle_shift,
    z3_hopping_matrix,
    z3_symmetry_check,
)
from .acoustic_metric import (
    acoustic_hawking_temperature,
    acoustic_horizon_radius,
    acoustic_line_element_at_radius,
    acoustic_metric_components,
    acoustic_surface_gravity,
    compare_acoustic_vs_gravitational_T_H,
    ctc_region_is_supersonic,
)
from .casimir_throat import (
    brown_maclay_T_minkowski,
    brown_maclay_at_lp_point,
    brown_maclay_energy_density,
    brown_maclay_normal_pressure,
    brown_maclay_trace,
    compare_throat_to_brown_maclay,
    topological_throat_coefficient,
    transverse_pressure,
)
from .back_reaction import (
    BackReactionLandscape,
    back_reaction_landscape,
    chronology_favoring_deltas,
    compare_to_ctc_extinction,
    optimal_delta_by_NK,
    pair_back_reaction_residual,
    pair_back_reaction_residual_at_r,
)
from .floquet_engineering import (
    StabilityMapResult,
    ctc_stability_gap,
    floquet_engineering_map,
    identify_floquet_resonances,
    stabilisation_efficacy,
)
from .dsi_observables import (
    LogPeriodicFit,
    box_count_dimension_1d,
    discrete_scale_invariance_test,
    fit_log_periodic_precursor,
    log_periodic_model,
    lomb_scargle_log_periodicity,
)
from .adm_export import (
    ADMSlice,
    adm_decompose_lp,
    adm_summary,
    export_to_einsteintoolkit_ascii,
    hamiltonian_constraint_residual,
)
from .d_ctc import (
    apply_channel,
    channel_superoperator,
    clifford_like_unitary,
    dctc_fixed_point,
    density_matrix_diagnostics,
    maximally_mixed_state,
    predict_convergence_via_spectrum,
    verify_fixed_point,
    z3_ctc_unitary,
)
from .wormhole_throat import (
    WormholeThroatReport,
    build_wormhole_report,
    effective_redshift_function,
    effective_shape_function,
    find_candidate_throats,
    flaring_out_check,
    z3_cover_throat_interpretation,
)
from .exotic_matter_accounting import (
    ExoticMatterReport,
    casimir_cavity_energy,
    exotic_vs_casimir_comparison,
    is_casimir_route_physically_plausible,
    morris_thorne_exotic_budget,
    morris_thorne_exotic_density,
    required_plate_separation,
)
from .chronology_protection import (
    ChronologyProtectionStudy,
    chronology_protection_study,
    chronology_protection_verdict,
    matched_pair_default_study,
)
from .wilson_loop import (
    field_strength_is_zero,
    flat_u1_connection,
    gauge_field_strength_4d,
    integrated_chern_number,
    verify_wilson_loop_matches_z3,
    wilson_loop,
    wilson_loop_sum_over_branches,
    z3_branch_holonomy,
)
from .dynamical_casimir import (
    DCEFluxResult,
    cavity_mode_frequency,
    dce_flux_estimate,
    dce_spectrum,
    linewidth_at_resonance,
    off_resonance_photon_number,
    on_resonance_photon_number,
    resonance_frequency_for_mode,
)
from .vacuum_states import (
    VacuumStateReport,
    adiabatic_well_defined,
    boulware_energy_density_proxy,
    boulware_well_defined,
    compare_vacua_at_radius,
    hartle_hawking_analog_exists,
    natural_vacuum_verdict,
    vacuum_selection_summary,
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
    "AdiabaticFloquetSpectrum",
    "adiabatic_floquet_spectrum",
    "adiabatic_floquet_validity",
    "nonadiabatic_floquet_correction",
    "static_pair_bound_states",
    "hurwitz_zeta_neg3",
    "standard_casimir_energy_density",
    "standard_casimir_force",
    "topological_casimir_coefficient",
    "topological_casimir_derivative",
    "z3_cover_fundamental_eigenvalue",
    "z3_cover_mode_density",
    "z3_cover_regularised_zeta_sum",
]
__version__ = "0.17.0"
