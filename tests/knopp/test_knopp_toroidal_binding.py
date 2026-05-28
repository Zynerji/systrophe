"""Tests for the binding mechanism catalogue."""

import math

import pytest

from systrophe.knopp.knopp_toroidal_binding import (
    BindingVerdict,
    CATALOGUE,
    DarkPhotonMechanism,
    HardCoreMechanism,
    LQGAreaMechanism,
    ScalarTensorMechanism,
    YukawaMechanism,
    brans_dicke_emission_factor,
    dark_photon_required_charge,
    gw_inspiral_force,
    gw_inspiral_power,
    hard_core_required_epsilon,
    lqg_area_gap_fractional,
    summarise_binding_survey,
    survey_binding_mechanisms,
    yukawa_required_coupling,
)


# ----- GW inspiral baseline ---------------------------------------------


def test_gw_power_peters():
    assert gw_inspiral_power(M=1.0, d=2.0) == pytest.approx(
        64.0 / 5.0 / 32.0, rel=1e-12,
    )


def test_gw_force_positive():
    assert gw_inspiral_force(M=1.0, d=2.0) > 0.0


def test_gw_power_scales_inverse_d_fifth():
    p1 = gw_inspiral_power(M=1.0, d=2.0)
    p10 = gw_inspiral_power(M=1.0, d=20.0)
    assert p1 / p10 == pytest.approx(10.0 ** 5, rel=1e-12)


# ----- Yukawa ------------------------------------------------------------


def test_yukawa_mechanism_does_not_rescue():
    assert YukawaMechanism().rescues_inspiral is False


def test_yukawa_required_coupling_finite_for_massless():
    alpha = yukawa_required_coupling(M=1.0, d=2.0, mu=0.0)
    assert math.isfinite(alpha)
    assert alpha > 0.0


def test_yukawa_required_coupling_grows_with_mass():
    """Heavier Yukawa boson -> stronger suppression -> need bigger coupling."""
    alpha_light = yukawa_required_coupling(M=1.0, d=2.0, mu=0.0)
    alpha_heavy = yukawa_required_coupling(M=1.0, d=2.0, mu=1.0)
    # NOTE: the formula yukawa_force ~ alpha * exp(-mu*d) so heavier mu
    # *reduces* yukawa force at fixed alpha -> need *larger* alpha to
    # produce the same force. Confirm.
    assert alpha_heavy > alpha_light


# ----- Hard core --------------------------------------------------------


def test_hard_core_mechanism_in_principle_rescues():
    assert HardCoreMechanism().rescues_inspiral is True


def test_hard_core_required_epsilon_formula():
    """epsilon = M^2 * d_eq^(n-1) / (2 n). At n=6, d=2M, M=1:
       epsilon = 1 * 2^5 / 12 = 32 / 12 = 2.667."""
    eps = hard_core_required_epsilon(M=1.0, d_eq=2.0, n=6)
    assert eps == pytest.approx(32.0 / 12.0, rel=1e-12)


def test_hard_core_rejects_n_less_than_2():
    with pytest.raises(ValueError):
        hard_core_required_epsilon(M=1.0, d_eq=2.0, n=1)


# ----- Scalar-tensor / Brans-Dicke -------------------------------------


def test_scalar_tensor_does_not_rescue():
    assert ScalarTensorMechanism().rescues_inspiral is False


def test_brans_dicke_factor_for_GR_limit():
    """As omega_BD -> infinity, BD -> GR. factor -> 1."""
    f_inf = brans_dicke_emission_factor(omega_BD=1e8)
    assert f_inf == pytest.approx(1.0, abs=1e-7)


def test_brans_dicke_factor_grows_for_finite_omega():
    """For omega_BD ~ 1, the factor is significantly > 1 (faster inspiral)."""
    f = brans_dicke_emission_factor(omega_BD=1.0)
    assert f > 1.0


# ----- Dark photon -----------------------------------------------------


def test_dark_photon_mechanism_in_principle_rescues():
    assert DarkPhotonMechanism().rescues_inspiral is True


def test_dark_photon_required_charge_equals_mass():
    """For F_D = q_D^2/r^2 to balance F_N = M^2/r^2, q_D = M."""
    for M in (0.5, 1.0, 10.0):
        for d in (2.0, 5.0, 20.0):
            assert dark_photon_required_charge(M, d) == pytest.approx(M, rel=1e-12)


# ----- LQG area ---------------------------------------------------------


def test_lqg_area_does_not_rescue():
    assert LQGAreaMechanism().rescues_inspiral is False


def test_lqg_area_gap_tiny_for_stellar_BH():
    """For M = 1 M_sun, fractional area gap is ~1e-77 -- exponentially
    too small to balance GW emission."""
    gap = lqg_area_gap_fractional(M_solar=1.0)
    assert gap < 1e-70


def test_lqg_area_gap_decreases_with_M_squared():
    """Heavier BH -> larger area -> smaller fractional gap."""
    g1 = lqg_area_gap_fractional(M_solar=1.0)
    g100 = lqg_area_gap_fractional(M_solar=100.0)
    assert g1 / g100 == pytest.approx(10000.0, rel=1e-6)


# ----- catalogue + verdict ---------------------------------------------


def test_catalogue_has_nine_mechanisms():
    """Extended catalogue: Yukawa, hard core, scalar-tensor, dark photon,
    LQG, f(R), braneworld, parity-charge, Casimir."""
    assert len(CATALOGUE) == 9


def test_most_mechanisms_are_BSM():
    """Eight of nine catalogued mechanisms require beyond-SM physics
    (the Casimir-vacuum-energy one is in-SM but quantitatively too weak)."""
    n_BSM = sum(1 for m in CATALOGUE if m.requires_BSM)
    assert n_BSM >= 8


def test_survey_returns_list_of_verdicts():
    out = survey_binding_mechanisms()
    assert isinstance(out, list)
    assert len(out) == 9
    for v in out:
        assert isinstance(v, BindingVerdict)


def test_survey_no_physical_candidate_anywhere():
    out = survey_binding_mechanisms()
    n_phys = sum(1 for v in out if v.physical_candidate)
    assert n_phys == 0


def test_survey_some_mechanisms_could_rescue():
    out = survey_binding_mechanisms()
    n_rescue = sum(1 for v in out if v.rescues_inspiral)
    # Hard core + dark photon both COULD in principle rescue
    assert n_rescue >= 2


def test_summary_mentions_closed_rescue_path():
    out = survey_binding_mechanisms()
    s = summarise_binding_survey(out)
    assert "CLOSED" in s
    assert "NEW physics" in s


# ----- new mechanisms (iteration) ----------------------------------------


def test_fr_gravity_yukawa_range_scales_with_alpha():
    from systrophe.knopp.knopp_toroidal_binding import (
        fr_gravity_yukawa_range,
    )
    # range ~ sqrt(6 * alpha), monotone in alpha
    r1 = fr_gravity_yukawa_range(alpha=1.0)
    r10 = fr_gravity_yukawa_range(alpha=100.0)
    assert r10 > r1


def test_braneworld_enhancement_negligible_at_BH_scales():
    from systrophe.knopp.knopp_toroidal_binding import (
        add_gravity_enhancement_factor,
    )
    # At BH-binary scales (d = 2 km in SI), with R_compact = 10^-5 m
    # in geometric units M=1 -> R_compact = 10^-5/M_solar_km
    # The factor should be ~ 1 + (negligibly tiny)
    factor = add_gravity_enhancement_factor(d=2.0)
    assert factor > 1.0
    assert factor - 1.0 < 1e-5  # negligible


def test_parity_charge_antiparallel_sum_zero():
    from systrophe.knopp.knopp_toroidal_binding import (
        chirality_charge_sum_antiparallel,
    )
    """For antiparallel maximal spins, the Chern-Simons chirality
    charges cancel exactly. This is what defeats the parity-violation
    rescue."""
    assert chirality_charge_sum_antiparallel(M=1.0) == 0.0
    assert chirality_charge_sum_antiparallel(M=10.0) == 0.0


# ----- rescuability landscape -------------------------------------------


def test_landscape_returns_list_of_points():
    from systrophe.knopp.knopp_toroidal_binding import (
        rescuability_landscape, RescuabilityPoint,
    )
    landscape = rescuability_landscape()
    assert len(landscape) >= 5
    for p in landscape:
        assert isinstance(p, RescuabilityPoint)


def test_landscape_only_repulsive_steep_creates_stable_eq():
    """The classification: ONLY repulsive, steeper-than-1/r^2 force
    laws create stable equilibria."""
    from systrophe.knopp.knopp_toroidal_binding import (
        rescuability_landscape,
    )
    landscape = rescuability_landscape()
    for p in landscape:
        if p.creates_stable_equilibrium:
            assert p.sign == "repulsive"
            assert p.power > 2.0


def test_landscape_summary_string():
    from systrophe.knopp.knopp_toroidal_binding import (
        rescuability_landscape, summarise_landscape,
    )
    landscape = rescuability_landscape()
    s = summarise_landscape(landscape)
    assert "REPULSIVE" in s
    assert "stable" in s.lower()
