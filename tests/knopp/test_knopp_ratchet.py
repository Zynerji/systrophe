"""Tests for the Knopp-Drive reversible-ratcheting pendulum bias controller."""

import numpy as np
import pytest

from systrophe.knopp.knopp_drive import KnoppDrive
from systrophe.knopp.knopp_ratchet import (
    BiasEnergyLedger,
    FeasibilityReport,
    RatchetTraversalReport,
    WarpParetoRatchet,
    WarpRatchetConfig,
    bias_energy_ledger,
    capacitor_accounting,
    casimir_pump_accounting,
    feasibility_report,
    ledger_Q_sweep,
    ratchet_traversal,
    reverse_asymmetry,
    summarise_ratchet,
)


# --- the ratchet pawl itself ------------------------------------------------


def test_floor_only_rises():
    """The pawl floor must be monotonically non-decreasing across advances."""
    r = WarpParetoRatchet(anchor_disp=1.0, anchor_safety=0.4, floor_pct=0.85)
    floors = []
    for disp in [1.1, 1.3, 1.2, 1.5, 1.4]:
        r.check(disp, 0.4)
        floors.append(r.disp_floor)
    assert all(b >= a - 1e-12 for a, b in zip(floors, floors[1:]))


def test_advance_on_product_improvement():
    r = WarpParetoRatchet(anchor_disp=1.0, anchor_safety=0.4, floor_pct=0.85)
    assert r.check(2.0, 0.5) == "advance"


def test_rollback_when_both_below_floor():
    r = WarpParetoRatchet(anchor_disp=1.0, anchor_safety=0.4, floor_pct=0.85)
    r.check(2.0, 0.5)  # raise the floor
    assert r.check(0.01, 0.001) == "rollback"
    assert r.rollback_count == 1


# --- the physics asymmetry --------------------------------------------------


def test_reverse_costs_more_than_forward():
    """The core claim: retreating one ratchet step costs strictly more exotic
    matter than advancing one. Asymmetry comes from forfeiting the free
    Tipler-gated recovery plus the pawl penalty."""
    a = reverse_asymmetry()
    assert a["e_reverse"] > a["e_advance"]
    assert a["asymmetry_ratio"] > 1.0


def test_asymmetry_grows_with_stricter_pawl():
    """A higher floor_pct (stricter pawl) makes reverse strictly more costly."""
    lo = reverse_asymmetry(WarpRatchetConfig(floor_pct=0.50))
    hi = reverse_asymmetry(WarpRatchetConfig(floor_pct=0.95))
    assert hi["asymmetry_ratio"] > lo["asymmetry_ratio"]


def test_cost_is_direction_symmetric_without_ratchet():
    """Sanity: composite_E_neg does not depend on heading, so the bare power
    stroke cost is the same forward and backward — proving the asymmetry is
    manufactured by the ratchet protocol, not the drive."""
    fwd = reverse_asymmetry(WarpRatchetConfig(heading=0.0))
    bwd = reverse_asymmetry(WarpRatchetConfig(heading=np.pi))
    assert fwd["e_advance"] == pytest.approx(bwd["e_advance"], rel=1e-9)


# --- the traversal ----------------------------------------------------------


def test_traversal_makes_net_forward_progress():
    rep = ratchet_traversal(WarpRatchetConfig(n_cycles=32))
    assert isinstance(rep, RatchetTraversalReport)
    assert rep.net_displacement > 0.0


def test_traversal_reports_novelty_verdict():
    """Standing project rule: every deliverable reports a catcher verdict."""
    rep = ratchet_traversal(WarpRatchetConfig(n_cycles=24))
    assert rep.novelty_verdict in {"novel_structure", "smooth", "uniform"}


def test_overstrong_pendulum_triggers_rollback():
    """Driving eps past the pinch threshold should provoke rollbacks (the
    stability mechanism), not a runaway."""
    sched = np.linspace(0.6, 1.2, 40)  # ramps past pinch (= 1.0)
    rep = ratchet_traversal(WarpRatchetConfig(n_cycles=40), eps_schedule=sched)
    assert rep.rollbacks >= 1


def test_summary_string():
    rep = ratchet_traversal(WarpRatchetConfig(n_cycles=8))
    s = summarise_ratchet(rep)
    assert "WarpRatchet" in s and "net_disp" in s


# --- Casimir / DCE pump accounting ------------------------------------------


def test_casimir_pump_no_exotic_reservoir():
    """The amplified-Casimir route: real pump power on the power stroke, free
    recovery, PF saturated (not beaten), no exotic-matter reservoir."""
    acc = casimir_pump_accounting()
    assert acc["needs_exotic_matter_reservoir"] is False
    assert acc["power_stroke_pump_power"] > 0.0
    assert acc["recovery_pump_power"] == 0.0
    assert acc["pfenning_ford_compatible"] is True
    assert acc["pf_product"] >= acc["pf_bound"]


def test_pump_power_falls_as_inverse_Q_squared():
    """P_drive ~ 1/Q^2: the dynamical-Casimir amplification trade."""
    from dataclasses import replace
    base = WarpRatchetConfig()
    lowQ = casimir_pump_accounting(
        replace(base, drive=replace(base.drive, Q=10.0)))
    highQ = casimir_pump_accounting(
        replace(base, drive=replace(base.drive, Q=100.0)))
    ratio = lowQ["power_stroke_pump_power"] / highQ["power_stroke_pump_power"]
    assert ratio == pytest.approx(100.0, rel=0.05)


# --- KnoppDrive high-level interface ----------------------------------------


def test_knoppdrive_bias_method():
    drive = KnoppDrive(Q=10.0, epsilon_horn=0.2)
    rep = drive.bias(heading=0.0, n_cycles=24)
    assert isinstance(rep, RatchetTraversalReport)
    assert rep.net_displacement > 0.0


def test_knoppdrive_asymmetry_and_pump_methods():
    drive = KnoppDrive(Q=10.0)
    a = drive.asymmetry(heading=0.0)
    assert a["asymmetry_ratio"] > 1.0
    p = drive.pump_accounting(heading=0.0)
    assert p["needs_exotic_matter_reservoir"] is False


# --- bias energy ledger / Ford-Roman quantum interest -----------------------


def test_ledger_bottom_line_is_sum_of_accounts():
    led = bias_energy_ledger(WarpRatchetConfig(n_cycles=32))
    assert isinstance(led, BiasEnergyLedger)
    assert led.e_ledger_total == pytest.approx(
        led.e_pump_total + led.e_repay_total, rel=1e-9)
    assert led.e_repay_total == pytest.approx(
        led.e_borrow_total + led.e_interest_total, rel=1e-9)


def test_repay_overcompensates_the_loan():
    """Ford-Roman: repayment must exceed the borrowed negative energy."""
    led = bias_energy_ledger(WarpRatchetConfig(n_cycles=16))
    assert led.e_repay_total > led.e_borrow_total
    assert led.interest_rate > 0.0


def test_alpha1_conserves_debt_at_the_floor():
    """At alpha=1 the per-stroke repayment equals the bare negative energy,
    so the ledger sits at floor + pump overhead and tends to the floor."""
    led = bias_energy_ledger(WarpRatchetConfig(n_cycles=8), interest_alpha=1.0)
    assert led.e_repay_total == pytest.approx(led.irreducible_floor, rel=1e-6)
    assert led.e_ledger_total > led.irreducible_floor  # pump overhead remains


def test_higher_Q_lowers_ledger_toward_floor_at_alpha1():
    lo = bias_energy_ledger(
        WarpRatchetConfig(drive=KnoppDrive(Q=5.0).config, n_cycles=8))
    hi = bias_energy_ledger(
        WarpRatchetConfig(drive=KnoppDrive(Q=200.0).config, n_cycles=8))
    assert hi.e_ledger_total < lo.e_ledger_total
    assert hi.e_ledger_total >= hi.irreducible_floor - 1e-9


def test_superlinear_interest_has_interior_optimum():
    """alpha>1 makes holding longer (high Q) increasingly expensive, so the
    cost-per-displacement Q-sweep has an interior optimum, not Q->inf."""
    # Optimum Q* = (1/(alpha-1))^(1/alpha); alpha=1.1 -> Q* ~ 8, interior.
    sweep = ledger_Q_sweep(n_Q=20, interest_alpha=1.1)
    assert 1.0 < sweep["optimal_Q"] < 200.0


def test_knoppdrive_energy_ledger_method():
    led = KnoppDrive(Q=10.0).energy_ledger(heading=0.0, n_cycles=16)
    assert led.energy_per_unit_displacement > 0.0
    assert led.reverse_undo_energy > led.e_ledger_total


# --- SI feasibility report --------------------------------------------------


def test_feasibility_floor_is_jupiter_scale_for_metre_bubble():
    """A 1 m, luminal bubble's irreducible principal is ~Jupiter mass-energy
    (the order-of-magnitude every warp metric hits)."""
    f = feasibility_report(bubble_radius_m=1.0, wall_thickness_m=1.0,
                           v_s_over_c=1.0, n_cycles=64)
    assert isinstance(f, FeasibilityReport)
    assert 0.1 < f.floor_jupiter_masses < 100.0
    assert f.irreducible_floor_J > 1e43


def test_feasibility_scales_as_R2_over_sigma():
    """E ~ R^2/sigma: doubling R quadruples the floor; doubling sigma halves it."""
    base = feasibility_report(bubble_radius_m=1.0, wall_thickness_m=1.0)
    big_R = feasibility_report(bubble_radius_m=2.0, wall_thickness_m=1.0)
    thick = feasibility_report(bubble_radius_m=1.0, wall_thickness_m=2.0)
    assert big_R.irreducible_floor_J == pytest.approx(
        4.0 * base.irreducible_floor_J, rel=1e-6)
    assert thick.irreducible_floor_J == pytest.approx(
        0.5 * base.irreducible_floor_J, rel=1e-6)


def test_feasibility_shortfall_is_dozens_of_orders():
    f = feasibility_report(lab_source_J=1e-15)
    assert f.shortfall_orders_of_magnitude > 40.0
    assert "INFEASIBLE" in f.verdict


def test_knoppdrive_feasibility_method():
    f = KnoppDrive(Q=100.0).feasibility_report(bubble_radius_m=10.0)
    assert f.floor_jupiter_masses > 0.0
    assert f.shortfall_orders_of_magnitude > 0.0


# --- surface capacitor accounting -------------------------------------------


def test_capacitor_reduces_peak_charge_by_Q():
    """The capacitor holds |E_neg|/Q at once: peak charge falls as 1/Q."""
    acc = capacitor_accounting(WarpRatchetConfig(drive=KnoppDrive(Q=50.0).config))
    assert acc["peak_charge_natural"] == pytest.approx(
        acc["bare_requirement_natural"] / 50.0, rel=1e-6)
    assert acc["peak_reduction_factor"] == pytest.approx(50.0)


def test_capacitor_throughput_is_Q_independent():
    """Total throughput (Ford-Roman principal) does NOT depend on Q, even as
    the peak charge shrinks — the capacitor solves peak, not total."""
    lo = capacitor_accounting(WarpRatchetConfig(drive=KnoppDrive(Q=10.0).config))
    hi = capacitor_accounting(WarpRatchetConfig(drive=KnoppDrive(Q=1000.0).config))
    assert hi["peak_charge_natural"] < lo["peak_charge_natural"]
    assert hi["total_throughput_natural"] == pytest.approx(
        lo["total_throughput_natural"], rel=1e-6)
