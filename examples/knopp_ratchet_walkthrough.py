"""Knopp Drive reversible-ratcheting pendulum walkthrough.

Demonstrates how a *direction-blind* warp drive is given a stable directional
bias by porting the proven TriCameral `ParetoRatchet` (rising-floor pawl) onto
the horn-toroidal steering control, and re-expresses the cost as amplified
(dynamical) Casimir pump power rather than a reservoir of exotic matter.

Sections:
  1. The bare drive is direction-blind (cost is the same forward and backward).
  2. The ratchet manufactures a tunable forward/reverse energy asymmetry.
  3. A multi-cycle biased traversal accumulates net directed displacement,
     with rollback-on-regression for stability.
  4. The negative energy is squeezed Casimir vacuum (DCE), pumped at 1/Q^2,
     with Pfenning-Ford saturated, not beaten — no exotic-matter reservoir.
  5. Address-space novelty-catcher verdict (standing project rule).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from systrophe.knopp_drive import KnoppDrive
from systrophe.knopp_ratchet import (
    WarpRatchetConfig,
    reverse_asymmetry,
    summarise_ratchet,
)


def main() -> None:
    print("=" * 70)
    print("Knopp Drive - reversible-ratcheting pendulum bias controller")
    print("=" * 70)
    print()

    drive = KnoppDrive(Q=10.0, epsilon_horn=0.2, sigma_shell=4.0)

    # 1. Direction-blindness of the bare drive ------------------------------
    print("1. The bare drive is direction-blind")
    print("   composite_E_neg does not depend on the horn-twist axis, so the")
    print("   bare exotic-matter cost is identical forward and backward:")
    a_fwd = drive.asymmetry(heading=0.0)
    a_bwd = drive.asymmetry(heading=math.pi)
    print(f"     E_advance(heading=0)   = {a_fwd['e_advance']:.5e}")
    print(f"     E_advance(heading=pi)  = {a_bwd['e_advance']:.5e}")
    print("   -> the drive has no built-in preferred direction.")
    print()

    # 2. The ratchet manufactures the asymmetry -----------------------------
    print("2. The rising-floor pawl manufactures a tunable asymmetry")
    print("   (port of TriCameral proven.py ParetoRatchet):")
    print(f"     {'floor_pct':>9} {'E_advance':>12} {'E_reverse':>12} "
          f"{'reverse/forward':>16}")
    for fp in (0.0, 0.5, 0.85, 0.95, 0.99):
        r = reverse_asymmetry(WarpRatchetConfig(drive=drive.config, floor_pct=fp))
        print(f"     {fp:9.2f} {r['e_advance']:12.4e} {r['e_reverse']:12.4e} "
              f"{r['asymmetry_ratio']:15.2f}x")
    print("   floor_pct = 0  -> reciprocal pendulum (scallop theorem)")
    print("   floor_pct -> 1 -> near-perfect one-way valve")
    print()

    # 3. Biased traversal ----------------------------------------------------
    print("3. Biased traversal (64 cycles, heading = 0)")
    rep = drive.bias(heading=0.0, n_cycles=64, eps_high=0.6, floor_pct=0.85)
    print("   " + summarise_ratchet(rep))
    rep_rev = drive.bias(heading=math.pi, n_cycles=64)
    print(f"   reverse heading net displacement = {rep_rev.net_displacement:.3e}"
          " (same magnitude, opposite lock) -> reversible")
    print()

    # 4. Amplified-Casimir pump accounting ----------------------------------
    print("4. The negative energy is squeezed Casimir vacuum (DCE), not a")
    print("   reservoir of exotic matter:")
    for Q in (8.0, 10.0, 100.0, 1000.0):
        d = KnoppDrive(Q=Q, sigma_shell=4.0)
        acc = d.pump_accounting(heading=0.0)
        print(f"     Q={Q:7.0f}  P_drive={acc['power_stroke_pump_power']:.3e}  "
              f"PF_ok={acc['pfenning_ford_compatible']}  "
              f"E*tau/bound={acc['pf_product'] / acc['pf_bound']:.1f}x")
    print("   P_drive ~ 1/Q^2 (real positive-energy pump); PF saturated, not")
    print("   beaten; recovery stroke coasts free through the Tipler band.")
    print()

    # 4b. Bias energy ledger with Ford-Roman quantum interest ---------------
    print("4b. Bias energy ledger (Ford-Roman quantum-interest payback)")
    led = drive.energy_ledger(heading=0.0, n_cycles=64, interest_alpha=1.0)
    print(f"     advancing strokes        = {led.n_advancing_strokes}")
    print(f"     net displacement         = {led.net_displacement:.3e}")
    print(f"     E_pump   (real input)    = {led.e_pump_total:.4e}")
    print(f"     E_borrow (neg held)      = {led.e_borrow_total:.4e}")
    print(f"     interest rate r          = {led.interest_rate:.2f}  "
          f"(alpha={led.interest_alpha})")
    print(f"     E_repay  (principal+int) = {led.e_repay_total:.4e}")
    print(f"     ---------------------------------------------")
    print(f"     E_ledger TOTAL           = {led.e_ledger_total:.4e}")
    print(f"     irreducible floor (Q->inf)= {led.irreducible_floor:.4e}")
    print(f"     energy / unit displacement= {led.energy_per_unit_displacement:.4e}")
    print(f"     reverse-undo energy      = {led.reverse_undo_energy:.4e} "
          f"(>> forward: the ratchet's point)")
    print("   At alpha=1, E_repay == floor (debt conserved); amplification")
    print("   only removes the pump overhead, never the quantum-interest")
    print("   principal. E_ledger -> floor as Q -> inf:")
    for Q in (1.0, 10.0, 100.0, 1000.0):
        L = KnoppDrive(Q=Q, sigma_shell=4.0).energy_ledger(n_cycles=64)
        over = 100.0 * (L.e_ledger_total / L.irreducible_floor - 1.0)
        print(f"     Q={Q:7.0f}  E_ledger={L.e_ledger_total:.4e}  "
              f"pump overhead = +{over:5.1f}% over floor")
    print()

    # 4c. SI feasibility report ---------------------------------------------
    print("4c. SI feasibility (irreducible Ford-Roman principal in joules)")
    print(f"     {'bubble R':>9} {'wall s':>7} {'floor [J]':>12} "
          f"{'Jupiter-m':>10} {'shortfall':>12}")
    for R_m, s_m in ((1.0, 1.0), (10.0, 1.0), (100.0, 0.001)):
        f = KnoppDrive(Q=100.0).feasibility_report(
            bubble_radius_m=R_m, wall_thickness_m=s_m, v_s_over_c=1.0)
        print(f"     {R_m:8.0f}m {s_m:6.3f}m {f.irreducible_floor_J:12.3e} "
              f"{f.floor_jupiter_masses:9.2g}x "
              f"1e{f.shortfall_orders_of_magnitude:4.0f}")
    f1 = KnoppDrive(Q=100.0).feasibility_report(bubble_radius_m=1.0,
                                                wall_thickness_m=1.0)
    print(f"   {f1.verdict}")
    print("   The amplified-Casimir ratchet is elegant control; the energy")
    print("   floor is the same Jupiter-mass wall every warp metric hits.")
    print()

    # 5. Novelty catcher -----------------------------------------------------
    print("5. Address-space novelty catcher")
    print(f"   traversal verdict: {rep.novelty_verdict} "
          f"({rep.novelty_n_sharp} sharp features)")
    print("   (smooth = continuously tunable bias, not an emergent transition;")
    print("    this is a speculative controller, NOT a validated result.)")
    print()

    out = {
        "direction_blind_e_advance_fwd": a_fwd["e_advance"],
        "direction_blind_e_advance_bwd": a_bwd["e_advance"],
        "asymmetry_at_floor_0.85": drive.asymmetry()["asymmetry_ratio"],
        "net_displacement_64cyc": rep.net_displacement,
        "total_exotic_cost": rep.total_exotic_cost,
        "pump_accounting_Q10": KnoppDrive(Q=10.0).pump_accounting(),
        "ledger_Q10": {
            "e_ledger_total": led.e_ledger_total,
            "e_pump_total": led.e_pump_total,
            "e_repay_total": led.e_repay_total,
            "irreducible_floor": led.irreducible_floor,
            "energy_per_unit_displacement": led.energy_per_unit_displacement,
            "reverse_undo_energy": led.reverse_undo_energy,
        },
        "feasibility_1m_bubble": {
            "irreducible_floor_J": f1.irreducible_floor_J,
            "floor_jupiter_masses": f1.floor_jupiter_masses,
            "shortfall_orders_of_magnitude": f1.shortfall_orders_of_magnitude,
            "verdict": f1.verdict,
        },
        "novelty_verdict": rep.novelty_verdict,
    }
    out_path = Path(__file__).with_name("knopp_ratchet_walkthrough_results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path.name}")


if __name__ == "__main__":
    main()
