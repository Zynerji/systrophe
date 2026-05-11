"""End-to-end demonstration of Systrophe v0.14 quantum-layer modules.

Runs:
  1. Hadamard <T_munu>_ren on a supercritical van Stockum exterior
  2. Z_3 anomaly inflow closure check at several gamma_eff values
  3. Acoustic vs gravitational Hawking T comparison at the horizon
  4. Back-reaction landscape sweep across pair offset delta
  5. Joint Floquet quasi-energy spectrum on Z_3 cover
  6. D-CTC fixed-point convergence
  7. DSI test on Tipler-sinusoid zero set

Writes machine-readable results to
`examples/quantum_layer_walkthrough_results.json`.
"""

from __future__ import annotations

import json
import os

import numpy as np

from systrophe import (
    SystrophePair,
    TiplerSinusoid,
    VanStockumInterior,
    analyze_floquet_mobius,
    apply_channel,
    back_reaction_landscape,
    compare_acoustic_vs_gravitational_T_H,
    dctc_fixed_point,
    discrete_scale_invariance_test,
    hadamard_offtrace_T,
    hadamard_T_trace,
    maximally_mixed_state,
    z3_anomaly_inflow_balance,
    z3_branch_etas,
    z3_ctc_unitary,
)
from systrophe.acoustic_metric import acoustic_horizon_radius


def to_native(o):
    """Convert numpy scalars / arrays to JSON-serialisable Python types."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_native(x) for x in o]
    return o


def step_1_hadamard_offtrace(vs, r_test):
    """Section 1: Hadamard <T_munu>_ren on supercritical VS."""
    T = hadamard_offtrace_T(vs, r=r_test)
    trace = hadamard_T_trace(T, vs, r_test)
    return {
        "r": r_test,
        "T_tt": float(T[0, 0]),
        "T_phi_phi": float(T[2, 2]),
        "trace": float(trace),
        "trace_anomaly_constant": 1 / (2880 * np.pi ** 2),
    }


def step_2_z3_anomaly(gamma_values):
    """Section 2: Z_3 anomaly closure across gamma_eff."""
    results = []
    for g in gamma_values:
        etas = z3_branch_etas(g)
        balance = z3_anomaly_inflow_balance(gamma_eff=g, B_z=0.0, area=1.0)
        results.append({
            "gamma_eff": float(g),
            "branch_etas": etas.tolist(),
            "sum_eta": float(np.sum(etas)),
            "required_bulk_inflow": balance["required_bulk_inflow"],
        })
    return results


def step_3_acoustic_hawking(vs):
    """Section 3: Acoustic vs gravitational Hawking T."""
    r_h = acoustic_horizon_radius(vs, r_min=1.05, r_max=20.0)
    if r_h is None:
        return {"error": "no horizon found"}
    cmp = compare_acoustic_vs_gravitational_T_H(vs, r_horizon=r_h)
    return {
        "r_horizon": float(r_h),
        "T_acoustic": cmp["T_acoustic"],
        "T_gravitational": cmp["T_gravitational"],
        "rel_diff": cmp["rel_diff"],
    }


def step_4_back_reaction_landscape():
    """Section 4: Back-reaction landscape sweep."""
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    deltas = np.linspace(0, 2 * np.pi, 16)
    rs = np.array([1.5, 2.0, 2.5])
    lndscp = back_reaction_landscape(s1, s2, deltas, rs)
    return {
        "deltas": lndscp.deltas.tolist(),
        "residuals": lndscp.residuals.tolist(),
        "min_delta": lndscp.min_delta,
        "min_residual": lndscp.min_residual,
        "max_delta": lndscp.max_delta,
        "max_residual": lndscp.max_residual,
        "extinction_at_pi_check": abs(lndscp.min_delta - np.pi) < 0.5,
    }


def step_5_floquet_mobius():
    """Section 5: Joint Floquet spectrum at a resonance."""
    energies = np.array([0.2, 0.5, 0.8])
    result = analyze_floquet_mobius(
        energies, hopping=0.1, drive_amp=0.3, omega_drive=2.5, n_steps=200,
    )
    return {
        "branch_energies": energies.tolist(),
        "quasi_energies": result.quasi_energies.tolist(),
        "drive_amp": result.drive_amp,
        "omega_drive": result.omega_drive,
    }


def step_6_dctc_fixed_point():
    """Section 6: D-CTC fixed-point on Z_3 cover."""
    dim_cr = 2
    U = z3_ctc_unitary(dim_cr=dim_cr, phase=0.2)
    sigma_cr = maximally_mixed_state(dim_cr)
    result = dctc_fixed_point(
        U, sigma_cr, dim_cr=dim_cr,
        rho_ctc_init=maximally_mixed_state(3),
        tol=1e-10, max_iter=500,
    )
    return {
        "converged": result["converged"],
        "iterations": result["iterations"],
        "residual": result["residual"],
        "fixed_point_trace": float(np.real(np.trace(result["rho_ctc"]))),
    }


def step_7_dsi_test():
    """Section 7: DSI test on Tipler-sinusoid zero set."""
    s = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    from systrophe.tipler_fractal import zero_set
    zeros = zero_set(s.R, s.alpha, s.delta, r_min=1.0, r_max=np.exp(8))
    result = discrete_scale_invariance_test(zeros)
    return {
        "n_zeros": int(len(zeros)),
        "best_ratio": result["best_ratio"],
        "expected_ratio_exp_pi_over_alpha": float(np.exp(np.pi / s.alpha)),
        "rms_log_dev": result["rms_log_dev"],
        "is_dsi": bool(result["is_dsi"]),
    }


def main(output_path: str = "examples/quantum_layer_walkthrough_results.json"):
    print("Systrophe v0.14 quantum-layer walkthrough")
    print("=" * 60)
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r_test = 2.0

    results = {}

    print("\n[1/7] Hadamard <T_munu>_ren at r = 2.0")
    results["step_1_hadamard"] = step_1_hadamard_offtrace(vs, r_test)
    print(f"      trace = {results['step_1_hadamard']['trace']:.3e}")

    print("\n[2/7] Z_3 anomaly closure")
    results["step_2_anomaly"] = step_2_z3_anomaly([0.0, 0.3, 1.0, 2.0])
    sum0 = results["step_2_anomaly"][0]["sum_eta"]
    print(f"      sum_eta @ gamma=0 = {sum0:.3e}  (closure)")

    print("\n[3/7] Acoustic vs gravitational Hawking T")
    results["step_3_acoustic_T"] = step_3_acoustic_hawking(vs)
    if "rel_diff" in results["step_3_acoustic_T"]:
        print(f"      rel diff = {results['step_3_acoustic_T']['rel_diff']:.3e}")

    print("\n[4/7] Back-reaction landscape sweep")
    results["step_4_back_reaction"] = step_4_back_reaction_landscape()
    print(f"      min @ delta = {results['step_4_back_reaction']['min_delta']:.3f}")
    print(f"      extinction-at-pi check: {results['step_4_back_reaction']['extinction_at_pi_check']}")

    print("\n[5/7] Joint Floquet quasi-energy spectrum")
    results["step_5_floquet"] = step_5_floquet_mobius()
    qe = results["step_5_floquet"]["quasi_energies"]
    print(f"      quasi-energies = {[f'{q:.4f}' for q in qe]}")

    print("\n[6/7] D-CTC fixed-point convergence")
    results["step_6_dctc"] = step_6_dctc_fixed_point()
    print(f"      converged: {results['step_6_dctc']['converged']}"
          f" in {results['step_6_dctc']['iterations']} iterations")

    print("\n[7/7] DSI test on Tipler-sinusoid zero set")
    results["step_7_dsi"] = step_7_dsi_test()
    print(f"      n_zeros = {results['step_7_dsi']['n_zeros']}; "
          f"DSI = {results['step_7_dsi']['is_dsi']}")

    print("\n" + "=" * 60)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(to_native(results), f, indent=2)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
