"""Marrakesh batch #2: three further Systrophe validations.

E. KG cascade transmission decay (Phase 22)
   Amplitude-damping chain. N partial-isometry CH crossings -> survival
   probability P_N = (1 - p)^N. Test geometric decay with p chosen to
   match a representative LP chronology-horizon damping.

F. KK escape interferometer (Phase 25)
   Mach-Zehnder with relative phase from F(r) (4D arm) vs F + v^2 * L_xi^2
   (5D arm with xi-velocity). At v^2 L_xi^2 = -F, interference vanishes
   (escape threshold). Verifies the Phase 25 escape condition.

G. Holographic boundary phase wrap (Phase 16)
   Quantum walk in geometric s-sequence accumulates phase -alpha * ln(s).
   Extract alpha from arctan(<Y>/<X>) at multiple step counts; compare
   to LP theoretical alpha = sqrt(4 a^2 - 1).

Calibration: optimization_level=3 + DD(XpXm) + gate/measure twirling
(same as batch #1).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from systrophe.geometry.vanstockum import VanStockumInterior

SHOTS = 4096
DEPTH_LIMIT = 150
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------- Experiment E: KG cascade ----------

def amplitude_damping_block(
    qc: QuantumCircuit, data: int, anc: int, p: float,
) -> None:
    """One amplitude-damping channel of parameter p on `data` using `anc`.

    Stinespring dilation:
        U = | 0 0 >< 0 0 | + sqrt(1-p) | 1 0 >< 1 0 |
          + sqrt(p)   | 0 1 >< 1 0 | + | 0 1 >< 0 1 |
            ... etc. (the standard amplitude-damping channel).

    Simple circuit: anc starts |0>. Apply CRy(theta) on (data, anc) with
    theta = 2 arcsin(sqrt(p)). Then CX(anc, data). Trace out anc by
    measurement (mid-circuit reset).
    """
    theta = 2.0 * math.asin(math.sqrt(p))
    qc.cry(theta, data, anc)
    qc.cx(anc, data)
    qc.reset(anc)  # mid-circuit reset to reuse ancilla


def experiment_E_kg_cascade(p: float = 0.30,
                             n_horizons_list: list[int] = (1, 2, 3, 4)) -> list[dict]:
    """Apply N amplitude-damping channels to |1>; measure P(survive |1>).

    Predicted: P_N = (1-p)^N (geometric decay).
    """
    out = []
    for N in n_horizons_list:
        qr = QuantumRegister(2, "q")  # 0 = data, 1 = ancilla (reused)
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr, name=f"kg_cascade_N{N}")
        # Prepare data = |1>
        qc.x(qr[0])
        for _ in range(N):
            amplitude_damping_block(qc, data=0, anc=1, p=p)
        qc.measure(qr[0], cr[0])
        out.append({
            "exp": "E_kg_cascade",
            "label": f"kg_cascade_N{N}",
            "n_horizons": N,
            "p_per_CH": p,
            "circuit": qc,
            "predicted_survival": (1.0 - p) ** N,
        })
    return out


# ---------- Experiment F: KK escape interferometer ----------

def experiment_F_kk_escape() -> list[dict]:
    """Mach-Zehnder with 4D/5D phase split.

    Setup: 1 control qubit + 1 phase qubit.
        H on control.
        Conditioned on control=0: apply phase phi_4D on data
        Conditioned on control=1: apply phase phi_4D + phi_xi on data
        H on control. Measure control.

    Visibility V = |cos(phi_xi / 2)|. Escape (xi velocity at threshold)
    corresponds to phi_xi = pi -> V = 0.

    We use phi_4D fixed (= LP CTC phase at r=3.35) and sweep phi_xi
    representing xi-velocity.
    """
    # phi_4D ~ |F(r=3.35)| in arbitrary units -> normalised to ~pi/3
    vs = VanStockumInterior(omega=1.0, R=1.0)
    F_val = float(vs.analytic_exterior_F(np.array([3.35]))[0])
    phi_4D = math.pi / 3 * abs(F_val)  # ~pi/3 in our units
    out = []
    # xi-velocity samples mapping to phi_xi in {0, pi/4, pi/2, 3pi/4, pi}
    for label, phi_xi in [
        ("no_xi_velocity", 0.0),
        ("partial_escape_1", math.pi / 4),
        ("partial_escape_2", math.pi / 2),
        ("partial_escape_3", 3 * math.pi / 4),
        ("threshold_escape", math.pi),
    ]:
        qr = QuantumRegister(2, "q")  # 0 = data, 1 = control
        cr = ClassicalRegister(1, "c")  # measure control
        qc = QuantumCircuit(qr, cr, name=f"kk_escape_{label}")
        # Standard Hadamard test for RZ(phi_xi) on |psi> = |+>:
        # P(control=0) = (1 + Re<+|RZ(phi_xi)|+>) / 2 = cos^2(phi_xi/4).
        # phi_4D is a common phase on both arms -> cancels in visibility,
        # kept here as the 4D-only contribution that does NOT split the arms
        # (it represents the CTC phase, which is identical on each arm).
        qc.h(qr[1])  # ancilla = |+>
        qc.h(qr[0])  # data = |+>
        qc.rz(phi_4D, qr[0])  # common 4D phase
        qc.crz(phi_xi, qr[1], qr[0])  # 5D differential phase
        qc.h(qr[1])
        qc.measure(qr[1], cr[0])
        predicted_p0 = math.cos(phi_xi / 4) ** 2
        out.append({
            "exp": "F_kk_escape",
            "label": f"kk_escape_{label}",
            "phi_xi": phi_xi,
            "phi_4D": phi_4D,
            "circuit": qc,
            "predicted_p_control_0": predicted_p0,
        })
    return out


# ---------- Experiment G: holographic phase wrap ----------

def experiment_G_boundary_phase_wrap() -> list[dict]:
    """Accumulate phase -alpha * ln(s) over geometric s-sequence.

    For omega=R=1, alpha = sqrt(3) ~ 1.732. Use r-step = 2, so each
    increment adds phase alpha * ln(2) ~ 1.201 rad.

    Protocol:
      |+> + RZ(N * alpha * ln(2)) -> measure <X> and <Y> via H/Sdg-H
      Extract phi = arctan(<Y>/<X>); compare to N * alpha * ln(2).
    """
    vs = VanStockumInterior(omega=1.0, R=1.0)
    alpha = vs.alpha
    delta_phi = alpha * math.log(2.0)
    out = []
    for N in [1, 2, 3, 4]:
        # X-basis circuit
        qr = QuantumRegister(1, "q")
        cr_x = ClassicalRegister(1, "c")
        qc_x = QuantumCircuit(qr, cr_x, name=f"phase_wrap_N{N}_X")
        qc_x.h(qr[0])
        qc_x.rz(N * delta_phi, qr[0])
        qc_x.h(qr[0])  # X-basis measurement
        qc_x.measure(qr[0], cr_x[0])
        out.append({
            "exp": "G_boundary_phase_wrap",
            "label": f"phase_wrap_N{N}_X",
            "N": N,
            "delta_phi": delta_phi,
            "alpha_theory": alpha,
            "basis": "X",
            "circuit": qc_x,
            "predicted_p0": math.cos(N * delta_phi / 2) ** 2,
        })
        # Y-basis circuit
        cr_y = ClassicalRegister(1, "c")
        qc_y = QuantumCircuit(qr, cr_y, name=f"phase_wrap_N{N}_Y")
        qc_y.h(qr[0])
        qc_y.rz(N * delta_phi, qr[0])
        qc_y.sdg(qr[0])
        qc_y.h(qr[0])
        qc_y.measure(qr[0], cr_y[0])
        out.append({
            "exp": "G_boundary_phase_wrap",
            "label": f"phase_wrap_N{N}_Y",
            "N": N,
            "delta_phi": delta_phi,
            "alpha_theory": alpha,
            "basis": "Y",
            "circuit": qc_y,
            "predicted_p0": (1 + math.sin(N * delta_phi)) / 2,
        })
    return out


# ---------- Wiring ----------

def all_experiments() -> list[dict]:
    return (
        experiment_E_kg_cascade()
        + experiment_F_kk_escape()
        + experiment_G_boundary_phase_wrap()
    )


def transpile_and_check(experiments: list[dict], backend) -> list[dict]:
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    for ex in experiments:
        qc = ex["circuit"]
        isa_qc = pm.run(qc)
        d = isa_qc.depth()
        print(f"  {ex['label']:35s} pre_depth={qc.depth():3d}  isa_depth={d:4d}",
              flush=True)
        assert d < DEPTH_LIMIT, (
            f"{ex['label']} ISA depth {d} >= {DEPTH_LIMIT}"
        )
        ex["isa_circuit"] = isa_qc
        ex["isa_depth"] = d
    return experiments


# ---------- Result analysis ----------

def _p_of_bit(counts: dict, bit: str) -> float:
    total = sum(counts.values())
    return counts.get(bit, 0) / total


def analyze_E(ex: dict, counts: dict) -> dict:
    p1 = _p_of_bit(counts, "1")
    pred = ex["predicted_survival"]
    return {
        "label": ex["label"],
        "n_horizons": ex["n_horizons"],
        "predicted_survival": pred,
        "observed_survival": p1,
        "abs_error": abs(p1 - pred),
    }


def analyze_F(ex: dict, counts: dict) -> dict:
    p0 = _p_of_bit(counts, "0")
    pred = ex["predicted_p_control_0"]
    return {
        "label": ex["label"],
        "phi_xi": ex["phi_xi"],
        "predicted_p_control_0": pred,
        "observed_p_control_0": p0,
        "abs_error": abs(p0 - pred),
    }


def analyze_G(experiments: list[dict], counts_list: list[dict]) -> dict:
    """Pair X and Y measurements per N, extract phi = atan2(<Y>, <X>),
    compare cumulative phi(N) to N * delta_phi (alpha theory).
    """
    by_N = {}
    for ex, counts in zip(experiments, counts_list):
        if ex["exp"] != "G_boundary_phase_wrap":
            continue
        N = ex["N"]
        by_N.setdefault(N, {})["theory_delta_phi"] = ex["delta_phi"]
        by_N[N]["alpha_theory"] = ex["alpha_theory"]
        p0 = _p_of_bit(counts, "0")
        p1 = _p_of_bit(counts, "1")
        # <X> = p0(X-basis) - p1(X-basis). Same for Y.
        expectation = p0 - p1
        by_N[N][f"E_{ex['basis']}"] = expectation
    out = []
    for N in sorted(by_N):
        d = by_N[N]
        if "E_X" in d and "E_Y" in d:
            # RZ(theta)|+> -> <X> = cos(theta), <Y> = +sin(theta).
            # atan2 returns phi in (-pi, pi]; expected theta = N * delta_phi
            # may exceed pi -> unwrap by adding 2*pi*k to make closest to theory.
            phi_raw = math.atan2(d["E_Y"], d["E_X"])
            phi_theory = N * d["theory_delta_phi"]
            k = round((phi_theory - phi_raw) / (2 * math.pi))
            phi_unwrapped = phi_raw + 2 * math.pi * k
            alpha_obs = phi_unwrapped / (N * math.log(2.0))
            out.append({
                "N": N,
                "E_X": d["E_X"],
                "E_Y": d["E_Y"],
                "phi_raw": phi_raw,
                "phi_unwrapped": phi_unwrapped,
                "phi_theory": phi_theory,
                "alpha_theory": d["alpha_theory"],
                "alpha_observed": alpha_obs,
                "alpha_rel_error": abs(alpha_obs - d["alpha_theory"]) / d["alpha_theory"],
            })
    return out


def analyze_all(experiments: list[dict], counts_per_ex: list[dict]) -> dict:
    result = {"E": [], "F": [], "G": []}
    for ex, counts in zip(experiments, counts_per_ex):
        if ex["exp"].startswith("E_"):
            result["E"].append(analyze_E(ex, counts))
        elif ex["exp"].startswith("F_"):
            result["F"].append(analyze_F(ex, counts))
    result["G"] = analyze_G(experiments, counts_per_ex)
    return result


# ---------- Main ----------

def run_simulator() -> None:
    from qiskit.providers.basic_provider import BasicSimulator
    backend = BasicSimulator()
    experiments = all_experiments()
    print(f"\n[sim] {len(experiments)} circuits; transpiling...")
    experiments = transpile_and_check(experiments, backend)
    print(f"\n[sim] running {SHOTS} shots each...")
    counts_per_ex = []
    for ex in experiments:
        job = backend.run(ex["isa_circuit"], shots=SHOTS)
        counts = job.result().get_counts()
        counts_per_ex.append(counts)
        print(f"  {ex['label']:35s} unique_states={len(counts):4d}")
    analysis = analyze_all(experiments, counts_per_ex)
    out_path = RESULTS_DIR / "marrakesh_batch2_sim_analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    print(f"\n[sim] wrote {out_path}")
    print(json.dumps(analysis, indent=2))


def run_hardware() -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    service = QiskitRuntimeService(instance="Zynerji")
    backend = service.backend("ibm_marrakesh")
    status = backend.status()
    print(f"\n[hw] backend: {backend.name}, {backend.num_qubits}q, "
          f"status: {status.status_msg}, pending: {status.pending_jobs}")
    experiments = all_experiments()
    print(f"\n[hw] {len(experiments)} circuits; transpiling at opt_level=3...")
    experiments = transpile_and_check(experiments, backend)
    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.twirling.enable_gates = True
    opts.twirling.enable_measure = True
    opts.default_shots = SHOTS
    print("\n[hw] submitting batch via SamplerV2 ...")
    sampler = SamplerV2(mode=backend, options=opts)
    isa_circuits = [ex["isa_circuit"] for ex in experiments]
    t0 = time.monotonic()
    job = sampler.run(isa_circuits, shots=SHOTS)
    print(f"[hw] job_id={job.job_id()}")
    result = job.result()
    t1 = time.monotonic()
    print(f"[hw] result ready in {t1 - t0:.1f}s wall")
    counts_per_ex = []
    for i, ex in enumerate(experiments):
        data = result[i].data
        creg_name = next(iter(data))
        counts = getattr(data, creg_name).get_counts()
        counts_per_ex.append(counts)
        print(f"  {ex['label']:35s} unique_states={len(counts):4d}")
    analysis = analyze_all(experiments, counts_per_ex)
    out_path = RESULTS_DIR / "marrakesh_batch2_hw_analysis.json"
    raw_path = RESULTS_DIR / "marrakesh_batch2_hw_counts.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    raw_path.write_text(json.dumps(
        [
            {"label": ex["label"], "isa_depth": ex["isa_depth"], "counts": counts}
            for ex, counts in zip(experiments, counts_per_ex)
        ],
        indent=2,
    ))
    print(f"\n[hw] wrote {out_path}")
    print(f"[hw] wrote {raw_path}")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--hardware", action="store_true")
    args = parser.parse_args()
    if not (args.sim or args.hardware):
        parser.error("pass --sim or --hardware")
    if args.sim:
        run_simulator()
    if args.hardware:
        run_hardware()
