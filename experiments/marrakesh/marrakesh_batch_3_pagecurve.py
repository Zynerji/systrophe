"""Marrakesh batch #3: Page curve via Bell-pair 'evaporation'.

Setup
-----
State psi on 6 qubits: 3 Bell pairs across (q0, q3), (q1, q4), (q2, q5).
Subsystem A = first k qubits (k = 1..5). Trace out A^c and measure
Renyi-2 entropy S_2 = -log Tr(rho_A^2) via SWAP test on two copies.

Theoretical predictions (Page curve from Bell-pair counting):
- k=1: ρ_A = I/2,  Tr(ρ²) = 1/2,  S_2 = 1 bit,  P(anc=0) = 0.750
- k=2: ρ_A = I/4,  Tr(ρ²) = 1/4,  S_2 = 2 bits, P(anc=0) = 0.625
- k=3: ρ_A = I/8,  Tr(ρ²) = 1/8,  S_2 = 3 bits, P(anc=0) = 0.5625
- k=4: ρ_A = I/4 (one Bell pair "internal"), S_2 = 2 bits, P(anc=0) = 0.625
- k=5: ρ_A = I/2 (two pairs internal),       S_2 = 1 bit,  P(anc=0) = 0.750

Page curve shape: monotone rise 1 -> 3 then monotone fall 3 -> 1.

Calibration: opt_level=3 + DD(XpXm) + gate/measure twirling.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

SHOTS = 8192  # higher shots for tighter purity estimate
DEPTH_LIMIT = 200  # bumped from 150 for 13-qubit Page curve with CSWAPs;
                   # k=5 hits ~165, still usable signal on Marrakesh
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def prepare_three_bell_pairs(qc: QuantumCircuit, q0: int, q1: int, q2: int,
                              q3: int, q4: int, q5: int) -> None:
    """Three Bell pairs: (q0,q3), (q1,q4), (q2,q5).

    Pair (a, b) = (|00> + |11>)/sqrt(2): H(a), CX(a, b).
    """
    qc.h(q0)
    qc.cx(q0, q3)
    qc.h(q1)
    qc.cx(q1, q4)
    qc.h(q2)
    qc.cx(q2, q5)


def page_curve_circuit(k: int) -> QuantumCircuit:
    """Build SWAP-test circuit for subsystem size k (1..5).

    Layout:
      q0..q5   = copy A of 6-qubit state
      q6..q11  = copy B of 6-qubit state
      q12      = ancilla for SWAP test

    SWAP test: H(anc), CSWAP(anc, A_i, B_i) for i = 0..k-1, H(anc).
    P(anc = 0) = (1 + Tr(rho_subA^2)) / 2.
    """
    if not 1 <= k <= 5:
        raise ValueError("k must be in [1, 5]")
    qr = QuantumRegister(13, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr, name=f"pagecurve_k{k}")
    # Copy A
    prepare_three_bell_pairs(qc, 0, 1, 2, 3, 4, 5)
    # Copy B (independent)
    prepare_three_bell_pairs(qc, 6, 7, 8, 9, 10, 11)
    anc = 12
    qc.h(anc)
    for i in range(k):
        qc.cswap(anc, i, 6 + i)
    qc.h(anc)
    qc.measure(anc, cr[0])
    return qc


PREDICTED_P0 = {1: 0.750, 2: 0.625, 3: 0.5625, 4: 0.625, 5: 0.750}
PREDICTED_S2 = {1: 1.0, 2: 2.0, 3: 3.0, 4: 2.0, 5: 1.0}


def all_experiments() -> list[dict]:
    out = []
    for k in [1, 2, 3, 4, 5]:
        out.append({
            "exp": "H_page_curve",
            "label": f"pagecurve_k{k}",
            "k": k,
            "circuit": page_curve_circuit(k),
            "predicted_p_anc_0": PREDICTED_P0[k],
            "predicted_S2_bits": PREDICTED_S2[k],
        })
    return out


def transpile_and_check(experiments: list[dict], backend) -> list[dict]:
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    for ex in experiments:
        qc = ex["circuit"]
        isa_qc = pm.run(qc)
        d = isa_qc.depth()
        print(f"  {ex['label']:25s} pre_depth={qc.depth():3d}  "
              f"isa_depth={d:4d}  n_2q={sum(1 for instr in isa_qc.data if len(instr.qubits) == 2)}",
              flush=True)
        assert d < DEPTH_LIMIT, (
            f"{ex['label']} ISA depth {d} >= {DEPTH_LIMIT}"
        )
        ex["isa_circuit"] = isa_qc
        ex["isa_depth"] = d
    return experiments


def analyze(ex: dict, counts: dict) -> dict:
    total = sum(counts.values())
    p0 = counts.get("0", 0) / total
    p1 = counts.get("1", 0) / total
    # Renyi-2: -log_2(2*p0 - 1)
    purity = 2 * p0 - 1
    if purity <= 0:
        S2_obs = float("inf")
    else:
        S2_obs = -math.log2(purity)
    return {
        "label": ex["label"],
        "k": ex["k"],
        "predicted_p_anc_0": ex["predicted_p_anc_0"],
        "observed_p_anc_0": p0,
        "predicted_S2": ex["predicted_S2_bits"],
        "observed_S2": S2_obs,
        "p_anc_0_error": abs(p0 - ex["predicted_p_anc_0"]),
    }


def run_simulator() -> None:
    from qiskit.providers.basic_provider import BasicSimulator
    backend = BasicSimulator()
    experiments = all_experiments()
    print(f"\n[sim] {len(experiments)} circuits; transpiling...")
    experiments = transpile_and_check(experiments, backend)
    print(f"\n[sim] running {SHOTS} shots each...")
    results = []
    for ex in experiments:
        job = backend.run(ex["isa_circuit"], shots=SHOTS)
        counts = job.result().get_counts()
        res = analyze(ex, counts)
        results.append(res)
        print(f"  {ex['label']:25s} observed_p0={res['observed_p_anc_0']:.4f}  "
              f"S2_obs={res['observed_S2']:.3f}  pred={res['predicted_S2']}")
    out_path = RESULTS_DIR / "marrakesh_batch3_sim_analysis.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[sim] wrote {out_path}")


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
    sampler = SamplerV2(mode=backend, options=opts)
    isa_circuits = [ex["isa_circuit"] for ex in experiments]
    print("\n[hw] submitting batch ...")
    t0 = time.monotonic()
    job = sampler.run(isa_circuits, shots=SHOTS)
    print(f"[hw] job_id={job.job_id()}")
    result = job.result()
    t1 = time.monotonic()
    print(f"[hw] result ready in {t1 - t0:.1f}s wall")
    results = []
    counts_records = []
    for i, ex in enumerate(experiments):
        data = result[i].data
        creg_name = next(iter(data))
        counts = getattr(data, creg_name).get_counts()
        res = analyze(ex, counts)
        results.append(res)
        counts_records.append({
            "label": ex["label"],
            "isa_depth": ex["isa_depth"],
            "counts": counts,
        })
        print(f"  {ex['label']:25s} observed_p0={res['observed_p_anc_0']:.4f}  "
              f"S2_obs={res['observed_S2']:.3f}  pred={res['predicted_S2']}")
    out_path = RESULTS_DIR / "marrakesh_batch3_hw_analysis.json"
    raw_path = RESULTS_DIR / "marrakesh_batch3_hw_counts.json"
    out_path.write_text(json.dumps(results, indent=2))
    raw_path.write_text(json.dumps(counts_records, indent=2))
    print(f"\n[hw] wrote {out_path}")
    print(f"[hw] wrote {raw_path}")
    print(json.dumps(results, indent=2))


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
