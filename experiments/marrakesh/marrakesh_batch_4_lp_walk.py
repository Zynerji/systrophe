"""Marrakesh batch #4: multi-band LP quantum walk + novelty catcher.

The breakthrough from batch 2 (KK escape, interference cancellation
exact at threshold) is extended to a chain of N=3 successive CTC bands
with phases LOG-PERIODICALLY scaled by the LP rotation parameter α.

Setup
-----
- 1 data qubit (the "particle phase register").
- N=3 path qubits, one per CTC band (records 4D-loop vs 5D-escape).
- For each band n=1,2,3: apply CRZ(path_n -> data, n * α * ln 2).
  The factor n encodes the LOG-PERIODIC structure: successive bands
  have radius ratio exp(π/α), so the cumulative log-coordinate at
  band n is n * π/α, and the phase contribution is α * (band's log
  coordinate increment) = α * π/α * n = n * π. Using α * ln 2
  instead of π keeps the angles non-degenerate.
- Initialize all 4 qubits in |+> -> superposition over all 2^3 path
  choices.
- Measure all 4 qubits.

Predicted bitstring distribution (theoretical via classical sim)
captures the path-conditional interference. Recovering α from the
distribution validates the LP framework operationally.

NOVELTY CATCHER (mandatory): the observed bitstring distribution is
hashed to bit addresses across a scan of α values (effective phase
scales); λ₂ on the resulting Hamming graph flags any unexpected
emergent structure.

Calibration: opt_level=3 + DD(XpXm) + gate/measure twirling.
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

from systrophe.catchers.novelty_catcher import (
    bitstring_counts_to_address,
    catch_novelty_in_distributions,
    probability_vector_to_address,
    scan_novelty,
    summarize_novelty_for_report,
)
from systrophe.geometry.vanstockum import VanStockumInterior

SHOTS = 8192
DEPTH_LIMIT = 200
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

vs = VanStockumInterior(omega=1.0, R=1.0)
ALPHA_LP = vs.alpha  # sqrt(3) ≈ 1.7321
LN2 = math.log(2.0)


def lp_walk_circuit(alpha_scale: float, n_bands: int = 3) -> QuantumCircuit:
    """Build the multi-band LP quantum walk circuit.

    Phase per band n: angle = n * alpha_scale * LN2.
    """
    n_qubits = 1 + n_bands  # data + path qubits
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qr, cr, name=f"lp_walk_alpha{alpha_scale:.4f}")
    # All in |+>
    for q in range(n_qubits):
        qc.h(qr[q])
    # data qubit = qr[0]; path qubits = qr[1..N]
    for n in range(1, n_bands + 1):
        angle = n * alpha_scale * LN2
        qc.crz(angle, qr[n], qr[0])
    # Hadamard on data for X-basis read-out structure
    qc.h(qr[0])
    qc.measure(qr, cr)
    return qc


def predicted_distribution(alpha_scale: float, n_bands: int = 3,
                            n_shots: int = SHOTS) -> dict[str, float]:
    """Classical simulation of the circuit, returning probability dict."""
    # State after all gates: enumerate over 2^N path values; for each,
    # compute the data qubit state and measurement probabilities.
    n_qubits = 1 + n_bands
    probs = {}
    for path in range(2 ** n_bands):
        # Path bitstring (q_N q_{N-1} ... q_1 in Qiskit's little-endian
        # ordering for measurement bitstrings)
        phase = 0.0
        path_bits = []
        for n in range(1, n_bands + 1):
            c_n = (path >> (n - 1)) & 1
            path_bits.append(c_n)
            if c_n == 1:
                phase += n * alpha_scale * LN2
        # data qubit was |+>, after RZ(phase) it's
        #   (e^(-i phase/2) |0> + e^(+i phase/2) |1>)/sqrt(2)
        # After H: (cos(phase/2) |0> - i sin(phase/2) |1>)
        # Pr(data = 0) = cos^2(phase/2), Pr(data = 1) = sin^2(phase/2)
        p_data_0 = math.cos(phase / 2) ** 2
        p_data_1 = math.sin(phase / 2) ** 2
        # Joint prob of path = (1/2^N) (since each path qubit was |+>
        # measured in Z, uniform), multiplied by p_data
        prob_path = 1.0 / (2 ** n_bands)
        # Bitstring in Qiskit measurement order: cr[0] cr[1] ... cr[N]
        # qc.measure(qr, cr) maps qr[i] -> cr[i]. Printed bitstring is
        # c[N] c[N-1] ... c[0] (left-to-right = high-index to low-index).
        # So for n_qubits=4 (1 data + 3 path), bitstring = c3 c2 c1 c0
        # where c0 = data, c1 = path_1, c2 = path_2, c3 = path_3.
        for d in [0, 1]:
            bits = [str(b) for b in path_bits]  # path_1, path_2, path_3
            # Construct bitstring: c3 c2 c1 c0
            bitstring = "".join(reversed(bits)) + str(d)
            p = prob_path * (p_data_0 if d == 0 else p_data_1)
            probs[bitstring] = probs.get(bitstring, 0.0) + p
    return probs


def transpile_and_check(experiments: list[dict], backend) -> list[dict]:
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    for ex in experiments:
        qc = ex["circuit"]
        isa_qc = pm.run(qc)
        d = isa_qc.depth()
        n_2q = sum(1 for instr in isa_qc.data if len(instr.qubits) == 2)
        print(f"  {ex['label']:30s} pre_depth={qc.depth():3d}  "
              f"isa_depth={d:4d}  n_2q={n_2q}", flush=True)
        assert d < DEPTH_LIMIT, f"{ex['label']} depth {d} >= {DEPTH_LIMIT}"
        ex["isa_circuit"] = isa_qc
        ex["isa_depth"] = d
    return experiments


def analyze_bitstring_distribution(
    counts: dict[str, int], predicted: dict[str, float],
) -> dict:
    total = sum(counts.values())
    observed = {k: v / total for k, v in counts.items()}
    # Total variation distance
    all_keys = set(observed) | set(predicted)
    tv = 0.5 * sum(abs(observed.get(k, 0.0) - predicted.get(k, 0.0))
                    for k in all_keys)
    return {
        "total_variation_distance": float(tv),
        "n_unique_states_observed": int(len(counts)),
        "top5_observed": dict(sorted(observed.items(),
                                       key=lambda x: -x[1])[:5]),
        "top5_predicted": dict(sorted(predicted.items(),
                                        key=lambda x: -x[1])[:5]),
    }


def extract_alpha_from_distribution(
    observed_counts: dict[str, int],
    alpha_grid: np.ndarray = None,
) -> dict:
    """Find alpha that minimizes TV distance between observed and predicted."""
    if alpha_grid is None:
        alpha_grid = np.linspace(0.5, 3.0, 100)
    total = sum(observed_counts.values())
    observed = {k: v / total for k, v in observed_counts.items()}
    best_alpha = None
    best_tv = float("inf")
    for a in alpha_grid:
        pred = predicted_distribution(float(a))
        all_keys = set(observed) | set(pred)
        tv = 0.5 * sum(abs(observed.get(k, 0.0) - pred.get(k, 0.0))
                        for k in all_keys)
        if tv < best_tv:
            best_tv = tv
            best_alpha = float(a)
    return {
        "alpha_recovered": best_alpha,
        "alpha_theoretical": ALPHA_LP,
        "alpha_relative_error": abs(best_alpha - ALPHA_LP) / ALPHA_LP,
        "best_TV_distance": best_tv,
    }


def all_experiments() -> list[dict]:
    """Five experiments scanning over alpha-scale: 0.5, 1.0, sqrt(3), 2.0, 2.5."""
    alphas = [0.5, 1.0, ALPHA_LP, 2.0, 2.5]
    out = []
    for a in alphas:
        ex = {
            "exp": "lp_walk",
            "label": f"lp_walk_alpha{a:.4f}",
            "alpha_scale": float(a),
            "circuit": lp_walk_circuit(a),
            "predicted_distribution": predicted_distribution(a),
        }
        out.append(ex)
    return out


def run_simulator() -> None:
    from qiskit.providers.basic_provider import BasicSimulator
    backend = BasicSimulator()
    experiments = all_experiments()
    print(f"\n[sim] {len(experiments)} circuits; transpiling...")
    experiments = transpile_and_check(experiments, backend)
    print(f"\n[sim] running {SHOTS} shots each...")
    all_distributions = []
    summaries = []
    for ex in experiments:
        job = backend.run(ex["isa_circuit"], shots=SHOTS)
        counts = job.result().get_counts()
        analysis = analyze_bitstring_distribution(counts, ex["predicted_distribution"])
        alpha_rec = extract_alpha_from_distribution(counts)
        summaries.append({
            "label": ex["label"],
            "alpha_scale": ex["alpha_scale"],
            **analysis,
            **alpha_rec,
        })
        # Build dense probability vector for novelty catcher
        all_keys = sorted(set(counts.keys()) | set(ex["predicted_distribution"].keys()))
        total = sum(counts.values())
        prob_vec = np.array([counts.get(k, 0) / total for k in all_keys])
        all_distributions.append(prob_vec)
        print(f"  {ex['label']:30s} TV={analysis['total_variation_distance']:.4f}  "
              f"alpha_rec={alpha_rec['alpha_recovered']:.3f} "
              f"(theory={ALPHA_LP:.3f})")
    # Novelty catcher on the distributions
    labels = [s["label"] for s in summaries]
    novelty = catch_novelty_in_distributions(all_distributions,
                                                labels=labels, n_bits=32)
    out_path = RESULTS_DIR / "marrakesh_batch4_sim_analysis.json"
    out_path.write_text(json.dumps({
        "per_circuit": summaries,
        "novelty_catcher": novelty,
    }, indent=2))
    print(f"\n[sim] novelty verdict: {novelty['verdict']}")
    print(f"[sim] sharp features: {len(novelty['sharp_features'])}")
    print(f"[sim] wrote {out_path}")


def run_hardware() -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    service = QiskitRuntimeService(instance="Zynerji")
    backend = service.backend("ibm_marrakesh")
    status = backend.status()
    print(f"\n[hw] backend: {backend.name}, status: {status.status_msg}, "
          f"pending: {status.pending_jobs}")
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
    t0 = time.monotonic()
    job = sampler.run(isa_circuits, shots=SHOTS)
    print(f"[hw] job_id={job.job_id()}")
    result = job.result()
    t1 = time.monotonic()
    print(f"[hw] result ready in {t1 - t0:.1f}s wall")
    all_distributions = []
    summaries = []
    raw_records = []
    for i, ex in enumerate(experiments):
        data = result[i].data
        creg_name = next(iter(data))
        counts = getattr(data, creg_name).get_counts()
        analysis = analyze_bitstring_distribution(counts, ex["predicted_distribution"])
        alpha_rec = extract_alpha_from_distribution(counts)
        summaries.append({
            "label": ex["label"],
            "alpha_scale": ex["alpha_scale"],
            **analysis,
            **alpha_rec,
        })
        all_keys = sorted(set(counts.keys()) | set(ex["predicted_distribution"].keys()))
        total = sum(counts.values())
        prob_vec = np.array([counts.get(k, 0) / total for k in all_keys])
        all_distributions.append(prob_vec)
        raw_records.append({"label": ex["label"], "isa_depth": ex["isa_depth"],
                             "counts": counts})
        print(f"  {ex['label']:30s} TV={analysis['total_variation_distance']:.4f}  "
              f"alpha_rec={alpha_rec['alpha_recovered']:.3f} "
              f"(theory={ALPHA_LP:.3f})")
    # Novelty catcher
    labels = [s["label"] for s in summaries]
    novelty = catch_novelty_in_distributions(all_distributions,
                                                labels=labels, n_bits=32)
    out_path = RESULTS_DIR / "marrakesh_batch4_hw_analysis.json"
    raw_path = RESULTS_DIR / "marrakesh_batch4_hw_counts.json"
    out_path.write_text(json.dumps({
        "per_circuit": summaries,
        "novelty_catcher": novelty,
    }, indent=2))
    raw_path.write_text(json.dumps(raw_records, indent=2))
    print(f"\n[hw] novelty verdict: {novelty['verdict']}")
    print(f"[hw] sharp features: {len(novelty['sharp_features'])}")
    if novelty["sharp_features"]:
        print("[hw] sharp feature details:")
        for sf in novelty["sharp_features"]:
            print(f"    {sf}")
    print(f"[hw] wrote {out_path}")


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
