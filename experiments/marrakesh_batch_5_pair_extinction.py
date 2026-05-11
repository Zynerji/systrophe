"""Marrakesh batch #5: Tipler-pair anti-phase extinction.

Theory predicts the SystrophePair CTC bands extinguish exactly at the
anti-phase offset δ=π (topological off-switch). Batch 4 explored a
single-cylinder log-periodic walk; this batch probes whether the
phase-offset machinery survives on hardware.

Setup
-----
- 1 data qubit (the particle-phase register).
- 3 path qubits, one per CTC band (records the band traversed).
- Initialise all 4 qubits in |+> -> superposition over all 2^3 band
  configurations.
- For each band n = 1, 2, 3: apply CRZ(path_n -> data, n · α · LN2)
  representing the phase accumulated through cylinder 1.
- For each band n = 1, 2, 3: apply CRZ(path_n -> data, n · α · LN2 + δ)
  representing the phase through cylinder 2 (offset by δ), implemented
  via X(path_n) ; CRZ ; X(path_n) so the phase is applied on the
  path=0 (cylinder-2 selected) branch.
- Final H on data; measure all 4 qubits.

A 4-qubit measurement has 16 unique bitstrings, giving the catcher
enough support to resolve probability shifts; a 2-qubit version was
proven insufficient (sim verdict: smooth across the extinction).

Seven circuits in the sweep, non-uniformly spaced to double-bracket
the extinction angle:
    δ ∈ {0, π/2, 7π/8, π, 9π/8, 3π/2, 2π}.

Predicted distribution (Mach-Zehnder style)
-------------------------------------------
At δ=0 (and δ=2π) the two branches add coherently → data biased.
At δ=π the two branches cancel exactly → data 50/50.

The novelty catcher MUST flag the 7π/8→π→9π/8 bracket as a sharp pair
of Hamming-graph transitions (one in each direction around the
extinction). If it does not, either the hardware (1) cannot resolve
the extinction angle, or (2) we have a real emergent that defies the
analytic prediction.

A uniform 5-point sweep over [0, π] cannot resolve the extinction:
the sim verdict on that geometry was `smooth, 0 sharp` even though
P(data=1) hit 0.5007 at δ=π, because every adjacent step was a
similar smooth bias shift. The asymmetric bracket fixes this.

NOVELTY CATCHER (mandatory): every bitstring distribution is hashed to
address space and tested for sharp Hamming steps along the δ sweep.

Calibration: opt_level=3 + DD(XpXm) + gate/measure twirling, 8192
shots.
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

from systrophe.novelty_catcher import (
    bitstring_counts_to_address,
    catch_novelty_in_distributions,
)
from systrophe.vanstockum import VanStockumInterior

SHOTS = 8192
DEPTH_LIMIT = 200
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

vs = VanStockumInterior(omega=1.0, R=1.0)
ALPHA_LP = vs.alpha  # sqrt(3) for a=1, R=1
LN2 = math.log(2.0)
N_BANDS = 3


def pair_extinction_circuit(delta: float) -> QuantumCircuit:
    """4-qubit Mach-Zehnder analog of the SystrophePair phase offset.

    qr[0] is data; qr[1..N_BANDS] are path qubits. After H on each,
    we are in a uniform superposition. For each band n we apply two
    CRZ gates on the data qubit, gated by path_n :
      cylinder 1: phase  n · α · LN2  when path_n = 1
      cylinder 2: phase  n · α · LN2 + δ when path_n = 0
                  (implemented via X(path_n) ; CRZ ; X(path_n)).
    A final H on data and Z-measurement of all 4 qubits.

    At δ = 0 cylinders are in phase, distribution has the same shape as
    batch 4's single-cylinder walk. At δ = π the two contributions
    cancel band-wise → the data-qubit phase is independent of the path
    register → joint distribution is uniform over the 8 path bitstrings
    × 50/50 data → 1/16 mass on each 4-bit outcome. The catcher should
    see this collapse to uniformity as a sharp address-space jump.
    """
    n_qubits = 1 + N_BANDS
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qr, cr, name=f"pair_delta{delta:.4f}")
    for q in range(n_qubits):
        qc.h(qr[q])
    for n in range(1, N_BANDS + 1):
        phi1 = n * ALPHA_LP * LN2
        phi2 = n * ALPHA_LP * LN2 + delta
        # cylinder 1 (path_n = 1) :
        qc.crz(phi1, qr[n], qr[0])
        # cylinder 2 (path_n = 0) via X-sandwich:
        qc.x(qr[n])
        qc.crz(phi2, qr[n], qr[0])
        qc.x(qr[n])
    qc.h(qr[0])
    qc.measure(qr, cr)
    return qc


def predicted_distribution(delta: float, n_bands: int = N_BANDS) -> dict[str, float]:
    """Classical simulation of the 4-qubit pair-extinction circuit.

    Enumerates over all 2^N path bitstrings; for each, computes the
    total accumulated phase on the data qubit, then the post-H Z-basis
    probabilities. Returns a dict keyed by Qiskit measurement bitstring
    `c{N} c{N-1} ... c1 c0` where c0 is the data bit.
    """
    n_qubits = 1 + n_bands
    probs: dict[str, float] = {}
    for path in range(2 ** n_bands):
        path_bits = [(path >> (n - 1)) & 1 for n in range(1, n_bands + 1)]
        phase = 0.0
        for n in range(1, n_bands + 1):
            phi1 = n * ALPHA_LP * LN2
            phi2 = n * ALPHA_LP * LN2 + delta
            phase += phi1 if path_bits[n - 1] == 1 else phi2
        p_data_0 = math.cos(phase / 2) ** 2
        p_data_1 = math.sin(phase / 2) ** 2
        prob_path = 1.0 / (2 ** n_bands)
        for d in (0, 1):
            # Bitstring: c{N} c{N-1} ... c1 c0
            bs = "".join(str(b) for b in reversed(path_bits)) + str(d)
            probs[bs] = probs.get(bs, 0.0) + prob_path * (p_data_0 if d == 0 else p_data_1)
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


def all_experiments() -> list[dict]:
    # Non-uniform sweep concentrating around the predicted extinction at
    # δ=π. The bracket {7π/8, π, 9π/8} guarantees the catcher sees the
    # sharp on both sides of the antipodal point; the wings {0, π/2,
    # 3π/2, 2π} establish the baseline distribution shape.
    deltas = [
        0.0,
        math.pi / 2,
        7 * math.pi / 8,
        math.pi,
        9 * math.pi / 8,
        3 * math.pi / 2,
        2 * math.pi,
    ]
    out = []
    for d in deltas:
        out.append({
            "exp": "pair_extinction",
            "label": f"pair_delta{d:.4f}",
            "delta": float(d),
            "circuit": pair_extinction_circuit(d),
            "predicted_distribution": predicted_distribution(d),
        })
    return out


def _tv(observed: dict[str, float], predicted: dict[str, float]) -> float:
    keys = set(observed) | set(predicted)
    return 0.5 * sum(abs(observed.get(k, 0.0) - predicted.get(k, 0.0))
                      for k in keys)


def run_simulator() -> None:
    from qiskit.providers.basic_provider import BasicSimulator
    backend = BasicSimulator()
    experiments = all_experiments()
    print(f"\n[sim] {len(experiments)} circuits", flush=True)
    experiments = transpile_and_check(experiments, backend)
    summaries = []
    distributions = []
    for ex in experiments:
        job = backend.run(ex["isa_circuit"], shots=SHOTS)
        counts = job.result().get_counts()
        total = sum(counts.values())
        observed = {k: v / total for k, v in counts.items()}
        tv = _tv(observed, ex["predicted_distribution"])
        # P(data=1) sums over both paths
        p_data1 = sum(v for k, v in observed.items() if k[-1] == "1")
        summaries.append({
            "label": ex["label"], "delta": ex["delta"],
            "tv_obs_vs_pred": tv,
            "P_data1_observed": float(p_data1),
            "P_data1_predicted": sum(v for k, v in ex["predicted_distribution"].items()
                                       if k[-1] == "1"),
        })
        # Build dense probability vector for catcher input
        all_keys = sorted(set(counts.keys()) | set(ex["predicted_distribution"].keys()))
        prob_vec = np.array([counts.get(k, 0) / total for k in all_keys])
        distributions.append(prob_vec)
        print(f"  {ex['label']:25s} TV={tv:.4f}  P(data=1)={p_data1:.4f}",
              flush=True)

    labels = [s["label"] for s in summaries]
    novelty = catch_novelty_in_distributions(distributions, labels=labels)
    out_path = RESULTS_DIR / "marrakesh_batch5_sim_analysis.json"
    out_path.write_text(json.dumps({
        "per_circuit": summaries,
        "novelty_catcher": novelty,
    }, indent=2))
    print(f"\n[sim] novelty verdict: {novelty['verdict']}, "
          f"sharp_features={len(novelty['sharp_features'])}", flush=True)
    for sf in novelty["sharp_features"]:
        print(f"    sharp: {sf.get('between')}  step={sf.get('hamming_step')}",
              flush=True)
    print(f"[sim] wrote {out_path}", flush=True)


def run_hardware() -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    service = QiskitRuntimeService(instance="Zynerji")
    backend = service.backend("ibm_marrakesh")
    status = backend.status()
    print(f"\n[hw] backend: {backend.name}, status: {status.status_msg}, "
          f"pending: {status.pending_jobs}", flush=True)
    experiments = all_experiments()
    print(f"\n[hw] {len(experiments)} circuits; transpiling at opt_level=3...",
          flush=True)
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
    print(f"[hw] job_id={job.job_id()}", flush=True)
    result = job.result()
    t1 = time.monotonic()
    print(f"[hw] result ready in {t1 - t0:.1f}s wall", flush=True)
    summaries = []
    raw = []
    distributions = []
    for i, ex in enumerate(experiments):
        data = result[i].data
        creg_name = next(iter(data))
        counts = getattr(data, creg_name).get_counts()
        total = sum(counts.values())
        observed = {k: v / total for k, v in counts.items()}
        tv = _tv(observed, ex["predicted_distribution"])
        p_data1 = sum(v for k, v in observed.items() if k[-1] == "1")
        summaries.append({
            "label": ex["label"], "delta": ex["delta"],
            "tv_obs_vs_pred": tv,
            "P_data1_observed": float(p_data1),
            "P_data1_predicted": sum(v for k, v in ex["predicted_distribution"].items()
                                       if k[-1] == "1"),
        })
        all_keys = sorted(set(counts.keys()) | set(ex["predicted_distribution"].keys()))
        prob_vec = np.array([counts.get(k, 0) / total for k in all_keys])
        distributions.append(prob_vec)
        raw.append({"label": ex["label"], "counts": counts,
                     "isa_depth": ex["isa_depth"]})
        print(f"  {ex['label']:25s} TV={tv:.4f}  P(data=1)={p_data1:.4f}",
              flush=True)

    labels = [s["label"] for s in summaries]
    novelty = catch_novelty_in_distributions(distributions, labels=labels)
    out_path = RESULTS_DIR / "marrakesh_batch5_hw_analysis.json"
    raw_path = RESULTS_DIR / "marrakesh_batch5_hw_counts.json"
    out_path.write_text(json.dumps({
        "per_circuit": summaries,
        "novelty_catcher": novelty,
        "submitted_job_id": job.job_id(),
    }, indent=2))
    raw_path.write_text(json.dumps(raw, indent=2))
    print(f"\n[hw] novelty verdict: {novelty['verdict']}, "
          f"sharp_features={len(novelty['sharp_features'])}", flush=True)
    for sf in novelty["sharp_features"]:
        print(f"    sharp: {sf.get('between')}  step={sf.get('hamming_step')}",
              flush=True)
    print(f"[hw] wrote {out_path}", flush=True)
    print(f"[hw] wrote {raw_path}", flush=True)


def submit_only() -> None:
    """Submit the job to IBM without waiting for the result.

    Prints the job_id and exits so the queue accrues while other work
    continues. Use experiments/recover_batch5_hw.py to pull the result
    once status flips to DONE.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    service = QiskitRuntimeService(instance="Zynerji")
    backend = service.backend("ibm_marrakesh")
    status = backend.status()
    print(f"\n[submit] backend={backend.name}  pending={status.pending_jobs}",
          flush=True)
    experiments = all_experiments()
    experiments = transpile_and_check(experiments, backend)
    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.twirling.enable_gates = True
    opts.twirling.enable_measure = True
    opts.default_shots = SHOTS
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run([ex["isa_circuit"] for ex in experiments], shots=SHOTS)
    submitted = {
        "job_id": job.job_id(),
        "backend": backend.name,
        "n_circuits": len(experiments),
        "deltas": [ex["delta"] for ex in experiments],
        "submitted_unix": int(time.time()),
    }
    out = RESULTS_DIR / "marrakesh_batch5_submitted.json"
    out.write_text(json.dumps(submitted, indent=2))
    print(f"[submit] job_id={job.job_id()}  wrote {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--hardware", action="store_true",
                        help="Submit and block until result.")
    parser.add_argument("--submit-only", action="store_true",
                        help="Submit to IBM, print job_id, exit.")
    args = parser.parse_args()
    if not (args.sim or args.hardware or args.submit_only):
        parser.error("pass --sim, --hardware, or --submit-only")
    if args.sim:
        run_simulator()
    if args.submit_only:
        submit_only()
    if args.hardware:
        run_hardware()
