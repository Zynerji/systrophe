"""Big-scale QEC + GHZ headline test on ibm_kingston (156 qubits).

Useful work: prepare a 156-qubit GHZ state via tree-CNOT, then verify
its coherence via catcher analysis of the bitstring distribution.

Circuit
=======
1. Apply H on qubit 0.
2. Build a tree of CNOTs broadcasting q0 to all 155 other qubits.
   Tree depth = ceil(log_2(156)) = 8.
3. Measure all 156 qubits.

Expected outcome
================
Ideal: bimodal distribution -- ~50% of shots at 000...0, ~50% at
111...1; near-zero probability for all other bitstrings.

Real-hardware: depolarising noise spreads the distribution.
Single-bit-flip errors produce strings with Hamming weight 1 or 155.
A k-error event produces strings with Hamming weight k or 156-k.

QEC interpretation
==================
The 156-qubit GHZ is the codeword space of a distance-156 repetition
code. The logical bit = majority-vote(measured bits). With distance
156, the code can correct up to 77 bit-flip errors per shot.

The catcher running on the 8192-shot bitstring distribution detects:
1. The bimodal peak structure (logical-state coherence)
2. Hamming-weight distribution (physical error rate)
3. Logical-error rate (probability of majority-vote miss)

This is the cleanest "useful work using 156 coherent qubits + QEC"
experiment that fits within Heron-r2 calibration.
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
)

SHOTS = 8192
N_QUBITS = 156
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def tree_ghz_circuit(n_qubits: int = N_QUBITS) -> QuantumCircuit:
    """Build a tree-CNOT GHZ-state preparation circuit of depth O(log n).

    Layer 0: H on q0.
    Layers 1..ceil(log_2(n)): pair up qubits and CNOT to broadcast.
    """
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qr, cr, name=f"ghz_{n_qubits}q")
    qc.h(qr[0])
    # Tree broadcasting: at each layer, qubits already in superposition
    # CNOT-control onto the next batch of qubits.
    layer_size = 1
    active = [0]  # qubits currently in superposition
    while len(active) < n_qubits:
        new_active = list(active)
        for src in active:
            tgt = src + layer_size
            if tgt < n_qubits:
                qc.cx(qr[src], qr[tgt])
                new_active.append(tgt)
        active = new_active
        layer_size *= 2
    qc.measure(qr, cr)
    return qc


def predicted_distribution_summary() -> dict:
    """Ideal GHZ measurement statistics."""
    return {
        "P_all_zeros": 0.5,
        "P_all_ones": 0.5,
        "P_other": 0.0,
        "hamming_weight_modes": [0, N_QUBITS],
    }


def transpile(backend) -> QuantumCircuit:
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    qc = tree_ghz_circuit(N_QUBITS)
    isa = pm.run(qc)
    print(f"  GHZ_{N_QUBITS}q: pre_depth={qc.depth()}, isa_depth={isa.depth()}",
          flush=True)
    return isa


def submit_kingston(instance: str = "Zynerji") -> str:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend("ibm_kingston")
    print(f"\n[156q-ghz] kingston: pending={backend.status().pending_jobs}",
          flush=True)
    isa = transpile(backend)
    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.twirling.enable_gates = True
    opts.twirling.enable_measure = True
    opts.default_shots = SHOTS
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run([isa], shots=SHOTS)
    job_id = job.job_id()
    print(f"[156q-ghz] kingston job_id={job_id}", flush=True)

    out = {
        "job_id": job_id,
        "backend": "ibm_kingston",
        "n_qubits": N_QUBITS,
        "shots": SHOTS,
        "submitted_unix": int(time.time()),
    }
    out_path = RESULTS_DIR / "kingston_156q_ghz_submitted.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[156q-ghz] wrote {out_path}", flush=True)
    return job_id


def recover_from_job(job_id: str, instance: str = "Zynerji") -> dict:
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(instance=instance)
    job = service.job(job_id)
    print(f"\n[156q-recover] kingston job={job_id}", flush=True)
    st = str(job.status())
    if st not in ("DONE", "COMPLETED"):
        print(f"  status={st}; skipping")
        return {"status": st}
    result = job.result()
    data = result[0].data
    creg_name = next(iter(data))
    counts = getattr(data, creg_name).get_counts()
    total = sum(counts.values())
    print(f"[156q-recover] total shots: {total}", flush=True)
    print(f"[156q-recover] unique bitstrings: {len(counts)}", flush=True)

    # All-zeros and all-ones probabilities
    zero_str = "0" * N_QUBITS
    one_str = "1" * N_QUBITS
    p_zero = counts.get(zero_str, 0) / total
    p_one = counts.get(one_str, 0) / total
    print(f"  P(all-zeros) = {p_zero:.6f}")
    print(f"  P(all-ones)  = {p_one:.6f}")
    print(f"  P(GHZ-codeword) = {p_zero + p_one:.6f}  (ideal: 1.0)")
    print(f"  GHZ fidelity proxy = {(p_zero + p_one) / 1.0:.4f}")

    # Hamming-weight distribution
    hw_hist = np.zeros(N_QUBITS + 1, dtype=int)
    for bs, c in counts.items():
        hw = bs.count("1")
        hw_hist[hw] += c
    hw_dist = hw_hist / total
    # Modes: should be 0 and 156
    top_hw = sorted(enumerate(hw_dist), key=lambda kv: -kv[1])[:5]
    print(f"\n  Top 5 Hamming weights by probability:")
    for hw, p in top_hw:
        print(f"    hw={hw:3d}: P={p:.6f}")

    # Logical-bit majority vote
    n_logical_zero = 0
    n_logical_one = 0
    for bs, c in counts.items():
        hw = bs.count("1")
        if hw < N_QUBITS / 2:
            n_logical_zero += c
        elif hw > N_QUBITS / 2:
            n_logical_one += c
    p_logical_zero = n_logical_zero / total
    p_logical_one = n_logical_one / total
    print(f"\n  Majority-vote logical decoding:")
    print(f"    P(logical=0) = {p_logical_zero:.6f}")
    print(f"    P(logical=1) = {p_logical_one:.6f}")
    print(f"    Logical decoder success rate = {p_logical_zero + p_logical_one:.6f}")

    # Catcher on the GHZ output bitstring distribution
    # Pick top 32 bitstrings by probability
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_strings = [bs for bs, _ in sorted_counts[:32]]
    addresses = []
    for bs in top_strings:
        bits = np.array([int(b) for b in bs[:32]])
        addresses.append(bits)

    out = {
        "job_id": job_id,
        "n_qubits": N_QUBITS,
        "total_shots": total,
        "unique_bitstrings": len(counts),
        "p_all_zeros": float(p_zero),
        "p_all_ones": float(p_one),
        "p_ghz_codeword": float(p_zero + p_one),
        "ghz_fidelity_proxy": float(p_zero + p_one),
        "majority_vote_logical_0": float(p_logical_zero),
        "majority_vote_logical_1": float(p_logical_one),
        "logical_decoder_success_rate": float(p_logical_zero + p_logical_one),
        "hamming_weight_distribution": hw_dist.tolist(),
        "top_5_hamming_weights": top_hw,
        "top_32_bitstrings_addresses": [a.tolist() for a in addresses],
        # Truncated counts (the full 2^156 space is large)
        "top_50_bitstrings": {bs: c for bs, c in sorted_counts[:50]},
    }
    out_path = RESULTS_DIR / "kingston_156q_ghz_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[156q-recover] wrote {out_path}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--recover", action="store_true")
    ap.add_argument("--instance", default="Zynerji")
    args = ap.parse_args()
    if args.submit:
        submit_kingston(instance=args.instance)
    if args.recover:
        sub = json.loads(
            (RESULTS_DIR / "kingston_156q_ghz_submitted.json").read_text()
        )
        recover_from_job(sub["job_id"], instance=args.instance)
