"""Full X+Z syndrome rotated surface code memory experiment.

Adds X-stabilizer extraction to the Z-memory experiment. Each round
measures BOTH X-stabilizers (which detect Z errors on data qubits)
and Z-stabilizers (which detect X errors). With both syndromes
available, the decoder can correct both bit-flip AND phase-flip
errors -- the canonical fault-tolerant quantum memory experiment.

For Z-MEMORY (logical |0_L> initial state, logical Z final measurement),
the X-syndrome data lets the decoder correct phase errors that
propagate into Z observables during the syndrome cycles. This should
incrementally improve the logical-Z recovery over the Z-only version.

For a true quantum memory test (logical superposition), we would
prepare |+_L> and measure logical X, but this requires a more complex
preparation circuit. We start with the Z-memory + X-syndromes
extension here.

d=5: 25 data + 12 X-anc + 12 Z-anc = 49 qubits
d=7: 49 data + 24 X-anc + 24 Z-anc = 97 qubits
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from surface_code_generic import build_stabilizers


def build_full_xz_z_memory_experiment(d: int, n_rounds: int):
    """|0>^(d*d) + n_rounds of X+Z syndrome extraction + Z measurement."""
    from qiskit import (
        QuantumCircuit, QuantumRegister, ClassicalRegister,
    )

    X_stabs, Z_stabs = build_stabilizers(d)
    n_data = d * d
    n_x = len(X_stabs)
    n_z = len(Z_stabs)

    data = QuantumRegister(n_data, "data")
    anc_x = QuantumRegister(n_x, "anc_x")
    anc_z = QuantumRegister(n_z, "anc_z")
    cr_data = ClassicalRegister(n_data, "data_meas")
    syndrome_regs = []
    for r in range(n_rounds):
        syndrome_regs.append(ClassicalRegister(n_x, f"sx_{r}"))
        syndrome_regs.append(ClassicalRegister(n_z, f"sz_{r}"))

    qc = QuantumCircuit(data, anc_x, anc_z, cr_data, *syndrome_regs)
    qc.barrier()

    for r in range(n_rounds):
        cr_x = syndrome_regs[2 * r]
        cr_z = syndrome_regs[2 * r + 1]
        # X-stab extraction: H + CNOT(anc -> data) + H + measure + reset
        for stab_idx, qs in enumerate(X_stabs):
            qc.h(anc_x[stab_idx])
            for q in qs:
                qc.cx(anc_x[stab_idx], data[q])
            qc.h(anc_x[stab_idx])
            qc.measure(anc_x[stab_idx], cr_x[stab_idx])
            qc.reset(anc_x[stab_idx])
        # Z-stab extraction: CNOT(data -> anc) + measure + reset
        for stab_idx, qs in enumerate(Z_stabs):
            for q in qs:
                qc.cx(data[q], anc_z[stab_idx])
            qc.measure(anc_z[stab_idx], cr_z[stab_idx])
            qc.reset(anc_z[stab_idx])
        qc.barrier()

    qc.measure(data, cr_data)
    return qc


def submit_full_xz(d: int = 5, n_rounds_list=(1, 4), shots: int = 8192,
                     instance: str = "Zynerji") -> str:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend("ibm_kingston")
    print(f"[full-XZ d={d}] pending={backend.status().pending_jobs}")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    circuits = []
    metadata = []
    for nr in n_rounds_list:
        surf_qc = build_full_xz_z_memory_experiment(d=d, n_rounds=nr)
        surf_isa = pm.run(surf_qc)
        print(f"  d={d}, n={nr}: qubits={surf_qc.num_qubits}, "
              f"ISA depth={surf_isa.depth()}")
        circuits.append(surf_isa)
        metadata.append({"d": d, "n_rounds": nr, "kind": "surface_full_xz",
                          "shots": shots})

    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.default_shots = shots
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run(circuits, shots=shots)
    print(f"[full-XZ d={d}] job_id={job.job_id()}")
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"surface_code_full_xz_d{d}_submitted.json"
    out_path.write_text(json.dumps({
        "job_id": job.job_id(),
        "d": d,
        "metadata_per_circuit": metadata,
        "shots_actual": shots,
        "submitted_unix": int(time.time()),
    }, indent=2))
    return job.job_id()


if __name__ == "__main__":
    submit_full_xz(d=5, n_rounds_list=(1, 4), shots=8192)
