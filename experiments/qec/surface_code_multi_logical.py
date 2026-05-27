"""Multiple parallel d=5 logical qubits on the same Heron-r2 chip.

Each d=5 patch uses 37 qubits (25 data + 12 Z-ancilla). Heron-r2 has
156 qubits, so up to 4 patches fit in principle (148 qubits).

This script submits 2 parallel d=5 patches running independent
Z-memory circuits at n_rounds=1. The two patches don't interact;
they're spatially separated on the chip. We expect both to give
the same logical Z=0 rate (within shot noise) as the single-patch
result, demonstrating that the QEC suppression scales linearly with
patch count.

Useful as evidence of multi-logical-qubit memory parallelism on
Heron-r2, paving the way for transversal logical CNOT and other
multi-qubit operations.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from surface_code_generic import build_stabilizers


def build_multi_d5_z_memory(n_patches: int, n_rounds: int):
    """Build a single circuit with n_patches independent d=5 surface
    code Z-memory experiments running in parallel.

    Returns the QuantumCircuit; transpiler will route to disjoint
    qubit groups."""
    from qiskit import (
        QuantumCircuit, QuantumRegister, ClassicalRegister,
    )

    d = 5
    _, Z_stabs = build_stabilizers(d)
    n_data = d * d
    n_z = len(Z_stabs)

    data_regs = [QuantumRegister(n_data, f"data_p{p}") for p in range(n_patches)]
    anc_regs = [QuantumRegister(n_z, f"anc_p{p}") for p in range(n_patches)]
    cr_data_regs = [ClassicalRegister(n_data, f"data_p{p}_meas")
                     for p in range(n_patches)]
    syndrome_regs = []
    for p in range(n_patches):
        for r in range(n_rounds):
            syndrome_regs.append(ClassicalRegister(n_z, f"sz_p{p}_r{r}"))

    qc = QuantumCircuit(*data_regs, *anc_regs, *cr_data_regs, *syndrome_regs)
    qc.barrier()

    for p in range(n_patches):
        data = data_regs[p]
        anc_z = anc_regs[p]
        for r in range(n_rounds):
            cr_z = syndrome_regs[p * n_rounds + r]
            for stab_idx, qs in enumerate(Z_stabs):
                for q in qs:
                    qc.cx(data[q], anc_z[stab_idx])
                qc.measure(anc_z[stab_idx], cr_z[stab_idx])
                qc.reset(anc_z[stab_idx])
            qc.barrier()
        qc.measure(data, cr_data_regs[p])

    return qc


def submit_multi_logical(n_patches: int = 2, n_rounds: int = 1,
                          shots: int = 8192, instance: str = "Zynerji") -> str:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend("ibm_kingston")
    print(f"[multi-d5] pending={backend.status().pending_jobs}")

    qc = build_multi_d5_z_memory(n_patches=n_patches, n_rounds=n_rounds)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa = pm.run(qc)
    print(f"[multi-d5] n_patches={n_patches}, n_rounds={n_rounds}, "
          f"qubits={qc.num_qubits}, ISA depth={isa.depth()}")

    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.default_shots = shots
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run([isa], shots=shots)
    print(f"[multi-d5] job_id={job.job_id()}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"surface_code_multi_d5_p{n_patches}_n{n_rounds}_submitted.json"
    out_path.write_text(json.dumps({
        "job_id": job.job_id(),
        "n_patches": n_patches,
        "n_rounds": n_rounds,
        "shots": shots,
        "submitted_unix": int(time.time()),
    }, indent=2))
    return job.job_id()


if __name__ == "__main__":
    submit_multi_logical(n_patches=2, n_rounds=1, shots=8192)
