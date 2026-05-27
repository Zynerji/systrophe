"""Test transversal logical X (X_L = product of X on all data qubits)
on the d=5 rotated surface code.

Procedure:
  1. Prepare |0_L> (trivial: |0>^25 for Z-memory mode)
  2. Apply X to every data qubit (transversal X_L)
  3. Run n_rounds of Z-syndrome extraction
  4. Measure data qubits
  5. Decode -> should give logical Z = 1 (flipped from 0)

If the transversal X_L is correct and the encoding survives, we
expect P(L = 1) close to the previous P(L = 0) numbers (~0.91 at
n_rounds=1 with Dijkstra-MWPM).

Submits {without_XL, with_XL} as paired circuits at n_rounds in {1, 4}.
4 circuits total.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from surface_code_generic import build_stabilizers


def build_d5_z_memory_with_optional_xl(n_rounds: int, apply_x_l: bool):
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    d = 5
    _, Z_stabs = build_stabilizers(d)
    n_data = d * d
    n_z = len(Z_stabs)

    data = QuantumRegister(n_data, "data")
    anc_z = QuantumRegister(n_z, "anc_z")
    cr_data = ClassicalRegister(n_data, "data_meas")
    syndrome_regs = [ClassicalRegister(n_z, f"sz_{r}") for r in range(n_rounds)]

    qc = QuantumCircuit(data, anc_z, cr_data, *syndrome_regs)
    if apply_x_l:
        # Transversal X_L: X on every data qubit
        for q in range(n_data):
            qc.x(data[q])
    qc.barrier()

    for r in range(n_rounds):
        cr_z = syndrome_regs[r]
        for stab_idx, qs in enumerate(Z_stabs):
            for q in qs:
                qc.cx(data[q], anc_z[stab_idx])
            qc.measure(anc_z[stab_idx], cr_z[stab_idx])
            qc.reset(anc_z[stab_idx])
        qc.barrier()
    qc.measure(data, cr_data)
    return qc


def submit_logical_x_test(n_rounds_list=(1, 4), shots: int = 8192,
                            instance: str = "Zynerji") -> str:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend("ibm_kingston")
    print(f"[logical-X] pending={backend.status().pending_jobs}")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

    circuits = []
    metadata = []
    for nr in n_rounds_list:
        for apply_x in (False, True):
            qc = build_d5_z_memory_with_optional_xl(n_rounds=nr, apply_x_l=apply_x)
            isa = pm.run(qc)
            print(f"  n_rounds={nr}, apply_X_L={apply_x}: "
                  f"ISA depth={isa.depth()}")
            circuits.append(isa)
            metadata.append({"n_rounds": nr, "apply_x_l": apply_x,
                              "shots": shots})

    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.default_shots = shots
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run(circuits, shots=shots)
    print(f"[logical-X] job_id={job.job_id()}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "surface_code_logical_x_submitted.json"
    out_path.write_text(json.dumps({
        "job_id": job.job_id(),
        "metadata_per_circuit": metadata,
        "shots": shots,
        "submitted_unix": int(time.time()),
    }, indent=2))
    return job.job_id()


if __name__ == "__main__":
    submit_logical_x_test()
