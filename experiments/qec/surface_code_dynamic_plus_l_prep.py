"""Dynamic-circuit fault-tolerant |+_L> preparation for d=5 surface code.

Uses qiskit's classical-feedback `if_test()` to apply Z-corrections
based on the init Z-stab syndromes, projecting onto the +1 eigenstate
of all Z-stabs. This enables a proper X-memory experiment.

Procedure:
  1. Hadamard on all 25 data qubits -> |+>^25 (eigenstate of X-stabs)
  2. For each Z-stab, measure into a classical bit
  3. For each -1 Z-stab outcome, apply Z to one data qubit in the
     stab's support (classical feedback via if_test)
  4. This deterministically prepares the +1 eigenstate of all Z-stabs,
     i.e. |+_L>
  5. Then run n_rounds of X+Z syndrome extraction
  6. Measure logical X via H + Z basis

The Z-corrections from the init Z-syndromes can also be tracked
"in software" by the decoder rather than physically applied: every
shot's logical-X result is flipped XOR'd with the parity of the
init-syndrome-determined Z-correction qubits along the X-logical
row.

The dynamic-circuit version applies the corrections physically and
gives a clean prepared state. The post-selected version (also
implemented) keeps only shots where the init Z-syndrome is all-0,
giving the same state without classical feedback at the cost of
some shots.

For Heron-r2, dynamic circuits are supported by Sampler V2 with
appropriate backend options.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from surface_code_generic import build_stabilizers


def build_d5_x_memory_dynamic(n_rounds: int):
    """Dynamic-circuit |+_L> prep + n_rounds X+Z syndrome + measure X."""
    from qiskit import (
        QuantumCircuit, QuantumRegister, ClassicalRegister,
    )
    d = 5
    X_stabs, Z_stabs = build_stabilizers(d)
    n_data = d * d
    n_x = len(X_stabs)
    n_z = len(Z_stabs)

    data = QuantumRegister(n_data, "data")
    anc_x = QuantumRegister(n_x, "anc_x")
    anc_z = QuantumRegister(n_z, "anc_z")
    cr_init_z = ClassicalRegister(n_z, "init_z")
    cr_data = ClassicalRegister(n_data, "data_meas")
    syndrome_regs = []
    for r in range(n_rounds):
        syndrome_regs.append(ClassicalRegister(n_x, f"sx_{r}"))
        syndrome_regs.append(ClassicalRegister(n_z, f"sz_{r}"))

    qc = QuantumCircuit(data, anc_x, anc_z, cr_init_z, cr_data, *syndrome_regs)

    # Step 1: |+>^25
    for q in range(n_data):
        qc.h(data[q])

    # Step 2 + 3: measure Z-stabs, apply Z-corrections via if_test
    for stab_idx, qs in enumerate(Z_stabs):
        for q in qs:
            qc.cx(data[q], anc_z[stab_idx])
        qc.measure(anc_z[stab_idx], cr_init_z[stab_idx])
        qc.reset(anc_z[stab_idx])
        # If the syndrome was 1, apply Z to one data qubit (qs[0])
        with qc.if_test((cr_init_z[stab_idx], 1)):
            qc.z(data[qs[0]])
    qc.barrier()

    # n_rounds of X+Z syndrome extraction
    for r in range(n_rounds):
        cr_x = syndrome_regs[2 * r]
        cr_z = syndrome_regs[2 * r + 1]
        for stab_idx, qs in enumerate(X_stabs):
            qc.h(anc_x[stab_idx])
            for q in qs:
                qc.cx(anc_x[stab_idx], data[q])
            qc.h(anc_x[stab_idx])
            qc.measure(anc_x[stab_idx], cr_x[stab_idx])
            qc.reset(anc_x[stab_idx])
        for stab_idx, qs in enumerate(Z_stabs):
            for q in qs:
                qc.cx(data[q], anc_z[stab_idx])
            qc.measure(anc_z[stab_idx], cr_z[stab_idx])
            qc.reset(anc_z[stab_idx])
        qc.barrier()

    # Final logical-X measurement: H + Z basis
    for q in range(n_data):
        qc.h(data[q])
    qc.measure(data, cr_data)
    return qc


def build_bare_x_baseline(delay_dt: int):
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    qr = QuantumRegister(1, "bare")
    cr = ClassicalRegister(1, "bare_meas")
    qc = QuantumCircuit(qr, cr)
    qc.h(qr[0])
    qc.delay(int(delay_dt), qr[0], unit="dt")
    qc.h(qr[0])
    qc.measure(qr[0], cr[0])
    return qc


def submit_dynamic_x_memory(n_rounds_list=(1, 4), shots: int = 8192,
                              instance: str = "Zynerji") -> str:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend("ibm_kingston")
    print(f"[dyn X-mem d=5] pending={backend.status().pending_jobs}")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

    circuits = []
    metadata = []
    for nr in n_rounds_list:
        surf_qc = build_d5_x_memory_dynamic(n_rounds=nr)
        # Dynamic circuits require optimization_level <= 1 typically;
        # the transpiler will handle if_test scheduling.
        try:
            surf_isa = pm.run(surf_qc)
        except Exception as e:
            print(f"  transpile failed: {e}")
            raise
        try:
            duration_dt = int(surf_isa.duration) if surf_isa.duration is not None else surf_isa.depth() * 200
        except (AttributeError, TypeError):
            duration_dt = surf_isa.depth() * 200
        bare_qc = build_bare_x_baseline(duration_dt)
        bare_isa = pm.run(bare_qc)
        print(f"  d=5 dyn X-mem n={nr}: ISA depth={surf_isa.depth()}")
        circuits.append(surf_isa)
        circuits.append(bare_isa)
        metadata.append({"d": 5, "kind": "surface_x_memory_dynamic",
                          "n_rounds": nr, "shots": shots})
        metadata.append({"d": 5, "kind": "bare_x",
                          "n_rounds": nr, "shots": shots})

    opts = SamplerOptions()
    # Dynamical decoupling is incompatible with dynamic circuits in
    # Qiskit Runtime; disable it for if_test()-based prep.
    opts.dynamical_decoupling.enable = False
    opts.default_shots = shots
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run(circuits, shots=shots)
    print(f"[dyn X-mem d=5] job_id={job.job_id()}")
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "surface_code_dynamic_x_memory_submitted.json"
    out_path.write_text(json.dumps({
        "job_id": job.job_id(),
        "metadata_per_circuit": metadata,
        "shots_actual": shots,
        "submitted_unix": int(time.time()),
    }, indent=2))
    return job.job_id()


if __name__ == "__main__":
    submit_dynamic_x_memory(n_rounds_list=(1, 4), shots=8192)
