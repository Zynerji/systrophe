"""X-memory: prepare |+_L>, run X+Z syndrome extraction, measure
logical X. Tests whether the d=5 surface code preserves logical
superposition states (Willow-style quantum memory).

For X-memory:
  - Initial state: |+_L> = (|0_L> + |1_L>) / sqrt(2)
  - Preparation: Hadamards on all 25 data qubits (|+>^25 is +1 eigenstate
    of all X-stabilizers; we then need to project onto +1 of Z-stabs)
  - n_rounds of FULL X+Z syndrome extraction (X-stab detects phase
    errors; Z-stab detects bit-flip errors)
  - Final measurement: Hadamards on all data + Z-measurement
    (equivalent to X-measurement)
  - Decoder: uses X-syndrome history to detect Z errors -> apply Z
    corrections (or equivalently, propagate to the X measurement
    result with sign flip)

For X-memory, the X-syndromes do real work (correcting phase errors
that would otherwise flip |+_L> -> |-_L>). This is where the full
X+Z code actually pays off.

Compare to a bare-qubit X-baseline: |+> -> delay -> H -> measure.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from surface_code_generic import build_stabilizers


def build_d5_x_memory(n_rounds: int):
    """|+_L> -> n rounds of X+Z syndromes -> H_L -> Z-measurement."""
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
    cr_data = ClassicalRegister(n_data, "data_meas")
    cr_init_z = ClassicalRegister(n_z, "init_z")
    syndrome_regs = []
    for r in range(n_rounds):
        syndrome_regs.append(ClassicalRegister(n_x, f"sx_{r}"))
        syndrome_regs.append(ClassicalRegister(n_z, f"sz_{r}"))

    qc = QuantumCircuit(data, anc_x, anc_z, cr_init_z, cr_data, *syndrome_regs)

    # |+_L> preparation: H on all data qubits.
    # |+>^25 is the +1 eigenstate of all X-stabilizers automatically;
    # we still need to project onto +1 of Z-stabs (which requires one
    # round of Z-syndrome measurement with classical-corrected Z flips).
    # For simplicity here we do ONE round of Z-stab measurement to
    # initialize; -1 syndromes mean a Z-correction is needed but we
    # just record them (the decoder can use them).
    for q in range(n_data):
        qc.h(data[q])
    # One initial Z-stab round (projects onto +1 eigenstate of Z-stabs
    # with random sign; record the sign).
    for stab_idx, qs in enumerate(Z_stabs):
        for q in qs:
            qc.cx(data[q], anc_z[stab_idx])
        qc.measure(anc_z[stab_idx], cr_init_z[stab_idx])
        qc.reset(anc_z[stab_idx])
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

    # Final logical-X measurement: H on all data + Z-measurement
    for q in range(n_data):
        qc.h(data[q])
    qc.measure(data, cr_data)
    return qc


def build_bare_x_baseline(delay_dt: int):
    """|+> -> delay -> H -> measure Z (equivalent to X measurement)."""
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    qr = QuantumRegister(1, "bare")
    cr = ClassicalRegister(1, "bare_meas")
    qc = QuantumCircuit(qr, cr)
    qc.h(qr[0])
    qc.delay(int(delay_dt), qr[0], unit="dt")
    qc.h(qr[0])
    qc.measure(qr[0], cr[0])
    return qc


def submit_x_memory(n_rounds_list=(1, 4), shots: int = 8192,
                      instance: str = "Zynerji") -> str:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend("ibm_kingston")
    print(f"[X-mem d=5] pending={backend.status().pending_jobs}")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

    circuits = []
    metadata = []
    for nr in n_rounds_list:
        surf_qc = build_d5_x_memory(n_rounds=nr)
        surf_isa = pm.run(surf_qc)
        try:
            duration_dt = int(surf_isa.duration) if surf_isa.duration is not None else surf_isa.depth() * 200
        except (AttributeError, TypeError):
            duration_dt = surf_isa.depth() * 200
        bare_qc = build_bare_x_baseline(duration_dt)
        bare_isa = pm.run(bare_qc)
        print(f"  d=5 X-mem n={nr}: ISA depth={surf_isa.depth()}, "
              f"duration_dt={duration_dt}")
        circuits.append(surf_isa)
        circuits.append(bare_isa)
        metadata.append({"d": 5, "kind": "surface_x_memory",
                          "n_rounds": nr, "duration_dt": duration_dt,
                          "shots": shots})
        metadata.append({"d": 5, "kind": "bare_x",
                          "n_rounds": nr, "duration_dt": duration_dt,
                          "shots": shots})

    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.default_shots = shots
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run(circuits, shots=shots)
    print(f"[X-mem d=5] job_id={job.job_id()}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "surface_code_x_memory_d5_submitted.json"
    out_path.write_text(json.dumps({
        "job_id": job.job_id(),
        "metadata_per_circuit": metadata,
        "shots_actual": shots,
        "submitted_unix": int(time.time()),
    }, indent=2))
    return job.job_id()


if __name__ == "__main__":
    submit_x_memory(n_rounds_list=(1, 4), shots=8192)
