"""d=9 surface code Z-memory on ibm_kingston.

Z-memory only fits Heron-r2's 156 qubits: 81 data + 40 Z-ancillas
= 121 qubits, well within budget. Full X+Z would need 161 qubits
which exceeds the chip.

Z-memory test: prepare |0_L> trivially, run n_rounds of Z-syndrome
extraction with Dijkstra-MWPM decoding, compare to wall-clock-matched
bare qubit.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from surface_code_generic import build_full_z_memory_experiment


def build_bare_baseline_with_delay(delay_dt: int):
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    qr = QuantumRegister(1, "bare")
    cr = ClassicalRegister(1, "bare_meas")
    qc = QuantumCircuit(qr, cr)
    qc.delay(int(delay_dt), qr[0], unit="dt")
    qc.measure(qr[0], cr[0])
    return qc


def submit_d9(n_rounds_list=(1, 2, 4), shots: int = 8192,
                instance: str = "Zynerji") -> str:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend("ibm_kingston")
    print(f"[d9-kingston] pending={backend.status().pending_jobs}")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

    circuits = []
    metadata = []
    for nr in n_rounds_list:
        surf_qc = build_full_z_memory_experiment(d=9, n_rounds=nr)
        try:
            surf_isa = pm.run(surf_qc)
        except Exception as e:
            print(f"  d=9 n={nr} TRANSPILE FAILED: {e}")
            raise
        try:
            duration_dt = int(surf_isa.duration) if surf_isa.duration is not None else surf_isa.depth() * 200
        except (AttributeError, TypeError):
            duration_dt = surf_isa.depth() * 200
        bare_qc = build_bare_baseline_with_delay(duration_dt)
        bare_isa = pm.run(bare_qc)
        print(f"  d=9 n={nr}: qubits={surf_qc.num_qubits}, "
              f"surf ISA depth={surf_isa.depth()}, duration_dt={duration_dt}")
        circuits.append(surf_isa)
        circuits.append(bare_isa)
        metadata.append({"d": 9, "kind": "surface", "n_rounds": nr,
                          "duration_dt": duration_dt, "backend": "ibm_kingston"})
        metadata.append({"d": 9, "kind": "bare", "n_rounds": nr,
                          "duration_dt": duration_dt, "backend": "ibm_kingston"})

    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.default_shots = shots
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run(circuits, shots=shots)
    print(f"[d9-kingston] job_id={job.job_id()}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "surface_code_d9_kingston_submitted.json"
    out_path.write_text(json.dumps({
        "job_id": job.job_id(),
        "metadata_per_circuit": metadata,
        "shots_actual": shots,
        "submitted_unix": int(time.time()),
    }, indent=2))
    return job.job_id()


if __name__ == "__main__":
    submit_d9()
