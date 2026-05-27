"""Cross-chip d=5 surface code Z-memory: same experiment on Marrakesh
as on Kingston, for chip-reproducibility test.

If d=5 surface code achieves break-even on BOTH Heron-r2 chips with
matching margins (modulo their respective gate fidelities), this is
cross-chip QEC reproducibility — a substantially stronger claim than
single-chip break-even.
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


def submit_d5_marrakesh(n_rounds_list=(1, 2, 4), shots: int = 8192,
                         instance: str = "Zynerji") -> str:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend("ibm_marrakesh")
    print(f"[d5-marrakesh] pending={backend.status().pending_jobs}")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

    circuits = []
    metadata = []
    for nr in n_rounds_list:
        surf_qc = build_full_z_memory_experiment(d=5, n_rounds=nr)
        surf_isa = pm.run(surf_qc)
        try:
            duration_dt = int(surf_isa.duration) if surf_isa.duration is not None else surf_isa.depth() * 200
        except (AttributeError, TypeError):
            duration_dt = surf_isa.depth() * 200
        bare_qc = build_bare_baseline_with_delay(duration_dt)
        bare_isa = pm.run(bare_qc)
        print(f"  d=5 n={nr}: surf ISA depth={surf_isa.depth()}, "
              f"duration_dt={duration_dt}")
        circuits.append(surf_isa)
        circuits.append(bare_isa)
        metadata.append({"d": 5, "kind": "surface", "n_rounds": nr,
                          "duration_dt": duration_dt, "backend": "ibm_marrakesh"})
        metadata.append({"d": 5, "kind": "bare", "n_rounds": nr,
                          "duration_dt": duration_dt, "backend": "ibm_marrakesh"})

    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.default_shots = shots
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run(circuits, shots=shots)
    print(f"[d5-marrakesh] job_id={job.job_id()}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "surface_code_d5_marrakesh_submitted.json"
    out_path.write_text(json.dumps({
        "job_id": job.job_id(),
        "backend": "ibm_marrakesh",
        "metadata_per_circuit": metadata,
        "shots_actual": shots,
        "submitted_unix": int(time.time()),
    }, indent=2))
    return job.job_id()


if __name__ == "__main__":
    submit_d5_marrakesh()
