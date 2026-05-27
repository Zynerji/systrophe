"""Distance-5 surface code Z-memory round sweep on ibm_kingston.

Batch-submits d=5 Z-memory circuits at n_rounds in {1, 2, 4} together
with a wall-clock-matched bare-qubit baseline per n_rounds.

This is the QEC SOTA-tier follow-up to the d=3 Steane sub-threshold
result: if d=5 has enough margin to cross break-even on Heron-r2, we
should see logical-rate > bare-baseline at some n_rounds.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from surface_code_d5 import (
    N_DATA, build_full_experiment, decode_lookup,
)


def build_bare_baseline_with_delay(delay_dt: int):
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    qr = QuantumRegister(1, "bare")
    cr = ClassicalRegister(1, "bare_meas")
    qc = QuantumCircuit(qr, cr)
    qc.delay(int(delay_dt), qr[0], unit="dt")
    qc.measure(qr[0], cr[0])
    return qc


def submit_batch(n_rounds_list: list[int], shots: int = 4096,
                  instance: str = "Zynerji") -> str:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend("ibm_kingston")
    print(f"[surf5-sweep] pending={backend.status().pending_jobs}")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

    circuits = []
    metadata = []
    for nr in n_rounds_list:
        surf_qc = build_full_experiment(n_rounds=nr, mode="z_memory")
        surf_isa = pm.run(surf_qc)
        try:
            duration_dt = surf_isa.duration
        except AttributeError:
            duration_dt = None
        if duration_dt is None:
            duration_dt = surf_isa.depth() * 200
        bare_qc = build_bare_baseline_with_delay(int(duration_dt))
        bare_isa = pm.run(bare_qc)
        print(f"  n_rounds={nr}: surf ISA depth={surf_isa.depth()}, "
              f"duration_dt={duration_dt}, bare depth={bare_isa.depth()}")
        circuits.append(surf_isa)
        circuits.append(bare_isa)
        metadata.append({"kind": "surface", "n_rounds": nr, "duration_dt": int(duration_dt)})
        metadata.append({"kind": "bare",    "n_rounds": nr, "duration_dt": int(duration_dt)})

    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.default_shots = shots
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run(circuits, shots=shots)
    print(f"[surf5-sweep] job_id={job.job_id()}")
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "surface_code_d5_round_sweep_submitted.json"
    out_path.write_text(json.dumps({
        "job_id": job.job_id(),
        "n_rounds_list": n_rounds_list,
        "shots": shots,
        "metadata_per_circuit": metadata,
        "submitted_unix": int(time.time()),
    }, indent=2))
    return job.job_id()


def recover(job_id: str, instance: str = "Zynerji") -> dict:
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=instance)
    job = service.job(job_id)
    if str(job.status()) not in ("DONE", "COMPLETED"):
        return {"status": str(job.status())}
    submitted = json.loads(
        (Path(__file__).parent / "results" / "surface_code_d5_round_sweep_submitted.json").read_text()
    )
    metadata = submitted["metadata_per_circuit"]
    result = job.result()
    per_circuit = []
    for i, meta in enumerate(metadata):
        data = result[i].data
        creg = next(iter(data))
        counts = getattr(data, creg).get_counts()
        if meta["kind"] == "surface":
            n_total = 0
            n_logical_zero = 0
            for bitstring, count in counts.items():
                parts = bitstring.split(" ")
                for part in parts:
                    if len(part) == N_DATA:
                        data_bits = tuple(int(b) for b in reversed(part))
                        logical = decode_lookup(data_bits)
                        if logical == 0:
                            n_logical_zero += count
                        n_total += count
                        break
            lz = n_logical_zero / max(n_total, 1)
            entry = {**meta, "decoded": {
                "logical_zero_rate": lz,
                "n_shots_total": n_total,
            }}
        else:
            total = sum(counts.values())
            zero_count = sum(v for k, v in counts.items() if k == "0")
            entry = {**meta, "decoded": {
                "physical_zero_rate": zero_count / total,
                "n_shots_total": total,
            }}
        per_circuit.append(entry)

    paired = {}
    for entry in per_circuit:
        nr = entry["n_rounds"]
        paired.setdefault(nr, {})
        paired[nr][entry["kind"]] = entry["decoded"]

    print()
    print("d=5 surface code Z-memory + bare baseline (ibm_kingston)")
    print("=" * 70)
    print(f"{'n_rounds':>10}  {'logical Z=0':>15}  {'bare Z=0':>15}  {'diff':>10}")
    for nr in sorted(paired.keys()):
        surf_r = paired[nr].get("surface", {}).get("logical_zero_rate", float("nan"))
        bare_r = paired[nr].get("bare", {}).get("physical_zero_rate", float("nan"))
        print(f"{nr:>10d}  {surf_r:>15.4f}  {bare_r:>15.4f}  {surf_r - bare_r:>+10.4f}")

    out_path = Path(__file__).parent / "results" / "surface_code_d5_round_sweep_analysis.json"
    out_path.write_text(json.dumps({
        "job_id": job_id,
        "per_circuit": per_circuit,
        "paired_by_n_rounds": paired,
    }, indent=2))
    return paired


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--recover", type=str)
    ap.add_argument("--n-rounds", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--shots", type=int, default=4096)
    args = ap.parse_args()
    if args.submit:
        submit_batch(args.n_rounds, args.shots)
    if args.recover:
        recover(args.recover)
