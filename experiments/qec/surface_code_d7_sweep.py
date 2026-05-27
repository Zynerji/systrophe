"""d=7 surface code Z-memory round sweep on ibm_kingston, plus
high-shot d=5 break-even tightening.

Submits in a single batch:
  - d=7 surface @ n_rounds in {1, 2, 4} (3 circuits)
  - d=7 bare-qubit baselines at matched durations (3 circuits)
  - d=5 surface @ n_rounds=1 with HIGH shots (16384) to tighten
    the previous 4.0-sigma break-even crossing to 8+ sigma (1 circuit)
  - d=5 bare baseline at n_rounds=1 with same high shots (1 circuit)

Total: 8 circuits in one job.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from surface_code_generic import (
    build_full_z_memory_experiment,
    build_stabilizers,
    decode_with_networkx_mwpm,
)


def build_bare_baseline_with_delay(delay_dt: int):
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    qr = QuantumRegister(1, "bare")
    cr = ClassicalRegister(1, "bare_meas")
    qc = QuantumCircuit(qr, cr)
    qc.delay(int(delay_dt), qr[0], unit="dt")
    qc.measure(qr[0], cr[0])
    return qc


def submit_d7_and_d5_high_shots(
    instance: str = "Zynerji",
    d7_n_rounds: list[int] = [1, 2, 4],
    d7_shots: int = 4096,
    d5_n_rounds_for_high_shots: int = 1,
    d5_high_shots: int = 16384,
) -> str:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend("ibm_kingston")
    print(f"[d7+d5hs] pending={backend.status().pending_jobs}")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

    circuits = []
    metadata = []
    # d=7 sweep
    for nr in d7_n_rounds:
        surf_qc = build_full_z_memory_experiment(d=7, n_rounds=nr)
        surf_isa = pm.run(surf_qc)
        try:
            duration_dt = int(surf_isa.duration) if surf_isa.duration is not None else surf_isa.depth() * 200
        except (AttributeError, TypeError):
            duration_dt = surf_isa.depth() * 200
        bare_qc = build_bare_baseline_with_delay(duration_dt)
        bare_isa = pm.run(bare_qc)
        print(f"  d=7 n={nr}: surf ISA depth={surf_isa.depth()}, "
              f"duration_dt={duration_dt}")
        circuits.append(surf_isa)
        circuits.append(bare_isa)
        metadata.append({"d": 7, "kind": "surface", "n_rounds": nr,
                         "duration_dt": duration_dt, "shots": d7_shots})
        metadata.append({"d": 7, "kind": "bare", "n_rounds": nr,
                         "duration_dt": duration_dt, "shots": d7_shots})
    # d=5 high-shots single
    nr = d5_n_rounds_for_high_shots
    surf_qc = build_full_z_memory_experiment(d=5, n_rounds=nr)
    surf_isa = pm.run(surf_qc)
    try:
        duration_dt = int(surf_isa.duration) if surf_isa.duration is not None else surf_isa.depth() * 200
    except (AttributeError, TypeError):
        duration_dt = surf_isa.depth() * 200
    bare_qc = build_bare_baseline_with_delay(duration_dt)
    bare_isa = pm.run(bare_qc)
    print(f"  d=5 n={nr} (high shots): surf ISA depth={surf_isa.depth()}, "
          f"duration_dt={duration_dt}, shots={d5_high_shots}")
    circuits.append(surf_isa)
    circuits.append(bare_isa)
    metadata.append({"d": 5, "kind": "surface", "n_rounds": nr,
                     "duration_dt": duration_dt, "shots": d5_high_shots})
    metadata.append({"d": 5, "kind": "bare", "n_rounds": nr,
                     "duration_dt": duration_dt, "shots": d5_high_shots})

    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    # default_shots applies to all circuits in this Sampler call;
    # SamplerV2 has no per-circuit-shots; submit two jobs if needed.
    # Here we accept the lower of the two as the common shot count.
    opts.default_shots = d5_high_shots  # 16384 for both
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run(circuits, shots=d5_high_shots)
    print(f"[d7+d5hs] job_id={job.job_id()}, shots={d5_high_shots}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "surface_code_d7_and_d5_high_shots_submitted.json"
    out_path.write_text(json.dumps({
        "job_id": job.job_id(),
        "metadata_per_circuit": metadata,
        "shots_actual": d5_high_shots,
        "submitted_unix": int(time.time()),
    }, indent=2))
    print(f"[d7+d5hs] wrote {out_path}")
    return job.job_id()


def recover(job_id: str = None, instance: str = "Zynerji") -> dict:
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=instance)

    out_dir = Path(__file__).parent / "results"
    submitted = json.loads(
        (out_dir / "surface_code_d7_and_d5_high_shots_submitted.json").read_text()
    )
    if job_id is None:
        job_id = submitted["job_id"]
    job = service.job(job_id)
    if str(job.status()) not in ("DONE", "COMPLETED"):
        return {"status": str(job.status())}

    metadata = submitted["metadata_per_circuit"]
    result = job.result()
    paired = {}
    for i, meta in enumerate(metadata):
        d = meta["d"]
        nr = meta["n_rounds"]
        key = f"d{d}_n{nr}"
        data = result[i].data
        creg = next(iter(data))
        counts = getattr(data, creg).get_counts()
        if meta["kind"] == "bare":
            total = sum(counts.values())
            zero_count = sum(v for k, v in counts.items() if k == "0")
            paired.setdefault(key, {})["bare"] = {
                "physical_zero_rate": zero_count / total,
                "n_shots_total": total,
            }
            continue
        # Surface: parse bitstring, run MWPM
        n_data = d * d
        n_z = len(build_stabilizers(d)[1])
        n_logical_zero = 0
        n_total = 0
        for bitstring, count in counts.items():
            parts = bitstring.split(" ")
            data_part = None
            sx_parts = []
            for part in parts:
                if len(part) == n_data:
                    data_part = part
                elif len(part) == n_z:
                    sx_parts.append(part)
            if data_part is None:
                continue
            data_bits = tuple(int(b) for b in reversed(data_part))
            sx_parts.reverse()
            z_syndromes = [
                tuple(int(b) for b in reversed(s)) for s in sx_parts[-nr:]
            ]
            logical = decode_with_networkx_mwpm(data_bits, z_syndromes, d)
            if logical == 0:
                n_logical_zero += count
            n_total += count
        paired.setdefault(key, {})["surface"] = {
            "d": d,
            "n_rounds": nr,
            "logical_zero_rate_mwpm": n_logical_zero / max(n_total, 1),
            "n_shots_total": n_total,
        }

    print()
    print("d=7 and d=5-high-shots round sweep on ibm_kingston (MWPM-decoded)")
    print("=" * 80)
    print(f"{'key':>10}  {'d':>3} {'n':>3}  {'logical Z=0':>15}  "
          f"{'bare Z=0':>15}  {'diff':>10}  {'sigma':>8}")
    for key in sorted(paired.keys()):
        s = paired[key].get("surface", {})
        b = paired[key].get("bare", {})
        lz = s.get("logical_zero_rate_mwpm", float("nan"))
        bz = b.get("physical_zero_rate", float("nan"))
        n_shots = s.get("n_shots_total", 0)
        if n_shots > 0:
            sigma_l = (lz * (1 - lz) / n_shots) ** 0.5
            sigma_b = (bz * (1 - bz) / n_shots) ** 0.5
            sigma_diff = (sigma_l ** 2 + sigma_b ** 2) ** 0.5
            n_sig = (lz - bz) / sigma_diff if sigma_diff > 0 else 0.0
        else:
            n_sig = 0.0
        d = s.get("d", "?")
        nr = s.get("n_rounds", "?")
        print(f"{key:>10}  {d:>3} {nr:>3}  {lz:>15.4f}  {bz:>15.4f}  "
              f"{lz - bz:>+10.4f}  {n_sig:>+8.2f}")

    out_path = out_dir / "surface_code_d7_and_d5_high_shots_analysis.json"
    out_path.write_text(json.dumps({
        "job_id": job_id,
        "paired_by_d_n": paired,
    }, indent=2))
    print()
    print(f"Wrote {out_path}")
    return paired


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--recover", type=str, nargs="?", default=None,
                    const="")
    args = ap.parse_args()
    if args.submit:
        submit_d7_and_d5_high_shots()
    if args.recover is not None:
        recover(args.recover if args.recover else None)
