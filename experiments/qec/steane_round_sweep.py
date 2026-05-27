"""Distance-3 Steane code round sweep + physical-baseline comparison.

For each n_rounds in {1, 2, 4, 8}, submit BOTH:
  (a) The full Steane [[7,1,3]] experiment with n_rounds of mid-circuit
      syndrome extraction.
  (b) A matched physical baseline: a single bare qubit prepared in |0>,
      delayed for the same wall-clock duration as the Steane circuit,
      then measured in Z.

Pair (a) and (b) measure logical and physical Z=0 rates respectively
at matched wall-clock time. The canonical QEC claim is logical_rate(t)
> physical_rate(t) for some t = t_break_even, the "memory threshold".

All 8 circuits go in one SamplerV2 batch to land in one allocation
window.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from steane_logical_qubit import (
    build_full_experiment_circuit,
    decode_results,
    H_STEANE,
)


def build_bare_baseline_circuit(steane_isa_circuit) -> "QuantumCircuit":
    """Bare-qubit baseline: a single qubit prepared in |0>, delayed
    for a duration matched to the compiled Steane circuit, then
    measured. We use the ISA-transpiled Steane circuit's duration
    via backend timing.

    Returns a QuantumCircuit (NOT yet transpiled).
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

    qr = QuantumRegister(1, "bare")
    cr = ClassicalRegister(1, "bare_meas")
    qc = QuantumCircuit(qr, cr)
    # We leave the duration to be matched at transpile-time via
    # delay() once we know the Steane circuit's compiled duration.
    qc.measure(qr[0], cr[0])
    return qc


def build_bare_baseline_with_delay(delay_dt: int, backend) -> "QuantumCircuit":
    """Bare-qubit |0> -> delay(dt) -> measure. delay_dt is in backend dt units."""
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

    qr = QuantumRegister(1, "bare")
    cr = ClassicalRegister(1, "bare_meas")
    qc = QuantumCircuit(qr, cr)
    qc.delay(int(delay_dt), qr[0], unit="dt")
    qc.measure(qr[0], cr[0])
    return qc


def submit_batch(n_rounds_list: list[int], shots: int = 4096,
                  instance: str = "Zynerji") -> str:
    """Submit Steane + bare-baseline pairs for all n_rounds in one batch."""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend("ibm_kingston")
    print(f"[steane-sweep] backend pending={backend.status().pending_jobs}")

    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

    circuits = []
    metadata = []
    for nr in n_rounds_list:
        steane_qc = build_full_experiment_circuit(n_rounds=nr)
        steane_isa = pm.run(steane_qc)
        # Estimate Steane duration in dt units. Use the ISA circuit's
        # depth times typical gate duration (rough; ISA scheduler will
        # do the right thing for delay matching).
        # Better: get explicit duration in seconds from the scheduled
        # circuit. For now use depth * 200 ns / dt.
        # dt for Heron-r2 is typically 0.5 ns = 5e-10 s
        # Each scheduled instruction has its own duration; the
        # total circuit duration depends on the schedule.
        # Use the transpiler-attached duration if present.
        try:
            duration_dt = steane_isa.duration
        except AttributeError:
            duration_dt = steane_isa.depth() * 200
        if duration_dt is None:
            duration_dt = steane_isa.depth() * 200

        # Bare baseline matched at the same dt
        bare_qc = build_bare_baseline_with_delay(int(duration_dt), backend)
        bare_isa = pm.run(bare_qc)

        print(f"  n_rounds={nr}: steane ISA depth={steane_isa.depth()}, "
              f"duration_dt={duration_dt}, bare ISA depth={bare_isa.depth()}")

        circuits.append(steane_isa)
        circuits.append(bare_isa)
        metadata.append({"kind": "steane",  "n_rounds": nr, "duration_dt": int(duration_dt)})
        metadata.append({"kind": "bare",    "n_rounds": nr, "duration_dt": int(duration_dt)})

    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.default_shots = shots
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run(circuits, shots=shots)
    print(f"[steane-sweep] job_id={job.job_id()}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "steane_round_sweep_submitted.json"
    out_path.write_text(json.dumps({
        "job_id": job.job_id(),
        "n_rounds_list": n_rounds_list,
        "shots": shots,
        "metadata_per_circuit": metadata,
        "submitted_unix": int(time.time()),
    }, indent=2))
    print(f"[steane-sweep] wrote {out_path}")
    return job.job_id()


def recover(job_id: str, instance: str = "Zynerji") -> dict:
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(instance=instance)
    job = service.job(job_id)
    if str(job.status()) not in ("DONE", "COMPLETED"):
        return {"status": str(job.status())}

    submitted = json.loads(
        (Path(__file__).parent / "results" / "steane_round_sweep_submitted.json").read_text()
    )
    metadata = submitted["metadata_per_circuit"]

    result = job.result()
    per_circuit = []
    for i, meta in enumerate(metadata):
        data = result[i].data
        creg = next(iter(data))
        counts = getattr(data, creg).get_counts()
        if meta["kind"] == "steane":
            decoded = decode_results(counts, n_rounds=meta["n_rounds"])
            entry = {**meta, "decoded": decoded}
        else:
            # Bare baseline: single qubit, single classical bit
            total = sum(counts.values())
            zero_count = sum(v for k, v in counts.items() if k == "0")
            entry = {**meta, "decoded": {
                "physical_zero_rate": zero_count / total,
                "n_shots_total": total,
            }}
        per_circuit.append(entry)

    # Pair steane / bare for each n_rounds
    paired = {}
    for entry in per_circuit:
        nr = entry["n_rounds"]
        paired.setdefault(nr, {})
        paired[nr][entry["kind"]] = entry["decoded"]

    print()
    print("Round sweep + physical baseline result")
    print("=" * 70)
    print(f"{'n_rounds':>10}  {'logical Z=0':>15}  {'bare Z=0 (baseline)':>22}  {'logical - bare':>16}")
    for nr in sorted(paired.keys()):
        st = paired[nr].get("steane", {})
        br = paired[nr].get("bare", {})
        log_r = st.get("logical_zero_rate", float("nan"))
        bare_r = br.get("physical_zero_rate", float("nan"))
        diff = log_r - bare_r
        print(f"{nr:>10d}  {log_r:>15.4f}  {bare_r:>22.4f}  {diff:>+16.4f}")

    out_path = Path(__file__).parent / "results" / "steane_round_sweep_analysis.json"
    out_path.write_text(json.dumps({
        "job_id": job_id,
        "per_circuit": per_circuit,
        "paired_by_n_rounds": paired,
    }, indent=2))
    print()
    print(f"Wrote {out_path}")
    return paired


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--recover", type=str)
    ap.add_argument("--n-rounds", type=int, nargs="+",
                     default=[1, 2, 4, 8])
    ap.add_argument("--shots", type=int, default=4096)
    args = ap.parse_args()

    if args.submit:
        submit_batch(n_rounds_list=args.n_rounds, shots=args.shots)
    if args.recover:
        recover(args.recover)


if __name__ == "__main__":
    main()
