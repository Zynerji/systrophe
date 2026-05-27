"""Snapshot IBM Quantum backend calibration data for HW experiments.

Pulls the current T1/T2/single-qubit/two-qubit gate error/readout
error properties for each backend at the time of submission, and
saves to per-chip JSON files alongside the job submission records.

This is critical for reconciling observed bitstring distributions
across chips: each chip has its own per-qubit error budget that
shifts the noise floor and can produce chip-specific deviations
the catcher might mistake for emergent structure.

Usage
-----
    python experiments/calibration_snapshot.py --backends ibm_marrakesh ibm_fez ibm_kingston

Output
------
    experiments/results/calibration_<backend>_<unix_timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def snapshot_backend_calibration(backend) -> dict:
    """Return a JSON-serializable summary of backend calibration data.

    Includes per-qubit T1/T2/frequency/anharmonicity/readout error and
    per-gate error rates. Compatible with Qiskit IBM Runtime
    `BackendV2.target` introspection.
    """
    name = backend.name
    target = backend.target
    n_qubits = target.num_qubits
    snapshot = {
        "backend": name,
        "n_qubits": n_qubits,
        "snapshot_unix": int(time.time()),
        "snapshot_iso": time.strftime("%Y-%m-%dT%H:%M:%S",
                                       time.gmtime()),
        "qubit_properties": [],
        "instruction_errors": {},
    }
    # Per-qubit properties
    for q in range(n_qubits):
        try:
            props = target.qubit_properties[q]
            snapshot["qubit_properties"].append({
                "qubit": q,
                "T1": getattr(props, "t1", None),
                "T2": getattr(props, "t2", None),
                "frequency": getattr(props, "frequency", None),
                "anharmonicity": getattr(props, "anharmonicity", None),
            })
        except (AttributeError, IndexError):
            snapshot["qubit_properties"].append({
                "qubit": q, "T1": None, "T2": None,
            })
    # Per-instruction error rates
    for inst_name, qargs_props in target.items():
        snapshot["instruction_errors"][inst_name] = {}
        for qargs, props in qargs_props.items():
            if props is None:
                continue
            qkey = str(qargs)
            err = getattr(props, "error", None)
            dur = getattr(props, "duration", None)
            snapshot["instruction_errors"][inst_name][qkey] = {
                "error": float(err) if err is not None else None,
                "duration_s": float(dur) if dur is not None else None,
            }
    return snapshot


def median_q_param(snapshot: dict, key: str) -> float | None:
    """Median of `key` across qubit_properties (e.g., 'T1', 'T2')."""
    vals = [q.get(key) for q in snapshot["qubit_properties"]]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(sorted(vals)[len(vals) // 2])


def median_gate_error(snapshot: dict, gate: str) -> float | None:
    """Median error rate for a named instruction across all qargs."""
    gate_errors = snapshot["instruction_errors"].get(gate, {})
    vals = [e["error"] for e in gate_errors.values()
            if e.get("error") is not None]
    if not vals:
        return None
    return float(sorted(vals)[len(vals) // 2])


def summary_line(snapshot: dict) -> str:
    """One-line digest suitable for logging."""
    T1 = median_q_param(snapshot, "T1")
    T2 = median_q_param(snapshot, "T2")
    sx_err = median_gate_error(snapshot, "sx")
    cz_err = median_gate_error(snapshot, "cz")
    rdo = median_gate_error(snapshot, "measure")
    return (
        f"{snapshot['backend']} n_q={snapshot['n_qubits']}  "
        f"T1={T1 * 1e6:.1f} us, T2={T2 * 1e6:.1f} us, "
        f"sx_err={sx_err:.2e}, cz_err={cz_err:.2e}, "
        f"readout_err={rdo:.2e}"
        if all(v is not None for v in (T1, T2, sx_err, cz_err, rdo))
        else f"{snapshot['backend']} (incomplete data)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backends", nargs="+",
        default=["ibm_marrakesh", "ibm_fez", "ibm_kingston"],
        help="IBM Quantum backend names to snapshot",
    )
    ap.add_argument("--instance", default="Zynerji")
    args = ap.parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(instance=args.instance)
    summaries = []
    for name in args.backends:
        print(f"[calib] snapshotting {name}...", flush=True)
        backend = service.backend(name)
        snapshot = snapshot_backend_calibration(backend)
        out_path = RESULTS_DIR / f"calibration_{name}_{snapshot['snapshot_unix']}.json"
        out_path.write_text(json.dumps(snapshot, indent=2, default=str))
        line = summary_line(snapshot)
        summaries.append(line)
        print(f"[calib] wrote {out_path}", flush=True)
        print(f"[calib] {line}", flush=True)

    # Also save an aggregate summary
    agg_path = RESULTS_DIR / f"calibration_summary_{int(time.time())}.json"
    agg_path.write_text(json.dumps({
        "summaries": summaries,
        "snapshot_unix": int(time.time()),
    }, indent=2))
    print(f"\n[calib] aggregate summary at {agg_path}", flush=True)


if __name__ == "__main__":
    main()
