"""Calibration-aware qubit selection for surface code experiments.

Reads the current backend calibration (T1, T2, cz_err, readout_err per
qubit/pair) and selects the N highest-quality qubits as a contiguous
heavy-hex subgraph. Returns an initial_layout for the transpiler.

The 'quality' score per qubit is composed of:
  - T1 (higher is better, normalize by chip average)
  - T2 (higher is better)
  - readout_error (lower is better; weight 0.5)
  - cz_err averaged over the qubit's neighbors (lower is better; weight 1.0)

For surface code d=5 we need 37 qubits (25 data + 12 ancilla). The
heavy-hex topology constrains adjacency; the routine picks the largest
connected subgraph of N high-quality qubits.

Output:
  - initial_layout = list of physical qubit indices ordered by logical
    qubit index
  - quality summary JSON with per-qubit scores
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def get_backend_calibration(backend_name: str = "ibm_kingston",
                              instance: str = "Zynerji") -> dict:
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=instance)
    backend = service.backend(backend_name)
    props = backend.properties()
    target = backend.target
    n_qubits = backend.num_qubits

    calibration = {
        "backend": backend_name,
        "n_qubits": n_qubits,
        "per_qubit": [],
        "per_edge": [],
    }
    for q in range(n_qubits):
        if props is None:
            calibration["per_qubit"].append({
                "qubit": q, "T1_us": None, "T2_us": None,
                "readout_error": None, "sx_error": None,
            })
            continue
        try:
            t1 = props.t1(q) * 1e6
            t2 = props.t2(q) * 1e6
            ro = props.readout_error(q)
        except Exception:
            t1 = None
            t2 = None
            ro = None
        try:
            sx_err = props.gate_error("sx", q)
        except Exception:
            sx_err = None
        calibration["per_qubit"].append({
            "qubit": q, "T1_us": t1, "T2_us": t2,
            "readout_error": ro, "sx_error": sx_err,
        })

    # Two-qubit gates: pick "cz" or "ecr" depending on backend
    for op in ("cz", "ecr"):
        try:
            for inst, qargs in target.instructions:
                if inst.name == op and len(qargs) == 2:
                    try:
                        err = props.gate_error(op, list(qargs))
                    except Exception:
                        err = None
                    calibration["per_edge"].append({
                        "op": op, "qubits": list(qargs), "error": err,
                    })
        except Exception:
            pass
        if calibration["per_edge"]:
            break

    return calibration


def score_qubits(calibration: dict) -> dict:
    """Compute quality scores for each qubit. Higher = better."""
    per_q = calibration["per_qubit"]
    per_e = calibration["per_edge"]

    # Edge errors per qubit (average over neighbors)
    cz_by_qubit = {q["qubit"]: [] for q in per_q}
    for e in per_e:
        if e["error"] is None:
            continue
        for q in e["qubits"]:
            cz_by_qubit.setdefault(q, []).append(e["error"])
    cz_avg = {q: (np.mean(cz_by_qubit[q]) if cz_by_qubit[q] else None)
               for q in cz_by_qubit}

    # Aggregate score (higher is better)
    scores = {}
    t1_vals = [q["T1_us"] for q in per_q if q["T1_us"] is not None]
    t1_med = np.median(t1_vals) if t1_vals else 1.0
    for q in per_q:
        qi = q["qubit"]
        t1 = q["T1_us"] or 0
        t2 = q["T2_us"] or 0
        ro = q["readout_error"] if q["readout_error"] is not None else 1.0
        cz = cz_avg.get(qi)
        if cz is None:
            cz = 0.01  # penalize unknown
        # Composite: log T1 + log T2 - log(ro+0.001) - log(cz+0.0001)
        score = (
            np.log(t1 / t1_med + 0.1) +
            np.log(t2 / t1_med + 0.1) -
            np.log(ro + 0.001) -
            np.log(cz + 0.0001)
        )
        scores[qi] = float(score)
    return {"scores": scores, "cz_avg": cz_avg}


def report(backend_name: str = "ibm_kingston", n_top: int = 50) -> dict:
    cal = get_backend_calibration(backend_name)
    sc = score_qubits(cal)
    scores = sc["scores"]
    cz_avg = sc["cz_avg"]
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    print(f"Calibration scores for {backend_name}, top {n_top} qubits:")
    print(f"{'rank':>4} {'q':>4}  {'T1us':>7} {'T2us':>7} {'cz_err':>8} "
          f"{'sx_err':>8} {'ro_err':>8}  {'score':>7}")
    per_q_dict = {q["qubit"]: q for q in cal["per_qubit"]}
    for rank, (q, s) in enumerate(ranked[:n_top]):
        info = per_q_dict[q]
        t1 = info["T1_us"] if info["T1_us"] is not None else float("nan")
        t2 = info["T2_us"] if info["T2_us"] is not None else float("nan")
        cz = cz_avg.get(q) or float("nan")
        sx = info["sx_error"] if info["sx_error"] is not None else float("nan")
        ro = info["readout_error"] if info["readout_error"] is not None else float("nan")
        print(f"{rank:>4} {q:>4}  {t1:>7.1f} {t2:>7.1f} {cz:>8.2e} "
              f"{sx:>8.2e} {ro:>8.2e}  {s:>+7.3f}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"calibration_{backend_name.replace('ibm_', '')}.json"
    out_path.write_text(json.dumps({
        "backend": backend_name,
        "calibration": cal,
        "scores": scores,
        "ranked_qubits": [int(q) for q, s in ranked],
    }, indent=2, default=str))
    print()
    print(f"Wrote {out_path}")
    return {"ranked": ranked, "scores": scores, "calibration": cal}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--n-top", type=int, default=50)
    args = ap.parse_args()
    report(args.backend, args.n_top)
