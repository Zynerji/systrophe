"""Cross-chip "metallic invariant" entangled-state imprint.

We cannot physically entangle qubits across distinct IBM Quantum
chips (no shared photonic interconnect). What we CAN do is prepare
the SAME locally-entangled Bell pair on each chip, with a phase
rotation by an irrational metallic-mean angle, and ask whether
the three chips' resulting distributions are catcher-identical.

If they are (verdict=uniform), this certifies cross-chip quantum
reproducibility at irrational-phase resolution.

If they are NOT (verdict=novel_structure), the catcher has detected
a chip-specific deviation that survives both bit-occupancy hashing
AND the irrational-phase encoding -- a sign of either
(a) chip-specific systematic gate-discretisation error, or
(b) something more exotic.

Metallic means used
===================

The metallic means are roots of x^2 - n x - 1 = 0:
  Golden phi = (1 + sqrt 5) / 2   ~ 1.61803
  Silver sigma = 1 + sqrt 2       ~ 2.41421
  Bronze beta = (3 + sqrt 13) / 2 ~ 3.30278
  Copper omega = 2 + sqrt 5       ~ 4.23607

We use bronze and copper because they are FAR ENOUGH from any
small-integer rational that gate-compilation rounding cannot
serendipitously align them.

Circuit per metallic mean
=========================
1. Prepare |Phi+> = (|00> + |11>) / sqrt 2 on a 2-qubit pair.
2. Apply controlled phase rotation Rz(pi / metal) on qubit 1.
3. Measure both qubits in the Z basis.

The same 2 circuits (one per metallic mean: bronze, copper) are
submitted to all three Heron-r2 chips simultaneously.

Total: 2 circuits x 3 chips = 6 jobs (actually 2-circuit batches
per chip = 3 jobs).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from systrophe.catchers.novelty_catcher import (
    catch_novelty_in_distributions,
    catch_novelty_per_quantity,
)

SHOTS = 8192
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Metallic-mean constants
GOLDEN  = (1.0 + math.sqrt(5.0)) / 2.0           # ~1.618
SILVER  = 1.0 + math.sqrt(2.0)                    # ~2.414
BRONZE  = (3.0 + math.sqrt(13.0)) / 2.0           # ~3.303
COPPER  = 2.0 + math.sqrt(5.0)                    # ~4.236


def metallic_phase_bell_circuit(metal: str) -> QuantumCircuit:
    """Bell |Phi+> with a metallic-mean-controlled Rz phase rotation.

    `metal` is one of 'bronze', 'copper'. The phase angle is pi / metal.
    """
    qr = QuantumRegister(2, "q")
    cr = ClassicalRegister(2, "c")
    qc = QuantumCircuit(qr, cr, name=f"metallic_{metal}")
    # Bell-pair preparation
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    # Metallic phase rotation
    if metal == "bronze":
        theta = math.pi / BRONZE
    elif metal == "copper":
        theta = math.pi / COPPER
    elif metal == "golden":
        theta = math.pi / GOLDEN
    elif metal == "silver":
        theta = math.pi / SILVER
    else:
        raise ValueError(f"Unknown metallic mean: {metal}")
    qc.rz(theta, qr[1])
    qc.measure(qr, cr)
    return qc


def all_metals() -> list[str]:
    return ["bronze", "copper"]


def submit_all_chips(instance: str = "Zynerji") -> dict[str, str]:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions

    service = QiskitRuntimeService(instance=instance)
    job_ids = {}
    metals = all_metals()
    for chip in ("ibm_marrakesh", "ibm_fez", "ibm_kingston"):
        backend = service.backend(chip)
        print(f"\n[metallic] {chip}: pending={backend.status().pending_jobs}",
              flush=True)
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        isa = []
        for m in metals:
            qc = metallic_phase_bell_circuit(m)
            isa_qc = pm.run(qc)
            isa.append(isa_qc)
            print(f"  {m:8s} (theta=pi/{globals()[m.upper()]:.3f}={math.pi/globals()[m.upper()]:.4f}): "
                  f"pre_depth={qc.depth()}, isa_depth={isa_qc.depth()}",
                  flush=True)
        opts = SamplerOptions()
        opts.dynamical_decoupling.enable = True
        opts.dynamical_decoupling.sequence_type = "XpXm"
        opts.twirling.enable_gates = True
        opts.twirling.enable_measure = True
        opts.default_shots = SHOTS
        sampler = SamplerV2(mode=backend, options=opts)
        job = sampler.run(isa, shots=SHOTS)
        job_ids[chip] = job.job_id()
        print(f"[metallic] {chip} job_id={job.job_id()}", flush=True)

    out_path = RESULTS_DIR / "cross_chip_metallic_submitted.json"
    out_path.write_text(json.dumps({
        "job_ids": job_ids,
        "metals": metals,
        "metallic_means": {
            "bronze": BRONZE, "copper": COPPER,
            "golden": GOLDEN, "silver": SILVER,
        },
        "submitted_unix": int(time.time()),
    }, indent=2))
    print(f"\n[metallic] wrote {out_path}", flush=True)
    return job_ids


def recover_from_jobs(
    job_ids: dict[str, str], instance: str = "Zynerji",
) -> dict:
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(instance=instance)
    metals = all_metals()
    per_chip: dict[str, dict] = {}
    for chip, jid in job_ids.items():
        print(f"\n[metallic-recover] {chip} job={jid}", flush=True)
        job = service.job(jid)
        st = str(job.status())
        if st not in ("DONE", "COMPLETED"):
            print(f"  status={st}; skipping")
            per_chip[chip] = {"status": st}
            continue
        result = job.result()
        per_chip[chip] = {"per_metal": {}}
        for i, m in enumerate(metals):
            data = result[i].data
            creg_name = next(iter(data))
            counts = getattr(data, creg_name).get_counts()
            total = sum(counts.values())
            probs = {k: v / total for k, v in counts.items()}
            per_chip[chip]["per_metal"][m] = {
                "counts": dict(counts),
                "probs": probs,
            }
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:4]
            print(f"  {m}: top distributions: {[(k, round(v, 3)) for k, v in top]}")

    # Cross-chip catcher (per-quantity: same metal across chips)
    per_q = {}
    for m in metals:
        per_q[f"metallic_{m}"] = {}
        for chip in job_ids:
            if "per_metal" in per_chip.get(chip, {}):
                pr = per_chip[chip]["per_metal"][m]["probs"]
                # Use the 4 outcome probabilities as a feature vector
                vec = np.array([pr.get("00", 0), pr.get("01", 0),
                                 pr.get("10", 0), pr.get("11", 0)])
                per_q[f"metallic_{m}"][chip] = vec
    nov = catch_novelty_per_quantity(per_q, n_bins=32)

    out = {
        "per_chip": per_chip,
        "cross_chip_per_quantity_catcher": nov,
    }
    out_path = RESULTS_DIR / "cross_chip_metallic_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[metallic-recover] wrote {out_path}", flush=True)
    print(f"\n[metallic-recover] aggregate verdict: {nov['aggregate_verdict']}")
    for q, r in nov["per_quantity"].items():
        print(f"  {q}: verdict={r['verdict']}, n_dist={r['n_distributions']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--recover", action="store_true")
    ap.add_argument("--instance", default="Zynerji")
    args = ap.parse_args()
    if args.submit:
        submit_all_chips(instance=args.instance)
    if args.recover:
        sub = json.loads(
            (RESULTS_DIR / "cross_chip_metallic_submitted.json").read_text()
        )
        recover_from_jobs(sub["job_ids"], instance=args.instance)
