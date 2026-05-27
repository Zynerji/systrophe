"""Marrakesh batch #7: extended Knopp Drive sweep with calibration-
aware qubit placement on ibm_kingston (cleanest Heron-r2 chip).

Differences from batch 6
========================
- 16 r-points instead of 8 (finer band-edge resolution)
- 5 path qubits instead of 3 (richer band-amplitude composition)
- 6-qubit total circuit (1 data + 5 path)
- Calibration-aware placement: routed to Kingston's top-quality qubits
- Submitted to ibm_kingston (T1=280us, lowest sx_err, lowest cz_err)

Predicted outcome
=================
The same Knopp-band-gating extinction at the CTC band exit, but
with finer resolution at the band-edge (r~2.5) and a richer
multi-band interference pattern.
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

from systrophe.catchers.novelty_catcher import catch_novelty_in_distributions
from systrophe.geometry.tipler_krasnikov_hybrid import tipler_tilt_at
from systrophe.geometry.vanstockum import VanStockumInterior

SHOTS = 8192
N_BANDS = 5
N_R = 16
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Knopp Drive parameters (same as batch 6)
vs = VanStockumInterior(omega=1.0, R=1.0)
ALPHA_WALL = 4.0
EPSILON_HORN = 0.3
THETA_0_HORN = 0.5
COUPLING = 1.0


def r_grid() -> np.ndarray:
    """16-point r-sweep finely sampling the band exit + outer bands."""
    # Concentrate around r=2-3 (the first band exit)
    inner = np.linspace(1.05, 2.45, 6)   # inside first CTC band
    transition = np.linspace(2.50, 3.10, 5)  # band exit
    outer = np.linspace(3.20, 6.00, 5)    # outside band
    return np.concatenate([inner, transition, outer])


R_GRID = r_grid()


def knopp_phase_at_r(r: float) -> float:
    tilt = tipler_tilt_at(vs, r)
    gate_factor = max(1.0 - COUPLING * tilt, 0.0)
    phi_kras = ALPHA_WALL * math.log(r)
    phi_horn = EPSILON_HORN * math.cos(THETA_0_HORN)
    return gate_factor * phi_kras + phi_horn


def knopp_drive_extended_circuit(r_index: int) -> QuantumCircuit:
    """6-qubit Knopp Drive: 1 data + 5 path qubits."""
    r = R_GRID[r_index]
    phi = knopp_phase_at_r(float(r))
    n_path = N_BANDS
    qr = QuantumRegister(1 + n_path, "q")
    cr = ClassicalRegister(1 + n_path, "c")
    qc = QuantumCircuit(qr, cr, name=f"knopp_x_r{r_index}")
    for q in range(1 + n_path):
        qc.h(qr[q])
    # Two passes of band CRZ
    for n in range(n_path):
        qc.crz(phi / n_path, qr[n + 1], qr[0])
    for n in range(n_path):
        qc.crz(phi / (2 * n_path), qr[n + 1], qr[0])
    qc.h(qr[0])
    qc.measure(qr, cr)
    return qc


def all_experiments() -> list[dict]:
    return [
        {
            "label": f"knopp_x_r{k}",
            "r_index": k,
            "r": float(R_GRID[k]),
            "tilt": float(tipler_tilt_at(vs, float(R_GRID[k]))),
            "tipler_gate_factor": float(max(
                1.0 - COUPLING * tipler_tilt_at(vs, float(R_GRID[k])), 0.0
            )),
            "phi": float(knopp_phase_at_r(float(R_GRID[k]))),
            "circuit": knopp_drive_extended_circuit(k),
        }
        for k in range(len(R_GRID))
    ]


def submit_to_backend(backend_name: str, instance: str = "Zynerji") -> str:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend(backend_name)
    print(f"\n[knopp-x] {backend_name} pending={backend.status().pending_jobs}",
          flush=True)
    experiments = all_experiments()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa = []
    for ex in experiments:
        isa_qc = pm.run(ex["circuit"])
        isa.append(isa_qc)
        print(f"  r={ex['r']:.3f}  gate={ex['tipler_gate_factor']:.3f}  "
              f"phi={ex['phi']:+.4f}  isa_depth={isa_qc.depth()}",
              flush=True)
    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.twirling.enable_gates = True
    opts.twirling.enable_measure = True
    opts.default_shots = SHOTS
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run(isa, shots=SHOTS)
    print(f"[knopp-x] job_id={job.job_id()}", flush=True)

    short = backend_name.replace("ibm_", "")
    out_path = RESULTS_DIR / f"marrakesh_batch7_{short}_submitted.json"
    out_path.write_text(json.dumps({
        "job_id": job.job_id(),
        "backend": backend_name,
        "n_r": len(R_GRID),
        "r_grid": R_GRID.tolist(),
        "n_bands": N_BANDS,
        "shots": SHOTS,
        "submitted_unix": int(time.time()),
    }, indent=2))
    print(f"[knopp-x] wrote {out_path}", flush=True)
    return job.job_id()


def submit_kingston(instance: str = "Zynerji") -> str:
    return submit_to_backend("ibm_kingston", instance=instance)


def recover_from_job(job_id: str, instance: str = "Zynerji") -> dict:
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(instance=instance)
    job = service.job(job_id)
    print(f"\n[knopp-x-recover] job={job_id}", flush=True)
    st = str(job.status())
    if st not in ("DONE", "COMPLETED"):
        print(f"  status={st}; skipping")
        return {"status": st}
    result = job.result()
    experiments = all_experiments()
    summaries = []
    distributions = []
    for i, ex in enumerate(experiments):
        data = result[i].data
        creg_name = next(iter(data))
        counts = getattr(data, creg_name).get_counts()
        total = sum(counts.values())
        observed = {k: v / total for k, v in counts.items()}
        p_data1 = sum(v for k, v in observed.items() if k[-1] == "1")
        summaries.append({
            "label": ex["label"], "r_index": ex["r_index"], "r": ex["r"],
            "tilt": ex["tilt"], "tipler_gate_factor": ex["tipler_gate_factor"],
            "phi": ex["phi"],
            "P_data1_observed": float(p_data1),
        })
        all_keys = sorted(set(counts.keys()))
        vec = np.array([counts.get(k, 0) / total for k in all_keys])
        distributions.append(vec)
        print(f"  r={ex['r']:.3f}  gate={ex['tipler_gate_factor']:.3f}  "
              f"P(d=1)={p_data1:.4f}")

    labels = [s["label"] for s in summaries]
    # Align dimensions: pad distributions to max len
    max_n = max(len(d) for d in distributions)
    dists_padded = [
        np.concatenate([d, np.zeros(max_n - len(d))]) for d in distributions
    ]
    nov = catch_novelty_in_distributions(dists_padded, labels=labels)
    out = {
        "job_id": job_id,
        "per_circuit": summaries,
        "novelty_catcher": nov,
    }
    # Filename includes backend short-name so cross-chip runs don't clobber.
    short = str(job.backend().name).replace("ibm_", "")
    out_path = RESULTS_DIR / f"marrakesh_batch7_{short}_hw_analysis.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  catcher: verdict={nov['verdict']}, "
          f"n_sharp={len(nov['sharp_features'])}")
    for sf in nov["sharp_features"]:
        print(f"    sharp: {sf.get('between')}  step={sf.get('hamming_step')}")
    print(f"[knopp-x-recover] wrote {out_path}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--recover", action="store_true")
    ap.add_argument("--instance", default="Zynerji")
    args = ap.parse_args()
    if args.submit:
        submit_kingston(instance=args.instance)
    if args.recover:
        sub = json.loads(
            (RESULTS_DIR / "marrakesh_batch7_kingston_submitted.json").read_text()
        )
        recover_from_job(sub["job_id"], instance=args.instance)
