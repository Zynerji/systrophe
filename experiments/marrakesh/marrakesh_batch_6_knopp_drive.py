"""Marrakesh batch #6: Knopp Drive hardware encoding.

We cannot put exotic matter on a 156-qubit superconducting chip, but
we CAN encode the Knopp Drive's four-mechanism amplitude structure
as a quantum circuit and observe hardware-confirmed extinction at
the Tipler CTC band boundaries.

The encoding
============
A 4-qubit circuit: 1 data qubit + 3 path qubits. For each r_index in
{0, 1, ..., 7} corresponding to the orbit radius r_k spanning the
first CTC band exit:

    r_k = R + 0.05 + k * 0.5     for k = 0, ..., 7

the data qubit accumulates a Knopp-Drive-shaped phase:

    phi_knopp(r_k) = (1 - tipler_gate(r_k)) * phi_Krasnikov(alpha_wall)
                   + epsilon_horn * cos(theta_0)

At r_k INSIDE the Tipler CTC band (k small), tipler_gate = 0, so
phi_knopp ~ epsilon_horn * cos(theta_0): the residual horn-twist phase
only. At r_k OUTSIDE the band (k larger), tipler_gate > 0, so a
chunk of the Krasnikov wall phase comes through.

Predicted distribution
======================
Mach-Zehnder with multi-CRZ:
- Inside band: data qubit phase = epsilon_horn * cos(theta_0) only.
  Final H on data gives P(data=1) ~ sin^2((epsilon * cos(theta_0)) / 2)
  ~ small.
- Outside band: data qubit phase includes the Krasnikov contribution.
  P(data=1) is biased to varying degree depending on r_k.

The catcher should flag the r_k where the data-qubit bias CROSSES
THE BAND BOUNDARY as a sharp Hamming transition.

Eight circuits in the sweep, one per r_k.

Calibration: opt_level=3 + DD(XpXm) + gate/measure twirling, 8192
shots.
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
DEPTH_LIMIT = 200
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Knopp Drive parameters
vs = VanStockumInterior(omega=1.0, R=1.0)
ALPHA_WALL = 4.0       # Krasnikov wall sharpness
EPSILON_HORN = 0.3     # horn twist amplitude
THETA_0_HORN = 0.5     # horn axis (radians)
COUPLING = 1.0         # Tipler-Krasnikov coupling

# r-index sweep across the first CTC band exit
N_R = 8
R_GRID = np.array([vs.R + 0.05 + k * 0.5 for k in range(N_R)])


def knopp_phase_at_r(r: float) -> float:
    """Knopp Drive effective data-qubit phase at radius r."""
    tilt = tipler_tilt_at(vs, r)
    gate_factor = max(1.0 - COUPLING * tilt, 0.0)
    # Krasnikov wall amplitude proxy: alpha * ln(r)
    phi_kras = ALPHA_WALL * math.log(r)
    # Knopp Drive phase = gated Krasnikov + horn twist
    phi_horn = EPSILON_HORN * math.cos(THETA_0_HORN)
    return gate_factor * phi_kras + phi_horn


def knopp_drive_circuit(r_index: int) -> QuantumCircuit:
    """4-qubit Knopp Drive encoding for r-index k.

    Encodes phi_knopp(r_k) as a controlled-rotation phase on the data
    qubit. Path qubits provide a superposition of "computational paths"
    so the data-qubit Z-bias becomes the experimental observable.
    """
    r = R_GRID[r_index]
    phi = knopp_phase_at_r(float(r))
    n_path = 3
    qr = QuantumRegister(1 + n_path, "q")
    cr = ClassicalRegister(1 + n_path, "c")
    qc = QuantumCircuit(qr, cr, name=f"knopp_r{r_index}")
    # All in |+>
    for q in range(1 + n_path):
        qc.h(qr[q])
    # Apply CRZ from each path qubit to data qubit with phi distributed
    # across the paths (so the catcher sees the full distribution).
    for n in range(n_path):
        qc.crz(phi / n_path, qr[n + 1], qr[0])
    # Optional cycle of "feedback" amplification: apply a second pass
    # of CRZ to model the Q-cavity standing-wave structure
    for n in range(n_path):
        qc.crz(phi / (2 * n_path), qr[n + 1], qr[0])
    # Final H on data for X-basis readout structure
    qc.h(qr[0])
    qc.measure(qr, cr)
    return qc


def predicted_distribution(r_index: int) -> dict[str, float]:
    """Classical simulation of the 4-qubit Knopp circuit at r_index."""
    r = R_GRID[r_index]
    phi = knopp_phase_at_r(float(r))
    n_path = 3
    probs: dict[str, float] = {}
    # Each path contributes phi/n_path via the first pass and phi/(2*n_path)
    # via the second pass, when its qubit is |1>.
    for path in range(2 ** n_path):
        path_bits = [(path >> n) & 1 for n in range(n_path)]
        total_phase = sum(
            (phi / n_path + phi / (2 * n_path)) * b for b in path_bits
        )
        p_data_0 = math.cos(total_phase / 2) ** 2
        p_data_1 = math.sin(total_phase / 2) ** 2
        prob_path = 1.0 / (2 ** n_path)
        # Bitstring: c_path3 c_path2 c_path1 c_data
        for d in (0, 1):
            bs = "".join(str(b) for b in reversed(path_bits)) + str(d)
            probs[bs] = probs.get(bs, 0.0) + prob_path * (
                p_data_0 if d == 0 else p_data_1
            )
    return probs


def transpile_and_check(experiments: list[dict], backend) -> list[dict]:
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    for ex in experiments:
        qc = ex["circuit"]
        isa_qc = pm.run(qc)
        d = isa_qc.depth()
        n_2q = sum(1 for instr in isa_qc.data if len(instr.qubits) == 2)
        print(f"  {ex['label']:18s} r={ex['r']:5.3f}  phi={ex['phi']:+7.4f}  "
              f"pre_depth={qc.depth():3d}  isa_depth={d:4d}  n_2q={n_2q}",
              flush=True)
        assert d < DEPTH_LIMIT, f"{ex['label']} depth {d} >= {DEPTH_LIMIT}"
        ex["isa_circuit"] = isa_qc
        ex["isa_depth"] = d
    return experiments


def all_experiments() -> list[dict]:
    out = []
    for k in range(N_R):
        r = float(R_GRID[k])
        phi = knopp_phase_at_r(r)
        tilt = tipler_tilt_at(vs, r)
        out.append({
            "exp": "knopp_drive",
            "label": f"knopp_r{k}",
            "r_index": k,
            "r": r,
            "tilt": float(tilt),
            "tipler_gate_factor": float(max(1.0 - COUPLING * tilt, 0.0)),
            "phi": float(phi),
            "circuit": knopp_drive_circuit(k),
            "predicted_distribution": predicted_distribution(k),
        })
    return out


def _tv(observed: dict[str, float], predicted: dict[str, float]) -> float:
    keys = set(observed) | set(predicted)
    return 0.5 * sum(abs(observed.get(k, 0.0) - predicted.get(k, 0.0))
                      for k in keys)


def run_simulator() -> None:
    from qiskit.providers.basic_provider import BasicSimulator
    backend = BasicSimulator()
    experiments = all_experiments()
    print(f"\n[sim] {len(experiments)} Knopp Drive circuits", flush=True)
    experiments = transpile_and_check(experiments, backend)
    summaries = []
    distributions = []
    for ex in experiments:
        job = backend.run(ex["isa_circuit"], shots=SHOTS)
        counts = job.result().get_counts()
        total = sum(counts.values())
        observed = {k: v / total for k, v in counts.items()}
        tv = _tv(observed, ex["predicted_distribution"])
        p_data1 = sum(v for k, v in observed.items() if k[-1] == "1")
        summaries.append({
            "label": ex["label"], "r_index": ex["r_index"], "r": ex["r"],
            "tilt": ex["tilt"], "tipler_gate_factor": ex["tipler_gate_factor"],
            "phi": ex["phi"], "tv_obs_vs_pred": tv,
            "P_data1_observed": float(p_data1),
            "P_data1_predicted": sum(
                v for k, v in ex["predicted_distribution"].items() if k[-1] == "1"
            ),
        })
        all_keys = sorted(set(counts.keys()) | set(ex["predicted_distribution"].keys()))
        prob_vec = np.array([counts.get(k, 0) / total for k in all_keys])
        distributions.append(prob_vec)
        print(f"  {ex['label']:18s} r={ex['r']:.3f}  gate={ex['tipler_gate_factor']:.3f}  "
              f"TV={tv:.4f}  P(data=1)={p_data1:.4f}", flush=True)

    labels = [s["label"] for s in summaries]
    novelty = catch_novelty_in_distributions(distributions, labels=labels)
    out_path = RESULTS_DIR / "marrakesh_batch6_sim_analysis.json"
    out_path.write_text(json.dumps({
        "per_circuit": summaries,
        "novelty_catcher": novelty,
    }, indent=2, default=str))
    print(f"\n[sim] novelty verdict: {novelty['verdict']}, "
          f"sharp_features={len(novelty['sharp_features'])}", flush=True)
    for sf in novelty["sharp_features"]:
        print(f"    sharp: {sf.get('between')}  step={sf.get('hamming_step')}",
              flush=True)
    print(f"[sim] wrote {out_path}", flush=True)


def submit_only() -> None:
    """Submit to IBM without blocking; print job_id."""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    service = QiskitRuntimeService(instance="Zynerji")
    backend = service.backend("ibm_marrakesh")
    status = backend.status()
    print(f"\n[submit] backend={backend.name}  pending={status.pending_jobs}",
          flush=True)
    experiments = all_experiments()
    experiments = transpile_and_check(experiments, backend)
    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.twirling.enable_gates = True
    opts.twirling.enable_measure = True
    opts.default_shots = SHOTS
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run([ex["isa_circuit"] for ex in experiments], shots=SHOTS)
    submitted = {
        "job_id": job.job_id(),
        "backend": backend.name,
        "n_circuits": len(experiments),
        "r_grid": R_GRID.tolist(),
        "phi_at_r": [knopp_phase_at_r(float(r)) for r in R_GRID],
        "submitted_unix": int(time.time()),
    }
    out = RESULTS_DIR / "marrakesh_batch6_submitted.json"
    out.write_text(json.dumps(submitted, indent=2))
    print(f"[submit] job_id={job.job_id()}  wrote {out}", flush=True)


def run_hardware() -> None:
    """Block-mode: submit and wait for result."""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    service = QiskitRuntimeService(instance="Zynerji")
    backend = service.backend("ibm_marrakesh")
    status = backend.status()
    print(f"\n[hw] backend: {backend.name}, status: {status.status_msg}, "
          f"pending: {status.pending_jobs}", flush=True)
    experiments = all_experiments()
    experiments = transpile_and_check(experiments, backend)
    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.twirling.enable_gates = True
    opts.twirling.enable_measure = True
    opts.default_shots = SHOTS
    sampler = SamplerV2(mode=backend, options=opts)
    t0 = time.monotonic()
    job = sampler.run([ex["isa_circuit"] for ex in experiments], shots=SHOTS)
    print(f"[hw] job_id={job.job_id()}", flush=True)
    result = job.result()
    print(f"[hw] result in {time.monotonic() - t0:.1f}s", flush=True)
    summaries = []
    distributions = []
    raw = []
    for i, ex in enumerate(experiments):
        data = result[i].data
        creg_name = next(iter(data))
        counts = getattr(data, creg_name).get_counts()
        total = sum(counts.values())
        observed = {k: v / total for k, v in counts.items()}
        tv = _tv(observed, ex["predicted_distribution"])
        p_data1 = sum(v for k, v in observed.items() if k[-1] == "1")
        summaries.append({
            "label": ex["label"], "r_index": ex["r_index"], "r": ex["r"],
            "tilt": ex["tilt"], "tipler_gate_factor": ex["tipler_gate_factor"],
            "phi": ex["phi"], "tv_obs_vs_pred": tv,
            "P_data1_observed": float(p_data1),
            "P_data1_predicted": sum(
                v for k, v in ex["predicted_distribution"].items() if k[-1] == "1"
            ),
        })
        all_keys = sorted(set(counts.keys()) | set(ex["predicted_distribution"].keys()))
        prob_vec = np.array([counts.get(k, 0) / total for k in all_keys])
        distributions.append(prob_vec)
        raw.append({"label": ex["label"], "counts": counts})
        print(f"  {ex['label']:18s} r={ex['r']:.3f}  gate={ex['tipler_gate_factor']:.3f}  "
              f"TV={tv:.4f}  P(data=1)={p_data1:.4f}", flush=True)
    labels = [s["label"] for s in summaries]
    novelty = catch_novelty_in_distributions(distributions, labels=labels)
    out_path = RESULTS_DIR / "marrakesh_batch6_hw_analysis.json"
    raw_path = RESULTS_DIR / "marrakesh_batch6_hw_counts.json"
    out_path.write_text(json.dumps({
        "per_circuit": summaries,
        "novelty_catcher": novelty,
        "submitted_job_id": job.job_id(),
    }, indent=2, default=str))
    raw_path.write_text(json.dumps(raw, indent=2))
    print(f"\n[hw] novelty verdict: {novelty['verdict']}, "
          f"sharp_features={len(novelty['sharp_features'])}", flush=True)
    print(f"[hw] wrote {out_path}", flush=True)
    print(f"[hw] wrote {raw_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--submit-only", action="store_true")
    args = parser.parse_args()
    if not (args.sim or args.hardware or args.submit_only):
        parser.error("pass --sim, --hardware, or --submit-only")
    if args.sim:
        run_simulator()
    if args.submit_only:
        submit_only()
    if args.hardware:
        run_hardware()
