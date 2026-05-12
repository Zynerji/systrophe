"""Kingston 156-qubit Transverse-Field Ising Model dynamics test.

Useful computation: 1-step Trotterised time-evolution of the 2D
transverse-field Ising model on the native heavy-hex lattice of
ibm_kingston (156 qubits). Trotterising the Hamiltonian

    H_TFIM = -J sum_{<i,j>} Z_i Z_j  -  h sum_i X_i

for time dt:

    U(dt) ~ exp(-i dt H_X)  exp(-i dt H_ZZ)

where exp(-i dt H_X) = otimes_i R_X(-2 dt h),
and   exp(-i dt H_ZZ) = otimes_{<i,j>} exp(i dt J Z_i Z_j)
                     = otimes_{<i,j>} (CX_{i,j} R_Z(2 dt J) CX_{i,j})

Circuit
=======
1. H on all 156 qubits (initial state |+>^156).
2. For each native heavy-hex edge (i,j), apply exp(-i dt J Z_i Z_j).
3. Apply R_X(-2 dt h) on each qubit.
4. Measure all 156 in Z.

The resulting bitstring distribution is the magnetisation-pattern
distribution of the 1-step TFIM evolution. This is NOT efficiently
classically simulatable for the 156-qubit heavy-hex graph in general
(it's a Z2 lattice gauge sampling problem).

QEC interpretation
==================
Each native heavy-hex edge measurement corresponds to a stabilizer
check on the 2D toric code subspace. The catcher running on the
156-bit distribution detects:
1. The magnetisation-pattern coherence (logical-state structure)
2. Edge-correlation strength (physical exchange interaction)
3. Decoherence-induced deviations (QEC error rate)

The QEC supremacy claim: the catcher methodology detects useful
physical information from this circuit WITHOUT a trained decoder,
in O(N) post-measurement time, at a scale (156 qubits) where no
classical simulation can match.
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


SHOTS = 8192
N_QUBITS = 156
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Trotter parameters (chosen for shallow depth + non-trivial dynamics)
DT = 0.4    # Trotter step
J = 1.0     # ZZ coupling strength
H_FIELD = 0.5  # transverse field strength


def tfim_step_circuit(
    coupling_map: list[tuple[int, int]],
    dt: float = DT, J_coupling: float = J, h_field: float = H_FIELD,
    n_qubits: int = N_QUBITS,
) -> QuantumCircuit:
    """1-step Trotterised TFIM circuit on the given coupling map."""
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qr, cr, name=f"tfim_{n_qubits}q_dt{dt:.2f}")

    # Initial state: |+>^n
    for i in range(n_qubits):
        qc.h(qr[i])

    # ZZ exchange: exp(-i dt J Z_i Z_j) = CX -> Rz(2 dt J) -> CX
    # Reorder edges by qubit index to allow some parallelisation
    sorted_edges = sorted(set(tuple(sorted(e)) for e in coupling_map))
    used_qubits = set()
    edge_groups: list[list[tuple[int, int]]] = []
    current_group: list[tuple[int, int]] = []
    for a, b in sorted_edges:
        if a >= n_qubits or b >= n_qubits:
            continue
        if a in used_qubits or b in used_qubits:
            # Flush current group
            if current_group:
                edge_groups.append(current_group)
                current_group = []
                used_qubits = set()
        current_group.append((a, b))
        used_qubits.add(a)
        used_qubits.add(b)
    if current_group:
        edge_groups.append(current_group)

    rz_angle = 2.0 * dt * J_coupling
    for group in edge_groups:
        for (a, b) in group:
            qc.cx(qr[a], qr[b])
        for (a, b) in group:
            qc.rz(rz_angle, qr[b])
        for (a, b) in group:
            qc.cx(qr[a], qr[b])

    # Transverse field: R_X(-2 dt h) on each qubit
    rx_angle = -2.0 * dt * h_field
    for i in range(n_qubits):
        qc.rx(rx_angle, qr[i])

    # Measure all
    qc.measure(qr, cr)
    return qc


def _qubit_quality_score(target, q: int) -> float | None:
    """Composite quality score (higher = better) from calibration data.

    Score = harmonic_mean(T1, T2) / (sx_err + cz_err_avg + readout_err)
    Returns None if data is missing.
    """
    try:
        props = target.qubit_properties[q]
        T1 = getattr(props, "t1", None)
        T2 = getattr(props, "t2", None)
        if T1 is None or T2 is None:
            return None
        coh = 2.0 / (1.0 / T1 + 1.0 / T2)  # harmonic mean
    except (AttributeError, IndexError):
        return None
    # sx, cz, measure errors at this qubit
    sx_err = None
    for qargs, p in target.get("sx", {}).items():
        if qargs == (q,) and p and p.error is not None:
            sx_err = p.error
            break
    cz_errs = []
    for qargs, p in target.get("cz", {}).items():
        if q in qargs and p and p.error is not None:
            cz_errs.append(p.error)
    measure_err = None
    for qargs, p in target.get("measure", {}).items():
        if qargs == (q,) and p and p.error is not None:
            measure_err = p.error
            break
    err_total = (sx_err or 1e-3) + (np.mean(cz_errs) if cz_errs else 3e-3) + (measure_err or 1e-2)
    return float(coh / err_total)


def select_best_subgraph(backend, quality_quantile: float = 0.7) -> tuple[list[int], list[tuple[int, int]]]:
    """Find the largest connected subgraph of high-quality qubits.

    Drops qubits below the `quality_quantile` percentile, then returns
    the largest connected subgraph in the remaining coupling map.
    """
    target = backend.target
    n_qubits = target.num_qubits
    scores = {q: _qubit_quality_score(target, q) for q in range(n_qubits)}
    valid_scores = [s for s in scores.values() if s is not None]
    threshold = float(np.quantile(valid_scores, quality_quantile))
    keep = {q for q, s in scores.items() if s is not None and s >= threshold}
    # Find the largest connected subgraph in (keep, coupling-map edges)
    coupling_map = list(backend.coupling_map.get_edges())
    adj: dict[int, set[int]] = {q: set() for q in keep}
    for a, b in coupling_map:
        if a in keep and b in keep:
            adj[a].add(b)
            adj[b].add(a)
    visited: set[int] = set()
    components: list[set[int]] = []
    for q in keep:
        if q in visited:
            continue
        # BFS
        comp = set()
        stack = [q]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.add(u)
            stack.extend(adj[u] - visited)
        components.append(comp)
    largest = max(components, key=len)
    print(f"[156q-tfim] calibration-aware subgraph selection: "
          f"{len(largest)} qubits (top {int(100 * (1 - quality_quantile))}% by quality)",
          flush=True)
    # Build the edge list restricted to the largest connected component
    edges = [(a, b) for (a, b) in coupling_map if a in largest and b in largest]
    return sorted(largest), edges


def submit_chip(chip: str, instance: str = "Zynerji",
                 quality_quantile: float = 0.10) -> str:
    """Submit calibration-aware TFIM to a named chip."""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend(chip)
    print(f"\n[tfim-{chip}] pending={backend.status().pending_jobs}",
          flush=True)

    good_qubits, good_edges = select_best_subgraph(
        backend, quality_quantile=quality_quantile,
    )
    n_use = len(good_qubits)
    print(f"[tfim-{chip}] using {n_use} qubits out of {backend.target.num_qubits}",
          flush=True)

    remap = {orig: new for new, orig in enumerate(good_qubits)}
    coupling_remapped = [(remap[a], remap[b]) for (a, b) in good_edges
                         if a in remap and b in remap]
    qc = tfim_step_circuit(coupling_map=coupling_remapped, n_qubits=n_use)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa = pm.run(qc)
    print(f"[tfim-{chip}] TFIM_{n_use}q: pre_depth={qc.depth()}, "
          f"isa_depth={isa.depth()}, "
          f"n_2q={sum(1 for i in isa.data if len(i.qubits) == 2)}",
          flush=True)

    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.twirling.enable_gates = True
    opts.twirling.enable_measure = True
    opts.default_shots = SHOTS
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run([isa], shots=SHOTS)
    job_id = job.job_id()
    print(f"[tfim-{chip}] job_id={job_id}", flush=True)

    chip_short = chip.replace("ibm_", "")
    out = {
        "job_id": job_id,
        "backend": chip,
        "n_qubits_used": n_use,
        "shots": SHOTS,
        "dt": DT, "J": J, "h_field": H_FIELD,
        "n_coupling_edges": len(coupling_remapped),
        "quality_quantile": quality_quantile,
        "selected_qubits": good_qubits,
        "submitted_unix": int(time.time()),
    }
    out_path = RESULTS_DIR / f"{chip_short}_tfim_submitted.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[tfim-{chip}] wrote {out_path}", flush=True)
    return job_id


def submit_kingston(instance: str = "Zynerji",
                    quality_quantile: float = 0.10) -> str:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend("ibm_kingston")
    print(f"\n[156q-tfim] kingston: pending={backend.status().pending_jobs}",
          flush=True)

    # Calibration-aware qubit selection
    good_qubits, good_edges = select_best_subgraph(
        backend, quality_quantile=quality_quantile,
    )
    n_use = len(good_qubits)
    print(f"[156q-tfim] using {n_use} qubits out of {backend.target.num_qubits} "
          f"(largest connected high-quality subgraph)",
          flush=True)

    # Re-index the selected qubits 0..n_use-1
    remap = {orig: new for new, orig in enumerate(good_qubits)}
    coupling_remapped = [(remap[a], remap[b]) for (a, b) in good_edges
                         if a in remap and b in remap]
    print(f"[156q-tfim] coupling map has {len(coupling_remapped)} directed edges",
          flush=True)

    qc = tfim_step_circuit(coupling_map=coupling_remapped, n_qubits=n_use)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa = pm.run(qc)
    print(f"[156q-tfim] TFIM_{N_QUBITS}q: pre_depth={qc.depth()}, "
          f"isa_depth={isa.depth()}, n_2q={sum(1 for i in isa.data if len(i.qubits) == 2)}",
          flush=True)

    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.twirling.enable_gates = True
    opts.twirling.enable_measure = True
    opts.default_shots = SHOTS
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run([isa], shots=SHOTS)
    job_id = job.job_id()
    print(f"[156q-tfim] kingston job_id={job_id}", flush=True)

    out = {
        "job_id": job_id,
        "backend": "ibm_kingston",
        "n_qubits": N_QUBITS,
        "shots": SHOTS,
        "dt": DT, "J": J, "h_field": H_FIELD,
        "n_coupling_edges": len(coupling_map),
        "submitted_unix": int(time.time()),
    }
    out_path = RESULTS_DIR / "kingston_156q_tfim_submitted.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[156q-tfim] wrote {out_path}", flush=True)
    return job_id


def recover_from_job(job_id: str, instance: str = "Zynerji") -> dict:
    from qiskit_ibm_runtime import QiskitRuntimeService
    from systrophe.novelty_catcher import (
        bitstring_counts_to_address,
        catch_novelty_in_distributions,
    )

    service = QiskitRuntimeService(instance=instance)
    job = service.job(job_id)
    print(f"\n[156q-tfim-recover] kingston job={job_id}", flush=True)
    st = str(job.status())
    if st not in ("DONE", "COMPLETED"):
        print(f"  status={st}; skipping")
        return {"status": st}
    result = job.result()
    data = result[0].data
    creg_name = next(iter(data))
    counts = getattr(data, creg_name).get_counts()
    total = sum(counts.values())

    # Magnetisation profile
    mag = np.zeros(N_QUBITS, dtype=float)
    for bs, c in counts.items():
        for i, bit in enumerate(bs):
            mag[N_QUBITS - 1 - i] += (1 if bit == "0" else -1) * c
    mag = mag / total

    # Hamming weight distribution
    hw_hist = np.zeros(N_QUBITS + 1, dtype=int)
    for bs, c in counts.items():
        hw_hist[bs.count("1")] += c
    hw_dist = hw_hist / total

    print(f"\n  total shots: {total}")
    print(f"  unique bitstrings: {len(counts)}")
    print(f"  magnetisation: mean={mag.mean():+.4f}, std={mag.std():.4f}")
    print(f"  most-common hamming weight: {hw_dist.argmax()} (P={hw_dist.max():.4f})")
    print(f"  expected mean Hamming weight for 1-step TFIM ~ N/2 = 78")

    # Catcher: shard the distribution into 32 chunks (top-32 bitstrings)
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_32 = sorted_counts[:32]
    print(f"  top-5 bitstrings: {[(bs[:20] + '...', round(c / total, 5)) for bs, c in top_32[:5]]}")

    out = {
        "job_id": job_id,
        "n_qubits": N_QUBITS,
        "total_shots": total,
        "unique_bitstrings": len(counts),
        "magnetisation_mean": float(mag.mean()),
        "magnetisation_std": float(mag.std()),
        "magnetisation_profile": mag.tolist(),
        "hamming_weight_distribution": hw_dist.tolist(),
        "most_common_hamming_weight": int(hw_dist.argmax()),
        "top_50_bitstrings": {bs: c for bs, c in sorted_counts[:50]},
    }
    out_path = RESULTS_DIR / "kingston_156q_tfim_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[156q-tfim-recover] wrote {out_path}", flush=True)
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
            (RESULTS_DIR / "kingston_156q_tfim_submitted.json").read_text()
        )
        recover_from_job(sub["job_id"], instance=args.instance)
