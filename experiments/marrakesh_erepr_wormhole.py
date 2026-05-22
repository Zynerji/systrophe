"""Run the ER=EPR quantum information channel on IBM Marrakesh.

The buildable wormhole capability (see systrophe.erepr_channel,
FINDINGS_KNOPP_RATCHET.md sec. 13) executed on real quantum hardware.

Pipeline
--------
1. Pull LIVE Marrakesh calibration (quantum_golden_pendulum.calibration) and
   select the best connected good-qubit subset (T1/T2/readout/CX thresholds).
2. Build the deterministic ER=EPR teleportation circuit (EPR resource +
   coherent X/Z corrections) -- the channel that gives fidelity 1 noiselessly.
   Measure the (reference, output) Bell fidelity F = (1 + <XX> - <YY> + <ZZ>)/4
   in three bases.
3. Validate noiselessly (Statevector) -> F = 1.
4. Transpile onto the selected physical qubits (opt_level=2, depth check).
5. Submit to ibm_marrakesh via the Zynerji instance with error mitigation
   (dynamical decoupling + gate/measure twirling).
6. Report the hardware fidelity honestly (noise-degraded, expected < 1).

Also transpiles a minimal SYK-scrambled wormhole circuit purely to REPORT its
depth -- demonstrating why a faithful dense-SYK wormhole cannot run clean on
current hardware (the reason the 2022 demo used a learned sparse Hamiltonian).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# QuantumGoldenPendulum calibration pipeline
sys.path.insert(0, str(Path("C:/Users/cknop/.local/bin/QuantumGoldenPendulum")))
from quantum_golden_pendulum.calibration import pull_calibration, select_qubit_subset

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

INSTANCE = "Zynerji"
BACKEND = "ibm_marrakesh"
SHOTS = 4096
RESULTS = Path(__file__).with_name("marrakesh_erepr_results.json")

# logical qubits: P(ref)=0, M(message)=1, A=2, B(far mouth)=3
P, M, A, B = 0, 1, 2, 3


def _teleport_body(qc: QuantumCircuit) -> None:
    """Deterministic ER=EPR teleportation of an unknown qubit M -> B.

    The unknown qubit is prepared as half of a Bell pair with reference P, so
    measuring the (P, B) Bell fidelity certifies the channel for any input.
    """
    qc.h(A); qc.cx(A, B)          # EPR resource A-B = the wormhole bridge
    qc.h(P); qc.cx(P, M)          # reference-message Bell (the unknown input)
    qc.cx(M, A); qc.h(M)          # Bell rotation on (M, A)
    qc.cx(A, B)                   # coherent X correction (controlled on A)
    qc.cz(M, B)                   # coherent Z correction (controlled on M)


def build_fidelity_circuits() -> dict[str, QuantumCircuit]:
    """Three circuits measuring <XX>, <YY>, <ZZ> on the (P, B) pair."""
    circuits = {}
    for basis in ("XX", "YY", "ZZ"):
        qc = QuantumCircuit(4, 2)
        _teleport_body(qc)
        # rotate (P, B) into the chosen Pauli basis, then measure
        for q in (P, B):
            if basis == "XX":
                qc.h(q)
            elif basis == "YY":
                qc.sdg(q); qc.h(q)
        qc.measure(P, 0)
        qc.measure(B, 1)
        circuits[basis] = qc
    return circuits


def _correlator(counts: dict, shots: int) -> float:
    """<P_a P_b> = sum (-1)^(bit_P + bit_B) / shots from a 2-bit counts dict."""
    exp = 0.0
    for bitstr, c in counts.items():
        b = bitstr.replace(" ", "")
        parity = (int(b[-1]) + int(b[-2])) % 2     # bits for clbits 0 and 1
        exp += ((-1) ** parity) * c
    return exp / shots


def bell_fidelity(xx: float, yy: float, zz: float) -> float:
    """F(|Phi+>) = (1 + <XX> - <YY> + <ZZ>) / 4."""
    return (1.0 + xx - yy + zz) / 4.0


def validate_noiseless() -> float:
    """Noiseless Statevector check that the channel gives F = 1."""
    qc = QuantumCircuit(4)
    _teleport_body(qc)
    sv = Statevector.from_instruction(qc)
    # reduced (P, B) overlap with |Phi+>
    from qiskit.quantum_info import partial_trace, DensityMatrix
    rho = partial_trace(sv, [M, A])          # keep P, B
    phi = Statevector.from_label("00").data * 0
    phi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return float(np.real(phi.conj() @ rho.data @ phi))


def _syk_wormhole_depth(service, snap) -> dict:
    """Transpile a minimal SYK-scrambled wormhole and report depth (why it
    can't run clean). Uses systrophe.size_winding / erepr_channel to build a
    1-Trotter-step SYK scramble on 3+3 qubits."""
    from systrophe.erepr_channel import syk_hamiltonian, _syk_coupling_V
    from scipy.linalg import expm
    nq = 3
    H = syk_hamiltonian(nq, seed=1)
    UL = expm(-1j * H * 1.0)
    V = _syk_coupling_V(nq)
    EgV = expm(1j * 4.0 * V)
    qc = QuantumCircuit(2 + 2 * nq)
    Ls = list(range(2, 2 + nq)); Rs = list(range(2 + nq, 2 + 2 * nq))
    for k in range(nq):
        qc.h(Ls[k]); qc.cx(Ls[k], Rs[k])
    qc.h(0); qc.cx(0, 1); qc.swap(1, Ls[0])
    qc.unitary(UL, Ls, label="U_L")
    qc.unitary(EgV, Ls + Rs, label="e^{igV}")
    qc.unitary(UL, Rs, label="U_R")
    backend = service.backend(BACKEND)
    tqc = transpile(qc, backend=backend, optimization_level=2)
    return {"syk_logical_qubits": 2 + 2 * nq,
            "syk_transpiled_depth": int(tqc.depth()),
            "syk_transpiled_2q_gates": int(sum(1 for inst in tqc.data
                                               if inst.operation.num_qubits == 2))}


def best_quality_quad(backend, snap) -> tuple[list[int], dict]:
    """Pick the 4 connected qubits with the LOWEST total CX + readout error.

    Greedy: start from the lowest-CX-error good edge, grow to 4 qubits each step
    adding the neighbor minimizing (readout error + connecting CX error). Returns
    (qubits ordered with the highest-internal-degree qubit first = the teleport
    hub, error_summary).
    """
    from quantum_golden_pendulum.calibration import _get_two_qubit_error
    target = backend.target
    good = set(snap.good_qubits)
    readout = {q: snap.qubit_info[q].readout_error for q in good}
    edge_err: dict[frozenset, float] = {}
    adj: dict[int, list[int]] = {q: [] for q in good}
    for (i, j) in snap.good_edges:
        e = _get_two_qubit_error(target, i, j)
        if e is None:
            e = _get_two_qubit_error(target, j, i)
        if e is None:
            continue
        edge_err[frozenset((i, j))] = e
        adj[i].append(j); adj[j].append(i)

    start = min(edge_err, key=edge_err.get)
    sel = set(start)
    while len(sel) < 4:
        cands = {}
        for q in sel:
            for nb in adj[q]:
                if nb in sel:
                    continue
                ces = [edge_err[frozenset((nb, s))] for s in sel
                       if frozenset((nb, s)) in edge_err]
                if ces:
                    cands[nb] = readout[nb] + min(ces)
        if not cands:
            break
        sel.add(min(cands, key=cands.get))
    quad = sorted(sel)
    # order: hub (max internal degree) first -> logical M; transpile routes rest
    deg = {q: sum(1 for r in quad if frozenset((q, r)) in edge_err) for q in quad}
    ordered = sorted(quad, key=lambda q: -deg[q])
    summary = {
        "qubits": quad,
        "readout_errors": {q: round(readout[q], 5) for q in quad},
        "internal_cx_errors": {f"{i}-{j}": round(edge_err[frozenset((i, j))], 5)
                               for i in quad for j in quad
                               if i < j and frozenset((i, j)) in edge_err},
        "total_error": round(sum(readout[q] for q in quad)
                             + sum(edge_err[frozenset((i, j))] for i in quad
                                   for j in quad if i < j
                                   and frozenset((i, j)) in edge_err), 5),
    }
    return ordered, summary


def run_zne(shots: int = 32768, noise_factors=(1, 3, 5),
            qubits: list[int] | None = None, backend_name: str = BACKEND) -> dict:
    """Push the fidelity higher: EstimatorV2 + Zero-Noise Extrapolation.

    Measures the Bell-fidelity observable O_F = (I + X_PX_B - Y_PY_B + Z_PZ_B)/4
    directly (no final measurement on the circuit), with ZNE (gate-folding at
    noise factors, extrapolated to zero noise) on top of DD + gate twirling, at
    higher shot count for a tight extrapolation.
    """
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2

    service = QiskitRuntimeService(instance=INSTANCE)
    backend = service.backend(backend_name)
    print(f"    Backend: {backend_name}")
    snap = pull_calibration(backend)
    if qubits is not None:
        phys = qubits
        print(f"    Using quality-selected qubits: {phys}")
    else:
        phys, _ = select_qubit_subset(snap, 4)
        print(f"    Selected good qubits (BFS): {phys}")

    qc = QuantumCircuit(4)
    _teleport_body(qc)                     # state-prep only; Estimator measures O
    tqc = transpile(qc, backend=backend, optimization_level=2,
                    initial_layout=phys)
    # Bell-fidelity operator on logical (P=0, B=3), then map to physical layout
    F_op = SparsePauliOp.from_sparse_list(
        [("", [], 0.25), ("XX", [0, 3], 0.25),
         ("YY", [0, 3], -0.25), ("ZZ", [0, 3], 0.25)],
        num_qubits=4,
    )
    F_phys = F_op.apply_layout(tqc.layout)

    est = EstimatorV2(mode=backend)
    est.options.default_shots = shots
    est.options.resilience_level = 2       # ZNE + twirling + readout (T-REx)
    est.options.dynamical_decoupling.enable = True
    est.options.dynamical_decoupling.sequence_type = "XpXm"
    try:
        est.options.resilience.zne_mitigation = True
        est.options.resilience.zne.noise_factors = list(noise_factors)
        est.options.resilience.zne.extrapolator = "exponential"
    except Exception as e:
        print("    (using resilience_level=2 default ZNE config:", repr(e)[:80], ")")

    print(f"[ZNE] Submitting EstimatorV2 ({shots} shots, ZNE factors "
          f"{noise_factors}, DD + twirling)...")
    job = est.run([(tqc, F_phys)])
    res = job.result()[0]
    F = float(np.asarray(res.data.evs).reshape(-1)[0])
    # The runtime's built-in extrapolator sometimes returns nan; if so,
    # extrapolate to zero noise from the per-noise-factor fidelities directly.
    per_nf = np.asarray(res.data.evs_noise_factors).ravel()[:len(noise_factors)]
    nf = np.array(noise_factors, dtype=float)
    F_lin = float(np.polyval(np.polyfit(nf, per_nf, 1), 0.0))
    F_exp = float(np.exp(np.polyval(np.polyfit(nf, np.log(per_nf), 1), 0.0)))
    if not np.isfinite(F):
        F = F_exp
    print(f"[ZNE] per-noise-factor F: {dict(zip(nf.tolist(), per_nf.round(4).tolist()))}")
    print(f"[ZNE] zero-noise F: exponential={F_exp:.4f} linear={F_lin:.4f}  "
          f"(job {job.job_id()})")
    return {"backend": backend_name,
            "zne_fidelity": F_exp, "zne_fidelity_linear": F_lin,
            "zne_factor1_fidelity": float(per_nf[0]),
            "zne_per_noise_factor": {str(k): float(v) for k, v in zip(nf, per_nf)},
            "zne_shots": shots, "zne_noise_factors": list(noise_factors),
            "zne_extrapolator": "exponential (manual)", "zne_job_id": job.job_id(),
            "selected_qubits": phys}


def main(submit: bool = True) -> None:
    print("=" * 70)
    print("ER=EPR channel on IBM Marrakesh  (instance:", INSTANCE, ")")
    print("=" * 70)

    f_ideal = validate_noiseless()
    print(f"\n[1] Noiseless validation: Bell fidelity = {f_ideal:.6f} (expect 1.0)")

    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=INSTANCE)
    backend = service.backend(BACKEND)
    print(f"[2] Backend: {backend.name} ({backend.num_qubits} qubits)")

    # full calibration pipeline
    snap = pull_calibration(backend, save_path=RESULTS.with_name("marrakesh_calib.json"))
    print(f"    Calibration: {len(snap.good_qubits)} good / {len(snap.bad_qubits)} bad / "
          f"{len(snap.dead_qubits)} dead; {len(snap.good_edges)} good edges")
    phys, edges = select_qubit_subset(snap, 4)
    print(f"    Selected good qubits: {phys}")

    circuits = build_fidelity_circuits()
    initial_layout = {0: phys[0], 1: phys[1], 2: phys[2], 3: phys[3]}
    tcircs = {b: transpile(c, backend=backend, optimization_level=2,
                           initial_layout=[phys[0], phys[1], phys[2], phys[3]])
              for b, c in circuits.items()}
    depths = {b: tc.depth() for b, tc in tcircs.items()}
    print(f"[3] Transpiled depths (teleport): {depths}")

    syk = _syk_wormhole_depth(service, snap)
    print(f"[4] SYK wormhole transpiled depth: {syk['syk_transpiled_depth']} "
          f"({syk['syk_transpiled_2q_gates']} 2q gates) -- shown as the reason a "
          f"faithful dense-SYK wormhole can't run clean on current hardware")

    out = {"noiseless_fidelity": f_ideal, "selected_qubits": phys,
           "teleport_depths": depths, "syk_depth": syk}

    if not submit:
        RESULTS.write_text(json.dumps(out, indent=2, default=str))
        print("\n[dry-run] not submitting. Results ->", RESULTS.name)
        return

    from qiskit_ibm_runtime import SamplerV2, Batch
    print(f"\n[5] Submitting to {BACKEND} ({SHOTS} shots/circuit, DD + twirling)...")
    corr = {}
    with Batch(backend=backend) as batch:
        sampler = SamplerV2(mode=batch)
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
        jobs = {b: sampler.run([tcircs[b]], shots=SHOTS) for b in ("XX", "YY", "ZZ")}
        for b, job in jobs.items():
            res = job.result()[0]
            counts = res.data.c.get_counts() if hasattr(res.data, "c") else \
                res.join_data().get_counts()
            corr[b] = _correlator(counts, SHOTS)
            print(f"    <{b}> = {corr[b]:+.4f}   (job {job.job_id()})")

    F_hw = bell_fidelity(corr["XX"], corr["YY"], corr["ZZ"])
    out.update({"correlators": corr, "hardware_fidelity": F_hw,
                "classical_bound": 0.5})
    print(f"\n[6] HARDWARE Bell fidelity = {F_hw:.4f}  (classical bound 0.5, "
          f"noiseless 1.0)")
    print("    Beats classical bound:" , F_hw > 0.5)
    RESULTS.write_text(json.dumps(out, indent=2, default=str))
    print("Results ->", RESULTS.name)


if __name__ == "__main__":
    if "--best-quad" in sys.argv:
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc = QiskitRuntimeService(instance=INSTANCE)
        bk = svc.backend(BACKEND)
        snap = pull_calibration(bk)
        quad, summ = best_quality_quad(bk, snap)
        bfs, _ = select_qubit_subset(snap, 4)
        print("BFS quad:", bfs)
        print("Quality quad (ordered, hub first):", quad)
        print("Quality quad error profile:", json.dumps(summ, indent=2))
        if "--dry-run" not in sys.argv:
            out = run_zne(shots=32768, qubits=quad)
            out["quality_quad_summary"] = summ
            prev = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
            prev.update({f"bestquad_{k}": v for k, v in out.items()})
            RESULTS.write_text(json.dumps(prev, indent=2, default=str))
            print("Results ->", RESULTS.name)
    elif "--zne" in sys.argv:
        out = run_zne(shots=32768)
        prev = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        prev.update(out)
        RESULTS.write_text(json.dumps(prev, indent=2, default=str))
        print("Results ->", RESULTS.name)
    else:
        main(submit="--dry-run" not in sys.argv)
