"""Deeper-circuit benchmark where PEC should beat ZNE/baseline (IBM Kingston).

PEC (probabilistic error cancellation) wins on DEEP, structured-noise circuits,
not the near-clean shallow ER=EPR teleportation (where it gave no gain). This
runs a controlled test with an exactly-known answer:

  Mirror / Loschmidt-echo circuit: a depth-L random entangling brickwork V,
  followed by V^dagger, on |0...0>. Ideally V V^dagger = I, so the state returns
  to |0...0> and the observable P_0 = prod_i (I+Z_i)/2 (probability of all-zeros)
  has ideal value 1. Depth degrades the raw value; a good mitigator recovers it.

Compares, on the best-quality Kingston quad, the recovered <P_0>:
  baseline (T-REx + DD)  vs  ZNE  vs  PEC.
PEC is expected to give the smallest deviation from the ideal 1.0 here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("C:/Users/cknop/.local/bin/QuantumGoldenPendulum")))
from quantum_golden_pendulum.calibration import pull_calibration

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector

INSTANCE = "Zynerji"
BACKEND = "ibm_kingston"
QUBITS = [140, 141, 142, 143]
N = 4
SHOTS = 32768
RESULTS = Path(__file__).with_name("marrakesh_pec_depth_results.json")


def build_mirror(n: int, layers: int, seed: int = 11) -> QuantumCircuit:
    """V (depth-`layers` random RY + brickwork CX) then V^dagger. Ideal -> |0>^n."""
    rng = np.random.default_rng(seed)
    V = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            V.ry(float(rng.uniform(0, 2 * np.pi)), q)
        for q in range(0, n - 1, 2):
            V.cx(q, q + 1)
        for q in range(1, n - 1, 2):
            V.cx(q, q + 1)
    qc = QuantumCircuit(n)
    qc.compose(V, inplace=True)
    qc.compose(V.inverse(), inplace=True)
    return qc


def projector_zero(n: int) -> SparsePauliOp:
    """P_0 = prod_i (I + Z_i)/2 as a SparsePauliOp (ideal <P_0> = 1 on |0>^n)."""
    terms = []
    for mask in range(2 ** n):
        label = "".join("Z" if (mask >> i) & 1 else "I" for i in range(n))
        terms.append((label, 1.0 / 2 ** n))
    return SparsePauliOp.from_list(terms)


def validate(layers: int) -> float:
    qc = build_mirror(N, layers)
    sv = Statevector.from_instruction(qc)
    return float(abs(sv.data[0]) ** 2)   # P(all-zeros), ideal 1.0


def main(submit: bool, layers: int = 6) -> None:
    print("=" * 70)
    print(f"PEC deep-circuit benchmark on {BACKEND} (mirror, {layers} layers)")
    print("=" * 70)
    print(f"[1] Noiseless P(all-zeros) = {validate(layers):.6f} (ideal 1.0)")

    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
    service = QiskitRuntimeService(instance=INSTANCE)
    backend = service.backend(BACKEND)
    pull_calibration(backend)

    qc = build_mirror(N, layers)
    tqc = transpile(qc, backend=backend, optimization_level=2,
                    initial_layout=QUBITS)
    P0 = projector_zero(N).apply_layout(tqc.layout)
    print(f"[2] Transpiled depth = {tqc.depth()}, "
          f"2q gates = {sum(1 for i in tqc.data if i.operation.num_qubits == 2)}")

    out = {"layers": layers, "noiseless": validate(layers),
           "transpiled_depth": int(tqc.depth())}

    if not submit:
        RESULTS.write_text(json.dumps(out, indent=2, default=str))
        print("[dry-run] not submitting ->", RESULTS.name)
        return

    def estimate(label: str, configure) -> dict:
        est = EstimatorV2(mode=backend)
        est.options.default_shots = SHOTS
        est.options.dynamical_decoupling.enable = True
        est.options.dynamical_decoupling.sequence_type = "XpXm"
        configure(est)
        job = est.run([(tqc, P0)])
        res = job.result()[0]
        ev = float(np.asarray(res.data.evs).reshape(-1)[0])
        # ZNE may need manual extrapolation if built-in returns nan
        if not np.isfinite(ev) and hasattr(res.data, "evs_noise_factors"):
            nf = np.array([1.0, 3.0, 5.0])
            per = np.asarray(res.data.evs_noise_factors).ravel()[:3]
            ev = float(np.exp(np.polyval(np.polyfit(nf, np.log(np.clip(per, 1e-6, None)), 1), 0.0)))
        std = float(np.asarray(res.data.stds).reshape(-1)[0]) if hasattr(res.data, "stds") else float("nan")
        print(f"    {label:9s}: <P0> = {ev:.4f} +/- {std:.4f}  (job {job.job_id()})")
        return {"value": ev, "std": std, "job": job.job_id()}

    def cfg_baseline(est):
        est.options.resilience_level = 1                  # T-REx only
    def cfg_zne(est):
        est.options.resilience_level = 2                  # ZNE
    def cfg_pec(est):
        est.options.resilience_level = 1
        est.options.resilience.measure_mitigation = True
        est.options.resilience.pec_mitigation = True
        est.options.resilience.pec.max_overhead = 200.0
        est.options.twirling.enable_gates = True

    print(f"[3] Running baseline / ZNE / PEC ({SHOTS} shots each)...")
    out["baseline"] = estimate("baseline", cfg_baseline)
    out["zne"] = estimate("ZNE", cfg_zne)
    out["pec"] = estimate("PEC", cfg_pec)

    ideal = 1.0
    errs = {k: abs(ideal - out[k]["value"]) for k in ("baseline", "zne", "pec")}
    best = min(errs, key=errs.get)
    out["errors_from_ideal"] = errs
    out["best_method"] = best
    print(f"\n[4] |ideal - measured|: baseline={errs['baseline']:.4f}, "
          f"ZNE={errs['zne']:.4f}, PEC={errs['pec']:.4f}  -> best: {best}")
    print(f"    PEC beats baseline: {errs['pec'] < errs['baseline']}; "
          f"PEC beats ZNE: {errs['pec'] < errs['zne']}")
    RESULTS.write_text(json.dumps(out, indent=2, default=str))
    print("Results ->", RESULTS.name)


if __name__ == "__main__":
    L = 6
    for a in sys.argv:
        if a.startswith("--layers="):
            L = int(a.split("=")[1])
    main(submit="--dry-run" not in sys.argv, layers=L)
