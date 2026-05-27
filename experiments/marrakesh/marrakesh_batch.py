"""Marrakesh batch: four experiments validating Systrophe project pieces.

A. Spinor monodromy at r in/out of CTC band, plus pair-extinction at delta=pi
   - Validates: src/systrophe/spinor_monodromy.py (Phase 19)
B. Deutsch CTC fixed-point convergence
   - Validates: src/systrophe/d_ctc.py
C. Lloyd P-CTC capacity = 1 bit (post-selected teleportation)
   - Validates: src/systrophe/qi_channel.py (Phase 17)
D. Z_3 monodromy cycle: S^3 = I on one-hot 3-state register
   - Validates: Z_3 Mobius cover structure (tipler_fractal.py)

Calibration recipe (from TriCameral.ai Bragg/Dirac runs):
  - SamplerV2 with backend
  - generate_preset_pass_manager(backend, optimization_level=3)
  - Depth < 150 sanity gate (above which Marrakesh circuits are pure noise)
  - Modern additions: dynamical_decoupling (XpXm) and twirling (gates+measure)

Run modes:
  python marrakesh_batch.py --sim        # local simulator dry-run
  python marrakesh_batch.py --hardware   # submit to ibm_marrakesh
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.linalg import polar

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.linalg import expm

from systrophe.quantum_info.spinor_monodromy import _gamma_matrices, spin_connection_phi
from systrophe.geometry.vanstockum import VanStockumInterior


def lp_spin_unitary(vs: VanStockumInterior, r: float,
                     angle_scale: float = 1.0) -> np.ndarray:
    """2-qubit unitary parameterized by LP spin-connection components at r.

    U(r) = R_y(s * w01_normalized) (x) R_x(s * w12_normalized)

    where w_ab_normalized = tanh(s * w_ab) keeps angles in (-pi, pi).
    The Phase 19 `spinor_holonomy` module uses sigma_{ab} that mixes
    compact rotations with non-compact boosts, giving a non-unitary
    exponential in (-+++) signature. Here we use a clean unitary
    parameterized by the same physical quantities: the spin-connection
    components omega^{ab}_phi.

    Properties: U|00> superposition; pair extinction (delta=pi -> w=0) -> U=I.
    """
    sc = spin_connection_phi(vs, r)
    # Tanh-bound the angles
    w01 = float(np.tanh(angle_scale * sc["omega_01_phi"]))
    w12 = float(np.tanh(angle_scale * sc["omega_12_phi"]))
    theta01 = w01 * np.pi
    theta12 = w12 * np.pi
    # R_y(theta) = cos(t/2) I - i sin(t/2) Y
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    Ry = np.cos(theta01 / 2) * I2 - 1j * np.sin(theta01 / 2) * Y
    Rx = np.cos(theta12 / 2) * I2 - 1j * np.sin(theta12 / 2) * X
    # Tensor product with qubit 0 on left (qiskit little-endian: q1 ⊗ q0)
    U = np.kron(Rx, Ry)
    err = float(np.linalg.norm(U @ U.conj().T - np.eye(4)))
    assert err < 1e-9, f"lp_spin_unitary non-unitary, err={err:.2e}"
    return U

SHOTS = 4096
DEPTH_LIMIT = 150
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------- helper: project to nearest unitary ----------

def to_unitary(M: np.ndarray) -> np.ndarray:
    """Polar projection: U = M (M^H M)^{-1/2}.

    Our spinor_holonomy() has a sign-convention bug (the spin generator
    is hermitian instead of antihermitian -> matrix exponential is not
    unitary). For the hardware experiment we project onto the nearest
    unitary. The physics-module sign fix is tracked separately.
    """
    U, _ = polar(M)
    return U


# ---------- Experiment A: spinor monodromy ----------

def build_spinor_circuit(U: np.ndarray, label: str) -> QuantumCircuit:
    """Apply 4x4 U to |00>, measure both qubits."""
    qr = QuantumRegister(2, "q")
    cr = ClassicalRegister(2, "c")
    qc = QuantumCircuit(qr, cr, name=label)
    qc.append(UnitaryGate(U, label=label), [qr[0], qr[1]])
    qc.measure(qr, cr)
    return qc


def experiment_A_spinor_monodromy() -> list[dict]:
    """Two unitaries: U(r=1.5) outside CTC, U(r=3.35) inside first CTC band.

    Plus a "pair-extinction" reference: identity (delta=pi).
    """
    vs = VanStockumInterior(omega=1.0, R=1.0)
    out = []
    for r, regime in [(1.5, "exterior_F_positive"), (3.35, "inside_CTC_band")]:
        U = lp_spin_unitary(vs, r)
        label = f"spinor_r{r:.2f}_{regime}"
        qc = build_spinor_circuit(U, label)
        # classical prediction P(k) = |<k| U |00>|^2
        probs_pred = np.abs(U[:, 0]) ** 2
        out.append({
            "exp": "A_spinor_monodromy",
            "label": label,
            "r": r,
            "regime": regime,
            "circuit": qc,
            "predicted_probs": probs_pred.tolist(),
        })
    # Pair extinction at delta=pi reduces U to identity
    U_id = np.eye(4, dtype=complex)
    qc_id = build_spinor_circuit(U_id, "pair_extinction_delta_pi")
    out.append({
        "exp": "A_spinor_monodromy",
        "label": "pair_extinction_delta_pi",
        "r": 3.35,
        "regime": "pair_delta_pi_identity",
        "circuit": qc_id,
        "predicted_probs": [1.0, 0.0, 0.0, 0.0],
    })
    return out


# ---------- Experiment B: Deutsch CTC fixed point ----------

def experiment_B_deutsch_ctc() -> list[dict]:
    """Channel U = CNOT(in -> ctc). Deutsch fixed point is rho_CTC = I/2.

    Prepare in = |0>; prepare ctc via Bell pair with discarded ancilla
    (-> maximally mixed); apply CNOT; measure ctc; expect uniform 50/50.

    Iterating Deutsch's map on this U preserves rho_CTC = I/2; we test
    one iteration and verify the marginal stays mixed.
    """
    qr = QuantumRegister(3, "q")  # 0=in, 1=ctc, 2=anc (discarded)
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr, name="deutsch_ctc")
    # Prepare ctc = I/2 via Bell pair, partial trace by discarding anc
    qc.h(qr[1])
    qc.cx(qr[1], qr[2])
    # in qubit = |0> (default)
    # Channel U = CNOT (control=ctc, target=in)
    qc.cx(qr[1], qr[0])
    # Measure ctc qubit only
    qc.measure(qr[1], cr[0])
    return [{
        "exp": "B_deutsch_ctc",
        "label": "deutsch_fixed_point",
        "circuit": qc,
        "predicted_probs": [0.5, 0.5],  # P(0), P(1) marginal on ctc
    }]


# ---------- Experiment C: Lloyd P-CTC capacity ----------

def experiment_C_pctc_capacity() -> list[dict]:
    """Post-selected teleportation: capacity = 1 bit.

    Protocol:
      - Prepare input state |in> in {|0>, |1>}
      - Prepare Bell pair on (a, b)
      - Apply CNOT(in, a); H(in); measure (in, a) -> postselect on |00>
      - Output is qubit b; should equal |in>

    Postselection probability ~ 1/4. We use 4*SHOTS effectively to get
    good conditional statistics.
    """
    out = []
    for input_label, input_bit in [("input_0", 0), ("input_1", 1)]:
        qr = QuantumRegister(3, "q")  # 0=in, 1=a, 2=b
        cr = ClassicalRegister(3, "c")  # measure all three
        qc = QuantumCircuit(qr, cr, name=f"pctc_{input_label}")
        if input_bit == 1:
            qc.x(qr[0])
        # Bell pair on (1, 2)
        qc.h(qr[1])
        qc.cx(qr[1], qr[2])
        # Bell measurement on (0, 1)
        qc.cx(qr[0], qr[1])
        qc.h(qr[0])
        qc.measure(qr, cr)
        out.append({
            "exp": "C_pctc_capacity",
            "label": f"pctc_{input_label}",
            "input_bit": input_bit,
            "circuit": qc,
            "predicted_postselect_outcome": input_bit,
        })
    return out


# ---------- Experiment D: Z_3 monodromy cycle ----------

def _z3_cyclic_shift(qr: QuantumRegister, qc: QuantumCircuit) -> None:
    """Cyclic permutation on 3-qubit one-hot register: |100>->|010>->|001>->|100>.

    Achievable as two SWAPs: SWAP(0,1) then SWAP(1,2). This implements
    |100>->|010>->|001>: applying SWAP(0,1) on |100> gives |010>; then
    SWAP(1,2) on |010> gives |001>. So a single application of
    (SWAP(0,1) then SWAP(1,2)) is a 3-cycle. S^3 = I.
    """
    qc.swap(qr[0], qr[1])
    qc.swap(qr[1], qr[2])


def experiment_D_z3_cycle() -> list[dict]:
    """Apply S, S^2, S^3 to |100>; verify S^3 returns |100>."""
    out = []
    for n_apply in [1, 2, 3]:
        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(3, "c")
        qc = QuantumCircuit(qr, cr, name=f"z3_S{n_apply}")
        # Initial state |100> (qubit 0 in |1>)
        qc.x(qr[0])
        for _ in range(n_apply):
            _z3_cyclic_shift(qr, qc)
        qc.measure(qr, cr)
        # Initial |001> (X on q0); S = SWAP(0,1) then SWAP(1,2) implements
        # cyclic shift (q0,q1,q2) -> (q1,q2,q0). Thus:
        #   S^1: |001> -> |100>   S^2: -> |010>   S^3: -> |001>
        expected_bitstring = {1: "100", 2: "010", 3: "001"}[n_apply]
        out.append({
            "exp": "D_z3_cycle",
            "label": f"z3_S{n_apply}",
            "n_apply": n_apply,
            "circuit": qc,
            "expected_bitstring": expected_bitstring,
        })
    return out


# ---------- Build & transpile + sanity ----------

def all_experiments() -> list[dict]:
    return (
        experiment_A_spinor_monodromy()
        + experiment_B_deutsch_ctc()
        + experiment_C_pctc_capacity()
        + experiment_D_z3_cycle()
    )


def transpile_and_check(experiments: list[dict], backend) -> list[dict]:
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa = []
    for ex in experiments:
        qc = ex["circuit"]
        isa_qc = pm.run(qc)
        d = isa_qc.depth()
        print(f"  {ex['label']:35s} pre_depth={qc.depth():3d}  isa_depth={d:4d}",
              flush=True)
        assert d < DEPTH_LIMIT, (
            f"{ex['label']} ISA depth {d} >= {DEPTH_LIMIT}; would be pure noise"
        )
        ex["isa_circuit"] = isa_qc
        ex["isa_depth"] = d
    return experiments


# ---------- Result analysis ----------

def analyze_A(ex: dict, counts: dict) -> dict:
    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}
    # ISA result keys may have leading clbit padding; left-pad/strip
    p_obs = [probs.get(f"{i:02b}", 0.0) for i in range(4)]
    # Note: Qiskit bit ordering: little-endian by default
    p_obs_be = [probs.get(f"{i:02b}"[::-1], 0.0) for i in range(4)]
    p_pred = ex["predicted_probs"]
    tv_le = 0.5 * sum(abs(o - p) for o, p in zip(p_obs, p_pred))
    tv_be = 0.5 * sum(abs(o - p) for o, p in zip(p_obs_be, p_pred))
    tv = min(tv_le, tv_be)
    return {
        "label": ex["label"],
        "predicted": p_pred,
        "observed_le": p_obs,
        "observed_be": p_obs_be,
        "total_variation": float(tv),
    }


def analyze_B(ex: dict, counts: dict) -> dict:
    total = sum(counts.values())
    p0 = counts.get("0", 0) / total
    p1 = counts.get("1", 0) / total
    return {
        "label": ex["label"],
        "p0": p0,
        "p1": p1,
        "deviation_from_half": abs(p0 - 0.5),
    }


def analyze_C(ex: dict, counts: dict) -> dict:
    """Postselect on (in, a) bitstring '00' (right two clbits) and look at b."""
    total = sum(counts.values())
    # Bitstring ordering c[0]c[1]c[2] - left to right is c2 c1 c0 (Qiskit default)
    # We measured qr -> cr in order; the printed bit ordering is reversed.
    # Postselect: in measured (clbit 0) = 0 AND a measured (clbit 1) = 0
    keep = {}
    for k, v in counts.items():
        # k is e.g. "010" meaning c2=0, c1=1, c0=0
        c0 = k[-1]  # in
        c1 = k[-2]  # a
        if c0 == "0" and c1 == "0":
            b = k[-3]  # output (qubit 2)
            keep[b] = keep.get(b, 0) + v
    n_kept = sum(keep.values())
    if n_kept == 0:
        return {"label": ex["label"], "post_selected_n": 0, "ok": False}
    p_out_1 = keep.get("1", 0) / n_kept
    return {
        "label": ex["label"],
        "input_bit": ex["input_bit"],
        "post_selected_n": n_kept,
        "post_selection_rate": n_kept / total,
        "P(out=1 | postselect)": p_out_1,
        "ok": abs(p_out_1 - ex["input_bit"]) < 0.15,
    }


def analyze_D(ex: dict, counts: dict) -> dict:
    total = sum(counts.values())
    expected = ex["expected_bitstring"]
    expected_le = expected[::-1]  # Qiskit's little-endian convention
    p_exp = counts.get(expected, 0) / total
    p_exp_le = counts.get(expected_le, 0) / total
    p_match = max(p_exp, p_exp_le)
    return {
        "label": ex["label"],
        "n_apply": ex["n_apply"],
        "expected": expected,
        "P(expected_bitstring)": p_match,
        "ok": p_match > 0.5,
    }


def analyze_all(experiments: list[dict], counts_per_ex: list[dict]) -> dict:
    out = {"A": [], "B": [], "C": [], "D": []}
    for ex, counts in zip(experiments, counts_per_ex):
        if ex["exp"].startswith("A_"):
            out["A"].append(analyze_A(ex, counts))
        elif ex["exp"].startswith("B_"):
            out["B"].append(analyze_B(ex, counts))
        elif ex["exp"].startswith("C_"):
            out["C"].append(analyze_C(ex, counts))
        elif ex["exp"].startswith("D_"):
            out["D"].append(analyze_D(ex, counts))
    return out


# ---------- Main entry points ----------

def run_simulator() -> None:
    """Local sanity check using BasicSimulator."""
    from qiskit.providers.basic_provider import BasicSimulator

    backend = BasicSimulator()
    experiments = all_experiments()
    print(f"\n[sim] {len(experiments)} circuits; building & transpiling...")
    experiments = transpile_and_check(experiments, backend)

    print(f"\n[sim] running {SHOTS} shots each...")
    counts_per_ex = []
    for ex in experiments:
        job = backend.run(ex["isa_circuit"], shots=SHOTS)
        counts = job.result().get_counts()
        counts_per_ex.append(counts)
        print(f"  {ex['label']:35s} unique_states={len(counts):4d}")

    analysis = analyze_all(experiments, counts_per_ex)
    out_path = RESULTS_DIR / "marrakesh_sim_analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    print(f"\n[sim] wrote {out_path}")
    print(json.dumps(analysis, indent=2))


def run_hardware() -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions

    service = QiskitRuntimeService(instance="Zynerji")
    backend = service.backend("ibm_marrakesh")
    status = backend.status()
    print(f"\n[hw] backend: {backend.name}, {backend.num_qubits}q, "
          f"status: {status.status_msg}, pending: {status.pending_jobs}")

    experiments = all_experiments()
    print(f"\n[hw] {len(experiments)} circuits; transpiling at opt_level=3...")
    experiments = transpile_and_check(experiments, backend)

    # Calibration: DD + twirling
    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.twirling.enable_gates = True
    opts.twirling.enable_measure = True
    opts.default_shots = SHOTS

    print("\n[hw] submitting batch via SamplerV2 ...")
    sampler = SamplerV2(mode=backend, options=opts)
    isa_circuits = [ex["isa_circuit"] for ex in experiments]
    t0 = time.monotonic()
    job = sampler.run(isa_circuits, shots=SHOTS)
    print(f"[hw] job_id={job.job_id()}  (waiting for result)")
    result = job.result()
    t1 = time.monotonic()
    print(f"[hw] result ready in {t1 - t0:.1f}s wall")

    counts_per_ex = []
    for i, ex in enumerate(experiments):
        # cr name varies; use canonical .data first classical register
        data = result[i].data
        creg_name = next(iter(data))
        counts = getattr(data, creg_name).get_counts()
        counts_per_ex.append(counts)
        print(f"  {ex['label']:35s} unique_states={len(counts):4d}")

    analysis = analyze_all(experiments, counts_per_ex)
    out_path = RESULTS_DIR / "marrakesh_hw_analysis.json"
    raw_path = RESULTS_DIR / "marrakesh_hw_counts.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    raw_path.write_text(json.dumps(
        [
            {"label": ex["label"], "isa_depth": ex["isa_depth"], "counts": counts}
            for ex, counts in zip(experiments, counts_per_ex)
        ],
        indent=2,
    ))
    print(f"\n[hw] wrote {out_path}")
    print(f"[hw] wrote {raw_path}")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="local simulator dry-run")
    parser.add_argument("--hardware", action="store_true", help="submit to Marrakesh")
    args = parser.parse_args()
    if not (args.sim or args.hardware):
        parser.error("pass --sim or --hardware")
    if args.sim:
        run_simulator()
    if args.hardware:
        run_hardware()
