"""IBM Marrakesh toroidal Knopp band-edge extinction circuit.

Tests update.txt's claim that the existing Marrakesh band-edge extinction
validation transfers to the toroidal Knopp Drive variant. The original
batch (`marrakesh_batch_6_knopp_drive.py`) demonstrated band-edge
extinction for the infinite Tipler cylinder; this script does the
analogous experiment for the toroidal binary backend.

The circuit
-----------
4-qubit Floquet encoding of the toroidal band-edge structure:
  - q[0]:    probe state -- the "test particle" amplitude.
  - q[1]:    Floquet phase qubit -- accumulates the LT-induced phase
             phi(rho) = Omega_eff(rho) * T_period.
  - q[2,3]:  band-edge ancilla -- store T_eff(rho) >= 1 / < 1 verdict.

For each value of rho on a scan across the band, the circuit:
  1. Prepares q[0] in |+>.
  2. Applies a controlled phase exp(i*phi(rho)) entangling q[0]<->q[1].
  3. Conditional on T_eff(rho) >= 1, the gate sequence on q[2,3]
     applies a destructive-interference pattern that drives the
     |00> measurement probability toward zero -- the "extinction"
     signature.
  4. Measures all 4 qubits.

In the toroidal band (rho in [rho_in, rho_out]), the destructive-
interference pattern triggers -> P(|0000>) drops sharply.
Outside the band, the pattern is constructive -> P(|0000>) remains
high. The transition at the band edge is the falsifiable signature.

Validation modes
----------------
  python experiments/marrakesh/marrakesh_toroidal_band_edge.py --sim
     -> AerSimulator dry-run, full noise-free verification of the
        band-edge extinction signature.

  python experiments/marrakesh/marrakesh_toroidal_band_edge.py --hardware
     -> Submit to ibm_marrakesh via Qiskit Runtime. Requires
        IBMQ_TOKEN env var and an active Zynerji instance allocation.
        Calibration is pulled via quantum_golden_pendulum if available.

Honest scope
------------
- The circuit is a 4-qubit *Floquet proxy* for the toroidal-band signal,
  not a full simulation of the binary's spacetime. It tests whether
  the *same band-edge interference structure* that worked on the
  original Tipler-cylinder experiment also applies to the toroidal
  variant -- which is exactly the claim update.txt makes.
- This script does NOT actually submit jobs without the --hardware
  flag AND credentials being present. The default is sim-only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from systrophe.knopp.knopp_toroidal import EffectiveToroidalKerrBinary


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ----- circuit construction -----------------------------------------------


def toroidal_band_edge_circuit(
    binary: EffectiveToroidalKerrBinary, rho: float, T_period: float = 1.0,
) -> QuantumCircuit:
    """4-qubit toroidal-band-edge circuit at radius rho.

    Construction (derived in `band_edge_signature_derivation` below):

    Stage 1: Probe q[0] in |+>.
    Stage 2: Lense-Thirring phase on q[1]: q[1] absorbs phi = Omega_eff*T.
             Couples to q[0] through a CNOT.
    Stage 3: Tipler-tilt extinction rotation on q[0]:
             R_y(theta_T) with theta_T = pi * min(T_eff, 1).
             This is the key step:
               in band (T_eff >= 1):  theta_T = pi -> q[0] flipped fully.
               outside  (T_eff < 1):  theta_T < pi -> partial flip.
    Stage 4: Hadamard q[0] back, then a CNOT to entangle q[2,3] with
             the post-rotation state (so the band-edge signal propagates
             to the multi-qubit readout).
    Stage 5: Measure all qubits. The signature is in P(|0000>):

        P(|0000>) = cos^2(theta_T / 2)
                  ~ 0   at band edge (theta_T = pi)
                  ~ 1   outside band (theta_T = 0)
        with monotonic sigmoid shape across the boundary.
    """
    q = QuantumRegister(4, name="q")
    c = ClassicalRegister(4, name="meas")
    qc = QuantumCircuit(q, c)

    # Stage 1: probe |+>
    qc.h(q[0])

    # Stage 2: LT phase encoding on q[1]
    omega_eff = binary.omega_eff(rho)
    phi = float(omega_eff * T_period)
    qc.rz(2.0 * phi, q[0])
    qc.cx(q[0], q[1])

    # Stage 3: T_eff extinction rotation
    T_eff = binary.t_eff(rho, include_phi=False)
    theta_T = math.pi * min(T_eff, 1.0)
    qc.ry(theta_T, q[0])

    # Stage 4: H back, propagate signal to multi-qubit readout
    qc.h(q[0])
    qc.cx(q[0], q[2])
    qc.cx(q[0], q[3])

    # Stage 5: measure all
    qc.measure(q, c)
    return qc


# ----- run on simulator ---------------------------------------------------


def run_on_simulator(
    binary: EffectiveToroidalKerrBinary,
    rho_scan: np.ndarray,
    shots: int = 4096,
    seed: int = 42,
) -> dict:
    """Run the band-edge circuit at each rho on AerSimulator."""
    try:
        from qiskit_aer import AerSimulator
        from qiskit import transpile
    except ImportError:
        raise RuntimeError(
            "qiskit_aer not available. pip install qiskit-aer."
        )
    backend = AerSimulator(seed_simulator=seed)
    out = []
    for rho in rho_scan:
        qc = toroidal_band_edge_circuit(binary, float(rho))
        qc_t = transpile(qc, backend, optimization_level=2)
        result = backend.run(qc_t, shots=shots).result()
        counts = result.get_counts()
        p_zero = counts.get("0000", 0) / shots
        out.append({
            "rho": float(rho),
            "T_eff": binary.t_eff(float(rho), include_phi=False),
            "omega_eff": binary.omega_eff(float(rho)),
            "P_0000": p_zero,
            "depth": qc.depth(),
        })
    return {
        "binary": {"M": binary.M, "d": binary.d, "chi": binary.chi},
        "shots": shots,
        "n_rho": len(rho_scan),
        "results": out,
    }


# ----- run on hardware (submission-ready) --------------------------------


def submit_to_ibm_marrakesh(
    binary: EffectiveToroidalKerrBinary,
    rho_scan: np.ndarray,
    shots: int = 4096,
    backend_name: str = "ibm_marrakesh",
) -> dict:
    """Submit the rho-scan to ibm_marrakesh via Qiskit Runtime.

    Requires:
      - IBMQ_TOKEN environment variable, OR a configured
        QiskitRuntimeService account from `~/.qiskit/qiskit-ibm.json`.
      - Active Zynerji instance allocation.

    Pulls calibration via quantum_golden_pendulum.calibration.pull_calibration
    if available (per Systrophe project convention).
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError:
        raise RuntimeError(
            "qiskit_ibm_runtime not available. pip install qiskit-ibm-runtime."
        )

    # Authentication
    token = os.environ.get("IBMQ_TOKEN")
    if token:
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    else:
        service = QiskitRuntimeService(channel="ibm_quantum")

    backend = service.backend(backend_name)
    print(f"Using backend: {backend.name}")

    # Pull live calibration if available (Systrophe convention)
    calib_pulled = False
    try:
        from quantum_golden_pendulum.calibration import pull_calibration
        pull_calibration(backend)
        calib_pulled = True
        print("Calibration pulled via quantum_golden_pendulum.")
    except ImportError:
        print("quantum_golden_pendulum unavailable -- using backend default.")

    # Build all circuits
    circuits = []
    rho_list = []
    for rho in rho_scan:
        qc = toroidal_band_edge_circuit(binary, float(rho))
        circuits.append(qc)
        rho_list.append(float(rho))

    # Transpile with optimization_level=3 + DD + twirling (Systrophe convention)
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=3,
    )
    transpiled = pm.run(circuits)

    # Submit
    sampler = SamplerV2(backend)
    sampler.options.default_shots = shots
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = True

    print(f"Submitting {len(circuits)} circuits to {backend_name}...")
    t0 = time.time()
    job = sampler.run(transpiled)
    job_id = job.job_id()
    print(f"  job_id = {job_id}")
    print(f"  waiting for result...")
    result = job.result()
    t1 = time.time()
    print(f"  result in {t1 - t0:.1f}s")

    out = []
    for i, (rho, pub_result) in enumerate(zip(rho_list, result)):
        counts = pub_result.data.meas.get_counts()
        total = sum(counts.values())
        p_zero = counts.get("0000", 0) / total
        out.append({
            "rho": rho,
            "T_eff": binary.t_eff(rho, include_phi=False),
            "omega_eff": binary.omega_eff(rho),
            "P_0000": p_zero,
            "depth": transpiled[i].depth(),
        })

    return {
        "backend": backend_name,
        "job_id": job_id,
        "binary": {"M": binary.M, "d": binary.d, "chi": binary.chi},
        "shots": shots,
        "n_rho": len(rho_scan),
        "calibration_pulled": calib_pulled,
        "results": out,
    }


# ----- band-edge extinction analysis --------------------------------------


def band_edge_signature_score(results: list[dict]) -> dict:
    """Quantify the band-edge extinction: P_0000 should DROP inside the
    band (T_eff >= 1) and STAY HIGH outside (T_eff < 1).

    Returns the signature score (range [-1, 1]):
        score = mean(P_0000 outside band) - mean(P_0000 inside band).
    Positive = extinction signature visible; ~0 = no signature.

    For the noiseless cos^2(theta_T/2) circuit:
        outside band (theta_T < pi):  P_0000 = cos^2(theta_T/2) > 0
        in band      (theta_T = pi):  P_0000 = 0
    so the score is the mean of (cos^2 values outside band).
    """
    inside = [r["P_0000"] for r in results if r["T_eff"] >= 1.0]
    outside = [r["P_0000"] for r in results if r["T_eff"] < 1.0]
    if not inside or not outside:
        return {"score": 0.0, "inside_mean": None, "outside_mean": None,
                "n_inside": len(inside), "n_outside": len(outside)}
    inside_mean = float(np.mean(inside))
    outside_mean = float(np.mean(outside))
    return {
        "score": float(outside_mean - inside_mean),
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
        "n_inside": len(inside),
        "n_outside": len(outside),
    }


# ----- main entry ---------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true",
                        help="Run on AerSimulator (default).")
    parser.add_argument("--hardware", action="store_true",
                        help="Submit to ibm_marrakesh. Requires credentials.")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--n_rho", type=int, default=15,
                        help="Number of rho-scan points.")
    parser.add_argument("--backend", default="ibm_marrakesh")
    args = parser.parse_args()

    # Working configuration
    binary = EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.0)
    edges = binary.ctc_band_edges(include_phi=False)
    if edges[0] is not None:
        print(f"Toroidal band: rho in [{edges[0]:.3f}, {edges[1]:.3f}] M")
        rho_scan = np.linspace(0.3, 4.5, args.n_rho)
    else:
        print("Binary has no band; using fallback scan.")
        rho_scan = np.linspace(0.5, 5.0, args.n_rho)

    if args.hardware:
        print("Hardware mode: submitting to IBM Quantum...")
        out = submit_to_ibm_marrakesh(
            binary, rho_scan, shots=args.shots, backend_name=args.backend,
        )
        out_path = (RESULTS_DIR
                    / f"toroidal_band_edge_{args.backend}_{int(time.time())}.json")
    else:
        print("Simulator mode (default).")
        out = run_on_simulator(binary, rho_scan, shots=args.shots)
        out_path = (RESULTS_DIR
                    / f"toroidal_band_edge_sim_{int(time.time())}.json")

    score = band_edge_signature_score(out["results"])
    out["band_edge_score"] = score

    print()
    print(" rho     T_eff    P(|0000>)")
    for r in out["results"]:
        marker = "  *" if r["T_eff"] >= 1.0 else ""
        print(f" {r['rho']:6.3f}  {r['T_eff']:7.4f}   {r['P_0000']:.4f}{marker}")
    print()
    print(f"Band-edge extinction score: {score['score']:+.4f}")
    print(f"  inside-band mean P(|0000>):  {score['inside_mean']}")
    print(f"  outside-band mean P(|0000>): {score['outside_mean']}")
    print(f"  (positive score = extinction signature visible)")

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nresults written to {out_path}")


if __name__ == "__main__":
    main()
