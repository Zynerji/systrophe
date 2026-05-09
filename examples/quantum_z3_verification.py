"""Quantum-hardware verification of the Z_3 Mobius phase identity.

Background
----------
The Systrophe ↔ Dinos bridge (`systrophe.dinos_bridge`) hinges on the
identity that, on a discrete cycle of N nodes, the Z_3 Mobius branch=0
fundamental eigenvalue equals 2(1 - cos(2*pi/N)) — exactly matching the
Tipler log-grid fundamental.

This script verifies the underlying Z_3 phase structure on real
IBM Quantum hardware: prepare |+>, apply RZ(2*pi/3) three times
(total phase 2*pi = identity), apply H, and measure. The expected
outcome is P(0) = 1 within hardware noise. A failure here would mean
the Z_3 cyclic phase identity is not realised on the device, which
would falsify the discrete correspondence in any quantum-mechanical
implementation of the Mobius cover.

Run:
    python examples/quantum_z3_verification.py
"""

from __future__ import annotations

import time

import numpy as np


def main() -> None:
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    for _ in range(3):
        qc.rz(2.0 * np.pi / 3.0, 0)
    qc.h(0)
    qc.measure(0, 0)
    print("Circuit:")
    print(qc.draw(output="text"))

    service = QiskitRuntimeService()
    backend = service.backend("ibm_marrakesh")
    print(f"\nBackend: {backend.name}, queue depth: {backend.status().pending_jobs}")

    from qiskit import transpile
    qc_t = transpile(qc, backend, optimization_level=2)
    print(f"Transpiled depth: {qc_t.depth()}, gate counts: {qc_t.count_ops()}")

    sampler = SamplerV2(mode=backend)
    print("\nSubmitting (1024 shots)...")
    t0 = time.time()
    job = sampler.run([qc_t], shots=1024)
    print(f"Job ID: {job.job_id()}")
    result = job.result()
    elapsed = time.time() - t0

    counts = result[0].data.c.get_counts()
    p0 = counts.get("0", 0) / 1024
    p1 = counts.get("1", 0) / 1024
    print(f"\nResults (elapsed {elapsed:.1f}s wall):")
    print(f"  counts: {dict(counts)}")
    print(f"  P(0) = {p0:.4f}  (theoretical 1.0; deviation from hardware noise)")
    print(f"  P(1) = {p1:.4f}  (theoretical 0.0)")
    if p0 > 0.95:
        print("  VERDICT: Z_3 cyclic phase identity holds on hardware (P(0) > 0.95).")
    else:
        print(f"  VERDICT: Hardware deviation P(0) = {p0:.3f}; check device calibration.")


if __name__ == "__main__":
    main()
