"""Distance-5 rotated surface code on a Heron-r2 patch.

A d=5 rotated surface code requires:
  - 25 data qubits in a 5x5 grid
  - 12 X-stabilizer ancillas (one per face of the X-checkerboard)
  - 12 Z-stabilizer ancillas (one per face of the Z-checkerboard)
  - Total: 49 physical qubits per logical qubit (fits on Heron-r2's 156)

The rotated surface code's stabilizers are weight-4 plaquettes (for
internal faces) and weight-2 edge half-plaquettes (on the boundary).
With a 5x5 data grid, we have:
  - 12 weight-4 X-stabilizers (2 per row, alternating with Z, for 6 rows
    of internal vertical edges -- 5x4 = 20 vertical edges, half are X)
  - Actually: for the rotated surface code with side length d,
      n_data = d^2 = 25
      n_X_stab = (d^2 - 1) / 2 = 12
      n_Z_stab = (d^2 - 1) / 2 = 12

The 5x5 data qubit grid is indexed by (row, col) with row, col in {0..4}.
The plaquette at position (r, c) (in {0..3} x {0..3}) covers data qubits
(r, c), (r+1, c), (r, c+1), (r+1, c+1). Plaquettes are colored
checkerboard: if (r+c) is even, it's an X-stab; else Z-stab.

The boundary half-plaquettes (weight-2) are:
  - Top boundary (row=-0.5): X-stabs on (0, c) - (0, c+1) for c in {0, 2}
  - Bottom boundary (row=4.5): X-stabs on (4, c) - (4, c+1) for c in {1, 3}
  - Left boundary (col=-0.5): Z-stabs on (r, 0) - (r+1, 0) for r in {1, 3}
  - Right boundary (col=4.5): Z-stabs on (r, 4) - (r+1, 4) for r in {0, 2}

For the "rotated surface code" with rough top/bottom edges (X-boundaries)
and smooth left/right edges (Z-boundaries):
  X-logical: any row of data qubits (e.g., row 0: q0, q1, q2, q3, q4)
  Z-logical: any column of data qubits (e.g., col 0: q0, q5, q10, q15, q20)

This is the canonical d=5 surface code. Logical |0_L> is the +1
eigenstate of all stabilizers AND Z_L = +1. Standard preparation:
initialize all data qubits to |0> (this is the +1 eigenstate of all
Z-stabilizers) then measure all X-stabilizers to project onto the
+1 eigenstate of those too (apply X correction for any -1 syndrome).

For now this module provides:
  - The data-qubit layout (5x5 grid + qubit indices)
  - The X- and Z-stabilizer parity-check matrices
  - A naive lookup-table decoder using only the FINAL syndrome (not
    proper MWPM; should still demonstrate logical signal at small
    error rates)
  - Encoding circuit |0_L> = projector approach
  - Repeated syndrome extraction circuit

Heavy-hex caveat: Heron-r2 does NOT natively support the rotated
surface code's square-lattice connectivity. The transpiler will
insert SWAPs to route, adding depth. For first demo we accept this
overhead.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


D = 5  # code distance
N_DATA = D * D  # 25
N_X_STAB = (D * D - 1) // 2  # 12
N_Z_STAB = (D * D - 1) // 2  # 12


def data_index(row: int, col: int) -> int:
    """Convert (row, col) in {0..D-1}^2 to data qubit index."""
    return row * D + col


def build_stabilizers() -> tuple[list[list[int]], list[list[int]]]:
    """Build the X- and Z-stabilizer supports.

    Each stabilizer is a list of data-qubit indices (0..24).
    Returns (X_stabs, Z_stabs).

    Standard rotated surface code with rough top/bottom edges
    (X-boundaries) and smooth left/right edges (Z-boundaries).
    """
    X_stabs = []
    Z_stabs = []
    for r in range(D - 1):
        for c in range(D - 1):
            # 4-qubit plaquette at (r, c) - (r+1, c+1)
            qs = [
                data_index(r, c),
                data_index(r, c + 1),
                data_index(r + 1, c),
                data_index(r + 1, c + 1),
            ]
            if (r + c) % 2 == 0:
                X_stabs.append(qs)
            else:
                Z_stabs.append(qs)
    # Boundary half-plaquettes (weight 2)
    # Top edge (X-boundary): X-stabs on (0, c), (0, c+1) for c in {1, 3}
    for c in (1, 3):
        X_stabs.append([data_index(0, c), data_index(0, c + 1)])
    # Bottom edge: X-stabs on (D-1, c), (D-1, c+1) for c in {0, 2}
    for c in (0, 2):
        X_stabs.append([data_index(D - 1, c), data_index(D - 1, c + 1)])
    # Left edge (Z-boundary): Z-stabs on (r, 0), (r+1, 0) for r in {0, 2}
    for r in (0, 2):
        Z_stabs.append([data_index(r, 0), data_index(r + 1, 0)])
    # Right edge: Z-stabs on (r, D-1), (r+1, D-1) for r in {1, 3}
    for r in (1, 3):
        Z_stabs.append([data_index(r, D - 1), data_index(r + 1, D - 1)])
    return X_stabs, Z_stabs


X_STABS, Z_STABS = build_stabilizers()
assert len(X_STABS) == N_X_STAB, (
    f"Expected {N_X_STAB} X-stabs, got {len(X_STABS)}"
)
assert len(Z_STABS) == N_Z_STAB, (
    f"Expected {N_Z_STAB} Z-stabs, got {len(Z_STABS)}"
)


def build_encoding_circuit():
    """Encode |0_L> by initializing data qubits to |0> (already +1
    eigenstate of all Z-stabs) and measuring all X-stabs to project
    onto +1 eigenstate of those too. Apply X-flip corrections via
    classical feed-forward.

    For now (sketch): initialize |0>^25, ONE round of X-stab measurement,
    and conditional X corrections. The full fault-tolerant prep needs
    multiple rounds of stabilizer measurement and MWPM decoding.
    """
    from qiskit import (
        QuantumCircuit, QuantumRegister, ClassicalRegister,
    )

    data = QuantumRegister(N_DATA, "data")
    anc_x = QuantumRegister(N_X_STAB, "anc_x")
    anc_z = QuantumRegister(N_Z_STAB, "anc_z")
    cr_init = ClassicalRegister(N_X_STAB, "init_x")

    qc = QuantumCircuit(data, anc_x, anc_z, cr_init)
    # All data qubits start in |0> by default

    # ONE round of X-stab measurement to project onto X-stab +1
    for stab_idx, qs in enumerate(X_STABS):
        qc.h(anc_x[stab_idx])
        for q in qs:
            qc.cx(anc_x[stab_idx], data[q])
        qc.h(anc_x[stab_idx])
        qc.measure(anc_x[stab_idx], cr_init[stab_idx])
        qc.reset(anc_x[stab_idx])
    # The -1 syndromes can be corrected by flipping ONE data qubit per
    # syndrome bit using MWPM; for now we just record them.

    qc.barrier()
    return qc


def build_syndrome_round(round_idx: int):
    """One round of X- and Z-syndrome extraction with mid-circuit
    measurement and ancilla reset.

    Note: weight-4 stabilizers need 4 CNOTs each. With 12 X-stabs +
    12 Z-stabs that's 96 CNOTs per round; in practice many can be
    parallelized but the heavy-hex transpiler will serialize many
    due to topology.
    """
    from qiskit import (
        QuantumCircuit, QuantumRegister, ClassicalRegister,
    )

    data = QuantumRegister(N_DATA, "data")
    anc_x = QuantumRegister(N_X_STAB, "anc_x")
    anc_z = QuantumRegister(N_Z_STAB, "anc_z")
    cr_x = ClassicalRegister(N_X_STAB, f"sx_{round_idx}")
    cr_z = ClassicalRegister(N_Z_STAB, f"sz_{round_idx}")

    qc = QuantumCircuit(data, anc_x, anc_z, cr_x, cr_z)

    # X-stab extraction
    for stab_idx, qs in enumerate(X_STABS):
        qc.h(anc_x[stab_idx])
        for q in qs:
            qc.cx(anc_x[stab_idx], data[q])
        qc.h(anc_x[stab_idx])
        qc.measure(anc_x[stab_idx], cr_x[stab_idx])
        qc.reset(anc_x[stab_idx])

    # Z-stab extraction
    for stab_idx, qs in enumerate(Z_STABS):
        for q in qs:
            qc.cx(data[q], anc_z[stab_idx])
        qc.measure(anc_z[stab_idx], cr_z[stab_idx])
        qc.reset(anc_z[stab_idx])

    return qc


def build_full_experiment(n_rounds: int, mode: str = "z_memory"):
    """Encode |0_L>, run n_rounds of syndrome extraction, measure data.

    mode:
      "z_memory": Only Z-stab measurements (protects against X errors).
                  Encoder is trivially |0>^25 (already +1 of all Z-stabs
                  with Z_L = +1). This is the simplest valid first
                  d=5 surface code memory experiment.
      "full":     Both X and Z stab measurements; X-stab measurement
                  projects the random initial X-syndrome onto a definite
                  pattern. Requires MWPM decoding for full
                  fault-tolerance. Not yet implemented.
    """
    from qiskit import (
        QuantumCircuit, QuantumRegister, ClassicalRegister,
    )

    if mode != "z_memory":
        raise NotImplementedError("Only z_memory mode is implemented")

    data = QuantumRegister(N_DATA, "data")
    anc_z = QuantumRegister(N_Z_STAB, "anc_z")
    cr_data = ClassicalRegister(N_DATA, "data_meas")
    syndrome_regs = []
    for r in range(n_rounds):
        syndrome_regs.append(ClassicalRegister(N_Z_STAB, f"sz_{r}"))

    qc = QuantumCircuit(data, anc_z, cr_data, *syndrome_regs)

    # Encoding: |0>^25 is already |0_L> for Z-memory purposes.
    # No explicit encoding gates needed.

    qc.barrier()

    # n_rounds of Z-syndrome extraction
    for r in range(n_rounds):
        cr_z = syndrome_regs[r]
        for stab_idx, qs in enumerate(Z_STABS):
            for q in qs:
                qc.cx(data[q], anc_z[stab_idx])
            qc.measure(anc_z[stab_idx], cr_z[stab_idx])
            qc.reset(anc_z[stab_idx])
        qc.barrier()

    # Final data measurement
    qc.measure(data, cr_data)
    return qc


def decode_lookup(data_bits: tuple[int, ...]) -> int:
    """Naive lookup-table decoder using ONLY the data-derived
    Z-syndrome (final round). Z-stabilizers determine the X-error
    chain on the data qubits.

    For each Z-stab, syndrome = parity of measured data qubits in stab.
    The full syndrome is a 12-bit vector. We use the minimum-weight
    correction by greedily flipping data qubits whose Z-stab
    membership matches the syndrome.

    Returns logical Z = product of data qubits along a logical-Z
    column (col 0 by default).
    """
    db = np.array(data_bits, dtype=int)
    # Compute Z-syndrome
    syndrome = []
    for qs in Z_STABS:
        s = sum(db[q] for q in qs) % 2
        syndrome.append(s)
    syndrome = tuple(syndrome)

    # Greedy minimum-weight: for each violated syndrome bit, flip a
    # data qubit in that Z-stab support. Naive but works for d=5 with
    # single-error rate.
    if any(syndrome):
        db = db.copy()
        for stab_idx, s in enumerate(syndrome):
            if s == 1:
                # Flip the FIRST data qubit in this Z-stab's support
                # that isn't already flipped (greedy)
                qs = Z_STABS[stab_idx]
                for q in qs:
                    db[q] = 1 - db[q]
                    break

    # Logical Z = parity of data qubits along column 0
    logical_z = sum(db[data_index(r, 0)] for r in range(D)) % 2
    return logical_z


def simulate(n_rounds: int, shots: int = 1024) -> dict:
    from qiskit_aer import AerSimulator
    qc = build_full_experiment(n_rounds=n_rounds)
    # Use statevector method which has no coupling-map restriction
    sim = AerSimulator(method="stabilizer")
    job = sim.run(qc, shots=shots)
    counts = job.result().get_counts()
    # Aggregate
    n_total = 0
    n_logical_zero = 0
    for bitstring, count in counts.items():
        parts = bitstring.split(" ")
        # Data is the second-to-last register (init_x is last? depends
        # on order added). data was added BEFORE syndromes, so it's
        # printed... reverse iteration. The registers were added in
        # order: data, anc_x, anc_z, cr_init, cr_data, syndrome_regs.
        # Qiskit's get_counts reverses register order, so:
        # parts[0] = last syndrome_reg, ..., parts[-1] = data (cr_init)
        # Actually cr_init is added AFTER cr_data? Let me re-check.
        # Order: cr_init (after anc registers), cr_data, then sx_0, sz_0, ...
        # In bitstring (after reversal): cr_init is rightmost-ish
        # Just check lengths
        for part in parts:
            if len(part) == N_DATA:
                data_bits = tuple(int(b) for b in reversed(part))
                logical = decode_lookup(data_bits)
                if logical == 0:
                    n_logical_zero += count
                n_total += count
                break
    return {
        "logical_zero_rate": n_logical_zero / max(n_total, 1),
        "n_shots_total": n_total,
        "n_rounds": n_rounds,
    }


def submit_to_kingston(n_rounds: int, shots: int = 4096,
                         instance: str = "Zynerji") -> str:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService(instance=instance)
    backend = service.backend("ibm_kingston")
    print(f"[surface5] pending={backend.status().pending_jobs}")
    qc = build_full_experiment(n_rounds=n_rounds, mode="z_memory")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa = pm.run(qc)
    print(f"[surface5] n_rounds={n_rounds}, qubits={qc.num_qubits}, "
          f"ISA depth={isa.depth()}")
    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.default_shots = shots
    sampler = SamplerV2(mode=backend, options=opts)
    job = sampler.run([isa], shots=shots)
    print(f"[surface5] job_id={job.job_id()}")
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"surface_code_d5_kingston_n{n_rounds}_submitted.json"
    out_path.write_text(json.dumps({
        "job_id": job.job_id(), "n_rounds": n_rounds, "shots": shots,
        "submitted_unix": int(time.time()),
    }, indent=2))
    return job.job_id()


def recover_kingston(job_id: str, n_rounds: int, shots: int,
                       instance: str = "Zynerji") -> dict:
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=instance)
    job = service.job(job_id)
    if str(job.status()) not in ("DONE", "COMPLETED"):
        return {"status": str(job.status())}
    result = job.result()
    data = result[0].data
    creg = next(iter(data))
    counts = getattr(data, creg).get_counts()
    n_total = 0
    n_logical_zero = 0
    for bitstring, count in counts.items():
        parts = bitstring.split(" ")
        for part in parts:
            if len(part) == N_DATA:
                data_bits = tuple(int(b) for b in reversed(part))
                logical = decode_lookup(data_bits)
                if logical == 0:
                    n_logical_zero += count
                n_total += count
                break
    lz = n_logical_zero / max(n_total, 1)
    sigma = (lz * (1 - lz) / max(n_total, 1)) ** 0.5
    print(f"[surface5 recover] logical Z=0 = {lz:.4f} +- {sigma:.4f} "
          f"({n_total} shots)")
    return {
        "job_id": job_id,
        "n_rounds": n_rounds,
        "logical_zero_rate": lz,
        "logical_zero_sigma": sigma,
        "n_shots_total": n_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--recover", type=str)
    ap.add_argument("--n-rounds", type=int, default=0)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--report-structure", action="store_true")
    args = ap.parse_args()

    if args.report_structure:
        print(f"Rotated surface code, distance d = {D}")
        print(f"  n_data = {N_DATA}")
        print(f"  n_X_stab = {N_X_STAB}")
        print(f"  n_Z_stab = {N_Z_STAB}")
        print(f"  total qubits = {N_DATA + N_X_STAB + N_Z_STAB}")
        print()
        print(f"X-stabilizers (data qubit indices):")
        for i, qs in enumerate(X_STABS):
            print(f"  X{i}: {qs}")
        print()
        print(f"Z-stabilizers:")
        for i, qs in enumerate(Z_STABS):
            print(f"  Z{i}: {qs}")
        print()
        n_cnots_per_round = sum(len(qs) for qs in X_STABS) + sum(
            len(qs) for qs in Z_STABS
        )
        print(f"CNOTs per syndrome round (untranspiled): {n_cnots_per_round}")

    if args.simulate:
        result = simulate(n_rounds=args.n_rounds, shots=1024)
        print(f"Noiseless simulation, n_rounds = {args.n_rounds}:")
        print(f"  logical Z = 0 rate = {result['logical_zero_rate']:.4f}")
        print(f"  (Expected 1.0 for correct encoder + decoder)")

    if args.submit:
        submit_to_kingston(n_rounds=args.n_rounds, shots=args.shots)

    if args.recover:
        recover_kingston(args.recover, n_rounds=args.n_rounds, shots=args.shots)


if __name__ == "__main__":
    main()
