"""Cross-chip Bell-violation correlation study.

We cannot establish true quantum entanglement between distinct
physical IBM Quantum processors (no shared photonic interconnect).
What we CAN do is run an identical Bell-pair circuit on three
Heron-r2 chips (ibm_marrakesh, ibm_fez, ibm_kingston) and ask
whether the catcher detects platform-specific deviations in the
4-setting CHSH measurement statistics.

Predicted outcome
=================
If the three chips are identical at the quantum level, all four
CHSH-setting bitstring distributions on chip X are platform-
independent up to noise. The catcher should report `smooth` when
comparing the four-distribution set across chips.

If a chip has systematic calibration drift, gate-error asymmetry,
or shared-environment decoherence (e.g., correlated 1/f noise),
the catcher should flag the chip-specific deviation as a sharp
Hamming-graph transition.

Circuit
=======
2-qubit Bell state: |Phi+> = (|00> + |11>) / sqrt 2
Four measurement settings: ZZ, ZX, XZ, XX.
Implemented as 4 circuits per chip, 8192 shots each.

Each chip sees the same 4 circuits. Total 12 circuits, 3 chips.

Submission
==========
Submitted to all 3 chips simultaneously. With each chip's queue
empty (pending=0 at submission time), all three jobs should run
in parallel within minutes.
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

from systrophe.novelty_catcher import (
    catch_novelty_in_distributions,
    catch_novelty_per_quantity,
)

SHOTS = 8192
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def bell_circuit(meas_setting: str) -> QuantumCircuit:
    """Bell |Phi+> with one of four measurement settings.

    `meas_setting` is one of 'ZZ', 'ZX', 'XZ', 'XX'.
    """
    qr = QuantumRegister(2, "q")
    cr = ClassicalRegister(2, "c")
    qc = QuantumCircuit(qr, cr, name=f"bell_{meas_setting}")
    # Prepare |Phi+>
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    # Apply basis rotation per measurement setting
    if meas_setting[0] == "X":
        qc.h(qr[0])
    if meas_setting[1] == "X":
        qc.h(qr[1])
    qc.measure(qr, cr)
    return qc


def chsh_violating_bell_circuit(meas_label: str) -> QuantumCircuit:
    """Bell |Phi+> with CHSH-violating rotated bases.

    CHSH-violating measurement angles:
      Alice (q0): angle 0 (A) or pi/2 (A')
      Bob   (q1): angle pi/4 (B) or 3 pi/4 (B')

    The four settings 'AB', 'AB_prime', 'A_primeB', 'A_primeB_prime'
    give CHSH parameter
      S = E(A,B) - E(A,B') + E(A',B) + E(A',B')
    with maximum 2 sqrt(2) at the Tsirelson bound for an ideal |Phi+>.

    `meas_label` in {'AB', 'AB_prime', 'A_primeB', 'A_primeB_prime'}.
    """
    qr = QuantumRegister(2, "q")
    cr = ClassicalRegister(2, "c")
    qc = QuantumCircuit(qr, cr, name=f"chsh_{meas_label}")
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])

    if meas_label == "AB":
        # A = Z (angle 0), B = (Z + X)/sqrt 2 (angle pi/4 about y)
        qc.ry(-math.pi / 4, qr[1])
    elif meas_label == "AB_prime":
        # A = Z, B' = (Z - X)/sqrt 2 (angle -pi/4)
        qc.ry(+math.pi / 4, qr[1])
    elif meas_label == "A_primeB":
        # A' = X (angle pi/2), B = (Z + X)/sqrt 2
        qc.ry(-math.pi / 2, qr[0])
        qc.ry(-math.pi / 4, qr[1])
    elif meas_label == "A_primeB_prime":
        # A' = X, B' = (Z - X)/sqrt 2
        qc.ry(-math.pi / 2, qr[0])
        qc.ry(+math.pi / 4, qr[1])
    else:
        raise ValueError(f"Unknown CHSH setting: {meas_label}")
    qc.measure(qr, cr)
    return qc


def all_chsh_settings() -> list[str]:
    return ["AB", "AB_prime", "A_primeB", "A_primeB_prime"]


def chsh_violating_S(E: dict[str, float]) -> float:
    """CHSH parameter with the rotated-basis convention:
        S = E_AB - E_AB' + E_A'B + E_A'B'
    Bell violation: |S| > 2; Tsirelson bound 2 sqrt 2 = 2.828.
    """
    return (
        E["AB"]
        - E["AB_prime"]
        + E["A_primeB"]
        + E["A_primeB_prime"]
    )


def chsh_from_counts(counts: dict[str, int]) -> float:
    """Expectation value <A B> from a 2-qubit measurement.

    For result bitstring 'ab', the eigenvalue is (-1)^(a+b).
    Sum of all eigenvalues / total = <A B>.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    e = 0
    for bs, c in counts.items():
        a, b = int(bs[0]), int(bs[1])
        sign = 1 if (a + b) % 2 == 0 else -1
        e += sign * c
    return float(e / total)


def chsh_S(e_zz: float, e_zx: float, e_xz: float, e_xx: float) -> float:
    """CHSH parameter S = <ZZ> - <ZX> + <XZ> + <XX>.

    Bell-CHSH violation: |S| > 2 (classical bound = 2; Tsirelson 2sqrt(2)).
    """
    return e_zz - e_zx + e_xz + e_xx


def all_settings() -> list[str]:
    return ["ZZ", "ZX", "XZ", "XX"]


def transpile_circuits(backend) -> list[QuantumCircuit]:
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    settings = all_settings()
    out = []
    for s in settings:
        qc = bell_circuit(s)
        isa = pm.run(qc)
        out.append(isa)
        print(f"  {s}: pre_depth={qc.depth()}, isa_depth={isa.depth()}",
              flush=True)
    return out


def submit_all_chips(instance: str = "Zynerji") -> dict[str, str]:
    """Submit the 4-circuit Bell test to all three Heron-r2 chips.

    Returns a dict {chip_name: job_id}.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_ibm_runtime.options import SamplerOptions

    service = QiskitRuntimeService(instance=instance)
    job_ids = {}
    for chip in ("ibm_marrakesh", "ibm_fez", "ibm_kingston"):
        backend = service.backend(chip)
        status = backend.status()
        print(f"\n[submit] backend={chip}  pending={status.pending_jobs}",
              flush=True)
        isa_circuits = transpile_circuits(backend)
        opts = SamplerOptions()
        opts.dynamical_decoupling.enable = True
        opts.dynamical_decoupling.sequence_type = "XpXm"
        opts.twirling.enable_gates = True
        opts.twirling.enable_measure = True
        opts.default_shots = SHOTS
        sampler = SamplerV2(mode=backend, options=opts)
        job = sampler.run(isa_circuits, shots=SHOTS)
        job_id = job.job_id()
        job_ids[chip] = job_id
        print(f"[submit] {chip} job_id={job_id}", flush=True)
    out_path = RESULTS_DIR / "cross_chip_bell_submitted.json"
    out_path.write_text(json.dumps({
        "job_ids": job_ids,
        "submitted_unix": int(time.time()),
        "settings": all_settings(),
        "shots": SHOTS,
    }, indent=2))
    print(f"\n[submit] wrote {out_path}", flush=True)
    return job_ids


def recover_from_jobs(
    job_ids: dict[str, str], instance: str = "Zynerji",
) -> dict:
    """Pull results from all 3 chips and run the catcher comparison."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(instance=instance)
    per_chip: dict[str, dict] = {}
    settings = all_settings()
    for chip, job_id in job_ids.items():
        print(f"\n[recover] {chip} job={job_id}", flush=True)
        job = service.job(job_id)
        st = str(job.status())
        if st not in ("DONE", "COMPLETED"):
            print(f"  status={st}; skipping")
            per_chip[chip] = {"status": st}
            continue
        result = job.result()
        chip_data = {"settings": {}}
        E = {}
        for i, s in enumerate(settings):
            data = result[i].data
            creg_name = next(iter(data))
            counts = getattr(data, creg_name).get_counts()
            E[s] = chsh_from_counts(counts)
            chip_data["settings"][s] = {
                "counts": dict(counts),
                "expectation": E[s],
            }
        S = chsh_S(E["ZZ"], E["ZX"], E["XZ"], E["XX"])
        chip_data["E"] = E
        chip_data["S_CHSH"] = float(S)
        chip_data["bell_violated"] = bool(abs(S) > 2.0)
        per_chip[chip] = chip_data
        print(f"  E: ZZ={E['ZZ']:+.3f} ZX={E['ZX']:+.3f} "
              f"XZ={E['XZ']:+.3f} XX={E['XX']:+.3f}")
        print(f"  S_CHSH = {S:+.4f}  (classical bound 2.0, "
              f"Tsirelson {2*math.sqrt(2):.4f})")
        print(f"  Bell-violated: {abs(S) > 2.0}")

    # Per-quantity catcher across chips for each setting
    per_q: dict[str, dict] = {}
    for s in settings:
        per_q[f"E_{s}"] = {}
        for chip, data in per_chip.items():
            if "E" in data:
                per_q[f"E_{s}"][chip] = np.array([data["E"][s]])
    cross_chip_nov = catch_novelty_per_quantity(per_q)

    # Catcher on the 4-setting distribution per chip vs each other
    distributions_by_chip = []
    chip_names = []
    for chip, data in per_chip.items():
        if "E" in data:
            vec = np.array([data["E"][s] for s in settings])
            distributions_by_chip.append(vec)
            chip_names.append(chip)
    catcher_across_chips = catch_novelty_in_distributions(
        distributions_by_chip, labels=chip_names,
    ) if len(distributions_by_chip) >= 2 else None

    out = {
        "per_chip": per_chip,
        "cross_chip_per_quantity_catcher": cross_chip_nov,
        "cross_chip_distribution_catcher": catcher_across_chips,
    }
    out_path = RESULTS_DIR / "cross_chip_bell_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[recover] wrote {out_path}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--instance", default="Zynerji")
    args = parser.parse_args()
    if args.submit:
        submit_all_chips(instance=args.instance)
    if args.recover:
        submitted = json.loads(
            (RESULTS_DIR / "cross_chip_bell_submitted.json").read_text()
        )
        recover_from_jobs(submitted["job_ids"], instance=args.instance)
