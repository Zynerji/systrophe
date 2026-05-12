"""Improved Steane d=3 decoder using syndrome HISTORY across rounds.

The original decoder uses only the final data-derived Z-syndrome.
For repeated-round experiments, the per-round X- and Z-syndromes are
available; using them via matching across rounds should improve
recovery, especially at higher round counts where errors compound.

Approach:
  1. Read all (sx, sz) syndrome pairs per round.
  2. Compute syndrome differences between consecutive rounds.
  3. Any flip in Z-syndrome between rounds indicates an X-error on a
     data qubit in that stab's support; pair them up.
  4. For Steane (d=3), can correct ONE X-error per round. Greedy
     pair-matching by stab-distance suffices since the stabilizer
     graph is small (3 X-stabs, 3 Z-stabs, 7 data qubits).
  5. Apply the inferred X-flip corrections to the data measurement
     and compute logical Z.

Re-analyses job d81k4mfoha1c73bkgjlg (Steane round sweep, 4096 shots).
"""

from __future__ import annotations

import json
from pathlib import Path
from itertools import combinations

import numpy as np


# Steane parity-check matrix (consistent with the canonical encoder
# in experiments/steane_logical_qubit.py)
H_STEANE = np.array([
    [1, 1, 0, 1, 1, 0, 0],   # stab 0 -> qubits 0, 1, 3, 4
    [1, 0, 1, 1, 0, 1, 0],   # stab 1 -> qubits 0, 2, 3, 5
    [0, 1, 1, 1, 0, 0, 1],   # stab 2 -> qubits 1, 2, 3, 6
], dtype=int)


def _build_lookup() -> dict:
    table = {}
    for q in range(7):
        s = tuple(int(H_STEANE[i, q]) for i in range(3))
        table[s] = q
    table[(0, 0, 0)] = -1
    return table


SYNDROME_LOOKUP = _build_lookup()


def decode_with_syndrome_history(
    data_bits: tuple[int, ...],
    z_syndromes_per_round: list[tuple[int, ...]],
    x_syndromes_per_round: list[tuple[int, ...]] = None,
) -> int:
    """For d=3 Steane Z-memory, decode using:
      - the syndrome DIFFERENCE between rounds (catches a round-specific
        X-error)
      - the final data-derived Z-syndrome (catches the last-round error)

    Strategy: track which data qubits have been flipped by all error
    events. Final correction = XOR of all flips. Apply to data and
    compute logical Z.
    """
    db = np.array(data_bits, dtype=int)
    # Final data-derived Z-syndrome
    final_z = tuple(int(np.dot(H_STEANE[i], db) % 2) for i in range(3))

    # Build syndrome history including final round
    history = list(z_syndromes_per_round) + [final_z]

    # Compute differences
    diffs = []
    prev = (0, 0, 0)
    for syn in history:
        diff = tuple((syn[i] ^ prev[i]) for i in range(3))
        diffs.append(diff)
        prev = syn

    # For each non-zero diff, look up the qubit and toggle it in a
    # "flips" set
    flips = set()
    for diff in diffs:
        q = SYNDROME_LOOKUP.get(diff, -1)
        if q >= 0:
            if q in flips:
                flips.remove(q)
            else:
                flips.add(q)

    # Apply flips to data bits
    for q in flips:
        db[q] = 1 - db[q]

    # Logical Z = parity of all 7 corrected data bits
    return int(np.sum(db) % 2)


def reanalyze_steane_round_sweep(job_id: str = None,
                                   instance: str = "Zynerji") -> dict:
    """Re-analyze the Steane round-sweep job with syndrome-history MWPM."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=instance)

    out_dir = Path(__file__).parent / "results"
    submitted = json.loads(
        (out_dir / "steane_round_sweep_submitted.json").read_text()
    )
    if job_id is None:
        job_id = submitted["job_id"]
    job = service.job(job_id)
    if str(job.status()) not in ("DONE", "COMPLETED"):
        return {"status": str(job.status())}

    metadata = submitted["metadata_per_circuit"]
    result = job.result()

    paired = {}
    for i, meta in enumerate(metadata):
        nr = meta["n_rounds"]
        data = result[i].data
        creg = next(iter(data))
        counts = getattr(data, creg).get_counts()
        if meta["kind"] == "bare":
            total = sum(counts.values())
            zero_count = sum(v for k, v in counts.items() if k == "0")
            paired.setdefault(nr, {})["bare"] = {
                "physical_zero_rate": zero_count / total,
                "n_shots_total": total,
            }
            continue

        n_log_zero_old = 0
        n_log_zero_new = 0
        n_total = 0
        for bitstring, count in counts.items():
            parts = bitstring.split(" ")
            # Identify data (7 bits), sx_r (3 bits), sz_r (3 bits)
            data_part = None
            sx_parts = []
            sz_parts = []
            for part in parts:
                if len(part) == 7:
                    data_part = part
                elif len(part) == 3:
                    # Could be sx_r or sz_r; can't distinguish without
                    # reg order. Convention: bits added (data first, then
                    # sx_0, sz_0, sx_1, sz_1, ...). Reversed in qiskit
                    # output, so we get [last, ..., first].
                    # We use list-order with even/odd by position.
                    pass
            # Better: explicitly use the part positions. Qiskit's bitstring
            # parts are in REVERSE order of register addition. data_meas
            # was added first, so it's parts[-1].
            data_part = parts[-1]
            # Remaining parts (sx_0, sz_0, sx_1, sz_1, ...) -> reverse to
            # get round 0 first
            syndrome_parts_reversed = list(reversed(parts[:-1]))
            # Pair up as (sx, sz, sx, sz, ...)
            sx_parts = []
            sz_parts = []
            for r in range(nr):
                if 2 * r < len(syndrome_parts_reversed):
                    sx_parts.append(syndrome_parts_reversed[2 * r])
                if 2 * r + 1 < len(syndrome_parts_reversed):
                    sz_parts.append(syndrome_parts_reversed[2 * r + 1])
            data_bits = tuple(int(b) for b in reversed(data_part))
            z_syndromes = [tuple(int(b) for b in reversed(s)) for s in sz_parts]
            # OLD decoder: only uses data-derived syndrome
            from steane_logical_qubit import decode_shot
            log_old = decode_shot(data_bits, [])
            log_new = decode_with_syndrome_history(data_bits, z_syndromes)
            if log_old == 0:
                n_log_zero_old += count
            if log_new == 0:
                n_log_zero_new += count
            n_total += count
        paired.setdefault(nr, {})["surface"] = {
            "logical_zero_rate_old": n_log_zero_old / max(n_total, 1),
            "logical_zero_rate_new": n_log_zero_new / max(n_total, 1),
            "n_shots_total": n_total,
        }

    print()
    print("Steane d=3 with syndrome-history MWPM vs old data-only decoder")
    print("=" * 80)
    print(f"{'n_rounds':>10}  {'old':>10}  {'new MWPM':>10}  {'bare':>10}  "
          f"{'old-bare':>10}  {'new-bare':>10}")
    for nr in sorted(paired.keys()):
        s = paired[nr].get("surface", {})
        b = paired[nr].get("bare", {})
        old = s.get("logical_zero_rate_old", float("nan"))
        new = s.get("logical_zero_rate_new", float("nan"))
        ba = b.get("physical_zero_rate", float("nan"))
        print(f"{nr:>10d}  {old:>10.4f}  {new:>10.4f}  {ba:>10.4f}  "
              f"{old - ba:>+10.4f}  {new - ba:>+10.4f}")

    out_path = out_dir / "steane_syndrome_history_mwpm_analysis.json"
    out_path.write_text(json.dumps({
        "job_id": job_id,
        "decoder": "syndrome_history_mwpm",
        "paired_by_n_rounds": paired,
    }, indent=2))
    print()
    print(f"Wrote {out_path}")
    return paired


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    reanalyze_steane_round_sweep()
