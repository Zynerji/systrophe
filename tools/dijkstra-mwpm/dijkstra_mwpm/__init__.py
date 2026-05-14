"""dijkstra-mwpm: a Dijkstra-shortest-path MWPM decoder for the rotated
surface code.

Provenance: this is the decoder that the Systrophe v0.19.2 release
landed on as the biggest single software win in the Heron-r2 QEC
program (`paper/surface_code_multidistance_break_even.pdf`). On the
exact same hardware data, switching from a naive single-qubit-
boundary-flip matching to a full Dijkstra-shortest-path chain
correction recovered **+5-25 percentage points** in logical-zero
probability across the d=5 round-sweep, and was the change that
sustained the d=5 and d=7 break-even crossings beyond n_rounds=1.

This tool packages the decoder as a standalone, lightly-typed library
that does not require Qiskit, IBM Quantum, or the Systrophe core. The
inputs are pure Python: a tuple of data-qubit bits, a list of Z-
syndrome bit-tuples per round, and the code distance d. The output is
the decoded logical bit.

Distinction from existing MWPM libraries (pymatching, etc.): this is
the **specific** flavour that fits the Systrophe Heron-r2 pipeline --
syndrome-difference history per round, plus a 1-cost virtual boundary
for unpaired stabs, plus a `min_weight_matching` from networkx. The
algorithm is documented in `decoder.py`; reference numbers come from
`paper/surface_code_multidistance_break_even.pdf` and from the JSON
files under `Systrophe/experiments/results/surface_code_d5_dijkstra_mwpm_analysis.json`
and `surface_code_d7_and_d5_high_shots_dijkstra_analysis.json`.
"""

from .surface_code import (
    build_stabilizers,
    data_index,
)
from .decoder import (
    MWPMDecoder,
    build_stab_adjacency,
    decode_with_dijkstra_mwpm,
    dijkstra_path,
)
from .naive import decode_with_naive_mwpm

__all__ = [
    "MWPMDecoder",
    "build_stabilizers",
    "build_stab_adjacency",
    "data_index",
    "decode_with_dijkstra_mwpm",
    "decode_with_naive_mwpm",
    "dijkstra_path",
]

__version__ = "0.1.0"
