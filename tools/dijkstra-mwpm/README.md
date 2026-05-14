# dijkstra-mwpm

**A Dijkstra-shortest-path MWPM decoder for the rotated surface code.**

The decoder that the Systrophe v0.19.2 QEC release identified as the
single biggest software win on the IBM Heron-r2 hardware program:
switching from a naive single-qubit-boundary-flip MWPM to a full
Dijkstra-shortest-path chain correction recovered **+5 to +25
percentage points** in logical-zero rate across the d=5 round-sweep,
and was the change that sustained break-even past `n_rounds=1` for
both d=5 and d=7.

This tool packages the decoder as a small standalone library that:

* does **not** require Qiskit, IBM Quantum, or the Systrophe core;
* takes pure-Python inputs (`tuple[int, ...]` data bits, list of
  per-round Z-syndrome tuples, distance `d`);
* exposes both procedural (`decode_with_dijkstra_mwpm`) and OO
  (`MWPMDecoder`) entry points;
* ships with a synthetic-noise A/B benchmark that **reproduces the
  Heron-r2 delta** on a benchmark that runs in 3 seconds on a laptop.

## Provenance

* Algorithm and reference implementation:
  `Systrophe/experiments/surface_code_dijkstra_mwpm.py`
* Hardware reference numbers: `Systrophe/paper/surface_code_multidistance_break_even.pdf`
* Heron-r2 result JSON (d=5 round sweep):
  `Systrophe/experiments/results/surface_code_d5_dijkstra_mwpm_analysis.json`

## Layout

```
dijkstra-mwpm/
├── README.md
├── dijkstra_mwpm/
│   ├── __init__.py
│   ├── surface_code.py        build_stabilizers + data_index (rotated d x d)
│   ├── decoder.py             decode_with_dijkstra_mwpm + dijkstra_path + MWPMDecoder
│   └── naive.py               decode_with_naive_mwpm (control)
├── examples/
│   └── compare_naive_vs_dijkstra.py     reproduces the +5-25pp delta
├── tests/
│   └── test_decoder.py        16 tests
└── requirements.txt
```

## Quick start

```python
from dijkstra_mwpm import MWPMDecoder

# d=5 rotated surface code; 24 Z-stabs.
decoder = MWPMDecoder(d=5)
print(decoder.n_z_stabs)   # 12 (half the stabs are Z)

# A perfect |0>^25 prep + 2 noise-free rounds + clean final measurement:
data_bits = (0,) * 25
z_syndromes = [(0,) * decoder.n_z_stabs] * 2
logical_z = decoder.decode(data_bits, z_syndromes)
print(logical_z)   # 0
```

## A/B reproduction of the Heron-r2 delta

```
$ PYTHONPATH=tools/dijkstra-mwpm python tools/dijkstra-mwpm/examples/compare_naive_vs_dijkstra.py
d=5  n_shots=200  n_rounds=2

       p       naive    dijkstra    delta_pp
---------------------------------------------
  0.0050      0.965      0.995       +3.00
  0.0100      0.915      0.975       +6.00
  0.0200      0.880      0.955       +7.50
  0.0400      0.805      0.950      +14.50
  0.0800      0.685      0.910      +22.50
  0.1500      0.595      0.780      +18.50
```

Compare to the Heron-r2 reference: "+5-25 percentage points across the
d=5 round-sweep" (`paper/surface_code_multidistance_break_even.pdf`).
The pattern reproduces -- the delta is small at low noise (the naive
matching is already nearly optimal), grows with noise, and saturates
at very high noise where both decoders are well below the break-even
threshold.

## When to use this vs `pymatching`

`pymatching` v2 is the standard MWPM decoder in the QEC community and
is well-optimised. Use it if you're writing a publication-grade
benchmark or need parallel batched decoding at scale.

Use this tool if you want:

* a **reproducible reference implementation** of the specific
  algorithm Systrophe used to land the d=5/d=7 break-even on Heron-r2;
* a **transparent, hackable codebase** -- 400 LOC across three files,
  no Cython, no C++;
* the **side-by-side naive control** for measuring algorithmic delta
  on your own data;
* a **dependency-light** library (numpy + networkx).

## Install

This tool ships as part of the Systrophe repo. To use it without
installing the full repo:

```bash
pip install -r tools/dijkstra-mwpm/requirements.txt   # numpy, networkx
PYTHONPATH=tools/dijkstra-mwpm python -c "from dijkstra_mwpm import MWPMDecoder"
```

## License

MIT, inherited from the Systrophe parent package.
