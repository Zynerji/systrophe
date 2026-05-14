# lp-analyser

**Analytic-CTC physics for the Lewis-Papapetrou exterior.**

A single-entry-point public API on top of the Systrophe roadmap items
shipped through v0.21.0: classical-GR backbone, energy conditions,
Phase 2a renormalised stress-energy, Phase 2b Hadamard biparametrix,
Phase 3a N-cylinder beam-forming, Phase 3b off-axis pair topology.

The tool is a thin re-export layer over `systrophe.*` — it does **not**
implement any new physics. Its job is to make the analytic-CTC stack
easy to drive from one entry point and to lock down a canonical set
of return-value shapes (`LPSummary`) so downstream consumers don't
need to track the internal Systrophe module layout.

## Provenance

Every method maps directly to a Systrophe module:

| LPAnalyser method                  | Backed by                                       |
|------------------------------------|-------------------------------------------------|
| `.F`, `.K`, `.L`                   | `systrophe.vanstockum.VanStockumInterior`       |
| `.cauchy_horizons`                 | `systrophe.quantum_diagnostics`                 |
| `.ctc_bands`                       | `systrophe.ctc.find_ctc_intervals`              |
| `.energy_conditions`               | `systrophe.energy_conditions` (Phase 1a)        |
| `.surface_gravity`, `.hawking_T`   | `systrophe.quantum_diagnostics`                 |
| `.stress_tensor` (3 states)        | `systrophe.stress_energy_ctc` (**Phase 2a**)    |
| `.chronology_protection_scan`      | `systrophe.stress_energy_ctc` (**Phase 2a**)    |
| `.hadamard_V_0`, `.hadamard_V_1`   | `systrophe.hadamard_modesum` (**Phase 2b**)     |
| `.hadamard_chronology_scan`        | `systrophe.hadamard_modesum` (**Phase 2b**)     |
| `.trace_anomaly_check`             | cross-check Phase 2b ↔ 4D point-splitting       |
| `.kretschmann`                     | `systrophe.point_splitting`                     |
| `PairAnalyser.L`, `.array_factor`  | `systrophe.array.SystropheArray` (**Phase 3a**) |
| `PairAnalyser.beam_steer`          | `systrophe.array.SystropheArray.beam_steer`     |
| `PairAnalyser.extinction_check`    | `systrophe.array.SystropheArray.extinction_check`|
| `PairAnalyser.ctc_region_topology` | `systrophe.off_axis.OffAxisPair` (**Phase 3b**) |
| `PairAnalyser.ergosurface_2d`      | `systrophe.off_axis.OffAxisPair`                |

The reference numbers reproduced by `examples/characterize_supercritical.py`
match `Systrophe/CHANGELOG.md` v0.20.0 + v0.21.0 line for line.

## Layout

```
lp-analyser/
├── README.md
├── lp_analyser/
│   ├── __init__.py
│   ├── analyser.py        LPAnalyser (single cylinder; Phases 1a, 2a, 2b)
│   └── pair.py            PairAnalyser (Phases 3a, 3b)
├── examples/
│   └── characterize_supercritical.py
├── tests/
│   └── test_lp_analyser.py    18 tests
└── requirements.txt
```

## Quick start

```python
from lp_analyser import LPAnalyser

a = LPAnalyser(omega=2.0, R=1.0)
s = a.summary()
print(s.first_horizon_r)                          # 1.4054
print(s.first_horizon_hawking_temperature)        # ~ 1/pi
print(s.boulware_T_tt_simple_pole_power)          # ~ -1.00
print(s.chronology_protection_verdict)            # 'chronology_protection_consistent'
```

```python
# Phase 2a / Phase 2b per-horizon scans
rep_2a = a.chronology_protection_scan(n_horizons=3)
for f in rep_2a.boulware_fits:
    print(f"r_H={f.r_horizon:.4f}  power={f.power:+.4f}")
# r_H=1.4054  power=-1.0074
# r_H=3.1629  power=-0.9973
# r_H=7.1181  power=-1.0013

rep_2b = a.hadamard_chronology_scan(n_horizons=3)
# V_1 powers ~ 0 at every horizon (locally bounded; v0.10 result)
```

```python
# Phase 3a beam-steering
from lp_analyser import PairAnalyser
from systrophe.vanstockum import VanStockumInterior

c = VanStockumInterior(omega=1.0, R=1.0)
p = PairAnalyser.beam_steer(r_target=3.0, cylinder=c, N=2)
print(p.L(np.array([3.0]))[0])    # ~ 0 (machine zero)
```

```python
# Phase 3b off-axis topology
p_off = PairAnalyser([c, c], separation=3.0)
topo = p_off.ctc_region_topology(-3, 6, -3, 3, nx=81, ny=41)
print(topo["n_components"], topo["n_holes"])     # 1, 2
```

## Reference reproduction

`python examples/characterize_supercritical.py` runs every method on
the canonical omega=2, R=1 supercritical Tipler exterior and prints
the Phase 1-3b headline numbers. Expected output (matches `CHANGELOG.md`
v0.20.0 and v0.21.0):

```
Phase 2a Boulware <T_tt> power: -1.007 / -0.997 / -1.001 at r_H in {1.405, 3.163, 7.118}
Phase 2a trace-anomaly residual: 2.22e-16 (machine precision)
Phase 2a verdict: chronology_protection_consistent

Phase 2b V_1 power: +0.022 / -0.010 / +0.004 (BOUNDED at every horizon)
Phase 2b trace-anomaly cross-check (V_1 vs point-splitting): rel diff 0.0

Phase 3a N-fold extinction (N=2..8): max|array_factor| < 4e-15 (machine zero)
Phase 3a beam-steer: L(r_target) = 0 to machine zero for r_target in {2, 3, 5, 7}

Phase 3b separation=3 off-axis pair: CTC region = 1 component + 2 holes
  (resolution-stable at 41x21 and 81x41 grids)
```

Total runtime: < 30 seconds.

## Install

This tool ships as part of the Systrophe repo. To use it without
installing the full repo:

```bash
pip install -r tools/lp-analyser/requirements.txt    # numpy, scipy
PYTHONPATH=src:tools/lp-analyser python -c "from lp_analyser import LPAnalyser"
```

## License

MIT, inherited from the Systrophe parent package.
