# gw-burst-catcher

**Unmodeled gravitational-wave burst detection via the Systrophe address-space λ₂ catcher.**

Apply the catcher to a whitened LIGO strain → Q-transform → sliding
window pipeline and report per-window Hamming-step + the largest-step
candidate-event time. No template bank required — the catcher fires
when the time-frequency content of the strain changes sharply.

This tool **promotes the previously-untracked `src/systrophe/gw_catcher.py`**
(266 LOC, referenced in commit `3999ef6`'s 5000-segment unsearched-sample
scan) into a properly-packaged derived tool with: synthetic-injection
test mode (no GWOSC network access required for CI), separate
preprocessing / detection modules, and a typed `BurstDetectionResult`
dataclass.

## Provenance

* Algorithm: the same address-space λ₂ catcher from
  `systrophe.catchers.novelty_catcher.scan_novelty` (the framework's
  load-bearing detection primitive; track record 26+ emergents).
* Use case: applying the catcher to the *time-frequency representation*
  of a whitened strain segment, then sliding a window across it.
  Sharp jumps in per-band log-energy → high Hamming step → candidate
  burst.

## Layout

```
gw-burst-catcher/
├── README.md
├── gw_burst_catcher/
│   ├── __init__.py
│   ├── events.py          KNOWN_EVENTS registry (GW150914, GW170817, GW170814)
│   ├── preprocess.py      whiten_strain, q_transform (numpy + scipy)
│   ├── detection.py       catcher_scan_qtile, detect_burst_in_strain,
│   │                       run_event_catcher (GWOSC; needs gwpy)
│   └── synthetic.py       make_gaussian_noise, inject_chirp -- for
│                          offline tests + smoke runs
├── examples/
│   └── synthetic_chirp_demo.py    locates an injected chirp in noise
├── tests/
│   └── test_gw_burst_catcher.py   13 tests, all-synthetic, no network
└── requirements.txt
```

## Quick start (offline / synthetic)

```python
from gw_burst_catcher import (
    make_gaussian_noise, inject_chirp,
)
from gw_burst_catcher.detection import detect_burst_in_strain

sr = 1024
strain = make_gaussian_noise(duration_s=8.0, sample_rate=sr, seed=0)
info = inject_chirp(strain, sample_rate=sr, t_inject_s=4.0,
                       f_start=50.0, f_end=250.0,
                       amplitude=10.0, duration_s=0.3)

res = detect_burst_in_strain(strain, sample_rate=sr)
print(res.max_hamming_step_time_s)   # ~ 4.0
print(res.verdict)                   # 'novel_structure' typically
```

## Real-event mode (needs gwpy + internet)

```python
from gw_burst_catcher import run_event_catcher

result = run_event_catcher(event_name="GW150914",
                              detectors=("H1", "L1"),
                              duration_s=32.0)
print(result["per_detector"]["H1"]["max_hamming_step_time_s"])
print(result["tof_coincidence_ms"])   # H1 vs L1 time-of-flight
```

The real-event runner requires `pip install gwpy` and an internet
connection to GWOSC.

## What this tool is *not*

This is a **catcher-side**, low-level pipeline. It is NOT a full LIGO
detection pipeline: no template bank, no calibrated PSDs, no proper
glitch-veto, no false-alarm-rate calibration. For publishable GW
science use PyCBC / GstLAL / cWB.

What this IS: a clean demonstration that the Systrophe catcher
flags unmodeled transients in time-frequency data, with a
reproducible synthetic-injection harness for development.

## License

MIT, inherited from the Systrophe parent package.
