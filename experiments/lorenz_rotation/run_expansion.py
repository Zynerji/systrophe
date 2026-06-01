"""Expansion orchestrator: many attractors + deep tests exploiting the construct.

Extends run_experiment.py (single Lorenz rotation) to:
  A. a registry of dissipative chaotic rotation laws (Lorenz/Rossler/Chen/Halvorsen),
     each pushed through the Lyapunov + correlation-dimension + CTC pipeline;
  B. deep tests specific to the chaotically-gated-CTC construct:
     - Takens faithfulness: the scalar CTC band-measure reconstructs the attractor;
     - Pecora-Carroll synchronization of a Tipler pair;
     - catcher chaos-onset universality across attractors.

Writes lorenz_expansion_results.json. Heavier than run_experiment.py (~8-12 min).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rotating_dust_lorenz import lyapunov_spectrum, correlation_dimension, rotation_parameter_timeseries
from chaotic_ctc import chaotic_ctc_timeseries
from attractors import registry
from deep_tests import observable_faithfulness, pecora_carroll_sync, conditional_lyapunov, chaos_onset_scan


def attractor_suite() -> list:
    print("\n=== A. Attractor suite: Lyapunov / dimension / CTC for each rotation law ===")
    rows = []
    for f in registry():
        spec = lyapunov_spectrum(f, f.default_s0, t_max=300.0, dt=0.005,
                                 renorm_every=20, t_transient=f.t_transient)
        traj = f.integrate(f.default_s0, 250.0, dt=0.01, t_transient=f.t_transient)
        mean_div = f.mean_divergence(traj["s"])
        d2 = correlation_dimension(f, f.default_s0, t_max=800.0, dt=0.01,
                                   t_transient=f.t_transient, n_points=5000)
        rot = rotation_parameter_timeseries(traj, a0=1.5, eps=0.2, clip_min=0.55)
        cts = chaotic_ctc_timeseries(rot, R=1.0, r_min=1.05, r_max=20.0, stride=8)
        row = {
            "flow": f.name, "physical": f.physical,
            "lyapunov": [round(x, 4) for x in spec["exponents"].tolist()],
            "lyapunov_sum": round(spec["sum"], 4),
            "mean_divergence": round(mean_div, 4),
            "kaplan_yorke_dim": round(spec["kaplan_yorke_dim"], 4),
            "correlation_dim_D2": round(d2["D2"], 4),
            "ctc_band_count_min": int(cts["n_bands"].min()),
            "ctc_band_count_max": int(cts["n_bands"].max()),
            "ctc_log_measure_std": round(float(cts["log_measure"].std()), 4),
        }
        rows.append(row)
        print(f"  {f.name:24s} LE={row['lyapunov']}  KY={row['kaplan_yorke_dim']}  "
              f"D2={row['correlation_dim_D2']}  Σλ={row['lyapunov_sum']}=meanDiv {row['mean_divergence']}  "
              f"CTC bands {row['ctc_band_count_min']}..{row['ctc_band_count_max']}")
    return rows


def faithfulness_suite() -> list:
    print("\n=== B1. Takens faithfulness: CTC observable reconstructs the attractor ===")
    rows = []
    for f in registry():
        res = observable_faithfulness(f, t_max=1000.0, dt=0.01, stride=4)
        rows.append({k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in res.items()})
        print(f"  {f.name:24s} D2_state={res['D2_state_space']:.3f}  "
              f"D2_from_CTC={res['D2_ctc_observable']:.3f}  rel.diff={res['relative_diff']:.1%}")
    return rows


def synchronization() -> dict:
    print("\n=== B2. Pecora-Carroll synchronization of a Tipler pair ===")
    out = {}
    for drv in ("x", "z"):
        syn = pecora_carroll_sync(drive=drv, t_max=60.0)
        cle = conditional_lyapunov(drive=drv)
        out[drv] = {
            "conditional_lyapunov_max": round(cle["conditional_lyapunov_max"], 4),
            "tail_mean_error": syn["tail_mean_error"],
            "locked": syn["locked"],
        }
        print(f"  drive={drv}: cond.Lyapunov={cle['conditional_lyapunov_max']:+.3f}  "
              f"tail_err={syn['tail_mean_error']:.2e}  locked={syn['locked']}")
    return out


def onset_universality() -> list:
    print("\n=== B3. Catcher chaos-onset universality across attractors ===")
    rows = []
    for f in registry():
        r = chaos_onset_scan(f, t_le=100.0)
        rows.append({k: v for k, v in r.items()
                     if k not in ("lyapunov_curve", "scan")})
        print(f"  {f.name:24s} LE_onset({r['param']})={r['lyapunov_onset']}  "
              f"catcher={r['catcher_verdict']} feats={r['catcher_features']}  "
              f"agree={r['onset_agreement']}")
    return rows


def main():
    t0 = time.time()
    results = {
        "attractor_suite": attractor_suite(),
        "faithfulness": faithfulness_suite(),
        "synchronization": synchronization(),
        "onset_universality": onset_universality(),
        "runtime_seconds": None,
    }
    results["runtime_seconds"] = round(time.time() - t0, 1)
    out_path = Path(__file__).with_name("lorenz_expansion_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}  ({results['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
