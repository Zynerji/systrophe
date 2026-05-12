"""Retrofit the derivative catcher to existing phase modules.

The base value-level catcher returned emergent #17 (feedback_amplified_shell
critical Q ≈ 7.86) — but it caught the QUALITATIVE on/off threshold via
scan_novelty. With the new derivative catcher (`src/systrophe/derivative_catcher.py`)
we can ask: are there ADDITIONAL gradient transitions inside these
modules that the value-level pass missed?

Targets:
  1. feedback_amplified_shell.required_drive_power(Q)
  2. optical_fiber_analog (analog-horizon onset)
  3. krasnikov_ring.pair_extinction_amplitude(sigma_noise)
  4. berry_phase_lp (Berry phase across the radial sweep)

For each, scan a scalar observable across its parameter range and ask
both the value-level and the derivative catchers what they catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systrophe.derivative_catcher import catch_smooth_transition
from systrophe.novelty_catcher import scan_novelty


def retrofit_feedback_shell() -> dict:
    from systrophe.feedback_amplified_shell import required_drive_power
    Q_grid = np.linspace(1, 100, 50)

    def fn_scalar(q):
        return float(required_drive_power(v_s=1.0, R=1.0, sigma=5.0, Q=float(q)))

    res = catch_smooth_transition(Q_grid, fn_scalar, n_bits=32)
    return _summarise("feedback_amplified_shell", "Q", Q_grid, fn_scalar, res)


def retrofit_optical_fiber() -> dict:
    # Use # of analog horizons (0 or 1) vs v_end across the onset.
    try:
        from systrophe.optical_fiber_analog import (
            linear_pump_profile, fiber_analog_horizon,
        )
    except ImportError:
        return {"label": "optical_fiber_analog", "skipped": True}
    v_grid = np.linspace(0.4, 0.95, 50)

    def fn_scalar(v_end):
        pump = linear_pump_profile(v_start=0.4, v_end=float(v_end))
        h = fiber_analog_horizon(pump)
        # First-horizon location if any, else x_end
        if h["n_horizons"] > 0:
            return float(h["horizons"][0])
        return float(h["x_range"][1])

    res = catch_smooth_transition(v_grid, fn_scalar, n_bits=32)
    return _summarise("optical_fiber_analog", "v_end", v_grid, fn_scalar, res)


def retrofit_krasnikov_ring() -> dict:
    try:
        from systrophe.krasnikov_ring import krasnikov_ring_with_noise_NEC_radial
    except ImportError:
        return {"label": "krasnikov_ring", "skipped": True}
    sigma_grid = np.linspace(0.0, np.pi, 30)
    x_grid = np.linspace(-3.0, 3.0, 51)
    rng_seeds = list(range(50))

    def fn_scalar(sigma_phase):
        # mean residual NEC over many trials, integrated over x
        residuals = []
        for seed in rng_seeds:
            vals = [
                krasnikov_ring_with_noise_NEC_radial(
                    float(x), 1.0, 0.0, 0.0, 4.0, 2,
                    float(sigma_phase), seed=seed,
                )
                for x in x_grid
            ]
            residuals.append(abs(float(np.trapezoid(np.array(vals), x_grid))))
        return float(np.mean(residuals))

    res = catch_smooth_transition(sigma_grid, fn_scalar, n_bits=32)
    return _summarise("krasnikov_ring", "sigma_phase", sigma_grid, fn_scalar, res)


def retrofit_berry_phase_lp() -> dict:
    # Not present as a top-level radial-sweep function with a single
    # scalar; skip cleanly.
    return {"label": "berry_phase_lp", "skipped": True,
             "reason": "no scalar radial sweep API"}


def _summarise(label: str, p_name: str, p_grid: np.ndarray,
                fn, res: dict) -> dict:
    raw_values = [float(fn(float(p))) for p in p_grid]
    return {
        "label": label,
        "param_name": p_name,
        "param_min": float(p_grid.min()),
        "param_max": float(p_grid.max()),
        "n_points": int(len(p_grid)),
        "value_min": float(min(raw_values)),
        "value_max": float(max(raw_values)),
        "value_verdict": res["value_scan"].verdict,
        "value_n_sharp": len(res["value_scan"].sharp_features),
        "derivative_verdict": res["derivative_scan"].verdict,
        "derivative_n_sharp": len(res["derivative_scan"].sharp_features),
        "kind": res["kind"],
        "estimated_transition_centre": res["estimated_transition_centre"],
        "derivative_sharp_features": [
            {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
             for k, v in sf.items()}
            for sf in res["derivative_scan"].sharp_features
        ],
    }


def main() -> None:
    print("=" * 70)
    print("Derivative-catcher retrofit on existing phase modules")
    print("=" * 70)
    print()

    runners = [
        ("feedback_amplified_shell", retrofit_feedback_shell),
        ("optical_fiber_analog", retrofit_optical_fiber),
        ("krasnikov_ring",         retrofit_krasnikov_ring),
        ("berry_phase_lp",         retrofit_berry_phase_lp),
    ]
    all_results = {}
    for label, runner in runners:
        try:
            r = runner()
        except Exception as e:
            r = {"label": label, "error": str(e)}
        all_results[label] = r
        if r.get("skipped"):
            print(f"--- {label}: SKIPPED (module not present) ---")
            print()
            continue
        if "error" in r:
            print(f"--- {label}: ERROR -- {r['error']} ---")
            print()
            continue
        print(f"--- {label} ---")
        print(f"  {r['param_name']} in [{r['param_min']:.3f}, {r['param_max']:.3f}], "
              f"n={r['n_points']}")
        print(f"  value range: [{r['value_min']:.4e}, {r['value_max']:.4e}]")
        print(f"  value catcher:      verdict={r['value_verdict']}, "
              f"n_sharp={r['value_n_sharp']}")
        print(f"  derivative catcher: verdict={r['derivative_verdict']}, "
              f"n_sharp={r['derivative_n_sharp']}, kind={r['kind']}")
        centre = r["estimated_transition_centre"]
        if centre is not None:
            print(f"  estimated transition centre at {r['param_name']} = {centre:.4f}")
        for sf in r["derivative_sharp_features"]:
            p_val = sf.get("parameter_value", "?")
            print(f"    derivative sharp: {r['param_name']} = {p_val}, "
                  f"hamming_step = {sf.get('hamming_step', '?')}")
        print()

    out_path = Path(__file__).parent / "retrofit_derivative_catcher_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
