"""Unified runner for all Systrophē Millennium-problem catcher
explorations.

Runs each catcher exploration and prints a single-line summary plus
links to the per-problem JSON result + findings doc. Designed to be
called from the README or CI to give a current snapshot.

Run with:
    python examples/run_all_millennium.py
or
    python examples/run_all_millennium.py --quick   # smaller N's
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).parent


def run_one(name: str, script: Path, env_override: dict | None = None) -> dict:
    print(f"\n{'=' * 70}")
    print(f"Running: {name}")
    print(f"  script: {script.name}")
    print(f"{'=' * 70}")
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True,
        cwd=script.parent.parent,
        env={**dict(__import__("os").environ), **(env_override or {})},
    )
    runtime = time.perf_counter() - t0
    return {
        "name": name,
        "script": str(script.relative_to(HERE.parent)),
        "exit_code": result.returncode,
        "runtime_s": runtime,
        "stdout_tail": "\n".join(result.stdout.splitlines()[-30:]),
        "stderr_tail": result.stderr[-1500:] if result.stderr else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Use smaller N's for faster CI")
    args = ap.parse_args()

    env = {"MILLENNIUM_QUICK": "1"} if args.quick else None

    runs = [
        ("Riemann Hypothesis (zeta-zero spacings)",
         HERE / "millennium_riemann_catcher.py"),
        ("P vs NP (3-SAT phase transition)",
         HERE / "millennium_sat_phase_transition.py"),
    ]

    print("Systrophe Millennium-problem catcher runner")
    print(f"Quick mode: {args.quick}")

    out_summary = []
    for name, script in runs:
        if not script.exists():
            print(f"  SKIP {name}: {script} missing")
            continue
        r = run_one(name, script, env_override=env)
        print(f"\n  -> {name}: exit={r['exit_code']}, runtime={r['runtime_s']:.1f}s")
        out_summary.append(r)

    summary_path = HERE / "run_all_millennium_summary.json"
    summary_path.write_text(json.dumps(out_summary, indent=2, default=str))

    print(f"\n{'=' * 70}")
    print(f"Summary written to {summary_path}")
    print(f"{'=' * 70}")
    print()
    for r in out_summary:
        status = "OK" if r["exit_code"] == 0 else "FAILED"
        print(f"  [{status}] {r['name']:60s} ({r['runtime_s']:.1f}s)")
    print()
    print("See FINDINGS_MILLENNIUM_PROGRESS.md for full interpretation.")


if __name__ == "__main__":
    main()
