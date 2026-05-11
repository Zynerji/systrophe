"""Recover batch 4 hardware result from an already-submitted IBM job.

The original `marrakesh_batch_4_lp_walk.py --hardware` run crashed after
submission but before `job.result()` returned. The job survived in the IBM
runtime queue; re-running the original script would duplicate-submit. This
script pulls the existing job by ID and runs the same analysis +
novelty-catcher pipeline locally, writing the standard output files:

  experiments/results/marrakesh_batch4_hw_analysis.json
  experiments/results/marrakesh_batch4_hw_counts.json

Usage
-----
    python experiments/recover_batch4_hw.py --job-id d80unaftjchs73bm64dg
    python experiments/recover_batch4_hw.py --job-id <id> --wait
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from systrophe.novelty_catcher import catch_novelty_in_distributions

from marrakesh_batch_4_lp_walk import (
    ALPHA_LP,
    RESULTS_DIR,
    all_experiments,
    analyze_bitstring_distribution,
    extract_alpha_from_distribution,
)


def _poll_until_done(job, poll_sec: int = 30) -> None:
    while True:
        st = job.status()
        s = str(st)
        print(f"[recover] status={s}", flush=True)
        if s in ("DONE", "COMPLETED", "ERROR", "CANCELLED", "FAILED"):
            return
        time.sleep(poll_sec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--wait", action="store_true",
                    help="Block-poll until DONE (every 30s).")
    ap.add_argument("--instance", default="Zynerji")
    args = ap.parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=args.instance)
    job = service.job(args.job_id)
    print(f"[recover] job={args.job_id} backend={job.backend().name} "
          f"primitive={getattr(job, 'primitive_id', '?')} "
          f"created={job.creation_date}", flush=True)

    st = str(job.status())
    if st not in ("DONE", "COMPLETED"):
        if not args.wait:
            print(f"[recover] job not finished (status={st}); rerun with --wait "
                  "or wait then rerun without --wait.", flush=True)
            return
        _poll_until_done(job)
        st = str(job.status())
        if st not in ("DONE", "COMPLETED"):
            raise SystemExit(f"[recover] job terminated with status={st}; "
                              "no result to analyse.")

    print("[recover] pulling result...", flush=True)
    result = job.result()

    experiments = all_experiments()  # rebuild predicted distributions / labels
    assert len(experiments) == len(result), (
        f"experiment count mismatch: {len(experiments)} vs {len(result)}")

    all_distributions = []
    summaries = []
    raw_records = []
    for i, ex in enumerate(experiments):
        data = result[i].data
        creg_name = next(iter(data))
        counts = getattr(data, creg_name).get_counts()
        analysis = analyze_bitstring_distribution(counts, ex["predicted_distribution"])
        alpha_rec = extract_alpha_from_distribution(counts)
        summaries.append({
            "label": ex["label"],
            "alpha_scale": ex["alpha_scale"],
            **analysis,
            **alpha_rec,
        })
        all_keys = sorted(set(counts.keys()) | set(ex["predicted_distribution"].keys()))
        total = sum(counts.values())
        prob_vec = np.array([counts.get(k, 0) / total for k in all_keys])
        all_distributions.append(prob_vec)
        raw_records.append({"label": ex["label"], "counts": counts})
        print(f"  {ex['label']:30s} TV={analysis['total_variation_distance']:.4f}  "
              f"alpha_rec={alpha_rec['alpha_recovered']:.3f} "
              f"(theory={ALPHA_LP:.3f})", flush=True)

    labels = [s["label"] for s in summaries]
    novelty = catch_novelty_in_distributions(all_distributions,
                                              labels=labels, n_bits=32)

    out_path = Path(RESULTS_DIR) / "marrakesh_batch4_hw_analysis.json"
    raw_path = Path(RESULTS_DIR) / "marrakesh_batch4_hw_counts.json"
    out_path.write_text(json.dumps({
        "per_circuit": summaries,
        "novelty_catcher": novelty,
        "recovered_from_job_id": args.job_id,
    }, indent=2))
    raw_path.write_text(json.dumps(raw_records, indent=2))

    print(f"\n[recover] novelty verdict: {novelty['verdict']}", flush=True)
    print(f"[recover] sharp features: {len(novelty['sharp_features'])}", flush=True)
    if novelty["sharp_features"]:
        print("[recover] sharp feature details:", flush=True)
        for sf in novelty["sharp_features"]:
            print(f"    {sf}", flush=True)
    print(f"[recover] wrote {out_path}", flush=True)
    print(f"[recover] wrote {raw_path}", flush=True)


if __name__ == "__main__":
    main()
