"""Recover batch 5 hardware result from an already-submitted IBM job.

Mirror of recover_batch4_hw.py for the Tipler-pair extinction batch.
Pulls the existing job_id and runs the same analyse + novelty-catcher
pipeline used in marrakesh_batch_5_pair_extinction.py's run_hardware,
writing the standard output files:

  experiments/results/marrakesh_batch5_hw_analysis.json
  experiments/results/marrakesh_batch5_hw_counts.json

Usage
-----
    python experiments/recover_batch5_hw.py --job-id d810s4voha1c73bjp0a0
    python experiments/recover_batch5_hw.py --job-id <id> --wait
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from systrophe.novelty_catcher import catch_novelty_in_distributions

from marrakesh_batch_5_pair_extinction import (
    RESULTS_DIR,
    all_experiments,
    _tv,
)


def _poll_until_done(job, poll_sec: int = 30) -> None:
    while True:
        s = str(job.status())
        print(f"[recover5] status={s}", flush=True)
        if s in ("DONE", "COMPLETED", "ERROR", "CANCELLED", "FAILED"):
            return
        time.sleep(poll_sec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--instance", default="Zynerji")
    args = ap.parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(instance=args.instance)
    job = service.job(args.job_id)
    print(f"[recover5] job={args.job_id} backend={job.backend().name} "
          f"primitive={getattr(job, 'primitive_id', '?')} "
          f"created={job.creation_date}", flush=True)

    st = str(job.status())
    if st not in ("DONE", "COMPLETED"):
        if not args.wait:
            print(f"[recover5] job not finished (status={st}); rerun with --wait.",
                  flush=True)
            return
        _poll_until_done(job)
        st = str(job.status())
        if st not in ("DONE", "COMPLETED"):
            raise SystemExit(f"[recover5] job terminated with status={st}")

    print("[recover5] pulling result...", flush=True)
    result = job.result()
    experiments = all_experiments()
    assert len(experiments) == len(result), (
        f"experiment count mismatch: {len(experiments)} vs {len(result)}")

    summaries = []
    raw = []
    distributions = []
    for i, ex in enumerate(experiments):
        data = result[i].data
        creg_name = next(iter(data))
        counts = getattr(data, creg_name).get_counts()
        total = sum(counts.values())
        observed = {k: v / total for k, v in counts.items()}
        tv = _tv(observed, ex["predicted_distribution"])
        p_data1 = sum(v for k, v in observed.items() if k[-1] == "1")
        summaries.append({
            "label": ex["label"], "delta": ex["delta"],
            "tv_obs_vs_pred": tv,
            "P_data1_observed": float(p_data1),
            "P_data1_predicted": sum(v for k, v in ex["predicted_distribution"].items()
                                       if k[-1] == "1"),
        })
        all_keys = sorted(set(counts.keys()) | set(ex["predicted_distribution"].keys()))
        prob_vec = np.array([counts.get(k, 0) / total for k in all_keys])
        distributions.append(prob_vec)
        raw.append({"label": ex["label"], "counts": counts})
        print(f"  {ex['label']:25s} TV={tv:.4f}  P(data=1)={p_data1:.4f}",
              flush=True)

    labels = [s["label"] for s in summaries]
    novelty = catch_novelty_in_distributions(distributions, labels=labels)

    out_path = Path(RESULTS_DIR) / "marrakesh_batch5_hw_analysis.json"
    raw_path = Path(RESULTS_DIR) / "marrakesh_batch5_hw_counts.json"
    out_path.write_text(json.dumps({
        "per_circuit": summaries,
        "novelty_catcher": novelty,
        "recovered_from_job_id": args.job_id,
    }, indent=2))
    raw_path.write_text(json.dumps(raw, indent=2))

    print(f"\n[recover5] novelty verdict: {novelty['verdict']}, "
          f"sharp_features={len(novelty['sharp_features'])}", flush=True)
    for sf in novelty["sharp_features"]:
        print(f"    sharp: {sf.get('between')}  step={sf.get('hamming_step')}",
              flush=True)
    print(f"[recover5] wrote {out_path}", flush=True)
    print(f"[recover5] wrote {raw_path}", flush=True)


if __name__ == "__main__":
    main()
