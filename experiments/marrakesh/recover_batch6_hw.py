"""Recover batch 6 Knopp Drive hardware result from a queued IBM job."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from systrophe.catchers.novelty_catcher import catch_novelty_in_distributions

from marrakesh_batch_6_knopp_drive import (
    RESULTS_DIR,
    _tv,
    all_experiments,
)


def _poll_until_done(job, poll_sec: int = 30) -> None:
    while True:
        s = str(job.status())
        print(f"[recover6] status={s}", flush=True)
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
    print(f"[recover6] job={args.job_id} backend={job.backend().name} "
          f"created={job.creation_date}", flush=True)

    st = str(job.status())
    if st not in ("DONE", "COMPLETED"):
        if not args.wait:
            print(f"[recover6] job not finished (status={st}); rerun with --wait",
                  flush=True)
            return
        _poll_until_done(job)
        st = str(job.status())
        if st not in ("DONE", "COMPLETED"):
            raise SystemExit(f"[recover6] terminated with status={st}")

    print("[recover6] pulling result...", flush=True)
    result = job.result()
    experiments = all_experiments()
    assert len(experiments) == len(result), (
        f"count mismatch: {len(experiments)} vs {len(result)}")

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
            "label": ex["label"], "r_index": ex["r_index"], "r": ex["r"],
            "tilt": ex["tilt"], "tipler_gate_factor": ex["tipler_gate_factor"],
            "phi": ex["phi"], "tv_obs_vs_pred": tv,
            "P_data1_observed": float(p_data1),
            "P_data1_predicted": sum(
                v for k, v in ex["predicted_distribution"].items() if k[-1] == "1"
            ),
        })
        all_keys = sorted(set(counts.keys()) | set(ex["predicted_distribution"].keys()))
        prob_vec = np.array([counts.get(k, 0) / total for k in all_keys])
        distributions.append(prob_vec)
        raw.append({"label": ex["label"], "counts": counts})
        print(f"  {ex['label']:18s} r={ex['r']:.3f}  gate={ex['tipler_gate_factor']:.3f}  "
              f"TV={tv:.4f}  P(data=1)={p_data1:.4f}", flush=True)

    labels = [s["label"] for s in summaries]
    novelty = catch_novelty_in_distributions(distributions, labels=labels)

    out_path = Path(RESULTS_DIR) / "marrakesh_batch6_hw_analysis.json"
    raw_path = Path(RESULTS_DIR) / "marrakesh_batch6_hw_counts.json"
    out_path.write_text(json.dumps({
        "per_circuit": summaries,
        "novelty_catcher": novelty,
        "recovered_from_job_id": args.job_id,
    }, indent=2, default=str))
    raw_path.write_text(json.dumps(raw, indent=2))
    print(f"\n[recover6] novelty verdict: {novelty['verdict']}, "
          f"sharp_features={len(novelty['sharp_features'])}", flush=True)
    for sf in novelty["sharp_features"]:
        print(f"    sharp: {sf.get('between')}  step={sf.get('hamming_step')}",
              flush=True)
    print(f"[recover6] wrote {out_path}", flush=True)
    print(f"[recover6] wrote {raw_path}", flush=True)


if __name__ == "__main__":
    main()
