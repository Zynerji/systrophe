"""Attack every MONETARY-PRIZE Erdos problem that has an OEIS sequence.

Breadth pass of the autonomous campaign: for every OPEN prize problem in
the teorth/erdosproblems linkage that carries an OEIS sequence, fetch the
b-file and point the validated DSI detector (Lomb-Scargle in ln n, AR(1)
red-noise significance) at it. Most are expected to be untestable (Ramsey
numbers, Golomb rulers, etc. have a handful of known terms); the few with
enough length get an honest DSI verdict, Bonferroni-corrected over the
testable set. Anchors (synthetic log-periodic + prime psi) validate the run.

Honest premise: a spectral instrument cannot PROVE an open conjecture, so it
cannot collect a proof-prize. This pass establishes which prize problems are
even in the toolkit's domain and whether any testable one shows real
discrete-scale-invariant structure (a lead), not a proof.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from erdos_dsi_sweep import dsi_scan, gen_control_logperiodic, gen_psi_error
from erdos_oeis_dsi_sweep import fetch_bfile, prepare_series

UA = {"User-Agent": "Mozilla/5.0 (Systrophe prize battery)"}
YAML_URL = ("https://raw.githubusercontent.com/teorth/erdosproblems/"
            "main/data/problems.yaml")


def open_prize_problems() -> list[dict]:
    y = urllib.request.urlopen(
        urllib.request.Request(YAML_URL, headers=UA), timeout=60
    ).read().decode("utf-8", "replace")
    recs = []
    for b in re.split(r'\n- number:', y):
        num = re.search(r'^\s*"?(\d+)"?', b)
        prize = re.search(r'prize:\s*"(\$\d+)"', b)
        if not (num and prize):
            continue
        state = re.search(r'state:\s*"(.*?)"', b)
        if not state or state.group(1) != "open":
            continue
        oeis = re.search(r'oeis:\s*\[(.*?)\]', b)
        ids = re.findall(r'A\d{6}', oeis.group(1)) if oeis else []
        tags = re.findall(r'"(.*?)"',
                          (re.search(r'tags:\s*\[(.*?)\]', b)
                           or re.match("", "")).group(1)) \
            if re.search(r'tags:\s*\[(.*?)\]', b) else []
        if ids:
            recs.append({"number": num.group(1),
                         "prize": int(prize.group(1)[1:]),
                         "oeis": ids, "tags": tags})
    recs.sort(key=lambda r: -r["prize"])
    return recs


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    recs = open_prize_problems()
    # Unique (aid, problem, prize, tags), highest prize wins per aid.
    seen, items = {}, []
    for r in recs:
        for aid in r["oeis"]:
            if aid not in seen:
                seen[aid] = True
                items.append((aid, r))

    tested, untestable = [], []
    for aid, r in items:
        terms = fetch_bfile(aid)
        if terms is None:
            untestable.append({"aid": aid, "problem": r["number"],
                               "prize": r["prize"], "reason": "fetch_failed"})
            continue
        prepared, why = prepare_series(aid, terms, min_terms=150,
                                       min_span=2.5, max_points=1500)
        if prepared is None:
            untestable.append({"aid": aid, "problem": r["number"],
                               "prize": r["prize"], "tags": r["tags"],
                               "reason": why, "n_terms": len(terms)})
            continue
        t, y = prepared
        items_scan = dsi_scan(t, y, n_freq=1000, n_boot=300)
        tested.append({"aid": aid, "problem": r["number"], "prize": r["prize"],
                       "tags": r["tags"], "n_terms": len(t), **items_scan})

    bonf = 0.05 / max(len(tested), 1)
    for x in tested:
        x["verdict"] = "DSI-lead" if x["p_value"] <= bonf else "null"
    tested.sort(key=lambda r: r["p_value"])

    # Anchors
    ct, cy, _ = gen_control_logperiodic(1500)
    pt, py, _ = gen_psi_error(1500, 1e7)
    anchors = {"control_logperiodic": dsi_scan(ct, cy, n_freq=1000, n_boot=300),
               "psi_error": dsi_scan(pt, py, n_freq=1000, n_boot=300)}

    print("=" * 80)
    print("Erdos MONETARY-PRIZE battery (open problems w/ OEIS; DSI ln-n, "
          "AR(1) sig)")
    print("=" * 80)
    print(f"  unique prize sequences: {len(items)}  | testable: {len(tested)}"
          f"  | untestable: {len(untestable)}  | Bonferroni p<={bonf:.5f}")
    print("\n  anchors:", {k: f"omega={v['peak_omega']:.2f},p={v['p_value']:.3f}"
                           for k, v in anchors.items()})
    print("\n  TESTABLE prize sequences:")
    print(f"  {'$':>5} {'#':>4} {'aid':>9} {'terms':>5} {'omega':>7} "
          f"{'power':>6} {'p':>7}  verdict")
    for x in tested:
        print(f"  {x['prize']:>5} {x['problem']:>4} {x['aid']:>9} "
              f"{x['n_terms']:>5} {x['peak_omega']:>7.3f} {x['peak_power']:>6.3f} "
              f"{x['p_value']:>7.4f}  {x['verdict']}")
    print("\n  UNTESTABLE (too short / unfit) — by prize:")
    for x in sorted(untestable, key=lambda r: -r["prize"]):
        nt = x.get("n_terms", "?")
        print(f"  ${x['prize']:>5} #{x['problem']:>4} {x['aid']}  "
              f"({x['reason']}, n={nt})")

    leads = [x for x in tested if x["verdict"] == "DSI-lead"]
    print(f"\n  DSI leads among prize problems: {len(leads)}")
    for h in leads:
        print(f"    ${h['prize']} #{h['problem']} {h['aid']}: "
              f"omega={h['peak_omega']:.3f} p={h['p_value']:.4f} {h['tags']}")

    out = {"n_unique": len(items), "n_tested": len(tested),
           "n_untestable": len(untestable), "bonferroni": bonf,
           "anchors": {k: v for k, v in anchors.items()},
           "tested": tested, "untestable": untestable}
    p = Path(__file__).parent / "erdos_prize_battery_results.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Wrote {p}")


if __name__ == "__main__":
    main()
