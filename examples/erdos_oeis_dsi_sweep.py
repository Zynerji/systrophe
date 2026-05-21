"""Erdos DSI sweep over REAL OEIS-linked sequences.

Extends `erdos_dsi_sweep.py` from a hand-built battery to the actual
Erdos<->OEIS linkage maintained at github.com/teorth/erdosproblems
(Bloom & Tao). It:

  1. pulls `data/problems.yaml` and extracts the OEIS A-numbers attached to
     Erdos problems (with their tags / status);
  2. fetches + caches each sequence's OEIS b-file;
  3. keeps the ones suitable for log-periodicity detection (positive,
     clean power-law-ish growth, enough terms and ln-n span);
  4. runs the SAME validated DSI detector as erdos_dsi_sweep (Lomb-Scargle
     in ln n, AR(1) red-noise significance, max-power look-elsewhere),
     with the two synthetic controls + prime psi(x)-x as anchors;
  5. Bonferroni-corrects across the real batch and ranks the results.

A surviving hit means an Erdos-linked OEIS sequence carries log-periodic
(discrete-scale-invariant) structure -- which would be new. The anchors
make the run self-validating: the controls must behave and psi must peak
at gamma_1 = 14.13, or the batch is not to be trusted.

Network note: OEIS 403s the default urllib UA; a browser UA is sent.
b-files are cached under examples/oeis_cache/ (not committed).

Usage
-----
    python examples/erdos_oeis_dsi_sweep.py --max-fetch 120
    python examples/erdos_oeis_dsi_sweep.py --quick
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

from erdos_dsi_sweep import (
    dsi_scan,
    gen_control_logperiodic,
    gen_control_noise,
    gen_psi_error,
    _detrend_logpower,
)

UA = {"User-Agent": "Mozilla/5.0 (Systrophe DSI research sweep)"}
YAML_URL = ("https://raw.githubusercontent.com/teorth/erdosproblems/"
            "main/data/problems.yaml")
CACHE = Path(__file__).parent / "oeis_cache"

# Tags whose problems tend to be about growing extremal/counting sequences.
GROWTH_TAGS = {"number theory", "additive combinatorics", "combinatorics",
               "primes", "arithmetic progressions", "sequences",
               "analytic number theory", "graph theory"}


def _get(url: str, tries: int = 3) -> str:
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=45).read().decode(
                "utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0)
    raise last


def parse_linkage() -> list[dict]:
    """Return [{number, oeis:[...], tags:[...], state}] from problems.yaml."""
    y = _get(YAML_URL)
    blocks = re.split(r'\n- number:', y)
    recs = []
    for b in blocks:
        num = re.search(r'^\s*"?(\d+)"?', b)
        if not num:
            continue
        oeis = re.search(r'oeis:\s*\[(.*?)\]', b)
        tags = re.search(r'tags:\s*\[(.*?)\]', b)
        state = re.search(r'state:\s*"(.*?)"', b)
        ids = re.findall(r'A\d{6}', oeis.group(1)) if oeis else []
        tg = re.findall(r'"(.*?)"', tags.group(1)) if tags else []
        if ids:
            recs.append({"number": num.group(1), "oeis": ids,
                         "tags": tg, "state": state.group(1) if state else "?"})
    return recs


def fetch_bfile(aid: str) -> list[tuple[int, int]] | None:
    CACHE.mkdir(exist_ok=True)
    cf = CACHE / f"{aid}.txt"
    if cf.exists():
        txt = cf.read_text(encoding="utf-8", errors="replace")
    else:
        num = aid[1:]
        try:
            txt = _get(f"https://oeis.org/{aid}/b{num}.txt")
        except Exception:
            return None
        cf.write_text(txt, encoding="utf-8")
        time.sleep(0.4)  # be polite to OEIS
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            out.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
    return out or None


def prepare_series(aid: str, terms: list[tuple[int, int]],
                   min_terms: int, min_span: float, max_points: int):
    """Reduce a b-file to (t=ln n, y=log-power-detrended) if suitable."""
    idx = np.array([n for n, _ in terms], dtype=float)
    keep = idx > 0
    idx = idx[keep]
    vals = [v for (n, v), k in zip(terms, keep) if k]
    if len(idx) < min_terms:
        return None, "too_short"
    if any(v <= 0 for v in vals):
        return None, "nonpositive"
    span = math.log(idx[-1]) - math.log(idx[0])
    if span < min_span:
        return None, "small_span"
    logy = np.array([math.log(v) for v in vals])  # math.log handles big ints
    t = np.log(idx)
    # clean power-law-ish growth: high linear correlation of logy vs ln n
    r = float(np.corrcoef(t, logy)[0, 1])
    if not np.isfinite(r) or r < 0.97:
        return None, f"not_growthlike(r={r:.2f})"
    # decimate to bound LS cost
    if len(t) > max_points:
        sel = np.unique(np.linspace(0, len(t) - 1, max_points).round()
                        .astype(int))
        t, logy = t[sel], logy[sel]
    y = _detrend_logpower(t, logy, deg=3)
    return (t, y), "ok"


def run(max_fetch: int, quick: bool) -> dict:
    n_boot = 120 if quick else 150
    n_freq = 800 if quick else 1000
    max_points = 1200 if quick else 1500
    min_terms = 200

    recs = parse_linkage()
    # Unique A-numbers from growth-flavoured problems, in problem order.
    seen, candidates = set(), []
    for r in recs:
        if not (set(r["tags"]) & GROWTH_TAGS):
            continue
        for aid in r["oeis"]:
            if aid not in seen:
                seen.add(aid)
                candidates.append((aid, r))
    candidates = candidates[:max_fetch]

    series, skipped = [], {}
    for aid, rec in candidates:
        terms = fetch_bfile(aid)
        if terms is None:
            skipped[aid] = "fetch_failed"
            continue
        prepared, why = prepare_series(aid, terms, min_terms, 3.0, max_points)
        if prepared is None:
            skipped[aid] = why
            continue
        t, y = prepared
        series.append({"aid": aid, "problem": rec["number"],
                       "tags": rec["tags"], "state": rec["state"],
                       "t": t, "y": y, "n_terms": len(t)})

    n_real = len(series)
    bonf = 0.05 / max(n_real, 1)

    # Anchors (self-validation): controls + prime psi.
    ct, cy, _ = gen_control_logperiodic(1500)
    nt, ny, _ = gen_control_noise(1500)
    pt, py, _ = gen_psi_error(1500, 1e7)
    anchors = [("__control_logperiodic", ct, cy),
               ("__control_rednoise", nt, ny),
               ("__anchor_psi_error", pt, py)]

    results = []
    for name, t, y in anchors:
        scan = dsi_scan(t, y, n_freq=n_freq, n_boot=n_boot)
        results.append({"name": name, "kind": "anchor", **scan,
                        "verdict": "DSI" if scan["p_value"] <= 0.05 else "null"})
    for s in series:
        scan = dsi_scan(s["t"], s["y"], n_freq=n_freq, n_boot=n_boot)
        results.append({
            "name": s["aid"], "kind": "oeis", "problem": s["problem"],
            "tags": s["tags"], "state": s["state"], "n_terms": s["n_terms"],
            **scan,
            "verdict": "DSI" if scan["p_value"] <= bonf else "null",
        })

    results.sort(key=lambda r: r["p_value"])
    return {
        "n_candidates": len(candidates),
        "n_real_scanned": n_real,
        "n_skipped": len(skipped),
        "bonferroni_threshold": bonf,
        "skipped_reasons": skipped,
        "results": results,
    }


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-fetch", type=int, default=120)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    out = run(args.max_fetch, args.quick)
    print("=" * 82)
    print("Erdos x OEIS DSI sweep (log-periodic; AR(1) red-noise sig; "
          "Bonferroni over batch)")
    print("=" * 82)
    print(f"  candidates fetched = {out['n_candidates']}, "
          f"suitable scanned = {out['n_real_scanned']}, "
          f"skipped = {out['n_skipped']}, "
          f"Bonferroni p <= {out['bonferroni_threshold']:.5f}")
    print("  " + "-" * 78)
    print(f"  {'sequence':>22}  {'prob':>5}  {'omega':>7}  {'ratio':>6}  "
          f"{'power':>6}  {'p_value':>8}  {'verdict':>7}")
    for r in out["results"]:
        prob = r.get("problem", "-")
        print(f"  {r['name']:>22}  {str(prob):>5}  {r['peak_omega']:>7.3f}  "
              f"{r['geometric_ratio']:>6.3f}  {r['peak_power']:>6.3f}  "
              f"{r['p_value']:>8.4f}  {r['verdict']:>7}")
    hits = [r for r in out["results"]
            if r["kind"] == "oeis" and r["verdict"] == "DSI"]
    print()
    print(f"  DSI hits among real OEIS sequences: {len(hits)}")
    for h in hits:
        print(f"    {h['name']} (problem {h['problem']}, {h['state']}): "
              f"omega={h['peak_omega']:.3f}, ratio={h['geometric_ratio']:.3f}, "
              f"p={h['p_value']:.4f}, tags={h['tags']}")
    print()
    out_path = Path(__file__).parent / "erdos_oeis_dsi_sweep_results.json"
    # strip arrays already gone; results are JSON-safe
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
