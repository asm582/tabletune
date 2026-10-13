"""Experiments 1-3 in the paper's terms, from experiments/results/h100-studies/baselines.

Usage: python baselines.py [closed|vary|open|drift ...]

  closed  (Exp 1)  each config's throughput as % of the best fixed length at each load, mean of runs; worst load
  vary    (Exp 1b) jumping load: % of the per-window best configuration, total vs no speculation
  open    (Exp 2)  open-loop: throughput, TTFT/ITL/latency per rate and arrival pattern
  drift   (Exp 3)  workload switch: stale table's loss after the switch, and how much each controller recovers
"""
import collections
import csv
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import STUDIES  # noqa: E402

D = os.path.join(STUDIES, "baselines")
m = st.mean


def closed():
    rows = [r for r in csv.DictReader(open(f"{D}/results.csv")) if r["study"] in ("closed-eagle", "closed-dflash")]
    t = collections.defaultdict(list)
    for r in rows:
        if r["done"] == "1":
            t[(r["study"], r["config"], int(r["point"][1:]), r["run"])] = float(r["tok_s"])
    C = [1, 2, 4, 8, 16, 32, 64, 96, 128]
    out = {}
    for study, fixed, order in [
            ("closed-eagle", ["k1", "k2", "k3"], ["nospec", "k1", "k2", "k3", "tt", "tt013", "tt013-ctrl", "ts", "nj"]),
            ("closed-dflash", ["df-k3", "df-k4", "df-k8"], ["df-k3", "df-k4", "df-k8", "df-tt", "df-tt-ctrl", "df-ts", "df-nj"])]:
        runs = sorted({k[3] for k in t if k[0] == study})
        mean = {}
        for cfg in order:
            for c in C:
                v = [t[(study, cfg, c, r)] for r in runs if (study, cfg, c, r) in t]
                if v:
                    mean[(cfg, c)] = (m(v), len(v))
        best = {c: max(mean[(f, c)][0] for f in fixed if (f, c) in mean) for c in C if any((f, c) in mean for f in fixed)}
        print(f"\n=== {study}: % of the best fixed length at each load (mean of runs; n = runs)")
        print(f"{'config':12s} " + " ".join(f"{c:>6d}" for c in C) + "   worst   n")
        for cfg in order:
            rel = [100 * mean[(cfg, c)][0] / best[c] for c in C if (cfg, c) in mean and c in best]
            if not rel:
                continue
            ns = {mean[(cfg, c)][1] for c in C if (cfg, c) in mean}
            print(f"{cfg:12s} " + " ".join(f"{100 * mean[(cfg, c)][0] / best[c]:6.1f}" if (cfg, c) in mean else "     -" for c in C)
                  + f"  {min(rel):6.1f}  {'/'.join(map(str, sorted(ns)))}")
            out[(study, cfg)] = {c: 100 * mean[(cfg, c)][0] / best[c] for c in C if (cfg, c) in mean}
        # paired run-to-run spread of the fixed lengths, for scale
        spread = [100 * (max(v) - min(v)) / m(v) for (s, cfg, c), v in
                  _group(t, study).items() if len(v) > 1]
        if spread:
            print(f"run-to-run spread over (config, load): median {st.median(spread):.1f}%, max {max(spread):.1f}%")
    return out


def _group(t, study):
    g = collections.defaultdict(list)
    for (s, cfg, c, r), v in t.items():
        if s == study:
            g[(s, cfg, c)].append(v)
    return g


def segs(path_noext, sfxs=("", "_r2", "_r3")):
    out = []
    for s in sfxs:
        f = f"{path_noext}{s}.json"
        if os.path.exists(f):
            out.append(json.load(open(f))["segments"])
    return out


def vary():
    R = f"{D}/vary/"
    cfgs = ["nospec", "k2", "tt", "tt013", "tt013_ctrl", "ts", "nj"]
    S = {c: segs(R + c) for c in cfgs}
    S = {c: v for c, v in S.items() if v}
    if not S:
        return
    W = range(1, 12)
    T = {c: [m([r[i]["tok_s_engine"] for r in S[c]]) for i in W] for c in S}
    best = [max(T[c][j] for c in S) for j in range(11)]
    conc = [S[next(iter(S))][0][i]["conc"] * 2 for i in W]
    print("\n=== jumping load: tok/s per 50 s window (mean of runs), open requests = 2 clients x conc")
    print("win  open " + " ".join(f"{c:>10s}" for c in S))
    for j in range(11):
        print(f"{j + 1:3d} {conc[j]:5d} " + " ".join(f"{T[c][j]:10.0f}" for c in S))
    tot = {c: m(T[c]) for c in S}
    for c in S:
        rel = [100 * T[c][j] / best[j] for j in range(11)]
        per_run = [m([r[i]["tok_s_engine"] for i in W]) for r in S[c]]
        vs = f"vs nospec {100 * (tot[c] / tot['nospec'] - 1):+.1f}%" if "nospec" in tot else ""
        print(f"{c:11s} n={len(S[c])} min {min(rel):5.1f}%  mean {m(rel):5.1f}% of per-window best; total {tot[c]:7.0f} {vs}"
              f"  per-run {[round(x) for x in per_run]}")


def open_loop():
    rows = [r for r in csv.DictReader(open(f"{D}/results.csv")) if r["study"] == "open" and r["done"] == "1"]
    if not rows:
        return
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["config"], r["point"])].append(r)
    cfgs = [c for c in ["nospec", "k1", "k3", "tt", "tt013", "ts", "nj"] if any(k[0] == c for k in g)]
    pts = sorted({k[1] for k in g}, key=lambda p: (p.split("-r")[0], int(p.split("-r")[1])))
    for metric, label in [("tok_s", "output tok/s"), ("itl_p50", "median ITL (ms)"), ("ttft_p50", "median TTFT (ms)"),
                          ("ttft_p99", "p99 TTFT (ms)"), ("lat_p50", "median request latency (ms)"), ("lat_p99", "p99 request latency (ms)")]:
        print(f"\n=== open loop: {label} (mean of runs)")
        print(f"{'point':14s} " + " ".join(f"{c:>9s}" for c in cfgs))
        for p in pts:
            vals = []
            for c in cfgs:
                v = [float(r[metric]) for r in g.get((c, p), []) if r[metric]]
                vals.append(f"{m(v):9.1f}" if v else "        -")
            print(f"{p:14s} " + " ".join(vals))


def drift():
    for scen, label in [("s1", "S1 code -> math @96"), ("s2", "S2 translation -> math @48")]:
        R = f"{D}/drift/res-{scen}/"
        cfgs = ["static_old", "static_new", "ctrl", "ts", "nj"]
        S = {c: segs(R + c) for c in cfgs}
        S = {c: v for c, v in S.items() if v}
        if "static_old" not in S or "static_new" not in S:
            continue
        PRE, POST = [1, 2, 3], [4, 5, 6, 7, 8]
        tok = lambda c, idx: [m([r[i]["tok_s_engine"] for i in idx]) for r in S[c]]
        old, new = m(tok("static_old", POST)), m(tok("static_new", POST))
        print(f"\n=== drift {label}: after the switch the stale table runs {old:.0f} tok/s, the new workload's own table "
              f"{new:.0f} (stale loss {100 * (1 - old / new):.1f}%)")
        for c in S:
            post, pre = m(tok(c, POST)), m(tok(c, PRE))
            rec = 100 * (post - old) / (new - old) if new != old else float("nan")
            print(f"  {c:11s} n={len(S[c])} pre {pre:7.0f}  post {post:7.0f}  vs own table {100 * (post / new - 1):+6.1f}%  "
                  f"vs stale {100 * (post / old - 1):+6.1f}%  recovers {rec:5.0f}% of the gap   per-window "
                  f"{[round(m([r[i]['tok_s_engine'] for r in S[c]])) for i in range(9)]}")
        dl = lambda c: [round(m([r[i].get("mean_draft_len", 0) for r in S[c]]), 2) for i in range(9)]
        for c in S:
            print(f"  {c:11s} draft length by window {dl(c)}")
        print("  batch by window (stale):", [round(m([r[i]["batch_mean"] for r in S["static_old"]]), 1) for i in range(9)])


if __name__ == "__main__":
    which = sys.argv[1:] or ["closed", "vary", "open", "drift"]
    for w in which:
        {"closed": closed, "vary": vary, "open": open_loop, "drift": drift}[w]()
