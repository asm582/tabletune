"""DFlash real-acceptance study and long-prompt study: each config as % of the best fixed length per load (closed loop), and the
merge-tolerance tables (mt-*). Data: experiments/results/h100-studies/credib/results.csv."""
import collections, csv, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import STUDIES  # noqa: E402
D = os.path.join(STUDIES, "credib")
t = collections.defaultdict(list)
for r in csv.DictReader(open(f"{D}/results.csv")):
    if r["done"] == "1":
        t[(r["study"], r["config"], int(r["point"][1:]))].append(float(r["tok_s"]))
for study, fixed, cfgs in [("dflash-acc", ["df-k3", "df-k4"], ["df-k3", "df-k4", "df-tt", "df-tt-acc", "df-tt-raw"]),
                           ("long1024", ["k1", "k2", "k3"], ["k1", "k2", "k3", "docs", "tt-short", "tt-long"]),
                           ("long3072", ["k1", "k2", "k3"], ["k1", "k2", "k3", "docs", "tt-short", "tt-long"]),
                           ("mt-eagle", ["k1", "k2", "k3"], ["k1", "k2", "k3", "old", "new"]),
                           ("mt-ngram", ["k1", "k3", "k5"], ["k1", "k3", "k5", "new"]),
                           ("mt-specbench", ["k1", "k2", "k3"], ["k1", "k2", "k3", "new"]),
                           ("mt-dflash", ["k3", "k4", "k8"], ["k3", "k4", "k8", "old", "new"])]:
    C = sorted({c for (s, _, c) in t if s == study})
    if not C:
        continue
    best = {c: max(st.mean(t[(study, f, c)]) for f in fixed if (study, f, c) in t) for c in C}
    print(f"=== {study}: % of the best of {fixed} at each load (n = runs)")
    print(f"{'config':10s} " + " ".join(f"{c:>6d}" for c in C) + "   worst  n")
    for cfg in cfgs:
        v = {c: st.mean(t[(study, cfg, c)]) for c in C if (study, cfg, c) in t}
        if v:
            rel = {c: 100 * v[c] / best[c] for c in v}
            n = sorted({len(t[(study, cfg, c)]) for c in v})
            print(f"{cfg:10s} " + " ".join(f"{rel[c]:6.1f}" if c in rel else "     -" for c in C) + f"  {min(rel.values()):6.1f}  {n}")
