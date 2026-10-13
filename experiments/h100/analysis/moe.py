"""Qwen3-30B-A3B (MoE) on the H100, V2 runner: each config as % of the best fixed length per load (closed loop), and
the controller's repair of vLLM's docs table / 4 against that table without it. Data: experiments/results/h100-studies/moe/results.csv."""
import collections, csv, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import STUDIES  # noqa: E402
D = os.path.join(STUDIES, "moe")
rows = [r for r in csv.DictReader(open(f"{D}/results.csv")) if r["done"] == "1"]
t = collections.defaultdict(list)
for r in rows:
    t[(r["study"], r["config"], int(r["point"][1:]))].append(float(r["tok_s"]))
mean = {k: (st.mean(v), len(v)) for k, v in t.items()}
C = [1, 2, 4, 8, 16, 32, 64, 96, 128]
best = {c: max(mean[("closed", f, c)][0] for f in ("k1", "k2", "k3") if ("closed", f, c) in mean) for c in C
        if any(("closed", f, c) in mean for f in ("k1", "k2", "k3"))}
print("=== MoE closed loop: % of the best fixed length at each load (n = runs)")
print(f"{'config':10s} " + " ".join(f"{c:>6d}" for c in C) + "   worst  n")
for cfg in ["nospec", "k1", "k2", "k3", "docs", "tt", "tt013", "ts", "nj"]:
    v = {c: mean[("closed", cfg, c)] for c in C if ("closed", cfg, c) in mean and c in best}
    if v:
        rel = {c: 100 * v[c][0] / best[c] for c in v}
        print(f"{cfg:10s} " + " ".join(f"{rel[c]:6.1f}" if c in rel else "     -" for c in C)
              + f"  {min(rel.values()):6.1f}  {'/'.join(map(str, sorted({n for _, n in v.values()})))}")
print("best fixed tok/s:", {c: round(best[c]) for c in best})
L = [4, 16, 32, 40, 64, 96]
on = {c: mean.get(("ctrl", "rescaled-ctrl", c)) for c in L}
off = {c: mean.get(("ctrl", "rescaled", c)) for c in L}
if any(on.values()) and any(off.values()):
    print("=== controller on vLLM's docs table / 4 (mismatched prior): gain over the same table without it")
    for c in L:
        if on[c] and off[c]:
            ref = f", tuned table {100 * (mean[('closed', 'tt', c)][0] / off[c][0] - 1):+.1f}%" if ("closed", "tt", c) in mean else ""
            print(f"  {c:4d}: {100 * (on[c][0] / off[c][0] - 1):+6.1f}%  (n={on[c][1]}/{off[c][1]}){ref}")
