"""H100 workload switch (e1) and jumping-load (e2) metrics, 3 runs, in the paper's terms (Table 2, Q1).
Usage: python shift.py [res-h100v2m | res-h100v2m013 | res-h100m]   (V2 all-length table, V2 0/1/3 table, V1)"""
import json, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import STUDIES  # noqa: E402
R = os.path.join(STUDIES, "shift", sys.argv[1] if len(sys.argv) > 1 else "res-h100v2m") + "/"
SFX = ["", "_r2", "_r3"]
def segs(name):  # per run: list of tok_s_engine per segment
    out = []
    for s in SFX:
        try: out.append(json.load(open(R + name + s + ".json"))["segments"])
        except FileNotFoundError: pass
    return out
def tok(name, idx):  # per-run mean tok/s over segment indices
    return [st.mean(r[i]["tok_s_engine"] for i in idx) for r in segs(name)]
m = lambda v: st.mean(v)
def avg(v):  # mean over runs, skipping windows a stalled run did not report
    v = [x for x in v if x is not None]
    return round(st.mean(v), 2) if v else None
def div(a, b): return a / b if b else float("nan")   # a stalled run (V2 all-length table) has zero throughput
def rng(v): return f"{m(v):8.1f} [{min(v):.0f}, {max(v):.0f}] n={len(v)}"
PRE, POST, LATE, LAST = [1, 2, 3], [4, 5, 6, 7, 8], [5, 6, 7, 8], [8]
for d, stale, right, ctrls in [
    ("m2t", "e1_m2t_static_math", "e1_m2t_static_trans", ["e1_m2t_ctrl", "e1_m2t_ctrl_pd", "e1_m2t_ctrl_both"]),
    ("t2m", "e1_t2m_static_trans", "e1_t2m_static_math", ["e1_t2m_ctrl", "e1_t2m_ctrl_explore", "e1_t2m_ctrl_pd", "e1_t2m_ctrl_both"])]:
    print(f"\n=== {d}  (stale={stale}, right={right})")
    s, r = tok(stale, POST), tok(right, POST)
    print(f"  post-switch: stale {rng(s)} | right {rng(r)} | stale loss {100*(1-m(s)/m(r)):.1f}%  per-run {[round(100*(1-div(a, b)),1) for a,b in zip(s,r)]}")
    for c in ctrls:
        p, pl, last, pre = tok(c, POST), tok(c, LATE), tok(c, LAST), tok(c, PRE)
        rec = lambda x, idx: 100 * div(m(x) - m(tok(stale, idx)), m(tok(right, idx)) - m(tok(stale, idx)))
        print(f"  {c:22s} post {rng(p)} recov5 {rec(p, POST):6.0f}%  recov4 {rec(pl, LATE):6.0f}%  last vs right {100*(m(last)/m(tok(right, LAST))-1):+.1f}%"
              f"  pre vs stale-start-table {100*(m(pre)/m(tok(stale, PRE))-1):+.1f}%  vs right {100*(m(p)/m(r)-1):+.1f}%")
        dl = [avg(x[i]["mean_draft_len"] for x in segs(c)) for i in range(9)]
        print(f"  {'':22s} draft_len by window {dl}")
    print("  B by window (stale):", [avg(x[i]["batch_mean"] for x in segs(stale)) for i in range(9)])
print("\n=== e2 jumping load (windows 1..11)")
cfgs = [c for c in ["e2_nospec", "e2_k1", "e2_k2", "e2_tuned", "e2_tuned_ctrl", "e2_docs"] if segs(c)]
if not cfgs: sys.exit()
W = range(1, 12)
T = {c: [m([r[i]["tok_s_engine"] for r in segs(c)]) for i in W] for c in cfgs}
conc = [segs("e2_tuned")[0][i]["conc"] for i in W]
best = [max(T[c][j] for c in cfgs) for j in range(11)]
print("win conc " + " ".join(f"{c[3:]:>10}" for c in cfgs))
for j in range(11): print(f"{j+1:3d} {conc[j]:4d} " + " ".join(f"{T[c][j]:10.0f}" for c in cfgs))
tot = {c: m(T[c]) for c in cfgs}
print("tot      " + " ".join(f"{tot[c]:10.0f}" for c in cfgs))
for c in cfgs:
    rel = [100 * T[c][j] / best[j] for j in range(11)]
    perrun = [st.mean(r[i]["tok_s_engine"] for i in W) for r in segs(c)]
    print(f"{c:15s} min {min(rel):5.1f}%  mean {m(rel):5.1f}% of per-window best;  total vs tuned {100*(tot[c]/tot['e2_tuned']-1):+.1f}%  per-run totals {[round(x) for x in perrun]}")
pr = [100*(a/b-1) for a,b in zip([st.mean(r[i]["tok_s_engine"] for i in W) for r in segs("e2_tuned")],[st.mean(r[i]["tok_s_engine"] for i in W) for r in segs("e2_nospec")])]
print("tuned vs nospec total, paired by run:", [round(x,1) for x in pr])
pr = [100*(a/b-1) for a,b in zip([st.mean(r[i]["tok_s_engine"] for i in W) for r in segs("e2_tuned_ctrl")],[st.mean(r[i]["tok_s_engine"] for i in W) for r in segs("e2_tuned")])]
print("ctrl vs tuned total, paired by run:", [round(x,1) for x in pr])
print("B by window (tuned):", [avg(x[i]["batch_mean"] for x in segs("e2_tuned")) for i in W])
