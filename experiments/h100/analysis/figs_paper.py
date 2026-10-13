"""Paper figures from the matrix data: fig_h100.pdf (H100 V2 vs V1, relative to the best fixed length) and
fig_ctrl_gpus.pdf ((a) the controller repairing the same mistuned table on T4, A10G, H100 V1, H100 V2; (b) a
workload switch with DFlash). Figures 1 and 4."""
import csv, glob, os, statistics as st, sys
import matplotlib; matplotlib.use("pdf")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import OUT, STUDIES, SWEEPS  # noqa: E402
OLD = V1 = V2 = SWEEPS
OUT += "/"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.6, "xtick.color": "#55554f", "ytick.color": "#55554f",
    "axes.labelcolor": "#2b2b28", "legend.frameon": False, "lines.linewidth": 1.6, "pdf.fonttype": 42})
GRID = "#e4e3dd"
def load(root, sweep):
    d = {}
    for p in [f"{root}/{sweep}"] + sorted(glob.glob(f"{root}/{sweep}-r[0-9]")):
        if os.path.exists(f"{p}/summary.csv"):
            for r in csv.DictReader(open(f"{p}/summary.csv")):
                if r["tok_s_gpu"]: d.setdefault((r["config"], int(r["concurrency"])), []).append(float(r["tok_s_gpu"]))
    return d
m = st.mean
C = [1, 2, 4, 8, 16, 32, 64, 96, 128]; x = list(range(len(C)))
S = {"eagle-tuned": ("TableTune table", "#2a78d6", "o", "-"), "k1": ("Fixed 1", "#1baf7a", "^", ":"),
     "k2": ("Fixed 2", "#eda100", "v", ":"), "k3": ("Fixed 3", "#e87ba4", "D", ":"), "nospec": ("No speculation", "#008300", "x", "-.")}
# Figure 1 (motivation): the problem only -- no single fixed draft length is best at every load (H100, V2 runner)
fig, ax = plt.subplots(figsize=(3.35, 2.15))
d = load(V2, "code128t-h100v2")
ax.axhline(100, color=GRID, lw=0.8, zorder=0)
for cfg in ["nospec", "k1", "k2", "k3"]:
    name, col, mk, ls = S[cfg]
    ys = [100 * m(d[(cfg, c)]) / max(m(d[(k, c)]) for k in ["k1", "k2", "k3"]) for c in C]
    ax.plot(x, ys, color=col, marker=mk, ms=3.5, ls=ls, lw=1.5, label=name)
ax.set_xticks(x); ax.set_xticklabels(C); ax.set_xlabel("Open requests"); ax.set_ylabel("% of best fixed length")
ax.set_ylim(48, 106); ax.grid(axis="y", color=GRID, lw=0.5); ax.tick_params(length=2)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.8, ncol=4, handlelength=2.0, columnspacing=1.0)
fig.tight_layout(); fig.savefig(OUT + "fig_h100.pdf"); fig.savefig(OUT + "fig_h100.png", dpi=200)
# controller on the stress table, every GPU
# reference: the GPU's tuned table (same workload, same graph mode as the controller runs), from its code128t sweep
G = [("T4", OLD, "ctrlval-refit", "ctrlval-off", "code128t-t4", "dsd-tuned-v3", "#e87ba4", "s", ":"),
     ("A10G", OLD, "ctrlval-refit-a10g", "ctrlval-off-a10g", "code128t-a10g", "eagle-tuned", "#eda100", "^", "--"),
     ("H100 V1", V1, "ctrlval-refit-h100", "ctrlval-off-h100", "code128t-h100", "eagle-tuned", "#1baf7a", "D", "-."),
     ("H100 V2", V2, "ctrlval-refit-h100v2", "ctrlval-off-h100v2", "code128t-h100v2", "eagle-tuned", "#2a78d6", "o", "-")]
L = [4, 16, 32, 40, 64, 96]; xl = list(range(len(L)))
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1, 1]})
ax.axhline(0, color="#b9b8b0", lw=0.8, zorder=0)
for name, root, on, off, sweep, tuned, col, mk, ls in G:
    a, b, t = load(root, on), load(root, off), load(root, sweep)
    ax.plot(xl, [100 * (m(a[("dsd-rescaled", c)]) / m(b[("dsd-rescaled", c)]) - 1) for c in L], color=col, marker=mk, ms=3.5, ls=ls, lw=2.0 if "V2" in name else 1.4, label=name, zorder=3)
    ref = [(i, 100 * (m(t[(tuned, c)]) / m(b[("dsd-rescaled", c)]) - 1)) for i, c in enumerate(L) if (tuned, c) in t]   # no sweep point at 40
    ax.scatter([i for i, _ in ref], [y for _, y in ref], s=46, facecolors="none", edgecolors=col, linewidths=1.0, zorder=4)
ax.scatter([], [], s=46, facecolors="none", edgecolors="#55554f", linewidths=1.0, label="tuned table (reference)")
ax.set_xticks(xl); ax.set_xticklabels(L); ax.set_xlabel("Open requests"); ax.set_ylabel("Gain over the mistuned table (%)")
ax.grid(axis="y", color=GRID, lw=0.5); ax.tick_params(length=2); ax.legend(loc="upper left", fontsize=6.6)
ax.set_title("(a) Repairing a mistuned table (EAGLE-3)", fontsize=8, loc="left")
# (b) workload switch with DFlash on the H100 (V2), data from the baseline study (h100-studies/baselines/drift)
import json
DR = os.path.join(STUDIES, "baselines", "drift", "res-s1") + "/"
def segs(name):
    return [json.load(open(f))["segments"] for f in [DR + name + sfx + ".json" for sfx in ("", "_r2", "_r3")] if os.path.exists(f)]
for name, label, col, mk, ls, lw in [("static_old", "Code's table (stale)", "#b9b8b0", "o", "-", 1.3),
                                     ("static_new", "Math's table", "#55554f", "o", ":", 1.3),
                                     ("ctrl", "Controller, from code's table", "#2a78d6", "D", "-", 2.0)]:
    S_ = segs(name)
    bx.plot(range(1, 9), [m([r[i]["tok_s_engine"] for r in S_]) / 1000 for i in range(1, 9)], color=col, marker=mk, ms=3, ls=ls, lw=lw, label=label)
bx.axvline(3.5, color="#b9b8b0", lw=0.8)
bx.text(3.4, 23.3, "prompts switch\nto math", fontsize=6.5, color="#55554f", ha="right")
bx.set_xticks(range(1, 9)); bx.set_xlabel("60 s window"); bx.set_ylabel("Output tokens/s (thousands)")
bx.set_title("(b) A workload switch (DFlash, H100 V2, 96 req.)", fontsize=8, loc="left")
bx.grid(axis="y", color=GRID, lw=0.5); bx.tick_params(length=2); bx.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=6.6)
fig.tight_layout(); fig.savefig(OUT + "fig_ctrl_gpus.pdf"); fig.savefig(OUT + "fig_ctrl_gpus.png", dpi=200); print("ok")
