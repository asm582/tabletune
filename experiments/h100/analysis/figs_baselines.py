"""fig_baselines.pdf (Figure 3): TableTune against TurboSpec-style and Nightjar-style control on the H100 (V2 runner).

(a) EAGLE-3 and (b) DFlash, closed loop: throughput as % of the best fixed length at each load, mean of runs.
"""
import collections
import csv
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import D  # noqa: E402
from paths import OUT  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.6, "xtick.color": "#55554f", "ytick.color": "#55554f",
                     "axes.labelcolor": "#2b2b28", "legend.frameon": False, "lines.linewidth": 1.6, "pdf.fonttype": 42})
GRID = "#e4e3dd"
TT, TS, NJ, FIX, NEW, OLD = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#55554f", "#b9b8b0"
C = [1, 2, 4, 8, 16, 32, 64, 96, 128]


def closed_rel():
    t = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{D}/results.csv")):
        if r["study"] in ("closed-eagle", "closed-dflash") and r["done"] == "1":
            t[(r["study"], r["config"], int(r["point"][1:]))].append(float(r["tok_s"]))
    mean = {k: st.mean(v) for k, v in t.items()}
    rel = {}
    for study, fixed in [("closed-eagle", ["k1", "k2", "k3"]), ("closed-dflash", ["df-k3", "df-k4", "df-k8"])]:
        best = {c: max(mean.get((study, f, c), 0) for f in fixed) for c in C}
        for (s, cfg, c), v in mean.items():
            if s == study and best[c]:
                rel.setdefault((s, cfg), {})[c] = 100 * v / best[c]
    return rel


def panel_closed(ax, rel, study, lines, title):
    x = list(range(len(C)))
    ax.axhline(100, color="#b9b8b0", lw=0.8, zorder=0)
    for cfg, name, col, mk, ls, lw in lines:
        v = rel.get((study, cfg))
        if v:
            ax.plot([i for i, c in zip(x, C) if c in v], [v[c] for c in C if c in v], color=col, marker=mk, ms=3.2,
                    ls=ls, lw=lw, label=name, zorder=4 if col == TT else 3)
    ax.set_xticks(x)
    ax.set_xticklabels(C)
    ax.set_xlabel("Open requests")
    ax.set_title(title, fontsize=8, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.tick_params(length=2)


def main():
    rel = closed_rel()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    panel_closed(a, rel, "closed-eagle", [
        ("tt", "TableTune table", TT, "o", "-", 2.0), ("tt013-ctrl", "TableTune table + controller", TT, "D", ":", 1.3),
        ("ts", "TurboSpec-style", TS, "s", "--", 1.3), ("nj", "Nightjar-style", NJ, "^", "-.", 1.3),
        ("k3", "Fixed length 3", FIX, "v", ":", 1.1)], "(a) EAGLE-3")
    panel_closed(b, rel, "closed-dflash", [
        ("df-tt", "TableTune table", TT, "o", "-", 2.0), ("df-ts", "TurboSpec-style", TS, "s", "--", 1.3),
        ("df-nj", "Nightjar-style", NJ, "^", "-.", 1.3), ("df-k3", "Fixed length 3", FIX, "v", ":", 1.1)], "(b) DFlash")
    a.set_ylabel("% of best fixed length")
    for ax in (a, b):
        ax.set_ylim(66, 105)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=6.8, handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(f"{OUT}/fig_baselines.pdf")
    fig.savefig(f"{OUT}/fig_baselines.png", dpi=200)
    print("ok")


if __name__ == "__main__":
    main()
