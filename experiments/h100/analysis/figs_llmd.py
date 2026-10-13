"""fig_llmd.pdf (Figure 5): llm-d optimized-baseline, throughput gain over fixed draft length 3: (a) the guide's own
layout (2 x Qwen3-32B at TP=2), (b) 4 x Qwen3-8B at TP=1. Mean of 3 runs. Data: h100-studies/llmd/<model>.csv."""
import os, statistics as st, sys
import matplotlib; matplotlib.use("pdf")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llmd import load  # noqa: E402
from paths import OUT  # noqa: E402
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.6, "xtick.color": "#55554f", "ytick.color": "#55554f",
    "axes.labelcolor": "#2b2b28", "legend.frameon": False, "lines.linewidth": 1.6, "pdf.fonttype": 42})
GRID = "#e4e3dd"; C = [4, 8, 16, 32, 64, 128, 256, 512, 1024]; x = list(range(len(C)))
S = [("tuned13", "TableTune table", "#2a78d6", "o", "-", 2.0),
     ("docs-ctrl", "Docs table + controller", "#1baf7a", "^", "-.", 1.3), ("docs", "Docs table", "#eb6834", "s", "--", 1.3)]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
for ax, model, title in [(a, "qwen3-32b", "(a) Guide as shipped: 2 x Qwen3-32B, TP=2"), (b, "qwen3-8b", "(b) 4 x Qwen3-8B, TP=1")]:
    data = load(model)
    v = lambda cfg, c: st.mean(r["tok_s"] for r in data[(cfg, c)]) if (cfg, c) in data else None
    ax.axhline(0, color="#b9b8b0", lw=0.8, zorder=0)   # reference: fixed length 3; the guide's default does not speculate
    for cfg, name, col, mk, ls, lw in S:
        ys = [(i, 100 * (v(cfg, c) / v("k3", c) - 1)) for i, c in enumerate(C) if v(cfg, c) and v("k3", c)]
        if ys:
            ax.plot([p for p, _ in ys], [q for _, q in ys], color=col, marker=mk, ms=3.5, ls=ls, lw=lw, label=name, zorder=4 if cfg == "tuned13" else 3)
    ax.set_title(title, fontsize=8, loc="left"); ax.set_xticks(x); ax.set_xticklabels(C); ax.set_xlabel("Open requests (all replicas)")
    ax.grid(axis="y", color=GRID, lw=0.5); ax.tick_params(length=2)
a.set_ylabel("Gain over fixed length 3 (%)")
h, l = a.get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, handlelength=2.4)
fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(f"{OUT}/fig_llmd.pdf"); fig.savefig(f"{OUT}/fig_llmd.png", dpi=200); print("ok")
