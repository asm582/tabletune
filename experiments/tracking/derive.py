"""Add comparison metrics and charts to the runs already in MLflow.

MLflow stores each benchmark point on its own; "how much faster than no
speculation?" needs the matching baseline run. This script matches points on
(dataset, temperature, concurrency) across all sweeps and logs, per point:

  concurrency                      numeric (the param is a string)
  gain_vs_nospec_pct               tok/s/GPU vs. no speculation
  gain_vs_best_fixed_k_pct         tok/s/GPU vs. the best fixed K of the same drafter
                                   (EAGLE-3: k1..k3; n-gram: ngram-k1..ngram-k5)
  gain_vs_controller_off_pct       controller-on runs: vs. the same table, controller off
  ttft_p50_vs_nospec_pct           TTFT change vs. no speculation (negative = faster)
  itl_p50_vs_nospec_pct            inter-token latency change vs. no speculation

and attaches gain_latency.png to every sweep's parent run: throughput gain,
TTFT and ITL (p50 and p99) against concurrency, one line per config. The
'compare-controller' run gets controller_value.png (all controller runs) and
controller_on_off.png (per starting table: controller off vs. on, gap shaded).

Baselines with several matching runs (the same point measured in two sweeps)
are averaged. Only reads/writes MLflow: runs on the laptop, no GPU.

Usage: MLFLOW_TRACKING_URI=http://localhost:5000 \\
       uv run --no-project --with mlflow-skinny==3.16.1 --with matplotlib python experiments/tracking/derive.py
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mlflow.entities import Metric  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from plot_pareto import COLORS, GRID, LABELS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY  # noqa: E402

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "vllm-perf-dsd")
FIXED_K = re.compile(r"(ngram-)?k\d+")   # fixed-K configs: k1..k3 (EAGLE-3), ngram-k1..k5


def base_sweep(sweep: str) -> str:
    """'code128t-t4-r2' -> 'code128t-t4': repeats are averaged with run 1 in the charts."""
    return re.sub(r"-r\d+$", "", sweep)


def drafter(cfg: str) -> str:
    """Configs are compared only with fixed K of the same drafter."""
    return "ngram" if cfg.startswith("ngram") else "eagle"
FALLBACK = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def mode(r) -> str:
    if r.data.tags.get("controller") != "on":
        return ""
    return " + hysteresis only" if r.data.params.get("controller_refit") == "off" else " + controller"


def setup_key(r) -> tuple:
    p = r.data.params
    return (p.get("dataset"), p.get("temperature"), int(p.get("concurrency", 0)))


def avg(runs, m):
    vals = [r.data.metrics[m] for r in runs if m in r.data.metrics]
    return mean(vals) if vals else None


def pct(new, base):
    return None if new is None or not base else 100.0 * (new / base - 1)


def main() -> None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        sys.exit("set MLFLOW_TRACKING_URI")
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([exp.experiment_id], "", max_results=5000)
    points = [r for r in runs if "run_key" in r.data.tags]
    parents = {r.info.run_id: r for r in runs if r.data.tags.get("kind") == "sweep"}

    nospec, fixed, ctrl_off = defaultdict(list), defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for r in points:
        cfg, key = r.data.tags["config"], setup_key(r)
        if cfg == "nospec":
            nospec[key].append(r)
        elif FIXED_K.fullmatch(cfg):
            fixed[(key, drafter(cfg))][cfg].append(r)
        if r.data.tags.get("controller") == "off" and r.data.params.get("dsd_table"):
            ctrl_off[(key, r.data.params["dsd_table"])].append(r)

    n = 0
    for r in points:
        key, m = setup_key(r), r.data.metrics
        derived = {"concurrency": float(key[2])}
        base = nospec.get(key, [])
        derived["gain_vs_nospec_pct"] = pct(m.get("tok_s_gpu"), avg(base, "tok_s_gpu"))
        derived["ttft_p50_vs_nospec_pct"] = pct(m.get("ttft_ms_p50"), avg(base, "ttft_ms_p50"))
        derived["itl_p50_vs_nospec_pct"] = pct(m.get("itl_ms_p50"), avg(base, "itl_ms_p50"))
        best = [avg(v, "tok_s_gpu")
                for v in fixed.get((key, drafter(r.data.tags["config"])), {}).values()]
        derived["gain_vs_best_fixed_k_pct"] = pct(m.get("tok_s_gpu"), max(best) if best else None)
        if r.data.tags.get("controller") == "on":
            off = ctrl_off.get((key, r.data.params.get("dsd_table")), [])
            derived["gain_vs_controller_off_pct"] = pct(m.get("tok_s_gpu"), avg(off, "tok_s_gpu"))
        # Timestamp now: MLflow shows the latest value, and on equal timestamps the
        # largest, so a fixed timestamp would keep stale larger values forever.
        now = int(time.time() * 1000)
        batch = [Metric(k, float(v), now, 0) for k, v in derived.items() if v is not None]
        client.log_batch(r.info.run_id, metrics=batch)
        r.data.metrics.update({k: v for k, v in derived.items() if v is not None})
        n += 1
    print(f"[derive] comparison metrics logged for {n} points")

    # One chart per sweep: gain, TTFT and ITL vs. concurrency, a line per config.
    by_parent = defaultdict(list)
    for r in points:
        by_parent[r.data.tags.get("mlflow.parentRunId")].append(r)
    for pid, rs in by_parent.items():
        if pid not in parents:
            continue
        sweep = parents[pid].info.run_name
        cfgs = sorted({r.data.tags["config"] for r in rs},
                      key=lambda c: list(COLORS).index(c) if c in COLORS else 99)
        suffix = {c: next((mode(r) for r in rs if r.data.tags["config"] == c), "") for c in cfgs}
        panels = [("gain_vs_nospec_pct", "Throughput gain vs. no speculation (%)"),
                  ("ttft_ms_p50", "TTFT p50 (ms)"), ("ttft_ms_p99", "TTFT p99 (ms)"),
                  ("itl_ms_p50", "Inter-token latency p50 (ms)"), ("itl_ms_p99", "Inter-token latency p99 (ms)")]
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.6), dpi=120)
        fig.patch.set_facecolor(SURFACE)
        for ax, (mname, title) in zip(axes, panels):
            ax.set_facecolor(SURFACE)
            for i, c in enumerate(cfgs):
                pts = sorted(((r.data.metrics["concurrency"], r.data.metrics.get(mname))
                              for r in rs if r.data.tags["config"] == c), key=lambda x: x[0])
                pts = [(x, y) for x, y in pts if y is not None]
                if not pts:
                    continue
                color = COLORS.get(c, FALLBACK[i % len(FALLBACK)])
                ax.plot(*zip(*pts), color=color, lw=2, marker="o", ms=4,
                        label=LABELS.get(c, c) + suffix[c])
            if mname == "gain_vs_nospec_pct":
                ax.axhline(0, color=TEXT_SECONDARY, lw=1, ls=(0, (4, 3)))
            ax.set_xscale("log", base=2)
            concs = sorted({r.data.metrics["concurrency"] for r in rs})
            ax.set_xticks(concs)
            ax.set_xticklabels([f"{c:g}" for c in concs])
            ax.minorticks_off()
            ax.set_title(title, fontsize=10, color=TEXT_PRIMARY, loc="left")
            ax.set_xlabel("Concurrency", color=TEXT_SECONDARY, fontsize=9)
            ax.grid(color=GRID, lw=0.6)
            ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False,
                   fontsize=9, labelcolor=TEXT_PRIMARY)
        ds = rs[0].data.params.get("dataset"); t = rs[0].data.params.get("temperature")
        fig.suptitle(f"{sweep}: {ds}, temperature {t}", x=0.01, ha="left", color=TEXT_PRIMARY)
        fig.tight_layout(rect=(0, 0.1, 1, 0.95))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "gain_latency.png"
            fig.savefig(path, facecolor=SURFACE)
            client.log_artifact(pid, str(path))
        plt.close(fig)
        print(f"[derive] chart attached to sweep {sweep}")

    controller_chart(client, exp.experiment_id, points)
    ngram_charts(client, exp.experiment_id, points)  # charts: mean of runs 1-3


CONTROLLER_SERIES = [  # (sweep, config, label, color)
    ("code128t-t4", "nospec", "No speculation", "#52514e"),
    ("code128t-t4", "dsd-tuned-v3", "Tuned table v3 (reference)", "#008300"),
    ("ctrlval-off", "dsd-docs", "Docs table, controller off", "#4a3aa7"),
    ("ctrlval-refit", "dsd-docs", "Docs table + controller", "#2a78d6"),
    ("ctrlval-off", "dsd-rescaled", "Hand table, controller off", "#e87ba4"),
    ("ctrlval-hyst", "dsd-rescaled", "Hand table + hysteresis only", "#eda100"),
    ("ctrlval-refit", "dsd-rescaled", "Hand table + controller", "#eb6834"),
]


def controller_chart(client: MlflowClient, exp_id: str, points) -> None:
    """One view of the controller experiment across its sweeps, as its own MLflow run."""
    idx = {(r.data.tags["sweep"], r.data.tags["config"]): [] for r in points}
    for r in points:
        idx[(r.data.tags["sweep"], r.data.tags["config"])].append(r)
    if not all((s, c) in idx for s, c, _, _ in CONTROLLER_SERIES):
        return
    concs = sorted({r.data.metrics["concurrency"] for s, c, _, _ in CONTROLLER_SERIES[2:]
                    for r in idx[(s, c)]})
    panels = [("tok_s_gpu", "Throughput (tok/s per GPU)"),
              ("gain_vs_nospec_pct", "Throughput gain vs. no speculation (%)"),
              ("ttft_ms_p50", "TTFT p50 (ms)"), ("itl_ms_p50", "Inter-token latency p50 (ms)")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    for ax, (m, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for s, c, label, color in CONTROLLER_SERIES:
            pts = sorted((r.data.metrics["concurrency"], r.data.metrics.get(m)) for r in idx[(s, c)]
                         if r.data.metrics["concurrency"] in concs and r.data.metrics.get(m) is not None)
            if pts:
                ls = (0, (4, 3)) if c in ("nospec", "dsd-tuned-v3") else "-"
                ax.plot(*zip(*pts), color=color, lw=2, ls=ls, marker="o", ms=4, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(concs)
        ax.set_xticklabels([f"{x:g}" for x in concs])
        ax.minorticks_off()
        ax.set_title(title, fontsize=10, color=TEXT_PRIMARY, loc="left")
        ax.set_xlabel("Concurrency", color=TEXT_SECONDARY, fontsize=9)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9,
               labelcolor=TEXT_PRIMARY)
    fig.suptitle("Controller value: instruct-coder, temperature 1.0, max_num_seqs 128",
                 x=0.01, ha="left", color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    found = client.search_runs([exp_id], "tags.group_key = 'compare-controller'")
    if found:
        run_id = found[0].info.run_id
    else:
        run_id = client.create_run(exp_id, run_name="compare-controller",
                                   tags={"group_key": "compare-controller", "kind": "comparison"}).info.run_id
        client.set_terminated(run_id)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "controller_value.png"
        fig.savefig(path, facecolor=SURFACE)
        client.log_artifact(run_id, str(path))
    plt.close(fig)
    print("[derive] controller comparison chart attached to run 'compare-controller'")
    controller_on_off_chart(client, run_id, idx)


def controller_on_off_chart(client: MlflowClient, run_id: str, idx) -> None:
    """Presentation view: per misaligned starting table, controller off vs. on, gap shaded, gains labelled."""
    def series(s, c):
        """Mean over the sweep and its repeats (<sweep>-r2, -r3, ...) per concurrency."""
        vals = defaultdict(list)
        for (sw, cfg), runs in idx.items():
            if cfg == c and base_sweep(sw) == s:
                for r in runs:
                    vals[r.data.metrics["concurrency"]].append(r.data.metrics["tok_s_gpu"])
        return {x: mean(v) for x, v in sorted(vals.items())}

    nospec, tuned = series("code128t-t4", "nospec"), series("code128t-t4", "dsd-tuned-v3")
    tables = [("dsd-docs", "Start: vLLM docs example"), ("dsd-rescaled", "Start: docs table ÷4 (K=0 above 32)")]
    fig, axes = plt.subplots(1, 2, figsize=(16.64, 4.6), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (cfg, title) in zip(axes, tables):
        off, on = series("ctrlval-off", cfg), series("ctrlval-refit", cfg)
        cs = [x for x in off if x in on]
        lo, hi = [off[x] for x in cs], [on[x] for x in cs]
        ref = [x for x in nospec if cs[0] <= x <= cs[-1]]
        ax.set_facecolor(SURFACE)
        ax.plot(ref, [nospec[x] for x in ref], ":", color="#8A919C", lw=2)
        ax.plot(ref, [tuned[x] for x in ref], "--", color="#14213D", lw=1.6)
        ax.fill_between(cs, lo, hi, where=[b > a for a, b in zip(lo, hi)], color="#1F5FA8", alpha=0.15, lw=0)
        ax.plot(cs, lo, "-o", color="#B8501F", lw=2.5, ms=7)
        ax.plot(cs, hi, "-o", color="#1F5FA8", lw=2.5, ms=7)
        for x, a, b in zip(cs, lo, hi):
            if b / a - 1 >= 0.05:  # label only wins well above single-run noise (~3%)
                last = x == cs[-1]
                ax.annotate(f"+{100 * (b / a - 1):.0f}%", (x, a + 0.35 * (b - a)), xytext=(-10 if last else 10, 0),
                            textcoords="offset points", va="center", ha="right" if last else "left",
                            fontsize=15, fontweight="bold", color="#1F5FA8",
                            bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))
        end = cs[-1]
        ax.annotate("ON", (end, on[end]), xytext=(8, 10), textcoords="offset points",
                    color="#1F5FA8", fontsize=15, fontweight="bold", va="center")
        ax.annotate("OFF", (end, off[end]), xytext=(8, -12), textcoords="offset points",
                    color="#B8501F", fontsize=15, fontweight="bold", va="center")
        ax.annotate("tuned", (ref[-1], tuned[ref[-1]]), xytext=(8, 26), textcoords="offset points",
                    color="#14213D", fontsize=13, va="center")
        ax.annotate("no spec", (ref[-1], nospec[ref[-1]]), xytext=(8, -4),
                    textcoords="offset points", color="#8A919C", fontsize=13, va="center")
        ax.set_title(title, loc="left", fontsize=17, color=TEXT_PRIMARY, fontweight="bold")
        ax.set_xticks(cs)
        ax.set_xlim(0, cs[-1] * 1.18)
        ax.set_xlabel("Concurrent requests", color=TEXT_SECONDARY, fontsize=14)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=13)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Output tok/s per GPU", color=TEXT_SECONDARY, fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.995, 0.005, "Mean of 3 runs · Tesla T4 · MLflow run compare-controller",
             ha="right", va="bottom", fontsize=11, color=TEXT_SECONDARY)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "controller_on_off.png"
        fig.savefig(path, facecolor=SURFACE)
        client.log_artifact(run_id, str(path))
    plt.close(fig)
    print("[derive] controller on/off chart attached to run 'compare-controller'")



# --------------------------------------------------------- n-gram (issue #3)
NGRAM_TUNER = Path(__file__).resolve().parent.parent / "results" / "ngram-tuner"
BLUE, ORANGE, INK, GRAY, TEAL, SAND = "#1F5FA8", "#B8501F", "#14213D", "#8A919C", "#2E8B7A", "#C9A227"


def ngram_charts(client: MlflowClient, exp_id: str, points) -> None:
    """Three views of the n-gram study across its sweeps, on a 'compare-ngram' run:
    ngram_tuned_vs_fixed.png, ngram_cost_model.png, ngram_controller.png."""
    # Mean over each sweep and its repeats (<sweep>-r2, -r3): (tok/s per GPU, live batch).
    acc: dict[tuple[str, str], dict[float, list]] = defaultdict(lambda: defaultdict(list))
    for r in points:
        m = r.data.metrics
        if r.data.tags.get("sweep", "").startswith("ngram") and "tok_s_gpu" in m:
            acc[(base_sweep(r.data.tags["sweep"]), r.data.tags["config"])][m["concurrency"]].append(
                (m["tok_s_gpu"], m.get("running_avg", m["concurrency"])))
    series = {k: {c: (mean(t for t, _ in v), mean(b for _, b in v)) for c, v in d.items()}
              for k, d in acc.items()}
    base = "ngram-apply-t4"
    fixed = {k: series.get((base, f"ngram-k{k}"), {}) for k in (1, 3, 5)}
    if not all(fixed.values()):
        return
    concs = sorted(fixed[3])
    best = {c: max(fixed[k][c][0] for k in fixed if c in fixed[k]) for c in concs}
    charts = []

    def axis(ax, title, ylabel, xs=None):
        xs = xs or concs
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{c:g}" for c in xs])
        ax.set_xlim(xs[0] * 0.8, xs[-1] * 2.6)
        ax.end_labels = []
        ax.minorticks_off()
        ax.set_title(title, loc="left", fontsize=15, color=TEXT_PRIMARY, fontweight="bold")
        ax.set_xlabel("Concurrent requests", color=TEXT_SECONDARY, fontsize=12)
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=12)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=11)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def line(ax, pts, label, color, style="-", lw=2.2, pct_of=None):
        xs = [c for c in concs if c in pts]
        ys = [100 * (pts[c][0] / pct_of[c] - 1) if pct_of else pts[c][0] for c in xs]
        ax.plot(xs, ys, style, color=color, lw=lw, marker="o", ms=5, label=label)
        ax.end_labels.append([ys[-1], xs[-1], label, color])

    def place_labels(ax):
        """End-of-line labels, pushed apart vertically so they do not overlap."""
        lo, hi = ax.get_ylim()
        gap = 0.07 * (hi - lo)
        items = sorted(ax.end_labels)
        for i in range(1, len(items)):
            items[i][0] = max(items[i][0], items[i - 1][0] + gap)
        over = items[-1][0] - (hi - 0.02 * (hi - lo)) if items else 0
        if over > 0:  # keep the stack inside the axes, pushing lower labels apart too
            items[-1][0] -= over
            for i in range(len(items) - 2, -1, -1):
                items[i][0] = min(items[i][0], items[i + 1][0] - gap)
        for y, x, label, color in items:
            ax.annotate(label, (x, y), xytext=(8, 0), textcoords="offset points",
                        color=color, fontsize=11, va="center", fontweight="bold",
                        annotation_clip=False)

    # 1. Fixed K vs. the tuned tables.
    v1 = series.get((base, "ngram-tuned"), {})
    v2 = series.get(("ngram-apply-t4-v2", "ngram-tuned-v2"), {})
    nospec = series.get((base, "nospec"), {})
    fig, axes = plt.subplots(1, 2, figsize=(16.64, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    axis(axes[0], "Throughput", "Output tok/s per GPU")
    if nospec:
        line(axes[0], nospec, "no spec", GRAY, ":")
    line(axes[0], fixed[3], "K=3", TEAL, lw=1.6)
    line(axes[0], fixed[5], "K=5", SAND, lw=1.6)
    if v2:
        line(axes[0], v2, "tuned table", BLUE, lw=3)
    axis(axes[1], "Versus the best fixed K at each load", "% vs. best fixed n-gram K")
    axes[1].axhline(0, color=INK, lw=0.8)
    line(axes[1], fixed[3], "K=3", TEAL, lw=1.6, pct_of=best)
    line(axes[1], fixed[5], "K=5", SAND, lw=1.6, pct_of=best)
    if v1:
        line(axes[1], v1, "average-cost table", ORANGE, "--", pct_of=best)
    if v2:
        line(axes[1], v2, "tuned table", BLUE, lw=3, pct_of=best)
    for ax in axes:
        place_labels(ax)
    charts.append((fig, "ngram_tuned_vs_fixed.png"))

    # 2. Cost model: K=5 / K=3 throughput, measured vs. predicted.
    try:
        sys.path.insert(0, str(NGRAM_TUNER.parents[2]))  # repo root, for dsd_tuning.core
        from dsd_tuning.core import load_acceptance, load_costs, tokens_per_step
        from dsd_tuning.tuner.validate_costmodel import predict_step_ms
        costs = load_costs(str(NGRAM_TUNER / "costs_k*.json"))
        accept, rate = load_acceptance(NGRAM_TUNER / "acceptance.json")
        tps = tokens_per_step(accept, proposal_rate=rate)
    except (ImportError, FileNotFoundError):
        costs = None
    if costs:
        fig, ax = plt.subplots(figsize=(16.64, 4.8), dpi=200)
        fig.patch.set_facecolor(SURFACE)
        axis(ax, "K=5 vs. K=3 throughput (above 1: K=5 is better)", "K=5 / K=3")
        ax.axhline(1, color=INK, lw=0.8)
        meas = {c: (fixed[5][c][0] / fixed[3][c][0], (fixed[5][c][1] + fixed[3][c][1]) / 2)
                for c in concs if c in fixed[5]}
        for model, label, color, style in (("mean", "predicted, average cost", ORANGE, "--"),
                                           ("full", "predicted, full cost", BLUE, "-")):
            pred = {c: ((tps[5] / predict_step_ms(costs, 5, b, rate, model))
                        / (tps[3] / predict_step_ms(costs, 3, b, rate, model)), b)
                    for c, (_, b) in meas.items()}
            line(ax, pred, label, color, style)
        line(ax, meas, "measured", INK, lw=3)
        place_labels(ax)
        charts.append((fig, "ngram_cost_model.png"))

    # 3. Controller runs, as % of the tuned table without the controller.
    if v2:
        runs = [("ngram-ctrl-off", "ngram-untuned", "untuned, off", ORANGE, "-"),
                ("ngram-ctrl-calib", "ngram-untuned", "untuned + ctrl", BLUE, "-"),
                ("ngram-ctrl-tuned", "ngram-tuned-v2", "tuned + ctrl", INK, "--")]
        runs = [r for r in runs if series.get(r[:2])]
        if runs:
            fig, ax = plt.subplots(figsize=(16.64, 4.8), dpi=200)
            fig.patch.set_facecolor(SURFACE)
            xs = sorted({c for r in runs for c in series[r[:2]]})
            axis(ax, "Controller runs vs. the tuned table without it", "% vs. tuned table, controller off", xs)
            ax.axhline(0, color=INK, lw=0.8)
            for sweep, cfg, label, color, style in runs:
                line(ax, series[(sweep, cfg)], label, color, style, pct_of={c: v2[c][0] for c in v2})
            place_labels(ax)
            charts.append((fig, "ngram_controller.png"))

    found = client.search_runs([exp_id], "tags.group_key = 'compare-ngram'")
    if found:
        run_id = found[0].info.run_id
    else:
        run_id = client.create_run(exp_id, run_name="compare-ngram",
                                   tags={"group_key": "compare-ngram", "kind": "comparison"}).info.run_id
        client.set_terminated(run_id)
    with tempfile.TemporaryDirectory() as d:
        for fig, name in charts:
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            fig.text(0.995, 0.005, "Mean of 3 runs · Tesla T4 · MLflow run compare-ngram",
                     ha="right", va="bottom", fontsize=11, color=TEXT_SECONDARY)
            fig.savefig(Path(d) / name, facecolor=SURFACE)
            client.log_artifact(run_id, str(Path(d) / name))
            plt.close(fig)
    print(f"[derive] n-gram charts attached to run 'compare-ngram': {[n for _, n in charts]}")


if __name__ == "__main__":
    main()
