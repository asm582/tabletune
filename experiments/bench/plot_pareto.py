"""Summarize a sweep run and plot the throughput/latency Pareto chart.

Reads experiments/results/sweeps/<run>/<config>/c<N>/profile_export_aiperf.json (aiperf)
and server_metrics_export.json (vLLM /metrics scraped by aiperf), writes
summary.csv and pareto.png into experiments/results/sweeps/<run>/.

Usage: python experiments/bench/plot_pareto.py <run> [config ...]
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "results" / "sweeps"

# Fixed color per config (entity), never by rank. Reference categorical
# palette, light mode (dataviz skill references/palette.md).
COLORS = {
    "nospec": "#2a78d6",
    "k1": "#eb6834",
    "k2": "#1baf7a",
    "k3": "#eda100",
    "dsd-rescaled": "#e87ba4",
    "dsd-tuned": "#008300",
    "dsd-docs": "#4a3aa7",
    "dsd-tuned-v2": "#e34948",
    # The palette has 8 slots. v3 reuses v1's green: v1 and v3 are never
    # plotted together (v1 ran on spec-bench, v3 on code at T=1.0, c<=128).
    "dsd-tuned-v3": "#008300",
    # Controller-value chart (compare-ctrl): these series are only plotted
    # together with dsd-tuned-v3, so the other slots are free to reuse.
    "docs": "#4a3aa7",
    "docs+ctrl": "#2a78d6",
    "hand": "#e87ba4",
    "hand+hyst": "#eda100",
    "hand+ctrl": "#eb6834",
}
LABELS = {
    "nospec": "No speculation",
    "k1": "EAGLE-3, K=1",
    "k2": "EAGLE-3, K=2",
    "k3": "EAGLE-3, K=3",
    "dsd-rescaled": "Dynamic SD, hand table (docs example ÷4)",
    "dsd-tuned": "Dynamic SD, tuned table v1",
    "dsd-docs": "Dynamic SD, docs example table (unscaled)",
    "dsd-tuned-v2": "Dynamic SD, tuned table v2",
    "dsd-tuned-v3": "Dynamic SD, tuned table v3",
    "docs": "Docs table, controller off",
    "docs+ctrl": "Docs table + controller (re-fit, hysteresis)",
    "hand": "Hand table (with K=0), controller off",
    "hand+hyst": "Hand table + hysteresis only",
    "hand+ctrl": "Hand table + controller (re-fit, hysteresis)",
}
TEXT_PRIMARY, TEXT_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def metric(d: dict, key: str, stat: str = "avg"):
    v = d.get(key) or {}
    return v.get(stat)


def spec_counters(server_metrics: Path) -> dict:
    """vllm:spec_decode_* counter deltas over the profiling window.

    aiperf (0.13) stores counters as metrics[name].series[].stats.total. The
    per-position counter has one series per `position` label; it is returned
    as a list ordered by position.
    """
    if not server_metrics.exists():
        return {}
    metrics = json.loads(server_metrics.read_text()).get("metrics", {})
    out: dict = {}
    for name, m in metrics.items():
        if not name.startswith("vllm:spec_decode"):
            continue
        series = m.get("series", [])
        if name.endswith("per_pos"):
            by_pos = {int(s["labels"].get("position", 0)): s["stats"]["total"] for s in series}
            out[name] = [by_pos[p] for p in sorted(by_pos)]
        else:
            out[name] = sum(s["stats"]["total"] for s in series)
    return out


def kv_pressure(server_metrics: Path) -> dict:
    """Preemptions and actual batch size: shows when the KV cache, not the
    client, limited concurrency."""
    if not server_metrics.exists():
        return {}
    m = json.loads(server_metrics.read_text()).get("metrics", {})

    def stats(name):
        s = m.get(name, {}).get("series", [])
        return s[0]["stats"] if s else {}

    return {
        "preemptions": stats("vllm:num_preemptions").get("total"),
        "kv_usage_max": round(stats("vllm:kv_cache_usage_perc").get("max", 0), 2),
        "running_avg": round(stats("vllm:num_requests_running").get("avg", 0), 1),
    }


def load(run: str, configs: list[str]) -> list[dict]:
    rows = []
    for cfg in configs:
        for cdir in sorted((ROOT / run / cfg).glob("c*/"), key=lambda p: int(p.name[1:])):
            f = cdir / "profile_export_aiperf.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            rows.append({
                "config": cfg,
                "concurrency": int(cdir.name[1:]),
                "tok_s_user": metric(d, "output_token_throughput_per_user"),
                "tok_s_gpu": metric(d, "output_token_throughput"),
                "ttft_ms_p50": metric(d, "time_to_first_token", "p50"),
                "itl_ms_p50": metric(d, "inter_token_latency", "p50"),
                "itl_ms_p99": metric(d, "inter_token_latency", "p99"),
                "requests": metric(d, "request_count"),
                "output_tokens": metric(d, "total_output_tokens"),
                "osl_avg": metric(d, "output_sequence_length"),
                "errors": len(d.get("error_summary") or []),
                **kv_pressure(cdir / "server_metrics_export.json"),
            })
    return rows


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper-right frontier: no other point is better on both axes."""
    front, best_y = [], float("-inf")
    for x, y in sorted(points, key=lambda p: -p[0]):
        if y > best_y:
            front.append((x, y))
            best_y = y
    return sorted(front)


def plot(rows: list[dict], configs: list[str], out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    all_pts = [(r["tok_s_user"], r["tok_s_gpu"]) for r in rows]
    front = pareto_front(all_pts)
    if len(front) > 1:
        ax.plot(*zip(*front), color=TEXT_SECONDARY, lw=1, ls=(0, (4, 3)), zorder=1,
                label="Best of all configs (envelope)")

    for cfg in configs:
        pts = sorted((r for r in rows if r["config"] == cfg), key=lambda r: r["concurrency"])
        if not pts:
            continue
        xs = [r["tok_s_user"] for r in pts]
        ys = [r["tok_s_gpu"] for r in pts]
        color = COLORS.get(cfg, TEXT_SECONDARY)
        ax.plot(xs, ys, color=color, lw=2, zorder=3, label=LABELS.get(cfg, cfg))
        # Filled marker = clean point; hollow ring = KV cache full (preemptions),
        # so the client's concurrency was not actually reached.
        clean = [(r["tok_s_user"], r["tok_s_gpu"]) for r in pts if not r.get("preemptions")]
        full = [(r["tok_s_user"], r["tok_s_gpu"]) for r in pts if r.get("preemptions")]
        if clean:
            ax.scatter(*zip(*clean), s=36, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        if full:
            ax.scatter(*zip(*full), s=56, facecolor=SURFACE, edgecolor=color, linewidth=2, zorder=5)
    # End labels collide where the curves converge, so identity comes from the
    # legend; a proxy entry explains the hollow markers.
    if any(r.get("preemptions") for r in rows):
        ax.scatter([], [], s=56, facecolor=SURFACE, edgecolor=TEXT_SECONDARY, linewidth=2,
                   label="KV cache full (requests preempted)")

    # Concurrency labels on the no-spec curve only (selective labeling).
    for r in rows:
        if r["config"] == configs[0]:
            ax.annotate(f"c={r['concurrency']}", (r["tok_s_user"], r["tok_s_gpu"]),
                        xytext=(0, -12), textcoords="offset points", fontsize=7,
                        color=TEXT_SECONDARY, ha="center")

    ax.set_xlabel("Output tokens/s per user (higher = lower latency)", color=TEXT_PRIMARY)
    ax.set_ylabel("Output tokens/s per GPU (throughput)", color=TEXT_PRIMARY)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


def main() -> None:
    run = sys.argv[1]
    configs = sys.argv[2:] or [p.name for p in sorted((ROOT / run).iterdir()) if p.is_dir()]
    configs = sorted(configs, key=lambda c: list(COLORS).index(c) if c in COLORS else 99)
    rows = load(run, configs)
    for r in rows:
        sc = spec_counters(ROOT / run / r["config"] / f"c{r['concurrency']}" / "server_metrics_export.json")
        drafts = sc.get("vllm:spec_decode_num_drafts")
        acc = sc.get("vllm:spec_decode_num_accepted_tokens")
        drafted = sc.get("vllm:spec_decode_num_draft_tokens")
        r["acceptance_rate"] = round(acc / drafted, 3) if acc and drafted else None
        r["tokens_per_step"] = round(1 + acc / drafts, 2) if acc and drafts else None
        per_pos = sc.get("vllm:spec_decode_num_accepted_tokens_per_pos")
        r["accept_by_pos"] = (" ".join(f"{a / drafts:.2f}" for a in per_pos)
                              if per_pos and drafts else None)

    out_csv = ROOT / run / "summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")
    for r in rows:
        print(f"{r['config']:10s} c={r['concurrency']:3d}  user={r['tok_s_user']:7.1f}  "
              f"gpu={r['tok_s_gpu']:8.1f}  itl_p50={r['itl_ms_p50']:6.1f}ms  "
              f"accept={r['acceptance_rate']}  tok/step={r['tokens_per_step']}  "
              f"by_pos=[{r['accept_by_pos']}]  preempt={r.get('preemptions')}  "
              f"running={r.get('running_avg')}  err={r['errors']}")
    # Dataset name from the run's meta.txt ("dataset X | concurrencies ...").
    meta = ROOT / run / "meta.txt"
    dataset, extra = "spec-bench", ""
    if meta.exists():
        for line in meta.read_text().splitlines():
            if line.startswith("dataset "):
                dataset = line.split()[1]
            if line.startswith("max_num_seqs ") and "temperature 0" not in line.replace("0.", "x."):
                extra = f", temperature {line.split('temperature ')[1]}"
    cs = sorted({r["concurrency"] for r in rows})
    plot(rows, configs, ROOT / run / "pareto.png",
         f"{run}: Qwen3-4B + EAGLE-3, {dataset}{extra}, concurrency {cs[0]}→{cs[-1]}")


if __name__ == "__main__":
    main()
