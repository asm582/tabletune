"""Mean and spread of repeated sweeps: <sweep> (run 1), <sweep>-r2, <sweep>-r3.

Writes experiments/results/repeats/<sweep>.csv and prints a markdown table of output
tok/s per GPU: mean, and min..max across runs, per config and concurrency.
Also reports the spread as a percentage of the mean (the run-to-run noise).

  python experiments/bench/aggregate_repeats.py code128t-t4 [--metric tok_s_gpu]
  MLFLOW_TRACKING_URI=http://localhost:5000 python experiments/bench/aggregate_repeats.py \
      code128t-t4 ngram-apply-t4 --mlflow      # attach CSV + table to run 'compare-repeats'
"""

import argparse
import csv
import statistics
from pathlib import Path

SWEEPS = Path(__file__).resolve().parent.parent / "results" / "sweeps"
OUT = Path(__file__).resolve().parent.parent / "results" / "repeats"
METRICS = ["tok_s_gpu", "tok_s_user", "ttft_ms_p50", "itl_ms_p50", "preemptions"]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps", nargs="+")
    ap.add_argument("--metric", default="tok_s_gpu")
    ap.add_argument("--mlflow", action="store_true",
                    help="attach each sweep's CSV and markdown table to the MLflow run 'compare-repeats'")
    a = ap.parse_args(argv)
    for sweep in a.sweeps:
        aggregate(sweep, a.metric, a.mlflow)


def aggregate(sweep: str, metric: str, to_mlflow: bool) -> dict:
    class A:  # keep the body below unchanged
        pass
    a = A()
    a.sweep, a.metric = sweep, metric
    runs = [SWEEPS / a.sweep] + sorted(SWEEPS.glob(f"{a.sweep}-r[0-9]"))
    runs = [r for r in runs if (r / "summary.csv").exists()]
    data: dict[tuple[str, int], dict[str, list[float]]] = {}
    for r in runs:
        for row in csv.DictReader(open(r / "summary.csv")):
            key = (row["config"], int(row["concurrency"]))
            for m in METRICS:
                if row.get(m) not in (None, ""):
                    data.setdefault(key, {}).setdefault(m, []).append(float(row[m]))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for (cfg, c), ms in sorted(data.items()):
        row = {"config": cfg, "concurrency": c, "runs": len(ms.get(a.metric, []))}
        for m, v in ms.items():
            row[f"{m}_mean"] = round(statistics.mean(v), 2)
            row[f"{m}_min"], row[f"{m}_max"] = round(min(v), 2), round(max(v), 2)
            row[f"{m}_std"] = round(statistics.stdev(v), 2) if len(v) > 1 else 0.0
        rows.append(row)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("config", "concurrency", "runs"), k))
    with open(OUT / f"{a.sweep}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{a.sweep}: {len(runs)} run(s) [{', '.join(r.name for r in runs)}], metric {a.metric}")
    print("| config | c | runs | mean | min..max | spread % of mean |")
    print("|---|---|---|---|---|---|")
    spreads = []
    for r in rows:
        mean, lo, hi = r[f"{a.metric}_mean"], r[f"{a.metric}_min"], r[f"{a.metric}_max"]
        sp = 100 * (hi - lo) / mean if mean else 0
        spreads.append(sp)
        print(f"| {r['config']} | {r['concurrency']} | {r['runs']} | {mean:.1f} | {lo:.1f}..{hi:.1f} | {sp:.1f} |")
    lines = [f"{a.sweep}: {len(runs)} run(s), metric {a.metric}",
             "| config | c | runs | mean | min..max | spread % of mean |", "|---|---|---|---|---|---|"]
    for r, sp in zip(rows, spreads):
        lines.append(f"| {r['config']} | {r['concurrency']} | {r['runs']} | {r[f'{a.metric}_mean']:.1f} | "
                     f"{r[f'{a.metric}_min']:.1f}..{r[f'{a.metric}_max']:.1f} | {sp:.1f} |")
    if spreads:
        summary = f"median spread {statistics.median(spreads):.1f}%, max {max(spreads):.1f}%"
        print(summary)
        lines.append(summary)
    (OUT / f"{a.sweep}.md").write_text("\n".join(lines) + "\n")
    if to_mlflow:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        exp = client.get_experiment_by_name("vllm-perf-dsd")
        found = client.search_runs([exp.experiment_id], "tags.group_key = 'compare-repeats'")
        run_id = found[0].info.run_id if found else client.create_run(
            exp.experiment_id, run_name="compare-repeats",
            tags={"group_key": "compare-repeats", "kind": "comparison"}).info.run_id
        for ext in ("csv", "md"):
            client.log_artifact(run_id, str(OUT / f"{a.sweep}.{ext}"))
        if spreads:
            client.log_metric(run_id, f"{a.sweep}.median_spread_pct", statistics.median(spreads))
            client.log_metric(run_id, f"{a.sweep}.max_spread_pct", max(spreads))
        print(f"[mlflow] {a.sweep}: attached to run 'compare-repeats'")
    return {"runs": [r.name for r in runs], "rows": rows}


if __name__ == "__main__":
    main()
