"""Check the formula's predictions against a fixed-K benchmark sweep.

For each concurrency, compares the measured throughput of every fixed K with
the goodput predicted at the sweep's live batch size, under two step-cost
models for drafters that propose on only some steps (p < 1): "mean" (step time
at p*K drafts) and "full" (step time at K, what the tuner uses). For p = 1 they
are the same. Reports the predicted vs. measured best K and the ratio errors.

  python dsd_tuning/tuner/validate_costmodel.py --sweep <sweep dir> \\
      --costs 'my-tune/costs_k*.json' --acceptance my-tune/acceptance.json --prefix ngram-k
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from dsd_tuning.core import interp, load_acceptance, load_costs, step_ms, tokens_per_step
except ImportError:  # run as a file from a clone that is not installed
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dsd_tuning.core import interp, load_acceptance, load_costs, step_ms, tokens_per_step


def step_ms_mean(costs, k: int, b: float, rate: float) -> float:
    """The rejected "mean" model: step time interpolated at p*K drafts per request."""
    if rate >= 1.0 or k == 0:
        return step_ms(costs, k, b)
    kbar = rate * k
    below = max((kk for kk in costs if kk <= kbar), default=min(costs))
    above = min((kk for kk in costs if kk >= kbar), default=max(costs))
    t0 = interp(*costs[below], b)
    if above == below:
        return t0
    return t0 + (interp(*costs[above], b) - t0) * (kbar - below) / (above - below)


def predict_step_ms(costs, k: int, b: float, rate: float, model: str) -> float:
    """model "full" is the tuner's (core.step_ms); "mean" is step_ms_mean."""
    return step_ms_mean(costs, k, b, rate) if model == "mean" else step_ms(costs, k, b)


def predict(costs, tps, k: int, b: float, rate: float, model: str) -> float:
    return b * tps[k] / (predict_step_ms(costs, k, b, rate, model) / 1000)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--acceptance", required=True)
    ap.add_argument("--prefix", default="ngram-k", help="fixed-K config names: <prefix><K>")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    costs = load_costs(a.costs)
    accept, rate = load_acceptance(a.acceptance)
    tps = tokens_per_step(accept, proposal_rate=rate)
    rows = list(csv.DictReader(open(Path(a.sweep) / "summary.csv")))
    meas: dict[int, dict[int, float]] = {}
    batch: dict[int, list[float]] = {}
    for r in rows:
        if not r["config"].startswith(a.prefix):
            continue
        k, c = int(r["config"][len(a.prefix):]), int(r["concurrency"])
        meas.setdefault(c, {})[k] = float(r["tok_s_gpu"])
        batch.setdefault(c, []).append(float(r.get("running_avg") or c))

    report = {"proposal_rate": rate, "rows": []}
    print(f"proposal rate {rate:.3f}; tokens per step {({k: round(v, 3) for k, v in tps.items()})}")
    print(f"{'c':>4} {'batch':>6} {'measured best':>14} {'mean: best':>11} {'full: best':>11}"
          f"   measured tok/s by K")
    err = {"mean": [], "full": []}
    for c in sorted(meas):
        m = meas[c]
        b = sum(batch[c]) / len(batch[c])
        best = max(m, key=m.get)
        line = {"c": c, "batch": round(b, 1), "measured": m, "measured_best": best}
        for model in ("mean", "full"):
            g = {k: predict(costs, tps, k, b, rate, model) for k in m}
            line[f"{model}_best"] = max(g, key=g.get)
            # Ratio of each K to the measured best: predicted vs measured.
            for k in m:
                err[model].append(abs(g[k] / g[best] - m[k] / m[best]))
            line[f"{model}_ratio"] = {k: round(g[k] / g[best], 3) for k in m}
        report["rows"].append(line)
        print(f"{c:4d} {b:6.1f} {best:14d} {line['mean_best']:11d} {line['full_best']:11d}   "
              + "  ".join(f"K={k}:{v:.0f}" for k, v in sorted(m.items())))
    for model in ("mean", "full"):
        e = err[model]
        report[f"{model}_mean_abs_ratio_error"] = sum(e) / len(e)
        hits = sum(r["measured_best"] == r[f"{model}_best"] for r in report["rows"])
        report[f"{model}_best_k_hits"] = f"{hits}/{len(report['rows'])}"
        print(f"model {model:4s}: best K right at {hits}/{len(report['rows'])} loads, "
              f"mean |ratio error| {100 * sum(e) / len(e):.1f} pp")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
