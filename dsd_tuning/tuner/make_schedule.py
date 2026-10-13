"""Build a table from cost and acceptance files (the formula is in dsd_tuning/core.py).

Also compares the table with the measured best K of a fixed-K sweep (--oracle)
and prints what each fixed K would lose (--compare-fixed-k).

  python dsd_tuning/tuner/make_schedule.py --costs 'my-tune/costs_k*.json' \\
      --acceptance my-tune/acceptance.json --out my-tune [--oracle <sweep>/summary.csv]
"""

import argparse
import csv
import json
import re
from pathlib import Path

try:
    from dsd_tuning.core import (EPS, MIN_WIDTH_FRAC, build_schedule, compress,
                                fixed_k_losses, goodput_table, load_costs, load_tokens_per_step)
except ImportError:  # run as a file from a clone that is not installed
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dsd_tuning.core import (EPS, MIN_WIDTH_FRAC, build_schedule, compress,
                                fixed_k_losses, goodput_table, load_costs, load_tokens_per_step)

MAX_NUM_SEQS = 128


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--costs", required=True)
    ap.add_argument("--acceptance", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--oracle")
    ap.add_argument("--max-num-seqs", type=int, default=MAX_NUM_SEQS)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--min-width-frac", type=float, default=MIN_WIDTH_FRAC)
    ap.add_argument("--merge-tol", type=float, default=None,
                    help="merge a narrow range only if that loses at most this fraction of its predicted goodput "
                         "(e.g. 0.02); default: always merge")
    ap.add_argument("--compare-fixed-k", action="store_true",
                    help="print each fixed K's predicted worst/average loss vs. the table")
    a = ap.parse_args(argv)
    max_b = a.max_num_seqs

    costs = load_costs(a.costs)
    al, rate = load_tokens_per_step(a.acceptance)
    if rate < 1.0:
        print(f"[tuner] proposal rate {rate:.4f}: tokens per step scale with p, step cost is taken at the full K")
    goodput = goodput_table(costs, al, max_b)
    ks = sorted(next(iter(goodput.values())))
    raw, best = build_schedule(goodput, a.eps, a.min_width_frac, a.merge_tol)

    raw_schedule, schedule = compress(raw), compress(best)
    table = [{"batch_size": b, **{f"goodput_k{k}": round(goodput[b][k], 1) for k in ks},
              "k_raw": raw[b], "k_best": best[b]} for b in goodput]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "schedule.json").write_text(json.dumps(schedule))
    (out / "schedule_raw.json").write_text(json.dumps(raw_schedule))
    with (out / "goodput.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    print(f"[tuner] tokens/step by K: {al}")
    print(f"[tuner] raw argmax schedule: {raw_schedule}")
    print(f"[tuner] schedule: {schedule}")
    print(f"[tuner] wrote {out / 'schedule.json'} and {out / 'goodput.csv'}")

    if a.compare_fixed_k:
        print("\n[tuner] predicted loss of each fixed K vs. the best K at each batch size:")
        for k, v in fixed_k_losses(goodput).items():
            print(f"  K={k}: worst {v['worst_loss']:5.1%} at B={v['worst_b']:3d}, "
                  f"average {v['avg_loss']:5.1%}")

    if a.oracle:
        # Empirical best K per client concurrency from the fixed-K sweep.
        rows = list(csv.DictReader(open(a.oracle)))
        by_c: dict[int, dict[int, float]] = {}
        for r in rows:
            m = re.fullmatch(r"(?:ngram-)?k(\d+)", r["config"])  # k1..k3, ngram-k1..
            k = int(m.group(1)) if m else None
            if k is not None and k in ks:
                by_c.setdefault(int(r["concurrency"]), {})[k] = float(r["tok_s_gpu"])
        print("\n[tuner] predicted vs measured (fixed-K sweep):")
        print(f"{'c':>4} {'predicted K':>12} {'oracle K':>9} {'tok/s at predicted':>19} "
              f"{'tok/s at oracle':>16} {'loss':>6}")
        for c in sorted(by_c):
            meas = by_c[c]
            ok = max(meas, key=meas.get)
            pk = best[min(c, max_b)]
            if pk not in meas:
                continue
            loss = 1 - meas[pk] / meas[ok]
            print(f"{c:4d} {pk:12d} {ok:9d} {meas[pk]:19.1f} {meas[ok]:16.1f} {loss:6.1%}")


if __name__ == "__main__":
    main()
