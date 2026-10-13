"""Collect every finished AIPerf point under a study results folder into one CSV (run where the results are).

  python collect.py <results dir> <out.csv>     e.g. $WORK/exp-h100/results

Columns: study, run, config, point (c<N> for closed loop, <pattern>-r<rate> for open loop), output tok/s,
request/s, requests, TTFT p50/p99, ITL p50, request latency p50/p99 (ms).
"""
import csv
import glob
import json
import os
import sys

ROOT = sys.argv[1]
OUT = sys.argv[2]


def get(d, key, stat):
    v = d.get(key)
    return round(v[stat], 3) if isinstance(v, dict) and v.get(stat) is not None else ""


rows = []
for f in sorted(glob.glob(f"{ROOT}/*/r*/*/*/profile_export_aiperf.json")):
    point_dir = os.path.dirname(f)
    cfg_dir = os.path.dirname(point_dir)
    run_dir = os.path.dirname(cfg_dir)
    study_dir = os.path.dirname(run_dir)
    try:
        d = json.load(open(f))
    except (OSError, ValueError):
        continue
    rows.append({
        "study": os.path.basename(study_dir), "run": os.path.basename(run_dir), "config": os.path.basename(cfg_dir),
        "point": os.path.basename(point_dir), "done": int(os.path.exists(f"{cfg_dir}/DONE")),
        "tok_s": get(d, "output_token_throughput", "avg"), "req_s": get(d, "request_throughput", "avg"),
        "requests": get(d, "request_count", "avg"),
        "ttft_p50": get(d, "time_to_first_token", "p50"), "ttft_p99": get(d, "time_to_first_token", "p99"),
        "itl_p50": get(d, "inter_token_latency", "p50"),
        "lat_p50": get(d, "request_latency", "p50"), "lat_p99": get(d, "request_latency", "p99"),
    })
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["study"])
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} points -> {OUT}")
