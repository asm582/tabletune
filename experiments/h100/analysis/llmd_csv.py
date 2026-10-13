"""Compact one llm-d results folder (driver.sh: <config>-r<run>/c<load>/profile_export_aiperf.json plus the vLLM metric
dumps c<load>.{pre,post}.<pod>.prom) into one CSV row per (config, run, load). Only finished runs (DONE).
Usage: python llmd_csv.py <results dir> <out.csv>"""
import csv
import glob
import json
import os
import re
import sys

COLS = ["config", "run", "conc", "tok_s", "ttft_p50", "itl_p50", "draft_len", "acc_per_draft", "preempt"]


def prom(path):
    d = {}
    for line in open(path):
        m = re.match(r'^(vllm:[a-z_]+)(\{[^}]*\})? ([0-9.e+]+)', line)
        if m:
            d[m.group(1)] = d.get(m.group(1), 0) + float(m.group(3))
    return d


def level(run_dir, c):
    j = f"{run_dir}/c{c}/profile_export_aiperf.json"
    if not os.path.exists(j):
        return None
    a = json.load(open(j))
    out = {"tok_s": a["output_token_throughput"]["avg"], "ttft_p50": a["time_to_first_token"]["p50"],
           "itl_p50": a["inter_token_latency"]["p50"]}
    pre, post = {}, {}
    for side, acc in (("pre", pre), ("post", post)):   # summed over the replicas
        for f in glob.glob(f"{run_dir}/c{c}.{side}.*.prom"):
            for k, v in prom(f).items():
                acc[k] = acc.get(k, 0) + v
    dd = lambda k: post.get(k, 0) - pre.get(k, 0)
    if dd("vllm:spec_decode_num_drafts_total") > 0:
        out["draft_len"] = dd("vllm:spec_decode_num_draft_tokens_total") / dd("vllm:spec_decode_num_drafts_total")
        out["acc_per_draft"] = dd("vllm:spec_decode_num_accepted_tokens_total") / dd("vllm:spec_decode_num_drafts_total")
    out["preempt"] = dd("vllm:num_preemptions_total")
    return out


def main(R, out):
    rows = []
    for rd in sorted(glob.glob(f"{R}/*-r[0-9]")):
        if not os.path.exists(f"{rd}/DONE"):
            continue
        cfg, run = os.path.basename(rd).rsplit("-r", 1)
        loads = sorted(int(os.path.basename(p)[1:]) for p in glob.glob(f"{rd}/c*") if re.fullmatch(r"c\d+", os.path.basename(p)))
        for c in loads:
            lv = level(rd, c)
            if lv:
                rows.append({"config": cfg, "run": int(run), "conc": c, **{k: round(v, 4) for k, v in lv.items()}})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
