"""Build the 2%-merge-tolerance tables from the committed H100 V2 profiles.
EAGLE-3 keeps the original table's lengths 2/3, so no step from length 2 to 1 (which stalls this vLLM nightly).
Usage: python build_mergetol.py [out.json]   (default: print; CI compares against mergetol_tables.json)"""
import json
import os
import sys

from dsd_tuning.core import load_acceptance, load_costs, make_schedule

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
SETTINGS = [("eagle", "tuner-eagle-h100v2", [2, 3], 128), ("ngram", "ngram-tuner-h100v2", None, 128),
            ("specbench", "tuner-v2-h100v2", None, 64), ("dflash", "dflash-tuner-h100v2", None, 128)]

out = {}
for name, d, ks, maxb in SETTINGS:
    acc, rate = load_acceptance(f"{R}/{d}/acceptance.json")
    out[name] = make_schedule(load_costs(f"{R}/{d}/costs_k*.json"), acc, maxb, allowed_ks=ks, proposal_rate=rate, merge_tol=0.02)
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"))
else:
    print(json.dumps(out))
