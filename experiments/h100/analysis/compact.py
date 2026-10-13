"""Copy the per-window segments of gen.py result files (jumping load, workload switch) into the repo.

gen.py writes every request into its JSON; the analysis needs only the per-window "segments". Usage:
  python compact.py <src dir> <dst dir>      # every <name>.json with segments; <name>.b.json etc. skipped
"""
import glob
import json
import os
import sys


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(src, "*.json"))):
        name = os.path.basename(f)
        if name.count(".") > 1:   # <name>.b.json: the second client of the same run, same windows
            continue
        d = json.load(open(f))
        if "segments" not in d:
            if name.startswith("sched"):
                json.dump(d, open(os.path.join(dst, name), "w"))
            continue
        json.dump({"segments": d["segments"], "errors": d["errors"] if isinstance(d.get("errors"), int) else len(d.get("errors", []))},
                  open(os.path.join(dst, name), "w"), indent=0)
        n += 1
    print(f"{n} files -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
