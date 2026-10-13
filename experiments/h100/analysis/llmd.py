"""llm-d optimized-baseline + TableTune (Q4): per-config throughput through the router, mean over runs.
Usage: python llmd.py [qwen3-32b | qwen3-8b]   (data: experiments/results/h100-studies/llmd/<model>.csv, llmd_csv.py)"""
import collections, csv, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import STUDIES  # noqa: E402

CFGS = ["guide", "nospec", "k3", "docs", "tuned", "tuned13", "tuned13-ctrl", "docs-ctrl"]
NAME = {"guide": "guide as shipped (v0.30.0, no spec)", "nospec": "no spec (nightly)", "k3": "fixed length 3",
        "docs": "docs table", "tuned": "TableTune table", "tuned13": "TableTune table (len 0/1/3)",
        "tuned13-ctrl": "TableTune(0/1/3) + controller", "docs-ctrl": "docs table + controller"}


def load(model):
    data = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(STUDIES, "llmd", f"{model}.csv"))):
        data[(r["config"], int(r["conc"]))].append({k: float(v) for k, v in r.items() if k not in ("config", "run") and v != ""})
    return data


def main(model):
    data = load(model)
    C = sorted({c for _, c in data})
    cfgs = [c for c in CFGS if any((c, x) in data for x in C)]
    m = lambda cfg, c, k="tok_s": st.mean(x[k] for x in data[(cfg, c)]) if (cfg, c) in data else float("nan")
    print(f"{model}: output tokens/s, all replicas, through the llm-d router (mean of runs; n = runs)")
    print("open req " + "".join(f"{x:>12}" for x in cfgs))
    for c in C:
        print(f"{c:8d} " + "".join(f"{m(x, c):12.0f}" for x in cfgs))
    print("runs     " + "".join(f"{len(data.get((x, C[0]), [])):12d}" for x in cfgs))
    if "guide" in cfgs:
        print("\nGain over the guide as shipped (%)")
        for x in cfgs:
            if x != "guide":
                print(f"{NAME[x]:34s}" + " ".join(f"{100 * (m(x, c) / m('guide', c) - 1):+6.0f}" for c in C))
    for t in ["tuned13", "tuned13-ctrl"]:
        for ref in ["k3", "docs", "nospec"]:
            if ref in cfgs and t in cfgs:
                print(f"{NAME[t]} vs {NAME[ref]} (%): " + " ".join(f"{c}:{100 * (m(t, c) / m(ref, c) - 1):+.1f}" for c in C))
    if "docs-ctrl" in cfgs and "docs" in cfgs:
        print("controller from docs table vs docs table (%): " + " ".join(f"{c}:{100 * (m('docs-ctrl', c) / m('docs', c) - 1):+.1f}" for c in C))
    print("\nMedian TTFT / ITL (ms) at each load:")
    for x in cfgs:
        print(f"{NAME[x]:34s}" + " ".join(f"{m(x, c, 'ttft_p50'):6.0f}/{m(x, c, 'itl_p50'):4.1f}" for c in C))
    print("\nMean draft length per draft (from vLLM counters):")
    for x in cfgs:
        if any("draft_len" in v for v in data.get((x, C[0]), [])):
            print(f"{NAME[x]:34s}" + " ".join(f"{m(x, c, 'draft_len'):5.2f}" for c in C))
    s = [100 * (max(v["tok_s"] for v in d) - min(v["tok_s"] for v in d)) / st.mean(v["tok_s"] for v in d) for d in data.values() if len(d) > 1]
    print("\nspread (max-min)/mean %, median and max over (config, load) with >= 2 runs:", f"{st.median(s):.1f} / {max(s):.1f}" if s else "n/a")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen3-32b")
