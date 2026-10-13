"""GEMM microbenchmark: Qwen3-4B's linear layers vs. number of tokens (M).

Checks the nsys finding that cuBLAS switches from a 16x16 WMMA kernel to
64-row-tile kernels above 16 tokens, and looks for further switches (e.g. at
64). For every M in 1..MAX_M, times F.linear (what vLLM's unquantized linear
layers call) for each of the model's four weight shapes and records which CUDA
kernel ran. Writes experiments/results/micro/gemm.csv and prints where the kernel changes.

Usage: python experiments/tools/gemm_microbench.py [--max-m 128]
"""

import argparse
import csv
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

# Qwen3-4B: hidden 2560, 32 query heads + 8 KV heads of 128, MLP 9728.
SHAPES = {                     # name: (in_features, out_features)
    "qkv_proj": (2560, (32 + 2 * 8) * 128),
    "o_proj": (32 * 128, 2560),
    "gate_up_proj": (2560, 2 * 9728),
    "down_proj": (9728, 2560),
}
LAYERS = 36
ITERS = 50


def time_us(fn) -> float:
    for _ in range(5):
        fn()
    samples = []
    for _ in range(ITERS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return statistics.median(samples)


def kernel_name(fn) -> str:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    return names[0].split("(")[0].replace("void ", "")[:70] if names else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-m", type=int, default=128)
    a = ap.parse_args()

    torch.manual_seed(0)
    weights = {n: torch.randn(o, i, dtype=torch.float16, device="cuda") * 0.02
               for n, (i, o) in SHAPES.items()}
    rows = []
    for m in range(1, a.max_m + 1):
        per_layer_us = 0.0
        for n, (i, _) in SHAPES.items():
            x = torch.randn(m, i, dtype=torch.float16, device="cuda")
            fn = lambda: F.linear(x, weights[n])  # noqa: E731
            t = time_us(fn)
            per_layer_us += t
            rows.append({"m": m, "shape": n, "time_us": round(t, 1), "kernel": kernel_name(fn)})
        rows.append({"m": m, "shape": "ALL x36 layers", "time_us": round(per_layer_us * LAYERS, 1),
                     "kernel": ""})

    out = Path(__file__).resolve().parent.parent / "results" / "micro" / "gemm.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[gemm] wrote {out}")

    print("\n[gemm] kernel changes per shape (M = tokens):")
    for n in SHAPES:
        prev = None
        for r in (r for r in rows if r["shape"] == n):
            if r["kernel"] != prev:
                print(f"  {n:13s} M={r['m']:3d}: {r['kernel']}  ({r['time_us']} us)")
                prev = r["kernel"]
    print("\n[gemm] all 4 GEMMs x 36 layers, ms per forward pass:")
    tot = {r["m"]: r["time_us"] / 1000 for r in rows if r["shape"].startswith("ALL")}
    for m in [1, 2, 4, 8, 12, 15, 16, 17, 18, 20, 24, 32, 40, 44, 48, 56, 60, 64, 65, 72,
              80, 96, 112, 128]:
        if m in tot:
            print(f"  M={m:3d}: {tot[m]:6.2f} ms")


if __name__ == "__main__":
    main()
