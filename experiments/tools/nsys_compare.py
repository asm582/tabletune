"""Compare per-kernel GPU time across the nsys spike cases (experiments/results/nsys).

Reads <case>_cuda_gpu_kern_sum.csv (written by experiments/tools/nsys_spike.sh) for each
case in the current folder, groups kernels by name, and prints total time and
call count per kernel side by side, so the kernel that changes between cases
stands out (the GEMM switch above 16 tokens per step, experiments/results/nsys/README.md).

  cd experiments/results/nsys && python ../../tools/nsys_compare.py [N]   (top N kernels, default 14)
"""
import csv, re, sys
cases = ["b4_k2", "b3_k3", "b4_k3", "b5_k3", "b6_k3"]
data = {}
for c in cases:
    try:
        rows = list(csv.DictReader(open(f"{c}_cuda_gpu_kern_sum.csv")))
    except FileNotFoundError:
        continue
    d = {}
    for r in rows:
        n = re.sub(r"\(.*$", "", r["Name"])
        n = n.replace("void ", "").replace("cutlass::Kernel2<", "")[:70]
        a = d.setdefault(n, [0.0, 0])
        a[0] += float(r["Total Time (ns)"]) / 1e6
        a[1] += int(r["Instances"])
    data[c] = d
print("total GPU kernel ms:", {c: round(sum(v[0] for v in d.values())) for c, d in data.items()})
names = sorted({n for d in data.values() for n in d},
               key=lambda n: -max(d.get(n, [0])[0] for d in data.values()))
print(f"{'kernel':70s} " + " ".join(f"{c:>15s}" for c in data))
for n in names[:int(sys.argv[1]) if len(sys.argv) > 1 else 14]:
    print(f"{n:70s} " + " ".join(f"{data[c].get(n,[0,0])[0]:7.0f}ms/{data[c].get(n,[0,0])[1]:6d}" for c in data))
