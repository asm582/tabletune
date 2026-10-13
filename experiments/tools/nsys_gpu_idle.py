"""GPU busy vs. idle time and CUDA launch counts in an nsys trace (issue #12).

Reads the SQLite export of a trace (nsys export --type sqlite, or the .sqlite
that `nsys stats` writes next to the .nsys-rep) and prints: GPU busy % over the
window (union of kernel, memcpy and memset intervals), idle gaps longer than
50 us, the number of kernel and CUDA-graph launches, and the CUDA calls where
the CPU spent the most time.

Reading: if the CPU is ahead of the GPU, GPU busy % is high and the CPU spends
time waiting in cudaEventSynchronize. If the CPU is the bottleneck (launch-
bound), GPU busy % drops, idle gaps add up, and no waiting call shows up.

Usage: python experiments/tools/nsys_gpu_idle.py trace.sqlite [trace.sqlite ...]
"""
import sqlite3
import sys


def summarize(db: str) -> None:
    c = sqlite3.connect(db)
    iv = []
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        try:
            iv += c.execute(f"select start, end from {table}").fetchall()
        except sqlite3.OperationalError:     # table absent: no activity of that kind
            pass
    if not iv:
        print(f"{db}: no GPU activity (was the load running during the capture?)")
        return
    iv.sort()
    busy, gaps = 0, []
    s0, e0 = iv[0]
    for s, e in iv[1:]:
        if s > e0:
            busy += e0 - s0
            gaps.append(s - e0)
            s0, e0 = s, e
        else:
            e0 = max(e0, e)
    busy += e0 - s0
    window = iv[-1][1] - iv[0][0]
    api = c.execute("""select s.value, count(*), sum(r.end - r.start)
                       from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId
                       group by s.value order by 3 desc""").fetchall()
    launches = {name: calls for name, calls, _ in api if "Launch" in name}
    top = [(name, calls, round(t / 1e6, 1)) for name, calls, t in api[:4]]
    big = [g for g in gaps if g > 50_000]
    print(f"{db}: window {window / 1e6:.0f} ms, GPU busy {100 * busy / window:.1f}%, "
          f"idle gaps > 50 us: {len(big)} totalling {sum(big) / 1e6:.0f} ms; "
          f"launches {launches}; most CPU time in CUDA calls (name, calls, ms): {top}")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        summarize(path)
