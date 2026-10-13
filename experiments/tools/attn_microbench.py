"""Attention microbenchmark: vLLM's Triton unified attention vs. query length.

nsys suggested this kernel costs more with 4 query tokens per request (K=3)
than with 3 (K=2), but the kernel source puts both in one BLOCK_Q=4 block
(Qwen3-4B: 32 query heads / 8 KV heads -> BLOCK_M=16, BLOCK_Q=4), so tiling
alone predicts equal cost. This calls the kernel directly, with nothing else
in the measurement (no drafter, no prefill), for a grid of
(requests, query tokens per request) at a fixed context length.

Usage: python experiments/tools/attn_microbench.py [--context 352]
"""

import argparse
import csv
import statistics
from pathlib import Path

import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention

HEADS, KV_HEADS, HEAD_DIM, BLOCK_SIZE = 32, 8, 128, 16
ITERS = 200
SEQS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 16)


def bench(num_seqs: int, q_len: int, context: int) -> float:
    tokens = num_seqs * q_len
    blocks_per_seq = -(-context // BLOCK_SIZE)
    num_blocks = num_seqs * blocks_per_seq
    q = torch.randn(tokens, HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
    k = torch.randn(num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    cu_q = torch.arange(0, tokens + 1, q_len, dtype=torch.int32, device="cuda")
    seqused_k = torch.full((num_seqs,), context, dtype=torch.int32, device="cuda")
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(
        num_seqs, blocks_per_seq)

    def call():
        unified_attention(q=q, k=k, v=v, out=out, cu_seqlens_q=cu_q, max_seqlen_q=q_len,
                          seqused_k=seqused_k, max_seqlen_k=context,
                          softmax_scale=HEAD_DIM ** -0.5, causal=True, window_size=(-1, -1),
                          block_table=block_table, softcap=0, q_descale=None,
                          k_descale=None, v_descale=None)

    for _ in range(10):
        call()
    samples = []
    for _ in range(ITERS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return statistics.median(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=352)  # 320 prompt + ~32 generated
    a = ap.parse_args()

    rows = []
    for num_seqs in SEQS:
        for q_len in (1, 2, 3, 4, 5):
            t = bench(num_seqs, q_len, a.context)
            rows.append({"num_seqs": num_seqs, "q_len": q_len, "tokens": num_seqs * q_len,
                         "context": a.context, "time_us": round(t, 1)})

    out = Path(__file__).resolve().parent.parent / "results" / "micro" / "attention.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[attn] wrote {out}  (context {a.context} tokens, median of {ITERS} calls, us)")
    print("       " + " ".join(f"q_len={q:<3d}" for q in (1, 2, 3, 4, 5)))
    for n in SEQS:
        print(f"  B={n:2d} " + " ".join(f"{r['time_us']:9.1f}" for r in rows if r["num_seqs"] == n))


if __name__ == "__main__":
    main()
