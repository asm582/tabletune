# Microbenchmarks: GEMM and attention cost (T4)

Timings of Qwen3-4B's matrix multiplies and of vLLM's Triton attention kernel, measured outside vLLM on the Tesla T4 (FP16) on 2026-09-25. They check the two causes nsys pointed to for the step-cost jumps.

| File | What | Made by |
|---|---|---|
| `gemm.csv` | Time and cuBLAS kernel of each GEMM vs. tokens per step (M = 1–128) | [`experiments/tools/gemm_microbench.py`](../../tools/gemm_microbench.py) |
| `attention.csv` | Attention time vs. requests and query tokens per request, context 352 | [`experiments/tools/attn_microbench.py`](../../tools/attn_microbench.py) |

- **Write-up:** [`../nsys/README.md`](../nsys/README.md) (the GEMM kernel switch at 17 tokens, the attention steps).
- **Used by:** deck appendix A4 (`docs/deck/images/micro.png` is drawn from these two files; the script that drew it is not in the repository).
