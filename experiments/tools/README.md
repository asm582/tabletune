# Diagnostics

One-off tools used to explain a measurement. Not needed to tune or serve.

| Tool | What it answers | Results |
|---|---|---|
| `nsys_spike.sh` + `nsys_compare.py` | Why K=3 costs more at 3–5 requests on the T4 (per-kernel nsys summaries) | [`experiments/results/nsys`](../results/nsys/README.md) |
| `gemm_microbench.py` | Qwen3-4B's matrix multiplies vs. tokens per step, outside vLLM (the kernel switch above 16 tokens) | [`experiments/results/micro`](../results/micro) |
| `attn_microbench.py` | vLLM's Triton attention vs. query tokens per request | [`experiments/results/micro`](../results/micro) |
| `copy_diagnostic.py` | How much of each answer copies the prompt (whether n-gram can work) | [`experiments/results/ngram`](../results/ngram/README.md) |
| `nsys_gpu_idle.py` | GPU busy vs. idle time and launch counts in a trace window (issue #12) | `experiments/results/dsd-overhead` |

Each file's header has its exact command.
