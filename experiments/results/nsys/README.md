# nsys: Why K=3 Is Expensive at B=3–5 on the T4

**Question.** The v2 cost profile shows the third draft token adding 12–18% to a decode step at batch sizes B=3–5, but only 3–5% at B=2 and B≥6. The benchmark confirms it: at c=4, K=3 gets about 107 tok/s in four runs vs. about 124 for K=2. This spike is why the best K on the T4 goes up and down with batch size (see [`../sweeps/README.md`](../sweeps/README.md)). What causes it?

**Method.** [`experiments/tools/nsys_spike.sh`](../../tools/nsys_spike.sh) runs the tuner's cost profiler (Dynamic SD path, synthetic zero acceptance, 320-token prompts) under Nsight Systems 2025.6.1 for five cases. Only the timed loop is captured, with CUDA graphs traced per kernel. [`experiments/tools/nsys_compare.py`](../../tools/nsys_compare.py) compares the per-kernel summaries. Step times under nsys match the untraced profile within 1%, so tracing didn't change the behavior.

## Result

Tokens per step = B × (K+1): the positions the main model verifies each step.

| Case | Tokens/step | Step time | Main-model GEMM kernel | GEMM calls | Attention median |
|---|---|---|---|---|---|
| B=4, K=2 | 12 | 48.5 ms | 16×16 WMMA (`cutlass_75_wmma…16x16`) | 29,994 | 212 µs |
| B=3, K=3 | 12 | 53.2 ms | 16×16 WMMA | 30,984 | 282 µs |
| B=4, K=3 | 16 | 55.9 ms | 16×16 WMMA | 30,984 | 349 µs |
| B=5, K=3 | 20 | 63.3 ms | **64-row-tile cuBLAS** (`turing_fp16_s1688gemm…256x64`, `128x64`, `64x64`) | 2,184 WMMA + ~29k others | 361 µs |
| B=6, K=3 | 24 | 63.1 ms | 64-row-tile cuBLAS (same set) | same as B=5 | 361 µs |

Two causes, in two parts of the spike:

1. **B=5 (the largest jump): the GEMM kernel changes at 16 tokens.** Up to 16 tokens per step, the main model's matrix multiplies run on a 16×16 WMMA kernel. Above 16 they switch to cuBLAS kernels with 64-row tiles, most of which is padding at 17–24 tokens. The switch costs about 7–9 ms per step, then stays flat (B=5 and B=6 at K=3 cost the same, 63.3 vs. 63.1 ms). K=3 crosses 16 tokens one batch size earlier than K=2 (at B=5: 20 vs. 15 tokens), so for that one batch size K=3 pays the switch and K=2 doesn't. At B=6 both have switched (24 vs. 18 tokens), and K=3's extra cost falls back to 1.7 ms. The same threshold shows up for every K in the v2 profile: +8.6 ms (K=1, 16→18 tokens), +7.1 ms (K=2, 15→18), +8.1 ms (K=3, 16→20).
2. **B=3–4: attention costs more with 4 query tokens per request.** GEMM time is identical here (one 16-row tile covers 12 and 16 tokens). The difference is Triton's `kernel_unified_attention`. At the same 12 tokens per step, 3 requests × 4 query tokens (K=3) take 282 µs per call vs. 212 µs for 4 requests × 3 query tokens (K=2), about 33% more. For B=4, K=3 vs. K=2 (same prefill work, so a clean comparison), attention accounts for about 70% of the extra step time; the drafter's extra GEMMs make up the rest. The attention microbenchmark below reproduces this in isolation (+34% at B=3, +45% at B=4, none at B=5–6).

## What this means

- **The spike is a property of this GPU's kernels, not noise.** cuBLAS kernel selection and the Triton attention backend on Turing (T4, sm_75) set where the steps in the cost curve fall. A GPU with FlashAttention or FlashInfer and different GEMM heuristics will have its steps elsewhere, or none at all.
- **This is why the tuner has to measure costs rather than assume a smooth or monotone shape.** v1's monotone rule smoothed over exactly this effect and lost 2–3% (see the corrections log in [`../sweeps/README.md`](../sweeps/README.md)).
- **It also suggests a cheap engine-level improvement:** keeping B×(K+1) at or below 16 tokens (for example, choosing K=2 instead of K=3 at B=5) sidesteps the GEMM switch. The v2 table already does this for B=3–5.

## Verification with microbenchmarks

The nsys result left two open claims, so each part was measured in isolation on the same T4 (no vLLM server, no drafter, no prefill). Data: [`experiments/results/micro/`](../micro).

### GEMM: [`experiments/tools/gemm_microbench.py`](../../tools/gemm_microbench.py)

`F.linear` in FP16 for Qwen3-4B's four weight shapes (qkv, o, gate_up, down) at every M = 1…128 tokens, with the kernel name recorded:

| Tokens (M) | 1 | 2–16 | **17** | 17–64 | **65** | 128 |
|---|---|---|---|---|---|---|
| All 4 GEMMs × 36 layers, per forward pass | 33.4 ms | 35.7–37.2 ms | **42.7 ms** | 42.7–44.5 ms | **45.5 ms** | 47.8 ms |
| Kernel (qkv_proj, gate_up_proj) | gemv | 16×16 WMMA | 64-row-tile cuBLAS | same family | 256×128 tile (gate_up) | |

- **Confirmed:** cuBLAS switches from the 16×16 WMMA kernel to 64-row-tile kernels at exactly **M=17**, adding **+5.5 ms per forward pass (+15%)**. Cost then stays flat up to 64.
- **The step at 64 exists but is small:** another kernel switch at M=65 adds only +1.0 ms, which is why the cost profile showed just 1.5–2.2 ms there. This answers the earlier "missing 64-token step".
- GEMM time is flat from 40 to 44 tokens, so the K=3 jump at 40→44 tokens is **not** a GEMM effect (see attention below).
- Caveat: for o_proj and down_proj the recorded "first kernel" is sometimes a `Memset` or reduction helper rather than the GEMM itself, so their kernel names are unreliable; their timings are fine.

### Attention: [`experiments/tools/attn_microbench.py`](../../tools/attn_microbench.py)

vLLM's Triton `unified_attention` called directly with synthetic tensors, 352 tokens of context, median of 200 calls (µs):

| Requests (B) | q=1 | q=2 | q=3 (K=2) | q=4 (K=3) | q=5 |
|---|---|---|---|---|---|
| 2 | 183 | 244 | 243 | 245 | 245 |
| 3 | 248 | 248 | **248** | **332** | 330 |
| 4 | 249 | 268 | **268** | **390** | 390 |
| 5 | 332 | 330 | 391 | 392 | 392 |
| 6 | 338 | 333 | 395 | 393 | 539 |
| 8–10 | ~394 | ~393 | ~394 | ~395 | 620–680 |
| **11** | **491** | **529** | **487** | **488** | 819 |
| 12–14 | ~537 | ~537 | ~538 | ~538 | 824–967 |
| 16 | 629 | 644 | 651 | 675 | 1057 |

- **Confirmed in isolation:** with 4 query tokens per request (K=3), attention costs **+34% at B=3 and +45% at B=4** vs. 3 tokens (K=2), and the same at B=5–6. That matches the spike band exactly, so the nsys observation wasn't caused by drafter calls mixed into the medians.
- **The tiling hypothesis is refuted.** With 32 query heads / 8 KV heads, the kernel uses `BLOCK_M=16`, `BLOCK_Q=4` ([source](https://github.com/vllm-project/vllm/blob/353e491efd3a23057004a3ab68fcfbda4abbee13/vllm/v1/attention/ops/triton_unified_attention.py#L934-L971)), so 3 and 4 query tokens both fit in one block per request and tiling predicts equal cost. The measured cost is instead a **staircase over the number of requests**, with extra steps at small B for longer queries.
- **This explains the smaller jumps:** attention steps up at **B=10→11 for every query length** (~394 → ~490–530 µs, about +3.3–4.9 ms over 36 layers). The cost profile shows a jump at exactly B=10→11 for K=1, 2 and 3 (+5.0, +5.8, +5.5 ms), including the K=3 "40→44 tokens" jump.

## Still open

- **Why the attention kernel steps where it does.** The steps don't follow its query tiling. A plausible cause is how many kernel programs are launched relative to the T4's 40 SMs (occupancy and waves), but the step positions only partly fit that. Confirming it needs Nsight Compute (achieved occupancy and waves per launch). Not done.
- **K=0's jump at B=8→9** (+3.7 ms). Decode with one query token can take a different (split-KV, "3D") attention path that this microbenchmark doesn't exercise.
- **Kernel totals in the nsys summaries include the prefill passes** in the timed loop, so cross-B totals mix prefill and decode. Medians, the same-B comparison (B=4, K=2 vs. K=3) and the microbenchmarks are the clean numbers.
- Raw traces (`*.nsys-rep`, about 5 MB each) are not committed. Regenerate them with `bash experiments/tools/nsys_spike.sh` on the instance.

## Files

- `*_cuda_gpu_kern_sum.csv`: per-kernel summary for each case (`nsys stats --report cuda_gpu_kern_sum`)
- `*.json`: step time for each case
- [`../micro/gemm.csv`](../micro/gemm.csv), [`../micro/attention.csv`](../micro/attention.csv): microbenchmark data
- Compare: `cd experiments/results/nsys && python ../../tools/nsys_compare.py 16`
