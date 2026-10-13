# EAGLE-3 tuner and controller on the A10G

A10G (sm_86, BF16, FlashAttention), Qwen3-4B + `AngelSlim/Qwen3-4B_eagle3`, code workload (instruct-coder, ~32-token prompts, T=1.0), `max_num_seqs=128`. Output tok/s per GPU, **mean of 3 runs** (sweeps `code128t-a10g`, `-r2`, `-r3`; MLflow `compare-repeats`; spread 1.2% median, 6.5% max).

**Tuned table:** `[[1,16,3],[17,60,2],[61,91,1],[92,128,0]]` · acceptance by position 0.61 / 0.34 / 0.16 (same as the T4).

| c | K=1 | K=2 | K=3 | Tuned | No spec | T4's table | Tuned vs. best fixed |
|---|---|---|---|---|---|---|---|
| 1 | 77.0 | 87.7 | 88.6 | 90.7 | 52.2 | 86.8 | +2.3% |
| 2 | 146.9 | 167.8 | 172.9 | 171.3 | 99.4 | 173.4 | −0.9% |
| 4 | 271.1 | 306.4 | 302.4 | 299.4 | 181.4 | 297.3 | −2.3% |
| 8 | 511.8 | 570.2 | 583.7 | 572.1 | 369.6 | 574.6 | −2.0% |
| 16 | 916.4 | 1045.1 | 1036.6 | 1041.5 | 689.8 | 1036.7 | −0.3% |
| 32 | 1590.1 | 1581.9 | 1572.0 | 1618.0 | 1278.1 | 1608.7 | +1.8% |
| 64 | 2240.2 | 2149.9 | 1943.6 | 2226.2 | 2243.6 | 2230.5 | −0.6% |
| 96 | 2406.8 | 2002.0 | 1921.8 | 2532.2 | 2713.9 | 2405.7 | **+5.2%** |
| 128 | 2522.6 | 2230.1 | 2010.4 | 2888.2 | 3058.2 | 2524.1 | **+14.5%** |

- **The crossover exists on this GPU:** from ~92 requests K=0 is best, so the table switches to K=0 and beats every fixed K at 96–128. The T4 never reached it (its plain step is heavier; deck slide 22).
- **Tables are per GPU:** the T4's table keeps K=1 at 128 requests and loses 12.6%.
- **Each fixed K loses 15–20% somewhere** (K=1 −15.0% at 2, K=2 −16.8% at 96, K=3 −20.3% at 128).
- **Limit:** no speculation still beats the tuned table by 7.2% / 5.9% at 96 / 128: K=0 keeps the drafter's pass and the Dynamic SD CUDA-graph mode (issues #1, #12).

**Controller** (hold 8, re-fit every 200 steps, this GPU's costs, spec-bench prior; OFF → ON, 3 runs): docs example table +15.5% at 64 requests (1942 → 2242); docs table ÷4 +13.8% at 40 (1511 → 1719) but −4.9% at 96 (2549 → 2424), where OFF already runs K=0.

**KV cache:** 81.4k tokens without speculation, 71.4k with EAGLE-3; no preemption at 128 requests (peak use 43%).

**Reproduce:** `GPU=a10g DTYPE=bfloat16 bash experiments/scripts/run_gpu_eagle.sh` (run 1); `GPU=a10g DTYPE=bfloat16 SUFFIX=-r2 bash experiments/scripts/run_gpu_eagle.sh sweep controller` (repeats); `python experiments/bench/aggregate_repeats.py code128t-a10g`. Logs: [`../logs/a10g/`](../logs/a10g/).
