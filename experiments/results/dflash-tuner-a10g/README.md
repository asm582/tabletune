# DFlash tuner and controller on the A10G (issue #7)

A10G, BF16, Qwen3-4B + `z-lab/Qwen3-4B-DFlash-b16` (drafts a block of K tokens in one pass), code workload (instruct-coder, ~32-token prompts, T=1.0), `max_num_seqs=128`. Output tok/s per GPU, **mean of 3 runs** (sweeps `dflash-code-a10g`, `-r2`, `-r3`; MLflow `compare-repeats`; spread 1.5% median, 5.6% max). **Needs BF16:** in FP16 (T4, and the A10G in FP16) almost no guess is accepted (4 of 2,085), see issue #7.

**Tuned table:** `[[1,7,8],[8,9,6],[10,12,4],[13,17,3],[18,28,4],[29,65,3],[66,94,1],[95,128,0]]` · acceptance by position 0.70 / 0.47 / 0.32 / 0.23 / 0.16 / 0.12 / 0.09 / 0.06.

| c | K=1 | K=3 | K=8 | Tuned | Tuned vs. best fixed |
|---|---|---|---|---|---|
| 1 | 69.1 | 102.2 | 122.6 | 121.6 | −0.9% |
| 2 | 133.1 | 192.1 | 219.0 | 221.5 | +1.1% |
| 4 | 243.9 | 345.9 | 422.0 | 400.0 | −5.2% |
| 8 | 459.6 | 628.6 | 731.6 | 729.7 | −0.3% |
| 16 | 833.8 | 1148.1 | 1044.5 | 1158.1 | +0.9% |
| 32 | 1449.8 | 1727.3 | 1139.8 | 1705.9 | −1.2% |
| 64 | 2085.6 | 2102.9 | 1394.1 | 2096.5 | −0.3% |
| 96 | 2253.2 | 2055.4 | 1431.1 | 2323.6 | +3.1% |
| 128 | 2382.8 | 2172.4 | 1386.0* | 2675.2 | **+12.3%** |

\* K=8 at 128 requests is the mean of runs 1 and 3: run 2 ran out of GPU memory twice (original and automatic re-run).

- **K=8 wins up to ~8 requests, then collapses** (−34% to −42% vs. the best K from 32 up); each fixed K loses 18–44% somewhere. The tuned table stays within ~1% of the best K except −5.2% at 4 requests, where it already uses K=8; the Dynamic SD path's overhead is the likely cause (issue #12).
- **Limit:** no speculation beats the tuned table by 16.8% / 14.3% at 96 / 128 requests: at K=0 the (large) drafter still runs.
- **KV cache:** 55.1k tokens vs. 81.4k without speculation (−32%): DFlash's drafter keeps its own KV for every token; the K guessed tokens add only ~1.6%. Memory did not bind here (peak 42–54% at 128 requests, no preemption).

**Controller** from EAGLE's T4 table `[[1,18,3],[19,43,2],[44,128,1]]` with DFlash's costs and any K (3 runs): it rebuilt exactly DFlash's tuned table in all three runs; +10.8% at 4 requests (338 → 374), +6.0% at 32 (1610 → 1708), +3.0% at 40 (1851 → 1906), within ±1% at 16, 64 and 96.

**Reproduce:** `GPU=a10g DTYPE=bfloat16 bash experiments/scripts/run_dflash.sh` (run 1); `GPU=a10g DTYPE=bfloat16 SUFFIX=-r2 bash experiments/scripts/run_dflash.sh sweep controller` (repeats); `python experiments/bench/aggregate_repeats.py dflash-code-a10g`. Logs: [`../logs/a10g/`](../logs/a10g/).
