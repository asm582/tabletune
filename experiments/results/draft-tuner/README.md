# Separate draft model (Qwen3-0.6B) with the tuner and the controller (issue #4)

> Paths and commands here use the package's current name, `dsd_tuning/`; when these runs were made it was `system/` (renamed in 0.0.1).

**Date:** 2026-09-27 · Tesla T4 · target Qwen3-4B FP16, drafter `Qwen/Qwen3-0.6B` (same tokenizer), `method: draft_model` · vLLM 353e491ef + the controller patches (since replaced by the vLLM plugin, issue #22) · MRv1 · `max_num_seqs=128`

**Workload:** the code workload the EAGLE-3 v3 table was built for (aiperf `instruct-coder`, ~32-token prompts, answers up to 256 tokens, temperature 1.0), so `../sweeps/code128t-t4` (EAGLE-3 and no speculation) is the direct comparison.

**Question:** the tokens-per-step side of the tuner was tested with n-gram (#3); does the tuner follow a very different *cost* curve, and what happens where memory, not compute, is the limit?

## Highlights

1. **Feasible and lossless:** Dynamic SD works with a separate draft model (not in vLLM's tested list); greedy outputs match no speculation with fixed K, the table and the controller ([../smoke/README.md](../smoke/README.md#draft-model-qwen3-06b-with-dynamic-sd-issue-4)). KV cache 11,360–12,400 tokens, as estimated (a third of no speculation's 35,264).
2. **The tuner follows the new cost curve.** Each extra guess costs a full 28-layer drafter pass, so K=5 is not best even for one request. The table `[[1,2,2],[3,17,3],[18,40,2],[41,128,1]]` is within 3.5% of the best fixed K at every load, while every fixed K loses 12–32% somewhere. Best K predicted right at 8/9 loads (3.6 pp ratio error).
3. **K=0 is not "off":** at K=0 the step still costs 40.1 ms (1 request) and 86.7 ms (64 requests), vs. 35.7 and 66.8 ms for EAGLE-3 at K=0 on the same workload.
4. **First crossover on the T4, and the formula misses it.** From 64 requests every draft-model configuration preempts heavily and flattens at ~655 tok/s, while no speculation reaches 704–923 tok/s: speculation stops paying. The tuner picks the right K there (K=1) but cannot choose "no drafter", and its formula has no memory term. Evidence for a memory-aware policy (#1).
5. **The controller** rebuilds the draft model's own table from EAGLE's v3 table. That table was already close to right, so there was no measurable gain (−2.3% to +1.0%, section 5).

## 1. Acceptance (160 code prompts, T=1.0)

| Drafter | Acceptance by position | Tokens per step |
|---|---|---|
| Qwen3-0.6B draft model, K=5 | 0.71 / 0.54 / 0.42 / 0.33 / 0.27 | 1.71 (K=1), 2.25 (K=2), 2.67 (K=3), 3.27 (K=5) |
| EAGLE-3 (`../tuner-v3-code`) | 0.61 / 0.34 / 0.16 | 2.10 (K=3) |

The separate drafter guesses much better than EAGLE-3's one-layer head. Proposal rate 1.02 (a draft model drafts every step; ≥ 0.98 counts as 1).

## 2. Step costs (ms per decode step; ISL 32)

| Requests | K=0 | K=1 | K=2 | K=3 | K=4 | K=5 |
|---|---|---|---|---|---|---|
| 1 | 40.1 | 43.1 | 49.9 | 62.7 | 75.7 | 94.8 |
| 64 | 86.7 | 100.9 | 149.5 | 188.5 | 258.5 | 288.1 |
| 104 | 113.1 | 152.0 | 239.5 | 291.0 | 417.8 | 454.0 |

EAGLE-3 for comparison (`../tuner-v3-code`, same ISL): K=0 → K=3 at one request is 35.7 → 42.2 ms (+18%); for the draft model it is 40.1 → 62.7 ms (+56%). The second guess costs far more than the first at load (+14 ms vs. +48 ms at 64 requests; not traced). Profiled up to 104 requests: with ~11.8k KV tokens larger batches would preempt during profiling.

## 3. Sweep (output tok/s per GPU)

| c | Draft K=1 | K=3 | K=5 | **Tuned table** | Tuned vs. best fixed | No spec | EAGLE-3 tuned v3 |
|---|---|---|---|---|---|---|---|
| 1 | 38.1 | 43.1 | 42.0 | **42.6** | −1.1% | 28.3 | 46.0 |
| 2 | 69.6 | 79.3 | 74.4 | **77.9** | −1.7% | 51.8 | 85.9 |
| 4 | 125.9 | 141.8 | 116.0 | **136.8** | −3.5% | 93.3 | 148.3 |
| 8 | 220.5 | 223.5 | 216.6 | **225.7** | +1.0% | 173.8 | 257.0 |
| 16 | 343.4 | 360.9 | 305.9 | **360.1** | −0.2% | 300.4 | 450.0 |
| 32 | 548.3 | 508.2 | 398.2 | **542.3** | −1.1% | 455.4 | 692.8 |
| 64 | 658.5 | 602.9 | 463.4 | **672.5** | +2.1% | 703.5 | 925.7 |
| 96 | 653.4 | 575.2 | 455.9 | **660.8** | +1.1% | 830.5 | 1022.4 |
| 128 | 657.2 | 576.1 | 447.7 | **655.4** | −0.3% | 922.9 | 1045.1 |

Sources: [../sweeps/draft-code-t4/summary.csv](../sweeps/draft-code-t4/summary.csv); no spec and EAGLE-3 from [../sweeps/code128t-t4/summary.csv](../sweeps/code128t-t4/summary.csv) (same workload, earlier run). Single runs.

- **No fixed K is safe:** K=1 loses 12% at 1–4 requests, K=3 loses 12% from 96 requests up, K=5 up to 32%. The tuned table stays within 3.5%.
- **Memory:** preemptions from 64 requests (tuned: 108 at c=64, 492 at 96, 705 at 128) and at most ~69 requests running whatever the concurrency. No speculation wins by 5–40% there (703.5 vs. 672.5 at c=64, 922.9 vs. 657.2 at c=128). EAGLE-3 beats the draft model at every load on this workload (cheaper steps, far less KV).

## 4. Validating the predictions

`python dsd_tuning/tuner/validate_costmodel.py --sweep experiments/results/sweeps/draft-code-t4 --costs 'experiments/results/draft-tuner/costs_k*.json' --acceptance experiments/results/draft-tuner/acceptance.json --prefix draft-k` ([validation.json](validation.json)): among the tested K (1, 3, 5), the best K is predicted right at 8/9 loads, mean ratio error 3.6 pp (n-gram: 5.4). The miss is c=32, where K=1 beats K=3 by 8% but the formula prefers K=3; the tuned table itself uses K=2 there and is within 1.1% of K=1. With a proposal rate of 1 the average- and full-cost models are identical. At 64–128 requests the predicted K is right *among the Ks* but no speculation beats all of them: the formula has no memory term.

## 5. Controller

EAGLE-3's v3 table `[[1,18,3],[19,43,2],[44,128,1]]` as the starting table (K in {1,2,3}), with the draft model's costs and prior (hysteresis 8, re-fit every 200 steps). For K in {1,2,3} the tuner builds the same table as with all K: `[[1,2,2],[3,17,3],[18,40,2],[41,128,1]]`. EAGLE's table happens to be close to it, so this checks convergence more than it can show a gain.

- The first re-fit rebuilt `[[1,2,2],[3,18,3],[19,40,2],[41,128,1]]`; later re-fits moved the boundaries by one request around the tuner's table (8 changes in 4,000 steps). Proposal-rate tracking stayed off, as intended for a drafter that proposes every step. Acceptance estimates 0.71–0.73 / 0.54–0.56 / 0.42–0.44, as calibrated.
- Throughput (output tok/s per GPU; sweeps `draft-ctrl-v3table`, `draft-ctrl-off`):

| c | EAGLE's v3 table, controller off | + controller | Controller vs. off | Tuned table (section 3) |
|---|---|---|---|---|
| 4 | 135.7 | 132.5 | -2.3% | 136.8 |
| 16 | 367.7 | 362.4 | -1.5% | 360.1 |
| 32 | 540.8 | 535.1 | -1.1% | 542.3 |
| 40 | 587.9 | 593.7 | +1.0% | n/a |
| 64 | 679.5 | 663.5 | -2.3% | 672.5 |
| 96 | 660.7 | 651.9 | -1.3% | 660.8 |

No measurable difference (−2.3% to +1.0%, within noise): EAGLE's table was already close to the draft model's, so there was little to fix. The controller converged to the right table; a stronger test would start from a clearly misaligned table (e.g. K=3 everywhere, which loses 12% from 96 requests up). The controller cannot fix the memory limit either: it only chooses K.

## Files

| File | What |
|---|---|
| `acceptance.json` | acceptance on 160 code prompts at T=1.0, K=5 |
| `costs_k0..5.json`, `.log` | step costs through the Dynamic SD path, ISL 32, B up to 104 |
| `schedule.json`, `goodput.csv`, `schedule.log` | the tuner's table and predicted fixed-K losses |
| `validation.json` | predictions vs. the fixed-K sweep |

Reproduce: `bash experiments/scripts/run_draft.sh` (stages: accept, costs, schedule, sweep, controller).
