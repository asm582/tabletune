# N-gram (prompt lookup) with the tuner and the controller (issue #3)

> Paths and commands here use the package's current name, `dsd_tuning/`; when these runs were made it was `system/` (renamed in 0.0.1).

**Date:** 2026-09-26 · Tesla T4 · Qwen3-4B FP16 · vLLM 353e491ef + the controller patches (since replaced by the vLLM plugin, issue #22) · MRv1 · `max_num_seqs=128`

**Question:** do the tuner and the in-engine controller, built and validated with EAGLE-3, do the right thing with a very different drafter? N-gram has no drafter model: it looks up the last 2–4 tokens earlier in the prompt and output and proposes what followed. It drafts for free, but only when it finds a match.

## Highlights

1. **N-gram needs a copy-heavy workload.** On code editing with an explanation, it proposed on 22% of steps (1.38 tokens per step at K=5). Asking for *only* the modified code (as code-apply tools do) raised that to 49% and 2.65 tokens per step ([../ngram/README.md](../ngram/README.md)).
2. **The tuner had two n-gram bugs that EAGLE could not show, both fixed:**
   - vLLM counts acceptance per draft; with drafts on only some steps that overstates the gain (2.74 vs. 1.38 tokens per step on editing). The tuner now uses the proposal rate p = drafts / decode steps.
   - Step cost: taking the step time at the *average* p·K drafted tokens made the tuner pick K=5 everywhere; the sweep shows K=3 beats K=5 by 5–8% from 64 requests up. Taking the step time at the **full K** fits the sweep (best K right at 8/9 loads vs. 5/9). The fixed table matches the measured best K at 9/9 loads.
3. **Result:** the tuned table ([[1,2,5],[3,3,4],[4,20,5],[21,128,3]]) is within 1% of the best fixed n-gram K at 8 of 9 loads (−6% at c=4, see below), +7% over the first table at 96–128 requests, and +40% to +131% over no speculation.
4. **Serving acceptance matches calibration on large samples:** 0.80 / 0.71 / 0.66 / 0.62 / 0.59 by position at 96–128 requests (384–512 prompts) vs. 0.79 / 0.71 / 0.66 / 0.62 / 0.59 from the 160 calibration prompts. At 1–16 requests (the first 16–64 prompts) positions 4–5 are lower, 0.47–0.58: acceptance depends on which prompts are in flight.
5. **The controller works with n-gram:** from an untuned table it learned the proposal rate live and rebuilt exactly the tuner's table (+2% to +22% vs. the same table without it), and left the right table alone. With a deliberately wrong prior (EAGLE's) it recovered the throughput but not the whole table: unseen positions stay at the prior (issue #2).
6. **EAGLE is unchanged:** every change keeps the EAGLE code path; the committed EAGLE tables recompute byte-identically ([tests/tuner/test_tuner_eagle_regression.py](../../../tests/tuner/test_tuner_eagle_regression.py), GPU check in [../integration/README.md §6](../integration/README.md#6-regression-check-after-adding-n-gram-support-issue-3)).

## Workload

`bench-apply.jsonl`: 600 InstructCoder rows (seed 42), each the instruction + the code to edit + "Reply with only the complete modified code, no explanation." Calibration uses 160 other rows (seed 1234). Prompt 151 tokens and answer 136 tokens on average; greedy (temperature 0); answers capped at 512 tokens. N-gram match length 2–4.

## 1. Acceptance (calibration, 160 prompts)

| Drafter | Proposal rate | Acceptance by position (per draft) | Tokens per step (per decode step) |
|---|---|---|---|
| N-gram, K=5 | 0.49 | 0.79 / 0.71 / 0.66 / 0.62 / 0.59 | 1.38 (K=1), 2.05 (K=3), **2.65 (K=5)** |
| EAGLE-3, K=3 | 1.00 | 0.78 / 0.58 / 0.33 | **2.69 (K=3)** |

N-gram's acceptance barely decays with position (a copied span continues), so deeper K keeps paying in tokens. EAGLE-3's measured proposal rate is 1.02 here: the decode-step count is ~2% off with these short answers (0.4% with longer ones); rates ≥ 0.98 are treated as 1.

## 2. Step costs (ms per decode step, K = 0..5)

Profiled through the Dynamic SD path with synthetic zero acceptance, prompts of 151 tokens. N-gram needs a match to draft, so each prompt repeats one line of code (`vN = vN + 1`): every step drafted exactly K tokens (`drafted_per_step` in the cost files). A repeated block of random tokens gave only 1.5–2.1 drafts at K=3 and understated the cost by 12% at 64 requests.

| Requests | K=0 | K=1 | K=2 | K=3 | K=4 | K=5 |
|---|---|---|---|---|---|---|
| 1 | 36.9 | 39.5 | 39.9 | 40.2 | 40.5 | 40.9 |
| 16 | 52.3 | 61.1 | 62.2 | 63.8 | 76.1 | 78.6 |
| 32 | 70.5 | 74.0 | 79.7 | 82.0 | 116.2 | 119.5 |
| 64 | 91.3 | 102.7 | 121.7 | 136.8 | 204.9 | 205.9 |
| 128 | 139.2 | 173.7 | 208.2 | 245.4 | 378.2 | 406.6 |

Drafting is free (K=0→5 adds 4 ms at one request), but K=4 is a step up from K=3 at every batch size from 16 (+19% at 16, +42% at 32, +54% at 128), far more than the extra tokens. It depends on tokens *per request* (5 query tokens), not on the batch total (80 tokens at B=16), which fits the attention-kernel change seen for EAGLE in nsys (appendix A3). Not traced with nsys for n-gram yet.

## 3. Sweep: fixed K vs. the tuned tables (output tok/s per GPU)

| c | No spec | N-gram K=1 | K=3 | K=5 | Tuned v1 (avg cost) | **Tuned v2** | v2 vs. best fixed | EAGLE-3 K=3 |
|---|---|---|---|---|---|---|---|---|
| 1 | 27.8 | 38.9 | 54.4 | 63.1 | 63.9 | **63.7** | +1.0% | 58.7 |
| 2 | 48.8 | 70.4 | 100.6 | 112.0 | 112.5 | **112.8** | +0.7% | 102.1 |
| 4 | 88.9 | 114.9 | 140.4 | 157.6 | 148.2 | **148.0** | −6.1% | 163.6 |
| 8 | 150.3 | 215.2 | 273.0 | 297.0 | 298.5 | **294.3** | −0.9% | 275.8 |
| 16 | 210.9 | 245.3 | 301.4 | 303.9 | 307.1 | **302.3** | −0.5% | 347.1 |
| 32 | 291.5 | 423.6 | 502.8 | 495.3 | 499.4 | **501.5** | −0.3% | 552.1 |
| 64 | 459.8 | 593.4 | 690.8 | 658.6 | 661.3 | **686.1** | −0.7% | 661.8 |
| 96 | 512.5 | 633.3 | 718.8 | 660.9 | 673.6 | **717.0** | −0.2% | 687.6 |
| 128 | 524.3 | 649.0 | 748.3 | 684.8 | 697.4 | **748.4** | 0.0% | 676.6 |

Sources: [../sweeps/ngram-apply-t4/summary.csv](../sweeps/ngram-apply-t4/summary.csv), [../sweeps/ngram-apply-t4-v2/summary.csv](../sweeps/ngram-apply-t4-v2/summary.csv); MLflow sweeps `ngram-apply-t4`, `ngram-apply-t4-v2`. Single runs.

- **v1 → v2:** the average-cost table [[1,128,5]] behaves like fixed K=5 and loses 5–8% at 64–128 requests; v2 switches to K=3 from 21 requests and recovers it.
- **c=4 (−6%):** both tables show it, but their per-user speed is *higher* than fixed K=5 (58.8 and 58.7 vs. 57.2 tok/s per user); only the per-GPU figure of this 16-request point is lower. Likely run timing noise; possibly the Dynamic SD path's PIECEWISE CUDA graphs. Not resolved with one run.
- **N-gram vs. EAGLE-3 on the same workload:** EAGLE-3 wins at 4, 16 and 32 requests (by up to 14% at 16); n-gram wins at 1–2, at 8 (297.0 vs. 275.8 tok/s) and from 64 up (by 11% at 128): no drafter compute, and a larger KV cache (30.6k vs. 27.5k tokens; at c=128: 15–40 preemptions for n-gram vs. 54 for EAGLE-3 K=3).
- **Summary column note:** `tokens_per_step` in summary.csv is vLLM's per-draft figure (4.4 at K=5); per decode step it is ~2.6.

## 4. Validating the cost model

`python dsd_tuning/tuner/validate_costmodel.py --sweep experiments/results/sweeps/ngram-apply-t4 --costs 'experiments/results/ngram-tuner/costs_k*.json' --acceptance experiments/results/ngram-tuner/acceptance.json --out experiments/results/ngram-tuner/validation.json` predicts the goodput of each fixed K at the sweep's live batch size under both models:

| c | Live batch | Measured best K | Average-cost model | Full-cost model |
|---|---|---|---|---|
| 1–16 | 0.9–8.6 | 5 | 5 | 5 |
| 32 | 19.8 | 3 (503 vs. 495 for K=5) | 5 | 5 |
| 64 | 47.9 | 3 | 5 | **3** |
| 96 | 71.0 | 3 | 5 | **3** |
| 128 | 95.8 | 3 | 5 | **3** |
| **Best K right** | | | **5/9** | **8/9** |
| Mean error, predicted vs. measured ratio to the best K | | | 8.6 pp | **5.4 pp** |

A step where some requests draft K costs about as much as one where all do, so the tuner now takes the step time at the full K (`dsd_tuning/core.py`, `step_ms`). For p = 1 (EAGLE) the two models are identical. `make_schedule.py --oracle` with the new table: predicted K = measured best K at 9/9 concurrencies ([schedule.log](schedule.log)).

## 5. Controller

Controller on: hysteresis 4, re-fit every 200 steps, n-gram step costs; proposal-rate tracking is on automatically for n-gram. The controller may only use the K values of its starting table (CUDA graphs). Untuned start: `[[1,16,1],[17,64,3],[65,128,5]]` (K in {1,3,5}, deep K at high load: backwards). For K in {1,3,5} the tuner would build `[[1,20,5],[21,128,3]]`.

| Run | Start table | Prior | Table after re-fits | Estimates at the end |
|---|---|---|---|---|
| (a) | untuned | calibrated n-gram | **`[[1,20,5],[21,128,3]]`** (after `[[1,21,5],[22,128,3]]`) | acceptance 0.79 / 0.70 / 0.64 / 0.61 / 0.57, proposal rate 0.52 |
| (b) | untuned | EAGLE-3's (3 positions, p = 1) | `[[1,2,5],[3,128,3]]`, 7 changes | 0.69 / 0.50 / 0.38 / **0.074 / 0.034**, proposal rate 0.71 |
| (c) | untuned | controller off | unchanged | n/a |
| (d) | tuned v2 | calibrated n-gram | unchanged (0 changes in 15 re-fits; 221 holds) | n/a |

Throughput (output tok/s per GPU):

| c | (c) untuned, off | (a) + controller | (b) + controller, wrong prior | Tuned v2, off | (d) tuned v2 + controller | (a) vs. (c) |
|---|---|---|---|---|---|---|
| 4 | 113.7 | 135.8 | 133.0 | 148.0 | 145.9 | +19% |
| 16 | 242.2 | 296.3 | 296.3 | 302.3 | 300.0 | +22% |
| 32 | 474.0 | 507.8 | 508.0 | 501.5 | 505.7 | +7% |
| 64 | 675.7 | 691.7 | 691.6 | 686.1 | 688.3 | +2% |
| 96 | 664.6 | 719.6 | 721.7 | 717.0 | 722.7 | +8% |
| 128 | 701.8 | 758.4 | 758.9 | 748.4 | 755.2 | +8% |

Sources: `../sweeps/ngram-ctrl-{calib,eagleprior,off,tuned}/summary.csv` and `controller_log.txt`; MLflow sweeps of the same names.

- **(a) passes:** the controller learned n-gram's proposal rate live (0.52 vs. 0.49 calibrated) and rebuilt exactly the tuner's table; from 16 requests up it matches the tuned table; +2% to +22% over the same start without the controller.
- **(d) passes the no-harm check:** with the right table it changed nothing and stayed within −1.4% to +0.9% of the tuned table without the controller.
- **(b) partly passes.** Throughput recovered as much as (a), because K=3 is right from ~21 requests up, but the table is misaligned at 3–20 requests (K=3 where K=5 is up to 9% better in the fixed-K sweep). Two limits of the controller, both known and now measured:
  1. *Ratchet on unseen positions:* the EAGLE prior, extended to 5 positions, made K=4–5 look worthless; K=5 was never chosen at mid batch sizes, so positions 4–5 were never drafted and stayed at 0.074 / 0.034 (true 0.62 / 0.59).
  2. *Slow learning:* prior weight 2000 plus halving the live counts at every re-fit keeps the prior dominant; positions 1–3 and the proposal rate moved only partway.
  
  Fix tried (opt-in exploration where a deep-K step is cheap, and a decaying prior weight): prior decay fixed positions 1–3 and the proposal rate; exploration unstuck positions 4–5 but their live estimates are biased low, so the table still does not match. Details in [../integration/README.md §7](../integration/README.md#7-exploration-and-prior-decay-issue-2).

## Files

| File | What |
|---|---|
| `acceptance.json` | n-gram acceptance on the calibration prompts (copy of `../ngram/accept-ngram-apply-t0.json`) |
| `costs_k0..5.json`, `.log` | step costs; rows include `drafted_per_step` |
| `schedule.json`, `goodput.csv`, `schedule.log` | the tuner's table (full-cost model) and its predictions |
| `schedule_mean.json` | the first table (average-cost model), run as `ngram-tuned` |
| `validation.json` | cost-model comparison against the sweep |

Reproduce: `bash experiments/scripts/run_ngram.sh` (stages: prompts, accept, costs, schedule, sweep, tuned2, controller).
