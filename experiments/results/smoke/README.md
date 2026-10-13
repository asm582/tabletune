# Smoke Test: No-Spec vs. EAGLE-3 vs. Dynamic SD (T4)

**Date:** 2026-09-25 · **Purpose:** confirm that vLLM, EAGLE-3 speculative decoding and Dynamic SD run correctly on the development GPU before starting benchmarks. **These are not benchmark numbers.** The workload is 11 short requests (710 output tokens), too small for performance conclusions.

## Result

All three modes ran, and **all produced token-identical greedy outputs** (11/11 requests).

| Mode | Throughput | vs. no-spec | Greedy match | Draft acceptance | Tokens per step |
|---|---|---|---|---|---|
| No speculation | 39.9 tok/s | 1.00× | reference | n/a | 1.00 |
| EAGLE-3, fixed K=3 | 66.9 tok/s | **1.68×** | 11/11 | 36.5% (372 / 1,020) | 2.09 |
| EAGLE-3 + Dynamic SD | 55.4 tok/s | 1.39× | 11/11 | 43.3% (340 / 785) | 1.92 |

Throughput is output tokens ÷ generation time, measured over one batch of 8 requests followed by 3 single requests. Tokens per step = 1 + accepted draft tokens ÷ verification steps.

**Per-position acceptance, fixed K=3** (340 verification steps):

| Draft position | 1 | 2 | 3 |
|---|---|---|---|
| Accepted | 214 (63%) | 116 (34%) | 42 (12%) |

## What this confirms

1. **Speculative decoding doesn't change output.** Greedy outputs match exactly with and without speculation, so exact match is a valid quality check for later experiments.
2. **Dynamic SD changes K at runtime.** Fixed K=3 drafts exactly 3 tokens per step (1,020 / 340). Dynamic SD averaged 2.12 (785 / 371): it used K=1 for the batch of 8 and K=3 for the single requests, as the table specifies.
3. **Dynamic SD forces PIECEWISE CUDA graphs on MRv1.** The log shows `Overriding cudagraph_mode from FULL_AND_PIECEWISE to PIECEWISE`. This is one of the baseline-tax causes reported in vLLM issue [#49986](https://github.com/vllm-project/vllm/issues/49986), reproduced here. Together with choosing K=1 for the batch of 8 (probably too cautious on this GPU), it explains why Dynamic SD came out slower than fixed K=3 in this run.
4. **Per-position acceptance counters are available** (`vllm:spec_decode_num_accepted_tokens_per_pos`). This is the input the planned schedule tuner needs. Acceptance drops steeply by position (63% → 34% → 12%), which already suggests K=3 is near the useful limit for this model pair.
5. **The development setup works:** editable vLLM install from the pinned commit, model download, torch.compile, and CUDA graph capture.

## Setup

| Item | Value |
|---|---|
| Instance | AWS g4dn.xlarge: 1× Tesla T4 16 GB (compute capability 7.5), 4 vCPU, 15 GB RAM |
| Image | Deep Learning Base AMI with Single CUDA (Ubuntu 24.04), driver 595.91.07, CUDA 13.2 |
| vLLM | `0.30.1rc1.dev169+g353e491ef`, editable install of commit [`353e491ef`](https://github.com/vllm-project/vllm/commit/353e491efd3a23057004a3ab68fcfbda4abbee13) using prebuilt binaries |
| PyTorch | 2.13.0+cu132 |
| Model runner | MRv1 (`VLLM_USE_V2_MODEL_RUNNER=0`) |
| Target model | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B), FP16 (the T4 has no BF16), thinking mode off |
| Drafter | [`AngelSlim/Qwen3-4B_eagle3`](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3), K=3 |
| Dynamic SD table | `[[1, 4, 3], [5, 8, 1], [9, 256, 0]]` |
| Engine settings | `max_model_len=4096`, `max_num_seqs=64`, `gpu_memory_utilization=0.90`, greedy (`temperature=0`), `max_tokens=128` |
| Attention backend | `TRITON_ATTN` (FlashAttention needs compute capability ≥ 8.0, and FlashInfer is disabled on the T4) |
| KV cache | 4.46 GiB = 32,496 tokens (≈ 8 concurrent requests at full 4k context) |

## Reproduce

On the instance:

```bash
# 1. Tools and vLLM (editable install from the pinned commit, no compilation)
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --filter=blob:none https://github.com/vllm-project/vllm.git ~/vllm
cd ~/vllm && git checkout -b dsd-control-loop 353e491efd3a23057004a3ab68fcfbda4abbee13
~/.local/bin/uv venv --python 3.12 ~/venv && source ~/venv/bin/activate
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=353e491efd3a23057004a3ab68fcfbda4abbee13 \
  ~/.local/bin/uv pip install -e . --torch-backend=auto

# 2. Smoke test (run from the project repo, not from ~, or Python
#    imports the ~/vllm source folder instead of the installed package)
cd ~/vllm-dsd-tuning
bash experiments/bench/smoke_test.sh
```

This writes `experiments/results/smoke/{nospec,eagle3,dsd}.json` (outputs, token IDs, speculation counters) and prints the comparison. Scripts: [`experiments/bench/smoke_test.py`](../../bench/smoke_test.py), [`experiments/bench/smoke_test.sh`](../../bench/smoke_test.sh).

## Caveats

- **Too small to benchmark.** One run per mode, 710 tokens, no warm-up runs discarded, and only batch sizes 1 and 8. Real numbers come from the concurrency sweep (plan Section 4).
- **The T4 is for development only.** It is limited to the Triton attention backend and FP16, and memory limits concurrency. Final results will run on a larger GPU (plan Section 3). *Update (2026-09-30):* this was the plan at the start; the T4 later became one of the two results GPUs, with 3-run repeats, next to an A10G of similar size (see the [top-level README](../../../README.md)).
- **Why the pinned commit is `353e491ef`:** the newest `main` commit at setup time (`31cc226`) had no prebuilt binaries yet, so the pin moved back to the newest commit that did.
- **Startup time** was 162–301 s per mode, dominated by the first model download and torch.compile. Later runs reuse the caches.
- **A harmless error line:** the log shows `ERROR ... FA2 is only supported on devices with compute capability >= 8`. vLLM falls back to the Triton backend and runs normally.

## N-gram (prompt lookup) with Dynamic SD (issue #3)

**Date:** 2026-09-26 · Same 11 requests, same Dynamic SD table `[[1,4,3],[5,8,1],[9,256,0]]`. N-gram searches the prompt and the output so far for a 2–4 token match and proposes the tokens that followed it; there is no drafter model. vLLM's Dynamic SD docs list only Eagle, Eagle-3 and DFlash as tested.

| Mode | Throughput | Greedy match | Steps with a proposal | Avg K proposed | Acceptance by position |
|---|---|---|---|---|---|
| N-gram, fixed K=3 | 41.7 tok/s | 11/11 | 72 | 3.00 (216 / 72) | 0.35 / 0.18 / 0.08 |
| N-gram + Dynamic SD | 41.1 tok/s | 11/11 | 77 | 2.32 (179 / 77) | 0.36 / 0.10 / 0.04 |
| N-gram + Dynamic SD + controller | 50.5 tok/s | 11/11 | 75 | 2.71 (203 / 75) | 0.36 / 0.13 / 0.05 |

Commands: `python experiments/bench/smoke_test.py --mode ngram`, `--mode ngram-dsd`, and the controller run with `VLLM_DSD_CONTROLLER=1 VLLM_DSD_HYSTERESIS=2 VLLM_DSD_REFIT_STEPS=20 VLLM_DSD_COSTS="experiments/results/tuner-v3-code/costs_k*.json" VLLM_DSD_PRIOR=experiments/results/tuner-v2/acceptance.json ... --mode ngram-dsd --tag _ctrl`.

**What this shows:**

1. **Dynamic SD works with n-gram, and stays lossless.** All three modes match no speculation token for token. The table is applied: fixed K proposes exactly 3 tokens, the table averages 2.32.
2. **N-gram proposes only when it finds a match.** Only ~75 of the ~700 decode steps had a proposal on these chat-style prompts, and acceptance is low (35% at position 1 vs. 63% for EAGLE-3). As expected, n-gram needs a workload where the output copies the prompt; these prompts don't.
3. **The controller runs with n-gram** (4 hooks unchanged, greedy 11/11, one table change). Throughput differences here are noise-level on 710 tokens.
4. **Finding: on a small workload the controller's acceptance estimate stays at the prior.** After re-fitting it logged acceptance `[0.607, 0.328, 0.152]`, which is the EAGLE-3 prior it started from, not n-gram's measured ~0.36 / 0.13 / 0.05. Reason: the estimate is `(w·prior + accepted) / (w + drafted)` with prior weight w = 2000 pseudo-tokens, and counts are halved after every re-fit; ~200 drafted tokens can't move it. In serving at c=64 with re-fit every 200 steps, ~25k counted tokens outweigh the prior. **At low load (c=1, re-fit every 500 steps, halving) the counts settle near ~1,000, so the prior keeps about two thirds of the weight.** That matters most for n-gram, whose acceptance is far from an EAGLE prior. Follow-up in issue #3: scale the prior weight to traffic, or decay by time instead of per re-fit.
5. **Step-cost profiling needs care.** Because n-gram proposes only on a match, a profile with ordinary prompts would measure mostly K=0 steps. The cost profile needs prompts that guarantee matches (e.g. repeated text).

## Draft model (Qwen3-0.6B) with Dynamic SD (issue #4)

**Date:** 2026-09-27 · Same 11 requests and Dynamic SD table. The drafter is a separate model, `Qwen/Qwen3-0.6B` (same tokenizer and vocabulary as Qwen3-4B), `method: draft_model`. vLLM's Dynamic SD docs list only Eagle, Eagle-3 and DFlash as tested.

| Mode | Throughput | Greedy match | Avg K drafted | Acceptance by position |
|---|---|---|---|---|
| Draft model, fixed K=3 | 53.7 tok/s | 11/11 | 3.00 (711 / 237) | 0.78 / 0.67 / 0.55 |
| Draft model + Dynamic SD | 48.7 tok/s | 11/11 | 1.95 (575 / 295) | 0.78 / 0.33 / 0.28 (per draft) |
| Draft model + Dynamic SD + controller | 81.9 tok/s | 11/11 | | |

**Memory** (server log; estimate in issue #4): weights 8.67 GiB (estimate ~9.0), KV cache memory 2.77 GiB with fixed K and 2.88 GiB with the table (~2.9), **KV cache 11,360 / 11,792 tokens** (11–12k). About a third of no speculation (35,264) and 43% of EAGLE-3 (27,472): the drafter keeps its own KV cache (112 KiB per token on top of the target's 144 KiB).

**What this shows:**
1. Dynamic SD works with a separate draft model and stays lossless, also with the controller on (log: `allowed K=[0, 1, 3]`, no "tracking proposal rate": draft models propose on every step, so tracking is off).
2. The 0.6B drafter's acceptance is high and decays slowly (0.78 / 0.67 / 0.55 at K=3 on these chat prompts; EAGLE-3 had 0.63 / 0.34 / 0.12 on the same smoke run), so K above 3 may pay at low load.
3. The memory estimate holds: preemption should start around 40 concurrent requests on the code workload.

Commands: `python experiments/bench/smoke_test.py --mode draft`, `--mode draft-dsd`, and with `VLLM_DSD_CONTROLLER=1 ... --mode draft-dsd --tag _ctrl` (costs and prior from `tuner-v3-code`, as a mechanism check only).

## DFlash (z-lab/Qwen3-4B-DFlash-b16) (issue #7)

**Date:** 2026-09-27 · Same 11 requests and table. DFlash drafts a block of tokens in one pass of a 5-layer drafter that reads target layers 1, 9, 17, 25, 33 (vLLM logs them as `(2, 10, 18, 26, 34)`: it indexes layer outputs, `i + 1`, deliberately).

| Mode | Throughput | Greedy match | Drafts / draft tokens | Accepted (by position) |
|---|---|---|---|---|
| DFlash, fixed K=3 | 34.0 tok/s | 11/11 | 695 / 2,085 | 4 (4 / 0 / 0) |
| DFlash + Dynamic SD | 31.5 tok/s | 11/11 | 695 / 1,535 | 4 (4 / 0 / 0) |
| DFlash + Dynamic SD + controller | 40.8 tok/s | 11/11 | 695 / 1,831 | 4 (4 / 0 / 0) |
| No speculation | 39.9 tok/s | reference | | |

Memory: weights 8.61 GiB, KV cache 12,176–13,744 tokens.

**Runs and stays lossless, but the drafter's guesses are wrong: 4 accepted out of 1,535–2,085 drafted (~0.2%), even at position 1.** Speculation is then pure overhead (slower than no speculation). The automatic pipeline was stopped; the results are not usable for the tuner.

Log facts: TRITON_ATTN backend (the non-causal path DFlash needs); both models cast from BF16 to FP16 (the T4 has no BF16); CUDA graphs and torch.compile of the drafter enabled. Candidate causes, each needing a short GPU test: the Triton non-causal path on sm_75 (retry with `VLLM_ATTENTION_BACKEND=FLEX_ATTENTION`), FP16 numerics of a BF16-trained drafter, MRv1 vs. MRv2 for DFlash (retry with `VLLM_USE_V2_MODEL_RUNNER=1`, fixed K, no table), or graphs/compile (`enforce_eager`).
