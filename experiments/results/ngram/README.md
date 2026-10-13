# N-gram acceptance: does code editing make n-gram worth it? (issue #3)

> **Follow-up:** the code-apply workload ("reply with only the modified code"), step costs, the tuned table, the sweep and the controller are in [../ngram-tuner/README.md](../ngram-tuner/README.md).

**Date:** 2026-09-26 · Tesla T4 · Qwen3-4B FP16 · 160 prompts per run, same InstructCoder rows (seed 1234) for both workloads · max 512 output tokens · acceptance only, no step costs yet.

- **Code editing** (`prompts-edit.jsonl`): instruction + the code to edit (InstructCoder `input` column). Prompt 140 tokens on average.
- **Code writing** (`prompts-write.jsonl`): instruction only, the shape of the `instruct-coder` sweeps. Prompt 32 tokens.
- N-gram: `prompt_lookup_min=2`, `prompt_lookup_max=4`, K=5 (drafting is free, so deeper K is worth measuring). EAGLE-3: K=3, as in all other results.

Command: `bash experiments/scripts/run_ngram_accept.sh` (acceptance), `python experiments/tools/copy_diagnostic.py` (how much answers copy the prompt).

## Two acceptance numbers

vLLM counts a "draft" only when the drafter proposes. EAGLE-3 proposes on every step, n-gram only when it finds a match. So accepted ÷ drafts (what the tuner used so far) overstates n-gram. `profile_acceptance.py` now also reports **per decode step**: accepted ÷ decode steps, and the proposal rate = drafts ÷ decode steps. Check: EAGLE-3's proposal rate comes out 1.003 and 1.004, so the decode-step count is right to within 0.4%.

| Run | Proposal rate | Acceptance by position, per draft | Tokens per step, per draft (K max) | **Tokens per step, per decode step** |
|---|---|---|---|---|
| EAGLE-3, edit, greedy | 1.00 | 0.66 / 0.41 / 0.21 | 2.28 | **2.29** |
| EAGLE-3, edit, T=1.0 | 1.00 | 0.64 / 0.39 / 0.19 | 2.22 | **2.22** |
| N-gram, edit, greedy | 0.23 | 0.56 / 0.40 / 0.31 / 0.26 / 0.23 | 2.76 | **1.40** |
| N-gram, edit, T=1.0 | 0.22 | 0.55 / 0.40 / 0.31 / 0.26 / 0.22 | 2.74 | **1.38** |
| N-gram, write, T=1.0 | 0.17 | 0.46 / 0.26 / 0.16 / 0.11 / 0.09 | 2.09 | **1.18** |

## How much do the answers copy? (40 greedy answers each, no speculation)

| Workload | Answer chars inside code blocks | Output tokens whose last 3 tokens occurred before | …and the next token also matched |
|---|---|---|---|
| Edit | 41% | 35% | 27% |
| Write | 32% | 18% | 11% |

A typical edit answer: one sentence, the full modified code (mostly copied), then an "Explanation" and "Note" section of new text ([sample](copy-edit.json)).

## Assumptions checked

| Assumption (from the plan) | Result | Verdict |
|---|---|---|
| Code-editing answers copy the prompt | Only 41% of the answer is code; the rest is explanation. A 3-token match exists at 35% of output tokens | **Partly false** |
| N-gram acceptance is high on code editing | High *when it proposes* (0.56 at position 1, and it decays slowly: 0.23 at position 5, vs. EAGLE's 0.21 at position 3). But it proposes on only 23% of steps | **Partly true** |
| N-gram is near zero on code writing | Proposes on 17% of steps, 1.18 tokens per step: answers repeat their own code (names, repeated lines) | **False** |
| A big acceptance swing between the two workloads (for the drift test) | 1.38 vs. 1.18 tokens per step (+38% vs. +18% over no speculation) | **Weak** |
| Temperature 1.0 lowers acceptance | N-gram: 1.40 → 1.38. EAGLE-3: 2.29 → 2.22 | **Barely** |
| N-gram can compete with EAGLE-3 on code editing | 1.40 vs. 2.29 tokens per step. N-gram's drafting is free, so the step-cost profile still decides, but it needs a much larger cost advantage than EAGLE's drafter costs | **False here** (costs pending) |

## What this means for the tuner and the controller

1. **The tuner must use per-step acceptance for n-gram.** With per-draft rates it would predict 2.74 tokens per step at K=5 instead of 1.38 and pick far too deep a K. This is a real generalization bug the EAGLE-only work could not show.
2. **Step cost depends on how often n-gram proposes.** With a proposal rate of ~0.2, a K=5 table entry verifies ~1 extra token per request on average, not 5. The cost model needs the mean proposed tokens (proposal rate × K), not the table K. The current cost profile (every request drafts K) would overstate n-gram's cost.
3. **The drift test needs a bigger swing.** Real code-apply tools ask for the full edited file with no explanation; that workload should push the proposal rate much higher. Next step: the same prompts with "Reply with only the complete modified code" and re-measure. Chat vs. that workload would then be a clear drift test.
