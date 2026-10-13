# Tuner (offline)

Builds vLLM's Dynamic SD table (`num_speculative_tokens_per_batch_size`: the K to use at each batch size) from measurements on the GPU you serve on.

| Step | File | GPU | Redo when |
|---|---|---|---|
| Step time for each batch size and K | `profile_costs.py` | yes | new GPU, model, drafter or prompt length |
| Acceptance of each guessed position | `profile_acceptance.py` | yes | new workload |
| The table: *goodput = batch size × tokens per step ÷ step time*, best K per batch size | [`../core.py`](../core.py) | no | seconds |

`sample_prompts.py` draws a prompt sample from a HuggingFace dataset (needs `uv pip install -e ".[prompts]"`); `make_schedule.py` builds a table from existing profiles (the pipelines in [`experiments/scripts/`](../../experiments/scripts/README.md) use it); `validate_costmodel.py` compares the formula's best K with a benchmark sweep.

## How step time is measured

For each draft length, `profile_costs.py` starts vLLM with the table pinned to that length and times batches of exactly B requests, with prompts at your workload's length (`--isl`). One run asks each request for 1 output token (prefill only), another for 65 (the same prefill plus 64 decode steps); step time = the difference ÷ 64.

All B requests must stay in the batch for all 64 steps. With normal speculation some guesses are accepted, requests gain different numbers of tokens per step, and the lucky ones finish early, so later steps run with fewer requests and look too cheap. The tuner therefore uses vLLM's synthetic rejection mode with 0% acceptance: every guess is still drafted and verified, so each step does its full work, but none is accepted, so every request gains exactly one token per step and all B finish together. The benefit of accepted guesses comes back through the acceptance measured separately.

## Run it

Measuring on a GPU (`dsd-tuner run`, `recalibrate`): [docs/TUNING.md](../../docs/TUNING.md). Without a GPU, from existing profiles:

```bash
dsd-tuner schedule --costs "tables/t4-eagle3/costs_k*.json" --acceptance tables/t4-eagle3/acceptance.json
dsd-tuner screen --costs "tables/t4-eagle3/costs_k*.json" --acceptance tables/t4-eagle3/acceptance.json
```

`schedule` builds the table; `screen` shows what each fixed K would lose.

## Table rules

The raw best K can switch many times as batch size grows. Two rules cut the switch points, where K would flip as the live batch moves: **ties keep the current K** (another K must be more than 0.5% better), and **narrow ranges merge** (a range must span at least 25% of its starting batch size). There is no "K only falls" rule: on the T4 the best K goes up and down ([experiments/results/nsys](../../experiments/results/nsys/README.md)).
