# Tune for another GPU or model

The [table catalog](../tables/) covers Qwen3-4B on a Tesla T4 and an A10G. For anything else, measure your own table. On an H100, or for a mixture-of-experts or tensor-parallel target, use vLLM's V2 runner: [Run it on an H100](RUN.md#h100-or-a-mixture-of-experts-target) has the install, tune and serve commands. Start from an install done with [`install.sh`](../install.sh), in the repository folder, with `source ~/venv/bin/activate`.

**1. Tune (~45 min on a T4).** Measures step time for each batch size and draft length on this GPU, and how often the drafter's guesses are accepted on your prompts (JSONL, one `{"prompt": ...}` per line; [examples](examples/README.md)).

- `--isl`: your workload's typical prompt length. Step time grows with context, so profile at the length you serve: a table built from 320-token step times kept draft length 2 where 32-token code prompts needed 1, and was 3–10% slower at 64–128 requests.
- `--max-num-seqs`: the server's (default 128). `--temperature`: what you serve with. `--dtype bfloat16` on A10G, L4 or newer.

```bash
dsd-tuner run --target Qwen/Qwen3-4B --drafter AngelSlim/Qwen3-4B_eagle3 --prompts-file docs/examples/code-prompts.jsonl --isl 32 --max-num-seqs 128 --dtype float16 --out my-tune
```

**2. Serve your table with the controller.** Leave out the first three lines to serve the table without the controller.

```bash
export VLLM_DSD_CONTROLLER=1
export VLLM_DSD_COSTS="my-tune/costs_k*.json"
export VLLM_DSD_PRIOR=my-tune/acceptance.json
export VLLM_USE_V2_MODEL_RUNNER=0
vllm serve Qwen/Qwen3-4B --dtype float16 --max-model-len 4096 --max-num-seqs 128 --speculative-config "$(cat my-tune/speculative_config.json)"
```

**3. New workload later: re-measure acceptance only (~2 min).**

```bash
dsd-tuner recalibrate --costs "my-tune/costs_k*.json" --prompts-file docs/examples/code-edit-prompts.jsonl --max-num-seqs 128 --dtype float16 --out my-tune-edit
```

**4. Benchmark it (optional, ~50 min on a T4).** Serves your table (`eagle-tuned`, from `my-tune/schedule.json`), no speculation, and fixed draft lengths 1 and 3, at 1 to 128 requests with the table's `--max-num-seqs`; writes `experiments/results/sweeps/my-bench/summary.csv`. This is the evidence for the `--measured` line in step 5. `eagle-tuned` covers EAGLE-3 tables; for another model set `TARGET_MODEL`, `DRAFTER_MODEL` and `DTYPE`, and set `TEMPERATURE` if you tuned with one (default 0).

```bash
uv tool install aiperf==0.13.0
EAGLE_OUT=my-tune MAX_SEQS=128 bash experiments/bench/run_configs.sh my-bench "nospec k1 k3 eagle-tuned" "1 16 64 128" instruct-coder
uv pip install -e ".[tracking]"
python experiments/bench/plot_pareto.py my-bench
```

**5. Share it** so others can serve it with one command: [CONTRIBUTING.md](../CONTRIBUTING.md).

How the tuner decides: [dsd_tuning/tuner/README.md](../dsd_tuning/tuner/README.md). Controller settings: [dsd_tuning/controller/README.md](../dsd_tuning/controller/README.md).
