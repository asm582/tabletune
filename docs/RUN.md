# Run it on a GPU

Install, serve and tune TableTune. To tune for another GPU or model in detail: [TUNING.md](TUNING.md). To reproduce the experiments: [REPRODUCE.md](REPRODUCE.md).

## Tesla T4 or A10G

Tested on AWS EC2 with the [Deep Learning Base AMI with Single CUDA (Ubuntu 24.04)](https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html) (NVIDIA driver 595.91.07, CUDA 13.2):

| GPU | EC2 instance | vCPU | RAM |
|---|---|---|---|
| Tesla T4 (16 GB) | `g4dn.xlarge` | 4 | 16 GB |
| A10G (24 GB) | `g5.xlarge` | 4 | 16 GB |

**Install** (once; 99 s on a T4, mostly downloads, so it depends on your network). `install.sh` checks for git, curl and an NVIDIA GPU, then installs what's missing at the tested versions: uv, Python 3.12 in `~/venv`, vLLM, and this package. The repository is private, so cloning needs a GitHub login with access.

```bash
git clone https://github.com/asm582/tabletune.git
cd tabletune
bash install.sh
```

**Upgrading an earlier install:** the Python package is now `dsd-tuning` (import `dsd_tuning`); before 0.0.1 shipped it was `vllm-dsd-tuning` (import `system`). Re-running `install.sh` removes the old one. If you installed with `pip` yourself, first run `uv pip uninstall vllm-dsd-tuning` (or `pip uninstall vllm-dsd-tuning`) and delete the clone's `vllm_dsd_tuning.egg-info/` folder, so vLLM doesn't find two copies of the plugin.

**Serve Qwen3-4B with the tuned table and the controller.** It checks which GPU you have and picks the matching table: the T4 table (`t4-eagle3`) on a Tesla T4, or the A10G table (`a10g-eagle3`) on an A10G. Any other GPU stops with a message pointing to [tuning your own table](TUNING.md).

```bash
source ~/venv/bin/activate
dsd-tuner serve
```

The first start downloads Qwen3-4B and its drafter (~8 GB; how long depends on your network), then loads and compiles them: on a T4 it took ~4.5 min to answer with the models already downloaded. It is ready when the log shows `Application startup complete`.

**Send a request** from a second terminal:

```bash
curl -s localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen3-4B", "prompt": "def fibonacci(n):", "max_tokens": 64}'
```

## H100, or a mixture-of-experts target

The catalog has six H100 tables (`dsd-tuner tables`), all measured with vLLM's V2 model runner, which keeps full CUDA graphs for every draft length; with the default V1 runner any table falls back to piecewise CUDA graphs and loses 30–43% on the H100. `dsd-tuner serve` sets the runner from the table, but it needs a vLLM that has it: the nightly below.

**Install** a separate environment with the vLLM nightly the H100 results used (`43b4aaea`, V2 runner); this needs [uv](https://docs.astral.sh/uv/), which `install.sh` also installs:

```bash
git clone https://github.com/asm582/tabletune.git && cd tabletune
uv venv --python 3.12 ~/venv-v2 && source ~/venv-v2/bin/activate
uv pip install --prerelease=allow --torch-backend=auto vllm --extra-index-url https://wheels.vllm.ai/43b4aaea3e40e20ef53ce2bfecfa1b72060bf2b5
uv pip install -e .
export VLLM_USE_V2_MODEL_RUNNER=1        # the tuner profiles with the V2 runner (dsd-tuner serve sets it itself)
```

**Serve** a catalog table with the controller:

```bash
dsd-tuner serve                               # Qwen3-4B + EAGLE-3 (h100-eagle3, picked on an H100)
dsd-tuner serve h100-moe-eagle3               # Qwen3-30B-A3B, mixture of experts
dsd-tuner serve h100-qwen3-32b-tp2-eagle3     # Qwen3-32B on two H100s (tensor parallel 2), 256 requests
```

The others: `h100-ngram`, `h100-dflash` and `h100-awq-eagle3` (4-bit target). Each table's `table.json` says what it was measured on and how well it did.

To tune your own table instead (your prompts, another model, another GPU):

**Tune** the mixture-of-experts target Qwen3-30B-A3B with its EAGLE-3 drafter (BF16 on one 80 GB H100; about 20 minutes of profiling plus model downloads). Use your own prompts, prompt length (`--isl`) and temperature if they differ:

```bash
dsd-tuner run --target Qwen/Qwen3-30B-A3B --drafter AngelSlim/Qwen3-a3B_eagle3 \
  --prompts-file docs/examples/code-prompts.jsonl --isl 32 --temperature 1.0 \
  --max-num-seqs 128 --dtype bfloat16 --out moe-tune
```

**Serve** your table with the controller (leave out the first line to serve the table alone), or add it to the catalog with `dsd-tuner add-table --model-runner v2` ([CONTRIBUTING.md](../CONTRIBUTING.md)):

```bash
export VLLM_DSD_CONTROLLER=1 VLLM_DSD_COSTS="moe-tune/costs_k*.json" VLLM_DSD_PRIOR=moe-tune/acceptance.json
vllm serve Qwen/Qwen3-30B-A3B --dtype bfloat16 --max-model-len 4096 --max-num-seqs 128 \
  --gpu-memory-utilization 0.90 --speculative-config "$(cat moe-tune/speculative_config.json)"
```

Things to know on this vLLM nightly:
- **A table that steps from draft length 2 to 1 can stall the V2 runner.** If `moe-tune/schedule.json` has such a step, rebuild it from lengths 0, 1 and 3 only: `dsd-tuner schedule --costs "moe-tune/costs_k[013].json" --acceptance moe-tune/acceptance.json --max-num-seqs 128 --out moe-tune-013`, and serve that one. (The Qwen3-30B-A3B table, `[[1,3,3],[4,4,2],[5,128,3]]`, has no such step.)
- **Tensor parallelism:** for a target served with `--tensor-parallel-size N`, tune with `DSD_TP=N` so the step times are those of the sharded model (Qwen3-32B at TP=2 in llm-d: [`experiments/h100/llmd/`](../experiments/h100/llmd/)).
- The controller chooses only among the draft lengths in its starting table (the ones vLLM captured CUDA graphs for), so start from a table that contains the lengths you want it to use.
