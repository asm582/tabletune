# TableTune

[![CI](https://github.com/asm582/tabletune/actions/workflows/ci.yml/badge.svg)](https://github.com/asm582/tabletune/actions/workflows/ci.yml) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Speculative decoding makes vLLM generate faster by guessing a few tokens ahead and checking them all in one step. How far ahead to guess depends on the GPU and on how busy the server is, and today it is set by hand. TableTune measures it for your GPU and keeps it right while you serve.

- ⚡ **The right draft length at every load:** one tuner run measures your GPU and builds the table, about 8 minutes on an H100. It never changes what the model generates.
- 🔁 **Adapts while serving:** an in-engine controller corrects a table that doesn't fit the GPU, model or traffic.
- 🔌 **Drop-in for vLLM:** a plugin, no changes to vLLM. Measured on a Tesla T4, an A10G and H100s, dense targets of 4B–32B and a mixture-of-experts target.

## Results: end to end in llm-d

llm-d's [optimized-baseline](https://github.com/llm-d/llm-d) deployment as shipped: two Qwen3-32B replicas at TP=2 on four H100s behind the llm-d router, EAGLE-3 drafter, code prompts, configuration changes only ([deployment](experiments/h100/llmd/qwen3-32b/)). Output tokens/s through the router, mean of 3 runs:

| Open requests | Fixed draft length 3 (tok/s) | vLLM's documented table (tok/s) | TableTune table (tok/s, vs. fixed 3 / vs. documented) | Documented table + controller (tok/s, vs. documented) |
|---|---|---|---|---|
| 256 | 11843 | 13319 | $\color{green}{\textbf{14233 (+20\\% / +7\\%)}}$ | $\color{green}{\textbf{14506 (+9\\%)}}$ |
| 512 | 12964 | 13747 | $\color{green}{\textbf{16581 (+28\\% / +21\\%)}}$ | $\color{green}{\textbf{17190 (+25\\%)}}$ |
| 1024 | 13078 | 13786 | $\color{green}{\textbf{17210 (+32\\% / +25\\%)}}$ | $\color{green}{\textbf{17744 (+29\\%)}}$ |

Up to 128 open requests all configurations are within 4% of each other; above that, length 3 no longer pays, and both the tuned table and the controller move to about 1 token per step. The tuned table is built from the length-0/1/3 profiles, because the vLLM nightly used here (`43b4aaea`) can stall when a table steps from length 2 to 1 ([data](experiments/results/h100-studies/llmd/qwen3-32b.csv)).

## Run it

On a Tesla T4 or an A10G, three commands install it and serve Qwen3-4B with the tuned table and the controller:

```bash
git clone https://github.com/asm582/tabletune.git && cd tabletune
bash install.sh && source ~/venv/bin/activate
dsd-tuner serve
```

On an H100 the catalog has tables for Qwen3-4B, Qwen3-30B-A3B (mixture of experts) and Qwen3-32B at tensor parallel 2, served the same way once vLLM's V2-runner nightly is installed: [docs/RUN.md](docs/RUN.md#h100-or-a-mixture-of-experts-target). To tune your own table: [docs/TUNING.md](docs/TUNING.md).

## Paper

Everything else is in the paper, [**docs/paper/paper.pdf**](docs/paper/paper.pdf): every fixed draft length on a T4, an A10G and an H100, a mixture-of-experts target, long prompts, open-loop and jumping load, TurboSpec- and Nightjar-style control, the controller on mistuned tables and workload switches, and what the engine must provide. [docs/REPRODUCE.md](docs/REPRODUCE.md) maps each result to its data and command.

## Repository layout

| | |
|---|---|
| [`dsd_tuning/`](dsd_tuning/README.md) | The package: the [tuner](dsd_tuning/tuner/README.md) and the [controller](dsd_tuning/controller/README.md) (how it plugs into vLLM, its settings, why both) |
| [`tables/`](tables/) | The table catalog `dsd-tuner serve` reads (`dsd-tuner tables` lists it): Tesla T4, A10G and H100; to add one: [CONTRIBUTING.md](CONTRIBUTING.md) |
| [`docs/`](docs/) | [Run it on a GPU](docs/RUN.md) · [tune another GPU or model](docs/TUNING.md) · [reproduce the paper](docs/REPRODUCE.md) · example prompts |
| [`experiments/`](experiments/README.md) | How every number was measured: data, scripts, harness |
| [`experiments/h100/`](experiments/h100/README.md) | The H100 studies in the paper: drivers, llm-d deployments, committed inputs, analysis and figure scripts |
| [`baselines/`](baselines/README.md) | TurboSpec-style and Nightjar-style controllers, reimplemented as separate vLLM plugins |
| [`experiments/h100-runbook/`](experiments/h100-runbook/README.md) | Setting up and running on OpenShift H100s (V1 matrix, V2 environment) |
| [`docs/paper/`](docs/paper/README.md) | The paper ([PDF](docs/paper/paper.pdf)), its source and figures |
| `tests/` | Tests that run without a GPU, same layout as `dsd_tuning/` |

## License

Apache License 2.0. See [LICENSE](LICENSE). Changes: [releases](https://github.com/asm582/tabletune/releases). Security: [SECURITY.md](SECURITY.md). Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
